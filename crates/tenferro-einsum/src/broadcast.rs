//! Singleton-dimension broadcasting between einsum operands.
//!
//! Every einsum surface asks the same question about two extents of one label:
//! are they equal, does one of them broadcast, or do they conflict?
//! [`classify_extents`] is the only place that answers it. The concrete size
//! dictionary, the eager/traced broadcast decision, symbolic output-shape
//! inference and the reverse-mode reduction of broadcast axes all build on it,
//! so no caller can fall through a branch the others handle.

use std::collections::HashMap;

#[cfg(feature = "autodiff")]
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::sym_dim::SymDim;

/// How two extents of one label relate under singleton broadcasting.
///
/// `lhs` is the extent already recorded for the label, `rhs` the one being
/// folded in (or, for reverse mode, `lhs` is the active operand's extent and
/// `rhs` a peer's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtentPair {
    /// Provably the same extent.
    Equal,
    /// `lhs` is known to be `1` and `rhs` is not: `lhs` broadcasts to `rhs`.
    LhsSingleton,
    /// `rhs` is known to be `1` and `lhs` is not: `rhs` broadcasts to `lhs`.
    RhsSingleton,
    /// `lhs` is a known constant other than `1` and `rhs` is symbolic. A valid
    /// binding makes `rhs` either equal to `lhs` or `1`, so the broadcast
    /// extent is `lhs` either way.
    LhsConstant,
    /// Mirror of [`ExtentPair::LhsConstant`]: the broadcast extent is `rhs`.
    RhsConstant,
    /// Two different symbolic extents: the broadcast extent is
    /// [`symbolic_broadcast_extent`] of the two.
    BothSymbolic,
    /// Two known extents that differ and are both not `1`.
    Conflict,
}

/// Classify two extents of one label.
///
/// `lhs_known`/`rhs_known` are the extents' constant values (`None` when
/// symbolic) and `same_expr` says whether the two are the same expression,
/// which proves equality even when both are symbolic.
pub(crate) fn classify_extents(
    lhs_known: Option<usize>,
    rhs_known: Option<usize>,
    same_expr: bool,
) -> ExtentPair {
    if same_expr {
        return ExtentPair::Equal;
    }
    match (lhs_known, rhs_known) {
        (Some(lhs), Some(rhs)) if lhs == rhs => ExtentPair::Equal,
        (Some(1), _) => ExtentPair::LhsSingleton,
        (_, Some(1)) => ExtentPair::RhsSingleton,
        (Some(_), Some(_)) => ExtentPair::Conflict,
        (Some(_), None) => ExtentPair::LhsConstant,
        (None, Some(_)) => ExtentPair::RhsConstant,
        (None, None) => ExtentPair::BothSymbolic,
    }
}

/// Classify two concrete extents.
pub(crate) fn classify_sizes(lhs: usize, rhs: usize) -> ExtentPair {
    classify_extents(Some(lhs), Some(rhs), lhs == rhs)
}

/// Classify two symbolic extents.
pub(crate) fn classify_sym_dims(lhs: &SymDim, rhs: &SymDim) -> ExtentPair {
    classify_extents(lhs.constant_value(), rhs.constant_value(), lhs == rhs)
}

/// The result of folding one symbolic extent into the extent already recorded
/// for its label.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExtentMerge {
    /// The label's extent after the fold.
    Merged(SymDim),
    /// The extents must be equal; the caller records that requirement.
    RequireEqual,
}

/// The singleton-broadcast extent of two symbolic extents, as an expression
/// that agrees with [`classify_sizes`] at every valid binding.
///
/// A valid binding is `(a, a)`, `(1, b)` or `(a, 1)`, and the broadcast extent
/// is the extent that is not the broadcast `1`. `max(a, b)` gets that right
/// except for an empty axis: `(0, 1)` broadcasts to `0`, not `1`. Multiplying
/// by `min(min(a, b), 1)`, which is `0` exactly when either extent is `0` and
/// `1` otherwise, fixes that case and leaves every other one at `max(a, b)`.
pub(crate) fn symbolic_broadcast_extent(lhs: &SymDim, rhs: &SymDim) -> SymDim {
    let nonempty = lhs.clone().min(rhs.clone()).min(1usize);
    lhs.clone().max(rhs.clone()) * nonempty
}

/// Fold `next` into `existing` for one label shared between operands.
///
/// Without broadcasting every shared extent must be equal. With broadcasting
/// each [`ExtentPair`] has exactly one merged extent; only a known conflict
/// falls back to the equality requirement, which then reports the mismatch.
pub(crate) fn merge_broadcast_extent(
    existing: &SymDim,
    next: &SymDim,
    allow_broadcast: bool,
) -> ExtentMerge {
    if !allow_broadcast {
        return ExtentMerge::RequireEqual;
    }
    match classify_sym_dims(existing, next) {
        ExtentPair::Equal | ExtentPair::RhsSingleton | ExtentPair::LhsConstant => {
            ExtentMerge::Merged(existing.clone())
        }
        ExtentPair::LhsSingleton | ExtentPair::RhsConstant => ExtentMerge::Merged(next.clone()),
        ExtentPair::BothSymbolic => ExtentMerge::Merged(symbolic_broadcast_extent(existing, next)),
        ExtentPair::Conflict => ExtentMerge::RequireEqual,
    }
}

/// How the reverse derivative treats one axis of the active operand.
///
/// The VJP einsum contracts the output cotangent with the other operands. Its
/// output extent for a label is the broadcast of that label's extents among
/// those operands, which is not the active operand's extent when either side
/// was a broadcast singleton. The two non-`Keep` variants both remove the label
/// from the VJP einsum's output, which sums the cotangent over it, and then
/// broadcast the result back to the active operand's shape.
#[cfg(any(feature = "autodiff", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisCotangent {
    /// Some other extent of the label equals the active extent, so the VJP
    /// einsum already produces the axis at the active operand's extent.
    Keep,
    /// The active axis is a size-1 axis that was broadcast: the cotangent is
    /// summed over the broadcast extent and the size-1 axis is restored.
    SumBroadcast,
    /// Every other extent of the label is a size-1 axis that broadcast to the
    /// active extent, so the cotangent is constant along the axis: it is
    /// computed without the label and broadcast to the active extent.
    ExpandSingletonPeers,
}

#[cfg(any(feature = "autodiff", test))]
impl AxisCotangent {
    /// Whether the VJP einsum leaves the label out of its output, so that it
    /// is summed there and restored by a broadcast to the active shape.
    pub(crate) fn sums_label_out(self) -> bool {
        !matches!(self, Self::Keep)
    }
}

/// The extents could not decide whether an active axis was broadcast.
#[cfg(any(feature = "autodiff", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UndecidedBroadcastAxis;

/// Classify two program extents resolved against the bound input shapes.
///
/// `None` is an extent that is not exactly known (an upper bound or unknown),
/// which never proves equality.
#[cfg(feature = "autodiff")]
pub(crate) fn classify_dim_exprs(lhs: Option<&DimExpr>, rhs: Option<&DimExpr>) -> ExtentPair {
    let known = |extent: Option<&DimExpr>| match extent {
        Some(DimExpr::Const(value)) => Some(*value),
        _ => None,
    };
    let same_expr = matches!((lhs, rhs), (Some(lhs), Some(rhs)) if lhs == rhs);
    classify_extents(known(lhs), known(rhs), same_expr)
}

/// Decide how the reverse derivative treats one axis of the active operand.
///
/// `pairs` classifies the active operand's extent for the axis (as `lhs`)
/// against every extent of the same label elsewhere (the other operands and
/// the output, as `rhs`). The answer must be exact, so an axis whose extents
/// leave open whether it was broadcast is reported as undecided rather than
/// guessed: summing a non-broadcast axis, or keeping a broadcast one, are both
/// wrong gradients.
///
/// Only an op that admits broadcasting asks; without it every shared extent
/// is equal and every axis is kept.
#[cfg(any(feature = "autodiff", test))]
pub(crate) fn broadcast_axis_reduce(
    pairs: impl IntoIterator<Item = ExtentPair>,
) -> Result<AxisCotangent, UndecidedBroadcastAxis> {
    let mut active_is_broadcast = false;
    let mut equal_peer = false;
    let mut singleton_peer = false;
    let mut constant_over_symbolic_peer = false;
    for pair in pairs {
        match pair {
            ExtentPair::Equal => equal_peer = true,
            ExtentPair::LhsSingleton => active_is_broadcast = true,
            ExtentPair::RhsSingleton => singleton_peer = true,
            // The peer is bound to the active extent or to 1; which one is
            // open until another peer settles it.
            ExtentPair::LhsConstant => constant_over_symbolic_peer = true,
            // A conflict was already rejected by the primal's shape inference.
            ExtentPair::Conflict => {}
            // A symbolic active extent may itself be a broadcast singleton.
            ExtentPair::RhsConstant | ExtentPair::BothSymbolic => {
                return Err(UndecidedBroadcastAxis);
            }
        }
    }
    if active_is_broadcast {
        // Summing is exact whatever the peers are bound to: a peer of extent 1
        // makes it a sum over one element.
        Ok(AxisCotangent::SumBroadcast)
    } else if equal_peer {
        Ok(AxisCotangent::Keep)
    } else if constant_over_symbolic_peer {
        Err(UndecidedBroadcastAxis)
    } else if singleton_peer {
        Ok(AxisCotangent::ExpandSingletonPeers)
    } else {
        // The label occurs nowhere else; the VJP einsum leaves it out anyway.
        Ok(AxisCotangent::Keep)
    }
}

/// One extent as the broadcast decision sees it: known, or still symbolic.
pub(crate) trait Extent {
    fn known(self) -> Option<usize>;
}

impl Extent for usize {
    fn known(self) -> Option<usize> {
        Some(self)
    }
}

impl Extent for Option<usize> {
    fn known(self) -> Option<usize> {
        self
    }
}

/// Whether a label shared between operands repeats with extents that differ
/// only by a size-1 axis, i.e. whether the operands need singleton-dimension
/// broadcasting.
///
/// This is the single broadcast decision both einsum surfaces call: eager
/// passes fully concrete extents, tracing passes the ones it already knows and
/// skips the rest, so a symbolic extent can only narrow the decision. A label
/// repeated inside one operand is a diagonal, whose extents must match exactly,
/// so each operand joins the size map only after it is read.
pub(crate) fn requires_broadcast<E: Extent + Copy>(shapes: &[&[E]], labels: &[Vec<u32>]) -> bool {
    let mut sizes = HashMap::<u32, usize>::new();
    for (labels, shape) in labels.iter().zip(shapes) {
        let mut operand_sizes = HashMap::<u32, usize>::new();
        for (&label, &extent) in labels.iter().zip(shape.iter()) {
            let Some(size) = extent.known() else { continue };
            if let Some(&previous) = sizes.get(&label) {
                if matches!(
                    classify_sizes(previous, size),
                    ExtentPair::LhsSingleton | ExtentPair::RhsSingleton
                ) {
                    return true;
                }
            }
            operand_sizes.insert(label, size);
        }
        sizes.extend(operand_sizes);
    }
    false
}

#[cfg(test)]
mod tests;
