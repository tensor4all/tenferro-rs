//! Automatic differentiation rules for [`StdTensorOp`].
//!
//! `linearize` and `transpose_rule` are separate graph-level contracts.
//! Core ops keep their rules here; extension ops own their own AD support
//! through the extension trait.

pub mod context;

#[cfg(feature = "autodiff")]
mod analytic;
#[cfg(feature = "autodiff")]
mod contraction;
#[cfg(feature = "autodiff")]
mod diagonal;
#[cfg(feature = "autodiff")]
mod dynamic;
#[cfg(feature = "autodiff")]
mod elementwise;
#[cfg(feature = "autodiff")]
mod indexing;
#[cfg(feature = "autodiff")]
pub(crate) mod registry;
#[cfg(feature = "autodiff")]
mod semiring;
#[cfg(feature = "autodiff")]
mod structural;
#[cfg(feature = "autodiff")]
#[doc(hidden)]
pub mod support;
#[cfg(feature = "autodiff")]
mod zeros;

#[cfg(feature = "autodiff")]
use computegraph::fragment::FragmentBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;

#[cfg(feature = "autodiff")]
use chainrules_core::{ADRuleKind, ADRuleResult};

#[cfg(feature = "autodiff")]
use crate::ext_op::{linearize_extension_rule, transpose_extension_rule};
#[cfg(feature = "autodiff")]
use crate::std_tensor_op::StdTensorOp;

/// Forward-mode AD (JVP) for `StdTensorOp`: given the primal op and its
/// tangent inputs, emit the linearized fragment into `builder` and return
/// the output tangents.
///
/// Rules per op live in the category submodules (`semiring`, `analytic`,
/// `elementwise`, `structural`, `contraction`, `indexing`, `diagonal`,
/// `dynamic`). `StdTensorOp::Extension(_)` delegates to the trait.
#[cfg(feature = "autodiff")]
pub fn linearize(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match try_linearize(op, builder, primal_in, primal_out, tangent_in, ctx) {
        Ok(tangents) => tangents,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible forward-mode AD (JVP) for `StdTensorOp`.
#[cfg(feature = "autodiff")]
pub fn try_linearize(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut context::ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    if let StdTensorOp::Extension(ext) = op {
        return linearize_extension_rule(
            ext.as_ref(),
            builder,
            primal_in,
            primal_out,
            tangent_in,
            ctx,
        );
    }

    let kind = op
        .primitive_kind()
        .expect("non-extension StdTensorOp must have a primitive kind");
    let rule = registry::primitive_ad_rule(kind)
        .ok_or_else(|| registry::missing_rule(kind, ADRuleKind::Linearize))?;
    rule.linearize(op, builder, primal_in, primal_out, tangent_in, ctx)
}

/// Reverse-mode AD (VJP) for `StdTensorOp`: given the primal op, its
/// inputs, and the output cotangent, emit the transposed fragment and
/// return the input cotangents.
///
/// See [`linearize`] for the category split; the same categories appear
/// here.
#[cfg(feature = "autodiff")]
pub fn transpose_rule(
    op: &StdTensorOp,
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match try_transpose_rule(op, emitter, cotangent_out, inputs, mode, ctx) {
        Ok(cotangents) => cotangents,
        Err(err) => panic!("{err}"),
    }
}

/// Fallible reverse-mode AD (VJP) for `StdTensorOp`.
#[cfg(feature = "autodiff")]
pub fn try_transpose_rule(
    op: &StdTensorOp,
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut context::ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    if let StdTensorOp::Extension(ext) = op {
        let emitter_dyn: &mut dyn OpEmitter<StdTensorOp> = emitter;
        return transpose_extension_rule(
            ext.as_ref(),
            emitter_dyn,
            cotangent_out,
            inputs,
            mode,
            ctx,
        );
    }

    let kind = op
        .primitive_kind()
        .expect("non-extension StdTensorOp must have a primitive kind");
    let rule = registry::primitive_ad_rule(kind)
        .ok_or_else(|| registry::missing_rule(kind, ADRuleKind::Transpose))?;
    let emitter_dyn: &mut dyn OpEmitter<StdTensorOp> = emitter;
    rule.transpose_rule(op, emitter_dyn, cotangent_out, inputs, mode, ctx)
}

#[cfg(test)]
mod tests;
