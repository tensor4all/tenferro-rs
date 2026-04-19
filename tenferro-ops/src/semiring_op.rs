//! **Non-mainline.** The semiring graph op is retained for Stage 2-6
//! compatibility but is no longer the architectural source of truth for the
//! traced AD substrate.
//!
//! The mainline traced op vocabulary is [`StdTensorOp`](crate::std_tensor_op::StdTensorOp);
//! `SemiringOp<Alg>` is an auxiliary path whose forward execution still runs
//! through [`SemiringBackend`](tenferro_tensor::SemiringBackend), but it does
//! not integrate with the mainline AD rules in [`crate::ad`] and should not be
//! reasoned about as co-equal with `StdTensorOp`.
//!
//! This module — together with [`crate::semiring_op_kind`] and
//! [`crate::semiring_ops`] — is scheduled for removal when the
//! `ExtensionOp` substrate lands in Stage 6. See
//! `docs/design/design_v3/30-algebra-and-tropical.md` ("Recommended Fate Of
//! `SemiringOp`") and `docs/design/design_v3/90-migration-plan.md` Stage 6
//! ("Implement The `ExtensionOp` Mechanism") for the retirement plan. New code
//! must not extend this path; lower tropical/semiring use cases through core
//! primitives or, once Stage 6 lands, through the `ExtensionOp` mechanism.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use computegraph::GraphOp;
use tenferro_algebra::Algebra;
use tenferro_tensor::{DotGeneralConfig, TypedTensor};

use crate::dim_expr::DimExpr;
use crate::semiring_op_kind::SemiringOpKind;
#[allow(deprecated)]
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SemiringInputKey {
    pub id: u64,
}

/// Legacy algebra-generic traced op (non-mainline).
///
/// **Deprecated**: non-mainline per
/// `docs/design/design_v3/30-algebra-and-tropical.md`
/// ("Recommended Fate Of `SemiringOp`"). Scheduled for removal in Stage 6 of
/// the `design_v3` migration plan. Use [`crate::std_tensor_op::StdTensorOp`]
/// and compose tropical-style operations from core primitives instead.
#[deprecated(
    since = "design_v3-stage-2",
    note = "non-mainline per docs/design/design_v3/30-algebra-and-tropical.md; scheduled for removal in Stage 6"
)]
pub struct SemiringOp<Alg: Algebra> {
    pub kind: SemiringOpKind,
    _marker: PhantomData<Alg>,
}

#[allow(deprecated)]
impl<Alg: Algebra> SemiringOp<Alg> {
    pub fn new(kind: SemiringOpKind) -> Self {
        Self {
            kind,
            _marker: PhantomData,
        }
    }
}

#[allow(deprecated)]
impl<Alg: Algebra> Clone for SemiringOp<Alg> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            _marker: PhantomData,
        }
    }
}

#[allow(deprecated)]
impl<Alg: Algebra> fmt::Debug for SemiringOp<Alg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemiringOp")
            .field("kind", &self.kind)
            .finish()
    }
}

#[allow(deprecated)]
impl<Alg: Algebra> Hash for SemiringOp<Alg> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

#[allow(deprecated)]
impl<Alg: Algebra> PartialEq for SemiringOp<Alg> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[allow(deprecated)]
impl<Alg: Algebra> Eq for SemiringOp<Alg> {}

#[allow(deprecated)]
impl<Alg> GraphOp for SemiringOp<Alg>
where
    Alg: Algebra + Send + Sync + 'static,
{
    type Operand = TypedTensor<Alg::Scalar>;
    type Context = ();
    type InputKey = SemiringInputKey;

    fn n_inputs(&self) -> usize {
        self.kind.n_inputs()
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

#[allow(deprecated)]
impl<Alg> SemiringOps for SemiringOp<Alg>
where
    Alg: Algebra + Send + Sync + 'static,
{
    fn add_op() -> Self {
        Self::new(SemiringOpKind::Add)
    }

    fn mul_op() -> Self {
        Self::new(SemiringOpKind::Mul)
    }

    fn dot_general(config: DotGeneralConfig, _lhs_rank: usize, _rhs_rank: usize) -> Self {
        // `SemiringOp` does not track rank on the op; ranks are recovered from
        // the tensor operands at execution time. The parameters are ignored.
        Self::new(SemiringOpKind::DotGeneral(config))
    }

    fn reduce_sum(axes: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::ReduceSum { axes })
    }

    fn transpose_op(perm: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::Transpose { perm })
    }

    fn reshape(_from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self {
        Self::new(SemiringOpKind::Reshape {
            shape: concrete_shape(to_shape),
        })
    }

    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::BroadcastInDim {
            shape: concrete_shape(shape),
            dims,
        })
    }

    fn extract_diag(axis_a: usize, axis_b: usize) -> Self {
        Self::new(SemiringOpKind::ExtractDiag { axis_a, axis_b })
    }

    fn embed_diag(axis_a: usize, axis_b: usize) -> Self {
        Self::new(SemiringOpKind::EmbedDiag { axis_a, axis_b })
    }
}

fn concrete_shape(shape: Vec<DimExpr>) -> Vec<usize> {
    shape
        .into_iter()
        .map(|dim| match dim {
            DimExpr::Const(value) => value,
            other => panic!("SemiringOp only supports concrete shape expressions, got {other:?}"),
        })
        .collect()
}
