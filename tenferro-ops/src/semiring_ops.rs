//! **Non-mainline.** The `SemiringOps` trait exists so that the algebra-generic
//! [`crate::semiring_op::SemiringOp`] and the mainline
//! [`crate::std_tensor_op::StdTensorOp`] can share a common constructor
//! vocabulary. Per design_v3, the trait is demoted together with the rest of
//! the semiring graph substrate and is kept only for Stage 2-6 compatibility;
//! Stage 6 removes it entirely.
//!
//! New graph code must construct `StdTensorOp` variants directly rather than
//! going through this trait. See
//! `docs/design/design_v3/30-algebra-and-tropical.md` and
//! `docs/design/design_v3/90-migration-plan.md` Stage 6 for context.

use computegraph::GraphOp;
use tenferro_tensor::DotGeneralConfig;

use crate::dim_expr::DimExpr;

/// Shared op-construction trait used to keep the legacy `SemiringOp` and
/// `StdTensorOp` paths symmetric for the einsum fragment builder.
///
/// **Deprecated**: this trait is kept only to keep the legacy semiring graph
/// path and the mainline `StdTensorOp` path looking symmetric (see
/// `docs/design/design_v3/30-algebra-and-tropical.md`). Once the semiring
/// pipeline is removed in Stage 6 of the `design_v3` migration plan, einsum
/// lowering will target `StdTensorOp` directly and this trait will be
/// deleted. New code must construct `StdTensorOp` variants directly.
///
/// Some method signatures exist only to reach the lowest-common-denominator
/// shape between the two implementations — e.g. [`SemiringOps::reshape`]
/// passes `from_shape` that both current implementations ignore, and
/// [`SemiringOps::dot_general`] passes `lhs_rank` / `rhs_rank` that both
/// current implementations ignore. Those asymmetric parameters document
/// the bridge and will disappear with the trait.
#[deprecated(
    since = "design_v3-stage-2",
    note = "bridge trait kept only for legacy semiring symmetry; scheduled for removal in Stage 6 of design_v3 (see docs/design/design_v3/30-algebra-and-tropical.md)"
)]
pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    /// Construct a DotGeneral op from the dim-numbering `config` together with
    /// the explicit operand ranks.
    ///
    /// The `lhs_rank` / `rhs_rank` parameters were historically required for
    /// implementors that tracked rank on the op variant (before
    /// `DotGeneralConfig`'s rank fields were removed in `#664`). Both current
    /// implementors (`StdTensorOp` and the legacy `SemiringOp`) ignore these
    /// parameters, and they exist only to keep the two paths looking
    /// symmetric. They will disappear with the trait in Stage 6.
    fn dot_general(config: DotGeneralConfig, lhs_rank: usize, rhs_rank: usize) -> Self;
    fn reduce_sum(axes: Vec<usize>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    /// Construct a Reshape op.
    ///
    /// The `from_shape` argument is passed for symmetry across
    /// implementations but is currently ignored by both the
    /// [`StdTensorOp`](crate::std_tensor_op::StdTensorOp) implementation
    /// (the mainline path stores only the target shape) and the legacy
    /// `SemiringOp` implementation. Kept for compatibility until the trait
    /// is removed in Stage 6.
    fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self;
    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self;
    fn extract_diag(axis_a: usize, axis_b: usize) -> Self;
    fn embed_diag(axis_a: usize, axis_b: usize) -> Self;
}
