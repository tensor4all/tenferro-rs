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

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    /// Construct a DotGeneral op from the dim-numbering `config` together with
    /// the explicit operand ranks.
    ///
    /// The `lhs_rank` / `rhs_rank` parameters are the actual ranks of the two
    /// operands. They are needed by implementors (e.g. `StdTensorOp::DotGeneral`)
    /// that carry rank on the op variant so downstream passes can recover rank
    /// without relying on the removed `DotGeneralConfig` rank fields (issue
    /// #664). Implementors that do not track rank (e.g. `SemiringOp`) may
    /// ignore them.
    fn dot_general(config: DotGeneralConfig, lhs_rank: usize, rhs_rank: usize) -> Self;
    fn reduce_sum(axes: Vec<usize>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self;
    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self;
    fn extract_diag(axis_a: usize, axis_b: usize) -> Self;
    fn embed_diag(axis_a: usize, axis_b: usize) -> Self;
}
