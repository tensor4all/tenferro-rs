use computegraph::GraphOp;
use tenferro_tensor::DotGeneralConfig;

use crate::dim_expr::DimExpr;

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>, input_shape: Vec<DimExpr>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self;
    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self;
    fn extract_diag(axis_a: usize, axis_b: usize) -> Self;
    fn embed_diag(axis_a: usize, axis_b: usize) -> Self;
}
