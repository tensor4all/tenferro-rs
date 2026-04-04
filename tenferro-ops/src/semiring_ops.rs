use crate::config::DotGeneralConfig;
use computegraph::GraphOp;

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(shape: Vec<usize>) -> Self;
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self;
}
