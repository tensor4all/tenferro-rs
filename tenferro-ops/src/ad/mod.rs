mod contraction;
mod diagonal;
mod semiring;
mod structural;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    match op {
        StdTensorOp::Add => semiring::linearize_add(builder, tangent_in),
        StdTensorOp::Mul => semiring::linearize_mul(builder, primal_in, tangent_in),
        StdTensorOp::Neg => semiring::linearize_neg(builder, tangent_in),
        StdTensorOp::Conj => semiring::linearize_conj(builder, tangent_in),
        StdTensorOp::DotGeneral(config) => {
            contraction::linearize_dot_general(builder, primal_in, tangent_in, config)
        }
        StdTensorOp::ReduceSum { axes, .. } => {
            contraction::linearize_reduce_sum(builder, tangent_in, op, axes)
        }
        StdTensorOp::Transpose { perm } => {
            structural::linearize_transpose(builder, tangent_in, perm)
        }
        StdTensorOp::Reshape { .. } => structural::linearize_reshape(builder, tangent_in, op),
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::linearize_broadcast_in_dim(builder, tangent_in, shape, dims)
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::linearize_extract_diag(builder, tangent_in, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::linearize_embed_diag(builder, tangent_in, *axis_a, *axis_b)
        }
        _ => todo!("linearize not implemented for {:?}", op),
    }
}

pub fn transpose_rule(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    match op {
        StdTensorOp::Add => semiring::transpose_add(cotangent_out),
        StdTensorOp::Mul => semiring::transpose_mul(builder, cotangent_out, inputs, mode),
        StdTensorOp::Neg => semiring::transpose_neg(builder, cotangent_out),
        StdTensorOp::Conj => semiring::transpose_conj(builder, cotangent_out),
        StdTensorOp::DotGeneral(config) => {
            contraction::transpose_dot_general(builder, cotangent_out, inputs, mode, config)
        }
        StdTensorOp::ReduceSum { .. } => {
            contraction::transpose_reduce_sum(builder, cotangent_out, op)
        }
        StdTensorOp::Transpose { perm } => {
            structural::transpose_transpose(builder, cotangent_out, perm)
        }
        StdTensorOp::Reshape { .. } => structural::transpose_reshape(builder, cotangent_out, op),
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::transpose_broadcast_in_dim(builder, cotangent_out, shape, dims)
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::transpose_extract_diag(builder, cotangent_out, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::transpose_embed_diag(builder, cotangent_out, *axis_a, *axis_b)
        }
        _ => todo!("transpose_rule not implemented for {:?}", op),
    }
}
