mod analytic;
mod contraction;
mod diagonal;
mod elementwise_tier2;
mod linalg;
mod semiring;
mod structural;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

fn linearize_non_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Div => {
            elementwise_tier2::linearize_div(builder, primal_in, primal_out, tangent_in)
        }
        StdTensorOp::Abs => elementwise_tier2::linearize_abs(builder, primal_in, tangent_in),
        StdTensorOp::Sign => elementwise_tier2::linearize_sign(builder, tangent_in),
        StdTensorOp::Scale { factor } => {
            elementwise_tier2::linearize_scale(builder, tangent_in, *factor)
        }
        StdTensorOp::Exp => analytic::linearize_exp(builder, primal_out, tangent_in),
        StdTensorOp::Log => analytic::linearize_log(builder, primal_in, tangent_in),
        StdTensorOp::Sin => analytic::linearize_sin(builder, primal_in, tangent_in),
        StdTensorOp::Cos => analytic::linearize_cos(builder, primal_in, tangent_in),
        StdTensorOp::Tanh => analytic::linearize_tanh(builder, primal_out, tangent_in),
        StdTensorOp::Sqrt => analytic::linearize_sqrt(builder, primal_out, tangent_in),
        StdTensorOp::Rsqrt => analytic::linearize_rsqrt(builder, primal_in, primal_out, tangent_in),
        StdTensorOp::Pow => analytic::linearize_pow(builder, primal_in, primal_out, tangent_in),
        StdTensorOp::Expm1 => analytic::linearize_expm1(builder, primal_out, tangent_in),
        StdTensorOp::Log1p => analytic::linearize_log1p(builder, primal_in, tangent_in),
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
        StdTensorOp::Tril { k } => structural::linearize_tril(builder, tangent_in, *k),
        StdTensorOp::Triu { k } => structural::linearize_triu(builder, tangent_in, *k),
        StdTensorOp::Solve => linalg::linearize_solve(builder, primal_in, primal_out, tangent_in),
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => linalg::linearize_triangular_solve(
            builder,
            primal_in,
            tangent_in,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
        ),
        StdTensorOp::Cholesky => linalg::linearize_cholesky(builder, primal_out, tangent_in),
        StdTensorOp::Svd { eps, m, n } => {
            linalg::linearize_svd(builder, primal_out, tangent_in, *eps, *m, *n)
        }
        StdTensorOp::Qr => linalg::linearize_qr(builder, primal_out, tangent_in),
        StdTensorOp::Eigh { eps } => linalg::linearize_eigh(builder, primal_out, tangent_in, *eps),
        _ => return None,
    })
}

fn linearize_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Add => semiring::linearize_add(builder, tangent_in),
        StdTensorOp::Mul => semiring::linearize_mul(builder, primal_in, tangent_in),
        StdTensorOp::Neg => semiring::linearize_neg(builder, tangent_in),
        StdTensorOp::Conj => semiring::linearize_conj(builder, tangent_in),
        _ => return None,
    })
}

fn transpose_non_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Div => elementwise_tier2::transpose_div(builder, cotangent_out, inputs, mode),
        StdTensorOp::Abs => elementwise_tier2::transpose_abs(builder, cotangent_out, inputs, mode),
        StdTensorOp::Sign => elementwise_tier2::transpose_sign(builder, cotangent_out, mode),
        StdTensorOp::Scale { factor } => {
            elementwise_tier2::transpose_scale(builder, cotangent_out, mode, *factor)
        }
        StdTensorOp::Exp => analytic::transpose_exp(builder, cotangent_out, inputs, mode),
        StdTensorOp::Log => analytic::transpose_log(builder, cotangent_out, inputs, mode),
        StdTensorOp::Sin => analytic::transpose_sin(builder, cotangent_out, inputs, mode),
        StdTensorOp::Cos => analytic::transpose_cos(builder, cotangent_out, inputs, mode),
        StdTensorOp::Tanh => analytic::transpose_tanh(builder, cotangent_out, inputs, mode),
        StdTensorOp::Sqrt => analytic::transpose_sqrt(builder, cotangent_out, inputs, mode),
        StdTensorOp::Rsqrt => analytic::transpose_rsqrt(builder, cotangent_out, inputs, mode),
        StdTensorOp::Pow => analytic::transpose_pow(builder, cotangent_out, inputs, mode),
        StdTensorOp::Expm1 => analytic::transpose_expm1(builder, cotangent_out, inputs, mode),
        StdTensorOp::Log1p => analytic::transpose_log1p(builder, cotangent_out, inputs, mode),
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
        StdTensorOp::Tril { k } => structural::transpose_tril(builder, cotangent_out, *k),
        StdTensorOp::Triu { k } => structural::transpose_triu(builder, cotangent_out, *k),
        StdTensorOp::Solve => linalg::transpose_solve(builder, cotangent_out, inputs, mode),
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => linalg::transpose_triangular_solve(
            builder,
            cotangent_out,
            inputs,
            mode,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
        ),
        _ => return None,
    })
}

fn transpose_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Add => semiring::transpose_add(cotangent_out),
        StdTensorOp::Mul => semiring::transpose_mul(builder, cotangent_out, inputs, mode),
        StdTensorOp::Neg => semiring::transpose_neg(builder, cotangent_out),
        StdTensorOp::Conj => semiring::transpose_conj(builder, cotangent_out),
        _ => return None,
    })
}

fn todo_linearize(op: &StdTensorOp) -> ! {
    todo!("linearize not implemented for {:?}", op)
}

fn todo_transpose_rule(op: &StdTensorOp) -> ! {
    todo!("transpose_rule not implemented for {:?}", op)
}

pub fn linearize(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    if let Some(result) = linearize_non_semiring(op, builder, primal_in, primal_out, tangent_in) {
        return result;
    }
    if let Some(result) = linearize_semiring(op, builder, primal_in, tangent_in) {
        return result;
    }
    todo_linearize(op)
}

pub fn transpose_rule(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    if let Some(result) = transpose_non_semiring(op, builder, cotangent_out, inputs, mode) {
        return result;
    }
    if let Some(result) = transpose_semiring(op, builder, cotangent_out, inputs, mode) {
        return result;
    }
    todo_transpose_rule(op)
}
