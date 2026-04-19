pub mod context;

mod analytic;
mod contraction;
mod diagonal;
mod dynamic;
mod elementwise_tier2;
mod linalg;
mod semiring;
mod structural;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::std_tensor_op::StdTensorOp;

fn linearize_non_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut context::ShapeGuardContext,
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Div => {
            elementwise_tier2::linearize_div(builder, primal_in, primal_out, tangent_in)
        }
        StdTensorOp::Abs => elementwise_tier2::linearize_abs(builder, primal_in, tangent_in),
        StdTensorOp::Sign => elementwise_tier2::linearize_sign(builder, tangent_in),
        StdTensorOp::Constant { .. } => vec![None],
        StdTensorOp::Compare(_) => vec![None],
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
        StdTensorOp::DotGeneral { config } => {
            contraction::linearize_dot_general(builder, primal_in, tangent_in, config, ctx)
        }
        StdTensorOp::NaryEinsum { subscripts, .. } => {
            contraction::linearize_nary_einsum(builder, primal_in, tangent_in, subscripts)
        }
        StdTensorOp::ReduceSum { axes, .. } => {
            contraction::linearize_reduce_sum(builder, tangent_in, op, axes)
        }
        StdTensorOp::ReduceProd { axes, input_shape } => contraction::linearize_reduce_prod(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            axes,
            input_shape,
        ),
        StdTensorOp::ReduceMax { axes, input_shape }
        | StdTensorOp::ReduceMin { axes, input_shape } => contraction::linearize_reduce_chooser(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            axes,
            input_shape,
        ),
        StdTensorOp::Transpose { perm } => {
            structural::linearize_transpose(builder, tangent_in, perm)
        }
        StdTensorOp::Reshape { .. } => {
            structural::linearize_reshape(builder, primal_in, tangent_in, op)
        }
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::linearize_broadcast_in_dim(builder, primal_in, tangent_in, shape, dims)
        }
        StdTensorOp::Convert { from, to } => {
            structural::linearize_convert(builder, tangent_in, *from, *to)
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::linearize_extract_diag(builder, tangent_in, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::linearize_embed_diag(builder, tangent_in, *axis_a, *axis_b)
        }
        StdTensorOp::Tril { k } => structural::linearize_tril(builder, tangent_in, *k),
        StdTensorOp::Triu { k } => structural::linearize_triu(builder, tangent_in, *k),
        StdTensorOp::Pad(config) => structural::linearize_pad(builder, tangent_in, config),
        StdTensorOp::DynamicTruncate { axis } => {
            dynamic::linearize_dynamic_truncate(builder, primal_in, tangent_in, *axis)
        }
        StdTensorOp::PadToMatch { axis } => {
            dynamic::linearize_pad_to_match(builder, primal_in, tangent_in, *axis)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],
        StdTensorOp::Lu { input_shape } => {
            linalg::linearize_lu(builder, primal_out, tangent_in, input_shape, ctx)
        }
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            lhs_shape,
            rhs_shape,
        } => linalg::linearize_triangular_solve(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
            lhs_shape,
            rhs_shape,
        ),
        StdTensorOp::Cholesky { input_shape } => {
            linalg::linearize_cholesky(builder, primal_out, tangent_in, input_shape)
        }
        StdTensorOp::Svd { eps, input_shape } => {
            linalg::linearize_svd(builder, primal_out, tangent_in, *eps, input_shape, ctx)
        }
        StdTensorOp::Qr { input_shape } => {
            linalg::linearize_qr(builder, primal_out, tangent_in, input_shape, ctx)
        }
        StdTensorOp::Eigh { eps, input_shape } => {
            linalg::linearize_eigh(builder, primal_out, tangent_in, *eps, input_shape)
        }
        StdTensorOp::Eig {
            input_dtype,
            input_shape,
        } => linalg::linearize_eig(builder, primal_out, tangent_in, *input_dtype, input_shape),
        StdTensorOp::ValidateNonsingular { .. } => vec![tangent_in[0]],
        _ => return None,
    })
}

fn linearize_semiring(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut context::ShapeGuardContext,
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
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut context::ShapeGuardContext,
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Div => elementwise_tier2::transpose_div(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Abs => elementwise_tier2::transpose_abs(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Sign => elementwise_tier2::transpose_sign(emitter, cotangent_out, mode),
        StdTensorOp::Constant { .. } => vec![],
        StdTensorOp::Compare(_) => vec![None, None],
        StdTensorOp::Exp => analytic::transpose_exp(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Log => analytic::transpose_log(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Sin => analytic::transpose_sin(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Cos => analytic::transpose_cos(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Tanh => analytic::transpose_tanh(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Sqrt => analytic::transpose_sqrt(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Rsqrt => analytic::transpose_rsqrt(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Pow => analytic::transpose_pow(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Expm1 => analytic::transpose_expm1(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Log1p => analytic::transpose_log1p(emitter, cotangent_out, inputs, mode),
        StdTensorOp::DotGeneral { config } => {
            contraction::transpose_dot_general(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::NaryEinsum {
            subscripts,
            n_inputs,
        } => contraction::transpose_nary_einsum(
            emitter,
            cotangent_out,
            inputs,
            mode,
            subscripts,
            *n_inputs,
        ),
        StdTensorOp::ReduceSum { .. } => {
            contraction::transpose_reduce_sum(emitter, cotangent_out, op, inputs)
        }
        StdTensorOp::ReduceProd { .. } => {
            contraction::transpose_reduce_prod(emitter, cotangent_out, inputs, op)
        }
        StdTensorOp::ReduceMax { .. } | StdTensorOp::ReduceMin { .. } => {
            contraction::transpose_reduce_chooser(emitter, cotangent_out, inputs, op)
        }
        StdTensorOp::Transpose { perm } => {
            structural::transpose_transpose(emitter, cotangent_out, perm)
        }
        StdTensorOp::Reshape { .. } => {
            structural::transpose_reshape(emitter, cotangent_out, op, inputs)
        }
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::transpose_broadcast_in_dim(emitter, cotangent_out, shape, dims)
        }
        StdTensorOp::Convert { from, to } => {
            structural::transpose_convert(emitter, cotangent_out, mode, *from, *to)
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::transpose_extract_diag(emitter, cotangent_out, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::transpose_embed_diag(emitter, cotangent_out, *axis_a, *axis_b)
        }
        StdTensorOp::Tril { k } => structural::transpose_tril(emitter, cotangent_out, *k),
        StdTensorOp::Triu { k } => structural::transpose_triu(emitter, cotangent_out, *k),
        StdTensorOp::DynamicTruncate { axis } => {
            dynamic::transpose_dynamic_truncate(emitter, cotangent_out, inputs, *axis)
        }
        StdTensorOp::PadToMatch { axis } => {
            dynamic::transpose_pad_to_match(emitter, cotangent_out, inputs, *axis)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            lhs_shape,
            rhs_shape,
        } => linalg::transpose_triangular_solve(
            emitter,
            cotangent_out,
            inputs,
            mode,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
            lhs_shape,
            rhs_shape,
        ),
        StdTensorOp::ValidateNonsingular { .. } => vec![cotangent_out[0]],
        _ => return None,
    })
}

fn transpose_semiring(
    op: &StdTensorOp,
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    _ctx: &mut context::ShapeGuardContext,
) -> Option<Vec<Option<LocalValId>>> {
    Some(match op {
        StdTensorOp::Add => semiring::transpose_add(cotangent_out),
        StdTensorOp::Mul => semiring::transpose_mul(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Neg => semiring::transpose_neg(emitter, cotangent_out),
        StdTensorOp::Conj => semiring::transpose_conj(emitter, cotangent_out),
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
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    if let Some(result) =
        linearize_non_semiring(op, builder, primal_in, primal_out, tangent_in, ctx)
    {
        return result;
    }
    if let Some(result) = linearize_semiring(op, builder, primal_in, tangent_in, ctx) {
        return result;
    }
    todo_linearize(op)
}

pub fn transpose_rule(
    op: &StdTensorOp,
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    if let Some(result) = transpose_non_semiring(op, emitter, cotangent_out, inputs, mode, ctx) {
        return result;
    }
    if let Some(result) = transpose_semiring(op, emitter, cotangent_out, inputs, mode, ctx) {
        return result;
    }
    todo_transpose_rule(op)
}

#[cfg(test)]
mod tests;
