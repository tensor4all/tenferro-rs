pub mod context;

mod analytic;
mod contraction;
mod diagonal;
mod dynamic;
mod elementwise;
mod indexing;
mod linalg;
mod semiring;
mod structural;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::std_tensor_op::StdTensorOp;

fn todo_linearize(op: &StdTensorOp) -> ! {
    todo!("linearize not implemented for {:?}", op)
}

fn todo_transpose_rule(op: &StdTensorOp) -> ! {
    todo!("transpose_rule not implemented for {:?}", op)
}

/// Forward-mode AD (JVP) for `StdTensorOp`: given the primal op and its
/// tangent inputs, emit the linearized fragment into `builder` and return
/// the output tangents.
///
/// Rules per op live in the category submodules (`semiring`, `analytic`,
/// `elementwise`, `structural`, `contraction`, `indexing`, `linalg`,
/// `diagonal`, `dynamic`). `StdTensorOp::Extension(_)` delegates to the
/// trait.
pub fn linearize(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match op {
        // Semiring-arithmetic family (Add/Mul/Neg/Conj form a commutative
        // semiring over the supported scalar dtypes).
        StdTensorOp::Add => semiring::linearize_add(builder, tangent_in),
        StdTensorOp::Mul => semiring::linearize_mul(builder, primal_in, tangent_in),
        StdTensorOp::Neg => semiring::linearize_neg(builder, tangent_in),
        StdTensorOp::Conj => semiring::linearize_conj(builder, tangent_in),

        // Elementwise (non-semiring) family.
        StdTensorOp::Div => elementwise::linearize_div(builder, primal_in, primal_out, tangent_in),
        StdTensorOp::Abs => elementwise::linearize_abs(builder, primal_in, tangent_in),
        StdTensorOp::Sign => elementwise::linearize_sign(builder, tangent_in),
        StdTensorOp::Constant { .. } => vec![None],
        StdTensorOp::Compare(_) => vec![None],

        // Analytic family.
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

        // Contraction family.
        StdTensorOp::DotGeneral { config } => {
            contraction::linearize_dot_general(builder, primal_in, tangent_in, config, ctx)
        }
        StdTensorOp::NaryEinsum { subscripts } => {
            contraction::linearize_nary_einsum(builder, primal_in, tangent_in, subscripts)
        }
        StdTensorOp::ReduceSum { axes } => {
            contraction::linearize_reduce_sum(builder, tangent_in, op, axes)
        }
        StdTensorOp::ReduceProd { axes } => contraction::linearize_reduce_prod(
            builder, primal_in, primal_out, tangent_in, axes, ctx,
        ),
        StdTensorOp::ReduceMax { axes } | StdTensorOp::ReduceMin { axes } => {
            contraction::linearize_reduce_chooser(
                builder, primal_in, primal_out, tangent_in, axes, ctx,
            )
        }

        // Structural family.
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
        StdTensorOp::Tril { k } => structural::linearize_tril(builder, tangent_in, *k),
        StdTensorOp::Triu { k } => structural::linearize_triu(builder, tangent_in, *k),
        StdTensorOp::Slice(config) => structural::linearize_slice(builder, tangent_in, config),
        StdTensorOp::Pad(config) => structural::linearize_pad(builder, tangent_in, config),
        StdTensorOp::Reverse { axes } => structural::linearize_reverse(builder, tangent_in, axes),

        // Diagonal family.
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::linearize_extract_diag(builder, tangent_in, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::linearize_embed_diag(builder, tangent_in, *axis_a, *axis_b)
        }

        // Indexing family.
        StdTensorOp::Gather(config) => {
            indexing::linearize_gather(builder, primal_in, tangent_in, config)
        }
        StdTensorOp::Scatter(config) => {
            indexing::linearize_scatter(builder, primal_in, tangent_in, config, ctx)
        }

        // Dynamic family.
        StdTensorOp::DynamicTruncate { axis } => dynamic::linearize_dynamic_truncate(
            builder, primal_in, primal_out, tangent_in, *axis, ctx,
        ),
        StdTensorOp::PadToMatch { axis } => {
            dynamic::linearize_pad_to_match(builder, primal_in, tangent_in, *axis)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],

        // Linalg family.
        StdTensorOp::Lu => linalg::linearize_lu(builder, primal_in, primal_out, tangent_in, ctx),
        StdTensorOp::FullPivLu => vec![None, None, None, None, None],
        StdTensorOp::FullPivLuSolve { transpose_a } => linalg::linearize_full_piv_lu_solve(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            *transpose_a,
            ctx,
        ),
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => linalg::linearize_triangular_solve(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
            ctx,
        ),
        StdTensorOp::Cholesky => {
            linalg::linearize_cholesky(builder, primal_in, primal_out, tangent_in, ctx)
        }
        StdTensorOp::Svd { eps } => {
            linalg::linearize_svd(builder, primal_in, primal_out, tangent_in, *eps, ctx)
        }
        StdTensorOp::Qr => linalg::linearize_qr(builder, primal_in, primal_out, tangent_in, ctx),
        StdTensorOp::Eigh { eps } => {
            linalg::linearize_eigh(builder, primal_in, primal_out, tangent_in, *eps, ctx)
        }
        StdTensorOp::Eig { input_dtype } => linalg::linearize_eig(
            builder,
            primal_in,
            primal_out,
            tangent_in,
            *input_dtype,
            ctx,
        ),
        StdTensorOp::ValidateNonsingular => vec![tangent_in[0]],

        // Extension substrate.
        StdTensorOp::Extension(ext) => {
            ext.linearize(builder, primal_in, primal_out, tangent_in, ctx)
        }

        _ => todo_linearize(op),
    }
}

/// Reverse-mode AD (VJP) for `StdTensorOp`: given the primal op, its
/// inputs, and the output cotangent, emit the transposed fragment and
/// return the input cotangents.
///
/// See [`linearize`] for the category split; the same categories appear
/// here.
pub fn transpose_rule(
    op: &StdTensorOp,
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut context::ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    match op {
        // Semiring-arithmetic family.
        StdTensorOp::Add => semiring::transpose_add(cotangent_out),
        StdTensorOp::Mul => semiring::transpose_mul(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Neg => semiring::transpose_neg(emitter, cotangent_out),
        StdTensorOp::Conj => semiring::transpose_conj(emitter, cotangent_out),

        // Elementwise (non-semiring) family.
        StdTensorOp::Div => elementwise::transpose_div(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Abs => elementwise::transpose_abs(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Sign => elementwise::transpose_sign(emitter, cotangent_out, mode),
        StdTensorOp::Constant { .. } => vec![],
        StdTensorOp::Compare(_) => vec![None, None],

        // Analytic family.
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

        // Contraction family.
        StdTensorOp::DotGeneral { config } => {
            contraction::transpose_dot_general(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::NaryEinsum { subscripts } => {
            contraction::transpose_nary_einsum(emitter, cotangent_out, inputs, mode, subscripts)
        }
        StdTensorOp::ReduceSum { .. } => {
            contraction::transpose_reduce_sum(emitter, cotangent_out, op, inputs, ctx)
        }
        StdTensorOp::ReduceProd { .. } => {
            contraction::transpose_reduce_prod(emitter, cotangent_out, inputs, op, ctx)
        }
        StdTensorOp::ReduceMax { .. } | StdTensorOp::ReduceMin { .. } => {
            contraction::transpose_reduce_chooser(emitter, cotangent_out, inputs, op, ctx)
        }

        // Structural family.
        StdTensorOp::Transpose { perm } => {
            structural::transpose_transpose(emitter, cotangent_out, perm)
        }
        StdTensorOp::Reshape { .. } => {
            structural::transpose_reshape(emitter, cotangent_out, op, inputs, ctx)
        }
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::transpose_broadcast_in_dim(emitter, cotangent_out, shape, dims)
        }
        StdTensorOp::Convert { from, to } => {
            structural::transpose_convert(emitter, cotangent_out, mode, *from, *to)
        }
        StdTensorOp::Tril { k } => structural::transpose_tril(emitter, cotangent_out, *k),
        StdTensorOp::Triu { k } => structural::transpose_triu(emitter, cotangent_out, *k),
        StdTensorOp::Slice(config) => {
            structural::transpose_slice(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::Pad(config) => {
            structural::transpose_pad(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::Reverse { axes } => {
            structural::transpose_reverse(emitter, cotangent_out, mode, axes)
        }

        // Diagonal family.
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            diagonal::transpose_extract_diag(emitter, cotangent_out, *axis_a, *axis_b)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            diagonal::transpose_embed_diag(emitter, cotangent_out, *axis_a, *axis_b)
        }

        // Indexing family.
        StdTensorOp::Gather(config) => {
            indexing::transpose_gather(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::Scatter(config) => {
            indexing::transpose_scatter(emitter, cotangent_out, inputs, mode, config, ctx)
        }

        // Dynamic family.
        StdTensorOp::DynamicTruncate { axis } => {
            dynamic::transpose_dynamic_truncate(emitter, cotangent_out, inputs, *axis)
        }
        StdTensorOp::PadToMatch { axis } => {
            dynamic::transpose_pad_to_match(emitter, cotangent_out, inputs, mode, *axis, ctx)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],

        // Linalg family.
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => linalg::transpose_triangular_solve(
            emitter,
            cotangent_out,
            inputs,
            mode,
            *left_side,
            *lower,
            *transpose_a,
            *unit_diagonal,
            ctx,
        ),
        StdTensorOp::FullPivLuSolve { transpose_a } => linalg::transpose_full_piv_lu_solve(
            emitter,
            cotangent_out,
            inputs,
            mode,
            *transpose_a,
            ctx,
        ),
        StdTensorOp::ValidateNonsingular => vec![cotangent_out[0]],

        // Extension substrate.
        StdTensorOp::Extension(ext) => {
            let emitter_dyn: &mut dyn OpEmitter<StdTensorOp> = emitter;
            ext.transpose_rule(emitter_dyn, cotangent_out, inputs, mode, ctx)
        }

        _ => todo_transpose_rule(op),
    }
}

#[cfg(test)]
mod tests;
