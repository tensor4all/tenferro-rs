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
mod semiring;
#[cfg(feature = "autodiff")]
mod structural;
#[cfg(feature = "autodiff")]
mod support;
#[cfg(feature = "autodiff")]
mod zeros;

#[cfg(feature = "autodiff")]
use computegraph::fragment::FragmentBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;

#[cfg(feature = "autodiff")]
use chainrules_core::ADRuleResult;

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
    let tangents = match op {
        // Semiring-arithmetic family (Add/Mul/Neg/Conj form a commutative
        // semiring over the supported scalar dtypes).
        StdTensorOp::Add => semiring::linearize_add(builder, tangent_in),
        StdTensorOp::Mul => semiring::linearize_mul(builder, primal_in, tangent_in),
        StdTensorOp::Neg => semiring::linearize_neg(builder, tangent_in),
        StdTensorOp::Conj => semiring::linearize_conj(builder, primal_in, tangent_in, ctx),

        // Elementwise (non-semiring) family.
        StdTensorOp::Div => elementwise::linearize_div(builder, primal_in, primal_out, tangent_in),
        StdTensorOp::Abs => elementwise::linearize_abs(builder, primal_in, tangent_in),
        StdTensorOp::Sign => elementwise::linearize_sign(builder, tangent_in),
        StdTensorOp::Maximum => elementwise::linearize_maximum(builder, primal_in, tangent_in),
        StdTensorOp::Minimum => elementwise::linearize_minimum(builder, primal_in, tangent_in),
        StdTensorOp::Select => elementwise::linearize_select(builder, primal_in, tangent_in),
        StdTensorOp::Clamp => elementwise::linearize_clamp(builder, primal_in, tangent_in),
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
            structural::linearize_reshape(builder, primal_in, tangent_in, op, ctx)
        }
        StdTensorOp::BroadcastInDim { shape, dims } => {
            structural::linearize_broadcast_in_dim(builder, primal_in, tangent_in, shape, dims, ctx)
        }
        StdTensorOp::Convert { from, to } => {
            structural::linearize_convert(builder, tangent_in, *from, *to)
        }
        StdTensorOp::Tril { k } => structural::linearize_tril(builder, tangent_in, *k),
        StdTensorOp::Triu { k } => structural::linearize_triu(builder, tangent_in, *k),
        StdTensorOp::Slice(config) => structural::linearize_slice(builder, tangent_in, config),
        StdTensorOp::Pad(config) => structural::linearize_pad(builder, tangent_in, config),
        StdTensorOp::Concatenate { axis, n_inputs } => {
            structural::linearize_concatenate(builder, primal_in, tangent_in, *axis, *n_inputs, ctx)
        }
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
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => indexing::linearize_gather_dynamic_slice_sizes(
            builder,
            primal_in,
            tangent_in,
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            *index_vector_dim,
            slice_sizes,
        ),
        StdTensorOp::Scatter(config) => {
            indexing::linearize_scatter(builder, primal_in, tangent_in, config, ctx)
        }
        StdTensorOp::DynamicSlice { slice_sizes } => {
            indexing::linearize_dynamic_slice(builder, primal_in, tangent_in, slice_sizes)
        }
        StdTensorOp::DynamicUpdateSlice => {
            indexing::linearize_dynamic_update_slice(builder, primal_in, tangent_in, ctx)
        }

        // Dynamic family.
        StdTensorOp::DynamicTruncate { axis } => dynamic::linearize_dynamic_truncate(
            builder, primal_in, primal_out, tangent_in, *axis, ctx,
        ),
        StdTensorOp::PadToMatch { axis } => {
            dynamic::linearize_pad_to_match(builder, primal_in, tangent_in, *axis)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],

        // Extension substrate.
        StdTensorOp::Extension(ext) => {
            return linearize_extension_rule(
                ext.as_ref(),
                builder,
                primal_in,
                primal_out,
                tangent_in,
                ctx,
            );
        }
    };
    Ok(tangents)
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
    let cotangents = match op {
        // Semiring-arithmetic family.
        StdTensorOp::Add => semiring::transpose_add(cotangent_out),
        StdTensorOp::Mul => semiring::transpose_mul(emitter, cotangent_out, inputs, mode, ctx),
        StdTensorOp::Neg => semiring::transpose_neg(emitter, cotangent_out),
        StdTensorOp::Conj => semiring::transpose_conj(emitter, cotangent_out, inputs, ctx),

        // Elementwise (non-semiring) family.
        StdTensorOp::Div => elementwise::transpose_div(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Abs => elementwise::transpose_abs(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Sign => elementwise::transpose_sign(emitter, cotangent_out, mode),
        StdTensorOp::Maximum => {
            elementwise::transpose_maximum(emitter, cotangent_out, inputs, mode)
        }
        StdTensorOp::Minimum => {
            elementwise::transpose_minimum(emitter, cotangent_out, inputs, mode)
        }
        StdTensorOp::Select => elementwise::transpose_select(emitter, cotangent_out, inputs, mode),
        StdTensorOp::Clamp => elementwise::transpose_clamp(emitter, cotangent_out, inputs, mode),
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
        StdTensorOp::Concatenate { axis, n_inputs } => structural::transpose_concatenate(
            emitter,
            cotangent_out,
            inputs,
            mode,
            *axis,
            *n_inputs,
            ctx,
        ),
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
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            ..
        } => indexing::transpose_gather_dynamic_slice_sizes(
            emitter,
            cotangent_out,
            inputs,
            mode,
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            *index_vector_dim,
            ctx,
        ),
        StdTensorOp::Scatter(config) => {
            indexing::transpose_scatter(emitter, cotangent_out, inputs, mode, config, ctx)
        }
        StdTensorOp::DynamicSlice { .. } => {
            indexing::transpose_dynamic_slice(emitter, cotangent_out, inputs, mode, ctx)
        }
        StdTensorOp::DynamicUpdateSlice => {
            indexing::transpose_dynamic_update_slice(emitter, cotangent_out, inputs, mode, ctx)
        }

        // Dynamic family.
        StdTensorOp::DynamicTruncate { axis } => {
            dynamic::transpose_dynamic_truncate(emitter, cotangent_out, inputs, *axis)
        }
        StdTensorOp::PadToMatch { axis } => {
            dynamic::transpose_pad_to_match(emitter, cotangent_out, inputs, mode, *axis, ctx)
        }
        StdTensorOp::ShapeOf { .. } => vec![None],

        // Extension substrate.
        StdTensorOp::Extension(ext) => {
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
    };
    Ok(cotangents)
}

#[cfg(test)]
mod tests;
