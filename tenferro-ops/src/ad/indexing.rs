//! AD rules for `StdTensorOp::Gather` and `StdTensorOp::Scatter`.
//!
//! The forward rules (`linearize_gather` / `linearize_scatter`) push the
//! tangent through the same indexing op, reusing the integer index operand
//! from the primal because indices are non-differentiable. The backward
//! rules (`transpose_gather` / `transpose_scatter`) flow the cotangent
//! through the inverse op by swapping `Gather` ↔ `Scatter`, with a
//! mechanical config inversion that mirrors:
//!
//! - `start_index_map` ↔ `scatter_dims_to_operand_dims`
//! - `collapsed_slice_dims` ↔ `inserted_window_dims`
//! - `offset_dims` ↔ `update_window_dims`
//! - `index_vector_dim` carried over unchanged
//!
//! Closing the core op vocabulary under AD is a Stage 7 prerequisite (the
//! tropical fused backward emits `Gather` / `Scatter` on this vocabulary).

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

/// Forward-mode AD rule for `Gather(operand, start_indices, config)`.
///
/// `start_indices` is integer-valued and non-differentiable, so its tangent
/// is ignored. The output tangent is the same gather applied to the
/// operand's tangent, reusing the primal indices.
pub fn linearize_gather(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    config: &GatherConfig,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(d_operand) => {
            let out = builder.add_op(
                StdTensorOp::Gather(config.clone()),
                vec![
                    ValRef::Local(d_operand),
                    ValRef::External(primal_in[1].clone()),
                ],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

/// Reverse-mode AD rule for `Gather(operand, start_indices, config)`.
///
/// Emits a `Scatter` whose config inverts the `GatherConfig` so that the
/// cotangent values are scattered into the positions the forward gather
/// read from. The scatter uses add-mode semantics, so multiple gathers
/// reading the same slot accumulate additively in the operand cotangent.
pub fn transpose_gather(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &GatherConfig,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask.clone(),
        OpMode::Primal => return vec![None, None],
    };

    if !active_mask.first().copied().unwrap_or(false) {
        return vec![None, None];
    }

    let inverse_config = ScatterConfig {
        update_window_dims: config.offset_dims.clone(),
        inserted_window_dims: config.collapsed_slice_dims.clone(),
        scatter_dims_to_operand_dims: config.start_index_map.clone(),
        index_vector_dim: config.index_vector_dim,
    };

    let out = emitter.add_op(
        StdTensorOp::Scatter(inverse_config),
        vec![inputs[0].clone(), inputs[1].clone(), ValRef::Local(ct)],
        OpMode::Linear {
            active_mask: vec![false, false, true],
        },
    );

    vec![Some(out[0]), None]
}

/// Forward-mode AD rule for `Scatter(operand, scatter_indices, updates, config)`.
///
/// `scatter_indices` is integer-valued and non-differentiable.
///
/// The CPU scatter implementation starts with a zero buffer shaped like
/// `operand` (rather than a copy of `operand`) and accumulates `updates`
/// additively. The operand therefore contributes only its shape to the
/// output, so the output tangent comes solely from the updates' tangent:
/// `Scatter(operand, indices, updates_dot, config)`. The operand tangent
/// is ignored because `d_out / d_operand = 0` under these semantics.
pub fn linearize_scatter(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    config: &ScatterConfig,
) -> Vec<Option<LocalValId>> {
    match tangent_in[2] {
        Some(d_updates) => {
            let out = builder.add_op(
                StdTensorOp::Scatter(config.clone()),
                vec![
                    ValRef::External(primal_in[0].clone()),
                    ValRef::External(primal_in[1].clone()),
                    ValRef::Local(d_updates),
                ],
                OpMode::Linear {
                    active_mask: vec![false, false, true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

/// Reverse-mode AD rule for `Scatter(operand, scatter_indices, updates, config)`.
///
/// Because the scatter semantics initialise the output with zeros (using
/// `operand` only for its shape), the operand makes no contribution to the
/// output value and its cotangent is `None`. Updates flow back through the
/// inverse `Gather`: `Gather(cot_out, scatter_indices, inverse_config)`.
/// The inverse config uses the primal `updates`' shape to derive the
/// `slice_sizes` vector required by `GatherConfig`.
pub fn transpose_scatter(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &ScatterConfig,
    ctx: &ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask.clone(),
        OpMode::Primal => return vec![None, None, None],
    };

    if !active_mask.get(2).copied().unwrap_or(false) {
        return vec![None, None, None];
    }

    let operand_shape = ctx.shape_of(&inputs[0]).to_vec();
    let updates_shape = ctx.shape_of(&inputs[2]).to_vec();
    let slice_sizes = compute_inverse_slice_sizes(&operand_shape, &updates_shape, config);

    let inverse_config = GatherConfig {
        offset_dims: config.update_window_dims.clone(),
        collapsed_slice_dims: config.inserted_window_dims.clone(),
        start_index_map: config.scatter_dims_to_operand_dims.clone(),
        index_vector_dim: config.index_vector_dim,
        slice_sizes,
    };

    let out = emitter.add_op(
        StdTensorOp::Gather(inverse_config),
        vec![ValRef::Local(ct), inputs[1].clone()],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    );

    vec![None, None, Some(out[0])]
}

/// Build `slice_sizes` for the inverse `GatherConfig` used in
/// `transpose_scatter`.
///
/// For each operand dim `d`:
/// - if `d ∈ inserted_window_dims`: `slice_sizes[d] = 1` (the dim is
///   collapsed / indexed, not windowed)
/// - otherwise: `d` is the `k`-th entry of operand-window dims
///   `(0..rank) \ inserted_window_dims`; the size comes from the primal
///   updates tensor at the corresponding `update_window_dims[k]` axis
fn compute_inverse_slice_sizes(
    operand_shape: &[SymDim],
    updates_shape: &[SymDim],
    config: &ScatterConfig,
) -> Vec<usize> {
    let rank = operand_shape.len();
    let operand_window_dims: Vec<usize> = (0..rank)
        .filter(|dim| !config.inserted_window_dims.contains(dim))
        .collect();
    assert_eq!(
        operand_window_dims.len(),
        config.update_window_dims.len(),
        "transpose_scatter: update_window_dims length ({}) does not match \
         operand window dims count ({})",
        config.update_window_dims.len(),
        operand_window_dims.len(),
    );

    let mut slice_sizes = vec![1usize; rank];
    for (k, &operand_dim) in operand_window_dims.iter().enumerate() {
        let update_axis = config.update_window_dims[k];
        let dim = updates_shape
            .get(update_axis)
            .unwrap_or_else(|| {
                panic!(
                    "transpose_scatter: update_window_dims axis {} out of range for updates rank {}",
                    update_axis,
                    updates_shape.len()
                )
            })
            .constant_value()
            .unwrap_or_else(|| {
                panic!(
                    "transpose_scatter: symbolic updates dim {} cannot be used as a slice size",
                    update_axis
                )
            });
        slice_sizes[operand_dim] = dim;
    }
    slice_sizes
}
