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
//! Scatter uses the StableHLO add-scatter semantics: the output is
//! `operand + scatter_add(zeros, indices, updates, config)`, so the
//! output depends on both the operand (identity passthrough plus the
//! base value at scattered slots) and the updates (accumulated at the
//! scattered slots). The forward rule treats the scatter as linear in
//! both operand and updates; the backward rule sends the output
//! cotangent to the operand as an identity passthrough and to the
//! updates through the inverse `Gather`.
//!
//! Closing the core op vocabulary under AD is a Stage 7 prerequisite (the
//! tropical fused backward emits `Gather` / `Scatter` on this vocabulary).

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{DType, GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
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
/// Because the underlying scatter now follows StableHLO add-scatter
/// semantics (output starts from `operand`), the inverse scatter must
/// use a zero operand so the operand cotangent is only the sum over
/// gather reads, not `original operand + sum over gather reads`.
pub fn transpose_gather(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &GatherConfig,
    ctx: &mut ShapeGuardContext,
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

    let operand_meta = ctx.metadata_of(&inputs[0]);
    let operand_rank = operand_meta.shape.len();
    let operand_dtype = operand_meta.dtype;
    let zero_operand = build_zero_like(emitter, operand_dtype, inputs[0].clone(), operand_rank);

    let out = emitter.add_op(
        StdTensorOp::Scatter(inverse_config),
        vec![
            ValRef::Local(zero_operand),
            inputs[1].clone(),
            ValRef::Local(ct),
        ],
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
/// Under StableHLO add-scatter semantics the scatter output is
/// `operand + scatter_add(zeros, indices, updates, config)`, so the
/// forward tangent factors into an identity contribution from
/// `operand_dot` and a scattered contribution from `updates_dot`.
/// Four cases:
///
/// - Both `operand_dot = None`, `updates_dot = None` → `[None]`.
/// - Only `operand_dot = Some` → identity passthrough `[Some(operand_dot)]`
///   (scattering a zero updates tangent adds nothing).
/// - Only `updates_dot = Some` → emit
///   `Scatter(zeros_like(operand), indices, updates_dot, config)` so the
///   output tangent is only the scattered contribution.
/// - Both `Some` → emit
///   `Scatter(operand_dot, indices, updates_dot, config)`; by linearity
///   this captures both contributions in a single scatter.
pub fn linearize_scatter(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    config: &ScatterConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let d_operand = tangent_in[0];
    let d_updates = tangent_in[2];

    match (d_operand, d_updates) {
        (None, None) => vec![None],
        (Some(d_op), None) => vec![Some(d_op)],
        (None, Some(d_up)) => {
            let operand_key = ValRef::External(primal_in[0].clone());
            let operand_meta = ctx.metadata_of(&operand_key);
            let operand_rank = operand_meta.shape.len();
            let operand_dtype = operand_meta.dtype;
            let zero_operand =
                build_zero_like(builder, operand_dtype, operand_key.clone(), operand_rank);
            let out = builder.add_op(
                StdTensorOp::Scatter(config.clone()),
                vec![
                    ValRef::Local(zero_operand),
                    ValRef::External(primal_in[1].clone()),
                    ValRef::Local(d_up),
                ],
                OpMode::Linear {
                    active_mask: vec![false, false, true],
                },
            );
            vec![Some(out[0])]
        }
        (Some(d_op), Some(d_up)) => {
            let out = builder.add_op(
                StdTensorOp::Scatter(config.clone()),
                vec![
                    ValRef::Local(d_op),
                    ValRef::External(primal_in[1].clone()),
                    ValRef::Local(d_up),
                ],
                OpMode::Linear {
                    active_mask: vec![true, false, true],
                },
            );
            vec![Some(out[0])]
        }
    }
}

/// Reverse-mode AD rule for `Scatter(operand, scatter_indices, updates, config)`.
///
/// Under StableHLO add-scatter semantics the output depends on both the
/// operand (identity passthrough for non-scattered slots, plus the base
/// value at scattered slots) and the updates (accumulated at the
/// scattered slots). The cotangent therefore flows back to both:
///
/// - operand cotangent: identity passthrough of `cot_out`. The first
///   entry of the active mask gates whether it is returned.
/// - scatter_indices cotangent: always `None` (integer-valued).
/// - updates cotangent: `Gather(cot_out, scatter_indices, inverse_config)`
///   with `slice_sizes` derived from the primal `updates`' shape. The
///   third entry of the active mask gates whether it is returned.
pub fn transpose_scatter(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &ScatterConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask.clone(),
        OpMode::Primal => return vec![None, None, None],
    };

    let operand_active = active_mask.first().copied().unwrap_or(false);
    let updates_active = active_mask.get(2).copied().unwrap_or(false);

    let mut result = vec![None, None, None];

    if operand_active {
        result[0] = Some(ct);
    }

    if updates_active {
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
        result[2] = Some(out[0]);
    }

    result
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

/// Emit a zero tensor shaped like `anchor` (rank `anchor_rank`) with
/// dtype `dtype`. Uses `StdTensorOp::Constant { bytes: zeros }` for the
/// scalar, then `BroadcastInDim` reading axis sizes from `anchor`.
///
/// When `anchor_rank == 0` returns the scalar `Constant` directly.
fn build_zero_like(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    dtype: DType,
    anchor: ValRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValId {
    let zero_scalar = emitter.add_op(
        StdTensorOp::Constant {
            dtype,
            bytes: zero_bytes(dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    if anchor_rank == 0 {
        return zero_scalar;
    }
    // `BroadcastInDim` evaluates `DimExpr::InputDim { input_idx, axis }`
    // against its own inputs at execution time. Input 0 is the scalar
    // `zero_scalar`; input 1 is the anchor, whose shape we match.
    let shape: Vec<DimExpr> = (0..anchor_rank)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let out = emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        vec![ValRef::Local(zero_scalar), anchor],
        OpMode::Primal,
    );
    out[0]
}

/// Bytes for a zero value of the given dtype (little-endian, matching
/// the repr `Constant { dtype, bytes }` expects).
fn zero_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 0.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 0.0_f64.to_le_bytes().to_vec(),
        DType::I64 => 0_i64.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    }
}
