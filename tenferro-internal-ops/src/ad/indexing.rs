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

use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
use crate::ad::zeros::build_zero_like;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

/// Forward-mode AD rule for `Gather(operand, start_indices, config)`.
///
/// `start_indices` is integer-valued and non-differentiable, so its tangent
/// is ignored. The output tangent is the same gather applied to the
/// operand's tangent, reusing the primal indices.
pub fn linearize_gather(
    builder: &mut dyn OpEmitter<StdTensorOp>,
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

/// Forward-mode AD rule for `GatherDynamicSliceSizes`.
///
/// The operand tangent flows through the same gather. Start indices and all
/// shape-source inputs are non-differentiable and are reused from the primal.
#[allow(clippy::too_many_arguments)]
pub fn linearize_gather_dynamic_slice_sizes(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    offset_dims: &[usize],
    collapsed_slice_dims: &[usize],
    start_index_map: &[usize],
    index_vector_dim: usize,
    slice_sizes: &[DimExpr],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(d_operand) => {
            let mut inputs = Vec::with_capacity(primal_in.len());
            inputs.push(ValRef::Local(d_operand));
            inputs.extend(primal_in.iter().skip(1).cloned().map(ValRef::External));

            let mut active_mask = vec![false; primal_in.len()];
            active_mask[0] = true;

            let out = builder.add_op(
                StdTensorOp::GatherDynamicSliceSizes {
                    offset_dims: offset_dims.to_vec(),
                    collapsed_slice_dims: collapsed_slice_dims.to_vec(),
                    start_index_map: start_index_map.to_vec(),
                    index_vector_dim,
                    slice_sizes: slice_sizes.to_vec(),
                },
                inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

/// Forward-mode AD rule for `DynamicSlice`.
///
/// The source tensor tangent flows through the same dynamic slice. Runtime
/// start indices are integer-valued and non-differentiable.
pub fn linearize_dynamic_slice(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    slice_sizes: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(d_operand) => {
            let out = builder.add_op(
                StdTensorOp::DynamicSlice {
                    slice_sizes: slice_sizes.to_vec(),
                },
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

/// Forward-mode AD rule for `DynamicUpdateSlice`.
///
/// The operand and update inputs are differentiable. Runtime start indices are
/// integer-valued and non-differentiable.
pub fn linearize_dynamic_update_slice(
    builder: &mut dyn OpEmitter<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    if tangent_in[0].is_none() && tangent_in[1].is_none() {
        return vec![None];
    }

    let operand = ValRef::External(primal_in[0].clone());
    let update = ValRef::External(primal_in[1].clone());
    let d_operand = tangent_in[0].unwrap_or_else(|| {
        let meta = ctx.metadata_of(&operand);
        build_zero_like(builder, meta.dtype, operand, meta.shape.len())
    });
    let d_update = tangent_in[1].unwrap_or_else(|| {
        let meta = ctx.metadata_of(&update);
        build_zero_like(builder, meta.dtype, update, meta.shape.len())
    });

    let out = builder.add_op(
        StdTensorOp::DynamicUpdateSlice,
        vec![
            ValRef::Local(d_operand),
            ValRef::Local(d_update),
            ValRef::External(primal_in[2].clone()),
        ],
        OpMode::Linear {
            active_mask: vec![tangent_in[0].is_some(), tangent_in[1].is_some(), false],
        },
    );
    vec![Some(out[0])]
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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

/// Reverse-mode AD rule for `GatherDynamicSliceSizes`.
///
/// The operand cotangent is the same inverse scatter as for concrete
/// `Gather`. Shape-source inputs only parameterize the primal slice sizes, so
/// their cotangents are always inactive.
#[allow(clippy::too_many_arguments)]
pub fn transpose_gather_dynamic_slice_sizes(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    offset_dims: &[usize],
    collapsed_slice_dims: &[usize],
    start_index_map: &[usize],
    index_vector_dim: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let mut result = vec![None; inputs.len()];
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return result,
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return result,
    };

    if !active_mask.first().copied().unwrap_or(false) {
        return result;
    }

    let inverse_config = ScatterConfig {
        update_window_dims: offset_dims.to_vec(),
        inserted_window_dims: collapsed_slice_dims.to_vec(),
        scatter_dims_to_operand_dims: start_index_map.to_vec(),
        index_vector_dim,
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

    result[0] = Some(out[0]);
    result
}

/// Reverse-mode AD rule for `DynamicSlice`.
///
/// The transpose of `DynamicSlice(x, starts, sizes)` writes the cotangent back
/// into a zero tensor shaped like `x` using the same start-adjustment semantics.
pub fn transpose_dynamic_slice(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None],
    };
    if !active_mask.first().copied().unwrap_or(false) {
        return vec![None, None];
    }

    let operand_meta = ctx.metadata_of(&inputs[0]);
    let zero_operand = build_zero_like(
        emitter,
        operand_meta.dtype,
        inputs[0].clone(),
        operand_meta.shape.len(),
    );
    let out = emitter.add_op(
        StdTensorOp::DynamicUpdateSlice,
        vec![
            ValRef::Local(zero_operand),
            ValRef::Local(ct),
            inputs[1].clone(),
        ],
        OpMode::Linear {
            active_mask: vec![false, true, false],
        },
    );
    vec![Some(out[0]), None]
}

/// Reverse-mode AD rule for `DynamicUpdateSlice`.
///
/// Operand cotangent keeps the output cotangent outside the update window and
/// zeros the updated window. Update cotangent is the matching dynamic slice of
/// the output cotangent.
pub fn transpose_dynamic_update_slice(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let ct = match cotangent_out[0] {
        Some(ct) => ct,
        None => return vec![None, None, None],
    };

    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None, None, None],
    };

    let mut result = vec![None, None, None];
    if active_mask.first().copied().unwrap_or(false) {
        let update_meta = ctx.metadata_of(&inputs[1]);
        let zero_update = build_zero_like(
            emitter,
            update_meta.dtype,
            inputs[1].clone(),
            update_meta.shape.len(),
        );
        let operand_ct = emitter.add_op(
            StdTensorOp::DynamicUpdateSlice,
            vec![
                ValRef::Local(ct),
                ValRef::Local(zero_update),
                inputs[2].clone(),
            ],
            OpMode::Linear {
                active_mask: vec![true, false, false],
            },
        )[0];
        result[0] = Some(operand_ct);
    }

    if active_mask.get(1).copied().unwrap_or(false) {
        let update_shape = exact_usize_shape(ctx, &inputs[1], "DynamicUpdateSlice transpose");
        let update_ct = emitter.add_op(
            StdTensorOp::DynamicSlice {
                slice_sizes: update_shape,
            },
            vec![ValRef::Local(ct), inputs[2].clone()],
            OpMode::Linear {
                active_mask: vec![true, false],
            },
        )[0];
        result[1] = Some(update_ct);
    }

    result
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
    builder: &mut dyn OpEmitter<StdTensorOp>,
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
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
        let inverse = compute_inverse_gather(&operand_shape, &updates_shape, config, &inputs[2]);

        let out = match inverse {
            InverseGather::Concrete(config) => emitter.add_op(
                StdTensorOp::Gather(config),
                vec![ValRef::Local(ct), inputs[1].clone()],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            ),
            InverseGather::Dynamic {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
                shape_sources,
            } => {
                let mut gather_inputs = vec![ValRef::Local(ct), inputs[1].clone()];
                let mut active_mask = vec![true, false];
                for shape_source in shape_sources {
                    gather_inputs.push(shape_source);
                    active_mask.push(false);
                }
                emitter.add_op(
                    StdTensorOp::GatherDynamicSliceSizes {
                        offset_dims,
                        collapsed_slice_dims,
                        start_index_map,
                        index_vector_dim,
                        slice_sizes,
                    },
                    gather_inputs,
                    OpMode::Linear { active_mask },
                )
            }
        };
        result[2] = Some(out[0]);
    }

    result
}

enum InverseGather {
    Concrete(GatherConfig),
    Dynamic {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<DimExpr>,
        shape_sources: Vec<ValRef<StdTensorOp>>,
    },
}

/// Build the inverse gather used in `transpose_scatter`.
///
/// For each operand dim `d`:
/// - if `d ∈ inserted_window_dims`: `slice_sizes[d] = 1` (the dim is
///   collapsed / indexed, not windowed)
/// - otherwise: `d` is the `k`-th entry of operand-window dims
///   `(0..rank) \ inserted_window_dims`; the size comes from the primal
///   updates tensor at the corresponding `update_window_dims[k]` axis
fn compute_inverse_gather(
    operand_shape: &[SymDim],
    updates_shape: &[SymDim],
    config: &ScatterConfig,
    updates_ref: &ValRef<StdTensorOp>,
) -> InverseGather {
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

    let mut concrete_slice_sizes = vec![1usize; rank];
    let mut dynamic_slice_sizes = vec![DimExpr::Const(1); rank];
    let mut has_dynamic_slice_size = false;
    for (k, &operand_dim) in operand_window_dims.iter().enumerate() {
        let update_axis = config.update_window_dims[k];
        let dim = updates_shape.get(update_axis).unwrap_or_else(|| {
            panic!(
                "transpose_scatter: update_window_dims axis {} out of range for updates rank {}",
                update_axis,
                updates_shape.len()
            )
        });
        if let Some(value) = dim.constant_value() {
            concrete_slice_sizes[operand_dim] = value;
            dynamic_slice_sizes[operand_dim] = DimExpr::Const(value);
        } else {
            has_dynamic_slice_size = true;
            dynamic_slice_sizes[operand_dim] = DimExpr::InputDim {
                input_idx: 2,
                axis: update_axis,
            };
        }
    }

    if has_dynamic_slice_size {
        InverseGather::Dynamic {
            offset_dims: config.update_window_dims.clone(),
            collapsed_slice_dims: config.inserted_window_dims.clone(),
            start_index_map: config.scatter_dims_to_operand_dims.clone(),
            index_vector_dim: config.index_vector_dim,
            slice_sizes: dynamic_slice_sizes,
            shape_sources: vec![updates_ref.clone()],
        }
    } else {
        InverseGather::Concrete(GatherConfig {
            offset_dims: config.update_window_dims.clone(),
            collapsed_slice_dims: config.inserted_window_dims.clone(),
            start_index_map: config.scatter_dims_to_operand_dims.clone(),
            index_vector_dim: config.index_vector_dim,
            slice_sizes: concrete_slice_sizes,
        })
    }
}

fn exact_usize_shape(
    ctx: &mut ShapeGuardContext,
    value: &ValRef<StdTensorOp>,
    op_name: &'static str,
) -> Vec<usize> {
    ctx.exact_shape_of(value)
        .unwrap_or_else(|| panic!("{op_name} requires exact update shape metadata"))
        .into_iter()
        .map(|dim| {
            dim.constant_value()
                .unwrap_or_else(|| panic!("{op_name} requires concrete update shape metadata"))
        })
        .collect()
}
