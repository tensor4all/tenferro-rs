use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{PadConfig, SliceConfig};

use crate::ad::context::ShapeGuardContext;
use crate::ad::support::is_differentiable_dtype;
use crate::ad::zeros::build_zero_like;
use crate::ad::PrimitiveRuleBuilder;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter()
        .enumerate()
        .all(|(index, &value)| index == value)
}

fn shape_exprs_match_primal_input(
    ctx: &mut ShapeGuardContext,
    primal_in: &[ValueKey<StdTensorOp>],
    shape: &[DimExpr],
) -> bool {
    if primal_in.is_empty() || DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx > 0) {
        return false;
    }

    let input_shape = ctx
        .shape_of(&ValueRef::External(primal_in[0].clone()))
        .to_vec();
    if input_shape.len() != shape.len() {
        return false;
    }

    let input_shapes = [input_shape.as_slice()];
    shape
        .iter()
        .zip(input_shape.iter())
        .all(|(expr, dim)| SymDim::from_dim_expr(expr, &input_shapes) == dim.clone())
}

pub fn linearize_transpose(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    perm: &[usize],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            if is_identity_perm(perm) {
                return vec![Some(dx)];
            }
            let out = builder.add_operation(
                StdTensorOp::Transpose {
                    perm: perm.to_vec(),
                },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_reshape(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    op: &StdTensorOp,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let StdTensorOp::Reshape { to_shape } = op else {
        unreachable!("linearize_reshape expects Reshape");
    };

    match tangent_in[0] {
        Some(dx) => {
            if shape_exprs_match_primal_input(ctx, primal_in, to_shape) {
                return vec![Some(dx)];
            }
            let needs_shape_source =
                DimExpr::max_input_idx_all(to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValueRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValueRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_operation(
                StdTensorOp::Reshape {
                    to_shape: to_shape.clone(),
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_broadcast_in_dim(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    shape: &[DimExpr],
    dims: &[usize],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            if dims.iter().copied().eq(0..dims.len())
                && shape_exprs_match_primal_input(ctx, primal_in, shape)
            {
                return vec![Some(dx)];
            }
            let needs_shape_source = DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValueRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValueRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_operation(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_convert(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValueId>> {
    if !is_differentiable_dtype(from) || !is_differentiable_dtype(to) {
        return vec![None];
    }

    match tangent_in[0] {
        Some(dt) => {
            let out = builder.add_operation(
                StdTensorOp::Convert { from, to },
                vec![ValueRef::Local(dt)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_tril(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    k: i64,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Tril { k },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_triu(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    k: i64,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Triu { k },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_slice(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    config: &SliceConfig,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Slice(config.clone()),
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_pad(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    config: &PadConfig,
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Pad(config.clone()),
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_concatenate(
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    axis: usize,
    input_count: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    if tangent_in.iter().all(Option::is_none) {
        return vec![None];
    }
    if input_count == 1 {
        return vec![tangent_in[0]];
    }

    let mut inputs = Vec::with_capacity(input_count);
    let mut active_mask = Vec::with_capacity(input_count);
    for input_index in 0..input_count {
        match tangent_in[input_index] {
            Some(tangent) => {
                inputs.push(ValueRef::Local(tangent));
                active_mask.push(true);
            }
            None => {
                let anchor = ValueRef::External(primal_in[input_index].clone());
                let meta = ctx.metadata_of(&anchor);
                let zero = build_zero_like(builder, meta.dtype, anchor, meta.shape.len());
                inputs.push(ValueRef::Local(zero));
                active_mask.push(false);
            }
        }
    }

    let out = builder.add_operation(
        StdTensorOp::Concatenate { axis, input_count },
        inputs,
        OperationRole::Linearized { active_mask },
    );
    vec![Some(out[0])]
}

pub fn linearize_reverse(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    axes: &[usize],
) -> Vec<Option<LocalValueId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::Reverse {
                    axes: axes.to_vec(),
                },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_transpose(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    perm: &[usize],
) -> Vec<Option<LocalValueId>> {
    let mut inv = vec![0; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        inv[value] = index;
    }

    match cotangent_out[0] {
        Some(ct) => {
            if is_identity_perm(&inv) {
                return vec![Some(ct)];
            }
            let out = builder.add_operation(
                StdTensorOp::Transpose { perm: inv },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_reshape(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    op: &StdTensorOp,
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let StdTensorOp::Reshape { to_shape: _ } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };

    let mut result = Vec::with_capacity(inputs.len());
    let primary = match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.shape_of(&inputs[0]).len();
            let remapped_to_shape = DimExpr::input_shape(1, input_rank);
            let needs_shape_source =
                DimExpr::max_input_idx_all(&remapped_to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValueRef::Local(ct)];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_operation(
                StdTensorOp::Reshape {
                    to_shape: remapped_to_shape,
                },
                op_inputs,
                OperationRole::Linearized { active_mask },
            );
            Some(out[0])
        }
        None => None,
    };
    result.push(primary);
    for _ in 1..inputs.len() {
        result.push(None);
    }
    result
}

pub fn transpose_broadcast_in_dim(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    shape: &[DimExpr],
    dims: &[usize],
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let (reduce_axes, needs_input_shape_restore) =
        broadcast_transpose_reduce_axes(shape, dims, inputs, ctx);

    let primary = match cotangent_out[0] {
        Some(ct) if reduce_axes.is_empty() => Some(ct),
        Some(ct) => {
            let reduced = builder.add_operation(
                StdTensorOp::ReduceSum { axes: reduce_axes },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            )[0];
            if needs_input_shape_restore {
                let input_rank = ctx.shape_of(&inputs[0]).len();
                let reshaped = builder.add_operation(
                    StdTensorOp::Reshape {
                        to_shape: DimExpr::input_shape(1, input_rank),
                    },
                    vec![ValueRef::Local(reduced), inputs[0].clone()],
                    OperationRole::Linearized {
                        active_mask: vec![true, false],
                    },
                );
                Some(reshaped[0])
            } else {
                Some(reduced)
            }
        }
        None => None,
    };

    let mut result = Vec::with_capacity(inputs.len());
    result.push(primary);
    for _ in 1..inputs.len() {
        result.push(None);
    }
    result
}

fn broadcast_transpose_reduce_axes(
    shape: &[DimExpr],
    dims: &[usize],
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> (Vec<usize>, bool) {
    let mut reduce_axes: Vec<usize> = (0..shape.len()).filter(|dim| !dims.contains(dim)).collect();
    let Some(input_shapes) = collect_known_input_shapes(inputs, ctx) else {
        return (reduce_axes, false);
    };
    if input_shapes.is_empty()
        || DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx >= input_shapes.len())
    {
        return (reduce_axes, false);
    }

    let input_shape = &input_shapes[0];
    let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
    let output_shape: Vec<_> = shape
        .iter()
        .map(|dim| SymDim::from_dim_expr(dim, &input_shape_refs))
        .collect();

    let one = SymDim::from(1usize);
    let mut needs_input_shape_restore = false;
    for (input_axis, &output_axis) in dims.iter().enumerate() {
        if input_axis >= input_shape.len() || output_axis >= output_shape.len() {
            continue;
        }
        if input_shape[input_axis] == one && output_shape[output_axis] != one {
            reduce_axes.push(output_axis);
            needs_input_shape_restore = true;
        }
    }

    reduce_axes.sort_unstable();
    reduce_axes.dedup();
    (reduce_axes, needs_input_shape_restore)
}

fn collect_known_input_shapes(
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Option<Vec<Vec<SymDim>>> {
    let mut shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        shapes.push(ctx.try_shape_of(input)?.to_vec());
    }
    Some(shapes)
}

pub fn transpose_convert(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    mode: &OperationRole,
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValueId>> {
    if !is_differentiable_dtype(from) || !is_differentiable_dtype(to) {
        return vec![None];
    }

    let is_active = matches!(
        mode,
        OperationRole::Linearized { active_mask } if active_mask.first().copied().unwrap_or(false)
    );
    if !is_active {
        return vec![None];
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::Convert { from: to, to: from },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_tril(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    k: i64,
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::Tril { k },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_triu(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    k: i64,
) -> Vec<Option<LocalValueId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_operation(
                StdTensorOp::Triu { k },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_slice(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    config: &SliceConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None];
    };
    if !first_input_active(mode) {
        return vec![None];
    }

    let input_shape = ctx.shape_of(&inputs[0]);
    let rank = input_shape.len();
    assert_eq!(
        config.starts.len(),
        rank,
        "transpose_slice: starts rank mismatch"
    );
    assert_eq!(
        config.limits.len(),
        rank,
        "transpose_slice: limits rank mismatch"
    );
    assert_eq!(
        config.strides.len(),
        rank,
        "transpose_slice: strides rank mismatch"
    );

    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);
    let mut interior_padding = Vec::with_capacity(rank);

    for axis in 0..rank {
        let input_extent = static_dim(input_shape, axis, "transpose_slice");
        let start = config.starts[axis];
        let limit = config.limits[axis];
        let stride = config.strides[axis];
        assert!(
            stride > 0,
            "transpose_slice: stride must be positive on axis {axis}"
        );
        assert!(
            start <= limit && limit <= input_extent,
            "transpose_slice: invalid start/limit on axis {axis}"
        );

        let selected_len = if limit == start {
            0
        } else {
            (limit - start).div_ceil(stride)
        };
        let covered = if selected_len == 0 {
            0
        } else {
            (selected_len - 1) * stride + 1
        };
        let high = input_extent - start - covered;

        edge_padding_low.push(usize_to_i64(start, "transpose_slice"));
        edge_padding_high.push(usize_to_i64(high, "transpose_slice"));
        interior_padding.push(usize_to_i64(stride - 1, "transpose_slice"));
    }

    let out = builder.add_operation(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        }),
        vec![ValueRef::Local(ct)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_pad(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    config: &PadConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None];
    };
    if !first_input_active(mode) {
        return vec![None];
    }

    let input_shape = ctx.shape_of(&inputs[0]);
    let rank = input_shape.len();
    assert_eq!(
        config.edge_padding_low.len(),
        rank,
        "transpose_pad: edge_padding_low rank mismatch"
    );
    assert_eq!(
        config.edge_padding_high.len(),
        rank,
        "transpose_pad: edge_padding_high rank mismatch"
    );
    assert_eq!(
        config.interior_padding.len(),
        rank,
        "transpose_pad: interior_padding rank mismatch"
    );

    let mut starts = Vec::with_capacity(rank);
    let mut limits = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);

    for axis in 0..rank {
        let input_extent = static_dim(input_shape, axis, "transpose_pad");
        let input_extent_i = input_extent as i128;
        let low = i128::from(config.edge_padding_low[axis]);
        let high = i128::from(config.edge_padding_high[axis]);
        let interior = i128::from(config.interior_padding[axis]);
        assert!(
            interior >= 0,
            "transpose_pad: interior padding must be non-negative on axis {axis}"
        );
        let stride = interior + 1;
        let base = if input_extent == 0 {
            0
        } else {
            (input_extent_i - 1) * stride + 1
        };
        let output_extent = low + high + base;
        assert!(
            output_extent >= 0,
            "transpose_pad: negative output extent on axis {axis}"
        );

        let first_kept = if low < 0 {
            ceil_div_i128(-low, stride)
        } else {
            0
        };
        let first_dropped_after = ceil_div_i128(output_extent - low, stride);
        let j_start = clamp_i128(first_kept, 0, input_extent_i);
        let mut j_end = clamp_i128(first_dropped_after, 0, input_extent_i);
        if j_end < j_start {
            j_end = j_start;
        }

        let (slice_start, slice_limit) = if j_end > j_start {
            let start = low + j_start * stride;
            let limit = low + (j_end - 1) * stride + 1;
            (start, limit)
        } else {
            let empty = clamp_i128(low + j_start * stride, 0, output_extent);
            (empty, empty)
        };
        assert!(
            0 <= slice_start && slice_start <= slice_limit && slice_limit <= output_extent,
            "transpose_pad: invalid inverse slice on axis {axis}"
        );

        starts.push(i128_to_usize(slice_start, "transpose_pad"));
        limits.push(i128_to_usize(slice_limit, "transpose_pad"));
        strides.push(i128_to_usize(stride, "transpose_pad"));
        edge_padding_low.push(i128_to_i64(j_start, "transpose_pad"));
        edge_padding_high.push(i128_to_i64(input_extent_i - j_end, "transpose_pad"));
    }

    let sliced = builder.add_operation(
        StdTensorOp::Slice(SliceConfig {
            starts,
            limits,
            strides,
        }),
        vec![ValueRef::Local(ct)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0];

    if edge_padding_low.iter().all(|&pad| pad == 0) && edge_padding_high.iter().all(|&pad| pad == 0)
    {
        return vec![Some(sliced)];
    }

    let out = builder.add_operation(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding: vec![0; rank],
        }),
        vec![ValueRef::Local(sliced)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_reverse(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    mode: &OperationRole,
    axes: &[usize],
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None];
    };
    if !first_input_active(mode) {
        return vec![None];
    }

    let out = builder.add_operation(
        StdTensorOp::Reverse {
            axes: axes.to_vec(),
        },
        vec![ValueRef::Local(ct)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_concatenate(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    axis: usize,
    input_count: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValueId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None; input_count];
    };
    let active_mask = match mode {
        OperationRole::Linearized { active_mask } => active_mask,
        OperationRole::Primary => return vec![None; input_count],
    };

    let mut result = vec![None; input_count];
    let mut axis_offset = 0usize;
    for input_index in 0..input_count {
        let input_shape = ctx.shape_of(&inputs[input_index]);
        let rank = input_shape.len();
        assert!(
            axis < rank,
            "transpose_concatenate: axis {axis} out of bounds for rank {rank}"
        );
        let axis_extent = static_dim(input_shape, axis, "transpose_concatenate");
        if active_mask.get(input_index).copied().unwrap_or(false) {
            let starts = vec_with_axis(rank, axis, axis_offset, 0);
            let limits = input_shape
                .iter()
                .enumerate()
                .map(|(dim, _)| {
                    if dim == axis {
                        axis_offset + axis_extent
                    } else {
                        static_dim(input_shape, dim, "transpose_concatenate")
                    }
                })
                .collect();
            let out = builder.add_operation(
                StdTensorOp::Slice(SliceConfig {
                    starts,
                    limits,
                    strides: vec![1; rank],
                }),
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            result[input_index] = Some(out[0]);
        }
        axis_offset += axis_extent;
    }

    result
}

fn first_input_active(mode: &OperationRole) -> bool {
    matches!(
        mode,
        OperationRole::Linearized { active_mask } if active_mask.first().copied().unwrap_or(false)
    )
}

fn vec_with_axis(rank: usize, axis: usize, axis_value: usize, other_value: usize) -> Vec<usize> {
    (0..rank)
        .map(|dim| if dim == axis { axis_value } else { other_value })
        .collect()
}

fn static_dim(shape: &[SymDim], axis: usize, op: &str) -> usize {
    shape[axis]
        .constant_value()
        .unwrap_or_else(|| panic!("{op}: symbolic input dim {axis} is unsupported"))
}

fn ceil_div_i128(numer: i128, denom: i128) -> i128 {
    assert!(denom > 0, "ceil_div_i128: denominator must be positive");
    if numer >= 0 {
        (numer + denom - 1) / denom
    } else {
        numer / denom
    }
}

fn clamp_i128(value: i128, min: i128, max: i128) -> i128 {
    value.max(min).min(max)
}

fn usize_to_i64(value: usize, op: &str) -> i64 {
    i64::try_from(value).unwrap_or_else(|_| panic!("{op}: usize value does not fit in i64"))
}

fn i128_to_usize(value: i128, op: &str) -> usize {
    usize::try_from(value).unwrap_or_else(|_| panic!("{op}: i128 value does not fit in usize"))
}

fn i128_to_i64(value: i128, op: &str) -> i64 {
    i64::try_from(value).unwrap_or_else(|_| panic!("{op}: i128 value does not fit in i64"))
}
