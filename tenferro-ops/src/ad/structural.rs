use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{PadConfig, SliceConfig};

use crate::ad::context::ShapeGuardContext;
use crate::ad::zeros::build_zero_like;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

pub fn linearize_transpose(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Transpose {
                    perm: perm.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape } = op else {
        unreachable!("linearize_reshape expects Reshape");
    };

    match tangent_in[0] {
        Some(dx) => {
            let needs_shape_source =
                DimExpr::max_input_idx_all(to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    to_shape: to_shape.clone(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let needs_shape_source = DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(dx)];
            let active_mask = if needs_shape_source {
                op_inputs.push(ValRef::External(primal_in[1].clone()));
                vec![true, false]
            } else {
                vec![true]
            };
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_convert(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dt) => {
            let out = builder.add_op(
                StdTensorOp::Convert { from, to },
                vec![ValRef::Local(dt)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_tril(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Tril { k },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_triu(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Triu { k },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_slice(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    config: &SliceConfig,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Slice(config.clone()),
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_pad(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    config: &PadConfig,
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Pad(config.clone()),
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_concatenate(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axis: usize,
    n_inputs: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    if tangent_in.iter().all(Option::is_none) {
        return vec![None];
    }
    if n_inputs == 1 {
        return vec![tangent_in[0]];
    }

    let mut inputs = Vec::with_capacity(n_inputs);
    let mut active_mask = Vec::with_capacity(n_inputs);
    for input_index in 0..n_inputs {
        match tangent_in[input_index] {
            Some(tangent) => {
                inputs.push(ValRef::Local(tangent));
                active_mask.push(true);
            }
            None => {
                let anchor = ValRef::External(primal_in[input_index].clone());
                let meta = ctx.metadata_of(&anchor);
                let zero = build_zero_like(builder, meta.dtype, anchor, meta.shape.len());
                inputs.push(ValRef::Local(zero));
                active_mask.push(false);
            }
        }
    }

    let out = builder.add_op(
        StdTensorOp::Concatenate { axis, n_inputs },
        inputs,
        OpMode::Linear { active_mask },
    );
    vec![Some(out[0])]
}

pub fn linearize_reverse(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    axes: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Reverse {
                    axes: axes.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_transpose(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    let mut inv = vec![0; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        inv[value] = index;
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Transpose { perm: inv },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_reshape(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape: _ } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.shape_of(&inputs[0]).len();
            let remapped_to_shape = DimExpr::input_shape(1, input_rank);
            let needs_shape_source =
                DimExpr::max_input_idx_all(&remapped_to_shape).is_some_and(|idx| idx > 0);
            let mut op_inputs = vec![ValRef::Local(ct)];
            let active_mask = if needs_shape_source {
                op_inputs.push(inputs[0].clone());
                vec![true, false]
            } else {
                vec![true]
            };
            let out = emitter.add_op(
                StdTensorOp::Reshape {
                    to_shape: remapped_to_shape,
                },
                op_inputs,
                OpMode::Linear { active_mask },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_broadcast_in_dim(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    let output_rank = shape.len();
    let broadcast_axes: Vec<usize> = (0..output_rank).filter(|dim| !dims.contains(dim)).collect();

    // When `shape` references input_idx > 0 the primal op carries
    // auxiliary shape-reference inputs. Those inputs contribute no
    // cotangent, but the transpose-rule contract requires one entry per
    // input. Pad with `None` for each shape-ref slot.
    let extra_inputs = DimExpr::max_input_idx_all(shape).unwrap_or(0);

    let primary = match cotangent_out[0] {
        Some(ct) if broadcast_axes.is_empty() => Some(ct),
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::ReduceSum {
                    axes: broadcast_axes,
                },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            Some(out[0])
        }
        None => None,
    };

    let mut result = Vec::with_capacity(1 + extra_inputs);
    result.push(primary);
    for _ in 0..extra_inputs {
        result.push(None);
    }
    result
}

pub fn transpose_convert(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
    from: tenferro_tensor::DType,
    to: tenferro_tensor::DType,
) -> Vec<Option<LocalValId>> {
    let is_active = matches!(
        mode,
        OpMode::Linear { active_mask } if active_mask.first().copied().unwrap_or(false)
    );
    if !is_active {
        return vec![None];
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Convert { from: to, to: from },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_tril(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Tril { k },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_triu(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    k: i64,
) -> Vec<Option<LocalValId>> {
    match cotangent_out[0] {
        Some(ct) => {
            let out = emitter.add_op(
                StdTensorOp::Triu { k },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_slice(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &SliceConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
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
            (limit - start + stride - 1) / stride
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

    let out = emitter.add_op(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        }),
        vec![ValRef::Local(ct)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_pad(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    config: &PadConfig,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
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

    let sliced = emitter.add_op(
        StdTensorOp::Slice(SliceConfig {
            starts,
            limits,
            strides,
        }),
        vec![ValRef::Local(ct)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];

    if edge_padding_low.iter().all(|&pad| pad == 0) && edge_padding_high.iter().all(|&pad| pad == 0)
    {
        return vec![Some(sliced)];
    }

    let out = emitter.add_op(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding: vec![0; rank],
        }),
        vec![ValRef::Local(sliced)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_reverse(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    mode: &OpMode,
    axes: &[usize],
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None];
    };
    if !first_input_active(mode) {
        return vec![None];
    }

    let out = emitter.add_op(
        StdTensorOp::Reverse {
            axes: axes.to_vec(),
        },
        vec![ValRef::Local(ct)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );
    vec![Some(out[0])]
}

pub fn transpose_concatenate(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    axis: usize,
    n_inputs: usize,
    ctx: &mut ShapeGuardContext,
) -> Vec<Option<LocalValId>> {
    let Some(ct) = cotangent_out[0] else {
        return vec![None; n_inputs];
    };
    let active_mask = match mode {
        OpMode::Linear { active_mask } => active_mask,
        OpMode::Primal => return vec![None; n_inputs],
    };

    let mut result = vec![None; n_inputs];
    let mut axis_offset = 0usize;
    for input_index in 0..n_inputs {
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
            let out = emitter.add_op(
                StdTensorOp::Slice(SliceConfig {
                    starts,
                    limits,
                    strides: vec![1; rank],
                }),
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            result[input_index] = Some(out[0]);
        }
        axis_offset += axis_extent;
    }

    result
}

fn first_input_active(mode: &OpMode) -> bool {
    matches!(
        mode,
        OpMode::Linear { active_mask } if active_mask.first().copied().unwrap_or(false)
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
