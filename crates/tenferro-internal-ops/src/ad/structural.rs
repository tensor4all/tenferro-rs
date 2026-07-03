use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{PadConfig, SliceConfig};
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::ad::context::{ShapeGuardContext, ShapeGuardError};
use crate::ad::support::{is_differentiable_dtype, linear_transpose_input_active};
use crate::ad::zeros::build_zero_like;
use crate::ad::PrimitiveRuleBuilder;
use crate::dim_expr::DimExpr;
use crate::shape_extent::ShapeExtent;
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

    // Exact shape is used only for an identity fast path; non-exact metadata
    // falls back to emitting the reshape/broadcast op.
    let Ok(input_shape) = ctx.shape_of(&ValueRef::External(primal_in[0].clone())) else {
        return false;
    };
    let input_shape = input_shape.to_vec();
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
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if tangent_in.iter().all(Option::is_none) {
        return Ok(vec![None]);
    }
    if input_count == 1 {
        return Ok(vec![tangent_in[0]]);
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
                let meta = ctx.metadata_of(&anchor)?;
                let zero = build_zero_like(builder, meta.dtype, anchor, meta.rank());
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
    Ok(vec![Some(out[0])])
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
    mode: &OperationRole,
    perm: &[usize],
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !linear_transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    let mut inv = vec![0; perm.len()];
    let mut seen = vec![false; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        if value >= perm.len() {
            return Err(ADRuleError::invalid_input(
                "transpose",
                ADRuleKind::Transpose,
                format!(
                    "permutation axis {value} is out of bounds for rank {}",
                    perm.len()
                ),
            ));
        }
        if seen[value] {
            return Err(ADRuleError::invalid_input(
                "transpose",
                ADRuleKind::Transpose,
                format!("permutation axis {value} appears more than once"),
            ));
        }
        seen[value] = true;
        inv[value] = index;
    }

    Ok(match cotangent_out[0] {
        Some(ct) => {
            if is_identity_perm(&inv) {
                return Ok(vec![Some(ct)]);
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
    })
}

pub fn transpose_reshape(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    op: &StdTensorOp,
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Reshape { to_shape: _ } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };

    let mut result = Vec::with_capacity(inputs.len());
    if !linear_transpose_input_active(mode, 0) {
        result.resize(inputs.len(), None);
        return Ok(result);
    }

    let primary = match cotangent_out[0] {
        Some(ct) => {
            let input_rank = ctx.rank_of(&inputs[0])?;
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
    Ok(result)
}

pub fn transpose_broadcast_in_dim(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    shape: &[DimExpr],
    dims: &[usize],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !linear_transpose_input_active(mode, 0) {
        let mut result = Vec::with_capacity(inputs.len());
        result.resize(inputs.len(), None);
        return Ok(result);
    }

    let (reduce_axes, needs_input_shape_restore) =
        broadcast_transpose_reduce_axes(shape, dims, inputs, ctx)?;

    let primary = match cotangent_out[0] {
        Some(ct) => {
            let reduced = if reduce_axes.is_empty() {
                ct
            } else {
                builder.add_operation(
                    StdTensorOp::ReduceSum {
                        axes: reduce_axes.clone(),
                    },
                    vec![ValueRef::Local(ct)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                )[0]
            };
            let restored_order = if let Some(perm) =
                broadcast_transpose_restore_perm(shape.len(), dims, &reduce_axes)
            {
                builder.add_operation(
                    StdTensorOp::Transpose { perm },
                    vec![ValueRef::Local(reduced)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                )[0]
            } else {
                reduced
            };
            if needs_input_shape_restore {
                let input_rank = ctx.rank_of(&inputs[0])?;
                let reshaped = builder.add_operation(
                    StdTensorOp::Reshape {
                        to_shape: DimExpr::input_shape(1, input_rank),
                    },
                    vec![ValueRef::Local(restored_order), inputs[0].clone()],
                    OperationRole::Linearized {
                        active_mask: vec![true, false],
                    },
                );
                Some(reshaped[0])
            } else {
                Some(restored_order)
            }
        }
        None => None,
    };

    let mut result = Vec::with_capacity(inputs.len());
    result.push(primary);
    for _ in 1..inputs.len() {
        result.push(None);
    }
    Ok(result)
}

fn broadcast_transpose_reduce_axes(
    shape: &[DimExpr],
    dims: &[usize],
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<(Vec<usize>, bool)> {
    let mut reduce_axes: Vec<usize> = (0..shape.len()).filter(|dim| !dims.contains(dim)).collect();
    let mut needs_input_shape_restore = false;

    if let Some(input_shapes) = collect_known_input_shapes(inputs, ctx) {
        if input_shapes.is_empty()
            || DimExpr::max_input_idx_all(shape).is_some_and(|idx| idx >= input_shapes.len())
        {
            return Ok((reduce_axes, false));
        }

        let input_shape = &input_shapes[0];
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let output_shape: Vec<_> = shape
            .iter()
            .map(|dim| SymDim::from_dim_expr(dim, &input_shape_refs))
            .collect();

        let one = SymDim::from(1usize);
        for (input_axis, &output_axis) in dims.iter().enumerate() {
            if input_axis >= input_shape.len() || output_axis >= output_shape.len() {
                continue;
            }
            if input_shape[input_axis] == one && output_shape[output_axis] != one {
                reduce_axes.push(output_axis);
                needs_input_shape_restore = true;
            }
        }
    } else if let Some(input) = inputs.first() {
        match ctx.extents_of(input) {
            Ok(input_extents) => {
                let input_extents = input_extents.to_vec();
                for (input_axis, &output_axis) in dims.iter().enumerate() {
                    if input_axis >= input_extents.len()
                        || !extent_is_definitely_one(&input_extents[input_axis])
                        || !output_axis_needs_singleton_reduction(
                            shape.get(output_axis),
                            input_axis,
                        )
                    {
                        continue;
                    }
                    reduce_axes.push(output_axis);
                    needs_input_shape_restore = true;
                }
            }
            Err(ShapeGuardError::MissingMetadata { .. }) => {}
            Err(err) => return Err(err.into()),
        }
    }

    reduce_axes.sort_unstable();
    reduce_axes.dedup();
    Ok((reduce_axes, needs_input_shape_restore))
}

fn extent_is_definitely_one(extent: &ShapeExtent<SymDim>) -> bool {
    match extent {
        ShapeExtent::Exact(dim) | ShapeExtent::UpperBound(dim) => dim.constant_value() == Some(1),
        ShapeExtent::Unknown => false,
    }
}

fn output_axis_needs_singleton_reduction(output_dim: Option<&DimExpr>, input_axis: usize) -> bool {
    match output_dim {
        Some(DimExpr::Const(1)) => false,
        Some(DimExpr::InputDim { input_idx: 0, axis }) if *axis == input_axis => false,
        Some(_) => true,
        None => false,
    }
}

fn broadcast_transpose_restore_perm(
    output_rank: usize,
    dims: &[usize],
    reduce_axes: &[usize],
) -> Option<Vec<usize>> {
    // INVARIANT: `reduce_axes` and the loop domain are bounded by tensor rank;
    // ShapeVec keeps common ranks inline, so linear membership beats hashing here.
    let remaining_output_axes: Vec<_> = (0..output_rank)
        .filter(|axis| !reduce_axes.contains(axis))
        .collect();
    let mut perm = Vec::new();
    for &output_axis in dims {
        if reduce_axes.contains(&output_axis) {
            continue;
        }
        let reduced_axis = remaining_output_axes
            .iter()
            .position(|&axis| axis == output_axis)?;
        perm.push(reduced_axis);
    }
    let is_identity = perm
        .iter()
        .enumerate()
        .all(|(axis, &mapped)| axis == mapped);
    (!is_identity).then_some(perm)
}

fn collect_known_input_shapes(
    inputs: &[ValueRef<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
) -> Option<Vec<Vec<SymDim>>> {
    let mut shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        shapes.push(ctx.shape_if_available(input)?.to_vec());
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

    if !linear_transpose_input_active(mode, 0) {
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
    mode: &OperationRole,
    k: i64,
) -> Vec<Option<LocalValueId>> {
    if !linear_transpose_input_active(mode, 0) {
        return vec![None];
    }

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
    mode: &OperationRole,
    k: i64,
) -> Vec<Option<LocalValueId>> {
    if !linear_transpose_input_active(mode, 0) {
        return vec![None];
    }

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
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None]);
    };
    if !first_input_active(mode) {
        return Ok(vec![None]);
    }

    // The inverse is a concrete Pad payload, so exact input extents are
    // required; non-static inputs are unsupported by this transpose rule.
    let input_shape = ctx.shape_of(&inputs[0])?;
    let rank = input_shape.len();
    if config.starts.len() != rank || config.limits.len() != rank || config.strides.len() != rank {
        return Ok(vec![None]);
    }

    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);
    let mut interior_padding = Vec::with_capacity(rank);

    for axis in 0..rank {
        let Some(input_extent) = try_static_dim(&input_shape, axis) else {
            return Ok(vec![None]);
        };
        let start = config.starts[axis];
        let limit = config.limits[axis];
        let stride = config.strides[axis];
        if stride == 0 || start > limit || limit > input_extent {
            return Ok(vec![None]);
        }

        let selected_len = if limit == start {
            0
        } else {
            (limit - start).div_ceil(stride)
        };
        let covered = if selected_len == 0 {
            0
        } else {
            let Some(covered_minus_one) = (selected_len - 1).checked_mul(stride) else {
                return Ok(vec![None]);
            };
            let Some(covered) = covered_minus_one.checked_add(1) else {
                return Ok(vec![None]);
            };
            covered
        };
        let high = input_extent - start - covered;

        let (Some(low), Some(high), Some(interior)) = (
            try_usize_to_i64(start),
            try_usize_to_i64(high),
            try_usize_to_i64(stride - 1),
        ) else {
            return Ok(vec![None]);
        };
        edge_padding_low.push(low);
        edge_padding_high.push(high);
        interior_padding.push(interior);
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
    Ok(vec![Some(out[0])])
}

pub fn transpose_pad(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    config: &PadConfig,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None]);
    };
    if !first_input_active(mode) {
        return Ok(vec![None]);
    }

    // The inverse is a concrete Slice payload, so exact input extents are
    // required; non-static inputs are unsupported by this transpose rule.
    let input_shape = ctx.shape_of(&inputs[0])?;
    let rank = input_shape.len();
    if config.edge_padding_low.len() != rank
        || config.edge_padding_high.len() != rank
        || config.interior_padding.len() != rank
    {
        return Ok(vec![None]);
    }

    let mut starts = Vec::with_capacity(rank);
    let mut limits = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);

    for axis in 0..rank {
        let Some(input_extent) = try_static_dim(&input_shape, axis) else {
            return Ok(vec![None]);
        };
        let input_extent_i = input_extent as i128;
        let low = i128::from(config.edge_padding_low[axis]);
        let high = i128::from(config.edge_padding_high[axis]);
        let interior = i128::from(config.interior_padding[axis]);
        if interior < 0 {
            return Ok(vec![None]);
        }
        let stride = interior + 1;
        let base = if input_extent == 0 {
            0
        } else {
            (input_extent_i - 1) * stride + 1
        };
        let output_extent = low + high + base;
        if output_extent < 0 {
            return Ok(vec![None]);
        }

        let first_kept = if low < 0 {
            let Some(first_kept) = try_ceil_div_i128(-low, stride) else {
                return Ok(vec![None]);
            };
            first_kept
        } else {
            0
        };
        let Some(first_dropped_after) = try_ceil_div_i128(output_extent - low, stride) else {
            return Ok(vec![None]);
        };
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
        if !(0 <= slice_start && slice_start <= slice_limit && slice_limit <= output_extent) {
            return Ok(vec![None]);
        }

        let (Some(start), Some(limit), Some(stride), Some(low), Some(high)) = (
            try_i128_to_usize(slice_start),
            try_i128_to_usize(slice_limit),
            try_i128_to_usize(stride),
            try_i128_to_i64(j_start),
            try_i128_to_i64(input_extent_i - j_end),
        ) else {
            return Ok(vec![None]);
        };
        starts.push(start);
        limits.push(limit);
        strides.push(stride);
        edge_padding_low.push(low);
        edge_padding_high.push(high);
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
        return Ok(vec![Some(sliced)]);
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
    Ok(vec![Some(out[0])])
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
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some(ct) = cotangent_out[0] else {
        return Ok(vec![None; input_count]);
    };
    let mut result = vec![None; input_count];
    let mut axis_offset = 0usize;
    let mut concrete_shapes = Vec::with_capacity(input_count);
    for input in inputs.iter().take(input_count) {
        // Concatenate transpose slices a concrete cotangent segment for each
        // input, so exact axis extents are required here.
        let input_shape = ctx.shape_of(input)?;
        let rank = input_shape.len();
        if axis >= rank {
            return Ok(vec![None; input_count]);
        }
        let Some(concrete_shape) = input_shape
            .iter()
            .map(SymDim::constant_value)
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(vec![None; input_count]);
        };
        concrete_shapes.push(concrete_shape);
    }
    if concrete_shapes.len() != input_count {
        return Ok(vec![None; input_count]);
    }

    for input_index in 0..input_count {
        let input_shape = &concrete_shapes[input_index];
        let rank = input_shape.len();
        let axis_extent = input_shape[axis];
        let Some(next_axis_offset) = axis_offset.checked_add(axis_extent) else {
            return Ok(vec![None; input_count]);
        };
        if linear_transpose_input_active(mode, input_index) {
            let starts = vec_with_axis(rank, axis, axis_offset, 0);
            let limits = input_shape
                .iter()
                .copied()
                .enumerate()
                .map(|(dim, extent)| {
                    if dim == axis {
                        next_axis_offset
                    } else {
                        extent
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
        axis_offset = next_axis_offset;
    }

    Ok(result)
}

fn first_input_active(mode: &OperationRole) -> bool {
    linear_transpose_input_active(mode, 0)
}

fn vec_with_axis(rank: usize, axis: usize, axis_value: usize, other_value: usize) -> Vec<usize> {
    (0..rank)
        .map(|dim| if dim == axis { axis_value } else { other_value })
        .collect()
}

fn try_static_dim(shape: &[SymDim], axis: usize) -> Option<usize> {
    shape.get(axis)?.constant_value()
}

fn try_ceil_div_i128(numer: i128, denom: i128) -> Option<i128> {
    if denom <= 0 {
        return None;
    }
    if numer >= 0 {
        Some(numer.checked_add(denom - 1)? / denom)
    } else {
        Some(numer / denom)
    }
}

fn clamp_i128(value: i128, min: i128, max: i128) -> i128 {
    value.max(min).min(max)
}

fn try_usize_to_i64(value: usize) -> Option<i64> {
    i64::try_from(value).ok()
}

fn try_i128_to_usize(value: i128) -> Option<usize> {
    usize::try_from(value).ok()
}

fn try_i128_to_i64(value: i128) -> Option<i64> {
    i64::try_from(value).ok()
}
