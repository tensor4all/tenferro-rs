use tenferro_runtime::{PadConfig, SliceConfig};

use super::*;

pub(super) fn linearize_concatenate(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
    axis: usize,
    input_count: usize,
) -> Result<AdValue, SemanticAdTransformError> {
    if tangent_inputs
        .iter()
        .all(|tangent| *tangent == AdValue::Absent)
    {
        return Ok(AdValue::Absent);
    }
    let mut inputs = Vec::with_capacity(input_count);
    for input_index in 0..input_count {
        match tangent_inputs[input_index] {
            AdValue::Value(tangent) => inputs.push(tangent),
            AdValue::Absent => {
                inputs.push(zero_like(
                    builder,
                    primal_inputs[input_index],
                    SemanticTransformRole::Jvp,
                )?);
            }
        }
    }
    Ok(AdValue::Value(
        builder.add_op(CoreSemanticOp::Concatenate { axis, input_count }, &inputs)?[0],
    ))
}

pub(super) fn slice_vjp(
    builder: &mut SemanticProgramBuilder,
    input: ProgramValue,
    cotangent: AdValue,
    active: bool,
    config: &SliceConfig,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    if !active {
        return Ok(vec![AdValue::Absent]);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent]);
    };
    let input_shape = const_usize_shape_for_inverse_slice_padding(builder, input, "slice input")?;
    let rank = input_shape.len();
    if config.starts.len() != rank || config.limits.len() != rank || config.strides.len() != rank {
        return Err(metadata_error(
            SemanticTransformRole::Vjp,
            "slice config rank does not match its input rank",
        ));
    }
    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);
    let mut interior_padding = Vec::with_capacity(rank);
    for (axis, input_extent) in input_shape.iter().copied().enumerate() {
        let start = config.starts[axis];
        let limit = config.limits[axis];
        let stride = config.strides[axis];
        if stride == 0 || start > limit || limit > input_extent {
            return Err(metadata_error(
                SemanticTransformRole::Vjp,
                "slice config is invalid for its concrete input shape",
            ));
        }
        let selected_len = if limit == start {
            0
        } else {
            (limit - start).div_ceil(stride)
        };
        let covered = if selected_len == 0 {
            0
        } else {
            (selected_len - 1)
                .checked_mul(stride)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    metadata_error(
                        SemanticTransformRole::Vjp,
                        "slice inverse padding overflowed",
                    )
                })?
        };
        edge_padding_low.push(usize_to_i64(start, "slice start")?);
        edge_padding_high.push(usize_to_i64(
            input_extent - start - covered,
            "slice high padding",
        )?);
        interior_padding.push(usize_to_i64(stride - 1, "slice interior padding")?);
    }
    let value = builder.add_op(
        CoreSemanticOp::Pad(PadConfig {
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        }),
        &[cotangent],
    )?[0];
    Ok(vec![normalize_ad_value(
        builder,
        AdValue::Value(value),
        true,
        input,
    )?])
}

pub(super) fn pad_vjp(
    builder: &mut SemanticProgramBuilder,
    input: ProgramValue,
    cotangent: AdValue,
    active: bool,
    config: &PadConfig,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    if !active {
        return Ok(vec![AdValue::Absent]);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent]);
    };
    let input_shape = exact_usize_shape(builder, input, "pad input")?;
    let rank = input_shape.len();
    if config.edge_padding_low.len() != rank
        || config.edge_padding_high.len() != rank
        || config.interior_padding.len() != rank
    {
        return Err(metadata_error(
            SemanticTransformRole::Vjp,
            "pad config rank does not match its input rank",
        ));
    }

    let mut starts = Vec::with_capacity(rank);
    let mut limits = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    let mut edge_padding_low = Vec::with_capacity(rank);
    let mut edge_padding_high = Vec::with_capacity(rank);
    for (axis, input_extent) in input_shape.iter().copied().enumerate() {
        let input_extent_i = input_extent as i128;
        let low = i128::from(config.edge_padding_low[axis]);
        let high = i128::from(config.edge_padding_high[axis]);
        let interior = i128::from(config.interior_padding[axis]);
        if interior < 0 {
            return Err(metadata_error(
                SemanticTransformRole::Vjp,
                "negative interior padding has no semantic transpose",
            ));
        }
        let stride = interior + 1;
        let base = if input_extent == 0 {
            0
        } else {
            (input_extent_i - 1) * stride + 1
        };
        let output_extent = low + high + base;
        if output_extent < 0 {
            return Err(metadata_error(
                SemanticTransformRole::Vjp,
                "pad output extent is negative",
            ));
        }
        let first_kept = if low < 0 { ceil_div(-low, stride)? } else { 0 };
        let first_dropped_after = ceil_div(output_extent - low, stride)?;
        let start_index = first_kept.clamp(0, input_extent_i);
        let end_index = first_dropped_after
            .clamp(0, input_extent_i)
            .max(start_index);
        let (slice_start, slice_limit) = if end_index > start_index {
            (
                low + start_index * stride,
                low + (end_index - 1) * stride + 1,
            )
        } else {
            let empty = (low + start_index * stride).clamp(0, output_extent);
            (empty, empty)
        };
        if !(0 <= slice_start && slice_start <= slice_limit && slice_limit <= output_extent) {
            return Err(metadata_error(
                SemanticTransformRole::Vjp,
                "pad transpose produced an invalid slice",
            ));
        }
        starts.push(i128_to_usize(slice_start, "pad transpose slice start")?);
        limits.push(i128_to_usize(slice_limit, "pad transpose slice limit")?);
        strides.push(i128_to_usize(stride, "pad transpose slice stride")?);
        edge_padding_low.push(i128_to_i64(start_index, "pad transpose low padding")?);
        edge_padding_high.push(i128_to_i64(
            input_extent_i - end_index,
            "pad transpose high padding",
        )?);
    }
    let sliced = builder.add_op(
        CoreSemanticOp::Slice(SliceConfig {
            starts,
            limits,
            strides,
        }),
        &[cotangent],
    )?[0];
    let value = if edge_padding_low.iter().all(|padding| *padding == 0)
        && edge_padding_high.iter().all(|padding| *padding == 0)
    {
        sliced
    } else {
        builder.add_op(
            CoreSemanticOp::Pad(PadConfig {
                edge_padding_low,
                edge_padding_high,
                interior_padding: vec![0; rank],
            }),
            &[sliced],
        )?[0]
    };
    Ok(vec![normalize_ad_value(
        builder,
        AdValue::Value(value),
        true,
        input,
    )?])
}

pub(super) fn concatenate_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    axis: usize,
    input_count: usize,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent; input_count]);
    };
    let mut result = vec![AdValue::Absent; input_count];
    let mut axis_offset = 0usize;
    for input_index in 0..input_count {
        let input_shape =
            exact_usize_shape(builder, primal_inputs[input_index], "concatenate input")?;
        if axis >= input_shape.len() {
            return Err(metadata_error(
                SemanticTransformRole::Vjp,
                "concatenate axis is outside its input rank",
            ));
        }
        let next_axis_offset = axis_offset.checked_add(input_shape[axis]).ok_or_else(|| {
            metadata_error(
                SemanticTransformRole::Vjp,
                "concatenate cotangent offset overflowed",
            )
        })?;
        if active_inputs[input_index] {
            let mut starts = vec![0; input_shape.len()];
            starts[axis] = axis_offset;
            let mut limits = input_shape.clone();
            limits[axis] = next_axis_offset;
            let value = builder.add_op(
                CoreSemanticOp::Slice(SliceConfig {
                    starts,
                    limits,
                    strides: vec![1; input_shape.len()],
                }),
                &[cotangent],
            )?[0];
            result[input_index] = normalize_ad_value(
                builder,
                AdValue::Value(value),
                true,
                primal_inputs[input_index],
            )?;
        }
        axis_offset = next_axis_offset;
    }
    Ok(result)
}

fn zero_like(
    builder: &mut SemanticProgramBuilder,
    anchor: ProgramValue,
    role: SemanticTransformRole,
) -> Result<ProgramValue, SemanticAdTransformError> {
    let one = one_like(builder, anchor, role)?;
    Ok(builder.add_op(CoreSemanticOp::Sub, &[one, one])?[0])
}

fn exact_usize_shape(
    builder: &SemanticProgramBuilder,
    value: ProgramValue,
    field: &'static str,
) -> Result<Vec<usize>, SemanticAdTransformError> {
    exact_value_shape(builder, value, SemanticTransformRole::Vjp, field)?
        .into_iter()
        .map(|extent| match extent {
            tenferro_ops::dim_expr::DimExpr::Const(value) => Ok(value),
            _ => Err(metadata_error(
                SemanticTransformRole::Vjp,
                format!("{field} requires concrete extents"),
            )),
        })
        .collect()
}

fn const_usize_shape_for_inverse_slice_padding(
    builder: &SemanticProgramBuilder,
    value: ProgramValue,
    field: &'static str,
) -> Result<Vec<usize>, SemanticAdTransformError> {
    builder
        .value_metadata(value)?
        .shape()
        .iter()
        .map(|extent| match extent {
            ShapeExtent::Exact(DimExpr::Const(value))
            | ShapeExtent::UpperBound(DimExpr::Const(value)) => Ok(*value),
            ShapeExtent::Exact(_) | ShapeExtent::UpperBound(_) | ShapeExtent::Unknown => {
                Err(metadata_error(
                    SemanticTransformRole::Vjp,
                    format!("{field} requires constant exact or upper-bound extents"),
                ))
            }
        })
        .collect()
}

fn ceil_div(numerator: i128, denominator: i128) -> Result<i128, SemanticAdTransformError> {
    if denominator <= 0 {
        return Err(metadata_error(
            SemanticTransformRole::Vjp,
            "pad transpose requires a positive stride",
        ));
    }
    Ok(numerator.div_euclid(denominator) + i128::from(numerator.rem_euclid(denominator) != 0))
}

fn usize_to_i64(value: usize, field: &'static str) -> Result<i64, SemanticAdTransformError> {
    i64::try_from(value).map_err(|_| {
        metadata_error(
            SemanticTransformRole::Vjp,
            format!("{field} does not fit in i64"),
        )
    })
}

fn i128_to_usize(value: i128, field: &'static str) -> Result<usize, SemanticAdTransformError> {
    usize::try_from(value).map_err(|_| {
        metadata_error(
            SemanticTransformRole::Vjp,
            format!("{field} does not fit in usize"),
        )
    })
}

fn i128_to_i64(value: i128, field: &'static str) -> Result<i64, SemanticAdTransformError> {
    i64::try_from(value).map_err(|_| {
        metadata_error(
            SemanticTransformRole::Vjp,
            format!("{field} does not fit in i64"),
        )
    })
}

fn metadata_error(
    role: SemanticTransformRole,
    message: impl Into<String>,
) -> SemanticAdTransformError {
    SemanticAdTransformError::UnsupportedMetadata {
        role,
        message: message.into(),
    }
}
