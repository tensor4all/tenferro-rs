use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ShapeExtent;
use tenferro_runtime::{GatherConfig, ScatterConfig};

use super::*;

pub(super) fn linearize_indexing(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
) -> Result<AdValue, SemanticAdTransformError> {
    match op {
        CoreSemanticOp::Gather(config) => linearize_gather(
            builder,
            primal_inputs,
            tangent_inputs[0],
            CoreSemanticOp::Gather(config.clone()),
        ),
        CoreSemanticOp::GatherDynamicSliceSizes { .. } | CoreSemanticOp::DynamicSlice { .. } => {
            linearize_gather(builder, primal_inputs, tangent_inputs[0], op.clone())
        }
        CoreSemanticOp::Scatter(config) => {
            linearize_scatter(builder, primal_inputs, tangent_inputs, config)
        }
        CoreSemanticOp::DynamicUpdateSlice => {
            linearize_dynamic_update_slice(builder, primal_inputs, tangent_inputs)
        }
        _ => Err(unsupported_core(SemanticTransformRole::Jvp, op)),
    }
}

pub(super) fn indexing_vjp(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    match op {
        CoreSemanticOp::Gather(config) => gather_vjp(
            builder,
            primal_inputs,
            cotangent,
            active_inputs,
            ScatterConfig {
                update_window_dims: config.offset_dims.clone(),
                inserted_window_dims: config.collapsed_slice_dims.clone(),
                scatter_dims_to_operand_dims: config.start_index_map.clone(),
                index_vector_dim: config.index_vector_dim,
            },
        ),
        CoreSemanticOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            ..
        } => gather_vjp(
            builder,
            primal_inputs,
            cotangent,
            active_inputs,
            ScatterConfig {
                update_window_dims: offset_dims.clone(),
                inserted_window_dims: collapsed_slice_dims.clone(),
                scatter_dims_to_operand_dims: start_index_map.clone(),
                index_vector_dim: *index_vector_dim,
            },
        ),
        CoreSemanticOp::Scatter(config) => {
            scatter_vjp(builder, primal_inputs, cotangent, active_inputs, config)
        }
        CoreSemanticOp::DynamicSlice { .. } => {
            dynamic_slice_vjp(builder, primal_inputs, cotangent, active_inputs)
        }
        CoreSemanticOp::DynamicUpdateSlice => {
            dynamic_update_slice_vjp(builder, primal_inputs, cotangent, active_inputs)
        }
        _ => Err(unsupported_core(SemanticTransformRole::Vjp, op)),
    }
}

fn linearize_gather(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent: AdValue,
    op: CoreSemanticOp,
) -> Result<AdValue, SemanticAdTransformError> {
    let AdValue::Value(tangent) = tangent else {
        return Ok(AdValue::Absent);
    };
    let mut inputs = Vec::with_capacity(primal_inputs.len());
    inputs.push(tangent);
    inputs.extend_from_slice(&primal_inputs[1..]);
    Ok(AdValue::Value(builder.add_op(op, &inputs)?[0]))
}

fn linearize_scatter(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
    config: &ScatterConfig,
) -> Result<AdValue, SemanticAdTransformError> {
    match (tangent_inputs[0], tangent_inputs[2]) {
        (AdValue::Absent, AdValue::Absent) => Ok(AdValue::Absent),
        (AdValue::Value(operand), AdValue::Absent) => Ok(AdValue::Value(operand)),
        (operand, AdValue::Value(updates)) => {
            let operand = match operand {
                AdValue::Value(operand) => operand,
                AdValue::Absent => {
                    zero_like(builder, primal_inputs[0], SemanticTransformRole::Jvp)?
                }
            };
            Ok(AdValue::Value(
                builder.add_op(
                    CoreSemanticOp::Scatter(config.clone()),
                    &[operand, primal_inputs[1], updates],
                )?[0],
            ))
        }
    }
}

fn linearize_dynamic_update_slice(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
) -> Result<AdValue, SemanticAdTransformError> {
    if matches!(tangent_inputs[0], AdValue::Absent) && matches!(tangent_inputs[1], AdValue::Absent)
    {
        return Ok(AdValue::Absent);
    }
    let operand = ad_value_or_zero(
        builder,
        tangent_inputs[0],
        primal_inputs[0],
        SemanticTransformRole::Jvp,
    )?;
    let update = ad_value_or_zero(
        builder,
        tangent_inputs[1],
        primal_inputs[1],
        SemanticTransformRole::Jvp,
    )?;
    Ok(AdValue::Value(
        builder.add_op(
            CoreSemanticOp::DynamicUpdateSlice,
            &[operand, update, primal_inputs[2]],
        )?[0],
    ))
}

fn gather_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    inverse_config: ScatterConfig,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let mut result = vec![AdValue::Absent; primal_inputs.len()];
    if !active_inputs[0] {
        return Ok(result);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(result);
    };
    let zero_operand = zero_like(builder, primal_inputs[0], SemanticTransformRole::Vjp)?;
    let value = builder.add_op(
        CoreSemanticOp::Scatter(inverse_config),
        &[zero_operand, primal_inputs[1], cotangent],
    )?[0];
    result[0] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[0])?;
    Ok(result)
}

fn scatter_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    config: &ScatterConfig,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let mut result = vec![AdValue::Absent; 3];
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(result);
    };
    if active_inputs[0] {
        result[0] = normalize_ad_value(builder, AdValue::Value(cotangent), true, primal_inputs[0])?;
    }
    if active_inputs[2] {
        let gather = inverse_gather(builder, primal_inputs[0], primal_inputs[2], config)?;
        let value = match gather {
            InverseGather::Concrete(config) => builder.add_op(
                CoreSemanticOp::Gather(config),
                &[cotangent, primal_inputs[1]],
            )?[0],
            InverseGather::Dynamic {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
            } => builder.add_op(
                CoreSemanticOp::GatherDynamicSliceSizes {
                    offset_dims,
                    collapsed_slice_dims,
                    start_index_map,
                    index_vector_dim,
                    slice_sizes,
                },
                &[cotangent, primal_inputs[1], primal_inputs[2]],
            )?[0],
        };
        result[2] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[2])?;
    }
    Ok(result)
}

fn dynamic_slice_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let mut result = vec![AdValue::Absent; 2];
    if !active_inputs[0] {
        return Ok(result);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(result);
    };
    let zero_operand = zero_like(builder, primal_inputs[0], SemanticTransformRole::Vjp)?;
    let value = builder.add_op(
        CoreSemanticOp::DynamicUpdateSlice,
        &[zero_operand, cotangent, primal_inputs[1]],
    )?[0];
    result[0] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[0])?;
    Ok(result)
}

fn dynamic_update_slice_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let mut result = vec![AdValue::Absent; 3];
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(result);
    };
    if active_inputs[0] {
        let zero_update = zero_like(builder, primal_inputs[1], SemanticTransformRole::Vjp)?;
        let value = builder.add_op(
            CoreSemanticOp::DynamicUpdateSlice,
            &[cotangent, zero_update, primal_inputs[2]],
        )?[0];
        result[0] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[0])?;
    }
    if active_inputs[1] {
        let update_shape = exact_usize_shape(builder, primal_inputs[1], "dynamic update input")?;
        let value = builder.add_op(
            CoreSemanticOp::DynamicSlice {
                slice_sizes: update_shape,
            },
            &[cotangent, primal_inputs[2]],
        )?[0];
        result[1] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[1])?;
    }
    Ok(result)
}

enum InverseGather {
    Concrete(GatherConfig),
    Dynamic {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<DimExpr>,
    },
}

fn inverse_gather(
    builder: &SemanticProgramBuilder,
    operand: ProgramValue,
    updates: ProgramValue,
    config: &ScatterConfig,
) -> Result<InverseGather, SemanticAdTransformError> {
    let operand_rank = builder.value_metadata(operand)?.shape().len();
    let updates_shape = builder.value_metadata(updates)?.shape();
    let operand_window_dims: Vec<_> = (0..operand_rank)
        .filter(|dim| !config.inserted_window_dims.contains(dim))
        .collect();
    if operand_window_dims.len() != config.update_window_dims.len() {
        return Err(metadata_error(
            "scatter window dimensions do not invert to one gather",
        ));
    }
    let mut concrete = vec![1; operand_rank];
    let mut dynamic = vec![DimExpr::Const(1); operand_rank];
    let mut has_dynamic = false;
    for (window_index, operand_dim) in operand_window_dims.into_iter().enumerate() {
        let update_axis = config.update_window_dims[window_index];
        let extent = updates_shape.get(update_axis).ok_or_else(|| {
            metadata_error("scatter update-window axis is outside the updates rank")
        })?;
        match extent {
            ShapeExtent::Exact(DimExpr::Const(value)) => {
                concrete[operand_dim] = *value;
                dynamic[operand_dim] = DimExpr::Const(*value);
            }
            ShapeExtent::Exact(_) | ShapeExtent::UpperBound(_) | ShapeExtent::Unknown => {
                has_dynamic = true;
                dynamic[operand_dim] = DimExpr::InputDim {
                    input_idx: 2,
                    axis: update_axis,
                };
            }
        }
    }
    if has_dynamic {
        Ok(InverseGather::Dynamic {
            offset_dims: config.update_window_dims.clone(),
            collapsed_slice_dims: config.inserted_window_dims.clone(),
            start_index_map: config.scatter_dims_to_operand_dims.clone(),
            index_vector_dim: config.index_vector_dim,
            slice_sizes: dynamic,
        })
    } else {
        Ok(InverseGather::Concrete(GatherConfig {
            offset_dims: config.update_window_dims.clone(),
            collapsed_slice_dims: config.inserted_window_dims.clone(),
            start_index_map: config.scatter_dims_to_operand_dims.clone(),
            index_vector_dim: config.index_vector_dim,
            slice_sizes: concrete,
        }))
    }
}

fn ad_value_or_zero(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    anchor: ProgramValue,
    role: SemanticTransformRole,
) -> Result<ProgramValue, SemanticAdTransformError> {
    match value {
        AdValue::Value(value) => Ok(value),
        AdValue::Absent => zero_like(builder, anchor, role),
    }
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
    builder
        .value_metadata(value)?
        .shape()
        .iter()
        .map(|extent| match extent {
            ShapeExtent::Exact(DimExpr::Const(value)) => Ok(*value),
            _ => Err(metadata_error(format!("{field} requires concrete extents"))),
        })
        .collect()
}

fn metadata_error(message: impl Into<String>) -> SemanticAdTransformError {
    SemanticAdTransformError::UnsupportedMetadata {
        role: SemanticTransformRole::Vjp,
        message: message.into(),
    }
}
