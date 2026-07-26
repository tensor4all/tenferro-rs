use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ShapeExtent;
use tenferro_runtime::SliceConfig;

use super::*;

pub(super) fn linearize_dynamic_shape(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent: AdValue,
) -> Result<AdValue, SemanticAdTransformError> {
    let AdValue::Value(tangent) = tangent else {
        return Ok(AdValue::Absent);
    };
    Ok(AdValue::Value(
        builder.add_op(op.clone(), &[tangent, primal_inputs[1]])?[0],
    ))
}

pub(super) fn dynamic_shape_vjp(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
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
    let value = match op {
        CoreSemanticOp::DynamicTruncate { axis } => builder.add_op(
            CoreSemanticOp::PadToMatch { axis: *axis },
            &[cotangent, primal_inputs[0]],
        )?[0],
        CoreSemanticOp::PadToMatch { axis } => {
            transpose_pad_to_match(builder, cotangent, primal_inputs[0], *axis)?
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Vjp, op)),
    };
    result[0] = AdValue::Value(value);
    Ok(result)
}

fn transpose_pad_to_match(
    builder: &mut SemanticProgramBuilder,
    cotangent: ProgramValue,
    input: ProgramValue,
    axis: usize,
) -> Result<ProgramValue, SemanticAdTransformError> {
    let metadata = builder.value_metadata(input)?;
    if axis >= metadata.shape().len() {
        return Err(SemanticAdTransformError::UnsupportedMetadata {
            role: SemanticTransformRole::Vjp,
            message: format!(
                "pad-to-match axis {axis} is outside input rank {}",
                metadata.shape().len()
            ),
        });
    }
    let concrete_shape = metadata
        .shape()
        .iter()
        .map(|extent| match extent {
            ShapeExtent::Exact(DimExpr::Const(value)) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if let Some(limits) = concrete_shape {
        let rank = limits.len();
        return Ok(builder.add_op(
            CoreSemanticOp::Slice(SliceConfig {
                starts: vec![0; rank],
                limits,
                strides: vec![1; rank],
            }),
            &[cotangent],
        )?[0]);
    }
    let size = builder.add_op(CoreSemanticOp::ShapeOf { axis }, &[input])?[0];
    Ok(builder.add_op(CoreSemanticOp::DynamicTruncate { axis }, &[cotangent, size])?[0])
}
