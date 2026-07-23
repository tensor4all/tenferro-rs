use super::*;

pub(super) fn linearize_nonlinear_reduction(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent: AdValue,
) -> Result<AdValue, SemanticAdTransformError> {
    let AdValue::Value(tangent) = tangent else {
        return Ok(AdValue::Absent);
    };
    let input = primal_inputs[0];
    let axes = reduction_axes(op)?;
    let answer = builder.add_op(op.clone(), &[input])?[0];
    let (coefficient, divisor) = match op {
        CoreSemanticOp::ReduceProd { .. } => (
            reduce_prod_derivative_coefficient(builder, input, answer, axes)?,
            None,
        ),
        CoreSemanticOp::ReduceMax { .. } | CoreSemanticOp::ReduceMin { .. } => {
            let (indicators, counts) = reduction_chooser_indicators(builder, input, answer, axes)?;
            (indicators, Some(counts))
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Jvp, op)),
    };
    let weighted = builder.add_op(CoreSemanticOp::Mul, &[coefficient, tangent])?[0];
    let tangent_sum = builder.add_op(
        CoreSemanticOp::ReduceSum {
            axes: axes.to_vec(),
        },
        &[weighted],
    )?[0];
    if let Some(counts) = divisor {
        Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Div, &[tangent_sum, counts])?[0],
        ))
    } else {
        Ok(AdValue::Value(tangent_sum))
    }
}

pub(super) fn nonlinear_reduction_vjp(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active: bool,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    if !active {
        return Ok(vec![AdValue::Absent]);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent]);
    };
    let input = primal_inputs[0];
    let axes = reduction_axes(op)?;
    let answer = builder.add_op(op.clone(), &[input])?[0];
    let cotangent = broadcast_reduction_output(builder, cotangent, input, axes)?;
    let coefficient = match op {
        CoreSemanticOp::ReduceProd { .. } => {
            reduce_prod_derivative_coefficient(builder, input, answer, axes)?
        }
        CoreSemanticOp::ReduceMax { .. } | CoreSemanticOp::ReduceMin { .. } => {
            let (indicators, counts) = reduction_chooser_indicators(builder, input, answer, axes)?;
            let counts = broadcast_reduction_output(builder, counts, input, axes)?;
            builder.add_op(CoreSemanticOp::Div, &[indicators, counts])?[0]
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Vjp, op)),
    };
    let coefficient = conjugate_if_complex(builder, coefficient)?;
    let result = builder.add_op(CoreSemanticOp::Mul, &[coefficient, cotangent])?[0];
    Ok(vec![normalize_ad_value(
        builder,
        AdValue::Value(result),
        true,
        input,
    )?])
}

fn reduction_axes(op: &CoreSemanticOp) -> Result<&[usize], SemanticAdTransformError> {
    match op {
        CoreSemanticOp::ReduceProd { axes }
        | CoreSemanticOp::ReduceMax { axes }
        | CoreSemanticOp::ReduceMin { axes } => Ok(axes),
        _ => Err(unsupported_core(SemanticTransformRole::Jvp, op)),
    }
}

fn broadcast_reduction_output(
    builder: &mut SemanticProgramBuilder,
    output: ProgramValue,
    input: ProgramValue,
    axes: &[usize],
) -> Result<ProgramValue, SemanticAdTransformError> {
    let input_shape = exact_value_shape(
        builder,
        input,
        SemanticTransformRole::Vjp,
        "nonlinear reduction input",
    )?;
    let kept_dims = (0..input_shape.len())
        .filter(|axis| !axes.contains(axis))
        .collect();
    Ok(builder.add_op(
        CoreSemanticOp::BroadcastInDim {
            shape: input_shape,
            dims: kept_dims,
        },
        &[output],
    )?[0])
}

fn reduce_prod_derivative_coefficient(
    builder: &mut SemanticProgramBuilder,
    input: ProgramValue,
    answer: ProgramValue,
    axes: &[usize],
) -> Result<ProgramValue, SemanticAdTransformError> {
    let dtype = builder.value_metadata(input)?.dtype();
    let one = one_like(builder, input, SemanticTransformRole::Jvp)?;
    let zero = builder.add_op(CoreSemanticOp::Sub, &[one, one])?[0];
    let zero_mask = builder.add_op(CoreSemanticOp::Compare(CompareDir::Eq), &[input, zero])?[0];
    let numeric_zero_mask = builder.add_op(
        CoreSemanticOp::Convert {
            from: DType::Bool,
            to: dtype,
        },
        &[zero_mask],
    )?[0];
    let zero_count = builder.add_op(
        CoreSemanticOp::ReduceSum {
            axes: axes.to_vec(),
        },
        &[numeric_zero_mask],
    )?[0];
    let zero_count = broadcast_reduction_output(builder, zero_count, input, axes)?;
    let safe_input = builder.add_op(CoreSemanticOp::Select, &[zero_mask, one, input])?[0];
    let nonzero_prod = builder.add_op(
        CoreSemanticOp::ReduceProd {
            axes: axes.to_vec(),
        },
        &[safe_input],
    )?[0];
    let nonzero_prod = broadcast_reduction_output(builder, nonzero_prod, input, axes)?;
    let answer = broadcast_reduction_output(builder, answer, input, axes)?;
    let quotient = builder.add_op(CoreSemanticOp::Div, &[answer, safe_input])?[0];
    let single_zero_coefficient =
        builder.add_op(CoreSemanticOp::Select, &[zero_mask, nonzero_prod, zero])?[0];
    let zero_count_is_zero =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Eq), &[zero_count, zero])?[0];
    let zero_count_is_one =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Eq), &[zero_count, one])?[0];
    let zero_case = builder.add_op(
        CoreSemanticOp::Select,
        &[zero_count_is_one, single_zero_coefficient, zero],
    )?[0];
    Ok(builder.add_op(
        CoreSemanticOp::Select,
        &[zero_count_is_zero, quotient, zero_case],
    )?[0])
}

fn reduction_chooser_indicators(
    builder: &mut SemanticProgramBuilder,
    input: ProgramValue,
    answer: ProgramValue,
    axes: &[usize],
) -> Result<(ProgramValue, ProgramValue), SemanticAdTransformError> {
    let dtype = builder.value_metadata(input)?.dtype();
    let answer = broadcast_reduction_output(builder, answer, input, axes)?;
    let locations = builder.add_op(CoreSemanticOp::Compare(CompareDir::Eq), &[input, answer])?[0];
    let numeric_locations = builder.add_op(
        CoreSemanticOp::Convert {
            from: DType::Bool,
            to: dtype,
        },
        &[locations],
    )?[0];
    let counts = builder.add_op(
        CoreSemanticOp::ReduceSum {
            axes: axes.to_vec(),
        },
        &[numeric_locations],
    )?[0];
    Ok((numeric_locations, counts))
}
