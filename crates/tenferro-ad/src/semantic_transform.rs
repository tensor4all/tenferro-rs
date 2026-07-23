//! Whole-program automatic differentiation over semantic SSA programs.

use std::collections::{HashMap, HashSet};

use tenferro_runtime::program::{
    CoreSemanticOp, FrozenProgram, ProgramBuildError, ProgramFinishError, ProgramImport,
    ProgramInputSpec, ProgramQueryError, ProgramValue, SemanticOpRef, SemanticProgramBuilder,
};
use tenferro_runtime::DType;

use crate::semantic_extension::{AdValue, SemanticAdError, SemanticExtensionRuleSet};

/// Semantic AD transform role used by typed diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticTransformRole {
    /// Forward-mode linearization.
    Jvp,
    /// Reverse-mode transposition.
    Vjp,
}

/// Failures produced by whole-program semantic AD.
#[derive(Debug, thiserror::Error)]
pub enum SemanticAdTransformError {
    /// An activity mask did not match the corresponding ordered value list.
    #[error("semantic {role:?} {field} expects {expected} entries, got {actual}")]
    ActivityArity {
        /// Transform role.
        role: SemanticTransformRole,
        /// Invalid mask.
        field: &'static str,
        /// Required entry count.
        expected: usize,
        /// Supplied entry count.
        actual: usize,
    },
    /// An active core operation has not yet been admitted to semantic AD.
    #[error("semantic {role:?} does not support active core operation {op}")]
    UnsupportedCore {
        /// Transform role.
        role: SemanticTransformRole,
        /// Bounded operation diagnostic.
        op: String,
    },
    /// A future semantic operation variant is unknown to this transform.
    #[error("semantic {role:?} does not support this semantic operation variant")]
    UnsupportedOperationVariant {
        /// Transform role.
        role: SemanticTransformRole,
    },
    /// Active derivative metadata is outside the admitted exact-shape subset.
    #[error("semantic {role:?} does not support derivative metadata: {message}")]
    UnsupportedMetadata {
        /// Transform role.
        role: SemanticTransformRole,
        /// Bounded metadata diagnostic.
        message: String,
    },
    /// Source-program metadata could not be queried.
    #[error("semantic AD source-program query failed: {0}")]
    Query(#[from] ProgramQueryError),
    /// Destination semantic-program construction failed.
    #[error("semantic AD program construction failed: {0}")]
    Build(#[from] ProgramBuildError),
    /// An extension-owned semantic AD rule failed.
    #[error("semantic extension AD failed: {0}")]
    Extension(#[from] SemanticAdError),
    /// The transformed program could not be frozen atomically.
    #[error("semantic AD program finalization failed: {0}")]
    Finish(#[from] ProgramFinishError),
}

/// One frozen derivative program plus ordered derivative input/output maps.
///
/// Original primal inputs retain their source order. Active derivative seed
/// inputs are appended in source order. Program outputs contain only active
/// derivative values, also in source order; `None` records an inactive value.
#[derive(Debug)]
pub struct SemanticAdProgram {
    frozen: FrozenProgram,
    derivative_input_indices: Box<[Option<usize>]>,
    derivative_output_indices: Box<[Option<usize>]>,
}

impl SemanticAdProgram {
    /// Borrow the frozen derivative program.
    pub const fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }

    /// Return transformed-program input indices for ordered derivative seeds.
    pub fn derivative_input_indices(&self) -> &[Option<usize>] {
        &self.derivative_input_indices
    }

    /// Return transformed-program output indices for ordered derivatives.
    pub fn derivative_output_indices(&self) -> &[Option<usize>] {
        &self.derivative_output_indices
    }

    /// Consume this result and return the frozen derivative program.
    pub fn into_frozen(self) -> FrozenProgram {
        self.frozen
    }
}

/// Build a forward-mode derivative program.
///
/// `active_inputs` follows source-program input order. Each active input gets
/// one appended tangent seed. The result maps source outputs to compact
/// derivative-program outputs.
///
/// # Errors
///
/// Returns typed mask, source-query, semantic-rule, build, or finish errors.
pub fn semantic_jvp(
    input: &FrozenProgram,
    active_inputs: &[bool],
    rules: &SemanticExtensionRuleSet,
) -> Result<SemanticAdProgram, SemanticAdTransformError> {
    validate_activity(
        SemanticTransformRole::Jvp,
        "active_inputs",
        input.program.inputs().len(),
        active_inputs.len(),
    )?;
    let mut builder = SemanticProgramBuilder::new();
    let values = import_source(input, &mut builder)?;
    let mut tangents = HashMap::new();
    let mut derivative_input_indices = vec![None; input.program.inputs().len()];
    let mut next_input = input.program.inputs().len();
    for (index, source) in input.program.inputs().iter().copied().enumerate() {
        if active_inputs[index] {
            let tangent = builder.input(ProgramInputSpec::from_metadata(
                input.program.value_metadata(source)?.clone(),
            ))?;
            derivative_input_indices[index] = Some(next_input);
            next_input += 1;
            tangents.insert(source, AdValue::Value(tangent));
        } else {
            tangents.insert(source, AdValue::Absent);
        }
    }

    let live = source_output_liveness(input);
    for operation in input.program.operations() {
        let tangent_inputs: Vec<_> = operation
            .inputs()
            .iter()
            .map(|value| tangents.get(value).copied().unwrap_or(AdValue::Absent))
            .collect();
        let active_outputs: Vec<_> = operation
            .outputs()
            .iter()
            .map(|value| live.contains(value))
            .collect();
        let tangent_outputs = if tangent_inputs
            .iter()
            .all(|value| matches!(value, AdValue::Absent))
        {
            vec![AdValue::Absent; operation.outputs().len()].into_boxed_slice()
        } else {
            match operation.op() {
                SemanticOpRef::Extension(_) => rules
                    .linearize_operation(
                        operation,
                        &mapped_values(operation.inputs(), &values),
                        &mapped_values(operation.outputs(), &values),
                        &tangent_inputs,
                        &active_outputs,
                        &mut builder,
                    )?
                    .tangent_outputs()
                    .into(),
                SemanticOpRef::Core(op) => linearize_core(
                    op,
                    &mapped_values(operation.inputs(), &values),
                    &tangent_inputs,
                    &mut builder,
                )?,
                _ => {
                    return Err(SemanticAdTransformError::UnsupportedOperationVariant {
                        role: SemanticTransformRole::Jvp,
                    });
                }
            }
        };
        for (source, tangent) in operation.outputs().iter().copied().zip(tangent_outputs) {
            tangents.insert(source, tangent);
        }
    }

    let outputs = input
        .program
        .outputs()
        .iter()
        .map(|value| tangents.get(value).copied().unwrap_or(AdValue::Absent))
        .collect();
    finish_derivative(builder, derivative_input_indices, outputs)
}

/// Build a reverse-mode derivative program using extension semantic rules.
///
/// `active_inputs` selects requested source-input cotangents and
/// `active_outputs` selects source outputs that receive appended cotangent
/// seeds. Both masks follow source order.
///
/// # Errors
///
/// Returns typed mask, source-query, semantic-rule, build, or finish errors.
pub fn semantic_vjp(
    input: &FrozenProgram,
    active_inputs: &[bool],
    active_outputs: &[bool],
    rules: &SemanticExtensionRuleSet,
) -> Result<SemanticAdProgram, SemanticAdTransformError> {
    validate_activity(
        SemanticTransformRole::Vjp,
        "active_inputs",
        input.program.inputs().len(),
        active_inputs.len(),
    )?;
    validate_activity(
        SemanticTransformRole::Vjp,
        "active_outputs",
        input.program.outputs().len(),
        active_outputs.len(),
    )?;
    let mut builder = SemanticProgramBuilder::new();
    let values = import_source(input, &mut builder)?;
    let forward_active = requested_input_reachability(input, active_inputs);
    let mut cotangents = HashMap::new();
    let mut derivative_input_indices = vec![None; input.program.outputs().len()];
    let mut next_input = input.program.inputs().len();
    for (index, source) in input.program.outputs().iter().copied().enumerate() {
        if active_outputs[index] {
            let cotangent = builder.input(ProgramInputSpec::from_metadata(
                input.program.value_metadata(source)?.clone(),
            ))?;
            derivative_input_indices[index] = Some(next_input);
            next_input += 1;
            accumulate_cotangent(&mut builder, &mut cotangents, source, cotangent)?;
        }
    }

    let operations: Vec<_> = input.program.operations().collect();
    for operation in operations.into_iter().rev() {
        let cotangent_outputs: Vec<_> = operation
            .outputs()
            .iter()
            .map(|value| {
                cotangents
                    .get(value)
                    .copied()
                    .map_or(AdValue::Absent, AdValue::Value)
            })
            .collect();
        if cotangent_outputs
            .iter()
            .all(|value| matches!(value, AdValue::Absent))
        {
            continue;
        }
        let active_operation_inputs: Vec<_> = operation
            .inputs()
            .iter()
            .map(|value| forward_active.contains(value))
            .collect();
        if active_operation_inputs.iter().all(|active| !active) {
            continue;
        }
        let cotangent_inputs = match operation.op() {
            SemanticOpRef::Extension(op) => {
                if rules.lookup_primal_vjp(op.family_id()).is_some() {
                    rules.primal_vjp_operation(
                        operation,
                        &mapped_values(operation.inputs(), &values),
                        &mapped_values(operation.outputs(), &values),
                        &cotangent_outputs,
                        &active_operation_inputs,
                        &mut builder,
                    )?
                } else {
                    let inactive_tangents = vec![AdValue::Absent; operation.inputs().len()];
                    let active_operation_outputs: Vec<_> = cotangent_outputs
                        .iter()
                        .map(|value| matches!(value, AdValue::Value(_)))
                        .collect();
                    let linearized = rules.linearize_operation(
                        operation,
                        &mapped_values(operation.inputs(), &values),
                        &mapped_values(operation.outputs(), &values),
                        &inactive_tangents,
                        &active_operation_outputs,
                        &mut builder,
                    )?;
                    rules.linear_transpose_operation(
                        operation,
                        &mapped_values(operation.inputs(), &values),
                        &mapped_values(operation.outputs(), &values),
                        &cotangent_outputs,
                        &active_operation_inputs,
                        linearized.residuals(),
                        &mut builder,
                    )?
                }
            }
            SemanticOpRef::Core(op) => vjp_core(
                op,
                &mapped_values(operation.inputs(), &values),
                &cotangent_outputs,
                &active_operation_inputs,
                &mut builder,
            )?,
            _ => {
                return Err(SemanticAdTransformError::UnsupportedOperationVariant {
                    role: SemanticTransformRole::Vjp,
                });
            }
        };
        for (source, cotangent) in operation.inputs().iter().copied().zip(cotangent_inputs) {
            if let AdValue::Value(cotangent) = cotangent {
                accumulate_cotangent(&mut builder, &mut cotangents, source, cotangent)?;
            }
        }
    }

    let outputs = input
        .program
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if active_inputs[index] {
                cotangents
                    .get(value)
                    .copied()
                    .map_or(AdValue::Absent, AdValue::Value)
            } else {
                AdValue::Absent
            }
        })
        .collect();
    finish_derivative(builder, derivative_input_indices, outputs)
}

fn import_source(
    input: &FrozenProgram,
    builder: &mut SemanticProgramBuilder,
) -> Result<HashMap<ProgramValue, ProgramValue>, SemanticAdTransformError> {
    let mut source_values = input.program.inputs().to_vec();
    source_values.extend(
        input
            .program
            .operations()
            .flat_map(|operation| operation.outputs().iter().copied()),
    );
    let imported = builder.import(ProgramImport {
        program: input.program.as_ref(),
        bindings: &input.bindings,
        roots: &source_values,
    })?;
    Ok(source_values
        .into_iter()
        .zip(imported.roots().iter().copied())
        .collect())
}

fn mapped_values(
    source: &[ProgramValue],
    values: &HashMap<ProgramValue, ProgramValue>,
) -> Vec<ProgramValue> {
    source.iter().map(|value| values[value]).collect()
}

fn source_output_liveness(input: &FrozenProgram) -> HashSet<ProgramValue> {
    let mut live: HashSet<_> = input.program.outputs().iter().copied().collect();
    let operations: Vec<_> = input.program.operations().collect();
    for operation in operations.into_iter().rev() {
        if operation
            .outputs()
            .iter()
            .any(|output| live.contains(output))
        {
            live.extend(operation.inputs().iter().copied());
        }
    }
    live
}

fn requested_input_reachability(
    input: &FrozenProgram,
    active_inputs: &[bool],
) -> HashSet<ProgramValue> {
    let mut active: HashSet<_> = input
        .program
        .inputs()
        .iter()
        .copied()
        .zip(active_inputs.iter().copied())
        .filter_map(|(value, is_active)| is_active.then_some(value))
        .collect();
    for operation in input.program.operations() {
        if operation
            .inputs()
            .iter()
            .any(|value| active.contains(value))
        {
            active.extend(operation.outputs().iter().copied());
        }
    }
    active
}

fn accumulate_cotangent(
    builder: &mut SemanticProgramBuilder,
    cotangents: &mut HashMap<ProgramValue, ProgramValue>,
    source: ProgramValue,
    cotangent: ProgramValue,
) -> Result<(), ProgramBuildError> {
    let combined = if let Some(existing) = cotangents.get(&source).copied() {
        builder.add_op(CoreSemanticOp::Add, &[existing, cotangent])?[0]
    } else {
        cotangent
    };
    cotangents.insert(source, combined);
    Ok(())
}

fn linearize_core(
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdTransformError> {
    let output = match op {
        CoreSemanticOp::Add => add_ad_values(builder, tangent_inputs[0], tangent_inputs[1])?,
        CoreSemanticOp::Sub => sub_ad_values(builder, tangent_inputs[0], tangent_inputs[1])?,
        CoreSemanticOp::Mul => {
            let lhs = multiply_ad_value(builder, tangent_inputs[0], primal_inputs[1])?;
            let rhs = multiply_ad_value(builder, tangent_inputs[1], primal_inputs[0])?;
            add_ad_values(builder, lhs, rhs)?
        }
        CoreSemanticOp::Neg | CoreSemanticOp::Conj => {
            unary_ad_value(builder, op.clone(), tangent_inputs[0])?
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Jvp, op)),
    };
    Ok([output].into())
}

fn vjp_core(
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    cotangent_outputs: &[AdValue],
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdTransformError> {
    let cotangent = cotangent_outputs[0];
    let inputs = match op {
        CoreSemanticOp::Add => vec![
            active_cotangent(builder, cotangent, active_inputs[0], primal_inputs[0])?,
            active_cotangent(builder, cotangent, active_inputs[1], primal_inputs[1])?,
        ],
        CoreSemanticOp::Sub => {
            let negated = unary_ad_value(builder, CoreSemanticOp::Neg, cotangent)?;
            vec![
                active_cotangent(builder, cotangent, active_inputs[0], primal_inputs[0])?,
                normalize_ad_value(builder, negated, active_inputs[1], primal_inputs[1])?,
            ]
        }
        CoreSemanticOp::Mul => {
            let rhs_coefficient = conjugate_if_complex(builder, primal_inputs[1])?;
            let lhs_coefficient = conjugate_if_complex(builder, primal_inputs[0])?;
            let lhs = multiply_ad_value(builder, cotangent, rhs_coefficient)?;
            let rhs = multiply_ad_value(builder, cotangent, lhs_coefficient)?;
            vec![
                normalize_ad_value(builder, lhs, active_inputs[0], primal_inputs[0])?,
                normalize_ad_value(builder, rhs, active_inputs[1], primal_inputs[1])?,
            ]
        }
        CoreSemanticOp::Neg => {
            let negated = unary_ad_value(builder, CoreSemanticOp::Neg, cotangent)?;
            vec![normalize_ad_value(
                builder,
                negated,
                active_inputs[0],
                primal_inputs[0],
            )?]
        }
        CoreSemanticOp::Conj => {
            let conjugated = unary_ad_value(builder, CoreSemanticOp::Conj, cotangent)?;
            vec![normalize_ad_value(
                builder,
                conjugated,
                active_inputs[0],
                primal_inputs[0],
            )?]
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Vjp, op)),
    };
    Ok(inputs.into_boxed_slice())
}

fn add_ad_values(
    builder: &mut SemanticProgramBuilder,
    lhs: AdValue,
    rhs: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    match (lhs, rhs) {
        (AdValue::Absent, value) | (value, AdValue::Absent) => Ok(value),
        (AdValue::Value(lhs), AdValue::Value(rhs)) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Add, &[lhs, rhs])?[0],
        )),
    }
}

fn sub_ad_values(
    builder: &mut SemanticProgramBuilder,
    lhs: AdValue,
    rhs: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    match (lhs, rhs) {
        (AdValue::Absent, AdValue::Absent) => Ok(AdValue::Absent),
        (value, AdValue::Absent) => Ok(value),
        (AdValue::Absent, AdValue::Value(rhs)) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Neg, &[rhs])?[0],
        )),
        (AdValue::Value(lhs), AdValue::Value(rhs)) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Sub, &[lhs, rhs])?[0],
        )),
    }
}

fn unary_ad_value(
    builder: &mut SemanticProgramBuilder,
    op: CoreSemanticOp,
    value: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    match value {
        AdValue::Absent => Ok(AdValue::Absent),
        AdValue::Value(value) => Ok(AdValue::Value(builder.add_op(op, &[value])?[0])),
    }
}

fn multiply_ad_value(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    coefficient: ProgramValue,
) -> Result<AdValue, ProgramBuildError> {
    match value {
        AdValue::Absent => Ok(AdValue::Absent),
        AdValue::Value(value) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Mul, &[value, coefficient])?[0],
        )),
    }
}

fn active_cotangent(
    builder: &mut SemanticProgramBuilder,
    cotangent: AdValue,
    active: bool,
    primal_input: ProgramValue,
) -> Result<AdValue, SemanticAdTransformError> {
    normalize_ad_value(builder, cotangent, active, primal_input)
}

fn normalize_ad_value(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    active: bool,
    primal_input: ProgramValue,
) -> Result<AdValue, SemanticAdTransformError> {
    if !active {
        return Ok(AdValue::Absent);
    }
    let AdValue::Value(mut value) = value else {
        return Ok(AdValue::Absent);
    };
    let target_metadata = builder.value_metadata(primal_input)?.clone();
    let value_metadata = builder.value_metadata(value)?.clone();
    let target_shape = exact_shape(
        target_metadata.shape(),
        SemanticTransformRole::Vjp,
        "primal input",
    )?;
    let value_shape = exact_shape(
        value_metadata.shape(),
        SemanticTransformRole::Vjp,
        "cotangent",
    )?;
    if value_shape.len() < target_shape.len() {
        return Err(SemanticAdTransformError::UnsupportedMetadata {
            role: SemanticTransformRole::Vjp,
            message: "cotangent rank is smaller than its primal-input rank".into(),
        });
    }
    let leading = value_shape.len() - target_shape.len();
    let mut axes: Vec<_> = (0..leading).collect();
    axes.extend(
        target_shape
            .iter()
            .zip(value_shape.iter().skip(leading))
            .enumerate()
            .filter_map(|(axis, (target, actual))| {
                (matches!(target, tenferro_ops::dim_expr::DimExpr::Const(1)) && target != actual)
                    .then_some(axis + leading)
            }),
    );
    if !axes.is_empty() {
        value = builder.add_op(CoreSemanticOp::ReduceSum { axes }, &[value])?[0];
    }
    if builder.value_metadata(value)?.shape() != target_metadata.shape() {
        value = builder.add_op(
            CoreSemanticOp::Reshape {
                to_shape: target_shape,
            },
            &[value],
        )?[0];
    }
    let value_dtype = builder.value_metadata(value)?.dtype();
    if value_dtype != target_metadata.dtype() {
        value = builder.add_op(
            CoreSemanticOp::Convert {
                from: value_dtype,
                to: target_metadata.dtype(),
            },
            &[value],
        )?[0];
    }
    Ok(AdValue::Value(value))
}

fn exact_shape(
    shape: &[tenferro_ops::ShapeExtent<tenferro_ops::dim_expr::DimExpr>],
    role: SemanticTransformRole,
    field: &'static str,
) -> Result<Vec<tenferro_ops::dim_expr::DimExpr>, SemanticAdTransformError> {
    shape
        .iter()
        .map(|extent| {
            extent.as_exact().cloned().ok_or_else(|| {
                SemanticAdTransformError::UnsupportedMetadata {
                    role,
                    message: format!("{field} has a bounded or unknown extent"),
                }
            })
        })
        .collect()
}

fn conjugate_if_complex(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
) -> Result<ProgramValue, ProgramBuildError> {
    if matches!(
        builder.value_metadata(value)?.dtype(),
        DType::C32 | DType::C64
    ) {
        Ok(builder.add_op(CoreSemanticOp::Conj, &[value])?[0])
    } else {
        Ok(value)
    }
}

fn finish_derivative(
    builder: SemanticProgramBuilder,
    derivative_input_indices: Vec<Option<usize>>,
    values: Vec<AdValue>,
) -> Result<SemanticAdProgram, SemanticAdTransformError> {
    let mut outputs = Vec::new();
    let derivative_output_indices = values
        .into_iter()
        .map(|value| match value {
            AdValue::Absent => None,
            AdValue::Value(value) => {
                let index = outputs.len();
                outputs.push(value);
                Some(index)
            }
        })
        .collect();
    Ok(SemanticAdProgram {
        frozen: builder.finish(&outputs)?,
        derivative_input_indices: derivative_input_indices.into_boxed_slice(),
        derivative_output_indices,
    })
}

fn validate_activity(
    role: SemanticTransformRole,
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), SemanticAdTransformError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SemanticAdTransformError::ActivityArity {
            role,
            field,
            expected,
            actual,
        })
    }
}

fn unsupported_core(role: SemanticTransformRole, op: &CoreSemanticOp) -> SemanticAdTransformError {
    SemanticAdTransformError::UnsupportedCore {
        role,
        op: format!("{op:?}"),
    }
}
