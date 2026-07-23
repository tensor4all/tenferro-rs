//! Whole-program automatic differentiation over semantic SSA programs.

use std::collections::{HashMap, HashSet};

use tenferro_runtime::program::{
    CoreSemanticOp, FrozenProgram, ProgramBuildError, ProgramFinishError, ProgramImport,
    ProgramInputSpec, ProgramQueryError, ProgramValue, SemanticOpRef, SemanticProgramBuilder,
};
use tenferro_runtime::{CompareDir, DType, DotGeneralConfig};

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
        CoreSemanticOp::Div => {
            let lhs = divide_ad_value(builder, tangent_inputs[0], primal_inputs[1])?;
            let rhs_numerator = multiply_ad_value(builder, tangent_inputs[1], primal_inputs[0])?;
            let denominator =
                builder.add_op(CoreSemanticOp::Mul, &[primal_inputs[1], primal_inputs[1]])?[0];
            let rhs = divide_ad_value(builder, rhs_numerator, denominator)?;
            sub_ad_values(builder, lhs, rhs)?
        }
        CoreSemanticOp::Pow => {
            let lhs = if matches!(tangent_inputs[0], AdValue::Value(_)) {
                let one = one_like(builder, primal_inputs[1], SemanticTransformRole::Jvp)?;
                let exponent_minus_one =
                    builder.add_op(CoreSemanticOp::Sub, &[primal_inputs[1], one])?[0];
                let power = builder
                    .add_op(CoreSemanticOp::Pow, &[primal_inputs[0], exponent_minus_one])?[0];
                let coefficient =
                    builder.add_op(CoreSemanticOp::Mul, &[primal_inputs[1], power])?[0];
                multiply_ad_value(builder, tangent_inputs[0], coefficient)?
            } else {
                AdValue::Absent
            };
            let rhs = if matches!(tangent_inputs[1], AdValue::Value(_)) {
                let log = builder.add_op(CoreSemanticOp::Log, &[primal_inputs[0]])?[0];
                let power =
                    builder.add_op(CoreSemanticOp::Pow, &[primal_inputs[0], primal_inputs[1]])?[0];
                let coefficient = builder.add_op(CoreSemanticOp::Mul, &[log, power])?[0];
                multiply_ad_value(builder, tangent_inputs[1], coefficient)?
            } else {
                AdValue::Absent
            };
            add_ad_values(builder, lhs, rhs)?
        }
        CoreSemanticOp::DotGeneral { config } => {
            linearize_dot_general(builder, primal_inputs, tangent_inputs, config)?
        }
        CoreSemanticOp::Abs => {
            let input_dtype = builder.value_metadata(primal_inputs[0])?.dtype();
            let sign = builder.add_op(CoreSemanticOp::Sign, &[primal_inputs[0]])?[0];
            let coefficient = if is_complex_dtype(input_dtype) {
                builder.add_op(CoreSemanticOp::Conj, &[sign])?[0]
            } else {
                sign
            };
            let tangent = multiply_ad_value(builder, tangent_inputs[0], coefficient)?;
            convert_ad_value(builder, tangent, input_dtype, abs_output_dtype(input_dtype))?
        }
        CoreSemanticOp::Sign => AdValue::Absent,
        CoreSemanticOp::Maximum | CoreSemanticOp::Minimum => {
            linearize_extrema(builder, op, primal_inputs, tangent_inputs)?
        }
        CoreSemanticOp::Select => select_ad_values(
            builder,
            primal_inputs[0],
            tangent_inputs[1],
            tangent_inputs[2],
        )?,
        CoreSemanticOp::Clamp => linearize_clamp(builder, primal_inputs, tangent_inputs)?,
        CoreSemanticOp::Neg | CoreSemanticOp::Conj => {
            unary_ad_value(builder, op.clone(), tangent_inputs[0])?
        }
        CoreSemanticOp::Exp
        | CoreSemanticOp::Log
        | CoreSemanticOp::Sin
        | CoreSemanticOp::Cos
        | CoreSemanticOp::Tanh
        | CoreSemanticOp::Sqrt
        | CoreSemanticOp::Rsqrt
        | CoreSemanticOp::Expm1
        | CoreSemanticOp::Log1p => {
            linearize_analytic_unary(builder, op, primal_inputs[0], tangent_inputs[0])?
        }
        CoreSemanticOp::Transpose { .. }
        | CoreSemanticOp::Reshape { .. }
        | CoreSemanticOp::BroadcastInDim { .. }
        | CoreSemanticOp::ReduceSum { .. }
        | CoreSemanticOp::ExtractDiag { .. }
        | CoreSemanticOp::EmbedDiag { .. }
        | CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. }
        | CoreSemanticOp::Reverse { .. } => {
            linearize_unary_core(builder, op.clone(), primal_inputs, tangent_inputs[0])?
        }
        CoreSemanticOp::Convert { from, to } => {
            if is_differentiable_dtype(*from) && is_differentiable_dtype(*to) {
                linearize_unary_core(builder, op.clone(), primal_inputs, tangent_inputs[0])?
            } else {
                AdValue::Absent
            }
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
        CoreSemanticOp::Div => {
            let rhs_coefficient = conjugate_if_complex(builder, primal_inputs[1])?;
            let lhs = divide_ad_value(builder, cotangent, rhs_coefficient)?;
            let lhs_coefficient = conjugate_if_complex(builder, primal_inputs[0])?;
            let denominator =
                builder.add_op(CoreSemanticOp::Mul, &[rhs_coefficient, rhs_coefficient])?[0];
            let rhs = multiply_ad_value(builder, cotangent, lhs_coefficient)?;
            let rhs = divide_ad_value(builder, rhs, denominator)?;
            let rhs = unary_ad_value(builder, CoreSemanticOp::Neg, rhs)?;
            vec![
                normalize_ad_value(builder, lhs, active_inputs[0], primal_inputs[0])?,
                normalize_ad_value(builder, rhs, active_inputs[1], primal_inputs[1])?,
            ]
        }
        CoreSemanticOp::Pow => {
            let lhs = if active_inputs[0] {
                let one = one_like(builder, primal_inputs[1], SemanticTransformRole::Vjp)?;
                let exponent_minus_one =
                    builder.add_op(CoreSemanticOp::Sub, &[primal_inputs[1], one])?[0];
                let power = builder
                    .add_op(CoreSemanticOp::Pow, &[primal_inputs[0], exponent_minus_one])?[0];
                let coefficient =
                    builder.add_op(CoreSemanticOp::Mul, &[primal_inputs[1], power])?[0];
                let coefficient = conjugate_if_complex(builder, coefficient)?;
                multiply_ad_value(builder, cotangent, coefficient)?
            } else {
                AdValue::Absent
            };
            let rhs = if active_inputs[1] {
                let log = builder.add_op(CoreSemanticOp::Log, &[primal_inputs[0]])?[0];
                let power =
                    builder.add_op(CoreSemanticOp::Pow, &[primal_inputs[0], primal_inputs[1]])?[0];
                let coefficient = builder.add_op(CoreSemanticOp::Mul, &[log, power])?[0];
                let coefficient = conjugate_if_complex(builder, coefficient)?;
                multiply_ad_value(builder, cotangent, coefficient)?
            } else {
                AdValue::Absent
            };
            vec![
                normalize_ad_value(builder, lhs, active_inputs[0], primal_inputs[0])?,
                normalize_ad_value(builder, rhs, active_inputs[1], primal_inputs[1])?,
            ]
        }
        CoreSemanticOp::DotGeneral { config } => {
            dot_general_vjp(builder, primal_inputs, cotangent, active_inputs, config)?
        }
        CoreSemanticOp::Abs => {
            let input_dtype = builder.value_metadata(primal_inputs[0])?.dtype();
            let output_dtype = abs_output_dtype(input_dtype);
            let cotangent = convert_ad_value(builder, cotangent, output_dtype, input_dtype)?;
            let sign = builder.add_op(CoreSemanticOp::Sign, &[primal_inputs[0]])?[0];
            let cotangent = multiply_ad_value(builder, cotangent, sign)?;
            vec![normalize_ad_value(
                builder,
                cotangent,
                active_inputs[0],
                primal_inputs[0],
            )?]
        }
        CoreSemanticOp::Sign => vec![AdValue::Absent],
        CoreSemanticOp::Maximum | CoreSemanticOp::Minimum => {
            extrema_vjp(builder, op, primal_inputs, cotangent, active_inputs)?
        }
        CoreSemanticOp::Select => {
            let (on_true, on_false) = split_select_cotangent(
                builder,
                primal_inputs[0],
                cotangent,
                active_inputs[1],
                active_inputs[2],
            )?;
            vec![
                AdValue::Absent,
                normalize_ad_value(builder, on_true, active_inputs[1], primal_inputs[1])?,
                normalize_ad_value(builder, on_false, active_inputs[2], primal_inputs[2])?,
            ]
        }
        CoreSemanticOp::Clamp => clamp_vjp(builder, primal_inputs, cotangent, active_inputs)?,
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
        CoreSemanticOp::Exp
        | CoreSemanticOp::Log
        | CoreSemanticOp::Sin
        | CoreSemanticOp::Cos
        | CoreSemanticOp::Tanh
        | CoreSemanticOp::Sqrt
        | CoreSemanticOp::Rsqrt
        | CoreSemanticOp::Expm1
        | CoreSemanticOp::Log1p => {
            let coefficient = analytic_unary_coefficient(
                builder,
                op,
                primal_inputs[0],
                SemanticTransformRole::Vjp,
            )?;
            let coefficient = conjugate_if_complex(builder, coefficient)?;
            let cotangent = multiply_ad_value(builder, cotangent, coefficient)?;
            vec![normalize_ad_value(
                builder,
                cotangent,
                active_inputs[0],
                primal_inputs[0],
            )?]
        }
        CoreSemanticOp::Transpose { perm } => {
            let transposed = unary_ad_value(
                builder,
                CoreSemanticOp::Transpose {
                    perm: inverse_permutation(perm),
                },
                cotangent,
            )?;
            primary_cotangent(builder, transposed, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::Reshape { .. } => {
            let reshaped = reshape_ad_value_to_input(builder, cotangent, primal_inputs[0])?;
            primary_cotangent(builder, reshaped, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::BroadcastInDim { dims, .. } => {
            let reduced =
                transpose_broadcast(builder, cotangent, primal_inputs[0], dims.as_slice())?;
            primary_cotangent(builder, reduced, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::Convert { from, to } => {
            let converted = if is_differentiable_dtype(*from) && is_differentiable_dtype(*to) {
                unary_ad_value(
                    builder,
                    CoreSemanticOp::Convert {
                        from: *to,
                        to: *from,
                    },
                    cotangent,
                )?
            } else {
                AdValue::Absent
            };
            primary_cotangent(builder, converted, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::ReduceSum { axes } => {
            let input_shape = exact_value_shape(
                builder,
                primal_inputs[0],
                SemanticTransformRole::Vjp,
                "reduce-sum input",
            )?;
            let dims = (0..input_shape.len())
                .filter(|axis| !axes.contains(axis))
                .collect();
            let broadcast = unary_ad_value(
                builder,
                CoreSemanticOp::BroadcastInDim {
                    shape: input_shape,
                    dims,
                },
                cotangent,
            )?;
            primary_cotangent(builder, broadcast, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::ExtractDiag { axis_a, axis_b } => {
            let embedded = unary_ad_value(
                builder,
                CoreSemanticOp::EmbedDiag {
                    axis_a: if axis_a < axis_b { *axis_a } else { axis_a - 1 },
                    axis_b: *axis_b,
                },
                cotangent,
            )?;
            let padded = match embedded {
                AdValue::Absent => AdValue::Absent,
                AdValue::Value(value) => {
                    let value = builder.add_op(
                        CoreSemanticOp::PadToMatch { axis: *axis_a },
                        &[value, primal_inputs[0]],
                    )?[0];
                    AdValue::Value(
                        builder.add_op(
                            CoreSemanticOp::PadToMatch { axis: *axis_b },
                            &[value, primal_inputs[0]],
                        )?[0],
                    )
                }
            };
            primary_cotangent(builder, padded, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::EmbedDiag { axis_a, axis_b } => {
            let source_axis = if axis_b <= axis_a {
                axis_a + 1
            } else {
                *axis_a
            };
            let extracted = unary_ad_value(
                builder,
                CoreSemanticOp::ExtractDiag {
                    axis_a: source_axis,
                    axis_b: *axis_b,
                },
                cotangent,
            )?;
            let restored = if axis_b < axis_a {
                let rank = builder.value_metadata(primal_inputs[0])?.shape().len();
                let mut perm: Vec<_> = (0..rank).collect();
                let diagonal_axis = perm.remove(*axis_b);
                perm.insert(*axis_a, diagonal_axis);
                unary_ad_value(builder, CoreSemanticOp::Transpose { perm }, extracted)?
            } else {
                extracted
            };
            primary_cotangent(builder, restored, active_inputs, primal_inputs, false)?
        }
        CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. }
        | CoreSemanticOp::Reverse { .. } => {
            let transformed = unary_ad_value(builder, op.clone(), cotangent)?;
            primary_cotangent(builder, transformed, active_inputs, primal_inputs, false)?
        }
        _ => return Err(unsupported_core(SemanticTransformRole::Vjp, op)),
    };
    Ok(inputs.into_boxed_slice())
}

fn linearize_unary_core(
    builder: &mut SemanticProgramBuilder,
    op: CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    let AdValue::Value(tangent) = tangent else {
        return Ok(AdValue::Absent);
    };
    let mut inputs = Vec::with_capacity(primal_inputs.len());
    inputs.push(tangent);
    inputs.extend_from_slice(&primal_inputs[1..]);
    Ok(AdValue::Value(builder.add_op(op, &inputs)?[0]))
}

fn linearize_dot_general(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
    config: &DotGeneralConfig,
) -> Result<AdValue, SemanticAdTransformError> {
    validate_dot_general_metadata(builder, primal_inputs, config, SemanticTransformRole::Jvp)?;
    let mut terms = Vec::with_capacity(2);
    if let AdValue::Value(tangent) = tangent_inputs[0] {
        terms.push(
            builder.add_op(
                CoreSemanticOp::DotGeneral {
                    config: config.clone(),
                },
                &[tangent, primal_inputs[1]],
            )?[0],
        );
    }
    if let AdValue::Value(tangent) = tangent_inputs[1] {
        terms.push(
            builder.add_op(
                CoreSemanticOp::DotGeneral {
                    config: config.clone(),
                },
                &[primal_inputs[0], tangent],
            )?[0],
        );
    }
    let mut terms = terms.into_iter();
    let Some(mut result) = terms.next() else {
        return Ok(AdValue::Absent);
    };
    for term in terms {
        result = builder.add_op(CoreSemanticOp::Add, &[result, term])?[0];
    }
    Ok(AdValue::Value(result))
}

fn dot_general_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    config: &DotGeneralConfig,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let (lhs_rank, rhs_rank) =
        validate_dot_general_metadata(builder, primal_inputs, config, SemanticTransformRole::Vjp)?;
    let lhs_free = dot_general_free_dims(
        lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
        SemanticTransformRole::Vjp,
    )?;
    let rhs_free = dot_general_free_dims(
        rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
        SemanticTransformRole::Vjp,
    )?;
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent, AdValue::Absent]);
    };
    let mut result = vec![AdValue::Absent, AdValue::Absent];

    if active_inputs[0] {
        let rhs = conjugate_if_complex(builder, primal_inputs[1])?;
        let (transpose_config, perm) =
            dot_general_transpose_plan_for_lhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free)?;
        let value = builder.add_op(
            CoreSemanticOp::DotGeneral {
                config: transpose_config,
            },
            &[cotangent, rhs],
        )?[0];
        let value = transpose_if_needed(builder, value, &perm)?;
        result[0] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[0])?;
    }
    if active_inputs[1] {
        let lhs = conjugate_if_complex(builder, primal_inputs[0])?;
        let (transpose_config, perm) =
            dot_general_transpose_plan_for_rhs(config, lhs_rank, rhs_rank, &lhs_free, &rhs_free)?;
        let value = builder.add_op(
            CoreSemanticOp::DotGeneral {
                config: transpose_config,
            },
            &[lhs, cotangent],
        )?[0];
        let value = transpose_if_needed(builder, value, &perm)?;
        result[1] = normalize_ad_value(builder, AdValue::Value(value), true, primal_inputs[1])?;
    }
    Ok(result)
}

fn validate_dot_general_metadata(
    builder: &SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    config: &DotGeneralConfig,
    role: SemanticTransformRole,
) -> Result<(usize, usize), SemanticAdTransformError> {
    let lhs_rank = builder.value_metadata(primal_inputs[0])?.shape().len();
    let rhs_rank = builder.value_metadata(primal_inputs[1])?.shape().len();
    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(|error| SemanticAdTransformError::UnsupportedMetadata {
            role,
            message: format!(
                "invalid dot_general dimensions for ranks {lhs_rank} and {rhs_rank}: {error}"
            ),
        })?;
    Ok((lhs_rank, rhs_rank))
}

fn dot_general_free_dims(
    rank: usize,
    contracting: &[usize],
    batch: &[usize],
    role: SemanticTransformRole,
) -> Result<Vec<usize>, SemanticAdTransformError> {
    let mut bound = vec![false; rank];
    for &axis in batch.iter().chain(contracting) {
        let Some(slot) = bound.get_mut(axis) else {
            return Err(SemanticAdTransformError::UnsupportedMetadata {
                role,
                message: format!("dot_general axis {axis} is out of bounds for rank {rank}"),
            });
        };
        *slot = true;
    }
    Ok((0..rank).filter(|axis| !bound[*axis]).collect())
}

fn dot_general_transpose_plan_for_lhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> Result<(DotGeneralConfig, Vec<usize>), SemanticAdTransformError> {
    let batch_count = config.lhs_batch_dims.len();
    let output_rank = lhs_free.len() + rhs_free.len() + batch_count;
    let rhs_free_positions = (lhs_free.len()..lhs_free.len() + rhs_free.len()).collect();
    let rhs_contracting_order = dot_general_free_dims(
        rhs_rank,
        rhs_free,
        &config.rhs_batch_dims,
        SemanticTransformRole::Vjp,
    )?;
    let mut result_order = lhs_free.to_vec();
    for rhs_axis in rhs_contracting_order {
        let Some(pair) = config
            .rhs_contracting_dims
            .iter()
            .position(|&axis| axis == rhs_axis)
        else {
            return Err(dot_general_transpose_metadata_error(format!(
                "rhs contracting axis {rhs_axis} has no lhs pair"
            )));
        };
        result_order.push(config.lhs_contracting_dims[pair]);
    }
    result_order.extend(config.lhs_batch_dims.iter().copied());
    Ok((
        DotGeneralConfig {
            lhs_contracting_dims: rhs_free_positions,
            rhs_contracting_dims: rhs_free.to_vec(),
            lhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
            rhs_batch_dims: config.rhs_batch_dims.clone(),
        },
        permutation_to_original_order(lhs_rank, &result_order)?,
    ))
}

fn dot_general_transpose_plan_for_rhs(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_free: &[usize],
    rhs_free: &[usize],
) -> Result<(DotGeneralConfig, Vec<usize>), SemanticAdTransformError> {
    let batch_count = config.lhs_batch_dims.len();
    let lhs_contracting_order = dot_general_free_dims(
        lhs_rank,
        lhs_free,
        &config.lhs_batch_dims,
        SemanticTransformRole::Vjp,
    )?;
    let mut result_order = Vec::with_capacity(rhs_rank);
    for lhs_axis in lhs_contracting_order {
        let Some(pair) = config
            .lhs_contracting_dims
            .iter()
            .position(|&axis| axis == lhs_axis)
        else {
            return Err(dot_general_transpose_metadata_error(format!(
                "lhs contracting axis {lhs_axis} has no rhs pair"
            )));
        };
        result_order.push(config.rhs_contracting_dims[pair]);
    }
    result_order.extend(rhs_free.iter().copied());
    result_order.extend(config.rhs_batch_dims.iter().copied());
    let output_rank = lhs_free.len() + rhs_free.len() + batch_count;
    Ok((
        DotGeneralConfig {
            lhs_contracting_dims: lhs_free.to_vec(),
            rhs_contracting_dims: (0..lhs_free.len()).collect(),
            lhs_batch_dims: config.lhs_batch_dims.clone(),
            rhs_batch_dims: (lhs_free.len() + rhs_free.len()..output_rank).collect(),
        },
        permutation_to_original_order(rhs_rank, &result_order)?,
    ))
}

fn permutation_to_original_order(
    rank: usize,
    result_order: &[usize],
) -> Result<Vec<usize>, SemanticAdTransformError> {
    let mut permutation = vec![0; rank];
    for (result_axis, &original_axis) in result_order.iter().enumerate() {
        let Some(slot) = permutation.get_mut(original_axis) else {
            return Err(dot_general_transpose_metadata_error(format!(
                "dot_general transpose axis {original_axis} is out of bounds for rank {rank}"
            )));
        };
        *slot = result_axis;
    }
    Ok(permutation)
}

fn transpose_if_needed(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
    permutation: &[usize],
) -> Result<ProgramValue, ProgramBuildError> {
    if permutation
        .iter()
        .enumerate()
        .all(|(axis, &mapped)| axis == mapped)
    {
        Ok(value)
    } else {
        Ok(builder.add_op(
            CoreSemanticOp::Transpose {
                perm: permutation.to_vec(),
            },
            &[value],
        )?[0])
    }
}

fn dot_general_transpose_metadata_error(message: String) -> SemanticAdTransformError {
    SemanticAdTransformError::UnsupportedMetadata {
        role: SemanticTransformRole::Vjp,
        message,
    }
}

fn primary_cotangent(
    builder: &mut SemanticProgramBuilder,
    cotangent: AdValue,
    active_inputs: &[bool],
    primal_inputs: &[ProgramValue],
    normalize: bool,
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let mut result = vec![AdValue::Absent; primal_inputs.len()];
    if active_inputs.first().copied().unwrap_or(false) {
        result[0] = if normalize {
            normalize_ad_value(builder, cotangent, true, primal_inputs[0])?
        } else {
            cotangent
        };
    }
    Ok(result)
}

fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (axis, mapped) in perm.iter().copied().enumerate() {
        inverse[mapped] = axis;
    }
    inverse
}

fn reshape_ad_value_to_input(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    primal_input: ProgramValue,
) -> Result<AdValue, SemanticAdTransformError> {
    let shape = exact_value_shape(
        builder,
        primal_input,
        SemanticTransformRole::Vjp,
        "reshape input",
    )?;
    Ok(unary_ad_value(
        builder,
        CoreSemanticOp::Reshape { to_shape: shape },
        value,
    )?)
}

fn transpose_broadcast(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    primal_input: ProgramValue,
    dims: &[usize],
) -> Result<AdValue, SemanticAdTransformError> {
    let AdValue::Value(mut value) = value else {
        return Ok(AdValue::Absent);
    };
    let input_shape = exact_value_shape(
        builder,
        primal_input,
        SemanticTransformRole::Vjp,
        "broadcast input",
    )?;
    let output_shape = exact_value_shape(
        builder,
        value,
        SemanticTransformRole::Vjp,
        "broadcast cotangent",
    )?;
    let mut reduce_axes: Vec<_> = (0..output_shape.len())
        .filter(|axis| !dims.contains(axis))
        .collect();
    reduce_axes.extend(
        dims.iter()
            .copied()
            .enumerate()
            .filter_map(|(input_axis, output_axis)| {
                (matches!(
                    input_shape[input_axis],
                    tenferro_ops::dim_expr::DimExpr::Const(1)
                ) && input_shape[input_axis] != output_shape[output_axis])
                    .then_some(output_axis)
            }),
    );
    reduce_axes.sort_unstable();
    reduce_axes.dedup();
    if !reduce_axes.is_empty() {
        value = builder.add_op(
            CoreSemanticOp::ReduceSum {
                axes: reduce_axes.clone(),
            },
            &[value],
        )?[0];
    }

    let remaining_output_axes: Vec<_> = (0..output_shape.len())
        .filter(|axis| !reduce_axes.contains(axis))
        .collect();
    let perm: Vec<_> = dims
        .iter()
        .copied()
        .filter(|axis| !reduce_axes.contains(axis))
        .map(|axis| {
            remaining_output_axes
                .iter()
                .position(|candidate| *candidate == axis)
                .expect("broadcast dims survive unless reduced")
        })
        .collect();
    if perm.iter().copied().ne(0..perm.len()) {
        value = builder.add_op(CoreSemanticOp::Transpose { perm }, &[value])?[0];
    }
    if builder.value_metadata(value)?.shape() != builder.value_metadata(primal_input)?.shape() {
        value = builder.add_op(
            CoreSemanticOp::Reshape {
                to_shape: input_shape,
            },
            &[value],
        )?[0];
    }
    Ok(AdValue::Value(value))
}

fn exact_value_shape(
    builder: &SemanticProgramBuilder,
    value: ProgramValue,
    role: SemanticTransformRole,
    field: &'static str,
) -> Result<Vec<tenferro_ops::dim_expr::DimExpr>, SemanticAdTransformError> {
    exact_shape(builder.value_metadata(value)?.shape(), role, field)
}

fn is_differentiable_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64)
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

fn abs_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::C32 => DType::F32,
        DType::C64 => DType::F64,
        other => other,
    }
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

fn divide_ad_value(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    denominator: ProgramValue,
) -> Result<AdValue, ProgramBuildError> {
    match value {
        AdValue::Absent => Ok(AdValue::Absent),
        AdValue::Value(value) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Div, &[value, denominator])?[0],
        )),
    }
}

fn convert_ad_value(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
    from: DType,
    to: DType,
) -> Result<AdValue, ProgramBuildError> {
    if from == to {
        return Ok(value);
    }
    unary_ad_value(builder, CoreSemanticOp::Convert { from, to }, value)
}

fn zero_from_ad_value(
    builder: &mut SemanticProgramBuilder,
    value: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    let negated = unary_ad_value(builder, CoreSemanticOp::Neg, value)?;
    add_ad_values(builder, value, negated)
}

fn select_ad_values(
    builder: &mut SemanticProgramBuilder,
    condition: ProgramValue,
    on_true: AdValue,
    on_false: AdValue,
) -> Result<AdValue, ProgramBuildError> {
    match (on_true, on_false) {
        (AdValue::Absent, AdValue::Absent) => Ok(AdValue::Absent),
        (AdValue::Value(on_true), AdValue::Value(on_false)) => Ok(AdValue::Value(
            builder.add_op(CoreSemanticOp::Select, &[condition, on_true, on_false])?[0],
        )),
        (AdValue::Value(on_true), AdValue::Absent) => {
            let AdValue::Value(zero) = zero_from_ad_value(builder, AdValue::Value(on_true))? else {
                unreachable!();
            };
            Ok(AdValue::Value(
                builder.add_op(CoreSemanticOp::Select, &[condition, on_true, zero])?[0],
            ))
        }
        (AdValue::Absent, AdValue::Value(on_false)) => {
            let AdValue::Value(zero) = zero_from_ad_value(builder, AdValue::Value(on_false))?
            else {
                unreachable!();
            };
            Ok(AdValue::Value(
                builder.add_op(CoreSemanticOp::Select, &[condition, zero, on_false])?[0],
            ))
        }
    }
}

fn split_select_cotangent(
    builder: &mut SemanticProgramBuilder,
    condition: ProgramValue,
    cotangent: AdValue,
    true_active: bool,
    false_active: bool,
) -> Result<(AdValue, AdValue), ProgramBuildError> {
    if !true_active && !false_active {
        return Ok((AdValue::Absent, AdValue::Absent));
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok((AdValue::Absent, AdValue::Absent));
    };
    let AdValue::Value(zero) = zero_from_ad_value(builder, AdValue::Value(cotangent))? else {
        unreachable!();
    };
    let on_true = if true_active {
        AdValue::Value(builder.add_op(CoreSemanticOp::Select, &[condition, cotangent, zero])?[0])
    } else {
        AdValue::Absent
    };
    let on_false = if false_active {
        AdValue::Value(builder.add_op(CoreSemanticOp::Select, &[condition, zero, cotangent])?[0])
    } else {
        AdValue::Absent
    };
    Ok((on_true, on_false))
}

fn linearize_extrema(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
) -> Result<AdValue, SemanticAdTransformError> {
    let output = builder.add_op(op.clone(), primal_inputs)?[0];
    let lhs_eq_output = builder.add_op(
        CoreSemanticOp::Compare(CompareDir::Eq),
        &[primal_inputs[0], output],
    )?[0];
    let rhs_eq_output = builder.add_op(
        CoreSemanticOp::Compare(CompareDir::Eq),
        &[primal_inputs[1], output],
    )?[0];
    let lhs = balanced_extrema_contribution(
        builder,
        tangent_inputs[0],
        lhs_eq_output,
        rhs_eq_output,
        SemanticTransformRole::Jvp,
    )?;
    let rhs = balanced_extrema_contribution(
        builder,
        tangent_inputs[1],
        rhs_eq_output,
        lhs_eq_output,
        SemanticTransformRole::Jvp,
    )?;
    Ok(add_ad_values(builder, lhs, rhs)?)
}

fn extrema_vjp(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let output = builder.add_op(op.clone(), primal_inputs)?[0];
    let lhs_eq_output = builder.add_op(
        CoreSemanticOp::Compare(CompareDir::Eq),
        &[primal_inputs[0], output],
    )?[0];
    let rhs_eq_output = builder.add_op(
        CoreSemanticOp::Compare(CompareDir::Eq),
        &[primal_inputs[1], output],
    )?[0];
    let lhs = balanced_extrema_contribution(
        builder,
        cotangent,
        lhs_eq_output,
        rhs_eq_output,
        SemanticTransformRole::Vjp,
    )?;
    let rhs = balanced_extrema_contribution(
        builder,
        cotangent,
        rhs_eq_output,
        lhs_eq_output,
        SemanticTransformRole::Vjp,
    )?;
    Ok(vec![
        normalize_ad_value(builder, lhs, active_inputs[0], primal_inputs[0])?,
        normalize_ad_value(builder, rhs, active_inputs[1], primal_inputs[1])?,
    ])
}

fn balanced_extrema_contribution(
    builder: &mut SemanticProgramBuilder,
    active: AdValue,
    self_eq_output: ProgramValue,
    other_eq_output: ProgramValue,
    role: SemanticTransformRole,
) -> Result<AdValue, SemanticAdTransformError> {
    let AdValue::Value(active) = active else {
        return Ok(AdValue::Absent);
    };
    let zero = builder.add_op(CoreSemanticOp::Sub, &[active, active])?[0];
    let selected = builder.add_op(CoreSemanticOp::Select, &[self_eq_output, active, zero])?[0];
    let one = one_like(builder, active, role)?;
    let two = builder.add_op(CoreSemanticOp::Add, &[one, one])?[0];
    let half = builder.add_op(CoreSemanticOp::Div, &[selected, two])?[0];
    Ok(AdValue::Value(
        builder.add_op(CoreSemanticOp::Select, &[other_eq_output, half, selected])?[0],
    ))
}

fn linearize_clamp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    tangent_inputs: &[AdValue],
) -> Result<AdValue, ProgramBuildError> {
    let masks = clamp_masks(builder, primal_inputs)?;
    let input = mask_ad_value(builder, tangent_inputs[0], &[masks[0], masks[1]])?;
    let lower = mask_ad_value(builder, tangent_inputs[1], &[masks[2], masks[3]])?;
    let upper = mask_ad_value(builder, tangent_inputs[2], &[masks[4]])?;
    let input_and_lower = add_ad_values(builder, input, lower)?;
    add_ad_values(builder, input_and_lower, upper)
}

fn clamp_vjp(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
) -> Result<Vec<AdValue>, SemanticAdTransformError> {
    let masks = clamp_masks(builder, primal_inputs)?;
    let input = mask_ad_value(builder, cotangent, &[masks[0], masks[1]])?;
    let lower = mask_ad_value(builder, cotangent, &[masks[2], masks[3]])?;
    let upper = mask_ad_value(builder, cotangent, &[masks[4]])?;
    Ok(vec![
        normalize_ad_value(builder, input, active_inputs[0], primal_inputs[0])?,
        normalize_ad_value(builder, lower, active_inputs[1], primal_inputs[1])?,
        normalize_ad_value(builder, upper, active_inputs[2], primal_inputs[2])?,
    ])
}

fn clamp_masks(
    builder: &mut SemanticProgramBuilder,
    primal_inputs: &[ProgramValue],
) -> Result<[ProgramValue; 5], ProgramBuildError> {
    let input = primal_inputs[0];
    let lower = primal_inputs[1];
    let upper = primal_inputs[2];
    let input_gt_lower =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Gt), &[input, lower])?[0];
    let input_lt_upper =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Lt), &[input, upper])?[0];
    let lower_gt_input =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Gt), &[lower, input])?[0];
    let lower_lt_upper =
        builder.add_op(CoreSemanticOp::Compare(CompareDir::Lt), &[lower, upper])?[0];
    let max_input_lower = builder.add_op(CoreSemanticOp::Maximum, &[input, lower])?[0];
    let upper_lt_max_input_lower = builder.add_op(
        CoreSemanticOp::Compare(CompareDir::Lt),
        &[upper, max_input_lower],
    )?[0];
    Ok([
        input_gt_lower,
        input_lt_upper,
        lower_gt_input,
        lower_lt_upper,
        upper_lt_max_input_lower,
    ])
}

fn mask_ad_value(
    builder: &mut SemanticProgramBuilder,
    active: AdValue,
    conditions: &[ProgramValue],
) -> Result<AdValue, ProgramBuildError> {
    let AdValue::Value(active) = active else {
        return Ok(AdValue::Absent);
    };
    let zero = builder.add_op(CoreSemanticOp::Sub, &[active, active])?[0];
    let mut value = active;
    for condition in conditions {
        value = builder.add_op(CoreSemanticOp::Select, &[*condition, value, zero])?[0];
    }
    Ok(AdValue::Value(value))
}

fn linearize_analytic_unary(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_input: ProgramValue,
    tangent: AdValue,
) -> Result<AdValue, SemanticAdTransformError> {
    if matches!(tangent, AdValue::Absent) {
        return Ok(AdValue::Absent);
    }
    let coefficient =
        analytic_unary_coefficient(builder, op, primal_input, SemanticTransformRole::Jvp)?;
    Ok(multiply_ad_value(builder, tangent, coefficient)?)
}

fn analytic_unary_coefficient(
    builder: &mut SemanticProgramBuilder,
    op: &CoreSemanticOp,
    primal_input: ProgramValue,
    role: SemanticTransformRole,
) -> Result<ProgramValue, SemanticAdTransformError> {
    let coefficient = match op {
        CoreSemanticOp::Exp | CoreSemanticOp::Expm1 => {
            builder.add_op(CoreSemanticOp::Exp, &[primal_input])?[0]
        }
        CoreSemanticOp::Log => {
            let one = one_like(builder, primal_input, role)?;
            builder.add_op(CoreSemanticOp::Div, &[one, primal_input])?[0]
        }
        CoreSemanticOp::Sin => builder.add_op(CoreSemanticOp::Cos, &[primal_input])?[0],
        CoreSemanticOp::Cos => {
            let sin = builder.add_op(CoreSemanticOp::Sin, &[primal_input])?[0];
            builder.add_op(CoreSemanticOp::Neg, &[sin])?[0]
        }
        CoreSemanticOp::Tanh => {
            let tanh = builder.add_op(CoreSemanticOp::Tanh, &[primal_input])?[0];
            let square = builder.add_op(CoreSemanticOp::Mul, &[tanh, tanh])?[0];
            let one = one_like(builder, primal_input, role)?;
            builder.add_op(CoreSemanticOp::Sub, &[one, square])?[0]
        }
        CoreSemanticOp::Sqrt => {
            let sqrt = builder.add_op(CoreSemanticOp::Sqrt, &[primal_input])?[0];
            let twice = builder.add_op(CoreSemanticOp::Add, &[sqrt, sqrt])?[0];
            let one = one_like(builder, primal_input, role)?;
            builder.add_op(CoreSemanticOp::Div, &[one, twice])?[0]
        }
        CoreSemanticOp::Rsqrt => {
            let rsqrt = builder.add_op(CoreSemanticOp::Rsqrt, &[primal_input])?[0];
            let negated = builder.add_op(CoreSemanticOp::Neg, &[rsqrt])?[0];
            let twice = builder.add_op(CoreSemanticOp::Add, &[primal_input, primal_input])?[0];
            builder.add_op(CoreSemanticOp::Div, &[negated, twice])?[0]
        }
        CoreSemanticOp::Log1p => {
            let one = one_like(builder, primal_input, role)?;
            let denominator = builder.add_op(CoreSemanticOp::Add, &[primal_input, one])?[0];
            builder.add_op(CoreSemanticOp::Div, &[one, denominator])?[0]
        }
        _ => return Err(unsupported_core(role, op)),
    };
    Ok(coefficient)
}

fn one_like(
    builder: &mut SemanticProgramBuilder,
    anchor: ProgramValue,
    role: SemanticTransformRole,
) -> Result<ProgramValue, SemanticAdTransformError> {
    let metadata = builder.value_metadata(anchor)?.clone();
    let dtype = metadata.dtype();
    let bytes = match dtype {
        DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 1.0_f64.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut bytes = 1.0_f32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = 1.0_f64.to_le_bytes().to_vec();
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
        _ => {
            return Err(SemanticAdTransformError::UnsupportedMetadata {
                role,
                message: format!("cannot construct a differentiable one for {dtype:?}"),
            });
        }
    };
    let scalar = builder.add_op(CoreSemanticOp::Constant { dtype, bytes }, &[])?[0];
    if metadata.shape().is_empty() {
        Ok(scalar)
    } else {
        let shape = exact_shape(metadata.shape(), role, "one-like anchor")?;
        Ok(builder.add_op(
            CoreSemanticOp::BroadcastInDim {
                shape,
                dims: Vec::new(),
            },
            &[scalar],
        )?[0])
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
