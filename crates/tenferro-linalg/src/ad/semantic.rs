use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::traits::GraphOperation;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::ad::PrimitiveTransposeInput;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim, TensorMeta};
use tenferro_runtime::program::{
    CoreSemanticOp, ProgramValue, ProgramValueMetadata, SemanticProgramBuilder,
};

use super::LinalgAdRule;
use crate::extension::{LinalgExtensionOp, LinalgOp};
use crate::LINALG_EXTENSION_FAMILY_ID;

/// Return the linalg semantic-program AD rule set.
///
/// # Errors
///
/// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] if the linalg
/// family identifier is invalid, or
/// [`SemanticExtensionRegistryError::DuplicateRule`] if a semantic rule role
/// is already registered.
///
/// # Examples
///
/// ```rust
/// let rules = tenferro_linalg::semantic_ad_rules().unwrap();
/// assert!(rules
///     .lookup_linearize(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID)
///     .is_some());
/// assert!(rules
///     .lookup_linear_transpose(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID)
///     .is_some());
/// assert!(rules
///     .lookup_primal_vjp(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID)
///     .is_none());
/// ```
pub fn semantic_ad_rules() -> Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(LinalgAdRule))?
        .with_linear_transpose(Arc::new(LinalgAdRule))
}

impl SemanticLinearizeRule for LinalgAdRule {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let op = semantic_linalg_op(request.op(), SemanticAdRuleRole::Linearize)?;
        if matches!(op.op(), LinalgOp::LuFactor | LinalgOp::SvdFull) {
            // These value-only operations can appear inside a differentiable
            // composite (for example, `solve` uses `LuFactor` outputs as
            // prepared-solve residuals).  Returning absent tangents lets the
            // composite rule differentiate through the primal inputs without
            // pretending that the factorization outputs are differentiable.
            // A caller requesting those outputs directly still observes that
            // no derivative output was produced, while VJP rejects their
            // unsupported transpose below.
            return Ok(SemanticLinearizeResult::new(
                std::iter::repeat_n(AdValue::Absent, request.primal_outputs().len()),
                [],
            ));
        }
        let legacy = LegacyInvocation::new(
            request.primal_inputs(),
            request.primal_outputs(),
            request.active_outputs(),
            builder,
        )?;
        let seed_values: Vec<_> = request
            .tangent_inputs()
            .iter()
            .copied()
            .map(AdValue::value)
            .collect();
        let tangent_inputs: Vec<_> = seed_values
            .iter()
            .enumerate()
            .map(|(index, value)| value.map(|_| index))
            .collect();
        let mut recorded = RecordedBuilder::with_seed_count(seed_values.len());
        let tangent_outputs = LinalgAdRule
            .linearize(
                op,
                &mut recorded,
                &legacy.input_keys,
                &legacy.output_keys,
                &tangent_inputs,
                &mut legacy.context.clone(),
            )
            .map_err(|error| legacy_error(SemanticAdRuleRole::Linearize, error))?;
        let locals = recorded.replay(
            &seed_values,
            &legacy.external_values,
            &legacy.shape_sources,
            builder,
            SemanticAdRuleRole::Linearize,
        )?;
        Ok(SemanticLinearizeResult::new(
            tangent_outputs.into_iter().map(|value| {
                value
                    .and_then(|local| locals.get(local).copied().flatten())
                    .map_or(AdValue::Absent, AdValue::Value)
            }),
            [],
        ))
    }
}

impl SemanticLinearTransposeRule for LinalgAdRule {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = semantic_linalg_op(request.op(), SemanticAdRuleRole::LinearTranspose)?;
        match op.op() {
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => semantic_triangular_solve_transpose(
                request.primal_inputs(),
                request.primal_outputs(),
                request.cotangent_outputs(),
                request.active_inputs(),
                builder,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            ),
            LinalgOp::LuSolvePrepared { .. } => {
                let active_inputs = lu_solve_prepared_transpose_active_inputs(
                    request.active_inputs(),
                    SemanticAdRuleRole::LinearTranspose,
                )?;
                semantic_custom_transpose(
                    request.op(),
                    request.primal_inputs(),
                    request.primal_outputs(),
                    request.cotangent_outputs(),
                    &active_inputs,
                    builder,
                    SemanticAdRuleRole::LinearTranspose,
                )
            }
            LinalgOp::FullPivLuSolve { .. } => semantic_custom_transpose(
                request.op(),
                request.primal_inputs(),
                request.primal_outputs(),
                request.cotangent_outputs(),
                request.active_inputs(),
                builder,
                SemanticAdRuleRole::LinearTranspose,
            ),
            LinalgOp::LuFactor | LinalgOp::SvdFull => Err(SemanticAdError::Unsupported {
                family_id: LINALG_EXTENSION_FAMILY_ID,
                role: SemanticAdRuleRole::LinearTranspose,
                message: format!("semantic linear transpose is unsupported for {:?}", op.op()),
            }),
            _ => semantic_linearized_transpose(
                request.op(),
                request.primal_inputs(),
                request.primal_outputs(),
                request.cotangent_outputs(),
                request.active_inputs(),
                builder,
            ),
        }
    }
}

fn lu_solve_prepared_transpose_active_inputs(
    active_inputs: &[bool],
    role: SemanticAdRuleRole,
) -> Result<[bool; 4], SemanticAdError> {
    let active_inputs: [bool; 4] = active_inputs.try_into().map_err(|_| {
        semantic_internal(
            role,
            format!(
                "lu_solve_prepared semantic transpose expected 4 active inputs, got {}",
                active_inputs.len()
            ),
        )
    })?;
    // Packed LU may be an active intermediate when `solve` lowers through
    // factorization. Pivot and parity slots remain non-cotangent-producing
    // residuals.
    Ok([active_inputs[0], false, false, active_inputs[3]])
}

#[allow(clippy::too_many_arguments)]
fn semantic_triangular_solve_transpose(
    primal_inputs: &[ProgramValue],
    primal_outputs: &[ProgramValue],
    cotangent_outputs: &[AdValue],
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let role = SemanticAdRuleRole::LinearTranspose;
    if primal_inputs.len() != 2
        || primal_outputs.len() != 1
        || cotangent_outputs.len() != 1
        || active_inputs.len() != 2
    {
        return Err(semantic_internal(
            role,
            "triangular_solve semantic transpose received malformed arity",
        ));
    }
    let Some(ct) = cotangent_outputs.first().copied().and_then(AdValue::value) else {
        return Ok(vec![AdValue::Absent; 2].into_boxed_slice());
    };

    let mut result = vec![AdValue::Absent; 2];
    if !active_inputs[0] && !active_inputs[1] {
        return Ok(result.into_boxed_slice());
    }

    let matrix_rank = builder.value_metadata(primal_inputs[0])?.shape().len();
    let rhs_rank = builder.value_metadata(primal_inputs[1])?.shape().len();
    if matrix_rank < 2 || rhs_rank < 2 {
        return Err(semantic_internal(
            role,
            "triangular_solve semantic transpose expects matrix operands",
        ));
    }
    if matrix_rank != rhs_rank {
        return Err(semantic_internal(
            role,
            "triangular_solve semantic transpose expects equal-rank operands",
        ));
    }

    let conjugated_a = conjugate_if_complex(builder, primal_inputs[0])?;
    let rhs_cotangent = builder.add_extension(
        Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
            left_side,
            lower,
            transpose_a: !transpose_a,
            unit_diagonal,
        })),
        &[conjugated_a, ct],
    )?[0];

    if active_inputs[1] {
        result[1] = AdValue::Value(rhs_cotangent);
    }
    if active_inputs[0] {
        let matrix_cotangent = semantic_solve_matrix_cotangent(
            builder,
            rhs_cotangent,
            primal_outputs[0],
            left_side,
            transpose_a,
            matrix_rank,
        )?;
        let k = if unit_diagonal {
            if lower {
                -1
            } else {
                1
            }
        } else {
            0
        };
        let projected = if lower {
            builder.add_op(CoreSemanticOp::Tril { k }, &[matrix_cotangent])?[0]
        } else {
            builder.add_op(CoreSemanticOp::Triu { k }, &[matrix_cotangent])?[0]
        };
        result[0] = AdValue::Value(projected);
    }

    Ok(result.into_boxed_slice())
}

fn semantic_linearized_transpose(
    op: &dyn tenferro_ad::extension::ExtensionOp,
    primal_inputs: &[ProgramValue],
    primal_outputs: &[ProgramValue],
    cotangent_outputs: &[AdValue],
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let legacy = LegacyInvocation::new(
        primal_inputs,
        primal_outputs,
        &cotangent_outputs
            .iter()
            .map(|value| matches!(value, AdValue::Value(_)))
            .collect::<Vec<_>>(),
        builder,
    )?;
    let tangent_inputs: Vec<_> = active_inputs
        .iter()
        .copied()
        .enumerate()
        .map(|(index, active)| active.then_some(index))
        .collect();
    let mut recorded = RecordedBuilder::with_seed_count(primal_inputs.len());
    let tangent_outputs = LinalgAdRule
        .linearize(
            op,
            &mut recorded,
            &legacy.input_keys,
            &legacy.output_keys,
            &tangent_inputs,
            &mut legacy.context.clone(),
        )
        .map_err(|error| legacy_error(SemanticAdRuleRole::LinearTranspose, error))?;
    recorded.transpose_linear_fragment(
        &tangent_outputs,
        cotangent_outputs,
        active_inputs,
        &legacy.external_values,
        &legacy.shape_sources,
        builder,
    )
}

fn semantic_custom_transpose(
    op: &dyn tenferro_ad::extension::ExtensionOp,
    primal_inputs: &[ProgramValue],
    primal_outputs: &[ProgramValue],
    cotangent_outputs: &[AdValue],
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let legacy = LegacyInvocation::new(
        primal_inputs,
        primal_outputs,
        &vec![true; primal_outputs.len()],
        builder,
    )?;
    let seed_values: Vec<_> = cotangent_outputs
        .iter()
        .copied()
        .map(AdValue::value)
        .collect();
    let cotangents: Vec<_> = seed_values
        .iter()
        .enumerate()
        .map(|(index, value)| value.map(|_| index))
        .collect();
    let transpose_inputs: Vec<_> = legacy
        .input_keys
        .iter()
        .cloned()
        .map(PrimitiveTransposeInput::Residual)
        .collect();
    let mut recorded = RecordedBuilder::with_seed_count(seed_values.len());
    let cotangent_inputs = LinalgAdRule
        .linear_transpose(
            op,
            &mut recorded,
            &cotangents,
            &transpose_inputs,
            active_inputs,
            &mut legacy.context.clone(),
        )
        .map_err(|error| legacy_error(role, error))?;
    let locals = recorded.replay(
        &seed_values,
        &legacy.external_values,
        &legacy.shape_sources,
        builder,
        role,
    )?;
    Ok(cotangent_inputs
        .into_iter()
        .map(|value| {
            value
                .and_then(|local| locals.get(local).copied().flatten())
                .map_or(AdValue::Absent, AdValue::Value)
        })
        .collect())
}

struct LegacyInvocation {
    context: ShapeGuardContext,
    input_keys: Vec<ValueKey<StdTensorOp>>,
    output_keys: Vec<ValueKey<StdTensorOp>>,
    external_values: HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    shape_sources: Vec<ProgramValue>,
}

impl LegacyInvocation {
    fn new(
        primal_inputs: &[ProgramValue],
        primal_outputs: &[ProgramValue],
        active_outputs: &[bool],
        builder: &SemanticProgramBuilder,
    ) -> Result<Self, SemanticAdError> {
        let values: Vec<_> = primal_inputs
            .iter()
            .chain(primal_outputs)
            .copied()
            .collect();
        let metadata: Vec<_> = values
            .iter()
            .copied()
            .map(|value| builder.value_metadata(value).cloned())
            .collect::<Result<_, _>>()?;
        let symbolic_inputs = synthetic_input_shapes(&metadata);
        let symbolic_input_refs: Vec<_> = symbolic_inputs.iter().map(Vec::as_slice).collect();
        let mut context = ShapeGuardContext::default();
        let mut external_values = HashMap::new();
        let keys: Vec<_> = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| {
                let key = ValueKey::Input(TensorInputKey::User {
                    id: u64::try_from(index + 1).expect("small semantic AD invocation"),
                });
                context.insert_metadata(
                    key.clone(),
                    legacy_metadata(&metadata[index], &symbolic_input_refs),
                );
                external_values.insert(key.clone(), value);
                key
            })
            .collect();
        let input_count = primal_inputs.len();
        let input_keys = keys[..input_count].to_vec();
        let output_keys = keys[input_count..].to_vec();
        let active_values: HashSet<_> = output_keys
            .iter()
            .zip(active_outputs)
            .filter(|(_, active)| **active)
            .map(|(key, _)| key.clone())
            .collect();
        context = context.with_linearize_active_values(Arc::new(active_values));
        Ok(Self {
            context,
            input_keys,
            output_keys,
            external_values,
            shape_sources: values,
        })
    }
}

fn legacy_metadata(metadata: &ProgramValueMetadata, input_shapes: &[&[SymDim]]) -> TensorMeta {
    let extents = metadata
        .shape()
        .iter()
        .cloned()
        .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, input_shapes)))
        .collect();
    TensorMeta::with_extents(metadata.dtype(), extents)
}

fn synthetic_input_shapes(metadata: &[ProgramValueMetadata]) -> Vec<Vec<SymDim>> {
    let mut ranks = Vec::<usize>::new();
    for expression in metadata
        .iter()
        .flat_map(ProgramValueMetadata::shape)
        .filter_map(ShapeExtent::bound_expr)
    {
        collect_input_ranks(expression, &mut ranks);
    }
    ranks
        .into_iter()
        .enumerate()
        .map(|(input, rank)| {
            (0..rank)
                .map(|axis| {
                    SymDim::tensor_axis(
                        u64::try_from(input + 1).expect("small semantic input index"),
                        axis,
                    )
                })
                .collect()
        })
        .collect()
}

fn collect_input_ranks(expression: &DimExpr, ranks: &mut Vec<usize>) {
    match expression {
        DimExpr::Const(_) => {}
        DimExpr::InputDim { input_idx, axis } => {
            if ranks.len() <= *input_idx {
                ranks.resize(*input_idx + 1, 0);
            }
            ranks[*input_idx] = ranks[*input_idx].max(*axis + 1);
        }
        DimExpr::Add(lhs, rhs)
        | DimExpr::Sub(lhs, rhs)
        | DimExpr::Mul(lhs, rhs)
        | DimExpr::FloorDiv(lhs, rhs)
        | DimExpr::Min(lhs, rhs)
        | DimExpr::Max(lhs, rhs) => {
            collect_input_ranks(lhs, ranks);
            collect_input_ranks(rhs, ranks);
        }
    }
}

struct RecordedOperation {
    operation: StdTensorOp,
    inputs: Vec<ValueRef<StdTensorOp>>,
    role: OperationRole,
    outputs: Vec<LocalValueId>,
}

struct RecordedBuilder {
    seed_count: usize,
    next_local: usize,
    operations: Vec<RecordedOperation>,
}

impl RecordedBuilder {
    fn with_seed_count(seed_count: usize) -> Self {
        Self {
            seed_count,
            next_local: seed_count,
            operations: Vec::new(),
        }
    }

    fn replay(
        &self,
        seeds: &[Option<ProgramValue>],
        external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
        shape_sources: &[ProgramValue],
        builder: &mut SemanticProgramBuilder,
        role: SemanticAdRuleRole,
    ) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
        let mut locals = vec![None; self.next_local];
        for (index, seed) in seeds.iter().copied().enumerate().take(self.seed_count) {
            locals[index] = seed;
        }
        for operation in &self.operations {
            let inputs: Vec<_> = operation
                .inputs
                .iter()
                .map(|input| match input {
                    ValueRef::External(key) => external_values.get(key).copied(),
                    ValueRef::Local(local) => locals.get(*local).copied().flatten(),
                })
                .collect::<Option<_>>()
                .ok_or_else(|| {
                    semantic_internal(
                        role,
                        "recorded linalg AD fragment references an unavailable value",
                    )
                })?;
            let outputs = emit_recorded_operation(
                &operation.operation,
                &inputs,
                shape_sources,
                builder,
                role,
            )?;
            for (local, value) in operation
                .outputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
            {
                locals[local] = Some(value);
            }
        }
        Ok(locals)
    }

    fn transpose_linear_fragment(
        &self,
        tangent_outputs: &[Option<LocalValueId>],
        cotangent_outputs: &[AdValue],
        active_inputs: &[bool],
        external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
        shape_sources: &[ProgramValue],
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let role = SemanticAdRuleRole::LinearTranspose;
        let fixed_locals = self.replay_fixed(external_values, shape_sources, builder, role)?;
        let mut cotangents = HashMap::<LocalValueId, ProgramValue>::new();
        for (tangent, cotangent) in tangent_outputs
            .iter()
            .copied()
            .zip(cotangent_outputs.iter().copied())
        {
            if let (Some(tangent), AdValue::Value(cotangent)) = (tangent, cotangent) {
                accumulate_local_cotangent(builder, &mut cotangents, tangent, cotangent)?;
            }
        }
        for operation in self.operations.iter().rev() {
            let Some(active_mask) = linear_active_mask(&operation.role) else {
                continue;
            };
            if !active_mask.iter().any(|active| *active) {
                continue;
            }
            let output_cotangents: Vec<_> = operation
                .outputs
                .iter()
                .map(|output| cotangents.remove(output))
                .collect();
            if output_cotangents.iter().all(Option::is_none) {
                continue;
            }
            let context = RecordedTransposeContext {
                recorded: self,
                external_values,
                fixed_locals: &fixed_locals,
                shape_sources,
                role,
            };
            let input_cotangents = transpose_recorded_operation(
                operation,
                &output_cotangents,
                active_mask,
                &context,
                builder,
            )?;
            for ((input, active), cotangent) in operation
                .inputs
                .iter()
                .zip(active_mask)
                .zip(input_cotangents)
            {
                if !active {
                    continue;
                }
                let (ValueRef::Local(input), Some(cotangent)) = (input, cotangent) else {
                    return Err(semantic_internal(
                        role,
                        "linear linalg fragment has a non-local active input",
                    ));
                };
                accumulate_local_cotangent(builder, &mut cotangents, *input, cotangent)?;
            }
        }
        Ok(active_inputs
            .iter()
            .copied()
            .enumerate()
            .map(|(input, active)| {
                if active {
                    cotangents
                        .remove(&input)
                        .map_or(AdValue::Absent, AdValue::Value)
                } else {
                    AdValue::Absent
                }
            })
            .collect())
    }

    fn replay_fixed(
        &self,
        external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
        shape_sources: &[ProgramValue],
        builder: &mut SemanticProgramBuilder,
        role: SemanticAdRuleRole,
    ) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
        let mut locals = vec![None; self.next_local];
        for operation in &self.operations {
            if linear_active_mask(&operation.role)
                .is_some_and(|mask| mask.iter().any(|active| *active))
            {
                continue;
            }
            let inputs =
                resolve_recorded_inputs(&operation.inputs, external_values, &locals, role)?;
            let outputs = emit_recorded_operation(
                &operation.operation,
                &inputs,
                shape_sources,
                builder,
                role,
            )?;
            for (local, value) in operation
                .outputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
            {
                locals[local] = Some(value);
            }
        }
        Ok(locals)
    }
}

impl PrimitiveRuleBuilder for RecordedBuilder {
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        let output_count = GraphOperation::output_count(&operation);
        let outputs: Vec<_> = (self.next_local..self.next_local + output_count).collect();
        self.next_local += output_count;
        self.operations.push(RecordedOperation {
            operation,
            inputs,
            role,
            outputs: outputs.clone(),
        });
        outputs
    }
}

fn linear_active_mask(role: &OperationRole) -> Option<&[bool]> {
    match role {
        OperationRole::Primary => None,
        OperationRole::Linearized { active_mask } => Some(active_mask),
    }
}

struct RecordedTransposeContext<'a> {
    recorded: &'a RecordedBuilder,
    external_values: &'a HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    fixed_locals: &'a [Option<ProgramValue>],
    shape_sources: &'a [ProgramValue],
    role: SemanticAdRuleRole,
}

fn resolve_recorded_inputs(
    inputs: &[ValueRef<StdTensorOp>],
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    locals: &[Option<ProgramValue>],
    role: SemanticAdRuleRole,
) -> Result<Vec<ProgramValue>, SemanticAdError> {
    inputs
        .iter()
        .map(|input| match input {
            ValueRef::External(key) => external_values.get(key).copied(),
            ValueRef::Local(local) => locals.get(*local).copied().flatten(),
        })
        .collect::<Option<_>>()
        .ok_or_else(|| {
            semantic_internal(
                role,
                "recorded linalg AD fragment references an unavailable fixed value",
            )
        })
}

fn emit_recorded_operation(
    operation: &StdTensorOp,
    inputs: &[ProgramValue],
    shape_sources: &[ProgramValue],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Box<[ProgramValue]>, SemanticAdError> {
    match operation {
        StdTensorOp::Extension(extension) => {
            Ok(builder.add_extension(Arc::clone(extension), inputs)?)
        }
        core => {
            let core = CoreSemanticOp::try_from(core).map_err(|_| {
                semantic_internal(role, "linalg AD emitted a non-semantic standard operation")
            })?;
            let recorded_core = core.clone();
            let (core, inputs) =
                localize_shape_expressions(core, inputs, shape_sources, builder, role).map_err(
                    |error| match error {
                        SemanticAdError::Invariant {
                            family_id,
                            role,
                            message,
                        } => SemanticAdError::Invariant {
                            family_id,
                            role,
                            message: format!("{message}; recorded operation {recorded_core:?}"),
                        },
                        other => other,
                    },
                )?;
            Ok(builder.add_op(core, &inputs)?)
        }
    }
}

fn localize_shape_expressions(
    operation: CoreSemanticOp,
    data_inputs: &[ProgramValue],
    shape_sources: &[ProgramValue],
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<(CoreSemanticOp, Vec<ProgramValue>), SemanticAdError> {
    let mut inputs = data_inputs.to_vec();
    let operation = match operation {
        CoreSemanticOp::Reshape { to_shape } => CoreSemanticOp::Reshape {
            to_shape: localize_dims(
                &to_shape,
                data_inputs,
                shape_sources,
                1,
                &mut inputs,
                builder,
                role,
            )?,
        },
        CoreSemanticOp::BroadcastInDim { shape, dims } => CoreSemanticOp::BroadcastInDim {
            shape: localize_dims(
                &shape,
                data_inputs,
                shape_sources,
                1,
                &mut inputs,
                builder,
                role,
            )?,
            dims,
        },
        CoreSemanticOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => CoreSemanticOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes: localize_dims(
                &slice_sizes,
                data_inputs,
                shape_sources,
                2,
                &mut inputs,
                builder,
                role,
            )?,
        },
        other => other,
    };
    Ok((operation, inputs))
}

fn localize_dims(
    dims: &[DimExpr],
    data_inputs: &[ProgramValue],
    shape_sources: &[ProgramValue],
    fixed_data_arity: usize,
    operation_inputs: &mut Vec<ProgramValue>,
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<DimExpr>, SemanticAdError> {
    dims.iter()
        .map(|dim| {
            localize_dim(
                dim,
                data_inputs,
                shape_sources,
                fixed_data_arity,
                operation_inputs,
                builder,
                role,
            )
        })
        .collect()
}

fn localize_dim(
    dim: &DimExpr,
    data_inputs: &[ProgramValue],
    shape_sources: &[ProgramValue],
    fixed_data_arity: usize,
    operation_inputs: &mut Vec<ProgramValue>,
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<DimExpr, SemanticAdError> {
    let binary = |lhs: &DimExpr,
                  rhs: &DimExpr,
                  constructor: fn(Box<DimExpr>, Box<DimExpr>) -> DimExpr,
                  operation_inputs: &mut Vec<ProgramValue>|
     -> Result<DimExpr, SemanticAdError> {
        Ok(constructor(
            Box::new(localize_dim(
                lhs,
                data_inputs,
                shape_sources,
                fixed_data_arity,
                operation_inputs,
                builder,
                role,
            )?),
            Box::new(localize_dim(
                rhs,
                data_inputs,
                shape_sources,
                fixed_data_arity,
                operation_inputs,
                builder,
                role,
            )?),
        ))
    };
    match dim {
        DimExpr::Const(value) => Ok(DimExpr::Const(*value)),
        DimExpr::InputDim { input_idx, axis } => {
            // Legacy linalg rules express shape dimensions in invocation
            // coordinates. Shape-aware primitive helpers append explicit
            // shape operands after the primitive's fixed data operands and
            // remap their dimensions into those local operand coordinates.
            // Keep those two coordinate spaces distinct; rank compatibility
            // cannot disambiguate them.
            let source = if *input_idx >= fixed_data_arity {
                data_inputs
                    .get(*input_idx)
                    .copied()
                    .or_else(|| shape_sources.get(*input_idx).copied())
            } else {
                shape_sources.get(*input_idx).copied()
            }
            .ok_or_else(|| {
                    semantic_internal(
                        role,
                        format!(
                            "linalg AD symbolic shape input {input_idx} is out of bounds for {} operation inputs and {} primal shape sources",
                            data_inputs.len(),
                            shape_sources.len()
                        ),
                    )
                })?;
            let rank = builder.value_metadata(source)?.shape().len();
            if *axis >= rank {
                return Err(semantic_internal(
                    role,
                    format!(
                        "linalg AD symbolic shape axis {axis} is out of bounds for source rank {rank}"
                    ),
                ));
            }
            let input_idx = operation_inputs
                .iter()
                .position(|value| *value == source)
                .unwrap_or_else(|| {
                    operation_inputs.push(source);
                    operation_inputs.len() - 1
                });
            debug_assert!(
                input_idx < data_inputs.len() + shape_sources.len(),
                "localized shape source must be an operation input"
            );
            Ok(DimExpr::InputDim {
                input_idx,
                axis: *axis,
            })
        }
        DimExpr::Add(lhs, rhs) => binary(lhs, rhs, DimExpr::Add, operation_inputs),
        DimExpr::Sub(lhs, rhs) => binary(lhs, rhs, DimExpr::Sub, operation_inputs),
        DimExpr::Mul(lhs, rhs) => binary(lhs, rhs, DimExpr::Mul, operation_inputs),
        DimExpr::FloorDiv(lhs, rhs) => binary(lhs, rhs, DimExpr::FloorDiv, operation_inputs),
        DimExpr::Min(lhs, rhs) => binary(lhs, rhs, DimExpr::Min, operation_inputs),
        DimExpr::Max(lhs, rhs) => binary(lhs, rhs, DimExpr::Max, operation_inputs),
    }
}

fn transpose_recorded_operation(
    operation: &RecordedOperation,
    cotangent_outputs: &[Option<ProgramValue>],
    active_mask: &[bool],
    context: &RecordedTransposeContext<'_>,
    builder: &mut SemanticProgramBuilder,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let Some(cotangent) = cotangent_outputs.first().copied().flatten() else {
        return Ok(vec![None; operation.inputs.len()]);
    };
    let fixed = |index: usize| {
        if active_mask.get(index).copied().unwrap_or(false) {
            None
        } else {
            match operation.inputs.get(index) {
                Some(ValueRef::External(key)) => context.external_values.get(key).copied(),
                Some(ValueRef::Local(local)) => context.fixed_locals.get(*local).copied().flatten(),
                None => None,
            }
        }
    };
    let unary = |value| Ok(vec![Some(value)]);
    match &operation.operation {
        StdTensorOp::Add => Ok(active_mask
            .iter()
            .map(|active| active.then_some(cotangent))
            .collect()),
        StdTensorOp::Sub => {
            let rhs = builder.add_op(CoreSemanticOp::Neg, &[cotangent])?[0];
            Ok(vec![
                active_mask[0].then_some(cotangent),
                active_mask[1].then_some(rhs),
            ])
        }
        StdTensorOp::Neg => {
            let value = builder.add_op(CoreSemanticOp::Neg, &[cotangent])?[0];
            unary(value)
        }
        StdTensorOp::Conj => {
            let value = builder.add_op(CoreSemanticOp::Conj, &[cotangent])?[0];
            unary(value)
        }
        StdTensorOp::Mul => transpose_mul(cotangent, active_mask, &fixed, builder, context.role),
        StdTensorOp::Div => transpose_div(cotangent, active_mask, &fixed, builder, context.role),
        StdTensorOp::DotGeneral { config } => transpose_matrix_dot(
            cotangent,
            config,
            active_mask,
            &fixed,
            builder,
            context.role,
        ),
        StdTensorOp::ReduceSum { axes } => {
            transpose_reduce_sum(context, operation, cotangent, axes, active_mask, builder)
        }
        StdTensorOp::Transpose { perm } => {
            let mut inverse = vec![0; perm.len()];
            for (output_axis, input_axis) in perm.iter().copied().enumerate() {
                inverse[input_axis] = output_axis;
            }
            let value =
                builder.add_op(CoreSemanticOp::Transpose { perm: inverse }, &[cotangent])?[0];
            unary(value)
        }
        StdTensorOp::Convert { from, to } => {
            let value = builder.add_op(
                CoreSemanticOp::Convert {
                    from: *to,
                    to: *from,
                },
                &[cotangent],
            )?[0];
            unary(value)
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            let value = builder.add_op(
                CoreSemanticOp::EmbedDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                &[cotangent],
            )?[0];
            unary(value)
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            let value = builder.add_op(
                CoreSemanticOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                &[cotangent],
            )?[0];
            unary(value)
        }
        StdTensorOp::Tril { k } => {
            let value = builder.add_op(CoreSemanticOp::Tril { k: *k }, &[cotangent])?[0];
            unary(value)
        }
        StdTensorOp::Triu { k } => {
            let value = builder.add_op(CoreSemanticOp::Triu { k: *k }, &[cotangent])?[0];
            unary(value)
        }
        StdTensorOp::Extension(extension) => transpose_linalg_extension(
            extension.as_ref(),
            operation,
            cotangent,
            active_mask,
            context.external_values,
            context.fixed_locals,
            builder,
        ),
        other => Err(semantic_internal(
            context.role,
            format!("unsupported linear linalg fragment operation {other:?}"),
        )),
    }
}

fn transpose_reduce_sum(
    context: &RecordedTransposeContext<'_>,
    operation: &RecordedOperation,
    cotangent: ProgramValue,
    axes: &[usize],
    active_mask: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    if operation.inputs.len() != 1 || active_mask.len() != 1 {
        return Err(semantic_internal(
            context.role,
            "linear reduce_sum fragment has malformed arity",
        ));
    }
    if !active_mask[0] {
        return Ok(vec![None]);
    }
    let mut cache = HashMap::new();
    let input_shape = recorded_value_shape(
        context.recorded,
        &operation.inputs[0],
        context.external_values,
        context.shape_sources,
        builder,
        context.role,
        &mut cache,
    )?
    .ok_or_else(|| {
        semantic_internal(
            context.role,
            "linear reduce_sum fragment is missing input shape metadata",
        )
    })?;
    if axes.iter().any(|axis| *axis >= input_shape.len()) {
        return Err(semantic_internal(
            context.role,
            format!(
                "linear reduce_sum axis is out of bounds for input rank {}",
                input_shape.len()
            ),
        ));
    }
    let dims: Vec<_> = (0..input_shape.len())
        .filter(|axis| !axes.contains(axis))
        .collect();
    let mut inputs = vec![cotangent];
    let shape = localize_dims(
        &input_shape,
        &[cotangent],
        context.shape_sources,
        1,
        &mut inputs,
        builder,
        context.role,
    )?;
    Ok(vec![Some(
        builder.add_op(CoreSemanticOp::BroadcastInDim { shape, dims }, &inputs)?[0],
    )])
}

fn recorded_value_shape(
    recorded: &RecordedBuilder,
    value: &ValueRef<StdTensorOp>,
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    shape_sources: &[ProgramValue],
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
    cache: &mut HashMap<LocalValueId, Option<Vec<DimExpr>>>,
) -> Result<Option<Vec<DimExpr>>, SemanticAdError> {
    match value {
        ValueRef::External(key) => {
            let source = external_values.get(key).copied().ok_or_else(|| {
                semantic_internal(
                    role,
                    "recorded linalg shape references missing external value",
                )
            })?;
            source_shape(source, shape_sources, builder, role).map(Some)
        }
        ValueRef::Local(local) => recorded_local_shape(
            recorded,
            *local,
            external_values,
            shape_sources,
            builder,
            role,
            cache,
        ),
    }
}

fn recorded_local_shape(
    recorded: &RecordedBuilder,
    local: LocalValueId,
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    shape_sources: &[ProgramValue],
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
    cache: &mut HashMap<LocalValueId, Option<Vec<DimExpr>>>,
) -> Result<Option<Vec<DimExpr>>, SemanticAdError> {
    if let Some(cached) = cache.get(&local) {
        return Ok(cached.clone());
    }
    let shape = if local < recorded.seed_count {
        let source = shape_sources.get(local).copied().ok_or_else(|| {
            semantic_internal(
                role,
                format!("recorded linalg seed local {local} has no shape source"),
            )
        })?;
        Some(source_shape(source, shape_sources, builder, role)?)
    } else {
        let (operation, output_index) = recorded
            .operations
            .iter()
            .find_map(|operation| {
                operation
                    .outputs
                    .iter()
                    .position(|output| *output == local)
                    .map(|index| (operation, index))
            })
            .ok_or_else(|| {
                semantic_internal(
                    role,
                    format!("recorded linalg local {local} has no producing operation"),
                )
            })?;
        recorded_operation_output_shape(
            recorded,
            operation,
            output_index,
            external_values,
            shape_sources,
            builder,
            role,
            cache,
        )?
    };
    cache.insert(local, shape.clone());
    Ok(shape)
}

fn source_shape(
    source: ProgramValue,
    shape_sources: &[ProgramValue],
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<DimExpr>, SemanticAdError> {
    let index = shape_sources
        .iter()
        .position(|candidate| *candidate == source)
        .ok_or_else(|| {
            semantic_internal(role, "shape source is not part of the linalg invocation")
        })?;
    let rank = builder.value_metadata(source)?.shape().len();
    Ok(DimExpr::input_shape(index, rank))
}

#[allow(clippy::too_many_arguments)]
fn recorded_operation_output_shape(
    recorded: &RecordedBuilder,
    operation: &RecordedOperation,
    output_index: usize,
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    shape_sources: &[ProgramValue],
    builder: &SemanticProgramBuilder,
    role: SemanticAdRuleRole,
    cache: &mut HashMap<LocalValueId, Option<Vec<DimExpr>>>,
) -> Result<Option<Vec<DimExpr>>, SemanticAdError> {
    let input_shape = |input_index: usize,
                       cache: &mut HashMap<LocalValueId, Option<Vec<DimExpr>>>|
     -> Result<Option<Vec<DimExpr>>, SemanticAdError> {
        let input = operation.inputs.get(input_index).ok_or_else(|| {
            semantic_internal(
                role,
                "recorded linalg shape requested missing operation input",
            )
        })?;
        recorded_value_shape(
            recorded,
            input,
            external_values,
            shape_sources,
            builder,
            role,
            cache,
        )
    };
    match &operation.operation {
        StdTensorOp::Extension(extension) => {
            let linalg = semantic_linalg_op(extension.as_ref(), role)?;
            match linalg.op() {
                LinalgOp::LuFactor => match output_index {
                    0 => input_shape(0, cache),
                    1 => Ok(input_shape(0, cache)?.map(|shape| {
                        let (rows, cols, batch) = recorded_matrix_shape_parts(&shape);
                        let mut pivots_shape =
                            vec![DimExpr::Min(Box::new(rows.clone()), Box::new(cols.clone()))];
                        pivots_shape.extend_from_slice(batch);
                        pivots_shape
                    })),
                    2 => Ok(input_shape(0, cache)?.map(|shape| shape[2..].to_vec())),
                    _ => Ok(None),
                },
                LinalgOp::LuSolvePrepared { .. } => input_shape(3, cache),
                _ => Ok(None),
            }
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => Ok(input_shape(0, cache)?
            .map(|shape| extract_diag_shape(&shape, *axis_a, *axis_b))
            .transpose()?),
        StdTensorOp::ReduceSum { axes } => Ok(input_shape(0, cache)?.map(|shape| {
            shape
                .into_iter()
                .enumerate()
                .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim))
                .collect()
        })),
        StdTensorOp::Convert { .. }
        | StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::Tril { .. }
        | StdTensorOp::Triu { .. } => input_shape(0, cache),
        StdTensorOp::Transpose { perm } => Ok(input_shape(0, cache)?
            .map(|shape| perm.iter().map(|axis| shape[*axis].clone()).collect())),
        _ => Ok(None),
    }
}

fn recorded_matrix_shape_parts(shape: &[DimExpr]) -> (&DimExpr, &DimExpr, &[DimExpr]) {
    (&shape[0], &shape[1], &shape[2..])
}

fn extract_diag_shape(
    shape: &[DimExpr],
    axis_a: usize,
    axis_b: usize,
) -> Result<Vec<DimExpr>, SemanticAdError> {
    if axis_a >= shape.len() || axis_b >= shape.len() || axis_a == axis_b {
        return Err(semantic_internal(
            SemanticAdRuleRole::LinearTranspose,
            "extract_diag shape derivation received invalid axes",
        ));
    }
    let diagonal = DimExpr::Min(
        Box::new(shape[axis_a].clone()),
        Box::new(shape[axis_b].clone()),
    );
    let mut output = Vec::with_capacity(shape.len() - 1);
    for (axis, dim) in shape.iter().enumerate() {
        if axis == axis_b {
            continue;
        }
        if axis == axis_a {
            output.push(diagonal.clone());
        } else {
            output.push(dim.clone());
        }
    }
    Ok(output)
}

fn transpose_mul(
    cotangent: ProgramValue,
    active_mask: &[bool],
    fixed: &impl Fn(usize) -> Option<ProgramValue>,
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let mut result = vec![None; 2];
    for input in 0..2 {
        if !active_mask[input] {
            continue;
        }
        let coefficient = fixed(1 - input).ok_or_else(|| {
            semantic_internal(role, "linear multiply is missing its fixed coefficient")
        })?;
        let coefficient = conjugate_if_complex(builder, coefficient)?;
        result[input] = Some(builder.add_op(CoreSemanticOp::Mul, &[cotangent, coefficient])?[0]);
    }
    Ok(result)
}

fn transpose_div(
    cotangent: ProgramValue,
    active_mask: &[bool],
    fixed: &impl Fn(usize) -> Option<ProgramValue>,
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let mut result = vec![None; 2];
    if active_mask[0] {
        let denominator = fixed(1).ok_or_else(|| {
            semantic_internal(role, "linear divide is missing its fixed denominator")
        })?;
        let denominator = conjugate_if_complex(builder, denominator)?;
        result[0] = Some(builder.add_op(CoreSemanticOp::Div, &[cotangent, denominator])?[0]);
    }
    if active_mask[1] {
        let numerator = fixed(0).ok_or_else(|| {
            semantic_internal(role, "linear divide is missing its fixed numerator")
        })?;
        let denominator = fixed(1).ok_or_else(|| {
            semantic_internal(role, "linear divide is missing its fixed denominator")
        })?;
        let square = builder.add_op(CoreSemanticOp::Mul, &[denominator, denominator])?[0];
        let coefficient = builder.add_op(CoreSemanticOp::Div, &[numerator, square])?[0];
        let coefficient = conjugate_if_complex(builder, coefficient)?;
        let value = builder.add_op(CoreSemanticOp::Mul, &[cotangent, coefficient])?[0];
        result[1] = Some(builder.add_op(CoreSemanticOp::Neg, &[value])?[0]);
    }
    Ok(result)
}

fn transpose_matrix_dot(
    cotangent: ProgramValue,
    config: &tenferro_tensor::DotGeneralConfig,
    active_mask: &[bool],
    fixed: &impl Fn(usize) -> Option<ProgramValue>,
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let rank = 2 + config.lhs_batch_dims.len();
    let expected_batch: Vec<_> = (2..rank).collect();
    if config.lhs_contracting_dims != [1]
        || config.rhs_contracting_dims != [0]
        || config.lhs_batch_dims != expected_batch
        || config.rhs_batch_dims != expected_batch
    {
        return Err(semantic_internal(
            role,
            "linalg AD emitted an unsupported dot-general configuration",
        ));
    }
    let mut result = vec![None; 2];
    if active_mask[0] {
        let rhs = fixed(1).ok_or_else(|| {
            semantic_internal(role, "linear matrix product is missing its fixed rhs")
        })?;
        let rhs_h = matrix_adjoint(builder, rhs, rank)?;
        result[0] = Some(
            builder.add_op(
                CoreSemanticOp::DotGeneral {
                    config: config.clone(),
                },
                &[cotangent, rhs_h],
            )?[0],
        );
    }
    if active_mask[1] {
        let lhs = fixed(0).ok_or_else(|| {
            semantic_internal(role, "linear matrix product is missing its fixed lhs")
        })?;
        let lhs_h = matrix_adjoint(builder, lhs, rank)?;
        result[1] = Some(
            builder.add_op(
                CoreSemanticOp::DotGeneral {
                    config: config.clone(),
                },
                &[lhs_h, cotangent],
            )?[0],
        );
    }
    Ok(result)
}

fn semantic_solve_matrix_cotangent(
    builder: &mut SemanticProgramBuilder,
    rhs_cotangent: ProgramValue,
    solution: ProgramValue,
    left_side: bool,
    transpose_a: bool,
    rank: usize,
) -> Result<ProgramValue, SemanticAdError> {
    let negative_rhs_cotangent = builder.add_op(CoreSemanticOp::Neg, &[rhs_cotangent])?[0];
    let solution_h = matrix_adjoint(builder, solution, rank)?;
    let config = semantic_matrix_multiply_config(rank)?;
    let matrix_cotangent = if left_side {
        builder.add_op(
            CoreSemanticOp::DotGeneral {
                config: config.clone(),
            },
            &[negative_rhs_cotangent, solution_h],
        )?[0]
    } else {
        builder.add_op(
            CoreSemanticOp::DotGeneral { config },
            &[solution_h, negative_rhs_cotangent],
        )?[0]
    };
    if transpose_a {
        semantic_matrix_transpose(builder, matrix_cotangent, rank)
    } else {
        Ok(matrix_cotangent)
    }
}

fn semantic_matrix_multiply_config(
    rank: usize,
) -> Result<tenferro_tensor::DotGeneralConfig, SemanticAdError> {
    if rank < 2 {
        return Err(semantic_internal(
            SemanticAdRuleRole::LinearTranspose,
            "matrix multiply semantic helper expects rank >= 2",
        ));
    }
    let batch_dims: Vec<usize> = (2..rank).collect();
    Ok(tenferro_tensor::DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: batch_dims.clone(),
        rhs_batch_dims: batch_dims,
    })
}

fn semantic_matrix_transpose(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
    rank: usize,
) -> Result<ProgramValue, SemanticAdError> {
    if rank < 2 {
        return Err(semantic_internal(
            SemanticAdRuleRole::LinearTranspose,
            "matrix transpose semantic helper expects rank >= 2",
        ));
    }
    let mut perm: Vec<_> = (0..rank).collect();
    perm.swap(0, 1);
    Ok(builder.add_op(CoreSemanticOp::Transpose { perm }, &[value])?[0])
}

fn matrix_adjoint(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
    rank: usize,
) -> Result<ProgramValue, SemanticAdError> {
    let value = conjugate_if_complex(builder, value)?;
    let mut perm: Vec<_> = (0..rank).collect();
    perm.swap(0, 1);
    Ok(builder.add_op(CoreSemanticOp::Transpose { perm }, &[value])?[0])
}

fn conjugate_if_complex(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
) -> Result<ProgramValue, SemanticAdError> {
    if matches!(
        builder.value_metadata(value)?.dtype(),
        tenferro_tensor::DType::C32 | tenferro_tensor::DType::C64
    ) {
        Ok(builder.add_op(CoreSemanticOp::Conj, &[value])?[0])
    } else {
        Ok(value)
    }
}

fn transpose_linalg_extension(
    extension: &dyn tenferro_ad::extension::ExtensionOp,
    operation: &RecordedOperation,
    cotangent: ProgramValue,
    active_mask: &[bool],
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    fixed_locals: &[Option<ProgramValue>],
    builder: &mut SemanticProgramBuilder,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let role = SemanticAdRuleRole::LinearTranspose;
    let linalg = semantic_linalg_op(extension, role)?;
    if !matches!(
        linalg.op(),
        LinalgOp::TriangularSolve { .. }
            | LinalgOp::LuSolvePrepared { .. }
            | LinalgOp::FullPivLuSolve { .. }
    ) {
        return Err(semantic_internal(
            role,
            format!(
                "linear linalg fragment contains unsupported extension {:?}",
                linalg.op()
            ),
        ));
    }
    let mut context = ShapeGuardContext::default();
    let mut replay_values = HashMap::new();
    let mut keys = Vec::with_capacity(operation.inputs.len());
    let mut shape_sources = Vec::with_capacity(operation.inputs.len());
    for (index, (input, active)) in operation.inputs.iter().zip(active_mask).enumerate() {
        let key = ValueKey::Input(TensorInputKey::User {
            id: 10_000 + u64::try_from(index).expect("small linalg extension arity"),
        });
        let value = if *active {
            cotangent
        } else {
            match input {
                ValueRef::External(key) => external_values.get(key).copied(),
                ValueRef::Local(local) => fixed_locals.get(*local).copied().flatten(),
            }
            .ok_or_else(|| {
                semantic_internal(role, "linear solve fragment is missing a fixed operand")
            })?
        };
        let metadata = builder.value_metadata(value)?.clone();
        let symbolic_shapes = synthetic_input_shapes(std::slice::from_ref(&metadata));
        let symbolic_shape_refs: Vec<_> = symbolic_shapes.iter().map(Vec::as_slice).collect();
        context.insert_metadata(
            key.clone(),
            legacy_metadata(&metadata, &symbolic_shape_refs),
        );
        if !active {
            replay_values.insert(key.clone(), value);
        }
        shape_sources.push(value);
        keys.push(key);
    }
    let inputs: Vec<_> = keys
        .iter()
        .cloned()
        .map(PrimitiveTransposeInput::Residual)
        .collect();
    let mut recorded = RecordedBuilder::with_seed_count(1);
    let outputs = LinalgAdRule
        .linear_transpose(
            extension,
            &mut recorded,
            &[Some(0)],
            &inputs,
            active_mask,
            &mut context,
        )
        .map_err(|error| legacy_error(role, error))?;
    let locals = recorded.replay(
        &[Some(cotangent)],
        &replay_values,
        &shape_sources,
        builder,
        role,
    )?;
    Ok(outputs
        .into_iter()
        .map(|output| output.and_then(|local| locals.get(local).copied().flatten()))
        .collect())
}

fn accumulate_local_cotangent(
    builder: &mut SemanticProgramBuilder,
    cotangents: &mut HashMap<LocalValueId, ProgramValue>,
    local: LocalValueId,
    cotangent: ProgramValue,
) -> Result<(), SemanticAdError> {
    if let Some(existing) = cotangents.get_mut(&local) {
        *existing = builder.add_op(CoreSemanticOp::Add, &[*existing, cotangent])?[0];
    } else {
        cotangents.insert(local, cotangent);
    }
    Ok(())
}

fn semantic_linalg_op(
    op: &dyn tenferro_ad::extension::ExtensionOp,
    role: SemanticAdRuleRole,
) -> Result<&LinalgExtensionOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: LINALG_EXTENSION_FAMILY_ID,
            role,
            message: "linalg semantic AD received an incompatible payload".into(),
        })
}

fn legacy_error(role: SemanticAdRuleRole, error: tenferro_ops::ad::ADRuleError) -> SemanticAdError {
    SemanticAdError::Rule {
        family_id: LINALG_EXTENSION_FAMILY_ID,
        role,
        source: Box::new(error),
    }
}

fn semantic_internal(role: SemanticAdRuleRole, message: impl Into<String>) -> SemanticAdError {
    SemanticAdError::Invariant {
        family_id: LINALG_EXTENSION_FAMILY_ID,
        role,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_runtime::program::ProgramInputSpec;
    use tenferro_tensor::DType;

    #[test]
    fn recorded_broadcast_prefers_primal_shape_source_over_rank_compatible_data_input() {
        let mut builder = SemanticProgramBuilder::new();
        let _row_anchor = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
            .unwrap();
        let _col_anchor = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                ],
            ))
            .unwrap();
        let vector = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Min(
                    Box::new(DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    }),
                    Box::new(DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    }),
                )],
            ))
            .unwrap();

        let (operation, inputs) = localize_shape_expressions(
            CoreSemanticOp::BroadcastInDim {
                shape: vec![
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 1,
                    },
                ],
                dims: vec![1],
            },
            &[vector],
            &[matrix],
            &builder,
            SemanticAdRuleRole::Linearize,
        )
        .unwrap();

        assert_eq!(inputs, vec![vector, matrix]);
        assert_eq!(
            operation,
            CoreSemanticOp::BroadcastInDim {
                shape: vec![
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 1,
                    },
                ],
                dims: vec![1],
            }
        );
    }
}
