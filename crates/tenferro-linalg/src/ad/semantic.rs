use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::traits::GraphOperation;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ad::extension::{ExtensionLinearTransposeRule, ExtensionLinearizeRule};
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim, TensorMeta};
use tenferro_runtime::program::{
    CoreSemanticOp, ProgramValue, ProgramValueMetadata, SemanticProgramBuilder,
};
use tidu::PrimitiveTransposeInput;

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
        let tangent_outputs = ExtensionLinearizeRule::linearize(
            self,
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
            LinalgOp::TriangularSolve { .. }
            | LinalgOp::LuSolvePrepared { .. }
            | LinalgOp::FullPivLuSolve { .. } => semantic_custom_transpose(
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
    let tangent_outputs = ExtensionLinearizeRule::linearize(
        &LinalgAdRule,
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
    let cotangent_inputs = ExtensionLinearTransposeRule::linear_transpose(
        &LinalgAdRule,
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
            let input_cotangents = transpose_recorded_operation(
                operation,
                &output_cotangents,
                active_mask,
                external_values,
                &fixed_locals,
                builder,
                role,
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
    external_values: &HashMap<ValueKey<StdTensorOp>, ProgramValue>,
    fixed_locals: &[Option<ProgramValue>],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> Result<Vec<Option<ProgramValue>>, SemanticAdError> {
    let Some(cotangent) = cotangent_outputs.first().copied().flatten() else {
        return Ok(vec![None; operation.inputs.len()]);
    };
    let fixed = |index: usize| {
        if active_mask.get(index).copied().unwrap_or(false) {
            None
        } else {
            match operation.inputs.get(index) {
                Some(ValueRef::External(key)) => external_values.get(key).copied(),
                Some(ValueRef::Local(local)) => fixed_locals.get(*local).copied().flatten(),
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
        StdTensorOp::Mul => transpose_mul(cotangent, active_mask, &fixed, builder, role),
        StdTensorOp::Div => transpose_div(cotangent, active_mask, &fixed, builder, role),
        StdTensorOp::DotGeneral { config } => {
            transpose_matrix_dot(cotangent, config, active_mask, &fixed, builder, role)
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
            external_values,
            fixed_locals,
            builder,
        ),
        other => Err(semantic_internal(
            role,
            format!("unsupported linear linalg fragment operation {other:?}"),
        )),
    }
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
    let outputs = ExtensionLinearTransposeRule::linear_transpose(
        &LinalgAdRule,
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

fn legacy_error(role: SemanticAdRuleRole, error: tidu::ADRuleError) -> SemanticAdError {
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
