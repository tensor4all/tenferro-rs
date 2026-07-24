//! Private bridge from tidu's core sweep to semantic extension AD.

use std::collections::{HashMap, HashSet};

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ExtensionAdDispatcher, ExtensionOp, ShapeGuardContext};
use tenferro_runtime::program::{
    FrozenProgram, ProgramInputSpec, ProgramValue, ProgramValueMetadata, SemanticOpRef,
    SemanticProgramBuilder,
};
use tidu::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};

use crate::semantic_extension::{SemanticAdError, SemanticExtensionRuleSet};
use crate::semantic_transform::{
    semantic_jvp, semantic_vjp, SemanticAdProgram, SemanticAdTransformError,
};

#[derive(Clone, Debug)]
pub(crate) struct SemanticCompatDispatcher {
    rules: SemanticExtensionRuleSet,
}

impl SemanticCompatDispatcher {
    pub(crate) fn new(rules: SemanticExtensionRuleSet) -> Self {
        Self { rules }
    }
}

impl ExtensionAdDispatcher for SemanticCompatDispatcher {
    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let source = source_program(op, primal_in, ctx, ADRuleKind::Jvp)?;
        let active_inputs: Vec<_> = tangent_in.iter().map(Option::is_some).collect();
        let derivative = semantic_jvp(&source, &active_inputs, &self.rules)
            .map_err(|error| semantic_error(op, ADRuleKind::Jvp, error))?;
        let source_inputs = primal_in
            .iter()
            .cloned()
            .map(ValueRef::External)
            .collect::<Vec<_>>();
        let seeds = tangent_in
            .iter()
            .copied()
            .map(|local| local.map(ValueRef::Local))
            .collect::<Vec<_>>();
        let primal_outputs = primal_out
            .iter()
            .cloned()
            .map(ValueRef::External)
            .collect::<Vec<_>>();
        replay_derivative(
            &derivative,
            &source_inputs,
            &seeds,
            Some(&primal_outputs),
            builder,
            op.family_id(),
            ADRuleKind::Jvp,
        )
    }

    fn transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveTransposeInput<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let input_keys = inputs
            .iter()
            .map(|input| match input {
                PrimitiveTransposeInput::Residual(key) => key.clone(),
                PrimitiveTransposeInput::Linear {
                    primal: Some(primal),
                    ..
                } => primal.clone(),
                PrimitiveTransposeInput::Linear { key, primal: None } => key.clone(),
            })
            .collect::<Vec<_>>();
        let source = source_program(op, &input_keys, ctx, ADRuleKind::Transpose)?;
        let active_inputs = match mode {
            OperationRole::Linearized { active_mask } => active_mask.clone(),
            OperationRole::Primary => inputs
                .iter()
                .map(|input| matches!(input, PrimitiveTransposeInput::Linear { .. }))
                .collect(),
        };
        let active_outputs: Vec<_> = cotangent_out.iter().map(Option::is_some).collect();
        let derivative = semantic_vjp(&source, &active_inputs, &active_outputs, &self.rules)
            .map_err(|error| semantic_error(op, ADRuleKind::Transpose, error))?;
        let source_inputs = input_keys
            .into_iter()
            .map(ValueRef::External)
            .collect::<Vec<_>>();
        let seeds = cotangent_out
            .iter()
            .copied()
            .map(|local| local.map(ValueRef::Local))
            .collect::<Vec<_>>();
        let primal_outputs = ctx.transpose_primal_outputs().map(|outputs| {
            outputs
                .iter()
                .cloned()
                .map(ValueRef::External)
                .collect::<Vec<_>>()
        });
        replay_derivative(
            &derivative,
            &source_inputs,
            &seeds,
            primal_outputs.as_deref(),
            builder,
            op.family_id(),
            ADRuleKind::Transpose,
        )
    }
}

fn source_program(
    op: &dyn ExtensionOp,
    input_keys: &[ValueKey<StdTensorOp>],
    ctx: &mut ShapeGuardContext,
    kind: ADRuleKind,
) -> ADRuleResult<FrozenProgram> {
    let mut builder = SemanticProgramBuilder::new();
    let mut inputs = Vec::with_capacity(input_keys.len());
    for (input_idx, key) in input_keys.iter().enumerate() {
        let metadata = ctx.metadata_of(&ValueRef::External(key.clone()))?;
        let extents: Vec<_> = metadata
            .extents()
            .iter()
            .enumerate()
            .map(|(axis, extent)| {
                extent
                    .as_exact()
                    .and_then(|dim| dim.to_dim_expr(&[]).ok())
                    .map(ShapeExtent::Exact)
                    .unwrap_or_else(|| ShapeExtent::Exact(DimExpr::InputDim { input_idx, axis }))
            })
            .collect();
        inputs.push(
            builder
                .input(ProgramInputSpec::from_metadata(
                    ProgramValueMetadata::from_extents(metadata.dtype, extents),
                ))
                .map_err(|error| {
                    ADRuleError::invalid_input(op.family_id(), kind, error.to_string())
                })?,
        );
    }
    let outputs = builder
        .add_extension(op.clone_arc(), &inputs)
        .map_err(|error| ADRuleError::invalid_input(op.family_id(), kind, error.to_string()))?;
    builder
        .finish(&outputs)
        .map_err(|error| ADRuleError::invalid_input(op.family_id(), kind, error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn replay_derivative(
    derivative: &SemanticAdProgram,
    source_inputs: &[ValueRef<StdTensorOp>],
    seeds: &[Option<ValueRef<StdTensorOp>>],
    primal_outputs: Option<&[ValueRef<StdTensorOp>]>,
    builder: &mut dyn PrimitiveRuleBuilder,
    family_id: &'static str,
    kind: ADRuleKind,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let program = derivative.frozen().program.as_ref();
    let mut values = HashMap::<ProgramValue, ValueRef<StdTensorOp>>::new();
    for (value, input) in program.inputs().iter().copied().zip(source_inputs) {
        values.insert(value, input.clone());
    }
    let mut linear_values = HashSet::new();
    for (source_index, derivative_index) in derivative
        .derivative_input_indices()
        .iter()
        .copied()
        .enumerate()
    {
        let (Some(derivative_index), Some(seed)) =
            (derivative_index, seeds.get(source_index).cloned().flatten())
        else {
            continue;
        };
        let value = program.inputs()[derivative_index];
        values.insert(value, seed);
        linear_values.insert(value);
    }

    for (operation_index, operation) in program.operations().enumerate() {
        if operation_index == 0 {
            if let Some(primal_outputs) = primal_outputs {
                if operation.outputs().len() != primal_outputs.len() {
                    return Err(ADRuleError::invalid_input(
                        family_id,
                        kind,
                        "semantic compatibility bridge primal output arity mismatch",
                    ));
                }
                for (value, output) in operation.outputs().iter().copied().zip(primal_outputs) {
                    values.insert(value, output.clone());
                }
                continue;
            }
        }
        let input_values = operation
            .inputs()
            .iter()
            .map(|value| values.get(value).cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                ADRuleError::invalid_input(
                    family_id,
                    kind,
                    "semantic compatibility bridge references an unavailable value",
                )
            })?;
        let active_mask = operation
            .inputs()
            .iter()
            .map(|value| linear_values.contains(value))
            .collect::<Vec<_>>();
        let role = if active_mask.iter().any(|active| *active) {
            OperationRole::Linearized {
                active_mask: active_mask.clone(),
            }
        } else {
            OperationRole::Primary
        };
        let std_op = match operation.op() {
            SemanticOpRef::Core(core) => StdTensorOp::from(core),
            SemanticOpRef::Extension(extension) => StdTensorOp::Extension(extension.clone_arc()),
            _ => {
                return Err(ADRuleError::invalid_input(
                    family_id,
                    kind,
                    "semantic compatibility bridge encountered an unknown operation",
                ));
            }
        };
        let outputs = builder.add_operation(std_op, input_values, role);
        if outputs.len() != operation.outputs().len() {
            return Err(ADRuleError::invalid_input(
                family_id,
                kind,
                "semantic compatibility bridge emitted the wrong output count",
            ));
        }
        let output_is_linear = active_mask.iter().any(|active| *active);
        for (value, local) in operation.outputs().iter().copied().zip(outputs) {
            values.insert(value, ValueRef::Local(local));
            if output_is_linear {
                linear_values.insert(value);
            }
        }
    }

    derivative
        .derivative_output_indices()
        .iter()
        .copied()
        .map(|index| {
            let Some(index) = index else {
                return Ok(None);
            };
            let output = program.outputs().get(index).ok_or_else(|| {
                ADRuleError::invalid_input(
                    family_id,
                    kind,
                    "semantic compatibility bridge derivative output index is invalid",
                )
            })?;
            match values.get(output) {
                Some(ValueRef::Local(local)) => Ok(Some(*local)),
                Some(ValueRef::External(_)) | None => Err(ADRuleError::invalid_input(
                    family_id,
                    kind,
                    "semantic compatibility bridge derivative output is not local",
                )),
            }
        })
        .collect()
}

fn semantic_error(
    op: &dyn ExtensionOp,
    kind: ADRuleKind,
    error: SemanticAdTransformError,
) -> ADRuleError {
    match error {
        SemanticAdTransformError::Extension(SemanticAdError::Unsupported { family_id, .. }) => {
            ADRuleError::unsupported(family_id, kind)
        }
        SemanticAdTransformError::Extension(SemanticAdError::MissingRule { .. })
        | SemanticAdTransformError::UnsupportedCore { .. }
        | SemanticAdTransformError::UnsupportedOperationVariant { .. }
        | SemanticAdTransformError::UnsupportedMetadata { .. } => {
            ADRuleError::unsupported(op.family_id(), kind)
        }
        other => ADRuleError::invalid_input(op.family_id(), kind, other.to_string()),
    }
}
