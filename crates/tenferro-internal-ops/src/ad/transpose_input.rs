use crate::ad::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};
use computegraph::types::{ValueKey, ValueRef};

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

pub struct TransposeInputRef<'a> {
    input: &'a PrimitiveTransposeInput<StdTensorOp>,
}

impl<'a> TransposeInputRef<'a> {
    pub fn new(input: &'a PrimitiveTransposeInput<StdTensorOp>) -> Self {
        Self { input }
    }

    pub fn key(&self) -> &ValueKey<StdTensorOp> {
        self.input.key()
    }

    pub fn metadata_value(&self) -> ValueRef<StdTensorOp> {
        ValueRef::External(self.key().clone())
    }

    pub fn fixed_value(&self, op: &str, index: usize) -> ADRuleResult<ValueRef<StdTensorOp>> {
        self.primal_or_residual_value(op, index, "retained as a tensor operand")
    }

    pub fn shape_source_value(
        &self,
        op: &str,
        index: usize,
    ) -> ADRuleResult<ValueRef<StdTensorOp>> {
        self.primal_or_residual_value(op, index, "used as a runtime shape source")
    }

    fn primal_or_residual_value(
        &self,
        op: &str,
        index: usize,
        use_case: &str,
    ) -> ADRuleResult<ValueRef<StdTensorOp>> {
        match self.input {
            PrimitiveTransposeInput::Residual(key) => Ok(ValueRef::External(key.clone())),
            PrimitiveTransposeInput::Linear {
                primal: Some(primal),
                ..
            } => Ok(ValueRef::External(primal.clone())),
            PrimitiveTransposeInput::Linear { key, primal: None } => {
                Err(ADRuleError::invalid_input(
                    op,
                    ADRuleKind::Transpose,
                    format!(
                        "transpose input {index} is linear-only and cannot be {use_case}: {key:?}"
                    ),
                ))
            }
        }
    }

    pub fn shape_operand(
        &self,
        rank: usize,
        source_idx: usize,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<(Vec<DimExpr>, Vec<ValueRef<StdTensorOp>>)> {
        let metadata_ref = ValueRef::External(self.key().clone());
        if !self.requires_runtime_shape_source() {
            if let Some(shape) = ctx.shape_if_available(&metadata_ref) {
                if let Some(resolved) =
                    shape_exprs_from_exact_metadata(&shape, rank, source_idx, ctx)
                {
                    return Ok(resolved);
                }
            }
        }

        self.runtime_shape_with_source(rank, source_idx)
    }

    fn requires_runtime_shape_source(&self) -> bool {
        match self.input {
            PrimitiveTransposeInput::Residual(key) => key_requires_runtime_shape_source(key),
            PrimitiveTransposeInput::Linear { key, primal } => {
                key_requires_runtime_shape_source(key)
                    || primal
                        .as_ref()
                        .is_some_and(key_requires_runtime_shape_source)
            }
        }
    }

    fn runtime_shape_with_source(
        &self,
        rank: usize,
        source_idx: usize,
    ) -> ADRuleResult<(Vec<DimExpr>, Vec<ValueRef<StdTensorOp>>)> {
        let shape = (0..rank)
            .map(|axis| DimExpr::InputDim {
                input_idx: source_idx,
                axis,
            })
            .collect();
        let shape_sources = if rank == 0 {
            Vec::new()
        } else {
            vec![self.runtime_shape_source()?]
        };
        Ok((shape, shape_sources))
    }

    fn runtime_shape_source(&self) -> ADRuleResult<ValueRef<StdTensorOp>> {
        match self.input {
            PrimitiveTransposeInput::Residual(key) => Ok(ValueRef::External(key.clone())),
            PrimitiveTransposeInput::Linear {
                primal: Some(primal),
                ..
            } => Ok(ValueRef::External(primal.clone())),
            PrimitiveTransposeInput::Linear { key, primal: None } => {
                Err(ADRuleError::invalid_input(
                    "transpose shape operand",
                    ADRuleKind::Transpose,
                    format!("linear-only value has no primal shape source: {key:?}"),
                ))
            }
        }
    }
}

pub(crate) fn metadata_value_refs(inputs: &[TransposeInputRef<'_>]) -> Vec<ValueRef<StdTensorOp>> {
    inputs
        .iter()
        .map(TransposeInputRef::metadata_value)
        .collect()
}

pub(crate) fn fixed_value_refs(
    op: &str,
    inputs: &[TransposeInputRef<'_>],
) -> ADRuleResult<Vec<ValueRef<StdTensorOp>>> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| input.fixed_value(op, index))
        .collect()
}

pub(crate) fn linearized_inputs_with_inactive_shape_sources(
    data: ValueRef<StdTensorOp>,
    shape_sources: Vec<ValueRef<StdTensorOp>>,
) -> (Vec<ValueRef<StdTensorOp>>, Vec<bool>) {
    let mut inputs = Vec::with_capacity(1 + shape_sources.len());
    inputs.push(data);
    inputs.extend(shape_sources);
    let mut active_mask = vec![false; inputs.len()];
    active_mask[0] = true;
    (inputs, active_mask)
}

pub(crate) fn shape_exprs_for_value_extent(
    input: &ValueRef<StdTensorOp>,
    rank: usize,
    source_idx: usize,
    ctx: &mut ShapeGuardContext,
) -> (Vec<DimExpr>, Vec<ValueRef<StdTensorOp>>) {
    if !value_ref_requires_runtime_shape_source(input) {
        if let Some(shape) = ctx.shape_if_available(input) {
            if let Some(resolved) = shape_exprs_from_exact_metadata(&shape, rank, source_idx, ctx) {
                return resolved;
            }
        }
    }

    runtime_shape_with_value_source(input, rank, source_idx)
}

fn shape_exprs_from_exact_metadata(
    shape: &[SymDim],
    rank: usize,
    source_idx: usize,
    ctx: &mut ShapeGuardContext,
) -> Option<(Vec<DimExpr>, Vec<ValueRef<StdTensorOp>>)> {
    let mut tensor_map = Vec::new();
    let mut shape_sources = Vec::new();
    for tensor_id in shape
        .iter()
        .flat_map(|dim| dim.referenced_tensor_ids().into_iter())
    {
        if tensor_map.iter().any(|(seen, _)| *seen == tensor_id) {
            continue;
        }
        let source = ctx.shape_source(tensor_id).cloned()?;
        let input_idx = source_idx + shape_sources.len();
        tensor_map.push((tensor_id, input_idx));
        shape_sources.push(ValueRef::External(source));
    }

    let converted = shape
        .iter()
        .map(|dim| dim.to_dim_expr(&tensor_map))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if converted.len() == rank {
        Some((converted, shape_sources))
    } else {
        None
    }
}

fn value_ref_requires_runtime_shape_source(input: &ValueRef<StdTensorOp>) -> bool {
    match input {
        ValueRef::Local(_) => true,
        ValueRef::External(key) => key_requires_runtime_shape_source(key),
    }
}

fn runtime_shape_with_value_source(
    input: &ValueRef<StdTensorOp>,
    rank: usize,
    source_idx: usize,
) -> (Vec<DimExpr>, Vec<ValueRef<StdTensorOp>>) {
    let shape = (0..rank)
        .map(|axis| DimExpr::InputDim {
            input_idx: source_idx,
            axis,
        })
        .collect();
    let shape_sources = if rank == 0 {
        Vec::new()
    } else {
        vec![input.clone()]
    };
    (shape, shape_sources)
}

fn key_requires_runtime_shape_source(key: &ValueKey<StdTensorOp>) -> bool {
    match key {
        ValueKey::Input(_) => false,
        ValueKey::Derived { operation, .. } => {
            matches!(
                operation.operation(),
                StdTensorOp::DynamicTruncate { .. } | StdTensorOp::PadToMatch { .. }
            ) || operation
                .inputs()
                .iter()
                .any(key_requires_runtime_shape_source)
        }
    }
}
