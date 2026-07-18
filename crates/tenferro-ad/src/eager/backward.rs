use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, TensorMeta};
use tenferro_runtime::{Error, ErrorPhase};
use tenferro_tensor::{DType, Tensor, TensorBackend, TypedTensor};
use tidu::eager::{BackwardExecutor, RecordedGraph};
use tidu::{ADRuleError, ADRuleResult, LinearizedGraph, PrimitiveGraph};

use crate::ad_rule_error::DeferredErrors;
use crate::eager_builder::EagerPrimitiveBuilder;
use crate::eager_exec::{exec_op_on_tensors, exec_op_on_tensors_with_extension_executor};
use crate::extension_runtime::ExtensionExecutor;
use crate::metadata::{
    push_metadata_scope, register_scoped_live_graph_metadata, tensor_meta_from_tensor,
    GlobalMetadataScope,
};

use super::{zero_like_tensor, EagerRuntime};

#[derive(Debug)]
pub(super) enum EagerAdFailure {
    Rule(ADRuleError),
    Runtime(Error),
}

impl std::fmt::Display for EagerAdFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rule(error) => error.fmt(formatter),
            Self::Runtime(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EagerAdFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rule(error) => Some(error),
            Self::Runtime(error) => Some(error),
        }
    }
}

impl From<ADRuleError> for EagerAdFailure {
    fn from(error: ADRuleError) -> Self {
        Self::Rule(error)
    }
}

impl From<Error> for EagerAdFailure {
    fn from(error: Error) -> Self {
        Self::Runtime(error)
    }
}

pub(crate) struct TenferroBackwardCallbacks<'a, B: TensorBackend + 'static> {
    backend: &'a mut B,
    extension_executor: Option<&'a mut ExtensionExecutor<B>>,
    runtime: Option<&'a EagerRuntime>,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    errors: DeferredErrors,
}

impl<'a, B: TensorBackend + 'static> TenferroBackwardCallbacks<'a, B> {
    #[cfg(test)]
    pub(crate) fn new(
        backend: &'a mut B,
        extension_executor: Option<&'a mut ExtensionExecutor<B>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Self {
        Self::with_runtime(backend, extension_executor, None, metadata_scopes)
    }

    pub(crate) fn with_runtime(
        backend: &'a mut B,
        extension_executor: Option<&'a mut ExtensionExecutor<B>>,
        runtime: Option<&'a EagerRuntime>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Self {
        Self {
            backend,
            extension_executor,
            runtime,
            metadata_scopes,
            errors: DeferredErrors::default(),
        }
    }

    pub(crate) fn take_error(&mut self) -> Option<ADRuleError> {
        self.errors.take_ad_rule()
    }

    pub(crate) fn take_typed_error(&mut self) -> Option<Error> {
        self.errors.take_typed()
    }

    fn record_error(&mut self, err: ADRuleError) {
        self.errors.record_ad_rule(err);
    }

    fn runtime_error(&mut self, err: Error) -> ADRuleError {
        self.errors.runtime("tenferro-ad.eager", err)
    }

    fn record_failure(&mut self, failure: EagerAdFailure) -> ADRuleError {
        match failure {
            EagerAdFailure::Rule(err) => {
                self.record_error(err.clone());
                err
            }
            EagerAdFailure::Runtime(err) => self.runtime_error(err),
        }
    }

    pub(super) fn execute_forward_graph(
        &mut self,
        graph: &Graph<StdTensorOp>,
        initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<ValueKey<StdTensorOp>, Arc<Tensor>> {
        if self.errors.has_error() {
            return initial_data.clone();
        }
        let mut all_values = initial_data.clone();
        let live_values = live_graph_values(graph);
        let mut input_metadata = Vec::with_capacity(graph.inputs().len());
        for &input_id in graph.inputs() {
            let key = graph.values()[input_id].key.clone();
            match eager_forward_input_metadata(&key, initial_data) {
                Ok(meta) => input_metadata.push((key, meta)),
                Err(err) => {
                    let ad_error = self.record_failure(err);
                    self.record_error(ad_error);
                    return all_values;
                }
            }
        }

        for op_node in graph.operations() {
            if linear_op_depends_on_tangents(&op_node.role) {
                continue;
            }
            if !op_node
                .outputs
                .iter()
                .any(|output_id| live_values.contains(output_id))
            {
                continue;
            }

            let mut resolved_values = Vec::with_capacity(op_node.inputs.len());
            for input in &op_node.inputs {
                let resolved = match input {
                    ValueRef::Local(local_id) => {
                        let key = &graph.values()[*local_id].key;
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                    ValueRef::External(key) => {
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                };
                match resolved {
                    Ok(value) => resolved_values.push(value),
                    Err(err) => {
                        let ad_error = self.record_failure(err);
                        self.record_error(ad_error);
                        return all_values;
                    }
                }
            }
            let resolved_inputs: Vec<&Tensor> =
                resolved_values.iter().map(|value| value.as_ref()).collect();
            let outputs_result =
                if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
                    exec_op_on_tensors_with_extension_executor(
                        &op_node.operation,
                        &resolved_inputs,
                        self.backend,
                        Some(extension_executor),
                    )
                } else {
                    exec_op_on_tensors(&op_node.operation, &resolved_inputs, self.backend)
                };
            let outputs = match outputs_result {
                Ok(outputs) => outputs,
                Err(err) => {
                    let ad_error = self.runtime_error(err);
                    self.record_error(ad_error);
                    return all_values;
                }
            };

            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                let key = graph.values()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        let metadata_scope =
            match register_scoped_live_graph_metadata(graph, &live_values, input_metadata) {
                Ok(scope) => scope,
                Err(err) => {
                    let ad_error = self.runtime_error(err);
                    self.record_error(ad_error);
                    return all_values;
                }
            };
        push_metadata_scope(&mut self.metadata_scopes, Arc::new(metadata_scope));

        all_values
    }
}

pub(super) fn missing_tangent_base_key(
    key: &ValueKey<StdTensorOp>,
) -> Option<ValueKey<StdTensorOp>> {
    let ValueKey::Input(tangent_key) = key else {
        return None;
    };
    let TensorInputKey::Tangent { of, .. } = tangent_key else {
        return None;
    };
    Some(ValueKey::Input((**of).clone()))
}

pub(super) fn eager_forward_input_metadata(
    key: &ValueKey<StdTensorOp>,
    initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
) -> Result<TensorMeta, EagerAdFailure> {
    if let Some(value) = initial_data.get(key) {
        return Ok(tensor_meta_from_tensor(value.as_ref()));
    }

    let base_key = missing_tangent_base_key(key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD forward metadata",
            ErrorPhase::Execution,
            format!("missing concrete eager value for {key:?}"),
        ))
    })?;
    let base = initial_data.get(&base_key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD forward metadata",
            ErrorPhase::Execution,
            format!("missing base eager value for {base_key:?}"),
        ))
    })?;
    Ok(tensor_meta_from_tensor(base.as_ref()))
}

pub(super) fn eager_forward_value<B: TensorBackend>(
    all_values: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    key: &ValueKey<StdTensorOp>,
    initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    backend: &mut B,
) -> Result<Arc<Tensor>, EagerAdFailure> {
    if let Some(value) = all_values.get(key) {
        return Ok(Arc::clone(value));
    }

    let base_key = missing_tangent_base_key(key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD forward value",
            ErrorPhase::Execution,
            format!("missing concrete eager value for {key:?}"),
        ))
    })?;
    let base = initial_data.get(&base_key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD forward value",
            ErrorPhase::Execution,
            format!("missing base eager value for {base_key:?}"),
        ))
    })?;
    let value =
        Arc::new(zero_like_tensor(base.as_ref(), backend).map_err(EagerAdFailure::Runtime)?);
    all_values.insert(key.clone(), Arc::clone(&value));
    Ok(value)
}

pub(super) fn live_graph_values(graph: &Graph<StdTensorOp>) -> HashSet<LocalValueId> {
    let mut producers = HashMap::new();
    for (op_index, op_node) in graph.operations().iter().enumerate() {
        for &output_id in &op_node.outputs {
            producers.insert(output_id, op_index);
        }
    }

    let mut live = HashSet::new();
    let mut stack = graph.outputs().to_vec();
    while let Some(local_id) = stack.pop() {
        if !live.insert(local_id) {
            continue;
        }
        let Some(&op_index) = producers.get(&local_id) else {
            continue;
        };
        for input in &graph.operations()[op_index].inputs {
            if let ValueRef::Local(input_id) = input {
                stack.push(*input_id);
            }
        }
    }

    live
}

fn linear_op_depends_on_tangents(mode: &OperationRole) -> bool {
    matches!(mode, OperationRole::Linearized { active_mask } if active_mask.iter().any(|is_active| *is_active))
}

pub(super) fn zero_from_exact_metadata<B: TensorBackend>(
    meta: &TensorMeta,
    backend: &mut B,
) -> Result<Option<Tensor>, EagerAdFailure> {
    let Some(shape) = meta.exact_shape() else {
        return Ok(None);
    };
    let shape = shape
        .into_iter()
        .map(|dim| dim.constant_value())
        .collect::<Option<Vec<_>>>();
    let Some(shape) = shape else {
        return Ok(None);
    };
    let host = match meta.dtype {
        DType::F32 => Tensor::F32(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
        DType::F64 => Tensor::F64(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
        DType::I32 => Tensor::I32(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
        DType::I64 => Tensor::I64(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
        DType::Bool => {
            let len = checked_zero_element_count(&shape)?;
            Tensor::Bool(
                TypedTensor::from_vec_col_major(shape, vec![false; len])
                    .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
            )
        }
        DType::C32 => Tensor::C32(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
        DType::C64 => Tensor::C64(
            TypedTensor::zeros(shape)
                .map_err(|err| EagerAdFailure::Runtime(Error::TensorRuntime(err)))?,
        ),
    };
    Ok(Some(backend.upload_host_tensor(&host).map_err(|err| {
        EagerAdFailure::Runtime(Error::TensorRuntime(err))
    })?))
}

pub(super) fn prefill_missing_linear_zero_values<B: TensorBackend>(
    linear: &LinearizedGraph<StdTensorOp>,
    external_data: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ctx: &mut ShapeGuardContext,
    backend: &mut B,
) -> Result<(), EagerAdFailure> {
    for value in linear.as_graph().values() {
        if external_data.contains_key(&value.key) {
            continue;
        }
        let Some(meta) = ctx
            .metadata_if_available(&ValueRef::External(value.key.clone()))
            .cloned()
        else {
            continue;
        };
        let Some(zero) = zero_from_exact_metadata(&meta, backend)? else {
            continue;
        };
        external_data.insert(value.key.clone(), Arc::new(zero));
    }
    Ok(())
}

pub(super) fn prefill_linear_residual_values<B: TensorBackend + 'static>(
    linear: &LinearizedGraph<StdTensorOp>,
    external_data: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    backend: &mut B,
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<(), EagerAdFailure> {
    let mut visited = HashSet::new();
    prefill_residual_graph_values(
        linear.residual_graph(),
        external_data,
        backend,
        &mut extension_executor,
        &mut visited,
    )
}

fn prefill_residual_graph_values<B: TensorBackend + 'static>(
    graph: &Graph<StdTensorOp>,
    external_data: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    backend: &mut B,
    extension_executor: &mut Option<&mut ExtensionExecutor<B>>,
    visited: &mut HashSet<*const Graph<StdTensorOp>>,
) -> Result<(), EagerAdFailure> {
    let graph_ptr: *const Graph<StdTensorOp> = graph;
    if !visited.insert(graph_ptr) {
        return Ok(());
    }

    for parent in graph.parents() {
        prefill_residual_graph_values(parent, external_data, backend, extension_executor, visited)?;
    }

    for op_node in graph.operations() {
        if op_node
            .outputs
            .iter()
            .all(|output_id| external_data.contains_key(&graph.values()[*output_id].key))
        {
            continue;
        }

        let mut resolved_values = Vec::with_capacity(op_node.inputs.len());
        for input in &op_node.inputs {
            let key = match input {
                ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                ValueRef::External(key) => key,
            };
            resolved_values.push(eager_residual_value(external_data, key, backend)?);
        }

        let resolved_inputs: Vec<&Tensor> =
            resolved_values.iter().map(|value| value.as_ref()).collect();
        let outputs = match extension_executor.as_deref_mut() {
            Some(executor) => exec_op_on_tensors_with_extension_executor(
                &op_node.operation,
                &resolved_inputs,
                backend,
                Some(executor),
            ),
            None => exec_op_on_tensors(&op_node.operation, &resolved_inputs, backend),
        }
        .map_err(EagerAdFailure::Runtime)?;

        for (output_id, output) in op_node.outputs.iter().zip(outputs) {
            let key = graph.values()[*output_id].key.clone();
            external_data.insert(key, Arc::new(output));
        }
    }

    Ok(())
}

pub(super) fn eager_residual_value<B: TensorBackend>(
    all_values: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    key: &ValueKey<StdTensorOp>,
    backend: &mut B,
) -> Result<Arc<Tensor>, EagerAdFailure> {
    if let Some(value) = all_values.get(key) {
        return Ok(Arc::clone(value));
    }

    let base_key = missing_tangent_base_key(key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD residual value",
            ErrorPhase::Execution,
            format!("missing concrete eager residual value for {key:?}"),
        ))
    })?;
    let base = all_values.get(&base_key).ok_or_else(|| {
        EagerAdFailure::Runtime(Error::runtime_state(
            "eager AD residual value",
            ErrorPhase::Execution,
            format!("missing base eager residual value for {base_key:?}"),
        ))
    })?;
    let value =
        Arc::new(zero_like_tensor(base.as_ref(), backend).map_err(EagerAdFailure::Runtime)?);
    all_values.insert(key.clone(), Arc::clone(&value));
    Ok(value)
}

fn checked_zero_element_count(shape: &[usize]) -> Result<usize, EagerAdFailure> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            EagerAdFailure::Runtime(Error::invalid_argument(
                "eager AD zero",
                ErrorPhase::Execution,
                "shape",
                format!("zero tensor shape product overflows for shape {shape:?}"),
            ))
        })
    })
}

impl<B: TensorBackend + 'static> BackwardExecutor<StdTensorOp>
    for TenferroBackwardCallbacks<'_, B>
{
    fn linearize_recorded_graph(
        &mut self,
        graph: &RecordedGraph<StdTensorOp>,
        output_slots: &[usize],
        ctx: &mut ShapeGuardContext,
    ) -> tidu::ADRuleResult<Arc<LinearizedGraph<StdTensorOp>>> {
        match self.runtime {
            Some(runtime) => runtime.cached_linearize_recorded_graph(graph, output_slots, ctx),
            None => graph.linearize(output_slots, ctx).map(Arc::new),
        }
    }

    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, StdTensorOp>,
        initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<ValueKey<StdTensorOp>, Arc<Tensor>> {
        self.execute_forward_graph(graph.as_graph(), initial_data)
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> tidu::ADRuleResult<Vec<Option<Arc<Tensor>>>> {
        if let Some(err) = self.take_error() {
            return Err(err);
        }
        let mut external_data = external_data.clone();
        prefill_linear_residual_values(
            linear,
            &mut external_data,
            self.backend,
            self.extension_executor.as_deref_mut(),
        )
        .map_err(|err| self.record_failure(err))?;
        ctx.refresh_global_metadata();
        prefill_missing_linear_zero_values(linear, &mut external_data, ctx, self.backend)
            .map_err(|err| self.record_failure(err))?;

        let mut builder = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            EagerPrimitiveBuilder::with_extension_executor(self.backend, extension_executor)
        } else {
            EagerPrimitiveBuilder::new(self.backend)
        };
        builder.external_data = external_data;
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| builder.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        let transpose_result =
            tidu::linear_transpose_with_builder(linear, &mut builder, &cotangent_seed_ids, ctx);
        let cotangent_ids = match transpose_result {
            Ok(ids) => ids,
            Err(err) => {
                let builder_typed_error = builder.take_typed_error();
                let builder_error = builder.take_error();
                drop(builder);
                if let Some(err) = builder_typed_error {
                    return Err(self.runtime_error(err));
                }
                if let Some(err) = builder_error {
                    return Err(err);
                }
                return Err(err);
            }
        };
        let cotangent_tensors_result = cotangent_ids
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| builder.tensor(id)).transpose())
            .collect::<ADRuleResult<Vec<_>>>();
        let builder_typed_error = builder.take_typed_error();
        let builder_error = builder.take_error();
        drop(builder);
        if let Some(err) = builder_typed_error {
            return Err(self.runtime_error(err));
        }
        if let Some(err) = builder_error {
            return Err(err);
        }
        cotangent_tensors_result
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        if self.errors.has_error() {
            return Arc::clone(a);
        }
        match self.backend.add(a.as_ref(), b.as_ref()) {
            Ok(sum) => Arc::new(sum),
            Err(err) => {
                let ad_error = self.runtime_error(Error::TensorRuntime(err));
                self.record_error(ad_error);
                Arc::clone(a)
            }
        }
    }
}
