use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GraphOperation, LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tenferro_runtime::ad_support::push_metadata_scope;
use tenferro_tensor::Tensor;
use tidu::eager::{BackwardExecutor, ForwardExecutor, ForwardTangentInput, RecordedGraph};
use tidu::{
    ADRuleError, ADRuleKind, ADRuleResult, LinearizedGraph, PrimitiveBuilder, PrimitiveGraph,
    PrimitiveValue,
};

use crate::ad_rule_error::{ad_rule_error_with_context, DeferredErrors};
use crate::eager_exec::exec_op_on_tensors_with_extension_executor;
use crate::error::{Error, Result};
use crate::extension_runtime::ExtensionExecutor;

use super::backward::{
    eager_forward_input_metadata, eager_forward_value, live_graph_values, missing_tangent_base_key,
    prefill_linear_residual_values, prefill_missing_linear_zero_values, EagerAdFailure,
};
use super::{
    eager_val_key, record_eager_outputs, tensor_ptr, zero_like_tensor, EagerRuntime, EagerTensor,
};
use crate::metadata::{
    push_metadata_scope as push_ad_metadata_scope, register_scoped_live_graph_metadata,
    GlobalMetadataScope,
};

pub(super) fn functional_vjp_optional(
    ctx: &Arc<EagerRuntime>,
    output: &EagerTensor,
    wrt: &EagerTensor,
    cotangent: &EagerTensor,
) -> Result<Option<EagerTensor>> {
    let seed = cotangent.materialized_arc()?;
    let mut backend = ctx.lock_backend()?;
    let mut extension_executor = ctx.lock_extension_executor()?;
    let mut callbacks = RecordingCallbacks::new(
        Arc::clone(ctx),
        &mut backend,
        Some(&mut *extension_executor),
        output.metadata_scopes.clone(),
    );
    callbacks.remember_tensor(output)?;
    callbacks.remember_tensor(wrt)?;
    callbacks.remember_tensor(cotangent)?;
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    if let Some(extension_rules) = &ctx.extension_rules {
        ad_ctx = ad_ctx.with_extension_rules(extension_rules.clone());
    }

    let cotangents_result = tidu::eager::backward(
        &output.key,
        output.trace.as_ref(),
        seed,
        &mut callbacks,
        &mut ad_ctx,
    );
    let callback_typed_error = callbacks.take_typed_error();
    let callback_error = callbacks.take_error();
    let cotangents = match (callback_typed_error, cotangents_result, callback_error) {
        (Some(err), _, _) => return Err(err),
        (_, _, Some(err)) => {
            return Err(ad_rule_error_with_context("vjp", err, &mut ad_ctx));
        }
        (_, Err(err), None) => {
            return Err(ad_rule_error_with_context("vjp", err, &mut ad_ctx));
        }
        (_, Ok(cotangents), None) => cotangents,
    };

    cotangents
        .get(&wrt.key)
        .map(|cotangent| callbacks.tensor_for_arc(cotangent))
        .transpose()
}

pub(super) fn functional_jvp(
    ctx: &Arc<EagerRuntime>,
    output: &EagerTensor,
    wrt: &EagerTensor,
    tangent: &EagerTensor,
) -> Result<Option<EagerTensor>> {
    let tangent_value = tangent.materialized_arc()?;
    let mut backend = ctx.lock_backend()?;
    let mut extension_executor = ctx.lock_extension_executor()?;
    let mut callbacks = RecordingCallbacks::new(
        Arc::clone(ctx),
        &mut backend,
        Some(&mut *extension_executor),
        output.metadata_scopes.clone(),
    );
    callbacks.remember_tensor(output)?;
    callbacks.remember_tensor(wrt)?;
    callbacks.remember_tensor(tangent)?;
    let tangent_seeds = HashMap::from([(wrt.key.clone(), tangent_value)]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    if let Some(extension_rules) = &ctx.extension_rules {
        ad_ctx = ad_ctx.with_extension_rules(extension_rules.clone());
    }

    let tangent_result = tidu::eager::try_forward(
        &output.key,
        output.trace.as_ref(),
        &tangent_seeds,
        &mut callbacks,
        &mut ad_ctx,
    );
    let callback_typed_error = callbacks.take_typed_error();
    let callback_error = callbacks.take_error();
    let tangent = match (callback_typed_error, tangent_result, callback_error) {
        (Some(err), _, _) => return Err(err),
        (_, _, Some(err)) => {
            return Err(ad_rule_error_with_context("jvp", err, &mut ad_ctx));
        }
        (_, Err(err), None) => {
            return Err(ad_rule_error_with_context("jvp", err, &mut ad_ctx));
        }
        (_, Ok(tangent), None) => tangent,
    };

    tangent
        .as_ref()
        .map(|tangent| callbacks.tensor_for_arc(tangent))
        .transpose()
}

struct RecordingCallbacks<'a> {
    ctx: Arc<EagerRuntime>,
    backend: &'a mut super::EagerBackend,
    extension_executor: Option<&'a mut ExtensionExecutor<super::EagerBackend>>,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    tensors_by_ptr: HashMap<usize, EagerTensor>,
    errors: DeferredErrors,
}

impl<'a> RecordingCallbacks<'a> {
    fn new(
        ctx: Arc<EagerRuntime>,
        backend: &'a mut super::EagerBackend,
        extension_executor: Option<&'a mut ExtensionExecutor<super::EagerBackend>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Self {
        Self {
            ctx,
            backend,
            extension_executor,
            metadata_scopes,
            tensors_by_ptr: HashMap::new(),
            errors: DeferredErrors::default(),
        }
    }

    fn take_error(&mut self) -> Option<ADRuleError> {
        self.errors.take_ad_rule()
    }

    fn take_typed_error(&mut self) -> Option<Error> {
        self.errors.take_typed()
    }

    fn record_error(&mut self, err: ADRuleError) {
        self.errors.record_ad_rule(err);
    }

    fn runtime_error(&mut self, err: Error) -> ADRuleError {
        self.errors.runtime("tenferro-ad.eager.functional", err)
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

    fn ptr(tensor: &Arc<Tensor>) -> usize {
        tensor_ptr(tensor)
    }

    fn remember_tensor(&mut self, tensor: &EagerTensor) -> Result<Arc<Tensor>> {
        let value = tensor.materialized_arc()?;
        self.tensors_by_ptr
            .insert(Self::ptr(&value), tensor.clone());
        Ok(value)
    }

    fn remember_result(&mut self, tensor: EagerTensor) -> ADRuleResult<Arc<Tensor>> {
        let value = tensor
            .materialized_arc()
            .map_err(|err| self.runtime_error(err))?;
        self.tensors_by_ptr.insert(Self::ptr(&value), tensor);
        Ok(value)
    }

    fn tensor_for_arc(&mut self, value: &Arc<Tensor>) -> Result<EagerTensor> {
        if let Some(tensor) = self.tensors_by_ptr.get(&Self::ptr(value)) {
            return Ok(tensor.clone());
        }
        let tensor = EagerTensor::new_result_arc(
            Arc::clone(&self.ctx),
            eager_val_key(),
            Arc::clone(value),
            false,
            None,
            Vec::new(),
        )?;
        self.tensors_by_ptr.insert(Self::ptr(value), tensor.clone());
        Ok(tensor)
    }

    fn tensor_for_key_or_value(
        &mut self,
        key: &ValueKey<StdTensorOp>,
        value: &Arc<Tensor>,
    ) -> ADRuleResult<EagerTensor> {
        if let Some(tensor) = self.tensors_by_ptr.get(&Self::ptr(value)) {
            return Ok(tensor.clone());
        }
        if let Some(record) = self
            .ctx
            .value_record_by_tensor(value)
            .map_err(|err| self.runtime_error(err))?
        {
            let tensor = EagerTensor::from_record(record);
            self.tensors_by_ptr.insert(Self::ptr(value), tensor.clone());
            return Ok(tensor);
        }
        if let Some(record) = self
            .ctx
            .value_record(key)
            .map_err(|err| self.runtime_error(err))?
        {
            let tensor = EagerTensor::from_record(record);
            self.tensors_by_ptr.insert(Self::ptr(value), tensor.clone());
            return Ok(tensor);
        }
        let tensor = EagerTensor::new_result_arc(
            Arc::clone(&self.ctx),
            key.clone(),
            Arc::clone(value),
            false,
            None,
            Vec::new(),
        )
        .map_err(|err| self.runtime_error(err))?;
        self.tensors_by_ptr.insert(Self::ptr(value), tensor.clone());
        Ok(tensor)
    }

    fn external_tensors(
        &mut self,
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> ADRuleResult<HashMap<ValueKey<StdTensorOp>, EagerTensor>> {
        external_data
            .iter()
            .map(|(key, value)| {
                self.tensor_for_key_or_value(key, value)
                    .map(|tensor| (key.clone(), tensor))
            })
            .collect()
    }

    fn add_recorded(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> ADRuleResult<Arc<Tensor>> {
        let lhs = self
            .tensor_for_arc(a)
            .map_err(|err| self.runtime_error(err))?;
        let rhs = self
            .tensor_for_arc(b)
            .map_err(|err| self.runtime_error(err))?;
        let mut builder = RecordingPrimitiveBuilder::new(
            Arc::clone(&self.ctx),
            self.backend,
            self.extension_executor.as_deref_mut(),
            HashMap::new(),
        );
        let outputs_result = builder.execute_operation(&StdTensorOp::Add, vec![lhs, rhs]);
        let builder_typed_error = builder.take_typed_error();
        let builder_error = builder.take_error();
        drop(builder);
        if let Some(err) = builder_typed_error {
            return Err(self.runtime_error(err));
        }
        if let Some(err) = builder_error {
            return Err(err);
        }
        let outputs = outputs_result?;
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| recording_error("eager AD add produced no outputs"))?;
        self.remember_result(output)
    }
}

impl BackwardExecutor<StdTensorOp> for RecordingCallbacks<'_> {
    fn linearize_recorded_graph(
        &mut self,
        graph: &RecordedGraph<StdTensorOp>,
        output_slots: &[usize],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Arc<LinearizedGraph<StdTensorOp>>> {
        self.ctx
            .cached_linearize_recorded_graph(graph, output_slots, ctx)
    }

    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, StdTensorOp>,
        initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<ValueKey<StdTensorOp>, Arc<Tensor>> {
        if self.errors.has_error() {
            return initial_data.clone();
        }
        let graph = graph.as_graph();
        let mut all_values = initial_data.clone();
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
            let outputs = match exec_op_on_tensors_with_extension_executor(
                &op_node.operation,
                &resolved_inputs,
                self.backend,
                self.extension_executor.as_deref_mut(),
            ) {
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

        let live_values = live_graph_values(graph);
        match register_scoped_live_graph_metadata(graph, &live_values, input_metadata) {
            Ok(scope) => push_ad_metadata_scope(&mut self.metadata_scopes, Arc::new(scope)),
            Err(err) => {
                let ad_error = self.runtime_error(err);
                self.record_error(ad_error);
            }
        }

        all_values
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<Arc<Tensor>>>> {
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
        let external_tensors = self.external_tensors(&external_data)?;
        let seed_tensors = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| {
                        self.tensor_for_arc(seed)
                            .map_err(|err| self.runtime_error(err))
                    })
                    .transpose()
            })
            .collect::<ADRuleResult<Vec<_>>>()?;
        let mut builder = RecordingPrimitiveBuilder::new(
            Arc::clone(&self.ctx),
            self.backend,
            self.extension_executor.as_deref_mut(),
            external_tensors,
        );
        let cotangent_seed_ids = seed_tensors
            .into_iter()
            .map(|maybe_tensor| maybe_tensor.map(|tensor| builder.push_tensor(tensor)))
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
        let cotangent_tensors = cotangent_ids
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| builder.tensor(id)).transpose())
            .collect::<ADRuleResult<Vec<_>>>()?;
        let builder_typed_error = builder.take_typed_error();
        let builder_error = builder.take_error();
        drop(builder);
        if let Some(err) = builder_typed_error {
            return Err(self.runtime_error(err));
        }
        if let Some(err) = builder_error {
            return Err(err);
        }
        cotangent_tensors
            .into_iter()
            .map(|maybe_tensor| {
                maybe_tensor
                    .map(|tensor| self.remember_result(tensor))
                    .transpose()
            })
            .collect()
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        if self.errors.has_error() {
            return Arc::clone(a);
        }
        match self.add_recorded(a, b) {
            Ok(sum) => sum,
            Err(err) => {
                self.record_error(err);
                Arc::clone(a)
            }
        }
    }
}

impl ForwardExecutor<StdTensorOp> for RecordingCallbacks<'_> {
    fn linearize_recorded_graph(
        &mut self,
        graph: &RecordedGraph<StdTensorOp>,
        output_slots: &[usize],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Arc<LinearizedGraph<StdTensorOp>>> {
        self.ctx
            .cached_linearize_recorded_graph(graph, output_slots, ctx)
    }

    fn run_linearized_forward(
        &mut self,
        linear: &LinearizedGraph<StdTensorOp>,
        tangent_in: &[ForwardTangentInput<StdTensorOp>],
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<Arc<Tensor>>>> {
        if let Some(err) = self.take_error() {
            return Err(err);
        }
        let mut external_data = external_data.clone();
        ctx.refresh_global_metadata();
        prefill_missing_linear_zero_values(linear, &mut external_data, ctx, self.backend)
            .map_err(|err| self.record_failure(err))?;
        let tangent_by_key: HashMap<_, _> = tangent_in
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let tangent_tensors = linear
            .tangent_inputs()
            .iter()
            .map(
                |(key, _)| match tangent_by_key.get(key).cloned().flatten() {
                    Some(tangent) => self
                        .tensor_for_arc(&tangent)
                        .map_err(|err| self.runtime_error(err)),
                    None => self.zero_tangent_for_key(key, &external_data),
                },
            )
            .collect::<ADRuleResult<Vec<_>>>()?;
        let external_tensors = self.external_tensors(&external_data)?;
        let mut builder = RecordingPrimitiveBuilder::new(
            Arc::clone(&self.ctx),
            self.backend,
            self.extension_executor.as_deref_mut(),
            external_tensors,
        );
        for ((_, expected_id), tangent) in linear.tangent_inputs().iter().zip(tangent_tensors) {
            let id = builder.push_tensor(tangent);
            if id != *expected_id {
                return Err(recording_error(format!(
                    "linearized forward expected tangent local {expected_id}, got {id}"
                )));
            }
        }

        for op_node in linear.as_graph().operations() {
            let inputs = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValueRef::Local(local_id) => PrimitiveValue::Local(*local_id),
                    ValueRef::External(key) => PrimitiveValue::External(key.clone()),
                })
                .collect();
            let output_ids =
                builder.add_primitive(op_node.operation.clone(), inputs, op_node.role.clone());
            let builder_typed_error = builder.take_typed_error();
            let builder_error = builder.take_error();
            if let Some(err) = builder_typed_error {
                drop(builder);
                return Err(self.runtime_error(err));
            }
            if let Some(err) = builder_error {
                drop(builder);
                return Err(err);
            }
            if output_ids != op_node.outputs {
                return Err(recording_error(format!(
                    "linearized forward expected outputs {:?}, got {:?}",
                    op_node.outputs, output_ids
                )));
            }
        }

        let tangent_tensors = linear
            .tangent_outputs()
            .iter()
            .map(|maybe_id| maybe_id.map(|id| builder.tensor(id)).transpose())
            .collect::<ADRuleResult<Vec<_>>>()?;
        let builder_typed_error = builder.take_typed_error();
        let builder_error = builder.take_error();
        drop(builder);
        if let Some(err) = builder_typed_error {
            return Err(self.runtime_error(err));
        }
        if let Some(err) = builder_error {
            return Err(err);
        }
        tangent_tensors
            .into_iter()
            .map(|maybe_tensor| {
                maybe_tensor
                    .map(|tensor| self.remember_result(tensor))
                    .transpose()
            })
            .collect()
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        <Self as BackwardExecutor<StdTensorOp>>::add_operands(self, a, b)
    }
}

impl RecordingCallbacks<'_> {
    fn zero_tangent_for_key(
        &mut self,
        key: &TensorInputKey,
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> ADRuleResult<EagerTensor> {
        let tangent_key = ValueKey::Input(key.clone());
        let base_key =
            missing_tangent_base_key(&tangent_key).unwrap_or_else(|| tangent_key.clone());
        let base = external_data.get(&base_key).ok_or_else(|| {
            recording_error(format!("missing tangent base eager value for {base_key:?}"))
        })?;
        let zero =
            zero_like_tensor(base.as_ref(), self.backend).map_err(|err| self.runtime_error(err))?;
        EagerTensor::new_result_arc(
            Arc::clone(&self.ctx),
            eager_val_key(),
            Arc::new(zero),
            false,
            None,
            Vec::new(),
        )
        .map_err(|err| self.runtime_error(err))
    }
}

struct RecordingPrimitiveBuilder<'a> {
    ctx: Arc<EagerRuntime>,
    backend: &'a mut super::EagerBackend,
    extension_executor: Option<&'a mut ExtensionExecutor<super::EagerBackend>>,
    external_data: HashMap<ValueKey<StdTensorOp>, EagerTensor>,
    results: Vec<EagerTensor>,
    errors: DeferredErrors,
}

impl<'a> RecordingPrimitiveBuilder<'a> {
    fn new(
        ctx: Arc<EagerRuntime>,
        backend: &'a mut super::EagerBackend,
        extension_executor: Option<&'a mut ExtensionExecutor<super::EagerBackend>>,
        external_data: HashMap<ValueKey<StdTensorOp>, EagerTensor>,
    ) -> Self {
        Self {
            ctx,
            backend,
            extension_executor,
            external_data,
            results: Vec::new(),
            errors: DeferredErrors::default(),
        }
    }

    fn push_tensor(&mut self, tensor: EagerTensor) -> LocalValueId {
        let id = self.results.len();
        self.results.push(tensor);
        id
    }

    fn tensor(&self, id: LocalValueId) -> ADRuleResult<EagerTensor> {
        self.results
            .get(id)
            .cloned()
            .ok_or_else(|| recording_error(format!("missing eager AD local {id}")))
    }

    fn take_error(&mut self) -> Option<ADRuleError> {
        self.errors.take_ad_rule()
    }

    fn take_typed_error(&mut self) -> Option<Error> {
        self.errors.take_typed()
    }

    fn record_error(&mut self, err: ADRuleError) {
        self.errors.record_ad_rule(err);
    }

    fn runtime_error(&mut self, err: Error) -> ADRuleError {
        self.errors.runtime("tenferro-ad.eager.functional", err)
    }

    fn dummy_output_ids(&self, operation: &StdTensorOp) -> Vec<LocalValueId> {
        (0..operation.output_count()).collect()
    }

    fn external_tensor(&mut self, key: &ValueKey<StdTensorOp>) -> ADRuleResult<EagerTensor> {
        if let Some(tensor) = self.external_data.get(key) {
            return Ok(tensor.clone());
        }
        let base_key = missing_tangent_base_key(key).ok_or_else(|| {
            recording_error(format!("missing external eager AD value for {key:?}"))
        })?;
        let base = self.external_data.get(&base_key).cloned().ok_or_else(|| {
            recording_error(format!(
                "missing tangent base eager AD value for {base_key:?}"
            ))
        })?;
        let base = base
            .materialized_arc()
            .map_err(|err| self.runtime_error(err))?;
        let zero =
            zero_like_tensor(base.as_ref(), self.backend).map_err(|err| self.runtime_error(err))?;
        let tensor = EagerTensor::new_result_arc(
            Arc::clone(&self.ctx),
            eager_val_key(),
            Arc::new(zero),
            false,
            None,
            Vec::new(),
        )
        .map_err(|err| self.runtime_error(err))?;
        self.external_data.insert(key.clone(), tensor.clone());
        Ok(tensor)
    }

    fn resolve_primitive_input(
        &mut self,
        input: PrimitiveValue<StdTensorOp>,
    ) -> ADRuleResult<EagerTensor> {
        match input {
            PrimitiveValue::Local(id) => self.tensor(id),
            PrimitiveValue::External(key) => self.external_tensor(&key),
        }
    }

    fn execute_operation(
        &mut self,
        operation: &StdTensorOp,
        inputs: Vec<EagerTensor>,
    ) -> ADRuleResult<Vec<EagerTensor>> {
        let input_values = inputs
            .iter()
            .map(|tensor| {
                tensor
                    .materialized_arc()
                    .map_err(|err| self.runtime_error(err))
            })
            .collect::<ADRuleResult<Vec<_>>>()?;
        let input_refs: Vec<_> = input_values.iter().map(|tensor| tensor.as_ref()).collect();
        let outputs = exec_op_on_tensors_with_extension_executor(
            operation,
            &input_refs,
            self.backend,
            self.extension_executor.as_deref_mut(),
        )
        .map_err(|err| self.runtime_error(err))?;
        let output_arcs: Vec<_> = outputs.into_iter().map(Arc::new).collect();

        if !inputs.iter().any(|tensor| tensor.requires_grad) {
            return output_arcs
                .into_iter()
                .map(|output| {
                    EagerTensor::new_result_arc(
                        Arc::clone(&self.ctx),
                        eager_val_key(),
                        output,
                        false,
                        None,
                        Vec::new(),
                    )
                    .map_err(|err| self.runtime_error(err))
                })
                .collect();
        }

        let input_refs: Vec<_> = inputs.iter().collect();
        let recorded = record_eager_outputs(operation, &output_arcs, &input_refs)
            .map_err(|err| self.runtime_error(err))?;
        if recorded.traces.len() != output_arcs.len() {
            return Err(recording_error(format!(
                "expected {} eager AD traces for {:?}, got {}",
                output_arcs.len(),
                operation,
                recorded.traces.len()
            )));
        }
        let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
        for input in &inputs {
            for scope in &input.metadata_scopes {
                push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
            }
        }

        recorded
            .traces
            .into_iter()
            .zip(output_arcs)
            .map(|(trace, output)| {
                EagerTensor::new_result_arc(
                    Arc::clone(&self.ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    metadata_scopes.clone(),
                )
                .map_err(|err| self.runtime_error(err))
            })
            .collect()
    }
}

impl PrimitiveBuilder<StdTensorOp> for RecordingPrimitiveBuilder<'_> {
    fn add_primitive(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<PrimitiveValue<StdTensorOp>>,
        _role: OperationRole,
    ) -> Vec<LocalValueId> {
        let resolved = inputs
            .into_iter()
            .map(|input| self.resolve_primitive_input(input))
            .collect::<ADRuleResult<Vec<_>>>();
        let outputs = match resolved.and_then(|inputs| self.execute_operation(&operation, inputs)) {
            Ok(outputs) => outputs,
            Err(err) => {
                self.record_error(err);
                return self.dummy_output_ids(&operation);
            }
        };
        let start = self.results.len();
        self.results.extend(outputs);
        (start..self.results.len()).collect()
    }
}

fn recording_error(message: impl ToString) -> ADRuleError {
    ADRuleError::invalid_input(
        "tenferro-ad.eager.functional",
        ADRuleKind::Transpose,
        message.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_walkers_cover_active_and_inactive_inputs() {
        let ctx = EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new());
        let active = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap();
        let inactive = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap();
        let seed = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
            Arc::clone(&ctx),
        )
        .unwrap();
        let output = active.mul(&active).unwrap();

        let vjp = functional_vjp_optional(&ctx, &output, &active, &seed)
            .unwrap()
            .unwrap();
        let jvp = functional_jvp(&ctx, &output, &active, &seed)
            .unwrap()
            .unwrap();
        assert_eq!(
            vjp.materialized().unwrap().as_slice::<f64>().unwrap(),
            &[4.0, 6.0]
        );
        assert_eq!(
            jvp.materialized().unwrap().as_slice::<f64>().unwrap(),
            &[4.0, 6.0]
        );

        assert!(functional_vjp_optional(&ctx, &output, &inactive, &seed)
            .unwrap()
            .is_none());
        assert!(functional_jvp(&ctx, &output, &inactive, &seed)
            .unwrap()
            .is_none());
    }

    #[test]
    fn recording_callbacks_keep_typed_failures_and_cache_materialized_values() {
        let ctx = EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new());
        let mut backend = ctx.lock_backend().unwrap();
        let mut extension_executor = ctx.lock_extension_executor().unwrap();
        let mut callbacks = RecordingCallbacks::new(
            Arc::clone(&ctx),
            &mut backend,
            Some(&mut *extension_executor),
            Vec::new(),
        );

        let rule_error = ADRuleError::invalid_input(
            "tenferro-ad.tests",
            ADRuleKind::Transpose,
            "synthetic callback rule failure",
        );
        let recorded_rule = callbacks.record_failure(EagerAdFailure::Rule(rule_error));
        assert!(matches!(recorded_rule, ADRuleError::InvalidInput { .. }));
        assert!(callbacks.take_error().is_some());

        let recorded_runtime =
            callbacks.record_failure(EagerAdFailure::Runtime(Error::runtime_state(
                "functional-tests",
                tenferro_runtime::ErrorPhase::Execution,
                "synthetic runtime failure",
            )));
        assert!(matches!(recorded_runtime, ADRuleError::InvalidInput { .. }));
        assert!(callbacks.take_typed_error().is_some());
        assert!(callbacks.take_error().is_some());

        let value = Arc::new(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap());
        let first = callbacks.tensor_for_arc(&value).unwrap();
        let second = callbacks.tensor_for_arc(&value).unwrap();
        assert_eq!(
            first.materialized().unwrap().as_slice::<f64>().unwrap(),
            &[1.0, 2.0]
        );
        assert_eq!(first.key, second.key);
    }
}
