use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::extension_runtime::ExtensionExecutor;
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{Tensor, TensorBackend, TypedTensor};
use tidu::{PrimitiveBuilder, PrimitiveValue};

use crate::eager_exec::{exec_op_on_tensors, exec_op_on_tensors_with_extension_executor};

pub struct EagerPrimitiveBuilder<'a, B: TensorBackend + 'static> {
    pub backend: &'a mut B,
    pub extension_executor: Option<&'a mut ExtensionExecutor<B>>,
    pub external_data: HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    pub results: Vec<Arc<Tensor>>,
}

impl<B: TensorBackend + 'static> fmt::Debug for EagerPrimitiveBuilder<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EagerPrimitiveBuilder")
            .field("backend_type", &std::any::type_name::<B>())
            .field("has_extension_executor", &self.extension_executor.is_some())
            .field("external_data_len", &self.external_data.len())
            .field("results_len", &self.results.len())
            .finish_non_exhaustive()
    }
}

impl<'a, B: TensorBackend + 'static> EagerPrimitiveBuilder<'a, B> {
    pub fn new(backend: &'a mut B) -> Self {
        Self {
            backend,
            extension_executor: None,
            external_data: HashMap::new(),
            results: Vec::new(),
        }
    }

    pub fn with_extension_executor(
        backend: &'a mut B,
        extension_executor: &'a mut ExtensionExecutor<B>,
    ) -> Self {
        Self {
            backend,
            extension_executor: Some(extension_executor),
            external_data: HashMap::new(),
            results: Vec::new(),
        }
    }

    pub fn push_tensor(&mut self, tensor: Arc<Tensor>) -> LocalValueId {
        let id = self.results.len();
        self.results.push(tensor);
        id
    }

    pub fn tensor(&self, id: LocalValueId) -> Arc<Tensor> {
        Arc::clone(&self.results[id])
    }

    fn external_tensor(&mut self, key: &ValueKey<StdTensorOp>) -> Arc<Tensor> {
        if let Some(tensor) = self.external_data.get(key) {
            return Arc::clone(tensor);
        }

        let base_key = missing_tangent_base_key(key)
            .unwrap_or_else(|| panic!("EagerPrimitiveBuilder: missing external {:?}", key));
        let base = self.external_data.get(&base_key).unwrap_or_else(|| {
            panic!("EagerPrimitiveBuilder: missing tangent base {:?}", base_key)
        });
        // tidu's PrimitiveBuilder trait has no error channel here; the backward
        // setup has already validated that backend zero creation is available.
        let zero = Arc::new(
            zero_like_tensor(base.as_ref(), self.backend)
                .expect("eager primitive tangent zero creation"),
        );
        self.external_data.insert(key.clone(), Arc::clone(&zero));
        zero
    }

    fn execute_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
    ) -> Vec<LocalValueId> {
        let concrete_values: Vec<Arc<Tensor>> = inputs
            .iter()
            .map(|value| match value {
                ValueRef::Local(id) => Arc::clone(&self.results[*id]),
                ValueRef::External(key) => self.external_tensor(key),
            })
            .collect();
        let concrete: Vec<&Tensor> = concrete_values
            .iter()
            .map(|tensor| tensor.as_ref())
            .collect();

        let outputs = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            exec_op_on_tensors_with_extension_executor(
                &operation,
                &concrete,
                self.backend,
                Some(extension_executor),
            )
        } else {
            exec_op_on_tensors(&operation, &concrete, self.backend)
        }
        .unwrap_or_else(|err| panic!("eager exec failed for {:?}: {}", operation, err));

        let base = self.results.len();
        for output in outputs {
            self.results.push(Arc::new(output));
        }
        (base..self.results.len()).collect()
    }
}

impl<B: TensorBackend + 'static> PrimitiveBuilder<StdTensorOp> for EagerPrimitiveBuilder<'_, B> {
    fn add_primitive(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<PrimitiveValue<StdTensorOp>>,
        _role: OperationRole,
    ) -> Vec<LocalValueId> {
        let inputs = inputs.into_iter().map(ValueRef::from).collect();
        self.execute_operation(operation, inputs)
    }
}

fn missing_tangent_base_key(key: &ValueKey<StdTensorOp>) -> Option<ValueKey<StdTensorOp>> {
    let ValueKey::Input(tangent_key) = key else {
        return None;
    };
    let TensorInputKey::Tangent { of, .. } = tangent_key else {
        return None;
    };
    Some(ValueKey::Input((**of).clone()))
}

fn zero_like_tensor<B: TensorBackend>(
    input: &Tensor,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let host = match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::I32(tensor) => Tensor::I32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::Bool(tensor) => Tensor::Bool(TypedTensor::from_vec_col_major(
            tensor.shape().to_vec(),
            vec![false; tensor.n_elements()],
        )?),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape().to_vec())?),
    };
    backend.upload_host_tensor(&host)
}

#[cfg(test)]
mod tests;
