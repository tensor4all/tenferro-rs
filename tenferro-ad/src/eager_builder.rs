use std::collections::HashMap;
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
        let zero = Arc::new(zero_like_tensor(base.as_ref()));
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

fn zero_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::I32(tensor) => Tensor::I32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::Bool(tensor) => Tensor::Bool(TypedTensor::from_vec_col_major(
            tensor.shape().to_vec(),
            vec![false; tensor.n_elements()],
        )),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape().to_vec())),
    }
}
