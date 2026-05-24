use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GlobalValKey, LocalValId, OpEmitter, OpMode, ValRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::extension_runtime::ExtensionExecutor;
use tenferro_tensor::{Tensor, TensorBackend, TypedTensor};

use crate::eager_exec::{exec_op_on_tensors, exec_op_on_tensors_with_extension_executor};

pub struct EagerEmitter<'a, B: TensorBackend + 'static> {
    pub backend: &'a mut B,
    pub extension_executor: Option<&'a mut ExtensionExecutor<B>>,
    pub external_data: HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    pub results: Vec<Arc<Tensor>>,
}

impl<'a, B: TensorBackend + 'static> EagerEmitter<'a, B> {
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

    pub fn push_tensor(&mut self, tensor: Arc<Tensor>) -> LocalValId {
        let id = self.results.len();
        self.results.push(tensor);
        id
    }

    pub fn tensor(&self, id: LocalValId) -> Arc<Tensor> {
        Arc::clone(&self.results[id])
    }

    fn external_tensor(&mut self, key: &GlobalValKey<StdTensorOp>) -> Arc<Tensor> {
        if let Some(tensor) = self.external_data.get(key) {
            return Arc::clone(tensor);
        }

        let base_key = missing_tangent_base_key(key)
            .unwrap_or_else(|| panic!("EagerEmitter: missing external {:?}", key));
        let base = self
            .external_data
            .get(&base_key)
            .unwrap_or_else(|| panic!("EagerEmitter: missing tangent base {:?}", base_key));
        let zero = Arc::new(zero_like_tensor(base.as_ref()));
        self.external_data.insert(key.clone(), Arc::clone(&zero));
        zero
    }
}

impl<B: TensorBackend + 'static> OpEmitter<StdTensorOp> for EagerEmitter<'_, B> {
    fn add_op(
        &mut self,
        op: StdTensorOp,
        inputs: Vec<ValRef<StdTensorOp>>,
        _mode: OpMode,
    ) -> Vec<LocalValId> {
        let concrete_values: Vec<Arc<Tensor>> = inputs
            .iter()
            .map(|value| match value {
                ValRef::Local(id) => Arc::clone(&self.results[*id]),
                ValRef::External(key) => self.external_tensor(key),
            })
            .collect();
        let concrete: Vec<&Tensor> = concrete_values
            .iter()
            .map(|tensor| tensor.as_ref())
            .collect();

        let outputs = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            exec_op_on_tensors_with_extension_executor(
                &op,
                &concrete,
                self.backend,
                Some(extension_executor),
            )
        } else {
            exec_op_on_tensors(&op, &concrete, self.backend)
        }
        .unwrap_or_else(|err| panic!("eager exec failed for {:?}: {}", op, err));

        let base = self.results.len();
        for output in outputs {
            self.results.push(Arc::new(output));
        }
        (base..self.results.len()).collect()
    }
}

fn missing_tangent_base_key(key: &GlobalValKey<StdTensorOp>) -> Option<GlobalValKey<StdTensorOp>> {
    let GlobalValKey::Input(tangent_key) = key else {
        return None;
    };
    let TensorInputKey::Tangent { of, .. } = tangent_key else {
        return None;
    };
    Some(GlobalValKey::Input((**of).clone()))
}

fn zero_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::I32(tensor) => Tensor::I32(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::Bool(tensor) => Tensor::Bool(TypedTensor::from_vec_col_major(
            tensor.shape.clone(),
            vec![false; tensor.n_elements()],
        )),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape.clone())),
    }
}
