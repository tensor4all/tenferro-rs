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
        let zero = Arc::new(zero_like_tensor(base.as_ref(), self.backend));
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

fn zero_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let host = match input {
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
    };
    backend
        .upload_host_tensor(&host)
        .unwrap_or_else(|err| panic!("eager primitive zero_like upload failed: {}", err))
}

#[cfg(test)]
mod tests {
    use computegraph::graph::GraphBuilder;
    use num_complex::{Complex32, Complex64};
    use tenferro_cpu::CpuBackend;
    use tidu::ADKey;

    use super::*;

    #[test]
    fn new_builder_executes_standard_primitives_without_extension_executor() {
        let mut backend = CpuBackend::new();
        let mut builder = EagerPrimitiveBuilder::new(&mut backend);
        let lhs = builder.push_tensor(Arc::new(Tensor::from_vec_col_major(
            vec![2],
            vec![1.0_f64, 2.0],
        )));
        let rhs = builder.push_tensor(Arc::new(Tensor::from_vec_col_major(
            vec![2],
            vec![3.0_f64, 4.0],
        )));

        let outputs = PrimitiveBuilder::add_primitive(
            &mut builder,
            StdTensorOp::Add,
            vec![PrimitiveValue::Local(lhs), PrimitiveValue::Local(rhs)],
            OperationRole::Primary,
        );

        assert_eq!(outputs.len(), 1);
        assert_eq!(
            builder.tensor(outputs[0]).as_slice::<f64>().unwrap(),
            &[4.0, 6.0]
        );
    }

    #[test]
    fn missing_tangent_external_uses_zero_like_primal_fallback() {
        let mut backend = CpuBackend::new();
        let mut builder = EagerPrimitiveBuilder::new(&mut backend);
        let primal_input = TensorInputKey::User { id: 7 };
        let primal_key = ValueKey::Input(primal_input.clone());
        let tangent_key = ValueKey::Input(primal_input.tangent_of(3));
        builder.external_data.insert(
            primal_key,
            Arc::new(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0])),
        );

        let outputs = PrimitiveBuilder::add_primitive(
            &mut builder,
            StdTensorOp::Neg,
            vec![PrimitiveValue::External(tangent_key.clone())],
            OperationRole::Primary,
        );

        assert_eq!(outputs.len(), 1);
        assert!(builder.external_data.contains_key(&tangent_key));
        assert_eq!(
            builder.tensor(outputs[0]).as_slice::<f64>().unwrap(),
            &[0.0, 0.0]
        );
    }

    #[test]
    fn missing_tangent_base_key_accepts_only_input_tangent_keys() {
        let primal_input = TensorInputKey::User { id: 11 };
        let primal_key = ValueKey::Input(primal_input.clone());
        assert_eq!(missing_tangent_base_key(&primal_key), None);

        let tangent_key = ValueKey::Input(primal_input.tangent_of(5));
        assert_eq!(missing_tangent_base_key(&tangent_key), Some(primal_key));

        let mut graph = GraphBuilder::<StdTensorOp>::new();
        let input = graph.add_input(TensorInputKey::User { id: 12 });
        let output = graph.add_operation(
            StdTensorOp::Neg,
            vec![ValueRef::Local(input)],
            OperationRole::Primary,
        )[0];
        let derived_key = graph.global_key(output).clone();
        assert_eq!(missing_tangent_base_key(&derived_key), None);
    }

    #[test]
    fn zero_like_tensor_covers_all_dtypes() {
        assert_zero_like_matches(Tensor::F32(TypedTensor::from_vec_col_major(
            vec![2],
            vec![1.0_f32, -2.0],
        )));
        assert_zero_like_matches(Tensor::F64(TypedTensor::from_vec_col_major(
            vec![2],
            vec![1.0_f64, -2.0],
        )));
        assert_zero_like_matches(Tensor::I32(TypedTensor::from_vec_col_major(
            vec![2],
            vec![1_i32, -2],
        )));
        assert_zero_like_matches(Tensor::I64(TypedTensor::from_vec_col_major(
            vec![2],
            vec![1_i64, -2],
        )));
        assert_zero_like_matches(Tensor::Bool(TypedTensor::from_vec_col_major(
            vec![2],
            vec![true, false],
        )));
        assert_zero_like_matches(Tensor::C32(TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 4.0)],
        )));
        assert_zero_like_matches(Tensor::C64(TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        )));
    }

    fn assert_zero_like_matches(input: Tensor) {
        let shape = input.shape().to_vec();
        let mut backend = CpuBackend::new();
        let zero = zero_like_tensor(&input, &mut backend);

        assert_eq!(zero.shape(), shape.as_slice());
        match zero {
            Tensor::F32(tensor) => assert_eq!(tensor.as_slice(), &[0.0_f32, 0.0]),
            Tensor::F64(tensor) => assert_eq!(tensor.as_slice(), &[0.0_f64, 0.0]),
            Tensor::I32(tensor) => assert_eq!(tensor.as_slice(), &[0_i32, 0]),
            Tensor::I64(tensor) => assert_eq!(tensor.as_slice(), &[0_i64, 0]),
            Tensor::Bool(tensor) => assert_eq!(tensor.as_slice(), &[false, false]),
            Tensor::C32(tensor) => assert_eq!(
                tensor.as_slice(),
                &[Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0)]
            ),
            Tensor::C64(tensor) => assert_eq!(
                tensor.as_slice(),
                &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
            ),
        }
    }
}
