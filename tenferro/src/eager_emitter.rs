use std::collections::HashMap;
use std::sync::Arc;

use computegraph::{GlobalValKey, LocalValId, OpEmitter, OpMode, ValRef};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{Tensor, TensorBackend};

use crate::eager_exec::exec_op_on_tensors;

pub struct EagerEmitter<'a, B: TensorBackend> {
    pub backend: &'a mut B,
    pub external_data: HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    pub results: Vec<Arc<Tensor>>,
}

impl<'a, B: TensorBackend> EagerEmitter<'a, B> {
    pub fn new(backend: &'a mut B) -> Self {
        Self {
            backend,
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
}

impl<B: TensorBackend> OpEmitter<StdTensorOp> for EagerEmitter<'_, B> {
    fn add_op(
        &mut self,
        op: StdTensorOp,
        inputs: Vec<ValRef<StdTensorOp>>,
        _mode: OpMode,
    ) -> Vec<LocalValId> {
        let concrete: Vec<&Tensor> = inputs
            .iter()
            .map(|value| match value {
                ValRef::Local(id) => self.results[*id].as_ref(),
                ValRef::External(key) => self
                    .external_data
                    .get(key)
                    .unwrap_or_else(|| panic!("EagerEmitter: missing external {:?}", key))
                    .as_ref(),
            })
            .collect();

        let outputs = exec_op_on_tensors(&op, &concrete, self.backend)
            .unwrap_or_else(|err| panic!("eager exec failed for {:?}: {}", op, err));

        let base = self.results.len();
        for output in outputs {
            self.results.push(Arc::new(output));
        }
        (base..self.results.len()).collect()
    }
}
