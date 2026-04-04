use std::sync::Arc;

use computegraph::fragment::Fragment;
use computegraph::LocalValId;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::v2::{DType, Tensor};

use super::engine::Engine;

// TracedTensor is standard-algebra only.
// Custom algebras use Fragment<SemiringOp<T>> directly (no TracedTensor).
pub struct TracedTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
}

impl TracedTensor {
    pub fn from_tensor(_tensor: Tensor) -> Self {
        todo!()
    }

    pub fn eval(&mut self, _engine: &mut Engine) -> &Tensor {
        todo!()
    }

    pub fn grad(&self, _wrt: &TracedTensor) -> TracedTensor {
        todo!()
    }

    pub fn jvp(&self, _wrt: &TracedTensor, _tangent: &TracedTensor) -> TracedTensor {
        todo!()
    }
}

// Evaluate multiple outputs together — shared primal nodes computed once.
pub fn eval_all(_engine: &mut Engine, _outputs: &mut [&mut TracedTensor]) -> Vec<Tensor> {
    todo!()
}
