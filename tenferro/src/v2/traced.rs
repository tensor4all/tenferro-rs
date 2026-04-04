use std::sync::Arc;

use computegraph::fragment::Fragment;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::v2::{DType, Tensor};

use super::engine::Engine;

pub struct TracedTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: computegraph::LocalValId,
    pub data: Option<Tensor>,
}

impl TracedTensor {
    pub fn from_tensor(_tensor: Tensor) -> Self {
        todo!()
    }

    pub fn eval(&self, _engine: &mut Engine) -> Tensor {
        todo!()
    }

    pub fn grad(
        _engine: &mut Engine,
        _output: &TracedTensor,
        _inputs: &[&TracedTensor],
    ) -> Vec<Tensor> {
        todo!()
    }

    pub fn jvp(
        _engine: &mut Engine,
        _primals: &[&TracedTensor],
        _tangents: &[&TracedTensor],
    ) -> Vec<TracedTensor> {
        todo!()
    }
}

pub fn eval_all(_engine: &mut Engine, _outputs: &[&TracedTensor]) -> Vec<Tensor> {
    todo!()
}
