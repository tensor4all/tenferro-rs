use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{CudaBackend, IndexingPrimsDescriptor, TensorIndexingPrims};

use super::CudaContext;

impl<S: Scalar> TensorIndexingPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &IndexingPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "indexing family descriptor {desc:?}: CUDA not yet implemented"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _inputs: &[&Tensor<S>],
        _indices: &Tensor<i64>,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "indexing family execution: CUDA not yet implemented".into(),
        ))
    }

    fn has_indexing_support(_desc: IndexingPrimsDescriptor) -> bool {
        false
    }
}
