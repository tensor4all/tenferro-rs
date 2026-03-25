use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{CudaBackend, SortPrimsDescriptor, TensorSortPrims};

use super::CudaContext;

impl<S: Scalar + PartialOrd> TensorSortPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &SortPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "sort family descriptor {desc:?}: CUDA not yet implemented"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _input: &Tensor<S>,
        _values_out: &mut Tensor<S>,
        _indices_out: &mut Tensor<i64>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "sort family execution: CUDA not yet implemented".into(),
        ))
    }

    fn has_sort_support(_desc: &SortPrimsDescriptor) -> bool {
        false
    }
}
