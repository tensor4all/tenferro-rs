use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{KernelLinalgScalar, SvdTensorResult};

pub(crate) fn thin_svd<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<SvdTensorResult<T>> {
    super::common::unsupported("thin_svd")
}
