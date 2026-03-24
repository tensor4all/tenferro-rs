use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::KernelLinalgScalar;

pub(crate) fn svdvals<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<Tensor<T::Real>> {
    super::common::unsupported("svdvals")
}
