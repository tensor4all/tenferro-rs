use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{EigTensorResult, EigenTensorResult, KernelLinalgScalar};

pub(crate) fn eigen_sym<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<EigenTensorResult<T>> {
    super::common::unsupported("eigen_sym")
}

pub(crate) fn eig<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<EigTensorResult<T>> {
    super::common::unsupported("eig")
}
