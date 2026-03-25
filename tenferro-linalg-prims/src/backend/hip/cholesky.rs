use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{CholeskyTensorExResult, KernelLinalgScalar};

pub(crate) fn cholesky_ex<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<CholeskyTensorExResult<T>> {
    super::common::unsupported("cholesky_ex")
}

pub(crate) fn cholesky<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<Tensor<T>> {
    super::common::unsupported("cholesky")
}
