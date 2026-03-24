use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{KernelLinalgScalar, LuTensorExResult, LuTensorResult};

pub(crate) fn lu_factor_ex<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<LuTensorExResult<T>> {
    super::common::unsupported("lu_factor_ex")
}

pub(crate) fn lu_factor<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<LuTensorResult<T>> {
    super::common::unsupported("lu_factor")
}

pub(crate) fn lu_factor_no_pivot<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<LuTensorResult<T>> {
    super::common::unsupported("lu_factor_no_pivot")
}
