use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{KernelLinalgScalar, SolveTensorExResult};

pub(crate) fn solve_ex<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
    _: &Tensor<T>,
) -> Result<SolveTensorExResult<T>> {
    super::common::unsupported("solve_ex")
}

pub(crate) fn solve<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
    _: &Tensor<T>,
) -> Result<Tensor<T>> {
    super::common::unsupported("solve")
}

pub(crate) fn lu_solve<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
    _: &Tensor<i32>,
    _: &Tensor<T>,
) -> Result<Tensor<T>> {
    super::common::unsupported("lu_solve")
}
