use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::KernelLinalgScalar;

pub(crate) fn solve_triangular<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
    _: &Tensor<T>,
    _: bool,
) -> Result<Tensor<T>> {
    super::common::unsupported("solve_triangular")
}
