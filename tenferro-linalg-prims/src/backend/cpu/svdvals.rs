use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::KernelLinalgScalar;

pub(crate) fn svdvals<T>(
    ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
) -> Result<Tensor<T::Real>>
where
    T: KernelLinalgScalar,
{
    Ok(super::thin_svd::thin_svd(ctx, a)?.s)
}
