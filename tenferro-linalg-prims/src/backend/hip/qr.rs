use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::{KernelLinalgScalar, QrTensorResult};

pub(crate) fn qr<T: KernelLinalgScalar>(
    _: &mut tenferro_prims::RocmContext,
    _: &Tensor<T>,
) -> Result<QrTensorResult<T>> {
    super::common::unsupported("qr")
}
