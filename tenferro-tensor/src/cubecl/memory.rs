//! Host-to-device memory transfer via the CubeCL allocator.

use crate::Tensor;

/// Upload a host tensor into CubeCL-managed storage.
///
/// # Examples
///
/// ```
/// let _upload = tenferro_tensor::cubecl::upload_tensor;
/// let _ = _upload;
/// ```
pub fn upload_tensor(_rt: &super::CubeclRuntime, _tensor: &Tensor) -> crate::Result<Tensor> {
    todo!()
}
