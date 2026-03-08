//! Conversion utilities between Burn tensor primitives and tenferro tensors.
//!
//! # Current Limitations
//!
//! These conversion functions currently only support `f64` element type.
//! Support for `f32` and other numeric types will be added in future versions.

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// Convert a Burn backend tensor primitive into a tenferro `Tensor<f64>`.
///
/// # Current Limitations
///
/// This function currently always returns `Tensor<f64>` regardless of the
/// backend's float element type. Support for other element types (e.g., `f32`)
/// will be added in future versions.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use tenferro_burn::convert::burn_to_tenferro;
///
/// let burn_prim: <NdArray<f64> as burn::tensor::backend::Backend>::FloatTensorPrimitive =
///     todo!();
/// let tenferro_t: tenferro_tensor::Tensor<f64> = burn_to_tenferro::<NdArray<f64>>(burn_prim);
/// ```
pub fn burn_to_tenferro<B: Backend>(_tensor: FloatTensor<B>) -> tenferro_tensor::Tensor<f64> {
    todo!()
}

/// Convert a tenferro `Tensor<f64>` into a Burn backend tensor primitive.
///
/// The `device` parameter specifies which Burn device the resulting tensor
/// should be placed on. For the `NdArray` backend this is typically
/// `NdArrayDevice::Cpu`, obtainable via `Default::default()`.
///
/// # Current Limitations
///
/// This function currently only accepts `Tensor<f64>`. Support for other
/// element types will be added in future versions.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use burn::backend::ndarray::NdArrayDevice;
/// use tenferro_burn::convert::tenferro_to_burn;
///
/// let tenferro_t: tenferro_tensor::Tensor<f64> = todo!();
/// let device = NdArrayDevice::Cpu;
/// let burn_prim = tenferro_to_burn::<NdArray<f64>>(tenferro_t, &device);
/// ```
pub fn tenferro_to_burn<B: Backend>(
    _tensor: tenferro_tensor::Tensor<f64>,
    _device: &B::Device,
) -> FloatTensor<B> {
    todo!()
}
