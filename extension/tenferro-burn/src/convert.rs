//! Conversion utilities between Burn tensor primitives and tenferro tensors.

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// Convert a Burn backend tensor primitive into a tenferro `Tensor<f64>`.
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
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use tenferro_burn::convert::tenferro_to_burn;
///
/// let tenferro_t: tenferro_tensor::Tensor<f64> = todo!();
/// let burn_prim = tenferro_to_burn::<NdArray<f64>>(tenferro_t, &Default::default());
/// ```
pub fn tenferro_to_burn<B: Backend>(
    _tensor: tenferro_tensor::Tensor<f64>,
    _device: &B::Device,
) -> FloatTensor<B> {
    todo!()
}
