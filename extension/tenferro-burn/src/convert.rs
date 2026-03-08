//! Conversion utilities between Burn tensor primitives and tenferro tensors.
//!
//! # Current Limitations
//!
//! These conversion functions currently only support `f64` element type.
//! Support for `f32` and other numeric types will be added in future versions.

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor as BurnTensor, TensorData, TensorPrimitive};
use tenferro_tensor::{MemoryOrder, Tensor as TfTensor};

fn row_major_strides(dims: &[usize]) -> Vec<isize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }

    let mut strides = vec![0isize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as isize;
    }
    strides
}

/// Convert a Burn backend tensor primitive into a tenferro `Tensor<f64>`.
///
/// # Current Limitations
///
/// This function currently supports only Burn backends whose float element
/// type is `f64`. Support for other element types (e.g., `f32`) will be added
/// in future versions.
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
pub fn burn_to_tenferro<B: Backend<FloatElem = f64>>(tensor: FloatTensor<B>) -> TfTensor<f64> {
    let data = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(tensor)).into_data();
    let dims = data.shape.clone();
    let values = data
        .into_vec::<f64>()
        .expect("burn_to_tenferro only supports f64 float tensors");

    TfTensor::from_vec(values, &dims, &row_major_strides(&dims), 0)
        .expect("Burn TensorData always provides a dense row-major layout")
}

/// Convert a tenferro `Tensor<f64>` into a Burn backend tensor primitive.
///
/// The `device` parameter specifies which Burn device the resulting tensor
/// should be placed on. For the `NdArray` backend this is typically
/// `NdArrayDevice::Cpu`, obtainable via `Default::default()`.
///
/// # Current Limitations
///
/// This function currently supports only Burn backends whose float element
/// type is `f64`. Support for other element types will be added in future
/// versions.
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
pub fn tenferro_to_burn<B: Backend<FloatElem = f64>>(
    tensor: TfTensor<f64>,
    device: &B::Device,
) -> FloatTensor<B> {
    let row_major = tensor.into_contiguous(MemoryOrder::RowMajor);
    let dims = row_major.dims().to_vec();
    let data = row_major
        .try_into_data_vec()
        .expect("into_contiguous returns a uniquely-owned CPU buffer");

    B::float_from_data(TensorData::new(data, dims), device)
}
