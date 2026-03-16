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

use crate::{panic_on_error, Error, Result};

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

/// Fallibly convert a Burn backend tensor primitive into a tenferro
/// `Tensor<f64>`.
///
/// Burn tensors are treated as row-major boundary values. The canonical bridge
/// normalizes them into tenferro's internal column-major tensor layout before
/// returning.
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
/// use tenferro_burn::convert::try_burn_to_tenferro;
///
/// let burn_prim: <NdArray<f64> as burn::tensor::backend::Backend>::FloatTensorPrimitive =
///     todo!();
/// let tenferro_t = try_burn_to_tenferro::<NdArray<f64>>(burn_prim).unwrap();
/// ```
pub fn try_burn_to_tenferro<B: Backend<FloatElem = f64>>(
    tensor: FloatTensor<B>,
) -> Result<TfTensor<f64>> {
    let data = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(tensor)).into_data();
    let dims = data.shape.clone();
    let values = data.into_vec::<f64>().map_err(|_| {
        Error::InvalidArgument("burn_to_tenferro only supports f64 float tensors".into())
    })?;

    let tensor =
        TfTensor::from_vec(values, &dims, &row_major_strides(&dims), 0).map_err(|err| {
            Error::InvalidArgument(format!("Burn TensorData must be dense row-major: {err}"))
        })?;
    Ok(tensor.into_contiguous(MemoryOrder::ColumnMajor))
}

/// Convert a Burn backend tensor primitive into a tenferro `Tensor<f64>`,
/// panicking if conversion fails.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use tenferro_burn::convert::burn_to_tenferro;
///
/// let burn_prim: <NdArray<f64> as burn::tensor::backend::Backend>::FloatTensorPrimitive =
///     todo!();
/// let tenferro_t = burn_to_tenferro::<NdArray<f64>>(burn_prim);
/// assert_eq!(tenferro_t.dims().len(), 1);
/// ```
pub fn burn_to_tenferro<B: Backend<FloatElem = f64>>(tensor: FloatTensor<B>) -> TfTensor<f64> {
    panic_on_error(try_burn_to_tenferro::<B>(tensor))
}

/// Convert a tenferro `Tensor<f64>` into a Burn backend tensor primitive.
///
/// The `device` parameter specifies which Burn device the resulting tensor
/// should be placed on. For the `NdArray` backend this is typically
/// `NdArrayDevice::Cpu`, obtainable via `Default::default()`.
/// The bridge always materializes a row-major owned buffer at this boundary.
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
/// use tenferro_burn::convert::try_tenferro_to_burn;
///
/// let tenferro_t: tenferro_tensor::Tensor<f64> = todo!();
/// let device = NdArrayDevice::Cpu;
/// let burn_prim = try_tenferro_to_burn::<NdArray<f64>>(tenferro_t, &device).unwrap();
/// ```
pub fn try_tenferro_to_burn<B: Backend<FloatElem = f64>>(
    tensor: TfTensor<f64>,
    device: &B::Device,
) -> Result<FloatTensor<B>> {
    let row_major = tensor.into_contiguous(MemoryOrder::RowMajor);
    let dims = row_major.dims().to_vec();
    let data = row_major
        .try_into_data_vec()
        .ok_or(Error::InternalInvariant(
            "into_contiguous must return a uniquely-owned CPU buffer",
        ))?;

    Ok(B::float_from_data(TensorData::new(data, dims), device))
}

/// Convert a tenferro `Tensor<f64>` into a Burn backend tensor primitive,
/// panicking if conversion fails.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use tenferro_burn::convert::tenferro_to_burn;
///
/// let device = Default::default();
/// let tenferro_t: tenferro_tensor::Tensor<f64> = todo!();
/// let _burn_prim = tenferro_to_burn::<NdArray<f64>>(tenferro_t, &device);
/// ```
pub fn tenferro_to_burn<B: Backend<FloatElem = f64>>(
    tensor: TfTensor<f64>,
    device: &B::Device,
) -> FloatTensor<B> {
    panic_on_error(try_tenferro_to_burn::<B>(tensor, device))
}
