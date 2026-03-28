//! Bridge between [ndarray](https://docs.rs/ndarray) arrays and tenferro tensors.
//!
//! This crate provides typed canonical conversion helpers between generic
//! `ndarray::ArrayBase<S, D>` inputs and [`tenferro_tensor::Tensor<T>`], plus
//! export back to owned `ndarray::ArrayD<T>`. An optional `frontend` feature
//! adds a convenience conversion into `tenferro::Tensor`.
//!
//! The canonical bridge normalizes at the boundary:
//!
//! - `ndarray -> tenferro_tensor::Tensor<T>` first materializes a standard
//!   row-major ndarray layout, then converts into tenferro's internal
//!   column-major canonical tensor layout
//! - `tenferro_tensor::Tensor<T> -> ndarray` always materializes a standard
//!   row-major owned ndarray result
//!
//! This keeps the interop contract simple and avoids exposing a separate
//! zero-copy expert mode with layout-dependent reshape semantics.
//!
//! # Examples
//!
//! ```ignore
//! use ndarray::Array2;
//! use tenferro_ext_ndarray::{ndarray_to_tensor, tensor_to_ndarray};
//!
//! let array = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let tensor = ndarray_to_tensor(array);
//! let roundtrip = tensor_to_ndarray(tensor);
//! assert_eq!(roundtrip.shape(), &[2, 2]);
//! ```

use ndarray::{ArrayBase, ArrayD, Data, Dimension, IxDyn};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

#[cfg(test)]
mod tests;

#[cfg(feature = "frontend")]
mod frontend {
    use num_complex::{Complex32, Complex64};

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex32 {}
    impl Sealed for Complex64 {}
}

fn shape_error(err: ndarray::ShapeError) -> Error {
    Error::InvalidArgument(format!("ndarray layout conversion failed: {err}"))
}

fn ensure_main_memory(space: LogicalMemorySpace) -> Result<()> {
    if space != LogicalMemorySpace::MainMemory {
        return Err(Error::InvalidArgument(
            "tenferro-ext-ndarray currently supports CPU/main-memory tensors only".into(),
        ));
    }
    Ok(())
}

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

fn standard_row_major_owned<T, S, D>(array: ArrayBase<S, D>) -> ArrayD<T>
where
    T: Clone,
    S: Data<Elem = T>,
    D: Dimension,
{
    let array = array.into_dyn();
    if array.is_standard_layout() {
        array.into_owned()
    } else {
        array.as_standard_layout().to_owned()
    }
}

fn into_owned_data<T: Scalar>(tensor: Tensor<T>, context: &str) -> Result<Vec<T>> {
    tensor
        .try_into_data_vec()
        .ok_or_else(|| Error::InvalidArgument(context.into()))
}

/// Fallibly converts an ndarray array into a typed tenferro tensor.
///
/// This is the canonical interop entry point. The bridge first materializes a
/// standard row-major ndarray layout, then normalizes it into tenferro's
/// internal column-major tensor layout.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ext_ndarray::try_ndarray_to_tensor;
///
/// let array = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
/// let tensor = try_ndarray_to_tensor(array).unwrap();
/// assert_eq!(tensor.dims(), &[1, 2]);
/// ```
pub fn try_ndarray_to_tensor<T, S, D>(array: ArrayBase<S, D>) -> Result<Tensor<T>>
where
    T: Scalar + Clone,
    S: Data<Elem = T>,
    D: Dimension,
{
    let array = standard_row_major_owned(array);
    let dims = array.shape().to_vec();
    let (data, offset) = array.into_raw_vec_and_offset();
    let tensor = Tensor::from_vec(
        data,
        &dims,
        &row_major_strides(&dims),
        offset.unwrap_or(0) as isize,
    )?;
    Ok(tensor.into_contiguous(MemoryOrder::ColumnMajor))
}

/// Converts an owned ndarray array into a typed tenferro tensor, panicking on
/// conversion failure.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ext_ndarray::ndarray_to_tensor;
///
/// let array = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
/// let tensor = ndarray_to_tensor(array);
/// assert_eq!(tensor.dims(), &[1, 2]);
/// ```
pub fn ndarray_to_tensor<T, S, D>(array: ArrayBase<S, D>) -> Tensor<T>
where
    T: Scalar + Clone,
    S: Data<Elem = T>,
    D: Dimension,
{
    try_ndarray_to_tensor(array).unwrap_or_else(|err| panic!("{err}"))
}

/// Fallibly converts a typed tenferro tensor into an owned ndarray array.
///
/// The canonical bridge always materializes a standard row-major owned ndarray
/// result. This keeps row-major downstream integrations simple while tenferro
/// continues to use column-major canonical tensors internally.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_ext_ndarray::try_tensor_to_ndarray;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tensor = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor).unwrap();
/// let array = try_tensor_to_ndarray(tensor).unwrap();
/// assert_eq!(array.shape(), &[2, 2]);
/// ```
pub fn try_tensor_to_ndarray<T: Scalar>(tensor: Tensor<T>) -> Result<ArrayD<T>> {
    ensure_main_memory(tensor.logical_memory_space())?;

    let row_major = tensor.into_contiguous(MemoryOrder::RowMajor);
    let dims = row_major.dims().to_vec();
    let data = into_owned_data(
        row_major,
        "into_contiguous(RowMajor) must yield an owned CPU buffer for ndarray export",
    )?;
    ArrayD::from_shape_vec(IxDyn(&dims), data).map_err(shape_error)
}

/// Converts a typed tenferro tensor into an owned ndarray array, panicking on
/// conversion failure.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_ext_ndarray::tensor_to_ndarray;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tensor = Tensor::<f64>::zeros(&[2], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
/// let array = tensor_to_ndarray(tensor);
/// assert_eq!(array.shape(), &[2]);
/// ```
pub fn tensor_to_ndarray<T: Scalar>(tensor: Tensor<T>) -> ArrayD<T> {
    try_tensor_to_ndarray(tensor).unwrap_or_else(|err| panic!("{err}"))
}

/// Marker trait for scalar dtypes supported by the optional frontend helper.
///
/// This trait is sealed and cannot be implemented outside this crate.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ext_ndarray::try_ndarray_to_frontend;
///
/// let array = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
/// let tensor = try_ndarray_to_frontend(array).unwrap();
/// assert_eq!(tensor.dims(), &[1, 2]);
/// ```
#[cfg(feature = "frontend")]
pub trait FrontendScalar: Scalar + frontend::Sealed {
    fn into_frontend_tensor(tensor: Tensor<Self>) -> tenferro::Tensor;
}

#[cfg(feature = "frontend")]
impl FrontendScalar for f32 {
    fn into_frontend_tensor(tensor: Tensor<Self>) -> tenferro::Tensor {
        tenferro::Tensor::from_tensor(tensor)
    }
}

#[cfg(feature = "frontend")]
impl FrontendScalar for f64 {
    fn into_frontend_tensor(tensor: Tensor<Self>) -> tenferro::Tensor {
        tenferro::Tensor::from_tensor(tensor)
    }
}

#[cfg(feature = "frontend")]
impl FrontendScalar for num_complex::Complex32 {
    fn into_frontend_tensor(tensor: Tensor<Self>) -> tenferro::Tensor {
        tenferro::Tensor::from_tensor(tensor)
    }
}

#[cfg(feature = "frontend")]
impl FrontendScalar for num_complex::Complex64 {
    fn into_frontend_tensor(tensor: Tensor<Self>) -> tenferro::Tensor {
        tenferro::Tensor::from_tensor(tensor)
    }
}

/// Fallibly converts an owned ndarray array into the public `tenferro::Tensor`
/// frontend.
///
/// This helper is enabled by the `frontend` feature and is intentionally thin:
/// it converts through the canonical typed `ndarray -> tenferro_tensor::Tensor`
/// path first.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ext_ndarray::try_ndarray_to_frontend;
///
/// let array = Array2::from_shape_vec((1, 2), vec![1.0_f64, 2.0]).unwrap();
/// let tensor = try_ndarray_to_frontend(array).unwrap();
/// assert_eq!(tensor.dims(), &[1, 2]);
/// ```
#[cfg(feature = "frontend")]
pub fn try_ndarray_to_frontend<T, S, D>(array: ArrayBase<S, D>) -> Result<tenferro::Tensor>
where
    T: FrontendScalar + Clone,
    S: Data<Elem = T>,
    D: Dimension,
{
    let tensor = try_ndarray_to_tensor(array)?;
    Ok(T::into_frontend_tensor(tensor))
}
