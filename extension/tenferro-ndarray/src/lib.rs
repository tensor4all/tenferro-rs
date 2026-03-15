//! Bridge between [ndarray](https://docs.rs/ndarray) arrays and tenferro tensors.
//!
//! This crate provides typed canonical conversion helpers between generic
//! `ndarray::ArrayBase<S, D>` inputs and [`tenferro_tensor::Tensor<T>`], plus
//! export back to owned `ndarray::ArrayD<T>`. An optional `frontend` feature
//! adds a convenience conversion into `tenferro::Tensor`.
//!
//! # Examples
//!
//! ```ignore
//! use ndarray::Array2;
//! use tenferro_ndarray::{ndarray_to_tensor, tensor_to_ndarray};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let array = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let tensor = ndarray_to_tensor(array);
//! let roundtrip = tensor_to_ndarray(tensor);
//! assert_eq!(roundtrip.shape(), &[2, 2]);
//! ```

use ndarray::{ArrayBase, ArrayD, Data, Dimension, IxDyn, ShapeBuilder};
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
            "tenferro-ndarray currently supports CPU/main-memory tensors only".into(),
        ));
    }
    Ok(())
}

fn usize_strides(strides: &[isize]) -> Result<Vec<usize>> {
    strides
        .iter()
        .map(|&stride| {
            usize::try_from(stride).map_err(|_| {
                Error::InvalidArgument(format!(
                    "negative ndarray/tenferro stride {stride} is not supported by the bridge"
                ))
            })
        })
        .collect()
}

fn can_zero_copy_tensor_to_ndarray<T: Scalar>(tensor: &Tensor<T>) -> bool {
    tensor.logical_memory_space() == LogicalMemorySpace::MainMemory
        && !tensor.is_conjugated()
        && tensor.offset() == 0
        && tensor.buffer().is_owned()
        && tensor.buffer().is_unique()
        && tensor
            .strides()
            .iter()
            .all(|&stride| stride > 0 || tensor.len() <= 1)
}

fn into_owned_data<T: Scalar>(tensor: Tensor<T>, context: &str) -> Result<Vec<T>> {
    tensor
        .try_into_data_vec()
        .ok_or_else(|| Error::InvalidArgument(context.into()))
}

/// Fallibly converts an ndarray array into a typed tenferro tensor.
///
/// This is the canonical interop entry point. The bridge preserves shape,
/// strides, and offset. Owned ndarray inputs use best-effort zero-copy by
/// moving CPU storage into the target tensor; borrowed or shared inputs fall
/// back to `into_owned()`.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ndarray::try_ndarray_to_tensor;
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
    let array = array.into_dyn().into_owned();
    let dims = array.shape().to_vec();
    let strides = array.strides().to_vec();
    let (data, offset) = array.into_raw_vec_and_offset();
    Tensor::from_vec(data, &dims, &strides, offset.unwrap_or(0) as isize)
}

/// Converts an owned ndarray array into a typed tenferro tensor, panicking on
/// conversion failure.
///
/// # Examples
///
/// ```ignore
/// use ndarray::Array2;
/// use tenferro_ndarray::ndarray_to_tensor;
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
/// The bridge attempts zero-copy for unique owned CPU buffers whose layout can
/// be expressed by ndarray. Otherwise it materializes a row-major CPU copy.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_ndarray::try_tensor_to_ndarray;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tensor = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
/// let array = try_tensor_to_ndarray(tensor).unwrap();
/// assert_eq!(array.shape(), &[2, 2]);
/// ```
pub fn try_tensor_to_ndarray<T: Scalar>(tensor: Tensor<T>) -> Result<ArrayD<T>> {
    ensure_main_memory(tensor.logical_memory_space())?;

    if can_zero_copy_tensor_to_ndarray(&tensor) {
        let dims = tensor.dims().to_vec();
        let strides = usize_strides(tensor.strides())?;
        let data = into_owned_data(
            tensor,
            "expected unique owned CPU tensor buffer for zero-copy ndarray export",
        )?;
        return ArrayD::from_shape_vec(IxDyn(&dims).strides(IxDyn(&strides)), data)
            .map_err(shape_error);
    }

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
/// use tenferro_ndarray::tensor_to_ndarray;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tensor = Tensor::<f64>::zeros(&[2], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
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
/// use tenferro_ndarray::try_ndarray_to_frontend;
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
/// use tenferro_ndarray::try_ndarray_to_frontend;
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
