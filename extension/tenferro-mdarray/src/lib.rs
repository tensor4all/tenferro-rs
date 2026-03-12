//! Bridge between [mdarray](https://docs.rs/mdarray) multidimensional arrays
//! and [tenferro](https://docs.rs/tenferro-tensor) tensors.
//!
//! This crate provides conversion functions between mdarray's
//! `Array<T, DynRank>` and tenferro's `Tensor<T>`, enabling convenient data
//! exchange between the two ecosystems.
//!
//! Due to Rust's orphan rules, [`From`]/[`Into`] trait impls cannot be provided
//! for two external types. Instead, use the standalone conversion functions
//! [`mdarray_to_tensor`] and [`tensor_to_mdarray`].
//!
//! **Zero-copy is a non-goal.** Both conversion directions copy element data.
//! The purpose of this crate is ergonomic interoperability, not performance-
//! critical data sharing.
//!
//! # Examples
//!
//! ```ignore
//! use mdarray::{Array, DynRank};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_mdarray::{mdarray_to_tensor, tensor_to_mdarray};
//!
//! // mdarray -> tenferro
//! let md: Array<f64, DynRank> = mdarray::tensor![1.0, 2.0, 3.0, 4.0].into_dyn();
//! let t: Tensor<f64> = mdarray_to_tensor(md);
//!
//! // tenferro -> mdarray
//! let t2: Tensor<f64> = Tensor::zeros(
//!     &[2, 3],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! );
//! let md2: Array<f64, DynRank> = tensor_to_mdarray(t2);
//! ```

use mdarray::{Array, DynRank};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

#[cfg(test)]
mod tests;

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

/// Fallibly converts an mdarray `Array<T, DynRank>` into a tenferro `Tensor<T>`.
///
/// This conversion copies all element data from the mdarray array into a
/// newly allocated tenferro tensor on the CPU. The shape is preserved.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
/// use tenferro_mdarray::try_mdarray_to_tensor;
///
/// let md: Array<f64, DynRank> = mdarray::tensor![1.0, 2.0, 3.0].into_dyn();
/// let t: Tensor<f64> = try_mdarray_to_tensor(md).unwrap();
/// ```
pub fn try_mdarray_to_tensor<T: Scalar>(array: Array<T, DynRank>) -> Result<Tensor<T>> {
    let dims = array.dims().to_vec();
    let strides = row_major_strides(&dims);
    Tensor::from_vec(array.into_vec(), &dims, &strides, 0)
}

/// Converts an mdarray `Array<T, DynRank>` into a tenferro `Tensor<T>`,
/// panicking if conversion fails.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_mdarray::mdarray_to_tensor;
///
/// let md: Array<f64, DynRank> = mdarray::tensor![1.0, 2.0, 3.0].into_dyn();
/// let t = mdarray_to_tensor(md);
/// assert_eq!(t.dims(), &[3]);
/// ```
pub fn mdarray_to_tensor<T: Scalar>(array: Array<T, DynRank>) -> Tensor<T> {
    try_mdarray_to_tensor(array).unwrap_or_else(|err| panic!("{err}"))
}

/// Fallibly converts a tenferro `Tensor<T>` into an mdarray `Array<T, DynRank>`.
///
/// This conversion copies all element data from the tenferro tensor into a
/// newly allocated mdarray array. The shape is preserved.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
/// use tenferro_mdarray::try_tensor_to_mdarray;
///
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::MemoryOrder;
///
/// let t: Tensor<f64> = Tensor::zeros(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let md: Array<f64, DynRank> = try_tensor_to_mdarray(t).unwrap();
/// ```
pub fn try_tensor_to_mdarray<T: Scalar>(tensor: Tensor<T>) -> Result<Array<T, DynRank>> {
    let row_major = tensor.into_contiguous(MemoryOrder::RowMajor);
    let dims = row_major.dims().to_vec();
    let data = row_major.try_into_data_vec().ok_or(Error::InvalidArgument(
        "into_contiguous must return an owned CPU buffer".into(),
    ))?;
    Ok(Array::from(data).into_shape(dims.as_slice()).into_dyn())
}

/// Converts a tenferro `Tensor<T>` into an mdarray `Array<T, DynRank>`,
/// panicking if conversion fails.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_mdarray::tensor_to_mdarray;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t: Tensor<f64> = Tensor::zeros(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let md = tensor_to_mdarray(t);
/// assert_eq!(md.shape().ndim(), 2);
/// ```
pub fn tensor_to_mdarray<T: Scalar>(tensor: Tensor<T>) -> Array<T, DynRank> {
    try_tensor_to_mdarray(tensor).unwrap_or_else(|err| panic!("{err}"))
}
