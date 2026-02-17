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
//! use tenferro_tensor::Tensor;
//! use tenferro_mdarray::{mdarray_to_tensor, tensor_to_mdarray};
//!
//! // mdarray -> tenferro
//! let md: Array<f64, DynRank> = mdarray::tensor![1.0, 2.0, 3.0, 4.0];
//! let t: Tensor<f64> = mdarray_to_tensor(md);
//!
//! // tenferro -> mdarray
//! let t2: Tensor<f64> = Tensor::zeros(&[2, 3]);
//! let md2: Array<f64, DynRank> = tensor_to_mdarray(t2);
//! ```

use mdarray::{Array, DynRank};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

/// Converts an mdarray `Array<T, DynRank>` into a tenferro `Tensor<T>`.
///
/// This conversion copies all element data from the mdarray array into a
/// newly allocated tenferro tensor on the CPU. The shape is preserved.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
/// use tenferro_mdarray::mdarray_to_tensor;
///
/// let md: Array<f64, DynRank> = mdarray::tensor![1.0, 2.0, 3.0];
/// let t: Tensor<f64> = mdarray_to_tensor(md);
/// ```
pub fn mdarray_to_tensor<T: Scalar>(_array: Array<T, DynRank>) -> Tensor<T> {
    todo!()
}

/// Converts a tenferro `Tensor<T>` into an mdarray `Array<T, DynRank>`.
///
/// This conversion copies all element data from the tenferro tensor into a
/// newly allocated mdarray array. The shape is preserved.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
/// use tenferro_mdarray::tensor_to_mdarray;
///
/// let t: Tensor<f64> = Tensor::zeros(&[3, 4]);
/// let md: Array<f64, DynRank> = tensor_to_mdarray(t);
/// ```
pub fn tensor_to_mdarray<T: Scalar>(_tensor: Tensor<T>) -> Array<T, DynRank> {
    todo!()
}
