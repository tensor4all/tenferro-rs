//! Bridge between the [Burn](https://burn.dev) deep learning framework and
//! tenferro tensor network operations.
//!
//! This crate allows Burn tensors to be used with tenferro's einsum and
//! tensor network contraction routines, enabling seamless integration of
//! tensor network methods into Burn-based deep learning pipelines.
//!
//! # Examples
//!
//! ```ignore
//! use burn::backend::NdArray;
//! use burn::tensor::Tensor;
//! use tenferro_burn::einsum;
//!
//! // Matrix multiplication via einsum
//! let a: Tensor<NdArray<f64>, 2> = Tensor::ones([3, 4], &Default::default());
//! let b: Tensor<NdArray<f64>, 2> = Tensor::ones([4, 5], &Default::default());
//! let c: Tensor<NdArray<f64>, 2> = einsum("ij,jk->ik", vec![a, b]);
//! ```

pub mod backward;
pub mod convert;
pub mod forward;

pub use convert::{burn_to_tenferro, tenferro_to_burn};

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::Tensor;

/// Trait for backends that support tenferro tensor network operations.
///
/// Implement this trait for a Burn backend to enable `einsum` and other
/// tensor network primitives on that backend's tensors.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use tenferro_burn::TensorNetworkOps;
///
/// // NdArray<f64> implements TensorNetworkOps
/// let result = <NdArray<f64> as TensorNetworkOps>::tn_einsum(
///     "ij,jk->ik",
///     vec![a_primitive, b_primitive],
/// );
/// ```
pub trait TensorNetworkOps: Backend {
    /// Perform an einsum contraction on raw backend tensor primitives.
    ///
    /// This operates at the primitive level. Prefer the high-level [`einsum`]
    /// function for typical usage.
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self>;
}

/// High-level einsum on Burn tensors, dispatching to the backend's
/// [`TensorNetworkOps::tn_einsum`] implementation.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use burn::tensor::Tensor;
/// use tenferro_burn::einsum;
///
/// let a: Tensor<NdArray<f64>, 2> = Tensor::ones([3, 4], &Default::default());
/// let b: Tensor<NdArray<f64>, 2> = Tensor::ones([4, 5], &Default::default());
/// let c: Tensor<NdArray<f64>, 2> = einsum("ij,jk->ik", vec![a, b]);
/// ```
pub fn einsum<B: TensorNetworkOps, const D: usize>(
    _subscripts: &str,
    _inputs: Vec<Tensor<B, D>>,
) -> Tensor<B, D> {
    todo!()
}
