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

#[cfg(test)]
mod tests;

pub use convert::{burn_to_tenferro, tenferro_to_burn};

use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorMetadata, TensorPrimitive};

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
pub trait TensorNetworkOps: Backend<FloatElem = f64> {
    /// Perform an einsum contraction on raw backend tensor primitives.
    ///
    /// This operates at the primitive level. Prefer the high-level [`einsum`]
    /// function for typical usage.
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self>;
}

pub(crate) fn primitive_einsum<B: Backend<FloatElem = f64>>(
    subscripts: &str,
    inputs: Vec<FloatTensor<B>>,
) -> FloatTensor<B> {
    assert!(
        !inputs.is_empty(),
        "tenferro-burn::einsum requires at least one input tensor"
    );

    let device = B::float_device(inputs.first().unwrap());
    let tenferro_inputs: Vec<_> = inputs.iter().cloned().map(burn_to_tenferro::<B>).collect();
    let operand_refs: Vec<_> = tenferro_inputs.iter().collect();
    let mut ctx = CpuContext::new(1);
    let output = tenferro_einsum::einsum::<Standard<f64>, CpuBackend>(
        &mut ctx,
        subscripts,
        &operand_refs,
        None,
    )
    .expect("tenferro-burn::einsum received invalid subscripts or incompatible shapes");

    tenferro_to_burn::<B>(output, &device)
}

/// High-level einsum on Burn tensors, dispatching to the backend's
/// [`TensorNetworkOps::tn_einsum`] implementation.
///
/// The const rank `D` is shared by the input and output Burn tensors, so this
/// wrapper is only suitable for contractions whose output rank stays equal to
/// the input rank. Use [`TensorNetworkOps::tn_einsum`] directly for
/// rank-changing contractions.
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
    subscripts: &str,
    inputs: Vec<Tensor<B, D>>,
) -> Tensor<B, D> {
    let primitive_inputs: Vec<_> = inputs
        .into_iter()
        .map(|tensor| tensor.into_primitive().tensor())
        .collect();
    let output = B::tn_einsum(subscripts, primitive_inputs);

    assert_eq!(
        output.rank(),
        D,
        "tenferro-burn::einsum expected output rank {D}, got {}",
        output.rank()
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
