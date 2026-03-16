//! Bridge between the [Burn](https://burn.dev) deep learning framework and
//! tenferro tensor network operations.
//!
//! This crate allows Burn tensors to be used with tenferro's einsum and
//! tensor network contraction routines, enabling seamless integration of
//! tensor network methods into Burn-based deep learning pipelines.
//!
//! Burn tensors are treated as row-major boundary values. The bridge
//! normalizes them into tenferro's internal column-major canonical layout for
//! computation, then materializes row-major buffers again when exporting back
//! to Burn.
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
use thiserror::Error as ThisError;

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Tensor, TensorMetadata, TensorPrimitive};

/// Error type for Burn/tenferro bridge failures.
///
/// # Examples
///
/// ```
/// use tenferro_burn::Error;
///
/// let err = Error::InternalInvariant("example");
/// assert!(err.to_string().contains("example"));
/// ```
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("invalid tenferro-burn argument: {0}")]
    InvalidArgument(String),
    #[error("tenferro-burn internal invariant violated: {0}")]
    InternalInvariant(&'static str),
}

/// Result type for Burn/tenferro bridge operations.
///
/// # Examples
///
/// ```
/// use tenferro_burn::{Error, Result};
///
/// let result: Result<()> = Err(Error::InvalidArgument("bad einsum".into()));
/// assert!(result.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn panic_on_error<T>(result: Result<T>) -> T {
    result.unwrap_or_else(|err| panic!("{err}"))
}

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

pub(crate) fn try_primitive_einsum<B: Backend<FloatElem = f64>>(
    subscripts: &str,
    inputs: Vec<FloatTensor<B>>,
) -> Result<FloatTensor<B>> {
    let first = inputs.first().ok_or_else(|| {
        Error::InvalidArgument("tenferro-burn::einsum requires at least one input tensor".into())
    })?;
    let device = B::float_device(first);
    let tenferro_inputs: Vec<_> = inputs
        .iter()
        .cloned()
        .map(convert::try_burn_to_tenferro::<B>)
        .collect::<Result<_>>()?;
    let operand_refs: Vec<_> = tenferro_inputs.iter().collect();
    let mut ctx = CpuContext::new(1);
    let output = tenferro_einsum::einsum::<Standard<f64>, CpuBackend>(
        &mut ctx,
        subscripts,
        &operand_refs,
        None,
    )
    .map_err(|err| Error::InvalidArgument(err.to_string()))?;

    convert::try_tenferro_to_burn::<B>(output, &device)
}

pub(crate) fn primitive_einsum<B: Backend<FloatElem = f64>>(
    subscripts: &str,
    inputs: Vec<FloatTensor<B>>,
) -> FloatTensor<B> {
    panic_on_error(try_primitive_einsum::<B>(subscripts, inputs))
}

/// Fallible high-level einsum on Burn tensors, dispatching to the backend's
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
/// use tenferro_burn::try_einsum;
///
/// let a: Tensor<NdArray<f64>, 2> = Tensor::ones([3, 4], &Default::default());
/// let b: Tensor<NdArray<f64>, 2> = Tensor::ones([4, 5], &Default::default());
/// let c: Tensor<NdArray<f64>, 2> = try_einsum("ij,jk->ik", vec![a, b]).unwrap();
/// ```
pub fn try_einsum<B: TensorNetworkOps, const D: usize>(
    subscripts: &str,
    inputs: Vec<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let primitive_inputs: Vec<_> = inputs
        .into_iter()
        .map(|tensor| tensor.into_primitive().tensor())
        .collect();
    let output = B::tn_einsum(subscripts, primitive_inputs);

    if output.rank() != D {
        return Err(Error::InvalidArgument(format!(
            "tenferro-burn::einsum expected output rank {D}, got {}",
            output.rank()
        )));
    }

    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

/// High-level infallible einsum convenience wrapper.
///
/// # Examples
///
/// ```ignore
/// use burn::backend::NdArray;
/// use burn::tensor::Tensor;
/// use tenferro_burn::einsum;
///
/// let a: Tensor<NdArray<f64>, 2> = Tensor::ones([2, 2], &Default::default());
/// let b: Tensor<NdArray<f64>, 2> = Tensor::ones([2, 2], &Default::default());
/// let c = einsum("ij,jk->ik", vec![a, b]);
/// assert_eq!(c.dims(), [2, 2]);
/// ```
pub fn einsum<B: TensorNetworkOps, const D: usize>(
    subscripts: &str,
    inputs: Vec<Tensor<B, D>>,
) -> Tensor<B, D> {
    panic_on_error(try_einsum(subscripts, inputs))
}
