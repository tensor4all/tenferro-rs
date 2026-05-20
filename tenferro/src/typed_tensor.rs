//! Typed tensor operations.
//!
//! Operations in this module execute immediately on [`TypedTensor`] values
//! through an explicit backend.

pub use tenferro_tensor::TypedTensor;
use tenferro_tensor::{Result, TensorBackend, TensorScalar};

use crate::einsum_subscripts::to_einsum_subscripts;
use crate::EinsumSubscripts;

/// Execute an einsum immediately on borrowed typed tensors.
///
/// # Examples
///
/// ```
/// use tenferro::typed_tensor::{einsum, TypedTensor};
/// use tenferro::CpuBackend;
///
/// let mut backend = CpuBackend::new();
/// let a = TypedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TypedTensor::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let c = einsum(&mut backend, &[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.shape.as_slice(), &[2, 2]);
/// assert_eq!(c.host_data(), &[23.0, 34.0, 31.0, 46.0]);
/// ```
pub fn einsum<T: TensorScalar>(
    ctx: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>> {
    tenferro_einsum::typed_eager_einsum(ctx, inputs, subscripts)
}

/// Execute an einsum immediately on borrowed typed tensors using integer labels.
///
/// # Examples
///
/// ```
/// use tenferro::typed_tensor::{einsum_subscripts, TypedTensor};
/// use tenferro::{CpuBackend, EinsumSubscripts};
///
/// let mut backend = CpuBackend::new();
/// let a = TypedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
/// let b = TypedTensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);
/// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
/// let dot = einsum_subscripts(&mut backend, &[&a, &b], &subscripts).unwrap();
///
/// assert!(dot.shape.is_empty());
/// assert_eq!(dot.host_data(), &[32.0]);
/// ```
pub fn einsum_subscripts<T: TensorScalar>(
    ctx: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &EinsumSubscripts,
) -> Result<TypedTensor<T>> {
    let parsed = to_einsum_subscripts(subscripts);
    let tensors: Vec<tenferro_tensor::Tensor> = inputs
        .iter()
        .map(|tensor| T::into_tensor(tensor.shape.clone(), tensor.host_data().to_vec()))
        .collect();
    let refs: Vec<&tenferro_tensor::Tensor> = tensors.iter().collect();
    let result = tenferro_einsum::eager_einsum_subscripts(ctx, &refs, &parsed)?;
    let actual = result.dtype();
    T::try_into_typed(result).ok_or_else(|| tenferro_tensor::Error::DTypeMismatch {
        op: "typed_eager_einsum",
        lhs: actual,
        rhs: T::dtype(),
    })
}
