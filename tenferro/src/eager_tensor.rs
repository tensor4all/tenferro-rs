//! Eager tensor operations.
//!
//! This module hosts free functions whose receiver is naturally n-ary. Unary
//! and binary eager operations remain methods on [`EagerTensor`].

use tenferro_tensor::TensorBackend;

use crate::error::Result;

pub use crate::eager::{EagerContext, EagerTensor};

/// Execute an einsum eagerly and record it when any input requires gradients.
///
/// # Examples
///
/// ```
/// use tenferro::eager_tensor::einsum;
/// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
///
/// let ctx = EagerContext::with_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(Tensor::from_vec(
///     vec![2, 3],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx.clone());
/// let b = EagerTensor::from_tensor_in(Tensor::from_vec(
///     vec![3, 2],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx);
///
/// let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.data().shape(), &[2, 2]);
/// assert_eq!(c.data().as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn einsum<B: TensorBackend>(
    inputs: &[&EagerTensor<B>],
    subscripts: &str,
) -> Result<EagerTensor<B>> {
    crate::eager_einsum::einsum(inputs, subscripts)
}
