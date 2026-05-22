//! Eager tensor operations.
//!
//! This module hosts free functions whose receiver is naturally n-ary. Unary
//! and binary eager operations remain methods on [`EagerTensor`].

use crate::error::Result;
use crate::EinsumSubscripts;

pub use crate::eager::{EagerRuntime, EagerTensor};

/// Execute an einsum eagerly and record it when any input requires gradients.
///
/// # Examples
///
/// ```
/// use tenferro::eager_tensor::einsum;
/// use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
///     vec![2, 3],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx.clone());
/// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
///     vec![3, 2],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx);
///
/// let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.data().shape(), &[2, 2]);
/// assert_eq!(c.data().as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
    crate::eager_einsum::einsum(inputs, subscripts)
}

/// Execute an einsum eagerly from integer labels and record it when any input requires gradients.
///
/// # Examples
///
/// ```
/// use tenferro::eager_tensor::{einsum_subscripts, EagerRuntime, EagerTensor};
/// use tenferro::{CpuBackend, EinsumSubscripts, Tensor};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
/// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx);
/// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
/// let dot = einsum_subscripts(&[&a, &b], &subscripts).unwrap();
///
/// assert_eq!(dot.data().as_slice::<f64>().unwrap(), &[32.0]);
/// ```
pub fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<EagerTensor> {
    crate::eager_einsum::einsum_subscripts(inputs, subscripts)
}
