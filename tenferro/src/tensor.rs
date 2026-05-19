//! Concrete tensor operations.
//!
//! Operations in this module execute immediately on [`Tensor`] values through
//! an explicit backend.

pub use tenferro_tensor::Tensor;
use tenferro_tensor::{Result, TensorBackend};

/// Execute an einsum immediately on borrowed concrete tensors.
///
/// # Examples
///
/// ```
/// use tenferro::tensor::einsum;
/// use tenferro::{CpuBackend, Tensor};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
/// let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);
/// let dot = einsum(&mut backend, &[&a, &b], "i,i->").unwrap();
///
/// assert_eq!(dot.as_slice::<f64>().unwrap(), &[32.0]);
/// ```
pub fn einsum(
    ctx: &mut impl TensorBackend,
    inputs: &[&Tensor],
    subscripts: &str,
) -> Result<Tensor> {
    tenferro_einsum::eager_einsum(ctx, inputs, subscripts)
}

/// Execute an einsum immediately, allowing owned inputs to be consumed.
///
/// # Examples
///
/// ```
/// use tenferro::tensor::einsum_owned;
/// use tenferro::{CpuBackend, Tensor};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
/// let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);
/// let dot = einsum_owned(&mut backend, vec![a, b], "i,i->").unwrap();
///
/// assert_eq!(dot.as_slice::<f64>().unwrap(), &[32.0]);
/// ```
pub fn einsum_owned(
    ctx: &mut impl TensorBackend,
    inputs: Vec<Tensor>,
    subscripts: &str,
) -> Result<Tensor> {
    tenferro_einsum::eager_einsum_owned(ctx, inputs, subscripts)
}
