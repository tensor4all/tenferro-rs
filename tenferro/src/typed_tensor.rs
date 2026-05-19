//! Typed tensor operations.
//!
//! Operations in this module execute immediately on [`TypedTensor`] values
//! through an explicit backend.

pub use tenferro_tensor::TypedTensor;
use tenferro_tensor::{Result, TensorBackend, TensorScalar};

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
