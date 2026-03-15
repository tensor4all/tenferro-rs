mod basics;
mod complex;
mod eager_linalg;
mod eager_scalar;
mod eager_tensor;
mod layout;
mod merge;
mod promotion;
mod pullback;
mod scalar_ops;
mod snapshot;

use num_complex::{Complex32, Complex64};

pub use eager_linalg::{
    DynAdEigResult, DynAdEigenResult, DynAdLstsqResult, DynAdLuResult, DynAdQrResult,
    DynAdSlogdetResult, DynAdSvdResult,
};

/// Runtime AD tensor wrapper.
///
/// `DynAdTensor` is the canonical dynamic tensor payload for eager tensor
/// algebra in `tenferro-dyadtensor`.
///
/// - rank-0 `DynAdTensor` values act as scalar coefficients
/// - mixed-dtype tensor ops apply the dynamic result-type promotion rule
///   internally before execution
///   (`complex` beats `real`, and 64-bit beats 32-bit)
/// - [`DynAdTensor::to_scalar_type`] is the explicit numeric cast boundary
/// - [`DynAdTensor::detach`] drops tape metadata while keeping the same dynamic
///   tensor object for storage or FFI boundaries
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex64;
/// use tenferro_dyadtensor::{DynAdTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
/// let x = DynAdTensor::new_primal(t);
/// assert_eq!(x.scalar_type(), ScalarType::F64);
///
/// let coeff = DynAdTensor::new_primal(
///     Tensor::<Complex64>::from_slice(&[Complex64::new(0.0, 2.0)], &[], MemoryOrder::ColumnMajor)
///         .unwrap(),
/// );
/// let y = x.scale(&coeff).unwrap();
/// assert_eq!(y.scalar_type(), ScalarType::C64);
/// ```
#[derive(Clone)]
pub enum DynAdTensor {
    F32(crate::AdTensor<f32>),
    F64(crate::AdTensor<f64>),
    C32(crate::AdTensor<Complex32>),
    C64(crate::AdTensor<Complex64>),
}
