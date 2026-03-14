mod basics;
mod complex;
mod layout;
mod merge;
mod promotion;
mod scalar_ops;
mod snapshot;

use num_complex::{Complex32, Complex64};

/// Runtime AD tensor wrapper.
///
/// `DynAdTensor` is the canonical dynamic tensor payload for eager tensor
/// algebra in `tenferro-dyadtensor`.
///
/// - rank-0 `DynAdTensor` values act as scalar coefficients
/// - [`DynAdTensor::promote_to`] performs the supported AD-aware dtype lifts
/// - [`DynAdTensor::primal_snapshot`] exposes a primal-only structured payload
///   for storage or FFI boundaries
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
