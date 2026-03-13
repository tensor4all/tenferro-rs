mod basics;
mod binary;
mod math;
mod traits;

use num_complex::{Complex32, Complex64};

/// Runtime AD scalar value wrapper.
///
/// Binary operator overloads (`+`, `-`, `*`, `/`) are fallible and return
/// [`crate::Result`] so mixed reverse-tape validation follows the checked API.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdValue, DynAdScalar};
///
/// let x: DynAdScalar = AdValue::forward(2.0_f64, 1.0_f64).into();
/// assert_eq!(x.mode(), AdMode::Forward);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DynAdScalar {
    F32(crate::AdValue<f32>),
    F64(crate::AdValue<f64>),
    C32(crate::AdValue<Complex32>),
    C64(crate::AdValue<Complex64>),
}

pub(crate) use binary::{promote_f32_to_c32, promote_f64_to_c64};
