use core::ops::{Add, Div, Mul, Sub};

use num_complex::{Complex32, Complex64};
use num_traits::Float;

/// Scalar trait used by elementary AD rule helpers.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::ScalarAd;
///
/// fn takes_scalar<S: ScalarAd>(_x: S) {}
///
/// takes_scalar(1.0_f32);
/// takes_scalar(1.0_f64);
/// ```
pub trait ScalarAd:
    Copy + PartialEq + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// Real exponent type for `powf`.
    type Real: Copy + Float;

    /// Complex conjugate (identity for real scalars).
    fn conj(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Exponential.
    fn exp(self) -> Self;

    /// `exp(self) - 1`.
    fn expm1(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// `ln(1 + self)`.
    fn log1p(self) -> Self;

    /// Sine.
    fn sin(self) -> Self;

    /// Cosine.
    fn cos(self) -> Self;

    /// Hyperbolic tangent.
    fn tanh(self) -> Self;

    /// Power by real exponent.
    fn powf(self, exponent: Self::Real) -> Self;

    /// Power by integer exponent.
    fn powi(self, exponent: i32) -> Self;

    /// Convert real scalar to this scalar type.
    fn from_real(value: Self::Real) -> Self;

    /// Convert signed integer to this scalar type.
    fn from_i32(value: i32) -> Self;
}

macro_rules! impl_scalar_ad_real {
    ($ty:ty) => {
        impl ScalarAd for $ty {
            type Real = $ty;

            fn conj(self) -> Self {
                self
            }

            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }

            fn exp(self) -> Self {
                <$ty>::exp(self)
            }

            fn expm1(self) -> Self {
                <$ty>::exp_m1(self)
            }

            fn ln(self) -> Self {
                <$ty>::ln(self)
            }

            fn log1p(self) -> Self {
                <$ty>::ln_1p(self)
            }

            fn sin(self) -> Self {
                <$ty>::sin(self)
            }

            fn cos(self) -> Self {
                <$ty>::cos(self)
            }

            fn tanh(self) -> Self {
                <$ty>::tanh(self)
            }

            fn powf(self, exponent: Self::Real) -> Self {
                <$ty>::powf(self, exponent)
            }

            fn powi(self, exponent: i32) -> Self {
                <$ty>::powi(self, exponent)
            }

            fn from_real(value: Self::Real) -> Self {
                value
            }

            fn from_i32(value: i32) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! impl_scalar_ad_complex {
    ($complex_ty:ty, $real_ty:ty, $one:expr) => {
        impl ScalarAd for $complex_ty {
            type Real = $real_ty;

            fn conj(self) -> Self {
                <$complex_ty>::conj(&self)
            }

            fn sqrt(self) -> Self {
                <$complex_ty>::sqrt(self)
            }

            fn exp(self) -> Self {
                <$complex_ty>::exp(self)
            }

            fn expm1(self) -> Self {
                <$complex_ty>::exp(self) - $one
            }

            fn ln(self) -> Self {
                <$complex_ty>::ln(self)
            }

            fn log1p(self) -> Self {
                <$complex_ty>::ln(self + $one)
            }

            fn sin(self) -> Self {
                <$complex_ty>::sin(self)
            }

            fn cos(self) -> Self {
                <$complex_ty>::cos(self)
            }

            fn tanh(self) -> Self {
                <$complex_ty>::tanh(self)
            }

            fn powf(self, exponent: Self::Real) -> Self {
                <$complex_ty>::powf(self, exponent)
            }

            fn powi(self, exponent: i32) -> Self {
                <$complex_ty>::powi(&self, exponent)
            }

            fn from_real(value: Self::Real) -> Self {
                <$complex_ty>::new(value, 0.0)
            }

            fn from_i32(value: i32) -> Self {
                <$complex_ty>::new(value as $real_ty, 0.0)
            }
        }
    };
}

impl_scalar_ad_real!(f32);
impl_scalar_ad_real!(f64);
impl_scalar_ad_complex!(Complex32, f32, Complex32::new(1.0, 0.0));
impl_scalar_ad_complex!(Complex64, f64, Complex64::new(1.0, 0.0));

/// PyTorch-style real-input / complex-gradient projection helper (`handle_r_to_c`).
///
/// This is equivalent to taking the real part when a gradient for real input
/// becomes complex during intermediate algebra.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::handle_r_to_c_f64;
/// use num_complex::Complex64;
///
/// let g = Complex64::new(1.25, -3.0);
/// assert_eq!(handle_r_to_c_f64(g), 1.25);
/// ```
pub fn handle_r_to_c_f64(gradient: Complex64) -> f64 {
    gradient.re
}

/// `f32` variant of [`handle_r_to_c_f64`].
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::handle_r_to_c_f32;
/// use num_complex::Complex32;
///
/// let g = Complex32::new(2.0, 4.0);
/// assert_eq!(handle_r_to_c_f32(g), 2.0);
/// ```
pub fn handle_r_to_c_f32(gradient: Complex32) -> f32 {
    gradient.re
}
