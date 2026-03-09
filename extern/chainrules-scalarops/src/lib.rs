//! Scalar AD helper rules for elementary operations.
//!
//! This crate provides stateless primal/frule/rrule helpers for scalar ops
//! used by wrapper-level AD APIs.
//!
//! Supported operations:
//!
//! - `add`, `sub`, `mul`, `div`
//! - `conj`
//! - `sqrt`
//! - `powf` (fixed real exponent)
//! - `powi` (fixed integer exponent)
//!
//! # Examples
//!
//! ```rust
//! use chainrules_scalarops::{powf_frule, powf_rrule};
//!
//! // y = x^3 at x=2, dx=1
//! let (y, dy) = powf_frule(2.0_f64, 3.0, 1.0);
//! assert_eq!(y, 8.0);
//! assert_eq!(dy, 12.0);
//!
//! // Reverse-mode: dL/dx = dL/dy * 3*x^2
//! let dx = powf_rrule(2.0_f64, 3.0, 1.0);
//! assert_eq!(dx, 12.0);
//! ```

use core::ops::{Add, Div, Mul, Sub};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, One, Zero};

/// Scalar trait used by elementary AD rule helpers.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::ScalarAd;
///
/// fn takes_scalar<S: ScalarAd>(_x: S) {}
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

    /// Power by real exponent.
    fn powf(self, exponent: Self::Real) -> Self;

    /// Power by integer exponent.
    fn powi(self, exponent: i32) -> Self;

    /// Convert real scalar to this scalar type.
    fn from_real(value: Self::Real) -> Self;

    /// Convert signed integer to this scalar type.
    fn from_i32(value: i32) -> Self;
}

impl ScalarAd for f32 {
    type Real = f32;

    fn conj(self) -> Self {
        self
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn powf(self, exponent: Self::Real) -> Self {
        f32::powf(self, exponent)
    }

    fn powi(self, exponent: i32) -> Self {
        f32::powi(self, exponent)
    }

    fn from_real(value: Self::Real) -> Self {
        value
    }

    fn from_i32(value: i32) -> Self {
        value as f32
    }
}

impl ScalarAd for f64 {
    type Real = f64;

    fn conj(self) -> Self {
        self
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn powf(self, exponent: Self::Real) -> Self {
        f64::powf(self, exponent)
    }

    fn powi(self, exponent: i32) -> Self {
        f64::powi(self, exponent)
    }

    fn from_real(value: Self::Real) -> Self {
        value
    }

    fn from_i32(value: i32) -> Self {
        value as f64
    }
}

impl ScalarAd for Complex32 {
    type Real = f32;

    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    fn sqrt(self) -> Self {
        Complex32::sqrt(self)
    }

    fn powf(self, exponent: Self::Real) -> Self {
        Complex32::powf(self, exponent)
    }

    fn powi(self, exponent: i32) -> Self {
        Complex32::powi(&self, exponent)
    }

    fn from_real(value: Self::Real) -> Self {
        Complex32::new(value, 0.0)
    }

    fn from_i32(value: i32) -> Self {
        Complex32::new(value as f32, 0.0)
    }
}

impl ScalarAd for Complex64 {
    type Real = f64;

    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    fn sqrt(self) -> Self {
        Complex64::sqrt(self)
    }

    fn powf(self, exponent: Self::Real) -> Self {
        Complex64::powf(self, exponent)
    }

    fn powi(self, exponent: i32) -> Self {
        Complex64::powi(&self, exponent)
    }

    fn from_real(value: Self::Real) -> Self {
        Complex64::new(value, 0.0)
    }

    fn from_i32(value: i32) -> Self {
        Complex64::new(value as f64, 0.0)
    }
}

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

/// Primal `add`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::add;
///
/// assert_eq!(add(1.5_f64, 2.0_f64), 3.5_f64);
/// ```
pub fn add<S: ScalarAd>(x: S, y: S) -> S {
    x + y
}

/// Forward rule for `add`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::add_frule;
///
/// let (y, dy) = add_frule(2.0_f64, 3.0_f64, 0.1_f64, 0.2_f64);
/// assert_eq!(y, 5.0_f64);
/// assert!((dy - 0.3_f64).abs() < 1e-12);
/// ```
pub fn add_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    (x + y, dx + dy)
}

/// Reverse rule for `add`.
///
/// Returns cotangents with respect to `(x, y)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::add_rrule;
///
/// let (dx, dy) = add_rrule(1.25_f64);
/// assert_eq!(dx, 1.25_f64);
/// assert_eq!(dy, 1.25_f64);
/// ```
pub fn add_rrule<S: ScalarAd>(cotangent: S) -> (S, S) {
    (cotangent, cotangent)
}

/// Primal `sub`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sub;
///
/// assert_eq!(sub(5.0_f64, 2.0_f64), 3.0_f64);
/// ```
pub fn sub<S: ScalarAd>(x: S, y: S) -> S {
    x - y
}

/// Forward rule for `sub`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sub_frule;
///
/// let (y, dy) = sub_frule(5.0_f64, 2.0_f64, 0.3_f64, 0.1_f64);
/// assert_eq!(y, 3.0_f64);
/// assert!((dy - 0.2_f64).abs() < 1e-12);
/// ```
pub fn sub_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    (x - y, dx - dy)
}

/// Reverse rule for `sub`.
///
/// Returns cotangents with respect to `(x, y)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sub_rrule;
///
/// let (dx, dy) = sub_rrule(2.0_f64);
/// assert_eq!(dx, 2.0_f64);
/// assert_eq!(dy, -2.0_f64);
/// ```
pub fn sub_rrule<S: ScalarAd>(cotangent: S) -> (S, S) {
    (cotangent, S::from_i32(-1) * cotangent)
}

/// Primal `mul`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::mul;
///
/// assert_eq!(mul(2.0_f64, 4.0_f64), 8.0_f64);
/// ```
pub fn mul<S: ScalarAd>(x: S, y: S) -> S {
    x * y
}

/// Forward rule for `mul`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::mul_frule;
///
/// let (y, dy) = mul_frule(2.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
/// assert_eq!(y, 8.0_f64);
/// assert_eq!(dy, 2.5_f64);
/// ```
pub fn mul_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    let primal = x * y;
    let tangent = dx * y.conj() + dy * x.conj();
    (primal, tangent)
}

/// Reverse rule for `mul`.
///
/// Returns cotangents with respect to `(x, y)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::mul_rrule;
///
/// let (dx, dy) = mul_rrule(2.0_f64, 4.0_f64, 1.0_f64);
/// assert_eq!(dx, 4.0_f64);
/// assert_eq!(dy, 2.0_f64);
/// ```
pub fn mul_rrule<S: ScalarAd>(x: S, y: S, cotangent: S) -> (S, S) {
    (cotangent * y.conj(), cotangent * x.conj())
}

/// Primal `div`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::div;
///
/// assert_eq!(div(8.0_f64, 2.0_f64), 4.0_f64);
/// ```
pub fn div<S: ScalarAd>(x: S, y: S) -> S {
    x / y
}

/// Forward rule for `div`.
///
/// Returns `(primal, tangent)`.
///
/// When `y` is zero, the derivative produces NaN/Inf following IEEE 754
/// semantics, consistent with standard AD behavior for division by zero.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::div_frule;
///
/// let (y, dy) = div_frule(8.0_f64, 2.0_f64, 0.5_f64, 0.25_f64);
/// assert_eq!(y, 4.0_f64);
/// assert!((dy + 0.25_f64).abs() < 1e-12);
/// ```
pub fn div_frule<S: ScalarAd>(x: S, y: S, dx: S, dy: S) -> (S, S) {
    let primal = x / y;
    let inv_y = S::from_i32(1) / y;
    let dfdx = inv_y.conj();
    let dfdy = (S::from_i32(-1) * x * inv_y * inv_y).conj();
    let tangent = dx * dfdx + dy * dfdy;
    (primal, tangent)
}

/// Reverse rule for `div`.
///
/// Returns cotangents with respect to `(x, y)`.
///
/// When `y` is zero, the derivatives produce NaN/Inf following IEEE 754
/// semantics, consistent with standard AD behavior for division by zero.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::div_rrule;
///
/// let (dx, dy) = div_rrule(8.0_f64, 2.0_f64, 1.0_f64);
/// assert_eq!(dx, 0.5_f64);
/// assert_eq!(dy, -2.0_f64);
/// ```
pub fn div_rrule<S: ScalarAd>(x: S, y: S, cotangent: S) -> (S, S) {
    let inv_y = S::from_i32(1) / y;
    let dfdx = inv_y.conj();
    let dfdy = (S::from_i32(-1) * x * inv_y * inv_y).conj();
    (cotangent * dfdx, cotangent * dfdy)
}

/// Primal `conj`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::conj;
/// use num_complex::Complex64;
///
/// assert_eq!(conj(3.0_f64), 3.0);
/// assert_eq!(conj(Complex64::new(1.0, 2.0)), Complex64::new(1.0, -2.0));
/// ```
pub fn conj<S: ScalarAd>(x: S) -> S {
    x.conj()
}

/// Forward rule for `conj`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::conj_frule;
/// use num_complex::Complex64;
///
/// let (y, dy) = conj_frule(Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0));
/// assert_eq!(y, Complex64::new(1.0, -2.0));
/// assert_eq!(dy, Complex64::new(3.0, 4.0));
/// ```
pub fn conj_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    (x.conj(), dx.conj())
}

/// Reverse rule for `conj`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::conj_rrule;
/// use num_complex::Complex64;
///
/// let dx = conj_rrule(Complex64::new(1.0, 2.0));
/// assert_eq!(dx, Complex64::new(1.0, -2.0));
/// ```
pub fn conj_rrule<S: ScalarAd>(cotangent: S) -> S {
    cotangent.conj()
}

/// Primal `sqrt`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sqrt;
///
/// assert_eq!(sqrt(9.0_f64), 3.0);
/// ```
pub fn sqrt<S: ScalarAd>(x: S) -> S {
    x.sqrt()
}

/// Forward rule for `sqrt`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sqrt_frule;
///
/// let (_y, dy) = sqrt_frule(9.0_f64, 1.0);
/// assert!((dy - (1.0 / 6.0)).abs() < 1e-12);
/// ```
pub fn sqrt_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.sqrt();
    let dy = dx / (S::from_i32(2) * y.conj());
    (y, dy)
}

/// Reverse rule for `sqrt`.
///
/// `result` is the primal output `sqrt(x)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::sqrt_rrule;
///
/// let result = 3.0_f64;
/// let dx = sqrt_rrule(result, 1.0_f64);
/// assert!((dx - (1.0 / 6.0)).abs() < 1e-12);
/// ```
pub fn sqrt_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent / (S::from_i32(2) * result.conj())
}

/// Primal `powf`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powf;
///
/// assert_eq!(powf(2.0_f64, 3.0), 8.0);
/// ```
pub fn powf<S: ScalarAd>(x: S, exponent: S::Real) -> S {
    x.powf(exponent)
}

/// Forward rule for `powf` with fixed exponent.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powf_frule;
///
/// let (y, dy) = powf_frule(2.0_f64, 3.0, 1.0);
/// assert_eq!(y, 8.0);
/// assert_eq!(dy, 12.0);
/// ```
pub fn powf_frule<S: ScalarAd>(x: S, exponent: S::Real, dx: S) -> (S, S) {
    let y = x.powf(exponent);
    let dy = if exponent == S::Real::zero() {
        S::from_real(S::Real::zero())
    } else {
        dx * (S::from_real(exponent) * x.powf(exponent - S::Real::one())).conj()
    };
    (y, dy)
}

/// Reverse rule for `powf` with fixed exponent.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powf_rrule;
///
/// let dx = powf_rrule(2.0_f64, 3.0, 1.0);
/// assert_eq!(dx, 12.0);
/// ```
pub fn powf_rrule<S: ScalarAd>(x: S, exponent: S::Real, cotangent: S) -> S {
    if exponent == S::Real::zero() {
        return S::from_real(S::Real::zero());
    }
    cotangent * (S::from_real(exponent) * x.powf(exponent - S::Real::one())).conj()
}

/// Primal `powi`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powi;
///
/// assert_eq!(powi(2.0_f64, 4), 16.0);
/// ```
pub fn powi<S: ScalarAd>(x: S, exponent: i32) -> S {
    x.powi(exponent)
}

/// Forward rule for `powi` with fixed integer exponent.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powi_frule;
///
/// let (y, dy) = powi_frule(2.0_f64, 4, 1.0);
/// assert_eq!(y, 16.0);
/// assert_eq!(dy, 32.0);
/// ```
pub fn powi_frule<S: ScalarAd>(x: S, exponent: i32, dx: S) -> (S, S) {
    let y = x.powi(exponent);
    let dy = if exponent == 0 {
        S::from_i32(0)
    } else {
        dx * (S::from_i32(exponent) * x.powi(exponent - 1)).conj()
    };
    (y, dy)
}

/// Reverse rule for `powi` with fixed integer exponent.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::powi_rrule;
///
/// let dx = powi_rrule(2.0_f64, 4, 1.0);
/// assert_eq!(dx, 32.0);
/// ```
pub fn powi_rrule<S: ScalarAd>(x: S, exponent: i32, cotangent: S) -> S {
    if exponent == 0 {
        return S::from_i32(0);
    }
    cotangent * (S::from_i32(exponent) * x.powi(exponent - 1)).conj()
}
