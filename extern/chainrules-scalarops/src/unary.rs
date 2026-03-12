use crate::ScalarAd;

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

/// Primal `exp`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::exp;
///
/// assert!((exp(1.0_f64) - std::f64::consts::E).abs() < 1e-12);
/// ```
pub fn exp<S: ScalarAd>(x: S) -> S {
    x.exp()
}

/// Forward rule for `exp`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::exp_frule;
///
/// let (y, dy) = exp_frule(1.0_f64, 0.25_f64);
/// assert!((y - std::f64::consts::E).abs() < 1e-12);
/// assert!((dy - 0.25_f64 * std::f64::consts::E).abs() < 1e-12);
/// ```
pub fn exp_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.exp();
    (y, dx * y.conj())
}

/// Reverse rule for `exp`.
///
/// `result` is the primal output `exp(x)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::exp_rrule;
///
/// let dx = exp_rrule(std::f64::consts::E, 0.5_f64);
/// assert!((dx - 0.5_f64 * std::f64::consts::E).abs() < 1e-12);
/// ```
pub fn exp_rrule<S: ScalarAd>(result: S, cotangent: S) -> S {
    cotangent * result.conj()
}

/// Primal `log`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::log;
///
/// assert!((log(std::f64::consts::E) - 1.0_f64).abs() < 1e-12);
/// ```
pub fn log<S: ScalarAd>(x: S) -> S {
    x.ln()
}

/// Forward rule for `log`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::log_frule;
///
/// let (y, dy) = log_frule(2.0_f64, 3.0_f64);
/// assert!((y - 2.0_f64.ln()).abs() < 1e-12);
/// assert!((dy - 1.5_f64).abs() < 1e-12);
/// ```
pub fn log_frule<S: ScalarAd>(x: S, dx: S) -> (S, S) {
    let y = x.ln();
    let dy = dx * (S::from_i32(1) / x).conj();
    (y, dy)
}

/// Reverse rule for `log`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::log_rrule;
///
/// let dx = log_rrule(2.0_f64, 3.0_f64);
/// assert!((dx - 1.5_f64).abs() < 1e-12);
/// ```
pub fn log_rrule<S: ScalarAd>(x: S, cotangent: S) -> S {
    cotangent * (S::from_i32(1) / x).conj()
}
