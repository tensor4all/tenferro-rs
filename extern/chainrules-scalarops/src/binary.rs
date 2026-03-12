use crate::ScalarAd;

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
