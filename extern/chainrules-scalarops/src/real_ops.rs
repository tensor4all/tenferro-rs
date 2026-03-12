use num_traits::Float;

use crate::ScalarAd;

/// Primal `atan2(y, x)` for ordered real scalars.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::atan2;
///
/// assert!((atan2(3.0_f64, 4.0_f64) - 3.0_f64.atan2(4.0_f64)).abs() < 1e-12);
/// ```
pub fn atan2<S>(y: S, x: S) -> S
where
    S: ScalarAd<Real = S> + Float,
{
    Float::atan2(y, x)
}

/// Forward rule for `atan2(y, x)`.
///
/// Returns `(primal, tangent)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::atan2_frule;
///
/// let (_y, dy) = atan2_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
/// assert!((dy - 0.05_f64).abs() < 1e-12);
/// ```
pub fn atan2_frule<S>(y: S, x: S, dy: S, dx: S) -> (S, S)
where
    S: ScalarAd<Real = S> + Float,
{
    let primal = Float::atan2(y, x);
    let denom = x * x + y * y;
    let tangent = dy * (x / denom) + dx * ((-y) / denom);
    (primal, tangent)
}

/// Reverse rule for `atan2(y, x)`.
///
/// Returns cotangents with respect to `(y, x)`.
///
/// # Examples
///
/// ```rust
/// use chainrules_scalarops::atan2_rrule;
///
/// let (dy, dx) = atan2_rrule(3.0_f64, 4.0_f64, 2.0_f64);
/// assert!((dy - 0.32_f64).abs() < 1e-12);
/// assert!((dx + 0.24_f64).abs() < 1e-12);
/// ```
pub fn atan2_rrule<S>(y: S, x: S, cotangent: S) -> (S, S)
where
    S: ScalarAd<Real = S> + Float,
{
    let denom = x * x + y * y;
    (cotangent * (x / denom), cotangent * ((-y) / denom))
}
