//! Reproducible `log` and `sincospi` for the sample transforms.
//!
//! Both functions are built only from IEEE 754 operations that are correctly
//! rounded on every tenferro backend (`+`, `-`, `*`, `/`, `sqrt`, explicit
//! `fma` through `mul_add`, exact rounding to an integer, and bit
//! manipulation), evaluated in a fixed order. The same input therefore gives
//! the same bits on every platform; no platform `log`, `sin` or `cos` is
//! called. They are not correctly rounded functions: the accuracy target is
//! 2 ulp, checked against mpmath in `tests/math.rs` (the measured maximum is
//! below 1 ulp).
//!
//! - `log`: `x = 2^k m` with `m` in `[sqrt(1/2), sqrt(2))`, `f = m - 1`,
//!   `s = f / (2 + f)` and `log(1 + f) = f - hfsq + s (hfsq + R(s^2))` with
//!   `hfsq = f^2 / 2` and `R` the truncated Taylor series of
//!   `2 atanh(s) / s - 2`. This is the classic reduction of fdlibm's `e_log.c`
//!   (Sun Microsystems); the code and its coefficients (`2 / (2j + 1)`) are
//!   written here, not copied.
//! - `sincospi`: `x = n/2 + r` with `n = round(2x)` and `|r| <= 1/4`, both
//!   exact; `sin(pi r)` and `cos(pi r)` are truncated Taylor series with
//!   `pi r` carried in two parts, and the quadrant `n mod 4` picks the signs.
//!   No rounded `pi` multiplies the argument before reduction.
//!
//! The constants are derived by `reference/math_constants.py`.
//!
//! The CUDA implementation must perform the same operations in the same
//! order (compiled with `--fmad=false`, no flush-to-zero, IEEE division and
//! square root), with every `mul_add` written as an explicit `fma`.

/// Evaluates `c[0] + z (c[1] + z (c[2] + ...))` with one fused multiply-add
/// per coefficient, starting from the last.
macro_rules! horner {
    ($z:expr, $coeffs:expr) => {{
        let coeffs = $coeffs;
        let mut acc = coeffs[coeffs.len() - 1];
        let mut index = coeffs.len() - 1;
        while index > 0 {
            index -= 1;
            acc = acc.mul_add($z, coeffs[index]);
        }
        acc
    }};
}

macro_rules! reproducible_math {
    (
        $float:ty, $bits:ty, $log:ident, $sincospi:ident,
        mantissa_bits: $mantissa_bits:expr,
        exponent_bias: $bias:expr,
        subnormal_scale: $scale:expr,
        ln2_hi: $ln2_hi:expr,
        ln2_lo: $ln2_lo:expr,
        sqrt_2: $sqrt_2:expr,
        log_r: $log_r:expr,
        pi_hi: $pi_hi:expr,
        pi_lo: $pi_lo:expr,
        sin: $sin:expr,
        cos: $cos:expr,
    ) => {
        pub(crate) fn $log(x: $float) -> $float {
            const MANTISSA_BITS: u32 = $mantissa_bits;
            const MANTISSA_MASK: $bits = (1 << MANTISSA_BITS) - 1;
            const ONE_BITS: $bits = (1.0 as $float).to_bits();
            const LOG_R: &[$float] = &$log_r;

            if x.is_nan() || x < 0.0 {
                return <$float>::NAN;
            }
            if x == 0.0 {
                return <$float>::NEG_INFINITY;
            }
            if x == <$float>::INFINITY {
                return x;
            }
            let (mut bits, mut k): ($bits, i32) = if x.is_subnormal() {
                // Scaling by a power of two is exact.
                const SCALE: $float =
                    <$float>::from_bits((($bias + $scale) as $bits) << MANTISSA_BITS);
                ((x * SCALE).to_bits(), -$scale)
            } else {
                (x.to_bits(), 0)
            };
            k += (bits >> MANTISSA_BITS) as i32 - $bias;
            bits = (bits & MANTISSA_MASK) | ONE_BITS;
            let mut m = <$float>::from_bits(bits);
            if m > $sqrt_2 {
                m *= 0.5;
                k += 1;
            }
            let f = m - 1.0;
            let hfsq = 0.5 * (f * f);
            let s = f / (2.0 + f);
            let z = s * s;
            let r = z * horner!(z, LOG_R);
            let kf = k as $float;
            let tail = s.mul_add(hfsq + r, kf * $ln2_lo);
            kf.mul_add($ln2_hi, -((hfsq - tail) - f))
        }

        pub(crate) fn $sincospi(x: $float) -> ($float, $float) {
            // Above this magnitude every value is an integer.
            const INTEGER_LIMIT: $float = (1u64 << $mantissa_bits) as $float;
            const SIN: &[$float] = &$sin;
            const COS: &[$float] = &$cos;

            if !x.is_finite() {
                return (<$float>::NAN, <$float>::NAN);
            }
            if x.abs() >= INTEGER_LIMIT {
                // `x * 0.5` is exact here; an odd integer has a fraction.
                let half = x * 0.5;
                let cos = if half == half.floor() { 1.0 } else { -1.0 };
                return ((0.0 as $float).copysign(x), cos);
            }
            let n = (x + x).round_ties_even();
            let r = x - 0.5 * n;
            let quadrant = (n - 4.0 * (0.25 * n).floor()) as u32;
            let z = r * r;

            let pi_r = $pi_hi * r;
            let pi_r_error = $pi_hi.mul_add(r, -pi_r);
            let tail = (r * z).mul_add(horner!(z, SIN), pi_r_error);
            let sin_r = pi_r + $pi_lo.mul_add(r, tail);
            let cos_r = z.mul_add(horner!(z, COS), 1.0);

            let (mut sin, mut cos) = match quadrant {
                0 => (sin_r, cos_r),
                1 => (cos_r, -sin_r),
                2 => (-sin_r, -cos_r),
                _ => (-cos_r, sin_r),
            };
            // IEEE 754 sinPi/cosPi: sinPi(n) has the sign of n, and
            // cosPi(n + 1/2) is +0.
            if sin == 0.0 {
                sin = (0.0 as $float).copysign(x);
            }
            if cos == 0.0 {
                cos = 0.0;
            }
            (sin, cos)
        }
    };
}

reproducible_math! {
    f64, u64, log_f64, sincospi_f64,
    mantissa_bits: 52,
    exponent_bias: 1023,
    subnormal_scale: 54,
    ln2_hi: 0.6931471805598903,
    ln2_lo: 5.497923018708371e-14,
    sqrt_2: core::f64::consts::SQRT_2,
    log_r: [
        0.6666666666666666,
        0.4,
        0.2857142857142857,
        0.2222222222222222,
        0.18181818181818182,
        0.15384615384615385,
        0.13333333333333333,
        0.11764705882352941,
        0.10526315789473684,
        0.09523809523809523,
        0.08695652173913043,
    ],
    pi_hi: core::f64::consts::PI,
    pi_lo: 1.2246467991473532e-16_f64,
    sin: [
        -5.16771278004997,
        2.5501640398773455,
        -0.5992645293207921,
        0.08214588661112823,
        -0.0073704309457143504,
        0.00046630280576761255,
        -2.1915353447830217e-05,
        7.952054001475513e-07,
    ],
    cos: [
        -4.934802200544679,
        4.0587121264167685,
        -1.3352627688545895,
        0.2353306303588932,
        -0.02580689139001406,
        0.0019295743094039231,
        -0.0001046381049248457,
        4.303069587032947e-06,
    ],
}

reproducible_math! {
    f32, u32, log_f32, sincospi_f32,
    mantissa_bits: 23,
    exponent_bias: 127,
    subnormal_scale: 25,
    ln2_hi: 0.69314575,
    ln2_lo: 1.4286068e-6,
    sqrt_2: core::f32::consts::SQRT_2,
    log_r: [
        0.6666667,
        0.4,
        0.2857143,
        0.22222222,
        0.18181819,
    ],
    pi_hi: core::f32::consts::PI,
    pi_lo: -8.742278e-8_f32,
    sin: [
        -5.1677127,
        2.550164,
        -0.5992645,
        0.082145885,
    ],
    cos: [
        -4.934802,
        4.058712,
        -1.3352628,
        0.23533063,
        -0.02580689,
    ],
}

/// The natural logarithm, reproducible bit for bit on every backend.
///
/// The function is a fixed sequence of IEEE 754 operations that every backend
/// rounds correctly (`+`, `-`, `*`, `/`, explicit `fma`, bit manipulation), so
/// the same input gives the same bits everywhere; no platform `log` is called.
/// It is not a correctly rounded logarithm: the result is within 2 ulp of the
/// exact value (below 1 ulp in the tests). The method is fdlibm's reduction
/// `x = 2^k (1 + f)`, `s = f / (2 + f)`, with a truncated Taylor series of
/// `atanh`.
///
/// `log(+0)` and `log(-0)` are `-inf`, a negative or NaN input gives NaN and
/// `log(+inf)` is `+inf`.
///
/// # Examples
///
/// ```
/// use tenferro_random_core::log;
///
/// assert_eq!(log(1.0_f64), 0.0);
/// assert!((log(2.0_f32) - core::f32::consts::LN_2).abs() <= f32::EPSILON);
/// assert_eq!(log(0.0_f64), f64::NEG_INFINITY);
/// assert!(log(-1.0_f64).is_nan());
/// ```
#[must_use]
pub fn log<T: crate::RandomFloat>(x: T) -> T {
    <T as crate::sealed::Float>::log(x)
}

/// `(sin(pi x), cos(pi x))`, reproducible bit for bit on every backend.
///
/// Like [`log`], the function is a fixed sequence of correctly rounded IEEE 754
/// operations, so the same input gives the same bits on every backend; no
/// platform `sin` or `cos` is called. The argument is reduced exactly
/// (`x = n/2 + r`, `|r| <= 1/4`), so no rounded `pi` enters, and each result is
/// within 2 ulp of the exact value (below 1 ulp in the tests).
///
/// Following IEEE 754 `sinPi`/`cosPi`, the sine of an integer is a zero with
/// the sign of `x` and the cosine of a half-integer is `+0`. A NaN or infinite
/// input gives `(NaN, NaN)`.
///
/// # Examples
///
/// ```
/// use tenferro_random_core::sincospi;
///
/// assert_eq!(sincospi(0.5_f64), (1.0, 0.0));
/// assert_eq!(sincospi(1.0_f32), (0.0, -1.0));
/// let (sin, cos) = sincospi(0.25_f64);
/// assert!((sin - core::f64::consts::FRAC_1_SQRT_2).abs() <= f64::EPSILON);
/// assert!((cos - core::f64::consts::FRAC_1_SQRT_2).abs() <= f64::EPSILON);
/// ```
#[must_use]
pub fn sincospi<T: crate::RandomFloat>(x: T) -> (T, T) {
    <T as crate::sealed::Float>::sincospi(x)
}
