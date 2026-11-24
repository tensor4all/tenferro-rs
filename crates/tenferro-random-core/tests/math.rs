//! `log` and `sincospi` at fixed points, special values and the inputs
//! Box–Muller produces.
//!
//! Each fixed point has two expected values: the frozen output bits (what
//! every backend must reproduce) and the exact value from mpmath rounded once
//! (`reference/samples.py`). The test asserts the frozen bits and that they
//! are within 1 ulp of the exact value; the design's accuracy target is 2 ulp.

mod common;

use common::{ulps_f32, ulps_f64};
use tenferro_random_core::{log, sincospi};

/// (input, frozen output, mpmath rounded once)
const LOG_F32: [(u32, u32, u32); 11] = [
    (0x34000000, 0xc17f1402, 0xc17f1402), // 2^-23: smallest 1 - a of an f32 uniform
    (0x3f7ffffe, 0xb4000001, 0xb4000001), // 1 - 2^-23: largest below 1
    (0x3f800000, 0x00000000, 0x00000000), // 1: a = 0
    (0x3f000000, 0xbf317218, 0xbf317218), // 0.5
    (0x3f400000, 0xbe934b10, 0xbe934b11), // 0.75
    (0x3f3504f2, 0xbeb1721c, 0xbeb1721b), // just below sqrt(1/2): reduction edge
    (0x3fb504f3, 0x3eb17218, 0x3eb17217), // sqrt(2) rounded: reduction edge
    (0x41200000, 0x40135d8e, 0x40135d8e), // 10
    (0x0da24260, 0xc28a27b5, 0xc28a27b5), // 1e-30
    (0x7f61b1e6, 0x42b13196, 0x42b13196), // 3e38
    (0x00000001, 0xc2ce8ed0, 0xc2ce8ed0), // smallest subnormal
];

const LOG_F64: [(u64, u64, u64); 11] = [
    (0x3cb0000000000000, 0xc04205966f2b4f12, 0xc04205966f2b4f12), // 2^-52
    (0x3feffffffffffffe, 0xbcb0000000000001, 0xbcb0000000000001), // 1 - 2^-52
    (0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000), // 1
    (0x3fe0000000000000, 0xbfe62e42fefa39ef, 0xbfe62e42fefa39ef), // 0.5
    (0x3fe8000000000000, 0xbfd269621134db92, 0xbfd269621134db92), // 0.75
    (0x3fe6a09e667f3bcd, 0xbfd62e42fefa39ee, 0xbfd62e42fefa39ee), // sqrt(1/2)
    (0x3ff6a09e667f3bcd, 0x3fd62e42fefa39f0, 0x3fd62e42fefa39f0), // sqrt(2)
    (0x4024000000000000, 0x40026bb1bbb55516, 0x40026bb1bbb55516), // 10
    (0x01a56e1fc2f8f359, 0xc085963447f87fb5, 0xc085963447f87fb5), // 1e-300
    (0x7fee42d130773b76, 0x40862dd08fdc6f88, 0x40862dd08fdc6f88), // 1.7e308
    (0x0000000000000001, 0xc0874385446d71c3, 0xc0874385446d71c3), // smallest subnormal
];

/// (input, frozen (sin, cos), mpmath (sin, cos))
const SINCOSPI_F32: [(u32, [u32; 2], [u32; 2]); 14] = [
    (
        0x00000000,
        [0x00000000, 0x3f800000],
        [0x00000000, 0x3f800000],
    ), // 0
    (
        0x34800000,
        [0x35490fdb, 0x3f800000],
        [0x35490fdb, 0x3f800000],
    ), // 2^-22: smallest angle 2b
    (
        0x3e7ffffe,
        [0x3f3504f2, 0x3f3504f5],
        [0x3f3504f2, 0x3f3504f4],
    ), // below 1/4
    (
        0x3e800000,
        [0x3f3504f3, 0x3f3504f4],
        [0x3f3504f3, 0x3f3504f3],
    ), // 1/4: quadrant edge
    (
        0x3e800002,
        [0x3f3504f5, 0x3f3504f1],
        [0x3f3504f5, 0x3f3504f1],
    ), // above 1/4
    (
        0x3efffffe,
        [0x3f800000, 0x34490fdb],
        [0x3f800000, 0x34490fdb],
    ), // below 1/2
    (
        0x3f000000,
        [0x3f800000, 0x00000000],
        [0x3f800000, 0x00000000],
    ), // 1/2
    (
        0x3f000002,
        [0x3f800000, 0xb4c90fdb],
        [0x3f800000, 0xb4c90fdb],
    ), // above 1/2
    (
        0x3f800000,
        [0x00000000, 0xbf800000],
        [0x00000000, 0xbf800000],
    ), // 1
    (
        0x3fc00000,
        [0xbf800000, 0x00000000],
        [0xbf800000, 0x00000000],
    ), // 3/2
    (
        0x3ffffffe,
        [0xb5490fdb, 0x3f800000],
        [0xb5490fdb, 0x3f800000],
    ), // 2 - 2^-22: largest 2b
    (
        0x3dcccccd,
        [0x3e9e377a, 0x3f737871],
        [0x3e9e377a, 0x3f737871],
    ), // 0.1
    (
        0xbf400000,
        [0xbf3504f3, 0xbf3504f4],
        [0xbf3504f3, 0xbf3504f3],
    ), // -3/4
    (
        0x4640e600,
        [0xbf800000, 0x00000000],
        [0xbf800000, 0x00000000],
    ), // 12345.5
];

const SINCOSPI_F64: [(u64, [u64; 2], [u64; 2]); 14] = [
    (
        0x0000000000000000,
        [0x0000000000000000, 0x3ff0000000000000],
        [0x0000000000000000, 0x3ff0000000000000],
    ),
    (
        0x3cc0000000000000,
        [0x3cd921fb54442d18, 0x3ff0000000000000],
        [0x3cd921fb54442d18, 0x3ff0000000000000],
    ),
    (
        0x3fcffffffffffffe,
        [0x3fe6a09e667f3bcb, 0x3fe6a09e667f3bce],
        [0x3fe6a09e667f3bcb, 0x3fe6a09e667f3bce],
    ),
    (
        0x3fd0000000000000,
        [0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd],
        [0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd],
    ),
    (
        0x3fd0000000000002,
        [0x3fe6a09e667f3bcf, 0x3fe6a09e667f3bca],
        [0x3fe6a09e667f3bcf, 0x3fe6a09e667f3bca],
    ),
    (
        0x3fdffffffffffffe,
        [0x3ff0000000000000, 0x3cb921fb54442d18],
        [0x3ff0000000000000, 0x3cb921fb54442d18],
    ),
    (
        0x3fe0000000000000,
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
    ),
    (
        0x3fe0000000000002,
        [0x3ff0000000000000, 0xbcc921fb54442d18],
        [0x3ff0000000000000, 0xbcc921fb54442d18],
    ),
    (
        0x3ff0000000000000,
        [0x0000000000000000, 0xbff0000000000000],
        [0x0000000000000000, 0xbff0000000000000],
    ),
    (
        0x3ff8000000000000,
        [0xbff0000000000000, 0x0000000000000000],
        [0xbff0000000000000, 0x0000000000000000],
    ),
    (
        0x3ffffffffffffffe,
        [0xbcd921fb54442d18, 0x3ff0000000000000],
        [0xbcd921fb54442d18, 0x3ff0000000000000],
    ),
    (
        0x3fb999999999999a,
        [0x3fd3c6ef372fe950, 0x3fee6f0e134454ff],
        [0x3fd3c6ef372fe950, 0x3fee6f0e134454ff],
    ),
    (
        0xbfe8000000000000,
        [0xbfe6a09e667f3bcc, 0xbfe6a09e667f3bcd],
        [0xbfe6a09e667f3bcd, 0xbfe6a09e667f3bcd],
    ),
    (
        0x40c81cc000000000,
        [0xbff0000000000000, 0x0000000000000000],
        [0xbff0000000000000, 0x0000000000000000],
    ),
];

#[test]
fn log_matches_frozen_bits_within_one_ulp_of_mpmath() {
    for (x, frozen, exact) in LOG_F32 {
        let got = log(f32::from_bits(x)).to_bits();
        assert_eq!(got, frozen, "log f32 {:e}", f32::from_bits(x));
        assert!(ulps_f32(got, exact) <= 1, "log f32 {:e}", f32::from_bits(x));
    }
    for (x, frozen, exact) in LOG_F64 {
        let got = log(f64::from_bits(x)).to_bits();
        assert_eq!(got, frozen, "log f64 {:e}", f64::from_bits(x));
        assert!(ulps_f64(got, exact) <= 1, "log f64 {:e}", f64::from_bits(x));
    }
}

#[test]
fn sincospi_matches_frozen_bits_within_one_ulp_of_mpmath() {
    for (x, frozen, exact) in SINCOSPI_F32 {
        let (sin, cos) = sincospi(f32::from_bits(x));
        assert_eq!(
            [sin.to_bits(), cos.to_bits()],
            frozen,
            "sincospi f32 {}",
            f32::from_bits(x)
        );
        assert!(ulps_f32(sin.to_bits(), exact[0]) <= 1 && ulps_f32(cos.to_bits(), exact[1]) <= 1);
    }
    for (x, frozen, exact) in SINCOSPI_F64 {
        let (sin, cos) = sincospi(f64::from_bits(x));
        assert_eq!(
            [sin.to_bits(), cos.to_bits()],
            frozen,
            "sincospi f64 {}",
            f64::from_bits(x)
        );
        assert!(ulps_f64(sin.to_bits(), exact[0]) <= 1 && ulps_f64(cos.to_bits(), exact[1]) <= 1);
    }
}

#[test]
fn log_special_values() {
    assert_eq!(log(0.0_f64), f64::NEG_INFINITY);
    assert_eq!(log(-0.0_f64), f64::NEG_INFINITY);
    assert_eq!(log(f64::INFINITY), f64::INFINITY);
    assert!(log(-1.0_f64).is_nan());
    assert!(log(f64::NEG_INFINITY).is_nan());
    assert!(log(f64::NAN).is_nan());
    assert_eq!(log(0.0_f32), f32::NEG_INFINITY);
    assert_eq!(log(f32::INFINITY), f32::INFINITY);
    assert!(log(-f32::MIN_POSITIVE).is_nan());
    assert!(log(f32::NAN).is_nan());
    // log(1) is +0, so a Box-Muller radius from a = 0 is +0.
    assert_eq!(log(1.0_f64).to_bits(), 0);
    assert_eq!(log(1.0_f32).to_bits(), 0);
}

#[test]
fn sincospi_special_values_follow_ieee_sinpi_cospi() {
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (sin, cos) = sincospi(x);
        assert!(sin.is_nan() && cos.is_nan());
    }
    let (sin, cos) = sincospi(f32::NAN);
    assert!(sin.is_nan() && cos.is_nan());
    // sinPi(n) is a zero with the sign of n; cosPi(n + 1/2) is +0.
    for n in [-3.0_f64, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 7.0] {
        let (sin, cos) = sincospi(n);
        assert_eq!(sin.to_bits(), 0.0_f64.copysign(n).to_bits(), "sinpi({n})");
        assert_eq!(cos.abs(), 1.0);
        let (_, cos) = sincospi(n + 0.5);
        assert_eq!(cos.to_bits(), 0, "cospi({n} + 1/2)");
    }
    // Above 2^52 (f64) or 2^23 (f32) every value is an integer.
    assert_eq!(sincospi(4_503_599_627_370_497.0_f64), (0.0, -1.0)); // 2^52 + 1, odd
    assert_eq!(sincospi(9_007_199_254_740_992.0_f64), (0.0, 1.0)); // 2^53
    assert_eq!(sincospi(-1e300_f64).0.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(sincospi(8_388_609.0_f32), (0.0, -1.0)); // 2^23 + 1, odd
    assert_eq!(sincospi(3e38_f32), (0.0, 1.0));
    // The largest value below 2^52 keeps its half: (2^52 - 1/2).
    let (sin, cos) = sincospi(4_503_599_627_370_495.5_f64);
    assert_eq!((sin, cos.to_bits()), (-1.0, 0));
}

#[test]
fn f32_functions_stay_within_one_ulp_over_the_box_muller_inputs() {
    // Every 1021st of the 2^23 inputs Box–Muller gives log (1 - k 2^-23) and
    // sincospi (k 2^-22). The reference is f64 arithmetic, whose own error is
    // below 2^-28 f32 ulp. The full sweep, run offline, has maxima 0.81 ulp
    // (log) and 0.99 ulp (sin, cos).
    fn ulp_error(got: f32, exact: f64) -> f64 {
        if exact == 0.0 {
            return if got == 0.0 { 0.0 } else { f64::INFINITY };
        }
        let exponent = (exact.abs().log2().floor() as i32).max(-126);
        (f64::from(got) - exact).abs() / 2f64.powi(exponent - 23)
    }
    let mut worst = (0.0_f64, 0.0_f64);
    for k in (0..1u32 << 23).step_by(1021).chain([1, (1 << 23) - 1]) {
        let u = 1.0 - k as f32 * 2f32.powi(-23);
        worst.0 = worst.0.max(ulp_error(log(u), f64::from(u).ln()));

        let x = k as f32 * 2f32.powi(-22);
        let (sin, cos) = sincospi(x);
        let x = f64::from(x);
        let n = (2.0 * x).round();
        let r = core::f64::consts::PI * (x - n / 2.0);
        let (s, c) = (r.sin(), r.cos());
        let (sin_exact, cos_exact) = match n as i64 % 4 {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        };
        worst.1 = worst
            .1
            .max(ulp_error(sin, sin_exact))
            .max(ulp_error(cos, cos_exact));
    }
    assert!(worst.0 < 1.0, "log f32 error {} ulp", worst.0);
    assert!(worst.1 < 1.0, "sincospi f32 error {} ulp", worst.1);
}
