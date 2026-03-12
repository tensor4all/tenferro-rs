use num_complex::{Complex32, Complex64};

use crate::{
    atan2, atan2_frule, conj, conj_frule, conj_rrule, exp, exp_frule, exp_rrule, handle_r_to_c_f32,
    handle_r_to_c_f64, log, log_frule, log_rrule, sqrt, sqrt_frule, sqrt_rrule, ScalarAd,
};

fn assert_close_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < 1.0e-5,
        "actual={actual}, expected={expected}",
    );
}

fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}",
    );
}

fn assert_close_c32(actual: Complex32, expected: Complex32) {
    assert_close_f32(actual.re, expected.re);
    assert_close_f32(actual.im, expected.im);
}

fn assert_close_c64(actual: Complex64, expected: Complex64) {
    assert_close_f64(actual.re, expected.re);
    assert_close_f64(actual.im, expected.im);
}

#[test]
fn scalar_ad_real_impls_match_std_real_ops() {
    let x32 = 0.25_f32;
    assert_close_f32(<f32 as ScalarAd>::expm1(x32), x32.exp_m1());
    assert_close_f32(<f32 as ScalarAd>::log1p(x32), x32.ln_1p());
    assert_close_f32(<f32 as ScalarAd>::sin(x32), x32.sin());
    assert_close_f32(<f32 as ScalarAd>::cos(x32), x32.cos());
    assert_close_f32(<f32 as ScalarAd>::tanh(x32), x32.tanh());
    assert_close_f32(<f32 as ScalarAd>::powf(x32, 2.5), x32.powf(2.5));
    assert_eq!(<f32 as ScalarAd>::powi(x32, 3), x32.powi(3));
    assert_eq!(<f32 as ScalarAd>::from_real(1.5), 1.5);
    assert_eq!(<f32 as ScalarAd>::from_i32(-2), -2.0);

    let x64 = 0.5_f64;
    assert_close_f64(<f64 as ScalarAd>::expm1(x64), x64.exp_m1());
    assert_close_f64(<f64 as ScalarAd>::log1p(x64), x64.ln_1p());
    assert_close_f64(<f64 as ScalarAd>::sin(x64), x64.sin());
    assert_close_f64(<f64 as ScalarAd>::cos(x64), x64.cos());
    assert_close_f64(<f64 as ScalarAd>::tanh(x64), x64.tanh());
    assert_close_f64(<f64 as ScalarAd>::powf(x64, 1.5), x64.powf(1.5));
    assert_eq!(<f64 as ScalarAd>::powi(x64, 4), x64.powi(4));
    assert_eq!(<f64 as ScalarAd>::from_real(2.5), 2.5);
    assert_eq!(<f64 as ScalarAd>::from_i32(3), 3.0);
}

#[test]
fn scalar_ad_complex_impls_match_std_complex_ops() {
    let x32 = Complex32::new(0.25, -0.5);
    assert_close_c32(<Complex32 as ScalarAd>::conj(x32), x32.conj());
    assert_close_c32(<Complex32 as ScalarAd>::sqrt(x32), x32.sqrt());
    assert_close_c32(<Complex32 as ScalarAd>::exp(x32), x32.exp());
    assert_close_c32(
        <Complex32 as ScalarAd>::expm1(x32),
        x32.exp() - Complex32::new(1.0, 0.0),
    );
    assert_close_c32(<Complex32 as ScalarAd>::ln(x32), x32.ln());
    assert_close_c32(
        <Complex32 as ScalarAd>::log1p(x32),
        (x32 + Complex32::new(1.0, 0.0)).ln(),
    );
    assert_close_c32(<Complex32 as ScalarAd>::sin(x32), x32.sin());
    assert_close_c32(<Complex32 as ScalarAd>::cos(x32), x32.cos());
    assert_close_c32(<Complex32 as ScalarAd>::tanh(x32), x32.tanh());
    assert_close_c32(<Complex32 as ScalarAd>::powf(x32, 2.0), x32.powf(2.0));
    assert_close_c32(<Complex32 as ScalarAd>::powi(x32, 3), x32.powi(3));
    assert_eq!(
        <Complex32 as ScalarAd>::from_real(1.5),
        Complex32::new(1.5, 0.0)
    );
    assert_eq!(
        <Complex32 as ScalarAd>::from_i32(-2),
        Complex32::new(-2.0, 0.0)
    );

    let x64 = Complex64::new(0.5, 0.75);
    assert_close_c64(<Complex64 as ScalarAd>::conj(x64), x64.conj());
    assert_close_c64(<Complex64 as ScalarAd>::sqrt(x64), x64.sqrt());
    assert_close_c64(<Complex64 as ScalarAd>::exp(x64), x64.exp());
    assert_close_c64(
        <Complex64 as ScalarAd>::expm1(x64),
        x64.exp() - Complex64::new(1.0, 0.0),
    );
    assert_close_c64(<Complex64 as ScalarAd>::ln(x64), x64.ln());
    assert_close_c64(
        <Complex64 as ScalarAd>::log1p(x64),
        (x64 + Complex64::new(1.0, 0.0)).ln(),
    );
    assert_close_c64(<Complex64 as ScalarAd>::sin(x64), x64.sin());
    assert_close_c64(<Complex64 as ScalarAd>::cos(x64), x64.cos());
    assert_close_c64(<Complex64 as ScalarAd>::tanh(x64), x64.tanh());
    assert_close_c64(<Complex64 as ScalarAd>::powf(x64, 1.5), x64.powf(1.5));
    assert_close_c64(<Complex64 as ScalarAd>::powi(x64, 2), x64.powi(2));
    assert_eq!(
        <Complex64 as ScalarAd>::from_real(2.5),
        Complex64::new(2.5, 0.0)
    );
    assert_eq!(
        <Complex64 as ScalarAd>::from_i32(4),
        Complex64::new(4.0, 0.0)
    );
}

#[test]
fn direct_entrypoints_match_real_projection_and_atan2_formulas() {
    assert_eq!(handle_r_to_c_f32(Complex32::new(2.0, -5.0)), 2.0);
    assert_eq!(handle_r_to_c_f64(Complex64::new(-3.0, 1.5)), -3.0);

    let primal = atan2(3.0_f64, 4.0_f64);
    assert_close_f64(primal, 3.0_f64.atan2(4.0));

    let (atan2_y, atan2_dy) = atan2_frule(3.0_f64, 4.0_f64, 0.5_f64, 0.25_f64);
    assert_close_f64(atan2_y, primal);
    assert_close_f64(atan2_dy, 0.05);
}

#[test]
fn unary_entrypoints_match_forward_and_reverse_formulas() {
    let complex = Complex32::new(1.0, -2.0);
    assert_eq!(conj(complex), complex.conj());
    let (_y, dy) = conj_frule(complex, Complex32::new(3.0, 4.0));
    assert_eq!(dy, Complex32::new(3.0, -4.0));
    assert_eq!(conj_rrule(complex), complex.conj());

    assert_eq!(sqrt(9.0_f32), 3.0);
    let (sqrt_y, sqrt_dy) = sqrt_frule(9.0_f32, 2.0_f32);
    assert_eq!(sqrt_y, 3.0);
    assert_close_f32(sqrt_dy, 1.0 / 3.0);
    assert_close_f32(sqrt_rrule(3.0_f32, 2.0_f32), 1.0 / 3.0);

    let exp_y = exp(1.0_f32);
    assert_close_f32(exp_y, std::f32::consts::E);
    let (exp_primal, exp_tangent) = exp_frule(1.0_f32, 0.25_f32);
    assert_close_f32(exp_primal, std::f32::consts::E);
    assert_close_f32(exp_tangent, 0.25 * std::f32::consts::E);
    assert_close_f32(exp_rrule(exp_primal, 0.5_f32), 0.5 * std::f32::consts::E);

    let log_y = log(std::f32::consts::E);
    assert_close_f32(log_y, 1.0);
    let (log_primal, log_tangent) = log_frule(2.0_f32, 3.0_f32);
    assert_close_f32(log_primal, 2.0_f32.ln());
    assert_close_f32(log_tangent, 1.5);
    assert_close_f32(log_rrule(2.0_f32, 3.0_f32), 1.5);
}
