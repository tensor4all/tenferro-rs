//! #1922: CPU complex division is scale-robust.
//!
//! The textbook `a * conj(b) / |b|^2` overflows or underflows `|b|^2` when the
//! quotient itself is representable. These cases mirror the CUDA regression
//! (`tenferro-gpu` `complex_division_is_scale_robust`) on the CPU backend, for
//! one-element tensors and for longer tensors that take the vectorized path.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{Tensor, TensorRead, TensorScalar};

const LONG: usize = 1024;

fn tensor<T: TensorScalar>(len: usize, value: T) -> Tensor {
    let shape = if len == 0 { vec![] } else { vec![len] };
    let count = len.max(1);
    Tensor::from_vec_col_major(shape, vec![value; count]).unwrap()
}

fn divide(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let mut engine = CpuEngine::serial();
    tenferro_tensor::TensorElementwise::div_read(
        &mut engine,
        TensorRead::from_tensor(lhs),
        TensorRead::from_tensor(rhs),
    )
    .unwrap()
}

fn assert_close64(actual: Complex64, expected: Complex64, context: &str) {
    let close = |a: f64, e: f64| {
        if e == 0.0 {
            a == 0.0
        } else {
            (a - e).abs() <= e.abs() * 1e-12
        }
    };
    assert!(
        close(actual.re, expected.re) && close(actual.im, expected.im),
        "{context}: {actual:?} against {expected:?}"
    );
}

fn assert_close32(actual: Complex32, expected: Complex32, context: &str) {
    let close = |a: f32, e: f32| {
        if e == 0.0 {
            a == 0.0
        } else {
            (a - e).abs() <= e.abs() * 1e-5
        }
    };
    assert!(
        close(actual.re, expected.re) && close(actual.im, expected.im),
        "{context}: {actual:?} against {expected:?}"
    );
}

fn each_c64(out: &Tensor) -> &[Complex64] {
    out.as_slice::<Complex64>().unwrap()
}

fn each_c32(out: &Tensor) -> &[Complex32] {
    out.as_slice::<Complex32>().unwrap()
}

#[test]
fn c64_complex_division_survives_extreme_divisors() {
    let huge = 2f64.powi(600);
    let tiny = 2f64.powi(-600);
    let min_normal = f64::MIN_POSITIVE;
    let cases = [
        // 1 / (2^600 (1 + i)) = 2^-601 (1 - i)
        (
            Complex64::new(1.0, 0.0),
            Complex64::new(huge, huge),
            Complex64::new(2f64.powi(-601), -2f64.powi(-601)),
        ),
        // 1 / (2^-600 (1 + i)) = 2^599 (1 - i)
        (
            Complex64::new(1.0, 0.0),
            Complex64::new(tiny, tiny),
            Complex64::new(2f64.powi(599), -2f64.powi(599)),
        ),
        // Near the subnormal boundary: the subnormal 2^-1060 divided by
        // 2^-1022 (1 + i) is 2^-39 (1 - i) (Julia Base agrees). `powi` cannot
        // build 2^-1060 directly because 2^1060 overflows.
        (
            Complex64::new(min_normal * 2f64.powi(-38), 0.0),
            Complex64::new(min_normal, min_normal),
            Complex64::new(2f64.powi(-39), -2f64.powi(-39)),
        ),
        // An infinite divisor gives zeros.
        (
            Complex64::new(1.0, 1.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, 0.0),
        ),
        (
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
            Complex64::new(0.0, 0.0),
        ),
    ];
    for len in [1, LONG] {
        for (index, (lhs, rhs, expected)) in cases.iter().enumerate() {
            let out = divide(&tensor(len, *lhs), &tensor(len, *rhs));
            for value in each_c64(&out) {
                assert_close64(*value, *expected, &format!("case {index}, len {len}"));
            }
        }
    }
}

#[test]
fn c32_complex_division_survives_extreme_divisors() {
    let cases = [
        // 1 / (2^60 (1 + i)) = 2^-61 (1 - i)
        (
            Complex32::new(1.0, 0.0),
            Complex32::new(2f32.powi(60), 2f32.powi(60)),
            Complex32::new(2f32.powi(-61), -2f32.powi(-61)),
        ),
        // 1 / (2^-60 (1 + i)) = 2^59 (1 - i)
        (
            Complex32::new(1.0, 0.0),
            Complex32::new(2f32.powi(-60), 2f32.powi(-60)),
            Complex32::new(2f32.powi(59), -2f32.powi(59)),
        ),
        // Near the subnormal boundary: 2^-140 / (2^-126 (1 + i)) = 2^-15 (1 - i).
        (
            Complex32::new(f32::MIN_POSITIVE * 2f32.powi(-14), 0.0),
            Complex32::new(f32::MIN_POSITIVE, f32::MIN_POSITIVE),
            Complex32::new(2f32.powi(-15), -2f32.powi(-15)),
        ),
        (
            Complex32::new(1.0, 1.0),
            Complex32::new(f32::NEG_INFINITY, 0.0),
            Complex32::new(0.0, 0.0),
        ),
    ];
    for len in [1, LONG] {
        for (index, (lhs, rhs, expected)) in cases.iter().enumerate() {
            let out = divide(&tensor(len, *lhs), &tensor(len, *rhs));
            for value in each_c32(&out) {
                assert_close32(*value, *expected, &format!("case {index}, len {len}"));
            }
        }
    }
}

#[test]
fn mixed_real_scalar_complex_division_survives_extreme_operands() {
    let huge = 2f64.powi(600);
    for len in [1, LONG] {
        let huge_complex = tensor(len, Complex64::new(huge, huge));
        // 2 / (2^600 (1 + i)) = 2^-600 (1 - i)
        let out = divide(&tensor(0, 2.0_f64), &huge_complex);
        for value in each_c64(&out) {
            assert_close64(
                *value,
                Complex64::new(2f64.powi(-600), -2f64.powi(-600)),
                &format!("real / complex, len {len}"),
            );
        }
        // 2^600 (1 + i) / 2 = 2^599 (1 + i)
        let out = divide(&huge_complex, &tensor(0, 2.0_f64));
        for value in each_c64(&out) {
            assert_close64(
                *value,
                Complex64::new(2f64.powi(599), 2f64.powi(599)),
                &format!("complex / real, len {len}"),
            );
        }
        // Complex / real is componentwise: 2^100 (1 + i) / 2^100 = 1 + i in C32,
        // although 2^100 squared overflows f32.
        let out = divide(
            &tensor(len, Complex32::new(2f32.powi(100), 2f32.powi(100))),
            &tensor(0, 2f32.powi(100)),
        );
        for value in each_c32(&out) {
            assert_close32(
                *value,
                Complex32::new(1.0, 1.0),
                &format!("c32 / real, len {len}"),
            );
        }
        // 2 / (2^70 (1 + i)) = 2^-70 (1 - i) in C32.
        let out = divide(
            &tensor(0, 2.0_f32),
            &tensor(len, Complex32::new(2f32.powi(70), 2f32.powi(70))),
        );
        for value in each_c32(&out) {
            assert_close32(
                *value,
                Complex32::new(2f32.powi(-70), -2f32.powi(-70)),
                &format!("real / c32, len {len}"),
            );
        }
    }
}
