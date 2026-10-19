//! Accuracy of the CPU `erf` kernel against a high-precision reference.

use super::*;
use tenferro_tensor::{TensorRead, TensorView};

/// `(x, erf(x) in f64, erf(x as f32))` from mpmath at 200 bits, rounded to f64.
///
/// The third column evaluates erf at the f32-rounded input, so it is the
/// reference for the f32 kernel after rounding to f32. Generated with
/// `mpmath.erf` (prec = 200) over the core interval, both tails, the
/// `erf(x) -> +-1` saturation region and subnormal inputs.
const ERF_REFERENCE: &[(f64, f64, f64)] = &[
    (5e-324_f64, 5e-324_f64, 0.0_f64),
    (1e-300_f64, 1.1283791670955126e-300_f64, 0.0_f64),
    (
        1e-20_f64,
        1.1283791670955125e-20_f64,
        1.1283791312869893e-20_f64,
    ),
    (
        1e-08_f64,
        1.1283791670955126e-08_f64,
        1.1283791602378209e-08_f64,
    ),
    (
        0.001_f64,
        0.0011283787909692365_f64,
        0.0011283788445643173_f64,
    ),
    (0.1_f64, 0.1124629160182849_f64, 0.11246291768297051_f64),
    (0.25_f64, 0.27632639016823696_f64, 0.27632639016823696_f64),
    (0.5_f64, 0.5204998778130465_f64, 0.5204998778130465_f64),
    (0.75_f64, 0.7111556336535151_f64, 0.7111556336535151_f64),
    (0.84375_f64, 0.7672256612323416_f64, 0.7672256612323416_f64),
    (1.0_f64, 0.8427007929497149_f64, 0.8427007929497149_f64),
    (1.25_f64, 0.9229001282564583_f64, 0.9229001282564583_f64),
    (1.5_f64, 0.9661051464753108_f64, 0.9661051464753108_f64),
    (2.0_f64, 0.9953222650189527_f64, 0.9953222650189527_f64),
    (2.5_f64, 0.999593047982555_f64, 0.999593047982555_f64),
    (3.0_f64, 0.9999779095030014_f64, 0.9999779095030014_f64),
    (3.5_f64, 0.9999992569016276_f64, 0.9999992569016276_f64),
    (4.0_f64, 0.9999999845827421_f64, 0.9999999845827421_f64),
    (5.0_f64, 0.9999999999984626_f64, 0.9999999999984626_f64),
    (5.9_f64, 0.9999999999999999_f64, 0.9999999999999999_f64),
    (6.0_f64, 1.0_f64, 1.0_f64),
    (8.0_f64, 1.0_f64, 1.0_f64),
    (10.0_f64, 1.0_f64, 1.0_f64),
    (26.0_f64, 1.0_f64, 1.0_f64),
    (30.0_f64, 1.0_f64, 1.0_f64),
    (-5e-324_f64, -5e-324_f64, 0.0_f64),
    (-1e-300_f64, -1.1283791670955126e-300_f64, 0.0_f64),
    (
        -1e-20_f64,
        -1.1283791670955125e-20_f64,
        -1.1283791312869893e-20_f64,
    ),
    (
        -1e-08_f64,
        -1.1283791670955126e-08_f64,
        -1.1283791602378209e-08_f64,
    ),
    (
        -0.001_f64,
        -0.0011283787909692365_f64,
        -0.0011283788445643173_f64,
    ),
    (-0.1_f64, -0.1124629160182849_f64, -0.11246291768297051_f64),
    (
        -0.25_f64,
        -0.27632639016823696_f64,
        -0.27632639016823696_f64,
    ),
    (-0.5_f64, -0.5204998778130465_f64, -0.5204998778130465_f64),
    (-0.75_f64, -0.7111556336535151_f64, -0.7111556336535151_f64),
    (
        -0.84375_f64,
        -0.7672256612323416_f64,
        -0.7672256612323416_f64,
    ),
    (-1.0_f64, -0.8427007929497149_f64, -0.8427007929497149_f64),
    (-1.25_f64, -0.9229001282564583_f64, -0.9229001282564583_f64),
    (-1.5_f64, -0.9661051464753108_f64, -0.9661051464753108_f64),
    (-2.0_f64, -0.9953222650189527_f64, -0.9953222650189527_f64),
    (-2.5_f64, -0.999593047982555_f64, -0.999593047982555_f64),
    (-3.0_f64, -0.9999779095030014_f64, -0.9999779095030014_f64),
    (-3.5_f64, -0.9999992569016276_f64, -0.9999992569016276_f64),
    (-4.0_f64, -0.9999999845827421_f64, -0.9999999845827421_f64),
    (-5.0_f64, -0.9999999999984626_f64, -0.9999999999984626_f64),
    (-5.9_f64, -0.9999999999999999_f64, -0.9999999999999999_f64),
    (-6.0_f64, -1.0_f64, -1.0_f64),
    (-8.0_f64, -1.0_f64, -1.0_f64),
    (-10.0_f64, -1.0_f64, -1.0_f64),
    (-26.0_f64, -1.0_f64, -1.0_f64),
    (-30.0_f64, -1.0_f64, -1.0_f64),
];

fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let key = |v: f64| {
        let bits = v.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

fn ulp_distance_f32(a: f32, b: f32) -> u64 {
    if a == b {
        return 0;
    }
    let key = |v: f32| {
        let bits = v.to_bits() as i32 as i64;
        if bits < 0 {
            i32::MIN as i64 - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn erf_f64_matches_high_precision_reference_within_one_ulp() {
    let xs: Vec<f64> = ERF_REFERENCE.iter().map(|row| row.0).collect();
    let input = Tensor::from_vec_col_major(vec![xs.len()], xs).unwrap();
    let out = crate::analytic::erf(&input).unwrap();
    for (got, row) in out.as_slice::<f64>().unwrap().iter().zip(ERF_REFERENCE) {
        let ulps = ulp_distance_f64(*got, row.1);
        assert!(
            ulps <= 1,
            "erf({}) = {got}, reference {} ({ulps} ulp)",
            row.0,
            row.1
        );
    }
}

#[test]
fn erf_f32_matches_high_precision_reference_within_one_ulp() {
    let xs: Vec<f32> = ERF_REFERENCE.iter().map(|row| row.0 as f32).collect();
    let input = Tensor::from_vec_col_major(vec![xs.len()], xs).unwrap();
    let out = crate::analytic::erf(&input).unwrap();
    for (got, row) in out.as_slice::<f32>().unwrap().iter().zip(ERF_REFERENCE) {
        let reference = row.2 as f32;
        let ulps = ulp_distance_f32(*got, reference);
        assert!(
            ulps <= 1,
            "erff({}) = {got}, reference {reference} ({ulps} ulp)",
            row.0
        );
    }
}

#[test]
fn erf_special_values_follow_ieee_limits() {
    let input = Tensor::from_vec_col_major(
        vec![5],
        vec![0.0_f64, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
    )
    .unwrap();
    let out = crate::analytic::erf(&input).unwrap();
    let out = out.as_slice::<f64>().unwrap();
    assert_eq!(out[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(out[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(out[2], 1.0);
    assert_eq!(out[3], -1.0);
    assert!(out[4].is_nan());

    let input = Tensor::from_vec_col_major(
        vec![5],
        vec![0.0_f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    )
    .unwrap();
    let out = crate::analytic::erf(&input).unwrap();
    let out = out.as_slice::<f32>().unwrap();
    assert_eq!(out[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(out[1].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(out[2], 1.0);
    assert_eq!(out[3], -1.0);
    assert!(out[4].is_nan());
}

#[test]
fn erf_session_path_reads_strided_views_and_rejects_non_real_dtypes() {
    let mut backend = crate::CpuEngine::serial();
    let base =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.5, 1.0, 2.0, 3.0]).unwrap();
    let transposed = base.as_view().transpose_view([1, 0]).unwrap();
    let out = crate::test_support::on_engine(&mut backend, |session| {
            session.erf_read(TensorRead::from_view(TensorView::F64(transposed)))
        })
        .unwrap()
        .unwrap();
    let out = out.as_slice::<f64>().unwrap();
    // Column-major [[0.5, 2.0], [1.0, 3.0]] transposed reads 0.5, 2.0, 1.0, 3.0.
    assert_eq!(out[1], libm::erf(2.0));
    assert_eq!(out[2], libm::erf(1.0));

    for input in [
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex64::new(0.5, 0.0)]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex32::new(0.5, 0.0)]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap(),
    ] {
        let err = crate::test_support::on_engine(&mut backend, |session| session.erf_read(TensorRead::from_tensor(&input)))
            .unwrap()
            .unwrap_err();
        assert!(
            matches!(err, crate::Error::UnsupportedDType { op: "erf", .. }),
            "{err:?}"
        );
    }
}
