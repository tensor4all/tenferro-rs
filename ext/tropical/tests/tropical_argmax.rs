use tenferro_ext_tropical::cpu::{tropical_gemm_with_argmax, TropicalGemmKind};
use tenferro_tensor::Error;

#[test]
fn maxplus_gemm_returns_values_and_first_winner_indices() {
    let a = vec![10.0, 0.0, 1.0, 5.0]; // shape [2, 2], column-major
    let b = vec![1.0, 10.0, 0.0, 1.0]; // shape [2, 2], column-major

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(out.values, vec![11.0, 15.0, 10.0, 6.0]);
    assert_eq!(out.argmax, vec![0, 1, 0, 1]);
}

#[test]
fn minplus_gemm_returns_values_and_first_winner_indices() {
    let a = vec![1.0, 4.0, 3.0, 2.0];
    let b = vec![5.0, 6.0, 7.0, 1.0];

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MinPlus, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(out.values, vec![6.0, 8.0, 4.0, 3.0]);
    assert_eq!(out.argmax, vec![0, 1, 1, 1]);
}

#[test]
fn ties_keep_first_winner_index() {
    let a = vec![1.0, 1.0]; // shape [1, 2], column-major
    let b = vec![2.0, 2.0]; // shape [2, 1], column-major

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 1, 2, &b, 1).unwrap();

    assert_eq!(out.values, vec![3.0]);
    assert_eq!(out.argmax, vec![0]);
}

#[test]
fn invalid_dimensions_return_invalid_config() {
    let err = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &[1.0_f64], 2, 2, &[1.0], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_gemm_with_argmax",
            ..
        }
    ));

    let err = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &[] as &[f64], 1, 0, &[], 1)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_gemm_with_argmax",
            ..
        }
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn oversized_contracting_dimension_returns_invalid_config_before_reading_inputs() {
    let too_large_k = u32::MAX as usize + 2;

    let err = tropical_gemm_with_argmax(
        TropicalGemmKind::MaxPlus,
        &[] as &[f64],
        1,
        too_large_k,
        &[],
        1,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tropical_gemm_with_argmax",
            ..
        }
    ));
}
