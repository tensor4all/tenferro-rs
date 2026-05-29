use tenferro_ext_tropical::{
    cpu::{tropical_gemm_with_argmax, tropical_gemm_with_argmax_generic, TropicalGemmKind},
    TropicalKind,
};
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
fn rectangular_column_major_shape_returns_values_and_argmax() {
    let a = vec![1.0, 4.0, 10.0, 0.0, 2.0, 6.0]; // shape [2, 3]
    let b = vec![
        0.0, 1.0, 2.0, 5.0, 0.0, 1.0, -10.0, 20.0, 0.0, 3.0, 3.0, 3.0,
    ]; // shape [3, 4]

    let out = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 3, &b, 4).unwrap();

    assert_eq!(
        out.values,
        vec![11.0, 8.0, 10.0, 9.0, 30.0, 20.0, 13.0, 9.0]
    );
    assert_eq!(out.argmax, vec![1, 2, 1, 0, 1, 1, 1, 2]);
}

#[test]
fn generic_gemm_returns_current_fallback_semantics() {
    let out = tropical_gemm_with_argmax_generic(
        TropicalGemmKind::MaxPlus,
        &[0.0_f64, f64::NAN],
        1,
        2,
        &[2.0_f64, 3.0],
        1,
    )
    .unwrap();

    assert_eq!(out.values, vec![2.0]);
    assert_eq!(out.argmax, vec![0]);
}

#[cfg(feature = "tropical-gemm")]
#[test]
fn tropical_gemm_feature_matches_generic_for_f64_maxplus_rectangular_shape() {
    let a = vec![1.0_f64, 4.0, 10.0, 0.0, 2.0, 6.0]; // shape [2, 3]
    let b = vec![
        0.0, 1.0, 2.0, 5.0, 0.0, 1.0, -10.0, 20.0, 0.0, 3.0, 3.0, 3.0,
    ]; // shape [3, 4]

    let actual = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 2, 3, &b, 4).unwrap();
    let expected =
        tropical_gemm_with_argmax_generic(TropicalGemmKind::MaxPlus, &a, 2, 3, &b, 4).unwrap();

    assert_eq!(actual, expected);
}

#[cfg(feature = "tropical-gemm")]
#[test]
fn tropical_gemm_feature_matches_generic_for_f32_minplus_rectangular_shape() {
    let a = vec![3.0_f32, 1.0, 5.0, 2.0, 7.0, 4.0]; // shape [3, 2]
    let b = vec![2.0_f32, 9.0, 0.5, 8.0, 1.0, 6.0]; // shape [2, 3]

    let actual = tropical_gemm_with_argmax(TropicalGemmKind::MinPlus, &a, 3, 2, &b, 3).unwrap();
    let expected =
        tropical_gemm_with_argmax_generic(TropicalGemmKind::MinPlus, &a, 3, 2, &b, 3).unwrap();

    assert_eq!(actual, expected);
}

#[cfg(feature = "tropical-gemm")]
#[test]
fn tropical_gemm_feature_preserves_first_winner_ties() {
    let a = vec![1.0_f64, 1.0]; // shape [1, 2], column-major
    let b = vec![2.0_f64, 2.0]; // shape [2, 1], column-major

    let actual = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 1, 2, &b, 1).unwrap();
    let expected =
        tropical_gemm_with_argmax_generic(TropicalGemmKind::MaxPlus, &a, 1, 2, &b, 1).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.argmax, vec![0]);
}

#[cfg(feature = "tropical-gemm")]
#[test]
fn tropical_gemm_feature_falls_back_when_infinities_can_make_nan_products() {
    let a = vec![0.0_f64, f64::INFINITY]; // shape [1, 2], column-major
    let b = vec![1.0_f64, f64::NEG_INFINITY]; // shape [2, 1], column-major

    let actual = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &a, 1, 2, &b, 1).unwrap();
    let expected =
        tropical_gemm_with_argmax_generic(TropicalGemmKind::MaxPlus, &a, 1, 2, &b, 1).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.values, vec![1.0]);
    assert_eq!(actual.argmax, vec![0]);
}

#[test]
fn nan_products_are_ignored_with_identity_when_all_nan() {
    let max = tropical_gemm_with_argmax(
        TropicalGemmKind::MaxPlus,
        &[f64::NAN, 1.0, f64::NEG_INFINITY],
        1,
        3,
        &[0.0, 0.0, 0.0],
        1,
    )
    .unwrap();
    assert_eq!(max.values, vec![1.0]);
    assert_eq!(max.argmax, vec![1]);

    let min = tropical_gemm_with_argmax(
        TropicalGemmKind::MinPlus,
        &[f64::NAN, 1.0, f64::INFINITY],
        1,
        3,
        &[0.0, 0.0, 0.0],
        1,
    )
    .unwrap();
    assert_eq!(min.values, vec![1.0]);
    assert_eq!(min.argmax, vec![1]);

    let all_nan =
        tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &[f64::NAN], 1, 1, &[0.0], 1).unwrap();
    assert_eq!(all_nan.values, vec![f64::NEG_INFINITY]);
    assert_eq!(all_nan.argmax, vec![0]);
}

#[test]
fn gemm_kind_accepts_crate_level_tropical_kind() {
    assert_eq!(
        TropicalGemmKind::from(TropicalKind::MaxPlus),
        TropicalGemmKind::MaxPlus
    );
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

#[test]
fn zero_contracting_dimension_is_allowed_only_for_empty_outputs() {
    let empty =
        tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &[] as &[f64], 0, 0, &[], 4).unwrap();
    assert!(empty.values.is_empty());
    assert!(empty.argmax.is_empty());

    let err = tropical_gemm_with_argmax(TropicalGemmKind::MaxPlus, &[] as &[f64], 2, 0, &[], 4)
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
