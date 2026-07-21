use num_complex::{Complex32, Complex64};

use super::*;

#[test]
fn checked_convert_follows_dtype_promotion_lattice() {
    assert!(can_convert_dtype(DType::F32, DType::F64));
    assert!(can_convert_dtype(DType::F64, DType::C64));
    assert!(can_convert_dtype(DType::Bool, DType::I64));

    assert!(!can_convert_dtype(DType::F64, DType::F32));
    assert!(!can_convert_dtype(DType::F64, DType::I32));
    assert!(!can_convert_dtype(DType::C64, DType::F64));
    assert!(!can_convert_dtype(DType::I32, DType::Bool));
}

#[test]
fn validate_convert_dtype_reports_typed_error() {
    let err = validate_convert_dtype("convert", DType::C64, DType::I32).unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedDTypeConversion {
            op: "convert",
            from: DType::C64,
            to: DType::I32,
            ..
        }
    ));
}

macro_rules! float_singular_tests {
    ($mod_name:ident, $t:ty) => {
        mod $mod_name {
            use super::*;

            fn tensor(shape: Vec<usize>, data: Vec<$t>) -> TypedTensor<$t> {
                TypedTensor::<$t>::from_vec_col_major(shape, data).unwrap()
            }

            #[test]
            fn nonsquare_tall_nonsingular() {
                let t = tensor(
                    vec![3, 2],
                    vec![
                        2.0 as $t, 1.0 as $t, 0.0 as $t, 0.0 as $t, 5.0 as $t, 4.0 as $t,
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn nonsquare_tall_singular() {
                let t = tensor(
                    vec![3, 2],
                    vec![
                        0.0 as $t, 1.0 as $t, 0.0 as $t, 0.0 as $t, 0.0 as $t, 4.0 as $t,
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nonsquare_wide_nonsingular() {
                let t = tensor(
                    vec![2, 3],
                    vec![
                        2.0 as $t, 0.0 as $t, 1.0 as $t, 3.0 as $t, 0.0 as $t, 4.0 as $t,
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn nonsquare_wide_singular() {
                let t = tensor(
                    vec![2, 3],
                    vec![
                        0.0 as $t, 0.0 as $t, 1.0 as $t, 3.0 as $t, 0.0 as $t, 4.0 as $t,
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn zero_diagonal() {
                let t = tensor(vec![2, 2], vec![0.0 as $t, 1.0 as $t, 1.0 as $t, 0.0 as $t]);
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nan_diagonal() {
                let t = tensor(vec![2, 2], vec![<$t>::NAN, 1.0 as $t, 0.0 as $t, 1.0 as $t]);
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn inf_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![<$t>::INFINITY, 1.0 as $t, 0.0 as $t, 1.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn neg_inf_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![<$t>::NEG_INFINITY, 1.0 as $t, 0.0 as $t, 1.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn single_element_singular() {
                let t = tensor(vec![1, 1], vec![0.0 as $t]);
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn single_element_nonsingular() {
                let t = tensor(vec![1, 1], vec![5.0 as $t]);
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn batched_singular() {
                let t = tensor(
                    vec![2, 2, 2],
                    vec![
                        1.0 as $t, 0.0 as $t, 0.0 as $t, 2.0 as $t, 0.0 as $t, 0.0 as $t,
                        0.0 as $t, 4.0 as $t,
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn batched_nonsingular() {
                let t = tensor(
                    vec![2, 2, 2],
                    vec![
                        1.0 as $t, 0.0 as $t, 0.0 as $t, 2.0 as $t, 3.0 as $t, 0.0 as $t,
                        0.0 as $t, 4.0 as $t,
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }
        }
    };
}

macro_rules! complex_singular_tests {
    ($mod_name:ident, $t:ty, $float:ty) => {
        mod $mod_name {
            use super::*;

            fn tensor(shape: Vec<usize>, data: Vec<$t>) -> TypedTensor<$t> {
                TypedTensor::<$t>::from_vec_col_major(shape, data).unwrap()
            }

            #[test]
            fn nonsquare_tall_nonsingular() {
                let t = tensor(
                    vec![3, 2],
                    vec![
                        <$t>::new(2.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(5.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn nonsquare_tall_singular() {
                let t = tensor(
                    vec![3, 2],
                    vec![
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nonsquare_wide_nonsingular() {
                let t = tensor(
                    vec![2, 3],
                    vec![
                        <$t>::new(2.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(3.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn nonsquare_wide_singular() {
                let t = tensor(
                    vec![2, 3],
                    vec![
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(3.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nonsingular() {
                let t = tensor(
                    vec![2, 2],
                    vec![
                        <$t>::new(2.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(3.0 as $float, 0.0 as $float),
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn zero_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nan_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![
                        <$t>::new(<$float>::NAN, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn inf_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![
                        <$t>::new(1.0 as $float, <$float>::INFINITY),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn neg_inf_diagonal() {
                let t = tensor(
                    vec![2, 2],
                    vec![
                        <$t>::new(<$float>::NEG_INFINITY, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn single_element_singular() {
                let t = tensor(vec![1, 1], vec![<$t>::new(0.0 as $float, 0.0 as $float)]);
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn single_element_nonsingular() {
                let t = tensor(vec![1, 1], vec![<$t>::new(5.0 as $float, 0.0 as $float)]);
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn batched_singular() {
                let t = tensor(
                    vec![2, 2, 2],
                    vec![
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(2.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn batched_nonsingular() {
                let t = tensor(
                    vec![2, 2, 2],
                    vec![
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(2.0 as $float, 0.0 as $float),
                        <$t>::new(3.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(4.0 as $float, 0.0 as $float),
                    ],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }
        }
    };
}

float_singular_tests!(f32_tests, f32);
float_singular_tests!(f64_tests, f64);
complex_singular_tests!(c32_tests, Complex32, f32);
complex_singular_tests!(c64_tests, Complex64, f64);

#[test]
fn rank_less_than_two_returns_error_instead_of_panicking() {
    for shape in [Vec::new(), vec![3]] {
        let data = if shape.is_empty() {
            vec![1.0]
        } else {
            vec![1.0, 2.0, 3.0]
        };
        let t = TypedTensor::<f64>::from_vec_col_major(shape.clone(), data).unwrap();
        let err = check_singular_diagonal(&t).unwrap_err();
        assert!(matches!(
            err,
            Error::Validation {
                op: "solve",
                source: tenferro_tensor_core::ValidationError::RankMismatch {
                    expected: 2,
                    actual,
                },
            } if actual == shape.len()
        ));
    }
}

#[test]
fn f64_batched_error_includes_batch_index_and_position() {
    let t = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("batch 1"),
        "expected batch index in error message, got: {msg}"
    );
    assert!(
        msg.contains("position [1,1]"),
        "expected exact diagonal position in error message, got: {msg}"
    );
}

#[test]
fn singular_diagonal_uses_checked_shape_products_before_indexing() {
    let source = include_str!("mod.rs");
    assert!(
        !source.contains("t.shape()[2..].iter().product()")
            && !source.contains("let slice_size = rows * cols"),
        "singular diagonal validation must not use unchecked shape products before indexing"
    );
}

#[test]
fn f64_unbatched_error_omits_batch_index_and_includes_position() {
    let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]).unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        !msg.contains("batch"),
        "unbatched error should not mention batch, got: {msg}"
    );
    assert!(
        msg.contains("position [0,0]"),
        "expected exact diagonal position in error message, got: {msg}"
    );
}

#[test]
fn f64_unbatched_error_second_diagonal_reports_correct_position() {
    let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 1.0, 1.0, 0.0]).unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("position [1,1]"),
        "expected second diagonal position in error message, got: {msg}"
    );
}

#[test]
fn f64_batched_error_first_batch_reports_correct_position() {
    let t = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2, 2],
        vec![0.0, 1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("batch 0"),
        "expected batch 0 in error message, got: {msg}"
    );
    assert!(
        msg.contains("position [0,0]"),
        "expected exact diagonal position in error message, got: {msg}"
    );
}

#[test]
fn f32_unbatched_error_includes_exact_position() {
    let t =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![0.0f32, 1.0, 1.0, 0.0]).unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("position [0,0]"),
        "expected exact diagonal position in error message, got: {msg}"
    );
    assert!(
        !msg.contains("batch"),
        "unbatched f32 error should not mention batch, got: {msg}"
    );
}

#[test]
fn c64_unbatched_error_includes_exact_position() {
    let t = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("position [0,0]"),
        "expected exact position in c64 error, got: {msg}"
    );
    assert!(
        !msg.contains("batch"),
        "unbatched c64 error should not mention batch, got: {msg}"
    );
}

#[test]
fn c32_unbatched_error_includes_exact_position() {
    let t = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = std::error::Error::source(&err).unwrap().to_string();
    assert!(
        msg.contains("position [0,0]"),
        "expected exact position in c32 error, got: {msg}"
    );
    assert!(
        !msg.contains("batch"),
        "unbatched c32 error should not mention batch, got: {msg}"
    );
}

#[test]
fn diagonal_validation_preserves_numerical_source_and_unsupported_dtype() {
    let singular = TypedTensor::<f64>::from_vec_col_major(vec![1, 1], vec![0.0]).unwrap();
    let singular_error = check_singular_diagonal(&singular).unwrap_err();
    assert_eq!(singular_error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        std::error::Error::source(&singular_error)
            .and_then(|source| source.downcast_ref::<DiagonalError>()),
        Some(DiagonalError::SingularOrNonFinite { index: 0 })
    ));

    let integer = Tensor::I32(TypedTensor::from_vec_col_major(vec![1, 1], vec![1]).unwrap());
    let unsupported_error = validate_nonsingular_u(&integer).unwrap_err();
    assert_eq!(unsupported_error.kind(), ErrorKind::Unsupported);
    assert!(matches!(
        std::error::Error::source(&unsupported_error)
            .and_then(|source| source.downcast_ref::<DiagonalError>()),
        Some(DiagonalError::UnsupportedDType { dtype: DType::I32 })
    ));
}

macro_rules! validate_nonsingular_u_test {
    ($mod_name:ident, $variant:ident, $inner:ty) => {
        mod $mod_name {
            use num_traits::{One, Zero};

            use super::*;

            #[test]
            fn singular() {
                let t = Tensor::$variant(
                    TypedTensor::<$inner>::from_vec_col_major(
                        vec![2, 2],
                        vec![
                            <$inner>::zero(),
                            <$inner>::one(),
                            <$inner>::one(),
                            <$inner>::zero(),
                        ],
                    )
                    .unwrap(),
                );
                let err = validate_nonsingular_u(&t).unwrap_err();
                assert!(matches!(
                    err,
                    Error::Extension {
                        kind: ErrorKind::NumericalFailure,
                        ..
                    }
                ));
            }

            #[test]
            fn nonsingular() {
                let t = Tensor::$variant(
                    TypedTensor::<$inner>::from_vec_col_major(
                        vec![2, 2],
                        vec![
                            <$inner>::one(),
                            <$inner>::zero(),
                            <$inner>::zero(),
                            <$inner>::one() + <$inner>::one(),
                        ],
                    )
                    .unwrap(),
                );
                assert!(validate_nonsingular_u(&t).is_ok());
            }
        }
    };
}

validate_nonsingular_u_test!(validate_f32, F32, f32);
validate_nonsingular_u_test!(validate_f64, F64, f64);
validate_nonsingular_u_test!(validate_c32, C32, Complex32);
validate_nonsingular_u_test!(validate_c64, C64, Complex64);

#[test]
fn c32_tiny_nonzero_complex_diagonal_is_nonsingular() {
    // What: a representable complex pivot remains nonzero even when its squared norm underflows.
    let tiny = 2.0_f32.powi(-80);
    let tensor =
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex32::new(tiny, 0.0)]).unwrap();

    assert!(check_singular_diagonal(&tensor).is_ok());
}

#[test]
fn c64_tiny_nonzero_complex_diagonal_is_nonsingular() {
    // What: double-precision complex validation does not infer zero from an underflowed norm.
    let tiny = 2.0_f64.powi(-600);
    let tensor =
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(0.0, tiny)]).unwrap();

    assert!(check_singular_diagonal(&tensor).is_ok());
}
