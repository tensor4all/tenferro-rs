use num_complex::{Complex32, Complex64};

use super::*;

macro_rules! float_singular_tests {
    ($mod_name:ident, $t:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn f64_nonsquare_tall_nonsingular() {
                let t =
                    TypedTensor::<f64>::from_vec(vec![3, 2], vec![2.0, 1.0, 0.0, 0.0, 5.0, 4.0]);
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn zero_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![0.0 as $t, 1.0 as $t, 1.0 as $t, 0.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn nan_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![<$t>::NAN, 1.0 as $t, 0.0 as $t, 1.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn inf_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![<$t>::INFINITY, 1.0 as $t, 0.0 as $t, 1.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn neg_inf_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![<$t>::NEG_INFINITY, 1.0 as $t, 0.0 as $t, 1.0 as $t],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn single_element_singular() {
                let t = TypedTensor::<$t>::from_vec(vec![1, 1], vec![0.0 as $t]);
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn single_element_nonsingular() {
                let t = TypedTensor::<$t>::from_vec(vec![1, 1], vec![5.0 as $t]);
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn batched_singular() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2, 2],
                    vec![
                        1.0 as $t, 0.0 as $t, 0.0 as $t, 2.0 as $t, 0.0 as $t, 0.0 as $t,
                        0.0 as $t, 4.0 as $t,
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn batched_nonsingular() {
                let t = TypedTensor::<$t>::from_vec(
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

            #[test]
            fn nonsingular() {
                let t = TypedTensor::<$t>::from_vec(
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
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn nan_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![
                        <$t>::new(<$float>::NAN, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn inf_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![
                        <$t>::new(1.0 as $float, <$float>::INFINITY),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn neg_inf_diagonal() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![2, 2],
                    vec![
                        <$t>::new(<$float>::NEG_INFINITY, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(0.0 as $float, 0.0 as $float),
                        <$t>::new(1.0 as $float, 0.0 as $float),
                    ],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn single_element_singular() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![1, 1],
                    vec![<$t>::new(0.0 as $float, 0.0 as $float)],
                );
                let err = check_singular_diagonal(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn single_element_nonsingular() {
                let t = TypedTensor::<$t>::from_vec(
                    vec![1, 1],
                    vec![<$t>::new(5.0 as $float, 0.0 as $float)],
                );
                assert!(check_singular_diagonal(&t).is_ok());
            }

            #[test]
            fn batched_singular() {
                let t = TypedTensor::<$t>::from_vec(
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
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn batched_nonsingular() {
                let t = TypedTensor::<$t>::from_vec(
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
fn f64_nonsquare_tall_nonsingular() {
    let t = TypedTensor::<f64>::from_vec(vec![3, 2], vec![2.0, 1.0, 0.0, 0.0, 5.0, 4.0]);
    assert!(check_singular_diagonal(&t).is_ok());
}

#[test]
fn f64_nonsquare_tall_singular() {
    let t = TypedTensor::<f64>::from_vec(vec![3, 2], vec![0.0, 1.0, 0.0, 0.0, 0.0, 4.0]);
    let err = check_singular_diagonal(&t).unwrap_err();
    assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
}

#[test]
fn f64_nonsquare_wide_nonsingular() {
    let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![2.0, 0.0, 1.0, 3.0, 0.0, 4.0]);
    assert!(check_singular_diagonal(&t).is_ok());
}

#[test]
fn f64_nonsquare_wide_singular() {
    let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![0.0, 0.0, 1.0, 3.0, 0.0, 4.0]);
    let err = check_singular_diagonal(&t).unwrap_err();
    assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
}

#[test]
fn f64_batched_error_includes_batch_index() {
    let t =
        TypedTensor::<f64>::from_vec(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = match &err {
        Error::BackendFailure { message, .. } => message.clone(),
        _ => unreachable!(),
    };
    assert!(
        msg.contains("batch 1"),
        "expected batch index in error message, got: {msg}"
    );
}

#[test]
fn f64_unbatched_error_omits_batch_index() {
    let t = TypedTensor::<f64>::from_vec(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]);
    let err = check_singular_diagonal(&t).unwrap_err();
    let msg = match &err {
        Error::BackendFailure { message, .. } => message.clone(),
        _ => unreachable!(),
    };
    assert!(
        !msg.contains("batch"),
        "unbatched error should not mention batch, got: {msg}"
    );
    assert!(
        msg.contains("position"),
        "expected diagonal position in error message, got: {msg}"
    );
}

macro_rules! validate_nonsingular_u_test {
    ($mod_name:ident, $variant:ident, $inner:ty) => {
        mod $mod_name {
            use num_traits::{One, Zero};

            use super::*;

            #[test]
            fn singular() {
                let t = Tensor::$variant(TypedTensor::<$inner>::from_vec(
                    vec![2, 2],
                    vec![
                        <$inner>::zero(),
                        <$inner>::one(),
                        <$inner>::one(),
                        <$inner>::zero(),
                    ],
                ));
                let err = validate_nonsingular_u(&t).unwrap_err();
                assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
            }

            #[test]
            fn nonsingular() {
                let t = Tensor::$variant(TypedTensor::<$inner>::from_vec(
                    vec![2, 2],
                    vec![
                        <$inner>::one(),
                        <$inner>::zero(),
                        <$inner>::zero(),
                        <$inner>::one() + <$inner>::one(),
                    ],
                ));
                assert!(validate_nonsingular_u(&t).is_ok());
            }
        }
    };
}

validate_nonsingular_u_test!(validate_f32, F32, f32);
validate_nonsingular_u_test!(validate_f64, F64, f64);
validate_nonsingular_u_test!(validate_c32, C32, Complex32);
validate_nonsingular_u_test!(validate_c64, C64, Complex64);
