// Run with: cargo test --features cuda -- --ignored
use crate::config::CompareDir;
use crate::cubecl::gpu_available;
use crate::{DType, DeviceKind, GpuBackendKind, Tensor};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::BackendId;
use tenferro_tensor::{
    TensorAnalytic, TensorElementwise, TensorFusion, TensorRead, TensorStructural,
};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, tensor_i32, tensor_i64, upload,
};

fn assert_complex_classes_and_values_match(actual: &Tensor, expected: &Tensor) {
    fn component_matches<T: num_traits::Float + std::fmt::Debug>(actual: T, expected: T) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN component, got {actual:?}");
        } else {
            assert_eq!(actual, expected);
            if expected == T::zero() {
                assert_eq!(
                    actual.is_sign_negative(),
                    expected.is_sign_negative(),
                    "zero sign mismatch: actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }

    match (actual, expected) {
        (Tensor::C32(actual), Tensor::C32(expected)) => {
            for (actual, expected) in actual
                .as_slice()
                .unwrap()
                .iter()
                .zip(expected.as_slice().unwrap())
            {
                component_matches(actual.re, expected.re);
                component_matches(actual.im, expected.im);
            }
        }
        (Tensor::C64(actual), Tensor::C64(expected)) => {
            for (actual, expected) in actual
                .as_slice()
                .unwrap()
                .iter()
                .zip(expected.as_slice().unwrap())
            {
                component_matches(actual.re, expected.re);
                component_matches(actual.im, expected.im);
            }
        }
        _ => panic!("expected matching complex tensor dtypes"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_real_scalar_complex_binary_ops_match_cpu() {
    if !gpu_available() {
        eprintln!("skipping real-scalar complex binary parity test - no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let cases = [
        (
            tensor_f32(vec![], vec![2.0]),
            tensor_c32(
                vec![4],
                vec![
                    Complex32::new(1.0, 2.0),
                    Complex32::new(0.0, -2.0),
                    Complex32::new(0.0, -0.0),
                    Complex32::new(-3.0, 4.0),
                ],
            ),
            DType::C32,
        ),
        (
            tensor_f64(vec![], vec![2.0]),
            tensor_c64(
                vec![4],
                vec![
                    Complex64::new(1.0, 2.0),
                    Complex64::new(0.0, -2.0),
                    Complex64::new(0.0, -0.0),
                    Complex64::new(-3.0, 4.0),
                ],
            ),
            DType::C64,
        ),
    ];

    for (scalar, complex, expected_dtype) in cases {
        let gpu_scalar = upload(&gpu, &scalar);
        let gpu_complex = upload(&gpu, &complex);
        for (case, expected, actual) in [
            (
                "scalar+complex",
                cpu.add(&scalar, &complex),
                gpu.add(&gpu_scalar, &gpu_complex),
            ),
            (
                "complex+scalar",
                cpu.add(&complex, &scalar),
                gpu.add(&gpu_complex, &gpu_scalar),
            ),
            (
                "scalar-complex",
                cpu.sub(&scalar, &complex),
                gpu.sub(&gpu_scalar, &gpu_complex),
            ),
            (
                "complex-scalar",
                cpu.sub(&complex, &scalar),
                gpu.sub(&gpu_complex, &gpu_scalar),
            ),
            (
                "scalar*complex",
                cpu.mul(&scalar, &complex),
                gpu.mul(&gpu_scalar, &gpu_complex),
            ),
            (
                "complex*scalar",
                cpu.mul(&complex, &scalar),
                gpu.mul(&gpu_complex, &gpu_scalar),
            ),
            (
                "scalar/complex",
                cpu.div(&scalar, &complex),
                gpu.div(&gpu_scalar, &gpu_complex),
            ),
            (
                "complex/scalar",
                cpu.div(&complex, &scalar),
                gpu.div(&gpu_complex, &gpu_scalar),
            ),
        ] {
            let expected = expected.unwrap();
            let actual = download(&gpu, &actual.unwrap());
            assert_eq!(actual.dtype(), expected_dtype);
            assert_eq!(actual.shape(), &[4], "unexpected shape for {case}");
            assert_complex_classes_and_values_match(&actual, &expected);
        }

        for (op, result, expected_lhs, expected_rhs) in [
            (
                "pow",
                gpu.pow(&gpu_scalar, &gpu_complex),
                scalar.dtype(),
                complex.dtype(),
            ),
            (
                "pow",
                gpu.pow(&gpu_complex, &gpu_scalar),
                complex.dtype(),
                scalar.dtype(),
            ),
            (
                "rem",
                gpu.rem(&gpu_scalar, &gpu_complex),
                scalar.dtype(),
                complex.dtype(),
            ),
            (
                "rem",
                gpu.rem(&gpu_complex, &gpu_scalar),
                complex.dtype(),
                scalar.dtype(),
            ),
        ] {
            assert!(matches!(
                result,
                Err(crate::Error::DTypeMismatch { op: actual, lhs, rhs })
                    if actual == op && lhs == expected_lhs && rhs == expected_rhs
            ));
        }
    }

    for (scalar, complex) in [
        (
            tensor_f32(vec![], vec![2.0]),
            tensor_c32(vec![1], vec![Complex32::new(1.0e38, 1.0e38)]),
        ),
        (
            tensor_f64(vec![], vec![2.0]),
            tensor_c64(vec![1], vec![Complex64::new(1.0e308, 1.0e308)]),
        ),
    ] {
        let expected = cpu.div(&scalar, &complex).unwrap();
        let actual = gpu
            .div(&upload(&gpu, &scalar), &upload(&gpu, &complex))
            .map(|value| download(&gpu, &value))
            .unwrap();
        assert_complex_classes_and_values_match(&actual, &expected);
    }

    for (scalar, complex) in [
        (
            tensor_f32(vec![], vec![2.0]),
            tensor_c32(
                vec![4],
                vec![
                    Complex32::new(0.0, -0.0),
                    Complex32::new(f32::INFINITY, 1.0),
                    Complex32::new(f32::NAN, 0.0),
                    Complex32::new(0.0, f32::INFINITY),
                ],
            ),
        ),
        (
            tensor_f64(vec![], vec![2.0]),
            tensor_c64(
                vec![4],
                vec![
                    Complex64::new(0.0, -0.0),
                    Complex64::new(f64::INFINITY, 1.0),
                    Complex64::new(f64::NAN, 0.0),
                    Complex64::new(0.0, f64::INFINITY),
                ],
            ),
        ),
    ] {
        let gpu_scalar = upload(&gpu, &scalar);
        let gpu_complex = upload(&gpu, &complex);
        for (expected, actual) in [
            (
                cpu.add(&scalar, &complex),
                gpu.add(&gpu_scalar, &gpu_complex),
            ),
            (
                cpu.add(&complex, &scalar),
                gpu.add(&gpu_complex, &gpu_scalar),
            ),
            (
                cpu.sub(&scalar, &complex),
                gpu.sub(&gpu_scalar, &gpu_complex),
            ),
            (
                cpu.sub(&complex, &scalar),
                gpu.sub(&gpu_complex, &gpu_scalar),
            ),
            (
                cpu.mul(&scalar, &complex),
                gpu.mul(&gpu_scalar, &gpu_complex),
            ),
            (
                cpu.mul(&complex, &scalar),
                gpu.mul(&gpu_complex, &gpu_scalar),
            ),
            (
                cpu.div(&scalar, &complex),
                gpu.div(&gpu_scalar, &gpu_complex),
            ),
            (
                cpu.div(&complex, &scalar),
                gpu.div(&gpu_complex, &gpu_scalar),
            ),
        ] {
            let expected = expected.unwrap();
            let actual = download(&gpu, &actual.unwrap());
            assert_complex_classes_and_values_match(&actual, &expected);
        }
    }

    for (complex, scalar) in [
        (
            tensor_c32(vec![1], vec![Complex32::new(1.0, 1.0)]),
            tensor_f32(vec![], vec![1.0e38]),
        ),
        (
            tensor_c64(vec![1], vec![Complex64::new(1.0, 1.0)]),
            tensor_f64(vec![], vec![1.0e308]),
        ),
    ] {
        let expected = cpu.div(&complex, &scalar).unwrap();
        let actual = gpu
            .div(&upload(&gpu, &complex), &upload(&gpu, &scalar))
            .map(|value| download(&gpu, &value))
            .unwrap();
        assert_complex_classes_and_values_match(&actual, &expected);
    }

    for (lhs, rhs, expected_lhs, expected_rhs) in [
        (
            tensor_f32(vec![2], vec![1.0, 2.0]),
            tensor_c32(vec![2], vec![Complex32::new(1.0, 1.0); 2]),
            DType::F32,
            DType::C32,
        ),
        (
            tensor_f64(vec![2], vec![1.0, 2.0]),
            tensor_c64(vec![2], vec![Complex64::new(1.0, 1.0); 2]),
            DType::F64,
            DType::C64,
        ),
        (
            tensor_f32(vec![], vec![1.0]),
            tensor_c64(vec![2], vec![Complex64::new(1.0, 1.0); 2]),
            DType::F32,
            DType::C64,
        ),
        (
            tensor_f64(vec![], vec![1.0]),
            tensor_c32(vec![2], vec![Complex32::new(1.0, 1.0); 2]),
            DType::F64,
            DType::C32,
        ),
    ] {
        let gpu_lhs = upload(&gpu, &lhs);
        let gpu_rhs = upload(&gpu, &rhs);
        for (op, result) in [
            ("add", gpu.add(&gpu_lhs, &gpu_rhs)),
            ("sub", gpu.sub(&gpu_lhs, &gpu_rhs)),
            ("mul", gpu.mul(&gpu_lhs, &gpu_rhs)),
            ("div", gpu.div(&gpu_lhs, &gpu_rhs)),
        ] {
            assert!(matches!(
                result,
                Err(crate::Error::DTypeMismatch { op: actual, lhs, rhs })
                    if actual == op && lhs == expected_lhs && rhs == expected_rhs
            ));
        }
        for (op, result) in [
            ("add", gpu.add(&gpu_rhs, &gpu_lhs)),
            ("sub", gpu.sub(&gpu_rhs, &gpu_lhs)),
            ("mul", gpu.mul(&gpu_rhs, &gpu_lhs)),
            ("div", gpu.div(&gpu_rhs, &gpu_lhs)),
        ] {
            assert!(matches!(
                result,
                Err(crate::Error::DTypeMismatch { op: actual, lhs, rhs })
                    if actual == op && lhs == expected_rhs && rhs == expected_lhs
            ));
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_scalar_div_rem_pow_match_cpu() {
    if !gpu_available() {
        eprintln!("skipping scalar div/rem/pow parity test - no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let cases = [
        (
            tensor_f32(vec![3], vec![2.0, -4.0, 8.0]),
            tensor_f32(vec![], vec![2.0]),
        ),
        (
            tensor_f64(vec![3], vec![2.0, -4.0, 8.0]),
            tensor_f64(vec![], vec![2.0]),
        ),
        (
            tensor_i32(vec![3], vec![2, 4, 8]),
            tensor_i32(vec![], vec![2]),
        ),
        (
            tensor_i64(vec![3], vec![2, 4, 8]),
            tensor_i64(vec![], vec![2]),
        ),
    ];

    for (tensor, scalar) in cases {
        let gpu_tensor = upload(&gpu, &tensor);
        let gpu_scalar = upload(&gpu, &scalar);
        for (expected, actual) in [
            (
                cpu.div(&scalar, &tensor).unwrap(),
                gpu.div(&gpu_scalar, &gpu_tensor)
                    .map(|value| download(&gpu, &value)),
            ),
            (
                cpu.div(&tensor, &scalar).unwrap(),
                gpu.div(&gpu_tensor, &gpu_scalar)
                    .map(|value| download(&gpu, &value)),
            ),
            (
                cpu.rem(&scalar, &tensor).unwrap(),
                gpu.rem(&gpu_scalar, &gpu_tensor)
                    .map(|value| download(&gpu, &value)),
            ),
            (
                cpu.rem(&tensor, &scalar).unwrap(),
                gpu.rem(&gpu_tensor, &gpu_scalar)
                    .map(|value| download(&gpu, &value)),
            ),
        ] {
            assert_tensor_close(&actual.unwrap(), &expected, 0.0);
        }
    }

    for (tensor, negative_zero, negative_multiple, divisor) in [
        (
            tensor_f32(vec![2], vec![-0.0, -4.0]),
            tensor_f32(vec![], vec![-0.0]),
            tensor_f32(vec![], vec![-4.0]),
            tensor_f32(vec![], vec![2.0]),
        ),
        (
            tensor_f64(vec![2], vec![-0.0, -4.0]),
            tensor_f64(vec![], vec![-0.0]),
            tensor_f64(vec![], vec![-4.0]),
            tensor_f64(vec![], vec![2.0]),
        ),
    ] {
        let gpu_tensor = upload(&gpu, &tensor);
        let gpu_negative_zero = upload(&gpu, &negative_zero);
        let gpu_negative_multiple = upload(&gpu, &negative_multiple);
        let gpu_divisor = upload(&gpu, &divisor);

        let expected = cpu.rem(&tensor, &divisor).unwrap();
        let actual = gpu.rem(&gpu_tensor, &gpu_divisor).unwrap();
        let actual = download(&gpu, &actual);
        assert_float_classes_and_zero_signs_match("scalar rhs rem", &actual, &expected);

        for (scalar, gpu_scalar) in [
            (&negative_zero, &gpu_negative_zero),
            (&negative_multiple, &gpu_negative_multiple),
        ] {
            let expected = cpu.rem(scalar, &tensor).unwrap();
            let actual = gpu.rem(gpu_scalar, &gpu_tensor).unwrap();
            let actual = download(&gpu, &actual);
            assert_float_classes_and_zero_signs_match("scalar lhs rem", &actual, &expected);
        }
    }

    for (dtype, lhs, zero) in [
        (
            DType::I32,
            tensor_i32(vec![2], vec![i32::MIN, 7]),
            tensor_i32(vec![], vec![0]),
        ),
        (
            DType::I64,
            tensor_i64(vec![2], vec![i64::MIN, 7]),
            tensor_i64(vec![], vec![0]),
        ),
    ] {
        let gpu_lhs = upload(&gpu, &lhs);
        let gpu_zero = upload(&gpu, &zero);
        assert!(matches!(
            gpu.div(&gpu_lhs, &gpu_zero),
            Err(crate::Error::DivisionByZero { op: "div", dtype: actual }) if actual == dtype
        ));
        assert!(matches!(
            gpu.rem(&gpu_lhs, &gpu_zero),
            Err(crate::Error::DivisionByZero { op: "rem", dtype: actual }) if actual == dtype
        ));

        let (scalar_one, zero_rhs) = match dtype {
            DType::I32 => (tensor_i32(vec![], vec![1]), tensor_i32(vec![2], vec![1, 0])),
            DType::I64 => (tensor_i64(vec![], vec![1]), tensor_i64(vec![2], vec![1, 0])),
            _ => unreachable!(),
        };
        let gpu_scalar_one = upload(&gpu, &scalar_one);
        let gpu_zero_rhs = upload(&gpu, &zero_rhs);
        assert!(matches!(
            gpu.div(&gpu_scalar_one, &gpu_zero_rhs),
            Err(crate::Error::DivisionByZero { op: "div", dtype: actual }) if actual == dtype
        ));
        assert!(matches!(
            gpu.rem(&gpu_scalar_one, &gpu_zero_rhs),
            Err(crate::Error::DivisionByZero { op: "rem", dtype: actual }) if actual == dtype
        ));

        let minus_one = match dtype {
            DType::I32 => tensor_i32(vec![], vec![-1]),
            DType::I64 => tensor_i64(vec![], vec![-1]),
            _ => unreachable!(),
        };
        let gpu_minus_one = upload(&gpu, &minus_one);
        let expected_div = cpu.div(&lhs, &minus_one).unwrap();
        let expected_rem = cpu.rem(&lhs, &minus_one).unwrap();
        let gpu_div = gpu.div(&gpu_lhs, &gpu_minus_one).unwrap();
        let gpu_rem = gpu.rem(&gpu_lhs, &gpu_minus_one).unwrap();
        let actual_div = download(&gpu, &gpu_div);
        let actual_rem = download(&gpu, &gpu_rem);
        assert_tensor_close(&actual_div, &expected_div, 0.0);
        assert_tensor_close(&actual_rem, &expected_rem, 0.0);

        let (min_scalar, minus_one_rhs) = match dtype {
            DType::I32 => (
                tensor_i32(vec![], vec![i32::MIN]),
                tensor_i32(vec![2], vec![-1, -1]),
            ),
            DType::I64 => (
                tensor_i64(vec![], vec![i64::MIN]),
                tensor_i64(vec![2], vec![-1, -1]),
            ),
            _ => unreachable!(),
        };
        let gpu_min_scalar = upload(&gpu, &min_scalar);
        let gpu_minus_one_rhs = upload(&gpu, &minus_one_rhs);
        let expected_div = cpu.div(&min_scalar, &minus_one_rhs).unwrap();
        let expected_rem = cpu.rem(&min_scalar, &minus_one_rhs).unwrap();
        let gpu_div = gpu.div(&gpu_min_scalar, &gpu_minus_one_rhs).unwrap();
        let gpu_rem = gpu.rem(&gpu_min_scalar, &gpu_minus_one_rhs).unwrap();
        assert_tensor_close(&download(&gpu, &gpu_div), &expected_div, 0.0);
        assert_tensor_close(&download(&gpu, &gpu_rem), &expected_rem, 0.0);
    }

    for (tensor, scalar) in [
        (
            tensor_f32(vec![2], vec![2.0, 3.0]),
            tensor_f32(vec![], vec![2.0]),
        ),
        (
            tensor_f64(vec![2], vec![2.0, 3.0]),
            tensor_f64(vec![], vec![2.0]),
        ),
        (tensor_i32(vec![2], vec![2, 3]), tensor_i32(vec![], vec![2])),
        (tensor_i64(vec![2], vec![2, 3]), tensor_i64(vec![], vec![2])),
    ] {
        let gpu_tensor = upload(&gpu, &tensor);
        let gpu_scalar = upload(&gpu, &scalar);
        for (expected, actual) in [
            (
                cpu.pow(&scalar, &tensor).unwrap(),
                gpu.pow(&gpu_scalar, &gpu_tensor)
                    .map(|value| download(&gpu, &value)),
            ),
            (
                cpu.pow(&tensor, &scalar).unwrap(),
                gpu.pow(&gpu_tensor, &gpu_scalar)
                    .map(|value| download(&gpu, &value)),
            ),
        ] {
            assert_tensor_close(&actual.unwrap(), &expected, 0.0);
        }
    }

    for (empty, scalar) in [
        (tensor_f32(vec![0], vec![]), tensor_f32(vec![], vec![2.0])),
        (tensor_f64(vec![0], vec![]), tensor_f64(vec![], vec![2.0])),
        (tensor_i32(vec![0], vec![]), tensor_i32(vec![], vec![2])),
        (tensor_i64(vec![0], vec![]), tensor_i64(vec![], vec![2])),
    ] {
        let gpu_empty = upload(&gpu, &empty);
        let gpu_scalar = upload(&gpu, &scalar);
        for (expected, actual) in [
            (
                cpu.pow(&scalar, &empty).unwrap(),
                gpu.pow(&gpu_scalar, &gpu_empty).unwrap(),
            ),
            (
                cpu.pow(&empty, &scalar).unwrap(),
                gpu.pow(&gpu_empty, &gpu_scalar).unwrap(),
            ),
        ] {
            assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);
            assert_eq!(actual.shape(), &[0]);
        }
    }

    for (base, exponent) in [
        (
            tensor_i32(vec![2], vec![2, 3]),
            tensor_i32(vec![], vec![-1]),
        ),
        (
            tensor_i64(vec![2], vec![2, 3]),
            tensor_i64(vec![], vec![-1]),
        ),
        (
            tensor_i32(vec![], vec![2]),
            tensor_i32(vec![2], vec![2, -1]),
        ),
        (
            tensor_i64(vec![], vec![2]),
            tensor_i64(vec![2], vec![2, -1]),
        ),
    ] {
        let dtype = base.dtype();
        let gpu_base = upload(&gpu, &base);
        let gpu_exponent = upload(&gpu, &exponent);
        assert!(matches!(
            gpu.pow(&gpu_base, &gpu_exponent),
            Err(crate::Error::NegativeIntegerExponent { op: "pow", dtype: actual })
                if actual == dtype
        ));
    }

    let unequal_lhs = upload(&gpu, &tensor_f32(vec![2], vec![2.0, 3.0]));
    let unequal_rhs = upload(&gpu, &tensor_f32(vec![3], vec![2.0, 3.0, 4.0]));
    assert!(matches!(
        gpu.pow(&unequal_lhs, &unequal_rhs),
        Err(crate::Error::ShapeMismatch { op: "pow", lhs, rhs })
            if lhs == vec![2] && rhs == vec![3]
    ));

    for (tensor, scalar) in [
        (
            tensor_f32(
                vec![6],
                vec![-0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -1.0],
            ),
            tensor_f32(vec![], vec![0.5]),
        ),
        (
            tensor_f64(
                vec![6],
                vec![-0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1.0],
            ),
            tensor_f64(vec![], vec![0.5]),
        ),
    ] {
        let gpu_tensor = upload(&gpu, &tensor);
        let gpu_scalar = upload(&gpu, &scalar);
        for (label, expected, actual) in [
            (
                "scalar exponent pow",
                cpu.pow(&tensor, &scalar).unwrap(),
                gpu.pow(&gpu_tensor, &gpu_scalar).unwrap(),
            ),
            (
                "scalar base pow",
                cpu.pow(&scalar, &tensor).unwrap(),
                gpu.pow(&gpu_scalar, &gpu_tensor).unwrap(),
            ),
        ] {
            assert_float_classes_and_zero_signs_match(label, &download(&gpu, &actual), &expected);
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_complex_abs_matches_cpu() {
    if !gpu_available() {
        eprintln!("skipping test_cubecl_complex_abs_matches_cpu - no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let cases = [
        tensor_c32(
            vec![9],
            vec![
                Complex32::new(3.0, 4.0),
                Complex32::new(5.0, 12.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(f32::MAX / 4.0, f32::MAX / 4.0),
                Complex32::new(f32::MIN_POSITIVE, f32::MIN_POSITIVE),
                Complex32::new(f32::INFINITY, 1.0),
                Complex32::new(1.0, f32::INFINITY),
                Complex32::new(f32::NAN, 1.0),
                Complex32::new(1.0, f32::NAN),
            ],
        ),
        tensor_c64(
            vec![9],
            vec![
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 12.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(f64::MAX / 4.0, f64::MAX / 4.0),
                Complex64::new(f64::MIN_POSITIVE, f64::MIN_POSITIVE),
                Complex64::new(f64::INFINITY, 1.0),
                Complex64::new(1.0, f64::INFINITY),
                Complex64::new(f64::NAN, 1.0),
                Complex64::new(1.0, f64::NAN),
            ],
        ),
    ];

    for input in cases {
        let expected = cpu.abs(&input).unwrap();
        let gpu_input = upload(&gpu, &input);
        let gpu_output = gpu.abs(&gpu_input).unwrap();
        let actual = download(&gpu, &gpu_output);

        assert_eq!(actual.dtype(), expected.dtype());
        assert_float_classes_and_zero_signs_match("abs", &actual, &expected);
        match (&actual, &expected) {
            (Tensor::F32(actual), Tensor::F32(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(&actual[..3], &[5.0, 13.0, 0.0]);
                for (&actual, &expected) in actual[3..5].iter().zip(&expected[3..5]) {
                    assert!(actual.is_finite() && actual > 0.0);
                    assert!((actual / expected - 1.0).abs() <= 2.0 * f32::EPSILON);
                }
            }
            (Tensor::F64(actual), Tensor::F64(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(&actual[..3], &[5.0, 13.0, 0.0]);
                for (&actual, &expected) in actual[3..5].iter().zip(&expected[3..5]) {
                    assert!(actual.is_finite() && actual > 0.0);
                    assert!((actual / expected - 1.0).abs() <= 2.0 * f64::EPSILON);
                }
            }
            _ => panic!("complex abs must produce the matching real dtype"),
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_broadcast_multiply_scalar_operands_match_cpu() {
    if !gpu_available() {
        eprintln!(
            "skipping test_broadcast_multiply_scalar_operands_match_cpu - no CUDA device found"
        );
        return;
    }
    let Ok(mut backend) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(gpu_backend))
    else {
        eprintln!(
            "skipping test_broadcast_multiply_scalar_operands_match_cpu - CUDA runtime could not be initialized"
        );
        return;
    };
    let scalar = tensor_f64(vec![], vec![2.0]);
    let vector = tensor_f64(vec![3], vec![3.0, -4.0, 5.0]);
    let gpu_scalar = upload(&backend, &scalar);
    let gpu_vector = upload(&backend, &vector);

    let lhs_scalar = backend
        .execute_broadcast_multiply(
            TensorRead::from_tensor(&gpu_scalar),
            &[3],
            &[],
            TensorRead::from_tensor(&gpu_vector),
            &[3],
            &[0],
        )
        .unwrap()
        .expect("scalar lhs broadcast multiply should fuse");
    let rhs_scalar = backend
        .execute_broadcast_multiply(
            TensorRead::from_tensor(&gpu_vector),
            &[3],
            &[0],
            TensorRead::from_tensor(&gpu_scalar),
            &[3],
            &[],
        )
        .unwrap()
        .expect("scalar rhs broadcast multiply should fuse");

    let expected = tensor_f64(vec![3], vec![6.0, -8.0, 10.0]);
    assert_tensor_close(&download(&backend, &lhs_scalar), &expected, 1e-12);
    assert_tensor_close(&download(&backend, &rhs_scalar), &expected, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_log1p_small_x_f32_precision() {
    if !gpu_available() {
        eprintln!("skipping test_log1p_small_x_f32_precision — no CUDA device found");
        return;
    }
    let mut backend = super::gpu_backend();
    let x_values = vec![1e-7_f32, 1e-6, 1e-5, 1e-4, 1e-3];
    let cpu_input = super::tensor_f32(vec![x_values.len()], x_values.clone());
    let gpu_input = super::upload(&backend, &cpu_input);

    let gpu_out = backend.log1p(&gpu_input).unwrap();
    let result = super::download(&backend, &gpu_out);
    let result_slice = match result {
        Tensor::F32(t) => t.as_slice().unwrap().to_vec(),
        _ => panic!("expected F32"),
    };

    for (x, got) in x_values.iter().zip(result_slice.iter()) {
        let expected = (*x).ln_1p();
        let rel_err = (got - expected).abs() / expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            rel_err < 1e-6,
            "log1p({x}): expected {expected}, got {got}, rel_err {rel_err}",
        );
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_expm1_small_x_f32_precision() {
    if !gpu_available() {
        eprintln!("skipping test_expm1_small_x_f32_precision — no CUDA device found");
        return;
    }
    let mut backend = super::gpu_backend();
    let x_values = vec![1e-7_f32, 1e-6, 1e-5, 1e-4, 1e-3];
    let cpu_input = super::tensor_f32(vec![x_values.len()], x_values.clone());
    let gpu_input = super::upload(&backend, &cpu_input);

    let gpu_out = backend.expm1(&gpu_input).unwrap();
    let result = super::download(&backend, &gpu_out);
    let result_slice = match result {
        Tensor::F32(t) => t.as_slice().unwrap().to_vec(),
        _ => panic!("expected F32"),
    };

    for (x, got) in x_values.iter().zip(result_slice.iter()) {
        let expected = (*x).exp_m1();
        let rel_err = (got - expected).abs() / expected.abs().max(f32::MIN_POSITIVE);
        assert!(
            rel_err < 1e-6,
            "expm1({x}): expected {expected}, got {got}, rel_err {rel_err}",
        );
    }
}

#[test]
#[ignore]
fn test_cubecl_binary_float_elementwise_matches_cpu() {
    let lhs = tensor_f64(vec![4], vec![1.5, -2.0, 3.0, 4.5]);
    let rhs = tensor_f64(vec![4], vec![0.5, 4.0, 2.0, -1.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);

    let expected = cpu.add(&lhs, &rhs).unwrap();
    let gpu_out = gpu.add(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.mul(&lhs, &rhs).unwrap();
    let gpu_out = gpu.mul(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.div(&lhs, &rhs).unwrap();
    let gpu_out = gpu.div(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.rem(&lhs, &rhs).unwrap();
    let gpu_out = gpu.rem(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.maximum(&lhs, &rhs).unwrap();
    let gpu_out = gpu.maximum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.minimum(&lhs, &rhs).unwrap();
    let gpu_out = gpu.minimum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .pow(
            &tensor_f64(vec![4], vec![1.5, 2.0, 3.0, 4.0]),
            &tensor_f64(vec![4], vec![2.0, 3.0, 0.5, 1.0]),
        )
        .unwrap();
    let gpu_base = upload(&gpu, &tensor_f64(vec![4], vec![1.5, 2.0, 3.0, 4.0]));
    let gpu_exp = upload(&gpu, &tensor_f64(vec![4], vec![2.0, 3.0, 0.5, 1.0]));
    let gpu_out = gpu.pow(&gpu_base, &gpu_exp).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

fn assert_float_classes_and_zero_signs_match(op: &str, actual: &Tensor, expected: &Tensor) {
    match (actual, expected) {
        (Tensor::F32(actual), Tensor::F32(expected)) => {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.n_elements(), expected.n_elements());
            for (index, (actual, expected)) in actual
                .as_slice()
                .unwrap()
                .iter()
                .zip(expected.as_slice().unwrap())
                .enumerate()
            {
                let context = || {
                    format!(
                        "{op} F32 index {index}: actual={actual:?} ({:#010x}), expected={expected:?} ({:#010x})",
                        actual.to_bits(),
                        expected.to_bits()
                    )
                };
                assert_eq!(actual.is_nan(), expected.is_nan(), "{}", context());
                assert_eq!(
                    actual.is_infinite(),
                    expected.is_infinite(),
                    "{}",
                    context()
                );
                if actual.is_infinite() || (*actual == 0.0 && *expected == 0.0) {
                    assert_eq!(
                        actual.is_sign_negative(),
                        expected.is_sign_negative(),
                        "{}",
                        context()
                    );
                } else if actual.is_finite() && expected.is_finite() {
                    assert!((actual - expected).abs() <= 1e-6, "{}", context());
                }
            }
        }
        (Tensor::F64(actual), Tensor::F64(expected)) => {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.n_elements(), expected.n_elements());
            for (index, (actual, expected)) in actual
                .as_slice()
                .unwrap()
                .iter()
                .zip(expected.as_slice().unwrap())
                .enumerate()
            {
                let context = || {
                    format!(
                        "{op} F64 index {index}: actual={actual:?} ({:#018x}), expected={expected:?} ({:#018x})",
                        actual.to_bits(),
                        expected.to_bits()
                    )
                };
                assert_eq!(actual.is_nan(), expected.is_nan(), "{}", context());
                assert_eq!(
                    actual.is_infinite(),
                    expected.is_infinite(),
                    "{}",
                    context()
                );
                if actual.is_infinite() || (*actual == 0.0 && *expected == 0.0) {
                    assert_eq!(
                        actual.is_sign_negative(),
                        expected.is_sign_negative(),
                        "{}",
                        context()
                    );
                } else if actual.is_finite() && expected.is_finite() {
                    assert!((actual - expected).abs() <= 1e-12, "{}", context());
                }
            }
        }
        _ => panic!("expected matching F32 or F64 tensors"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_float_div_rem_preserve_ieee_special_values() {
    if !gpu_available() {
        eprintln!(
            "skipping test_cubecl_float_div_rem_preserve_ieee_special_values — no CUDA device found"
        );
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let cases = [
        (
            super::tensor_f32(
                vec![7],
                vec![1.0, 1.0, 0.0, f32::NAN, f32::INFINITY, -0.0, -4.0],
            ),
            super::tensor_f32(vec![7], vec![0.0, -0.0, 0.0, 1.0, f32::INFINITY, 2.0, 2.0]),
            super::tensor_f32(vec![7], vec![0.0, -0.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        ),
        (
            tensor_f64(
                vec![7],
                vec![1.0, 1.0, 0.0, f64::NAN, f64::INFINITY, -0.0, -4.0],
            ),
            tensor_f64(vec![7], vec![0.0, -0.0, 0.0, 1.0, f64::INFINITY, 2.0, 2.0]),
            tensor_f64(vec![7], vec![0.0, -0.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        ),
    ];

    for (lhs, div_rhs, rem_rhs) in cases {
        let gpu_lhs = upload(&gpu, &lhs);
        let gpu_div_rhs = upload(&gpu, &div_rhs);
        let gpu_rem_rhs = upload(&gpu, &rem_rhs);

        let expected = cpu.div(&lhs, &div_rhs).unwrap();
        let gpu_out = gpu.div(&gpu_lhs, &gpu_div_rhs).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_float_classes_and_zero_signs_match("div", &actual, &expected);

        let expected = cpu.rem(&lhs, &rem_rhs).unwrap();
        let gpu_out = gpu.rem(&gpu_lhs, &gpu_rem_rhs).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_float_classes_and_zero_signs_match("rem", &actual, &expected);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_float_unary_special_values_match_cpu() {
    if !gpu_available() {
        eprintln!("skipping test_float_unary_special_values_match_cpu — no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let cases = [
        super::tensor_f32(vec![5], vec![-0.0, 0.0, -2.5, 3.0, f32::NAN]),
        tensor_f64(vec![5], vec![-0.0, 0.0, -2.5, 3.0, f64::NAN]),
    ];

    for input in cases {
        let gpu_input = upload(&gpu, &input);

        let expected_abs = cpu.abs(&input).unwrap();
        let gpu_abs = gpu.abs(&gpu_input).unwrap();
        let actual_abs = download(&gpu, &gpu_abs);
        match (&actual_abs, &expected_abs) {
            (Tensor::F32(actual), Tensor::F32(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1..4], expected[1..4]);
                assert!(actual[4].is_nan());
            }
            (Tensor::F64(actual), Tensor::F64(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1..4], expected[1..4]);
                assert!(actual[4].is_nan());
            }
            _ => panic!("expected matching F32 or F64 abs tensors"),
        }

        let expected_sign = cpu.sign(&input).unwrap();
        let gpu_sign = gpu.sign(&gpu_input).unwrap();
        let actual_sign = download(&gpu, &gpu_sign);
        match (&actual_sign, &expected_sign) {
            (Tensor::F32(actual), Tensor::F32(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1].to_bits(), expected[1].to_bits());
                assert_eq!(actual[2..4], expected[2..4]);
                assert!(actual[4].is_nan());
            }
            (Tensor::F64(actual), Tensor::F64(expected)) => {
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1].to_bits(), expected[1].to_bits());
                assert_eq!(actual[2..4], expected[2..4]);
                assert!(actual[4].is_nan());
            }
            _ => panic!("expected matching F32 or F64 sign tensors"),
        }
    }
}

#[test]
#[ignore]
fn test_cubecl_unary_float_elementwise_matches_cpu() {
    let positive = tensor_f64(vec![4], vec![0.25, 0.5, 1.5, 3.0]);
    let signed = tensor_f64(vec![4], vec![-2.0, -0.0, 3.5, -4.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_positive = upload(&gpu, &positive);
    let gpu_signed = upload(&gpu, &signed);

    let expected = cpu.neg(&signed).unwrap();
    let gpu_out = gpu.neg(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.abs(&signed).unwrap();
    let gpu_out = gpu.abs(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sign(&signed).unwrap();
    let gpu_out = gpu.sign(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.exp(&positive).unwrap();
    let gpu_out = gpu.exp(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.log(&positive).unwrap();
    let gpu_out = gpu.log(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sin(&positive).unwrap();
    let gpu_out = gpu.sin(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.cos(&positive).unwrap();
    let gpu_out = gpu.cos(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.tanh(&positive).unwrap();
    let gpu_out = gpu.tanh(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sqrt(&positive).unwrap();
    let gpu_out = gpu.sqrt(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.rsqrt(&positive).unwrap();
    let gpu_out = gpu.rsqrt(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.expm1(&positive).unwrap();
    let gpu_out = gpu.expm1(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.log1p(&positive).unwrap();
    let gpu_out = gpu.log1p(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_float_compare_select_and_clamp_match_cpu() {
    let lhs = tensor_f64(vec![4], vec![1.0, 3.0, 2.0, 4.0]);
    let rhs = tensor_f64(vec![4], vec![2.0, 3.0, 1.0, 5.0]);
    let lower = tensor_f64(vec![4], vec![0.5, 2.0, 1.5, 3.5]);
    let upper = tensor_f64(vec![4], vec![1.5, 4.0, 2.5, 4.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let gpu_lower = upload(&gpu, &lower);
    let gpu_upper = upload(&gpu, &upper);

    let expected = cpu.compare(&lhs, &rhs, &CompareDir::Ge).unwrap();
    let gpu_out = gpu.compare(&gpu_lhs, &gpu_rhs, &CompareDir::Ge).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.select(&expected, &lhs, &rhs).unwrap();
    let gpu_pred = upload(&gpu, &actual);
    let gpu_out = gpu.select(&gpu_pred, &gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.clamp(&lhs, &lower, &upper).unwrap();
    let gpu_out = gpu.clamp(&gpu_lhs, &gpu_lower, &gpu_upper).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_integer_add_mul_compare_select_match_cpu() {
    if !gpu_available() {
        eprintln!(
            "skipping test_cubecl_integer_add_mul_compare_select_match_cpu — no CUDA device found"
        );
        return;
    }

    let i32_lhs = tensor_i32(vec![2, 3], vec![1, -2, 3, 4, -5, 6]);
    let i32_rhs = tensor_i32(vec![2, 3], vec![6, 5, -4, 3, 2, -1]);
    assert_integer_binary_and_select_matches_cpu(&i32_lhs, &i32_rhs);

    let i64_lhs = tensor_i64(vec![2, 3], vec![10, -20, 30, 40, -50, 60]);
    let i64_rhs = tensor_i64(vec![2, 3], vec![7, 6, -5, 4, 3, -2]);
    assert_integer_binary_and_select_matches_cpu(&i64_lhs, &i64_rhs);

    let i32_lhs = tensor_i32(vec![3], vec![i32::MAX, i32::MIN, 50]);
    let i32_rhs = tensor_i32(vec![3], vec![1, -1, i32::MAX]);
    assert_integer_binary_and_select_matches_cpu(&i32_lhs, &i32_rhs);

    let i64_lhs = tensor_i64(vec![2], vec![i64::MAX, i64::MIN]);
    let i64_rhs = tensor_i64(vec![2], vec![1, -1]);
    assert_integer_binary_and_select_matches_cpu(&i64_lhs, &i64_rhs);
}

fn assert_integer_binary_and_select_matches_cpu(lhs: &Tensor, rhs: &Tensor) {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, lhs);
    let gpu_rhs = upload(&gpu, rhs);

    let expected = cpu.add(lhs, rhs).unwrap();
    let gpu_out = gpu.add(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.sub(lhs, rhs).unwrap();
    let gpu_out = gpu.sub(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.mul(lhs, rhs).unwrap();
    let gpu_out = gpu.mul(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.div(lhs, rhs).unwrap();
    let gpu_out = gpu.div(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.rem(lhs, rhs).unwrap();
    let gpu_out = gpu.rem(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let pow_rhs = nonnegative_integer_exponents_like(rhs);
    let gpu_pow_rhs = upload(&gpu, &pow_rhs);
    let expected = cpu.pow(lhs, &pow_rhs).unwrap();
    let gpu_out = gpu.pow(&gpu_lhs, &gpu_pow_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.maximum(lhs, rhs).unwrap();
    let gpu_out = gpu.maximum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.minimum(lhs, rhs).unwrap();
    let gpu_out = gpu.minimum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.neg(lhs).unwrap();
    let gpu_out = gpu.neg(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.abs(lhs).unwrap();
    let gpu_out = gpu.abs(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.sign(lhs).unwrap();
    let gpu_out = gpu.sign(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected_pred = cpu.compare(lhs, rhs, &CompareDir::Ge).unwrap();
    let gpu_pred = gpu.compare(&gpu_lhs, &gpu_rhs, &CompareDir::Ge).unwrap();
    let actual_pred = download(&gpu, &gpu_pred);
    assert_tensor_close(&actual_pred, &expected_pred, 0.0);

    let expected = cpu.select(&expected_pred, lhs, rhs).unwrap();
    let gpu_out = gpu.select(&gpu_pred, &gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

fn nonnegative_integer_exponents_like(tensor: &Tensor) -> Tensor {
    match tensor {
        Tensor::I32(tensor) => tensor_i32(
            tensor.shape().to_vec(),
            (0..tensor.n_elements())
                .map(|idx| (idx % 5) as i32)
                .collect(),
        ),
        Tensor::I64(tensor) => tensor_i64(
            tensor.shape().to_vec(),
            (0..tensor.n_elements())
                .map(|idx| (idx % 5) as i64)
                .collect(),
        ),
        _ => panic!("expected integer tensor"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_integer_domain_errors_match_cpu() {
    if !gpu_available() {
        eprintln!("skipping test_cubecl_integer_domain_errors_match_cpu — no CUDA device found");
        return;
    }

    let mut gpu = gpu_backend();
    let lhs = tensor_i32(vec![2], vec![1, 2]);
    let zero_rhs = tensor_i32(vec![2], vec![1, 0]);
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_zero_rhs = upload(&gpu, &zero_rhs);

    let err = gpu.div(&gpu_lhs, &gpu_zero_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::DivisionByZero {
            op: "div",
            dtype: DType::I32
        }
    ));

    let err = gpu.rem(&gpu_lhs, &gpu_zero_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::DivisionByZero {
            op: "rem",
            dtype: DType::I32
        }
    ));

    let exp = tensor_i32(vec![2], vec![2, -1]);
    let gpu_exp = upload(&gpu, &exp);
    let err = gpu.pow(&gpu_lhs, &gpu_exp).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::NegativeIntegerExponent {
            op: "pow",
            dtype: DType::I32
        }
    ));
}

#[test]
#[ignore]
fn test_cubecl_complex_elementwise_matches_cpu_and_rejects_unsupported_ops() {
    let lhs = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(-3.0, 0.5),
            num_complex::Complex64::new(0.25, -1.25),
        ],
    );
    let rhs = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(-0.5, 1.0),
            num_complex::Complex64::new(2.0, -1.5),
            num_complex::Complex64::new(0.5, 0.25),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);

    let expected = cpu.add(&lhs, &rhs).unwrap();
    let gpu_out = gpu.add(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sub(&lhs, &rhs).unwrap();
    let gpu_out = gpu.sub(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.mul(&lhs, &rhs).unwrap();
    let gpu_out = gpu.mul(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.div(&lhs, &rhs).unwrap();
    let gpu_out = gpu.div(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu.rem(&gpu_lhs, &gpu_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::UnsupportedOpDType {
            op: "rem",
            dtype: DType::C64,
            backend: BackendId::Cuda,
        }
    ));

    let expected = cpu.neg(&lhs).unwrap();
    let gpu_out = gpu.neg(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.conj(&lhs).unwrap();
    let gpu_out = gpu.conj(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.abs(&lhs).unwrap();
    let gpu_out = gpu.abs(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu.exp(&gpu_lhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::UnsupportedOpDType {
            op: "exp",
            dtype: DType::C64,
            backend: tenferro_tensor::BackendId::Cuda,
        }
    ));

    let err = gpu
        .compare(&gpu_lhs, &gpu_rhs, &CompareDir::Eq)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "compare", .. }
    ));

    let err = gpu.select(&gpu_lhs, &gpu_lhs, &gpu_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "select", .. }
    ));

    let err = gpu.clamp(&gpu_lhs, &gpu_lhs, &gpu_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "clamp", .. }
    ));

    let converted = gpu.convert(&gpu_lhs, DType::C64).unwrap();
    let actual = download(&gpu, &converted);
    assert_tensor_close(&actual, &lhs, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_float_to_complex_convert_preserves_resident_device() {
    if !gpu_available() {
        eprintln!(
            "skipping test_cubecl_float_to_complex_convert_preserves_resident_device — no CUDA device found"
        );
        return;
    }
    let mut gpu = gpu_backend();
    let input = tensor_f64(vec![2], vec![1.0, -2.0]);
    let gpu_input = upload(&gpu, &input);

    let converted = gpu.convert(&gpu_input, DType::C64).unwrap();

    let Tensor::C64(tensor) = converted else {
        panic!("expected C64 output");
    };
    let resident = tensor
        .placement()
        .device
        .as_ref()
        .expect("converted tensor should preserve CUDA resident device");
    assert_eq!(resident.kind, DeviceKind::Gpu(GpuBackendKind::Cuda));
    assert_eq!(resident.ordinal, gpu.runtime().device_ordinal());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_conj_real_clone_rejects_missing_resident_device_metadata() {
    if !gpu_available() {
        eprintln!(
            "skipping test_cubecl_conj_real_clone_rejects_missing_resident_device_metadata — no CUDA device found"
        );
        return;
    }
    let mut gpu = gpu_backend();
    let input = tensor_f64(vec![2], vec![1.0, -2.0]);
    let mut gpu_input = match upload(&gpu, &input) {
        Tensor::F64(tensor) => tensor,
        _ => panic!("expected F64 upload"),
    };
    let mut placement = gpu_input.placement().clone();
    placement.device = None;
    gpu_input.set_placement(placement);

    let err = gpu.conj(&Tensor::F64(gpu_input)).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "conj", .. }
    ));
}
