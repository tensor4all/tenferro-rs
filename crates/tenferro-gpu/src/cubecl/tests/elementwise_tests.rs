// Run with: cargo test --features cuda -- --ignored
use crate::config::CompareDir;
use crate::cubecl::gpu_available;
use crate::{DType, DeviceKind, GpuBackendKind, Tensor};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    BackendSessionHost, ErrorKind, TensorAnalytic, TensorElementwise, TensorFusion, TensorRead,
    TensorStructural, TensorView, TensorWrite,
};

use super::{
    assert_cuda_numerical_error, assert_cuda_unsupported_dtype, assert_dtype_mismatch,
    assert_shape_mismatch, assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32,
    tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn elementwise_read_into_writes_owned_output_without_mutating_inputs() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let lhs = tensor_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = tensor_f32(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let expected = lhs
        .as_typed::<f32>()
        .unwrap()
        .as_slice()
        .unwrap()
        .iter()
        .zip(rhs.as_typed::<f32>().unwrap().as_slice().unwrap())
        .map(|(lhs, rhs)| lhs * rhs)
        .collect::<Vec<_>>();

    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let mut gpu_out = upload(&gpu, &tensor_f32(vec![2, 2], vec![0.0; 4]));
    gpu.mul_read_into(
        TensorRead::from_tensor(&gpu_lhs),
        TensorRead::from_tensor(&gpu_rhs),
        TensorWrite::from_tensor(&mut gpu_out),
    )
    .unwrap();

    let actual = download(&gpu, &gpu_out);
    assert_eq!(
        actual.as_typed::<f32>().unwrap().as_slice().unwrap(),
        expected
    );
    assert_tensor_close(&download(&gpu, &gpu_lhs), &lhs, 0.0);
    assert_tensor_close(&download(&gpu, &gpu_rhs), &rhs, 0.0);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn elementwise_read_compact_view_chain_uses_native_kernels() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let host_lhs = tensor_f32(vec![2, 2], vec![0.5, -1.0, 2.0, 3.5]);
    let host_rhs = tensor_f32(vec![2, 2], vec![2.0, 4.0, -0.5, 1.5]);
    let mut cpu = cpu_backend();
    let sum = cpu.add(&host_lhs, &host_rhs).unwrap();
    let product = cpu.mul(&sum, &host_rhs).unwrap();
    let expected = cpu.tanh(&product).unwrap();

    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &host_lhs);
    let rhs = upload(&gpu, &host_rhs);
    let sum = gpu
        .add_read(
            TensorRead::from_view(TensorView::F32(lhs.as_typed::<f32>().unwrap().as_view())),
            TensorRead::from_view(TensorView::F32(rhs.as_typed::<f32>().unwrap().as_view())),
        )
        .unwrap();
    let product = gpu
        .mul_read(
            TensorRead::from_view(TensorView::F32(sum.as_typed::<f32>().unwrap().as_view())),
            TensorRead::from_view(TensorView::F32(rhs.as_typed::<f32>().unwrap().as_view())),
        )
        .unwrap();
    let actual = gpu
        .tanh_read(TensorRead::from_view(TensorView::F32(
            product.as_typed::<f32>().unwrap().as_view(),
        )))
        .unwrap();

    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-6);
    assert_tensor_close(&download(&gpu, &lhs), &host_lhs, 0.0);
    assert_tensor_close(&download(&gpu, &rhs), &host_rhs, 0.0);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn elementwise_read_preserves_offset_and_strided_layouts() {
    use tenferro_tensor::StridedSliceSpec;
    let host = tensor_f64(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let mut gpu = gpu_backend();
    let mut cpu = cpu_backend();
    let device = upload(&gpu, &host);
    let view = device.as_typed::<f64>().unwrap().as_view();
    let host_view = host.as_typed::<f64>().unwrap().as_view();
    fn layouts(
        view: tenferro_tensor::TypedTensorView<'_, f64>,
    ) -> [tenferro_tensor::TypedTensorView<'_, f64>; 4] {
        [
            view.clone(),
            view.transpose_view([1, 0]).unwrap(),
            view.try_slice_axis(1, StridedSliceSpec::new(1, None, 1))
                .unwrap(),
            view.try_slice_axis(0, StridedSliceSpec::reverse()).unwrap(),
        ]
    }
    for (device_view, host_view) in layouts(view).into_iter().zip(layouts(host_view)) {
        let read = TensorRead::from_view(TensorView::F64(device_view));
        let expected_read = TensorRead::from_view(TensorView::F64(host_view));
        let expected = cpu
            .add_read(expected_read.clone(), expected_read.clone())
            .unwrap();
        let actual = gpu.add_read(read.clone(), read.clone()).unwrap();
        assert_tensor_close(&download(&gpu, &actual), &expected, 1e-12);
        let expected = cpu.tanh_read(expected_read).unwrap();
        let actual = gpu.tanh_read(read).unwrap();
        assert_tensor_close(&download(&gpu, &actual), &expected, 1e-12);
    }
    assert_tensor_close(&download(&gpu, &device), &host, 0.0);
    let mut host_output = tensor_f64(vec![2, 3], vec![0.; 6]);
    assert!(gpu
        .neg_read_into(
            TensorRead::from_tensor(&device),
            TensorWrite::from_tensor(&mut host_output)
        )
        .is_err());
}

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

    match (actual.dtype(), expected.dtype()) {
        (DType::C32, DType::C32) => {
            let actual = actual
                .as_typed::<Complex32>()
                .expect("the dtype guard selects this arm");
            let expected = expected
                .as_typed::<Complex32>()
                .expect("the dtype guard selects this arm");
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
        (DType::C64, DType::C64) => {
            let actual = actual
                .as_typed::<Complex64>()
                .expect("the dtype guard selects this arm");
            let expected = expected
                .as_typed::<Complex64>()
                .expect("the dtype guard selects this arm");
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
    assert!(gpu_available(), "CUDA test requires an available device");

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
            let error = result.expect_err("complex dtype mismatch must be rejected");
            assert_dtype_mismatch(&error, op, expected_lhs, expected_rhs);
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
            let error = result.expect_err("mixed dtype operation must be rejected");
            assert_dtype_mismatch(&error, op, expected_lhs, expected_rhs);
        }
        for (op, result) in [
            ("add", gpu.add(&gpu_rhs, &gpu_lhs)),
            ("sub", gpu.sub(&gpu_rhs, &gpu_lhs)),
            ("mul", gpu.mul(&gpu_rhs, &gpu_lhs)),
            ("div", gpu.div(&gpu_rhs, &gpu_lhs)),
        ] {
            let error = result.expect_err("mixed dtype operation must be rejected");
            assert_dtype_mismatch(&error, op, expected_rhs, expected_lhs);
        }
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_scalar_div_rem_pow_match_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");

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
        let error = gpu.div(&gpu_lhs, &gpu_zero).unwrap_err();
        assert_cuda_numerical_error(&error, "div", dtype, false);
        let error = gpu.rem(&gpu_lhs, &gpu_zero).unwrap_err();
        assert_cuda_numerical_error(&error, "rem", dtype, false);

        let (scalar_one, zero_rhs) = match dtype {
            DType::I32 => (tensor_i32(vec![], vec![1]), tensor_i32(vec![2], vec![1, 0])),
            DType::I64 => (tensor_i64(vec![], vec![1]), tensor_i64(vec![2], vec![1, 0])),
            _ => unreachable!(),
        };
        let gpu_scalar_one = upload(&gpu, &scalar_one);
        let gpu_zero_rhs = upload(&gpu, &zero_rhs);
        let error = gpu.div(&gpu_scalar_one, &gpu_zero_rhs).unwrap_err();
        assert_cuda_numerical_error(&error, "div", dtype, false);
        let error = gpu.rem(&gpu_scalar_one, &gpu_zero_rhs).unwrap_err();
        assert_cuda_numerical_error(&error, "rem", dtype, false);

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
        let error = gpu.pow(&gpu_base, &gpu_exponent).unwrap_err();
        assert_cuda_numerical_error(&error, "pow", dtype, true);
    }

    let unequal_lhs = upload(&gpu, &tensor_f32(vec![2], vec![2.0, 3.0]));
    let unequal_rhs = upload(&gpu, &tensor_f32(vec![3], vec![2.0, 3.0, 4.0]));
    let error = gpu.pow(&unequal_lhs, &unequal_rhs).unwrap_err();
    assert_shape_mismatch(&error, "pow", &[2], &[3]);

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
    assert!(gpu_available(), "CUDA test requires an available device");

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
        match (actual.dtype(), expected.dtype()) {
            (DType::F32, DType::F32) => {
                let actual = actual
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let expected = expected
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(&actual[..3], &[5.0, 13.0, 0.0]);
                for (&actual, &expected) in actual[3..5].iter().zip(&expected[3..5]) {
                    assert!(actual.is_finite() && actual > 0.0);
                    assert!((actual / expected - 1.0).abs() <= 2.0 * f32::EPSILON);
                }
            }
            (DType::F64, DType::F64) => {
                let actual = actual
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
                let expected = expected
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
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
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
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
fn test_broadcast_multiply_integer_overflow_matches_cpu_wrapping() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for (scalar, vector) in [
        (
            tensor_i32(vec![], vec![i32::MAX]),
            tensor_i32(vec![2], vec![2, -1]),
        ),
        (
            tensor_i64(vec![], vec![i64::MAX]),
            tensor_i64(vec![2], vec![2, -1]),
        ),
    ] {
        let expected = cpu.mul(&scalar, &vector).unwrap();
        let gpu_scalar = upload(&gpu, &scalar);
        let gpu_vector = upload(&gpu, &vector);
        let actual = gpu
            .execute_broadcast_multiply(
                TensorRead::from_tensor(&gpu_scalar),
                &[2],
                &[],
                TensorRead::from_tensor(&gpu_vector),
                &[2],
                &[0],
            )
            .unwrap()
            .expect("integer broadcast multiply should fuse");

        assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_log1p_small_x_f32_precision() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = super::gpu_backend();
    let x_values = vec![1e-7_f32, 1e-6, 1e-5, 1e-4, 1e-3];
    let cpu_input = super::tensor_f32(vec![x_values.len()], x_values.clone());
    let gpu_input = super::upload(&backend, &cpu_input);

    let gpu_out = backend.log1p(&gpu_input).unwrap();
    let result = super::download(&backend, &gpu_out);
    let result_slice = result
        .as_typed::<f32>()
        .expect("expected F32")
        .as_slice()
        .unwrap()
        .to_vec();

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
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = super::gpu_backend();
    let x_values = vec![1e-7_f32, 1e-6, 1e-5, 1e-4, 1e-3];
    let cpu_input = super::tensor_f32(vec![x_values.len()], x_values.clone());
    let gpu_input = super::upload(&backend, &cpu_input);

    let gpu_out = backend
        .with_backend_session(|__s| {
            __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&gpu_input))
        })
        .unwrap();
    let result = super::download(&backend, &gpu_out);
    let result_slice = result
        .as_typed::<f32>()
        .expect("expected F32")
        .as_slice()
        .unwrap()
        .to_vec();

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

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_maximum_minimum_propagate_nan_independent_of_argument_order() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for (lhs, rhs) in [
        (
            tensor_f32(vec![2], vec![f32::NAN, 1.0]),
            tensor_f32(vec![2], vec![1.0, f32::NAN]),
        ),
        (
            tensor_f64(vec![2], vec![f64::NAN, 1.0]),
            tensor_f64(vec![2], vec![1.0, f64::NAN]),
        ),
    ] {
        let gpu_lhs = upload(&gpu, &lhs);
        let gpu_rhs = upload(&gpu, &rhs);
        for (label, expected, actual) in [
            (
                "maximum",
                cpu.maximum(&lhs, &rhs).unwrap(),
                gpu.maximum(&gpu_lhs, &gpu_rhs).unwrap(),
            ),
            (
                "minimum",
                cpu.minimum(&lhs, &rhs).unwrap(),
                gpu.minimum(&gpu_lhs, &gpu_rhs).unwrap(),
            ),
        ] {
            assert_float_classes_and_zero_signs_match(label, &download(&gpu, &actual), &expected);
        }
    }
}

fn assert_float_classes_and_zero_signs_match(op: &str, actual: &Tensor, expected: &Tensor) {
    match (actual.dtype(), expected.dtype()) {
        (DType::F32, DType::F32) => {
            let actual = actual
                .as_typed::<f32>()
                .expect("the dtype guard selects this arm");
            let expected = expected
                .as_typed::<f32>()
                .expect("the dtype guard selects this arm");
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
        (DType::F64, DType::F64) => {
            let actual = actual
                .as_typed::<f64>()
                .expect("the dtype guard selects this arm");
            let expected = expected
                .as_typed::<f64>()
                .expect("the dtype guard selects this arm");
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
    assert!(gpu_available(), "CUDA test requires an available device");

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
    assert!(gpu_available(), "CUDA test requires an available device");

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
        match (actual_abs.dtype(), expected_abs.dtype()) {
            (DType::F32, DType::F32) => {
                let actual = actual_abs
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let expected = expected_abs
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1..4], expected[1..4]);
                assert!(actual[4].is_nan());
            }
            (DType::F64, DType::F64) => {
                let actual = actual_abs
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
                let expected = expected_abs
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
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
        match (actual_sign.dtype(), expected_sign.dtype()) {
            (DType::F32, DType::F32) => {
                let actual = actual_sign
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let expected = expected_sign
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                let actual = actual.as_slice().unwrap();
                let expected = expected.as_slice().unwrap();
                assert_eq!(actual[0].to_bits(), expected[0].to_bits());
                assert_eq!(actual[1].to_bits(), expected[1].to_bits());
                assert_eq!(actual[2..4], expected[2..4]);
                assert!(actual[4].is_nan());
            }
            (DType::F64, DType::F64) => {
                let actual = actual_sign
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
                let expected = expected_sign
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
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

    let expected = cpu
        .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(&positive)))
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(&gpu_positive)))
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .with_backend_session(|__s| {
            __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&positive))
        })
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| {
            __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&gpu_positive))
        })
        .unwrap();
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

    cpu.with_backend_session(|__s| {
        __s.select_read(
            TensorRead::from_tensor(&expected),
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
        )
    })
    .unwrap();
    let gpu_pred = upload(&gpu, &actual);
    gpu.with_backend_session(|__s| {
        __s.select_read(
            TensorRead::from_tensor(&gpu_pred),
            TensorRead::from_tensor(&gpu_lhs),
            TensorRead::from_tensor(&gpu_rhs),
        )
    })
    .unwrap();
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
    assert!(gpu_available(), "CUDA test requires an available device");

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

    cpu.with_backend_session(|__s| {
        __s.select_read(
            TensorRead::from_tensor(&expected_pred),
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
        )
    })
    .unwrap();
    gpu.with_backend_session(|__s| {
        __s.select_read(
            TensorRead::from_tensor(&gpu_pred),
            TensorRead::from_tensor(&gpu_lhs),
            TensorRead::from_tensor(&gpu_rhs),
        )
    })
    .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

fn nonnegative_integer_exponents_like(tensor: &Tensor) -> Tensor {
    match tensor.dtype() {
        crate::DType::I32 => {
            let tensor = tensor.as_typed::<i32>().expect("expected integer tensor");
            tensor_i32(
                tensor.shape().to_vec(),
                (0..tensor.n_elements())
                    .map(|idx| (idx % 5) as i32)
                    .collect(),
            )
        }
        crate::DType::I64 => {
            let tensor = tensor.as_typed::<i64>().expect("expected integer tensor");
            tensor_i64(
                tensor.shape().to_vec(),
                (0..tensor.n_elements())
                    .map(|idx| (idx % 5) as i64)
                    .collect(),
            )
        }
        _ => panic!("expected integer tensor"),
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_integer_domain_errors_match_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut gpu = gpu_backend();
    let lhs = tensor_i32(vec![2], vec![1, 2]);
    let zero_rhs = tensor_i32(vec![2], vec![1, 0]);
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_zero_rhs = upload(&gpu, &zero_rhs);

    let err = gpu.div(&gpu_lhs, &gpu_zero_rhs).unwrap_err();
    assert_cuda_numerical_error(&err, "div", DType::I32, false);

    let err = gpu.rem(&gpu_lhs, &gpu_zero_rhs).unwrap_err();
    assert_cuda_numerical_error(&err, "rem", DType::I32, false);

    let exp = tensor_i32(vec![2], vec![2, -1]);
    let gpu_exp = upload(&gpu, &exp);
    let err = gpu.pow(&gpu_lhs, &gpu_exp).unwrap_err();
    assert_cuda_numerical_error(&err, "pow", DType::I32, true);
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
    assert_cuda_unsupported_dtype(&err, "rem", DType::C64);

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
    assert_cuda_unsupported_dtype(&err, "exp", DType::C64);

    let err = gpu
        .compare(&gpu_lhs, &gpu_rhs, &CompareDir::Eq)
        .unwrap_err();
    assert_cuda_unsupported_dtype(&err, "compare", DType::C64);

    gpu.with_backend_session(|__s| {
        __s.select_read(
            TensorRead::from_tensor(&gpu_lhs),
            TensorRead::from_tensor(&gpu_lhs),
            TensorRead::from_tensor(&gpu_rhs),
        )
    })
    .unwrap_err();
    assert_cuda_unsupported_dtype(&err, "select", DType::C64);

    let err = gpu.clamp(&gpu_lhs, &gpu_lhs, &gpu_rhs).unwrap_err();
    assert_cuda_unsupported_dtype(&err, "clamp", DType::C64);

    let converted = gpu.convert(&gpu_lhs, DType::C64).unwrap();
    let actual = download(&gpu, &converted);
    assert_tensor_close(&actual, &lhs, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_float_to_complex_convert_preserves_resident_device() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = gpu_backend();
    let input = tensor_f64(vec![2], vec![1.0, -2.0]);
    let gpu_input = upload(&gpu, &input);

    let converted = gpu.convert(&gpu_input, DType::C64).unwrap();

    let Some(tensor) = converted.as_typed::<Complex64>() else {
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
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = gpu_backend();
    let input = tensor_f64(vec![2], vec![1.0, -2.0]);
    let mut gpu_input = upload(&gpu, &input)
        .into_typed::<f64>()
        .expect("expected F64 upload");
    let mut placement = gpu_input.placement().clone();
    placement.device = None;
    gpu_input.set_placement(placement);

    let err = gpu.conj(&Tensor::from_typed::<f64>(gpu_input)).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
}
