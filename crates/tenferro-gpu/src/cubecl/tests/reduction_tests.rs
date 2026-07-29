// Run with: cargo test --features cuda -- --ignored
use tenferro_tensor::{DType, TensorRead, TensorReduction};

use super::{
    assert_cuda_unsupported_dtype, assert_tensor_close, cpu_backend, download, gpu_backend,
    tensor_bool, tensor_c32, tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};

#[test]
#[ignore]
fn test_cubecl_full_axis_reductions_preserve_scalar_shape_and_values() {
    let inputs = [
        tensor_f32(vec![2, 2], vec![1.0, -2.0, 3.0, 4.0]),
        tensor_f64(vec![2, 2], vec![1.0, -2.0, 3.0, 4.0]),
        tensor_i32(vec![2, 2], vec![1, -2, 3, 4]),
        tensor_i64(vec![2, 2], vec![1, -2, 3, 4]),
        tensor_c32(
            vec![2, 2],
            vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(2.0, -1.0),
                num_complex::Complex32::new(-0.5, 0.25),
                num_complex::Complex32::new(3.0, 2.0),
            ],
        ),
        tensor_c64(
            vec![2, 2],
            vec![
                num_complex::Complex64::new(1.0, 1.0),
                num_complex::Complex64::new(2.0, -1.0),
                num_complex::Complex64::new(-0.5, 0.25),
                num_complex::Complex64::new(3.0, 2.0),
            ],
        ),
    ];

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for input in &inputs {
        let gpu_input = upload(&gpu, input);
        for (expected, gpu_out) in [
            (
                cpu.reduce_sum(input, &[0, 1]).unwrap(),
                gpu.reduce_sum(&gpu_input, &[0, 1]).unwrap(),
            ),
            (
                cpu.reduce_prod(input, &[0, 1]).unwrap(),
                gpu.reduce_prod(&gpu_input, &[0, 1]).unwrap(),
            ),
        ] {
            assert!(expected.shape().is_empty());
            assert!(gpu_out.shape().is_empty());
            let actual = download(&gpu, &gpu_out);
            assert_tensor_close(&actual, &expected, 1e-5);
        }

        if !matches!(input, crate::Tensor::C32(_) | crate::Tensor::C64(_)) {
            for (expected, gpu_out) in [
                (
                    cpu.reduce_min(input, &[0, 1]).unwrap(),
                    gpu.reduce_min(&gpu_input, &[0, 1]).unwrap(),
                ),
                (
                    cpu.reduce_max(input, &[0, 1]).unwrap(),
                    gpu.reduce_max(&gpu_input, &[0, 1]).unwrap(),
                ),
            ] {
                assert!(expected.shape().is_empty());
                assert!(gpu_out.shape().is_empty());
                let actual = download(&gpu, &gpu_out);
                assert_tensor_close(&actual, &expected, 0.0);
            }
        }
    }
}

#[test]
#[ignore]
fn test_cubecl_float_reductions_match_cpu() {
    let input = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_prod(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_sum_squares_matches_cpu_for_multi_axis_and_empty_axes() {
    let inputs = [
        tensor_f32(vec![2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]),
        tensor_f64(vec![2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]),
    ];
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    for input in &inputs {
        let gpu_input = upload(&gpu, input);
        for axes in [&[0, 1][..], &[][..]] {
            let expected = cpu
                .reduce_sum_squares_read(TensorRead::from_tensor(input), axes)
                .unwrap();
            let gpu_output = gpu
                .reduce_sum_squares_read(TensorRead::from_tensor(&gpu_input), axes)
                .unwrap();
            assert_tensor_close(&download(&gpu, &gpu_output), &expected, 1e-5);
        }
    }

    let integer = upload(&gpu, &tensor_i32(vec![2], vec![1, 2]));
    let error = gpu
        .reduce_sum_squares_read(TensorRead::from_tensor(&integer), &[0])
        .unwrap_err();
    assert_cuda_unsupported_dtype(&error, "reduce_sum_squares", DType::I32);
}

#[test]
#[ignore]
fn test_cubecl_sum_squares_does_not_contract_multiply_and_add() {
    // These values distinguish same-dtype multiply-then-add from FMA.
    let first = 0.000_442_517_25_f32;
    let second = 0.684_452_06_f32;
    let input = tensor_f32(vec![2], vec![first, second]);
    let expected = first * first + second * second;
    let contracted = second.mul_add(second, first * first);
    assert_ne!(expected.to_bits(), contracted.to_bits());

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_output = gpu
        .reduce_sum_squares_read(TensorRead::from_tensor(&gpu_input), &[0])
        .unwrap();
    let actual = download(&gpu, &gpu_output);
    let crate::Tensor::F32(actual) = actual else {
        panic!("sum-of-squares output must remain f32");
    };
    let actual = actual.as_slice().unwrap()[0];

    assert_eq!(actual.to_bits(), expected.to_bits());
    assert_ne!(actual.to_bits(), contracted.to_bits());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_float_max_min_reductions_propagate_nan_for_unit_and_plane() {
    if !crate::cubecl::gpu_available() {
        eprintln!("skipping reduction NaN propagation parity test - no CUDA device found");
        return;
    }

    fn assert_all_nan(tensor: &crate::Tensor) {
        match tensor {
            crate::Tensor::F32(tensor) => {
                assert!(tensor
                    .as_slice()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()));
            }
            crate::Tensor::F64(tensor) => {
                assert!(tensor
                    .as_slice()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()));
            }
            other => panic!("expected float reduction output, got {:?}", other.dtype()),
        }
    }

    let mut inputs = Vec::new();
    for len in [4, 1024] {
        let mut f32_values = vec![1.0_f32; len];
        f32_values[len / 2 + 1] = f32::NAN;
        inputs.push(tensor_f32(vec![len], f32_values));

        let mut f64_values = vec![1.0_f64; len];
        f64_values[len / 2 + 1] = f64::NAN;
        inputs.push(tensor_f64(vec![len], f64_values));
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for input in inputs {
        let gpu_input = upload(&gpu, &input);
        for (expected, actual) in [
            (
                cpu.reduce_max(&input, &[0]).unwrap(),
                gpu.reduce_max(&gpu_input, &[0]).unwrap(),
            ),
            (
                cpu.reduce_min(&input, &[0]).unwrap(),
                gpu.reduce_min(&gpu_input, &[0]).unwrap(),
            ),
        ] {
            assert_all_nan(&expected);
            assert_all_nan(&download(&gpu, &actual));
        }
    }
}

#[test]
#[ignore]
fn test_cubecl_complex_sum_and_prod_match_cpu() {
    let input = tensor_c64(
        vec![2, 2],
        vec![
            num_complex::Complex64::new(1.0, 1.0),
            num_complex::Complex64::new(2.0, -1.0),
            num_complex::Complex64::new(-0.5, 0.25),
            num_complex::Complex64::new(3.0, 2.0),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_prod(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu.reduce_max(&gpu_input, &[0]).unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_max", DType::C64);

    let err = gpu.reduce_min(&gpu_input, &[0]).unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_min", DType::C64);
}

#[test]
#[ignore]
fn test_cubecl_i64_sum_and_prod_match_cpu() {
    let input = tensor_i64(vec![2, 3, 2], vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[2]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_i32_sum_and_prod_match_cpu() {
    let input = tensor_i32(vec![2, 3, 2], vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[2]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_integer_reductions_wrap_on_overflow() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let input = tensor_i32(vec![2, 2], vec![i32::MAX, 1, i32::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, -1, i32::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_prod(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 1], vec![i64::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, 1, i32::MAX, -5]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 2], vec![i64::MIN, 7, i64::MAX, -9]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![1024], vec![i32::MAX; 1024]);
    let gpu_input = upload(&gpu, &input);
    for (expected, gpu_out) in [
        (
            cpu.reduce_sum(&input, &[0]).unwrap(),
            gpu.reduce_sum(&gpu_input, &[0]).unwrap(),
        ),
        (
            cpu.reduce_prod(&input, &[0]).unwrap(),
            gpu.reduce_prod(&gpu_input, &[0]).unwrap(),
        ),
    ] {
        let actual = download(&gpu, &gpu_out);
        assert_tensor_close(&actual, &expected, 0.0);
    }
}

#[test]
#[ignore]
fn test_cubecl_bool_reductions_are_unsupported() {
    let input = tensor_bool(vec![2, 3], vec![true, false, true, true, false, false]);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let err = gpu.reduce_sum(&gpu_input, &[0]).unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_sum", DType::Bool);

    let err = gpu.reduce_prod(&gpu_input, &[1]).unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_prod", DType::Bool);
}

#[test]
#[ignore]
fn test_cubecl_reductions_column_major_3d_axes_match_cpu() {
    let input = tensor_f64(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64 - 7.0).collect(),
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    for axes in [&[0][..], &[1][..], &[2][..], &[0, 2][..]] {
        let expected = cpu.reduce_sum(&input, axes).unwrap();
        let gpu_out = gpu.reduce_sum(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_prod(&input, axes).unwrap();
        let gpu_out = gpu.reduce_prod(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_max(&input, axes).unwrap();
        let gpu_out = gpu.reduce_max(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_min(&input, axes).unwrap();
        let gpu_out = gpu.reduce_min(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);
    }
}
