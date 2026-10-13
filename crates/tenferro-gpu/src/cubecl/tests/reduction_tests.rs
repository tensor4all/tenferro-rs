// Run with: cargo test --features cuda -- --ignored
use tenferro_tensor::{DType, TensorRead, TensorView};

use super::{
    assert_cuda_unsupported_dtype, assert_tensor_close, cpu_backend, download, gpu_backend,
    tensor_bool, tensor_c32, tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};
use tenferro_tensor::BackendSessionHost;

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
                cpu.with_backend_session(|__s| {
                    __s.reduce_sum_read(TensorRead::from_tensor(input), &[0, 1])
                })
                .unwrap()
                .unwrap(),
                gpu.with_backend_session(|__s| {
                    __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0, 1])
                })
                .unwrap()
                .unwrap(),
            ),
            (
                cpu.with_backend_session(|__s| {
                    __s.reduce_prod_read(TensorRead::from_tensor(input), &[0, 1])
                })
                .unwrap()
                .unwrap(),
                gpu.with_backend_session(|__s| {
                    __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[0, 1])
                })
                .unwrap()
                .unwrap(),
            ),
        ] {
            assert!(expected.shape().is_empty());
            assert!(gpu_out.shape().is_empty());
            let actual = download(&gpu, &gpu_out);
            assert_tensor_close(&actual, &expected, 1e-5);
        }

        if !matches!(input.dtype(), crate::DType::C32 | crate::DType::C64) {
            for (expected, gpu_out) in [
                (
                    cpu.with_backend_session(|__s| {
                        __s.reduce_min_read(TensorRead::from_tensor(input), &[0, 1])
                    })
                    .unwrap()
                    .unwrap(),
                    gpu.with_backend_session(|__s| {
                        __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[0, 1])
                    })
                    .unwrap()
                    .unwrap(),
                ),
                (
                    cpu.with_backend_session(|__s| {
                        __s.reduce_max_read(TensorRead::from_tensor(input), &[0, 1])
                    })
                    .unwrap()
                    .unwrap(),
                    gpu.with_backend_session(|__s| {
                        __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0, 1])
                    })
                    .unwrap()
                    .unwrap(),
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

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[1]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&input), &[1]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap();
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
                .with_backend_session(|__s| {
                    __s.reduce_sum_squares_read(TensorRead::from_tensor(input), axes)
                })
                .unwrap()
                .unwrap();
            let gpu_output = gpu
                .with_backend_session(|__s| {
                    __s.reduce_sum_squares_read(TensorRead::from_tensor(&gpu_input), axes)
                })
                .unwrap()
                .unwrap();
            assert_tensor_close(&download(&gpu, &gpu_output), &expected, 1e-5);
            let gpu_view = match gpu_input.dtype() {
                DType::F32 => TensorView::F32(gpu_input.as_typed::<f32>().unwrap().as_view()),
                DType::F64 => TensorView::F64(gpu_input.as_typed::<f64>().unwrap().as_view()),
                _ => unreachable!("sum-of-squares test uses float inputs"),
            };
            let view_output = gpu
                .with_backend_session(|session| {
                    session.reduce_sum_squares_read(TensorRead::from_view(gpu_view), axes)
                })
                .unwrap()
                .unwrap();
            assert_tensor_close(&download(&gpu, &view_output), &expected, 1e-5);
        }
    }

    let integer = upload(&gpu, &tensor_i32(vec![2], vec![1, 2]));
    let error = gpu
        .with_backend_session(|__s| {
            __s.reduce_sum_squares_read(TensorRead::from_tensor(&integer), &[0])
        })
        .unwrap()
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
        .with_backend_session(|__s| {
            __s.reduce_sum_squares_read(TensorRead::from_tensor(&gpu_input), &[0])
        })
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_output);
    let Some(actual) = actual.as_typed::<f32>() else {
        panic!("sum-of-squares output must remain f32");
    };
    let actual = actual.as_slice().unwrap()[0];

    assert_eq!(actual.to_bits(), expected.to_bits());
    assert_ne!(actual.to_bits(), contracted.to_bits());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_float_max_min_reductions_propagate_nan_for_unit_and_plane() {
    assert!(
        crate::cubecl::gpu_available(),
        "CUDA test requires an available device"
    );

    fn assert_all_nan(tensor: &crate::Tensor) {
        match tensor.dtype() {
            crate::DType::F32 => {
                let tensor = tensor
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm");
                assert!(tensor
                    .as_slice()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()));
            }
            crate::DType::F64 => {
                let tensor = tensor
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm");
                assert!(tensor
                    .as_slice()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan()));
            }
            other => panic!("expected float reduction output, got {:?}", other),
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
                cpu.with_backend_session(|__s| {
                    __s.reduce_max_read(TensorRead::from_tensor(&input), &[0])
                })
                .unwrap()
                .unwrap(),
                gpu.with_backend_session(|__s| {
                    __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0])
                })
                .unwrap()
                .unwrap(),
            ),
            (
                cpu.with_backend_session(|__s| {
                    __s.reduce_min_read(TensorRead::from_tensor(&input), &[0])
                })
                .unwrap()
                .unwrap(),
                gpu.with_backend_session(|__s| {
                    __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[0])
                })
                .unwrap()
                .unwrap(),
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

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[1]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_max", DType::C64);

    let err = gpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_min", DType::C64);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_cubecl_complex_sum_and_prod_match_cpu_on_plane_path() {
    if !crate::cubecl::gpu_available() {
        eprintln!("skipping complex plane reduction parity test - no CUDA device found");
        return;
    }

    // Axes longer than the warp/plane size force the plane (warp-shuffle)
    // reduction kernel, which must shuffle the complex components separately.
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for len in [33usize, 64] {
        let c64: Vec<_> = (0..len)
            .map(|i| num_complex::Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0) * 0.25)
            .collect();
        let c32: Vec<_> = (0..len)
            .map(|i| num_complex::Complex32::new((i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0) * 0.25)
            .collect();

        for input in [tensor_c64(vec![len, 1], c64), tensor_c32(vec![len, 1], c32)] {
            let gpu_input = upload(&gpu, &input);
            for (expected, actual) in [
                (
                    cpu.with_backend_session(|__s| {
                        __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0])
                    })
                    .unwrap()
                    .unwrap(),
                    gpu.with_backend_session(|__s| {
                        __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0])
                    })
                    .unwrap()
                    .unwrap(),
                ),
                (
                    cpu.with_backend_session(|__s| {
                        __s.reduce_prod_read(TensorRead::from_tensor(&input), &[0])
                    })
                    .unwrap()
                    .unwrap(),
                    gpu.with_backend_session(|__s| {
                        __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[0])
                    })
                    .unwrap()
                    .unwrap(),
                ),
            ] {
                assert_tensor_close(&download(&gpu, &actual), &expected, 1e-5);
            }
        }
    }
}

#[test]
#[ignore]
fn test_cubecl_i64_sum_and_prod_match_cpu() {
    let input = tensor_i64(vec![2, 3, 2], vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[2]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[2]))
        .unwrap()
        .unwrap();
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

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[2]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[2]))
        .unwrap()
        .unwrap();
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
    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, -1, i32::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 1], vec![i64::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, 1, i32::MAX, -5]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&input), &[1]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 2], vec![i64::MIN, 7, i64::MAX, -9]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&input), &[0]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&input), &[1]))
        .unwrap()
        .unwrap();
    let gpu_out = gpu
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![1024], vec![i32::MAX; 1024]);
    let gpu_input = upload(&gpu, &input);
    for (expected, gpu_out) in [
        (
            cpu.with_backend_session(|__s| {
                __s.reduce_sum_read(TensorRead::from_tensor(&input), &[0])
            })
            .unwrap()
            .unwrap(),
            gpu.with_backend_session(|__s| {
                __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0])
            })
            .unwrap()
            .unwrap(),
        ),
        (
            cpu.with_backend_session(|__s| {
                __s.reduce_prod_read(TensorRead::from_tensor(&input), &[0])
            })
            .unwrap()
            .unwrap(),
            gpu.with_backend_session(|__s| {
                __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[0])
            })
            .unwrap()
            .unwrap(),
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

    let err = gpu
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0]))
        .unwrap()
        .unwrap_err();
    assert_cuda_unsupported_dtype(&err, "reduce_sum", DType::Bool);

    let err = gpu
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), &[1]))
        .unwrap()
        .unwrap_err();
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
        let expected = cpu
            .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&input), axes))
            .unwrap()
            .unwrap();
        let gpu_out = gpu
            .with_backend_session(|__s| {
                __s.reduce_sum_read(TensorRead::from_tensor(&gpu_input), axes)
            })
            .unwrap()
            .unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu
            .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&input), axes))
            .unwrap()
            .unwrap();
        let gpu_out = gpu
            .with_backend_session(|__s| {
                __s.reduce_prod_read(TensorRead::from_tensor(&gpu_input), axes)
            })
            .unwrap()
            .unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu
            .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&input), axes))
            .unwrap()
            .unwrap();
        let gpu_out = gpu
            .with_backend_session(|__s| {
                __s.reduce_max_read(TensorRead::from_tensor(&gpu_input), axes)
            })
            .unwrap()
            .unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu
            .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&input), axes))
            .unwrap()
            .unwrap();
        let gpu_out = gpu
            .with_backend_session(|__s| {
                __s.reduce_min_read(TensorRead::from_tensor(&gpu_input), axes)
            })
            .unwrap()
            .unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);
    }
}

/// A full `reduce_sum` over a complex tensor must compile and reduce
/// componentwise (#2133).
///
/// The CUDA backend shuffles each component separately on the plane path. The
/// 0.10 pins emitted `__shfl_xor_sync` on `cuFloatComplex` itself, which NVRTC
/// refuses (`no instance of overloaded function "__shfl_xor_sync" matches the
/// argument list`); the fork's `feat(plane): lower complex plane reductions for
/// CUDA` fixed it, and this pins the behaviour so a future pin cannot drop it
/// silently.
///
/// `len` is the interesting parameter: 1 takes the unit path, 1e3 and 1e6 the
/// plane path with one output element, which is also the single-warp case #2134
/// measures.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_cubecl_complex_full_reduction_compiles_and_matches_cpu() {
    if !crate::cubecl::gpu_available() {
        eprintln!("skipping complex full reduction test - no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for len in [1usize, 1_000, 1_000_000] {
        let c64: Vec<_> = (0..len)
            .map(|i| num_complex::Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0))
            .collect();
        let c32: Vec<_> = (0..len)
            .map(|i| num_complex::Complex32::new((i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0))
            .collect();

        for (label, input) in [
            ("c64", tensor_c64(vec![len], c64)),
            ("c32", tensor_c32(vec![len], c32)),
        ] {
            let gpu_input = upload(&gpu, &input);

            let expected = cpu
                .with_backend_session(|session| {
                    session.reduce_sum_read(TensorRead::from_tensor(&input), &[0])
                })
                .unwrap()
                .unwrap();
            let actual = gpu
                .with_backend_session(|session| {
                    session.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0])
                })
                .unwrap()
                .unwrap();
            let actual = download(&gpu, &actual);
            assert!(
                actual.shape().is_empty(),
                "{label} len={len}: reducing the only axis of a rank-1 input must give a rank-0 result, got {:?}",
                actual.shape()
            );
            assert_tensor_close(&actual, &expected, 1e-4);

            // The reported sequence: `reduce_sum(sub(x, x))` is exactly zero.
            let difference = gpu
                .with_backend_session(|session| {
                    session.sub_read(
                        TensorRead::from_tensor(&gpu_input),
                        TensorRead::from_tensor(&gpu_input),
                    )
                })
                .unwrap()
                .unwrap();
            let reduced = gpu
                .with_backend_session(|session| {
                    session.reduce_sum_read(TensorRead::from_tensor(&difference), &[0])
                })
                .unwrap()
                .unwrap();
            let reduced = download(&gpu, &reduced);
            match reduced.dtype() {
                DType::C64 => {
                    for (index, value) in reduced
                        .as_slice::<num_complex::Complex64>()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        assert_eq!(
                            (value.re, value.im),
                            (0.0, 0.0),
                            "{label} len={len} [{index}]: reduce_sum(sub(x, x)) must be zero"
                        );
                    }
                }
                DType::C32 => {
                    for (index, value) in reduced
                        .as_slice::<num_complex::Complex32>()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        assert_eq!(
                            (value.re, value.im),
                            (0.0, 0.0),
                            "{label} len={len} [{index}]: reduce_sum(sub(x, x)) must be zero"
                        );
                    }
                }
                other => panic!("{label} len={len}: unexpected dtype {other:?}"),
            }
        }
    }
}

/// A long-axis full reduction is split across cubes (#2134): pass 1 writes
/// per-chunk partials and pass 2 combines them, instead of one warp walking the
/// whole axis.
///
/// The lengths straddle the split threshold (`2 * MIN_CHUNK = 512`) and the
/// remainder case (513, where the last chunk runs to `axis_len` rather than to
/// `start + chunk_len`), then cover the sizes the acceptance measures. The first
/// launch of each kernel compiles it with NVRTC, so the warm time is printed for
/// the record and only correctness is asserted.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_cubecl_split_reduction_large_axis_matches_cpu() {
    if !crate::cubecl::gpu_available() {
        eprintln!("skipping split reduction test - no CUDA device found");
        return;
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let runtime = gpu.runtime().clone();
    for len in [511usize, 512, 513, 600, 1_000_000, 10_000_000] {
        let values: Vec<f64> = (0..len).map(|i| ((i % 17) as f64) * 0.25 - 2.0).collect();
        let expected: f64 = values.iter().sum();
        let input = tensor_f64(vec![len], values);
        let gpu_input = upload(&gpu, &input);

        // The expected value also comes from the CPU backend, so a wrong host
        // sum cannot make a wrong device result look right.
        let reference = cpu
            .with_backend_session(|session| {
                session.reduce_sum_read(TensorRead::from_tensor(&input), &[0])
            })
            .unwrap()
            .unwrap();
        assert!(
            (reference.as_slice::<f64>().unwrap()[0] - expected).abs()
                <= 1e-9 * expected.abs().max(1.0),
            "len={len}: the CPU reference must agree with the host sum"
        );

        let mut warm = std::time::Duration::ZERO;
        let mut actual = None;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            actual = Some(
                gpu.with_backend_session(|session| {
                    session.reduce_sum_read(TensorRead::from_tensor(&gpu_input), &[0])
                })
                .unwrap()
                .unwrap(),
            );
            runtime.synchronize().unwrap();
            warm = started.elapsed();
        }
        eprintln!("len={len}: warm device time {warm:?}");

        let actual = download(&gpu, &actual.unwrap());
        let got = actual.as_slice::<f64>().unwrap()[0];
        assert!((got - expected).abs() <= 1e-9 * expected.abs().max(1.0), "len={len}: got {got}, expected {expected}");

        // The reported exactness case survives the split: every chunk sums to
        // exactly zero, so the combine pass does too.
        let difference = gpu
            .with_backend_session(|session| {
                session.sub_read(
                    TensorRead::from_tensor(&gpu_input),
                    TensorRead::from_tensor(&gpu_input),
                )
            })
            .unwrap()
            .unwrap();
        let reduced = gpu
            .with_backend_session(|session| {
                session.reduce_sum_read(TensorRead::from_tensor(&difference), &[0])
            })
            .unwrap()
            .unwrap();
        assert_eq!(
            download(&gpu, &reduced).as_slice::<f64>().unwrap()[0],
            0.0,
            "len={len}: reduce_sum(sub(x, x)) must stay exactly zero"
        );
    }
}
