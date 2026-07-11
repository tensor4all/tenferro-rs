// Run with: cargo test --features cuda -- --ignored
use crate::config::{PadConfig, ScatterConfig, SliceConfig};
use num_complex::Complex64;
use tenferro_tensor::{Error, TensorIndexing};

use super::{
    assert_tensor_close, cpu_backend, diagonal_scatter_config, download, gpu_backend,
    simple_gather_config, tensor_bool, tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64,
    upload,
};

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_float_index_validation_matches_cpu() {
    fn check_index(
        cpu: &mut tenferro_cpu::CpuBackend,
        gpu: &mut crate::cubecl::CudaBackend,
        index: crate::Tensor,
        valid: bool,
        label: &str,
        failures: &mut Vec<String>,
    ) {
        let operand = tensor_f64(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let updates = tensor_f64(vec![1], vec![5.0]);
        let gather_indices = match &index {
            crate::Tensor::F32(values) => {
                tensor_f32(vec![1, 1], values.as_slice().unwrap().to_vec())
            }
            crate::Tensor::F64(values) => {
                tensor_f64(vec![1, 1], values.as_slice().unwrap().to_vec())
            }
            _ => unreachable!("matrix only contains float indices"),
        };
        let scatter_config = ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        };

        let gpu_operand = upload(gpu, &operand);
        let gpu_index = upload(gpu, &index);
        let gpu_gather_indices = upload(gpu, &gather_indices);
        let gpu_updates = upload(gpu, &updates);

        let cases = [
            (
                "dynamic_slice",
                cpu.dynamic_slice(&operand, &index, &[1]),
                gpu.dynamic_slice(&gpu_operand, &gpu_index, &[1]),
            ),
            (
                "gather",
                cpu.gather(&operand, &gather_indices, &simple_gather_config()),
                gpu.gather(&gpu_operand, &gpu_gather_indices, &simple_gather_config()),
            ),
            (
                "scatter",
                cpu.scatter(&operand, &gather_indices, &updates, &scatter_config),
                gpu.scatter(
                    &gpu_operand,
                    &gpu_gather_indices,
                    &gpu_updates,
                    &scatter_config,
                ),
            ),
        ];

        for (op, cpu_result, gpu_result) in cases {
            if valid {
                let expected = cpu_result.unwrap_or_else(|err| panic!("{label} {op} CPU: {err:?}"));
                let Ok(actual_gpu) = gpu_result else {
                    failures.push(format!("{label} {op}: CUDA rejected CPU-valid input"));
                    continue;
                };
                let actual = download(gpu, &actual_gpu);
                assert_tensor_close(&actual, &expected, 0.0);
            } else {
                let expected = cpu_result
                    .err()
                    .unwrap_or_else(|| panic!("{label} {op} CPU unexpectedly succeeded"));
                match gpu_result {
                    Ok(_) => failures.push(format!("{label} {op}: CUDA unexpectedly succeeded")),
                    Err(actual) if actual != expected => {
                        failures.push(format!("{label} {op}: CUDA {actual:?} != CPU {expected:?}"))
                    }
                    Err(_) => {}
                }
            }
        }
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let mut failures = Vec::new();
    for (label, value, valid) in [
        ("f32 integral", 1.0_f32, true),
        ("f32 fractional", 1.5, false),
        ("f32 NaN", f32::NAN, false),
        ("f32 +inf", f32::INFINITY, false),
        ("f32 -inf", f32::NEG_INFINITY, false),
        ("f32 +boundary", 16_777_216.0, true),
        ("f32 -boundary", -16_777_216.0, true),
        ("f32 +outside", 16_777_218.0, false),
        ("f32 -outside", -16_777_218.0, false),
    ] {
        check_index(
            &mut cpu,
            &mut gpu,
            tensor_f32(vec![1], vec![value]),
            valid,
            label,
            &mut failures,
        );
    }
    for (label, value, valid) in [
        ("f64 integral", 1.0_f64, true),
        ("f64 fractional", 1.5, false),
        ("f64 NaN", f64::NAN, false),
        ("f64 +inf", f64::INFINITY, false),
        ("f64 -inf", f64::NEG_INFINITY, false),
        ("f64 +boundary", 9_007_199_254_740_992.0, true),
        ("f64 -boundary", -9_007_199_254_740_992.0, true),
        ("f64 +outside", 9_007_199_254_740_994.0, false),
        ("f64 -outside", -9_007_199_254_740_994.0, false),
    ] {
        check_index(
            &mut cpu,
            &mut gpu,
            tensor_f64(vec![1], vec![value]),
            valid,
            label,
            &mut failures,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_float_index_validation_reports_first_invalid_value() {
    fn check_first_error(
        cpu: &mut tenferro_cpu::CpuBackend,
        gpu: &mut crate::cubecl::CudaBackend,
        starts: crate::Tensor,
        gather_indices: crate::Tensor,
        label: &str,
    ) {
        let operand = tensor_f64(vec![4, 4], (0..16).map(|value| value as f64).collect());
        let updates = tensor_f64(vec![2, 4], vec![1.0; 8]);
        let gather_config = crate::config::GatherConfig {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1, 4],
        };
        let scatter_config = ScatterConfig {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        };
        let gpu_operand = upload(gpu, &operand);
        let gpu_starts = upload(gpu, &starts);
        let gpu_indices = upload(gpu, &gather_indices);
        let gpu_updates = upload(gpu, &updates);

        for (op, expected, actual) in [
            (
                "dynamic_slice",
                cpu.dynamic_slice(&operand, &starts, &[1, 1]),
                gpu.dynamic_slice(&gpu_operand, &gpu_starts, &[1, 1]),
            ),
            (
                "gather",
                cpu.gather(&operand, &gather_indices, &gather_config),
                gpu.gather(&gpu_operand, &gpu_indices, &gather_config),
            ),
            (
                "scatter",
                cpu.scatter(&operand, &gather_indices, &updates, &scatter_config),
                gpu.scatter(&gpu_operand, &gpu_indices, &gpu_updates, &scatter_config),
            ),
        ] {
            assert_eq!(
                actual.unwrap_err(),
                expected.unwrap_err(),
                "{label} {op} must report the lower flat invalid index"
            );
        }
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    // The higher flat indices take the short non-finite branch while the
    // lower indices require fractional/bounds checks, so concurrent workers
    // may publish the higher invalid index first. Atomic-min must still retain
    // the lower flat index and therefore CPU's first-value error message.
    check_first_error(
        &mut cpu,
        &mut gpu,
        tensor_f32(vec![2], vec![1.5, f32::NAN]),
        tensor_f32(vec![2, 1], vec![1.5, f32::NAN]),
        "F32 fractional before NaN",
    );
    check_first_error(
        &mut cpu,
        &mut gpu,
        tensor_f64(vec![2], vec![9_007_199_254_740_994.0, f64::NEG_INFINITY]),
        tensor_f64(vec![2, 1], vec![9_007_199_254_740_994.0, f64::NEG_INFINITY]),
        "F64 outside-bound before -inf",
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_bool_indexing_ops_match_cpu() {
    let input = tensor_bool(vec![4], vec![true, false, true, false]);
    let starts = tensor_i64(vec![1], vec![1]);
    let indices = tensor_i64(vec![2, 1], vec![0, 3]);
    let starts_i32 = tensor_i32(vec![1], vec![1]);
    let empty = tensor_bool(vec![0], vec![]);
    let empty_starts = tensor_i64(vec![1], vec![0]);
    let empty_indices = tensor_i64(vec![0, 1], vec![]);
    let slice = SliceConfig {
        starts: vec![1],
        limits: vec![4],
        strides: vec![2],
    };
    let pad = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![1],
    };
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gi = upload(&gpu, &input);
    let gs = upload(&gpu, &starts);
    let gx = upload(&gpu, &indices);
    let gs_i32 = upload(&gpu, &starts_i32);
    let ge = upload(&gpu, &empty);
    let ges = upload(&gpu, &empty_starts);
    let gex = upload(&gpu, &empty_indices);
    macro_rules! parity {
        ($cpu:expr, $gpu:expr) => {{
            let expected = $cpu.unwrap();
            let out = $gpu.unwrap();
            let actual = download(&gpu, &out);
            assert_tensor_close(&actual, &expected, 0.0);
        }};
    }
    macro_rules! error_parity {
        ($cpu:expr, $gpu:expr) => {{
            assert_eq!($cpu.unwrap_err(), $gpu.unwrap_err());
        }};
    }
    parity!(cpu.slice(&input, &slice), gpu.slice(&gi, &slice));
    parity!(
        cpu.dynamic_slice(&input, &starts, &[2]),
        gpu.dynamic_slice(&gi, &gs, &[2])
    );
    parity!(
        cpu.dynamic_slice(&input, &starts_i32, &[2]),
        gpu.dynamic_slice(&gi, &gs_i32, &[2])
    );
    parity!(cpu.pad(&input, &pad), gpu.pad(&gi, &pad));
    parity!(
        cpu.gather(&input, &indices, &simple_gather_config()),
        gpu.gather(&gi, &gx, &simple_gather_config())
    );
    let invalid = SliceConfig {
        starts: vec![0],
        limits: vec![5],
        strides: vec![1],
    };
    let empty_slice = SliceConfig {
        starts: vec![0],
        limits: vec![0],
        strides: vec![1],
    };
    let empty_pad = PadConfig {
        edge_padding_low: vec![0],
        edge_padding_high: vec![0],
        interior_padding: vec![0],
    };
    parity!(
        cpu.slice(&empty, &empty_slice),
        gpu.slice(&ge, &empty_slice)
    );
    parity!(
        cpu.dynamic_slice(&empty, &empty_starts, &[0]),
        gpu.dynamic_slice(&ge, &ges, &[0])
    );
    parity!(cpu.pad(&empty, &empty_pad), gpu.pad(&ge, &empty_pad));
    parity!(
        cpu.gather(&input, &empty_indices, &simple_gather_config()),
        gpu.gather(&gi, &gex, &simple_gather_config())
    );

    error_parity!(cpu.slice(&input, &invalid), gpu.slice(&gi, &invalid));
    let bad_starts = tensor_i64(vec![2], vec![0, 1]);
    let gpu_bad_starts = upload(&gpu, &bad_starts);
    error_parity!(
        cpu.dynamic_slice(&input, &bad_starts, &[2]),
        gpu.dynamic_slice(&gi, &gpu_bad_starts, &[2])
    );
    let bad_pad = PadConfig {
        edge_padding_low: vec![],
        edge_padding_high: vec![],
        interior_padding: vec![],
    };
    error_parity!(cpu.pad(&input, &bad_pad), gpu.pad(&gi, &bad_pad));
    let bad_gather = crate::config::GatherConfig {
        start_index_map: vec![1],
        ..simple_gather_config()
    };
    error_parity!(
        cpu.gather(&input, &indices, &bad_gather),
        gpu.gather(&gi, &gx, &bad_gather)
    );

    let updates = tensor_bool(vec![2], vec![true, false]);
    let scatter_indices = tensor_i64(vec![2, 1], vec![0, 1]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let expected_scatter_error = Error::BackendFailure {
        op: "scatter",
        message: "Bool data tensors are not supported by additive scatter".into(),
    };
    assert_eq!(
        cpu.scatter(&input, &scatter_indices, &updates, &config)
            .unwrap_err(),
        expected_scatter_error
    );
    let gpu_updates = upload(&gpu, &updates);
    let gpu_scatter_indices = upload(&gpu, &scatter_indices);
    assert_eq!(
        gpu.scatter(&gi, &gpu_scatter_indices, &gpu_updates, &config)
            .unwrap_err(),
        expected_scatter_error
    );
}

#[test]
#[ignore]
fn test_cubecl_slice_dynamic_slice_and_pad_match_cpu() {
    let input = tensor_f64(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let slice_config = SliceConfig {
        starts: vec![1, 1],
        limits: vec![4, 4],
        strides: vec![2, 1],
    };
    let pad_config = PadConfig {
        edge_padding_low: vec![1, 1],
        edge_padding_high: vec![1, 0],
        interior_padding: vec![0, 1],
    };
    let starts = tensor_i64(vec![2], vec![2, 3]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_starts = upload(&gpu, &starts);

    let expected = cpu.slice(&input, &slice_config).unwrap();
    let gpu_out = gpu.slice(&gpu_input, &slice_config).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.dynamic_slice(&input, &starts, &[2, 2]).unwrap();
    let gpu_out = gpu.dynamic_slice(&gpu_input, &gpu_starts, &[2, 2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.pad(&input, &pad_config).unwrap();
    let gpu_out = gpu.pad(&gpu_input, &pad_config).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_signed_pad_cropping_matches_cpu_values() {
    let cases = [
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![-1],
                edge_padding_high: vec![0],
                interior_padding: vec![0],
            },
            tensor_f64(vec![3], vec![2.0, 3.0, 4.0]),
        ),
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![-1],
                interior_padding: vec![0],
            },
            tensor_f64(vec![3], vec![1.0, 2.0, 3.0]),
        ),
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![-1],
                edge_padding_high: vec![0],
                interior_padding: vec![1],
            },
            tensor_f64(vec![6], vec![0.0, 2.0, 0.0, 3.0, 0.0, 4.0]),
        ),
        (
            tensor_f64(vec![2], vec![1.0, 2.0]),
            PadConfig {
                edge_padding_low: vec![i64::MIN],
                edge_padding_high: vec![i64::MAX],
                interior_padding: vec![0],
            },
            tensor_f64(vec![1], vec![0.0]),
        ),
    ];

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for (input, config, expected_values) in cases {
        let expected = cpu.pad(&input, &config).unwrap();
        assert_eq!(expected.shape(), expected_values.shape());
        assert_tensor_close(&expected, &expected_values, 0.0);

        let gpu_input = upload(&gpu, &input);
        let gpu_output = gpu.pad(&gpu_input, &config).unwrap();
        let actual = download(&gpu, &gpu_output);
        assert_eq!(actual.shape(), expected_values.shape());
        assert_tensor_close(&actual, &expected_values, 0.0);
        assert_tensor_close(&actual, &expected, 0.0);
    }
}

#[test]
#[ignore]
fn test_cubecl_gather_and_scatter_match_cpu() {
    let operand = tensor_f64(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let start_indices = tensor_i64(vec![3, 1], vec![0, 2, 4]);

    let scatter_operand = tensor_f64(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );
    let scatter_indices = tensor_i64(vec![3, 2], vec![0, 1, 2, 0, 1, 2]);
    let updates = tensor_f64(vec![3], vec![5.0, 6.0, 7.0]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let gpu_operand = upload(&gpu, &operand);
    let gpu_start_indices = upload(&gpu, &start_indices);
    let expected = cpu
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap();
    let gpu_out = gpu
        .gather(&gpu_operand, &gpu_start_indices, &simple_gather_config())
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let gpu_scatter_operand = upload(&gpu, &scatter_operand);
    let gpu_scatter_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);
    let expected = cpu
        .scatter(
            &scatter_operand,
            &scatter_indices,
            &updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    let gpu_out = gpu
        .scatter(
            &gpu_scatter_operand,
            &gpu_scatter_indices,
            &gpu_updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_scatter_skips_invalid_windows_like_cpu() {
    let operand = tensor_f64(vec![4], vec![9.0, 8.0, 7.0, 6.0]);
    let scatter_indices = tensor_i64(vec![3, 1], vec![-1, 2, 4]);
    let updates = tensor_f64(vec![3], vec![5.0, 6.0, 7.0]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_operand = upload(&gpu, &operand);
    let gpu_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);

    let expected = cpu
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap();
    let gpu_out = gpu
        .scatter(&gpu_operand, &gpu_indices, &gpu_updates, &config)
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_scatter_accumulates_overlapping_updates_like_cpu() {
    let operand = tensor_f64(vec![4], vec![1.0, 10.0, 100.0, 1000.0]);
    let scatter_indices = tensor_i64(vec![4, 1], vec![1, 1, -1, 3]);
    let updates = tensor_f64(vec![4], vec![2.0, 3.0, 99.0, 5.0]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_operand = upload(&gpu, &operand);
    let gpu_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);

    let expected = cpu
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap();
    let gpu_out = gpu
        .scatter(&gpu_operand, &gpu_indices, &gpu_updates, &config)
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_complex_scatter_accumulates_overlapping_updates_like_cpu() {
    let operand = tensor_c64(
        vec![3],
        vec![
            Complex64::new(1.0, 10.0),
            Complex64::new(2.0, 20.0),
            Complex64::new(3.0, 30.0),
        ],
    );
    let scatter_indices = tensor_i64(vec![3, 1], vec![0, 0, 2]);
    let updates = tensor_c64(
        vec![3],
        vec![
            Complex64::new(4.0, 1.0),
            Complex64::new(5.0, 2.0),
            Complex64::new(6.0, 3.0),
        ],
    );
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_operand = upload(&gpu, &operand);
    let gpu_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);

    let expected = cpu
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap();
    let gpu_out = gpu
        .scatter(&gpu_operand, &gpu_indices, &gpu_updates, &config)
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}
