// Run with: cargo test --features cuda -- --ignored
use crate::config::{PadConfig, ScatterConfig, SliceConfig};
use num_complex::Complex64;
use tenferro_tensor::{
    Buffer, DeviceId, DeviceKind, Error, GpuBackendKind, MemoryKind, Placement, TensorIndexing,
    TypedTensor,
};

use super::{
    assert_error_parity, assert_runtime_state, assert_tensor_close, assert_unsupported,
    assert_validation_kind, cpu_backend, diagonal_scatter_config, download, gpu_backend,
    simple_gather_config, tensor_bool, tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64,
    upload,
};
use tenferro_tensor::{ValidationError, ValidationKind};

fn with_cuda_ordinal<T: Clone + 'static>(
    tensor: &TypedTensor<T>,
    ordinal: usize,
) -> TypedTensor<T> {
    TypedTensor::from_buffer_col_major(
        tensor.shape().to_vec(),
        tensor.buffer().clone(),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal,
            }),
        },
    )
    .unwrap()
}

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
                    Err(actual) if actual.kind() != expected.kind() => {
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
fn cuda_bool_dynamic_slice_float_starts_match_cpu() {
    let input = tensor_bool(vec![4], vec![true, false, true, false]);
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    for (label, starts) in [
        ("F32 integral", tensor_f32(vec![1], vec![1.0])),
        (
            "F32 positive exact boundary",
            tensor_f32(vec![1], vec![16_777_216.0]),
        ),
        (
            "F32 negative exact boundary",
            tensor_f32(vec![1], vec![-16_777_216.0]),
        ),
        ("F64 integral", tensor_f64(vec![1], vec![2.0])),
        (
            "F64 positive exact boundary",
            tensor_f64(vec![1], vec![9_007_199_254_740_992.0]),
        ),
        (
            "F64 negative exact boundary",
            tensor_f64(vec![1], vec![-9_007_199_254_740_992.0]),
        ),
    ] {
        let expected = cpu.dynamic_slice(&input, &starts, &[2]).unwrap();
        let actual_gpu = gpu
            .dynamic_slice(&gpu_input, &upload(&gpu, &starts), &[2])
            .unwrap_or_else(|err| panic!("{label}: {err:?}"));
        assert_tensor_close(&download(&gpu, &actual_gpu), &expected, 0.0);
    }

    for (label, starts) in [
        ("F32 fractional", tensor_f32(vec![1], vec![1.5])),
        ("F32 NaN", tensor_f32(vec![1], vec![f32::NAN])),
        (
            "F32 positive infinity",
            tensor_f32(vec![1], vec![f32::INFINITY]),
        ),
        (
            "F32 negative infinity",
            tensor_f32(vec![1], vec![f32::NEG_INFINITY]),
        ),
        (
            "F32 positive outside exact bounds",
            tensor_f32(vec![1], vec![16_777_218.0]),
        ),
        (
            "F32 negative outside exact bounds",
            tensor_f32(vec![1], vec![-16_777_218.0]),
        ),
        ("F64 fractional", tensor_f64(vec![1], vec![1.5])),
        ("F64 NaN", tensor_f64(vec![1], vec![f64::NAN])),
        (
            "F64 positive infinity",
            tensor_f64(vec![1], vec![f64::INFINITY]),
        ),
        (
            "F64 negative infinity",
            tensor_f64(vec![1], vec![f64::NEG_INFINITY]),
        ),
        (
            "F64 positive outside exact bounds",
            tensor_f64(vec![1], vec![9_007_199_254_740_994.0]),
        ),
        (
            "F64 negative outside exact bounds",
            tensor_f64(vec![1], vec![-9_007_199_254_740_994.0]),
        ),
    ] {
        let actual = gpu
            .dynamic_slice(&gpu_input, &upload(&gpu, &starts), &[2])
            .unwrap_err();
        let expected = cpu.dynamic_slice(&input, &starts, &[2]).unwrap_err();
        assert_eq!(actual.kind(), expected.kind(), "{label}");
        assert_error_parity(expected, actual);
    }

    let invalid_starts = tensor_f64(vec![1], vec![f64::NAN]);
    let err = gpu
        .dynamic_slice(&gpu_input, &upload(&gpu, &invalid_starts), &[5])
        .unwrap_err();
    assert_validation_kind(&err, "dynamic_slice", ValidationKind::InvalidArgument);
    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::InvalidArgument { argument: "slice_sizes", message },
            ..
        } if message.contains("slice size exceeds dimension on axis 0")
    ));

    let empty = tensor_bool(vec![0], vec![]);
    for starts in [
        tensor_f32(vec![1], vec![0.0]),
        tensor_f64(vec![1], vec![0.0]),
    ] {
        let expected = cpu.dynamic_slice(&empty, &starts, &[0]).unwrap();
        let actual_gpu = gpu
            .dynamic_slice(&upload(&gpu, &empty), &upload(&gpu, &starts), &[0])
            .unwrap();
        assert_tensor_close(&download(&gpu, &actual_gpu), &expected, 0.0);
        let expected = cpu.dynamic_slice(&input, &starts, &[0]).unwrap();
        let actual_gpu = gpu
            .dynamic_slice(&gpu_input, &upload(&gpu, &starts), &[0])
            .unwrap();
        assert_tensor_close(&download(&gpu, &actual_gpu), &expected, 0.0);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_indexing_invalid_config_precedes_invalid_float_index_values() {
    fn assert_result_error_parity(
        cpu: tenferro_tensor::Result<crate::Tensor>,
        gpu: tenferro_tensor::Result<crate::Tensor>,
    ) {
        super::assert_error_parity(cpu.unwrap_err(), gpu.unwrap_err());
    }

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let operand_f64 = tensor_f64(vec![2], vec![1.0, 2.0]);
    let operand_f32 = tensor_f32(vec![2], vec![1.0, 2.0]);
    let starts_f64 = tensor_f64(vec![1], vec![f64::NAN]);
    let starts_f32 = tensor_f32(vec![1], vec![0.5]);
    let valid_starts = tensor_i32(vec![1], vec![0]);
    for (operand, invalid_starts) in [
        (operand_f64.clone(), starts_f64),
        (operand_f32.clone(), starts_f32),
    ] {
        assert!(cpu.dynamic_slice(&operand, &valid_starts, &[3]).is_err());
        let err = gpu
            .dynamic_slice(
                &upload(&gpu, &operand),
                &upload(&gpu, &invalid_starts),
                &[3],
            )
            .unwrap_err();
        assert_validation_kind(&err, "dynamic_slice", ValidationKind::InvalidArgument);
        assert_result_error_parity(
            cpu.dynamic_slice(&operand, &invalid_starts, &[1]),
            gpu.dynamic_slice(
                &upload(&gpu, &operand),
                &upload(&gpu, &invalid_starts),
                &[1],
            ),
        );
    }

    let operand_bool = tensor_bool(vec![2], vec![true, false]);
    let starts_i32 = tensor_i32(vec![2], vec![0, 1]);
    assert_result_error_parity(
        cpu.dynamic_slice(&operand_bool, &starts_i32, &[1]),
        gpu.dynamic_slice(
            &upload(&gpu, &operand_bool),
            &upload(&gpu, &starts_i32),
            &[1],
        ),
    );

    let bad_gather = crate::config::GatherConfig {
        start_index_map: vec![1],
        ..simple_gather_config()
    };
    let gather_indices = tensor_f64(vec![1, 1], vec![f64::NAN]);
    let valid_gather_indices = tensor_i32(vec![1, 1], vec![0]);
    for operand in [
        operand_f64.clone(),
        tensor_c64(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        ),
        operand_bool.clone(),
    ] {
        let expected = cpu
            .gather(&operand, &valid_gather_indices, &bad_gather)
            .unwrap_err();
        let actual = gpu
            .gather(
                &upload(&gpu, &operand),
                &upload(&gpu, &gather_indices),
                &bad_gather,
            )
            .unwrap_err();
        assert_error_parity(expected, actual);
        assert_result_error_parity(
            cpu.gather(&operand, &gather_indices, &simple_gather_config()),
            gpu.gather(
                &upload(&gpu, &operand),
                &upload(&gpu, &gather_indices),
                &simple_gather_config(),
            ),
        );
    }

    let bad_scatter = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![1],
        index_vector_dim: 1,
    };
    let valid_scatter_indices = tensor_i32(vec![1, 1], vec![0]);
    let valid_scatter = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let updates_f64 = tensor_f64(vec![1], vec![3.0]);
    let expected = cpu
        .scatter(
            &operand_f64,
            &valid_scatter_indices,
            &updates_f64,
            &bad_scatter,
        )
        .unwrap_err();
    let actual = gpu
        .scatter(
            &upload(&gpu, &operand_f64),
            &upload(&gpu, &gather_indices),
            &upload(&gpu, &updates_f64),
            &bad_scatter,
        )
        .unwrap_err();
    assert_error_parity(expected, actual);
    assert_result_error_parity(
        cpu.scatter(&operand_f64, &gather_indices, &updates_f64, &valid_scatter),
        gpu.scatter(
            &upload(&gpu, &operand_f64),
            &upload(&gpu, &gather_indices),
            &upload(&gpu, &updates_f64),
            &valid_scatter,
        ),
    );
    let operand_c64 = tensor_c64(vec![2], vec![Complex64::new(1.0, 2.0); 2]);
    let updates_c64 = tensor_c64(vec![1], vec![Complex64::new(3.0, 4.0)]);
    let expected = cpu
        .scatter(
            &operand_c64,
            &valid_scatter_indices,
            &updates_c64,
            &bad_scatter,
        )
        .unwrap_err();
    let actual = gpu
        .scatter(
            &upload(&gpu, &operand_c64),
            &upload(&gpu, &gather_indices),
            &upload(&gpu, &updates_c64),
            &bad_scatter,
        )
        .unwrap_err();
    assert_error_parity(expected, actual);
    assert_result_error_parity(
        cpu.scatter(&operand_c64, &gather_indices, &updates_c64, &valid_scatter),
        gpu.scatter(
            &upload(&gpu, &operand_c64),
            &upload(&gpu, &gather_indices),
            &upload(&gpu, &updates_c64),
            &valid_scatter,
        ),
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_indexing_zero_domains_validate_wrong_device_and_malformed_buffers() {
    let mut gpu = gpu_backend();
    let other_gpu = gpu_backend();
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let empty_operand = upload(&gpu, &tensor_f64(vec![0], vec![]));
    let indices = upload(&gpu, &tensor_i32(vec![0, 1], vec![]));
    let other_indices = upload(&other_gpu, &tensor_i32(vec![0, 1], vec![]));
    // INVARIANT: CUDA runtime residency is identified by device ordinal, not
    // by the `CudaBackend` wrapper. Same-device CubeCL clients share the
    // primary context, so a second cuda:0 backend remains compatible.
    gpu.scatter(
        &empty_operand,
        &other_indices,
        &upload(&gpu, &tensor_f64(vec![0], vec![])),
        &config,
    )
    .unwrap();

    let wrong_device_message =
        "expected GPU tensor resident on cuda:0, got Gpu(Cuda):1".to_string();
    let wrong_starts = crate::Tensor::I32(with_cuda_ordinal(
        match &upload(&gpu, &tensor_i32(vec![1], vec![0])) {
            crate::Tensor::I32(tensor) => tensor,
            _ => unreachable!(),
        },
        1,
    ));
    let err = gpu
        .dynamic_slice(&empty_operand, &wrong_starts, &[0])
        .unwrap_err();
    assert_runtime_state(&err, "dynamic_slice", &wrong_device_message);

    let empty_bool = upload(&gpu, &tensor_bool(vec![0], vec![]));
    let wrong_float_starts = crate::Tensor::F32(with_cuda_ordinal(
        match &upload(&gpu, &tensor_f32(vec![1], vec![0.0])) {
            crate::Tensor::F32(tensor) => tensor,
            _ => unreachable!(),
        },
        1,
    ));
    let err = gpu
        .dynamic_slice(&empty_bool, &wrong_float_starts, &[0])
        .unwrap_err();
    assert_runtime_state(&err, "dynamic_slice", &wrong_device_message);

    let malformed_bool = crate::Tensor::Bool(
        TypedTensor::from_buffer_col_major(
            vec![0],
            Buffer::Host(vec![]),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    );
    let err = gpu
        .dynamic_slice(
            &malformed_bool,
            &upload(&gpu, &tensor_f64(vec![1], vec![f64::NAN])),
            &[0],
        )
        .unwrap_err();
    assert_runtime_state(
        &err,
        "dynamic_slice",
        "expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops.",
    );

    let wrong_indices = crate::Tensor::I32(with_cuda_ordinal(
        match &indices {
            crate::Tensor::I32(tensor) => tensor,
            _ => unreachable!(),
        },
        1,
    ));
    let gather_operand = upload(&gpu, &tensor_f64(vec![2], vec![1.0, 2.0]));
    let err = gpu
        .gather(&gather_operand, &wrong_indices, &simple_gather_config())
        .unwrap_err();
    assert_runtime_state(&err, "gather", &wrong_device_message);

    let host_updates = tensor_f64(vec![0], vec![]);
    let host_message = "expected CubeCL GPU tensor, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops.".to_string();
    let err = gpu
        .scatter(&empty_operand, &indices, &host_updates, &config)
        .unwrap_err();
    assert_runtime_state(&err, "scatter", &host_message);

    let operand = upload(&gpu, &tensor_f64(vec![2], vec![1.0, 2.0]));
    let updates = upload(&gpu, &tensor_f64(vec![0], vec![]));
    let err = gpu
        .scatter(&operand, &wrong_indices, &updates, &config)
        .unwrap_err();
    assert_runtime_state(&err, "scatter", &wrong_device_message);

    let complex_operand = upload(&gpu, &tensor_c64(vec![0], vec![]));
    let complex_updates = upload(&gpu, &tensor_c64(vec![0], vec![]));
    let err = gpu
        .scatter(&complex_operand, &wrong_indices, &complex_updates, &config)
        .unwrap_err();
    assert_runtime_state(&err, "scatter", &wrong_device_message);
    let nonempty_complex_operand = upload(
        &gpu,
        &tensor_c64(vec![2], vec![Complex64::new(1.0, 2.0); 2]),
    );
    let err = gpu
        .scatter(
            &nonempty_complex_operand,
            &wrong_indices,
            &complex_updates,
            &config,
        )
        .unwrap_err();
    assert_runtime_state(&err, "scatter", &wrong_device_message);

    let malformed_updates = crate::Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![0],
            Buffer::Host(vec![]),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    );
    let err = gpu
        .scatter(&empty_operand, &indices, &malformed_updates, &config)
        .unwrap_err();
    assert_runtime_state(&err, "scatter", &host_message);
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
            let actual = actual.unwrap_err();
            let expected = expected.unwrap_err();
            assert_eq!(actual.kind(), expected.kind(), "{label} {op}");
            assert_error_parity(expected, actual);
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
            super::assert_error_parity($cpu.unwrap_err(), $gpu.unwrap_err());
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
    let expected_scatter_error = cpu
        .scatter(&input, &scatter_indices, &updates, &config)
        .unwrap_err();
    assert_unsupported(
        &expected_scatter_error,
        "scatter",
        "Bool data tensors are not supported by additive scatter",
    );
    let gpu_updates = upload(&gpu, &updates);
    let gpu_scatter_indices = upload(&gpu, &scatter_indices);
    let actual = gpu
        .scatter(&gi, &gpu_scatter_indices, &gpu_updates, &config)
        .unwrap_err();
    assert_unsupported(
        &actual,
        "scatter",
        "Bool data tensors are not supported by additive scatter",
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
