// Run with: cargo test --features cuda -- --ignored
use std::hint::black_box;
use std::time::Instant;

use crate::{DType, DeviceId, DeviceKind, Error, MemoryKind, Placement, Tensor, TypedTensor};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    backend::BackendSessionHost, BackendSession, GpuBackendKind, StridedSliceSpec,
    TensorElementwise, TensorIndexing, TensorRead, TensorReduction, TensorStructural, TensorView,
    TensorViewCanonicalization, TensorViewMut, TensorWrite,
};

use super::super::CudaBackend;
use super::{
    assert_cuda_unsupported_dtype, assert_error_parity, assert_runtime_state,
    assert_shape_mismatch, assert_tensor_close, cpu_backend, download, gpu_backend, tensor_bool,
    tensor_c32, tensor_c64, tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};

fn with_cuda_ordinal<T>(mut tensor: TypedTensor<T>, ordinal: usize) -> TypedTensor<T> {
    tensor.set_placement(Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal,
        }),
        cpu_affinity: None,
    });
    tensor
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_bool_structural_ops_match_cpu() {
    let matrix = tensor_bool(vec![2, 2], vec![true, false, false, true]);
    let vector = tensor_bool(vec![2], vec![true, false]);
    let scalar = tensor_bool(vec![], vec![true]);
    let empty = tensor_bool(vec![0, 2], vec![]);
    let empty_matrix = tensor_bool(vec![0, 0], vec![]);
    let empty_vector = tensor_bool(vec![0], vec![]);
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gm = upload(&gpu, &matrix);
    let gv = upload(&gpu, &vector);
    let gs = upload(&gpu, &scalar);
    let ge = upload(&gpu, &empty);
    let gem = upload(&gpu, &empty_matrix);
    let gev = upload(&gpu, &empty_vector);
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
            assert_error_parity($cpu.unwrap_err(), $gpu.unwrap_err());
        }};
    }
    parity!(cpu.transpose(&matrix, &[1, 0]), gpu.transpose(&gm, &[1, 0]));
    parity!(
        cpu.broadcast_in_dim(&scalar, &[2, 2], &[]),
        gpu.broadcast_in_dim(&gs, &[2, 2], &[])
    );
    parity!(
        cpu.extract_diagonal(&matrix, 0, 1),
        gpu.extract_diagonal(&gm, 0, 1)
    );
    parity!(
        cpu.embed_diagonal(&vector, 0, 1),
        gpu.embed_diagonal(&gv, 0, 1)
    );
    parity!(cpu.tril(&matrix, 0), gpu.tril(&gm, 0));
    parity!(cpu.triu(&matrix, 0), gpu.triu(&gm, 0));
    parity!(
        cpu.concatenate(&[&matrix, &matrix], 0),
        gpu.concatenate(&[&gm, &gm], 0)
    );
    parity!(cpu.reverse(&matrix, &[0]), gpu.reverse(&gm, &[0]));
    parity!(cpu.transpose(&empty, &[1, 0]), gpu.transpose(&ge, &[1, 0]));
    parity!(
        cpu.broadcast_in_dim(&empty_vector, &[0, 2], &[0]),
        gpu.broadcast_in_dim(&gev, &[0, 2], &[0])
    );
    parity!(
        cpu.extract_diagonal(&empty_matrix, 0, 1),
        gpu.extract_diagonal(&gem, 0, 1)
    );
    parity!(
        cpu.embed_diagonal(&empty_vector, 0, 1),
        gpu.embed_diagonal(&gev, 0, 1)
    );
    parity!(cpu.tril(&empty_matrix, 0), gpu.tril(&gem, 0));
    parity!(cpu.triu(&empty_matrix, 0), gpu.triu(&gem, 0));
    parity!(
        cpu.concatenate(&[&empty, &empty], 0),
        gpu.concatenate(&[&ge, &ge], 0)
    );
    parity!(cpu.reverse(&empty, &[1]), gpu.reverse(&ge, &[1]));

    error_parity!(cpu.transpose(&matrix, &[0, 0]), gpu.transpose(&gm, &[0, 0]));
    error_parity!(
        cpu.broadcast_in_dim(&vector, &[2, 2], &[]),
        gpu.broadcast_in_dim(&gv, &[2, 2], &[])
    );
    error_parity!(
        cpu.extract_diagonal(&matrix, 0, 0),
        gpu.extract_diagonal(&gm, 0, 0)
    );
    error_parity!(
        cpu.embed_diagonal(&vector, 0, 3),
        gpu.embed_diagonal(&gv, 0, 3)
    );
    error_parity!(cpu.tril(&vector, 0), gpu.tril(&gv, 0));
    error_parity!(cpu.triu(&vector, 0), gpu.triu(&gv, 0));
    error_parity!(
        cpu.concatenate(&[&matrix, &matrix], 2),
        gpu.concatenate(&[&gm, &gm], 2)
    );
    error_parity!(cpu.reverse(&matrix, &[2]), gpu.reverse(&gm, &[2]));
}

/// The traced runtime prepares operation operands as [`TensorRead`], which is
/// an owned tensor or a borrowed view over already-resident storage. Every
/// CUDA `_read` entry point must accept that shape; the unoverridden trait
/// defaults reject it, which broke traced linalg AD graphs with
/// "backend does not accept borrowed tensor views at this execution boundary".
#[test]
#[ignore]
fn test_cuda_read_entry_points_accept_borrowed_views() {
    let host = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scalar = tensor_f64(vec![], vec![2.0]);
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &host);
    let device_scalar = upload(&gpu, &scalar);

    let Some(device_typed) = device.as_typed::<f64>() else {
        unreachable!("f64 upload preserves the dtype")
    };
    let Some(scalar_typed) = device_scalar.as_typed::<f64>() else {
        unreachable!("f64 upload preserves the dtype")
    };
    let view = || TensorRead::from_view(TensorView::F64(device_typed.as_view()));
    let scalar_view = || TensorRead::from_view(TensorView::F64(scalar_typed.as_view()));

    let expected = cpu.transpose(&host, &[1, 0]).unwrap();
    let out = gpu.transpose_read(view(), &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &out), &expected, 1e-12);

    let expected = cpu.reshape(&host, &[3, 2]).unwrap();
    let out = gpu.reshape_read(view(), &[3, 2]).unwrap();
    assert_tensor_close(&download(&gpu, &out), &expected, 1e-12);

    let expected = cpu.broadcast_in_dim(&scalar, &[2, 3], &[]).unwrap();
    let out = gpu
        .broadcast_in_dim_read(scalar_view(), &[2, 3], &[])
        .unwrap();
    assert_tensor_close(&download(&gpu, &out), &expected, 1e-12);

    let expected = cpu.reduce_sum(&host, &[1]).unwrap();
    let out = gpu.reduce_sum_read(view(), &[1]).unwrap();
    assert_tensor_close(&download(&gpu, &out), &expected, 1e-12);

    let expected = cpu.add(&host, &host).unwrap();
    let out = gpu.add_read(view(), view()).unwrap();
    assert_tensor_close(&download(&gpu, &out), &expected, 1e-12);

    // The traced runtime reaches these entry points through the erased backend
    // session, so the session must forward the borrowed-view spellings too.
    let expected = cpu.transpose(&host, &[1, 0]).unwrap();
    let session_out = gpu.with_backend_session(|session| {
        session
            .transpose_read(
                TensorRead::from_view(TensorView::F64(device_typed.as_view())),
                &[1, 0],
            )
            .unwrap()
    });
    assert_tensor_close(&download(&gpu, &session_out), &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_structural_ops_match_cpu() {
    let input = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scalar = tensor_f64(vec![], vec![7.5]);
    let vector = tensor_f64(vec![3], vec![10.0, 20.0, 30.0]);
    let matrix = tensor_f64(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_scalar = upload(&gpu, &scalar);
    let gpu_vector = upload(&gpu, &vector);

    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let gpu_out = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reshape(&input, &[3, 2]).unwrap();
    let gpu_out = gpu.reshape(&gpu_input, &[3, 2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.broadcast_in_dim(&scalar, &[2, 3], &[]).unwrap();
    let gpu_out = gpu.broadcast_in_dim(&gpu_scalar, &[2, 3], &[]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reverse(&input, &[1]).unwrap();
    let gpu_out = gpu.reverse(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.concatenate(&[&input, &input], 1).unwrap();
    let gpu_concat = gpu.concatenate(&[&gpu_input, &gpu_input], 1).unwrap();
    let actual = download(&gpu, &gpu_concat);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.extract_diagonal(&matrix, 0, 1).unwrap();
    let gpu_matrix = upload(&gpu, &matrix);
    let gpu_out = gpu.extract_diagonal(&gpu_matrix, 0, 1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.embed_diagonal(&vector, 0, 1).unwrap();
    let gpu_out = gpu.embed_diagonal(&gpu_vector, 0, 1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.tril(&matrix, 0).unwrap();
    let gpu_out = gpu.tril(&gpu_matrix, 0).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.triu(&matrix, -1).unwrap();
    let gpu_out = gpu.triu(&gpu_matrix, -1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_i64_structural_ops_match_cpu() {
    let input = tensor_i64(vec![2, 3], vec![1, -2, 3, -4, 5, -6]);
    let scalar = tensor_i64(vec![], vec![7]);
    let vector = tensor_i64(vec![3], vec![10, -20, 30]);
    let matrix = tensor_i64(vec![3, 3], vec![1, -2, 3, -4, 5, -6, 7, -8, 9]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_scalar = upload(&gpu, &scalar);
    let gpu_vector = upload(&gpu, &vector);

    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let gpu_out = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.reshape(&input, &[3, 2]).unwrap();
    let gpu_out = gpu.reshape(&gpu_input, &[3, 2]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.broadcast_in_dim(&scalar, &[2, 3], &[]).unwrap();
    let gpu_out = gpu.broadcast_in_dim(&gpu_scalar, &[2, 3], &[]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.reverse(&input, &[1]).unwrap();
    let gpu_out = gpu.reverse(&gpu_input, &[1]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.concatenate(&[&input, &input], 1).unwrap();
    let gpu_out = gpu.concatenate(&[&gpu_input, &gpu_input], 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let gpu_matrix = upload(&gpu, &matrix);
    let expected = cpu.extract_diagonal(&matrix, 0, 1).unwrap();
    let gpu_out = gpu.extract_diagonal(&gpu_matrix, 0, 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.embed_diagonal(&vector, 0, 1).unwrap();
    let gpu_out = gpu.embed_diagonal(&gpu_vector, 0, 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.tril(&matrix, 0).unwrap();
    let gpu_out = gpu.tril(&gpu_matrix, 0).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.triu(&matrix, -1).unwrap();
    let gpu_out = gpu.triu(&gpu_matrix, -1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_i32_structural_ops_match_cpu() {
    let input = tensor_i32(vec![2, 3], vec![1, -2, 3, -4, 5, -6]);
    let scalar = tensor_i32(vec![], vec![7]);
    let vector = tensor_i32(vec![3], vec![10, -20, 30]);
    let matrix = tensor_i32(vec![3, 3], vec![1, -2, 3, -4, 5, -6, 7, -8, 9]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_scalar = upload(&gpu, &scalar);
    let gpu_vector = upload(&gpu, &vector);

    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let gpu_out = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.reshape(&input, &[3, 2]).unwrap();
    let gpu_out = gpu.reshape(&gpu_input, &[3, 2]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.broadcast_in_dim(&scalar, &[2, 3], &[]).unwrap();
    let gpu_out = gpu.broadcast_in_dim(&gpu_scalar, &[2, 3], &[]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.reverse(&input, &[1]).unwrap();
    let gpu_out = gpu.reverse(&gpu_input, &[1]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.concatenate(&[&input, &input], 1).unwrap();
    let gpu_out = gpu.concatenate(&[&gpu_input, &gpu_input], 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let gpu_matrix = upload(&gpu, &matrix);
    let expected = cpu.extract_diagonal(&matrix, 0, 1).unwrap();
    let gpu_out = gpu.extract_diagonal(&gpu_matrix, 0, 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.embed_diagonal(&vector, 0, 1).unwrap();
    let gpu_out = gpu.embed_diagonal(&gpu_vector, 0, 1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.tril(&matrix, 0).unwrap();
    let gpu_out = gpu.tril(&gpu_matrix, 0).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);

    let expected = cpu.triu(&matrix, -1).unwrap();
    let gpu_out = gpu.triu(&gpu_matrix, -1).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_bool_reshape_round_trips() {
    let input = tensor_bool(vec![2, 3], vec![true, false, true, true, false, false]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reshape(&input, &[3, 2]).unwrap();
    let gpu_out = gpu.reshape(&gpu_input, &[3, 2]).unwrap();
    assert_tensor_close(&download(&gpu, &gpu_out), &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_convert_matches_cpu() {
    let real = tensor_f64(vec![3], vec![1.5, -2.25, 3.75]);
    let complex = tensor_c64(
        vec![2],
        vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(-3.5, 0.5),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_real = upload(&gpu, &real);
    let gpu_complex = upload(&gpu, &complex);

    let expected = cpu.cast(&real, DType::F32).unwrap();
    let gpu_out = gpu.cast(&gpu_real, DType::F32).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-6);

    let expected = cpu.convert(&real, DType::C64).unwrap();
    let gpu_out = gpu.convert(&gpu_real, DType::C64).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.cast(&complex, DType::F64).unwrap();
    let gpu_out = gpu.cast(&gpu_complex, DType::F64).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cuda_explicit_cast_matrix_matches_cpu() {
    let sources = [
        tensor_f32(vec![4], vec![0.0, -2.75, 3.5, f32::NAN]),
        tensor_f64(vec![4], vec![0.0, -2.75, 3.5, f64::NAN]),
        tensor_i32(vec![4], vec![0, -2, 3, i32::MAX]),
        tensor_i64(vec![4], vec![0, -2, 3, i64::MAX]),
        tensor_bool(vec![4], vec![false, true, true, false]),
        tensor_c32(
            vec![4],
            vec![
                Complex32::new(0.0, 0.0),
                Complex32::new(-2.75, 4.0),
                Complex32::new(3.5, -1.0),
                Complex32::new(f32::NAN, 0.0),
            ],
        ),
        tensor_c64(
            vec![4],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(-2.75, 4.0),
                Complex64::new(3.5, -1.0),
                Complex64::new(f64::NAN, 0.0),
            ],
        ),
    ];
    let targets = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    for source in &sources {
        let gpu_source = upload(&gpu, source);
        for &target in &targets {
            let expected = cpu.cast(source, target);
            let actual = gpu.cast(&gpu_source, target);
            match (expected, actual) {
                (Err(expected), Err(actual)) => assert_error_parity(expected, actual),
                (Ok(expected), Ok(actual)) => {
                    let actual = download(&gpu, &actual);
                    assert_cast_tensor_equal(&actual, &expected);
                }
                (expected, actual) => panic!(
                    "cast {:?} -> {target:?} differs: CPU={expected:?}, CUDA={actual:?}",
                    source.dtype()
                ),
            }
        }
    }

    let empty_sources = [
        tensor_f32(vec![0], vec![]),
        tensor_f64(vec![0], vec![]),
        tensor_i32(vec![0], vec![]),
        tensor_i64(vec![0], vec![]),
        tensor_bool(vec![0], vec![]),
        tensor_c32(vec![0], vec![]),
        tensor_c64(vec![0], vec![]),
    ];
    for source in &empty_sources {
        let gpu_source = upload(&gpu, source);
        for &target in &targets {
            let expected = cpu.cast(source, target).unwrap();
            let gpu_actual = gpu.cast(&gpu_source, target).unwrap();
            let actual = download(&gpu, &gpu_actual);
            assert_cast_tensor_equal(&actual, &expected);
        }
    }

    for source in [
        tensor_f32(vec![3], vec![0.0, -2.75, 3.5]),
        tensor_f64(vec![3], vec![0.0, -2.75, 3.5]),
        tensor_c32(
            vec![3],
            vec![
                Complex32::new(0.0, 9.0),
                Complex32::new(-2.75, 4.0),
                Complex32::new(3.5, -1.0),
            ],
        ),
        tensor_c64(
            vec![3],
            vec![
                Complex64::new(0.0, 9.0),
                Complex64::new(-2.75, 4.0),
                Complex64::new(3.5, -1.0),
            ],
        ),
    ] {
        let gpu_source = upload(&gpu, &source);
        for target in [DType::I32, DType::I64] {
            let expected = cpu.cast(&source, target).unwrap();
            let gpu_actual = gpu.cast(&gpu_source, target).unwrap();
            assert_cast_tensor_equal(&download(&gpu, &gpu_actual), &expected);
        }
    }

    for (source, target) in [
        (tensor_f64(vec![1], vec![f64::INFINITY]), DType::I32),
        (tensor_f64(vec![1], vec![f64::INFINITY]), DType::I64),
        (tensor_f64(vec![1], vec![i32::MAX as f64 + 1.0]), DType::I32),
        (
            tensor_f64(vec![1], vec![9_223_372_036_854_775_808.0]),
            DType::I64,
        ),
        (
            tensor_c64(vec![1], vec![Complex64::new(f64::NEG_INFINITY, 7.0)]),
            DType::I64,
        ),
    ] {
        let gpu_source = upload(&gpu, &source);
        assert_error_parity(
            cpu.cast(&source, target).unwrap_err(),
            gpu.cast(&gpu_source, target).unwrap_err(),
        );
    }

    for source in [
        tensor_f32(vec![2], vec![-2_147_483_648.0, 2_147_483_520.0]),
        tensor_c32(
            vec![2],
            vec![
                Complex32::new(-2_147_483_648.0, 11.0),
                Complex32::new(2_147_483_520.0, -9.0),
            ],
        ),
    ] {
        let gpu_source = upload(&gpu, &source);
        let expected = cpu.cast(&source, DType::I32).unwrap();
        let actual = gpu.cast(&gpu_source, DType::I32).unwrap();
        assert_cast_tensor_equal(&download(&gpu, &actual), &expected);
    }

    for source in [
        tensor_f32(vec![2], vec![2_147_483_648.0, f32::NAN]),
        tensor_f32(vec![1], vec![-2_147_483_904.0]),
        tensor_c32(
            vec![2],
            vec![
                Complex32::new(2_147_483_648.0, 3.0),
                Complex32::new(f32::NAN, 0.0),
            ],
        ),
        tensor_c32(vec![1], vec![Complex32::new(-2_147_483_904.0, 3.0)]),
    ] {
        let gpu_source = upload(&gpu, &source);
        assert_error_parity(
            cpu.cast(&source, DType::I32).unwrap_err(),
            gpu.cast(&gpu_source, DType::I32).unwrap_err(),
        );
    }

    let i64_upper_exclusive = 9_223_372_036_854_775_808.0_f32;
    let i64_upper_valid = f32::from_bits(i64_upper_exclusive.to_bits() - 1);
    let i64_lower_valid = -i64_upper_exclusive;
    let i64_lower_invalid = f32::from_bits(i64_lower_valid.to_bits() + 1);
    for source in [
        tensor_f32(vec![2], vec![i64_lower_valid, i64_upper_valid]),
        tensor_c32(
            vec![2],
            vec![
                Complex32::new(i64_lower_valid, 2.0),
                Complex32::new(i64_upper_valid, -2.0),
            ],
        ),
    ] {
        let gpu_source = upload(&gpu, &source);
        let expected = cpu.cast(&source, DType::I64).unwrap();
        let actual = gpu.cast(&gpu_source, DType::I64).unwrap();
        assert_cast_tensor_equal(&download(&gpu, &actual), &expected);
    }
    for source in [
        tensor_f32(vec![1], vec![i64_upper_exclusive]),
        tensor_f32(vec![1], vec![i64_lower_invalid]),
        tensor_c32(vec![1], vec![Complex32::new(i64_upper_exclusive, 2.0)]),
        tensor_c32(vec![1], vec![Complex32::new(i64_lower_invalid, 2.0)]),
    ] {
        let gpu_source = upload(&gpu, &source);
        assert_error_parity(
            cpu.cast(&source, DType::I64).unwrap_err(),
            gpu.cast(&gpu_source, DType::I64).unwrap_err(),
        );
    }
}

fn assert_cast_tensor_equal(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.shape(), expected.shape());
    macro_rules! scalar_equal {
        ($ty:ty, $eq:expr) => {{
            let actual = actual.as_slice::<$ty>().unwrap();
            let expected = expected.as_slice::<$ty>().unwrap();
            assert!(
                actual.iter().zip(expected).all($eq),
                "actual={actual:?} expected={expected:?}"
            );
        }};
    }
    match actual.dtype() {
        DType::F32 => scalar_equal!(f32, |(a, e): (&f32, &f32)| a == e
            || (a.is_nan() && e.is_nan())),
        DType::F64 => scalar_equal!(f64, |(a, e): (&f64, &f64)| a == e
            || (a.is_nan() && e.is_nan())),
        DType::I32 => scalar_equal!(i32, |(a, e): (&i32, &i32)| a == e),
        DType::I64 => scalar_equal!(i64, |(a, e): (&i64, &i64)| a == e),
        DType::Bool => scalar_equal!(bool, |(a, e): (&bool, &bool)| a == e),
        DType::C32 => scalar_equal!(Complex32, |(a, e): (&Complex32, &Complex32)| {
            (a.re == e.re || (a.re.is_nan() && e.re.is_nan()))
                && (a.im == e.im || (a.im.is_nan() && e.im.is_nan()))
        }),
        DType::C64 => scalar_equal!(Complex64, |(a, e): (&Complex64, &Complex64)| {
            (a.re == e.re || (a.re.is_nan() && e.re.is_nan()))
                && (a.im == e.im || (a.im.is_nan() && e.im.is_nan()))
        }),
        // Test fixtures cover the preset dtypes; an externally defined scalar has no
        // fixture and would change what this test asserts.
        DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
    }
}

#[test]
#[ignore]
fn cuda_to_contiguous_keeps_tensor_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
    let gpu_input = upload(&gpu, &input);
    let Some(gpu_tensor) = gpu_input.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };
    let view = gpu_tensor.as_view().transpose_view([1, 0]).unwrap();

    let compact = gpu.to_contiguous(&view).unwrap();

    assert_eq!(compact.shape(), &[3, 2]);
    assert_eq!(compact.placement().memory_kind, MemoryKind::Device);
    assert!(matches!(
        compact
            .placement()
            .device
            .as_ref()
            .map(|device| &device.kind),
        Some(tenferro_tensor::DeviceKind::Gpu(GpuBackendKind::Cuda))
    ));
    let actual = download(&gpu, &Tensor::from_typed::<i32>(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 5, 2, 4, 6]);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and cuTENSOR"]
fn cuda_cutensor_permutation_transpose_and_to_contiguous_match_cpu() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    assert_eq!(
        gpu.cutensor_permutation_plan_cache_stats().unwrap().entries,
        0
    );

    let input = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let actual = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-12);

    let cache_after_first = gpu.cutensor_permutation_plan_cache_stats().unwrap();
    assert_eq!(cache_after_first.entries, 1);
    assert_eq!(cache_after_first.misses, 1);

    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let actual = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-12);
    let cache_after_second = gpu.cutensor_permutation_plan_cache_stats().unwrap();
    assert_eq!(cache_after_second.entries, 1);
    assert_eq!(cache_after_second.hits, 1);

    let Some(gpu_tensor) = gpu_input.as_typed::<f64>() else {
        panic!("expected f64 tensor");
    };
    let materialize_view = gpu_tensor.as_view().transpose_view([1, 0]).unwrap();
    let materialized = gpu
        .to_contiguous_read(TensorRead::from_view(TensorView::F64(materialize_view)))
        .unwrap();
    let actual = download(&gpu, &materialized);
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );

    let complex = tensor_c32(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, 6.0),
            Complex32::new(7.0, 8.0),
        ],
    );
    let gpu_complex = upload(&gpu, &complex);
    let expected = cpu.transpose(&complex, &[1, 0]).unwrap();
    let actual = gpu.transpose(&gpu_complex, &[1, 0]).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);

    let view = gpu_tensor
        .as_view()
        .try_slice_axis(0, StridedSliceSpec::reverse())
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    let compact = gpu.to_contiguous(&view).unwrap();
    let actual = download(&gpu, &Tensor::from_typed::<f64>(compact));
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        &[2.0, 4.0, 6.0, 1.0, 3.0, 5.0]
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_materialization_is_object_safe_and_stays_on_device() {
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &tensor_i32(vec![2, 3], vec![1, 2, 3, 4, 5, 6]));
    let Some(input) = gpu_input.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };
    let view = input.as_view().transpose_view([1, 0]).unwrap();
    let exec: &mut dyn BackendSession = &mut gpu;

    let output = exec
        .to_contiguous_read(TensorRead::from_view(TensorView::I32(view)))
        .unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.placement().memory_kind, MemoryKind::Device);
    let actual = download(&gpu, &output);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 5, 2, 4, 6]);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_copy_is_object_safe_and_updates_strided_destination() {
    let mut gpu = gpu_backend();
    let gpu_src = upload(&gpu, &tensor_i32(vec![2, 2], vec![1, 2, 3, 4]));
    let mut gpu_dst = upload(&gpu, &tensor_i32(vec![2, 2], vec![0, 0, 0, 0]));
    let Some(dst) = gpu_dst.as_typed_mut::<i32>() else {
        panic!("expected i32 destination");
    };
    let dst_view = dst.as_view_mut().transpose_view([1, 0]).unwrap();
    let exec: &mut dyn BackendSession = &mut gpu;

    exec.copy_read_into(
        TensorRead::from_tensor(&gpu_src),
        TensorWrite::from_view(TensorViewMut::I32(dst_view)),
    )
    .unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_copy_into_cutensor_matches_destination_reuse_and_survives_source_mutation() {
    let mut gpu = gpu_backend();
    let mut gpu_src = upload(
        &gpu,
        &tensor_f64(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let mut gpu_dst = upload(&gpu, &tensor_f64(vec![2, 3], vec![0.0; 6]));
    let replacement = upload(&gpu, &tensor_f64(vec![3, 2], vec![9.0; 6]));

    let Some(dst) = gpu_dst.as_typed_mut::<f64>() else {
        panic!("expected f64 destination");
    };
    let dst_view = dst.as_view_mut().transpose_view([1, 0]).unwrap();
    gpu.copy_read_into(
        TensorRead::from_tensor(&gpu_src),
        TensorWrite::from_view(TensorViewMut::F64(dst_view)),
    )
    .unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let Some(replacement) = replacement.as_typed::<f64>() else {
        panic!("expected f64 replacement");
    };
    let Some(src_mut) = gpu_src.as_typed_mut::<f64>() else {
        panic!("expected mutable f64 source");
    };
    gpu.copy_into(&replacement.as_view(), &mut src_mut.as_view_mut())
        .unwrap();

    let after_source_mutation = download(&gpu, &gpu_dst);
    assert_eq!(
        after_source_mutation.as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_copy_into_cutensor_matches_complex_destination_reuse() {
    let mut gpu = gpu_backend();
    let gpu_src = upload(
        &gpu,
        &tensor_c32(
            vec![2, 2, 2],
            vec![
                Complex32::new(1.0, 2.0),
                Complex32::new(3.0, 4.0),
                Complex32::new(5.0, 6.0),
                Complex32::new(7.0, 8.0),
                Complex32::new(9.0, 10.0),
                Complex32::new(11.0, 12.0),
                Complex32::new(13.0, 14.0),
                Complex32::new(15.0, 16.0),
            ],
        ),
    );
    let mut gpu_dst = upload(
        &gpu,
        &tensor_c32(vec![2, 2, 2], vec![Complex32::new(0.0, 0.0); 8]),
    );
    let Some(dst) = gpu_dst.as_typed_mut::<Complex32>() else {
        panic!("expected complex destination");
    };
    let dst_view = dst.as_view_mut().transpose_view([1, 0, 2]).unwrap();
    gpu.copy_read_into(
        TensorRead::from_tensor(&gpu_src),
        TensorWrite::from_view(TensorViewMut::C32(dst_view)),
    )
    .unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(
        actual.as_slice::<Complex32>().unwrap(),
        &[
            Complex32::new(1.0, 2.0),
            Complex32::new(5.0, 6.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(7.0, 8.0),
            Complex32::new(9.0, 10.0),
            Complex32::new(13.0, 14.0),
            Complex32::new(11.0, 12.0),
            Complex32::new(15.0, 16.0),
        ]
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU with a max single allocation above 4 GiB"]
fn cuda_runtime_copy_into_1522_a100_destination_reuse_benchmark() {
    const MIN_MAX_PAGE_SIZE: u64 = 4 * 1024 * 1024 * 1024;

    let mut gpu = gpu_backend();
    let max_page_size = gpu.runtime().client().properties().memory.max_page_size;
    if max_page_size <= MIN_MAX_PAGE_SIZE {
        eprintln!("skipping #1522 A100 benchmark: max single allocation is {max_page_size} bytes");
        return;
    }

    fn measure(
        gpu: &mut CudaBackend,
        source: &Tensor,
        destination: &mut Tensor,
        permutation: &[usize],
    ) -> Vec<f64> {
        let mut run = || {
            let Some(dst) = destination.as_typed_mut::<f64>() else {
                panic!("expected f64 destination");
            };
            let view = dst.as_view_mut().transpose_view(permutation).unwrap();
            let start = Instant::now();
            let result = gpu.copy_read_into(
                TensorRead::from_tensor(black_box(source)),
                TensorWrite::from_view(TensorViewMut::F64(view)),
            );
            black_box(result).unwrap();
            gpu.runtime().synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };

        for _ in 0..3 {
            black_box(run());
        }
        let mut samples: Vec<f64> = (0..7).map(|_| run()).collect();
        samples.sort_by(f64::total_cmp);
        samples
    }

    let source_2d = upload(
        &gpu,
        &tensor_f64(vec![32_768, 16_384], vec![0.0; 32_768 * 16_384]),
    );
    let mut destination_2d = upload(
        &gpu,
        &tensor_f64(vec![16_384, 32_768], vec![0.0; 32_768 * 16_384]),
    );
    let samples_2d = measure(&mut gpu, &source_2d, &mut destination_2d, &[1, 0]);

    let source_3d = upload(
        &gpu,
        &tensor_f64(vec![1024, 1024, 512], vec![0.0; 1024 * 1024 * 512]),
    );
    let mut destination_3d = upload(
        &gpu,
        &tensor_f64(vec![512, 1024, 1024], vec![0.0; 1024 * 1024 * 512]),
    );
    let samples_3d = measure(&mut gpu, &source_3d, &mut destination_3d, &[1, 2, 0]);

    println!("#1522 A100 2D sorted samples (ms): {samples_2d:?}");
    println!("#1522 A100 3D sorted samples (ms): {samples_3d:?}");
}

/// Issue #1832: the erased read-into entry must consume an arbitrary-stride
/// source view directly instead of rejecting it.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_copy_read_into_consumes_transposed_source() {
    let mut gpu = gpu_backend();
    let gpu_src = upload(&gpu, &tensor_i32(vec![2, 2], vec![1, 2, 3, 4]));
    let mut gpu_dst = upload(&gpu, &tensor_i32(vec![2, 2], vec![0, 0, 0, 0]));
    let Some(src) = gpu_src.as_typed::<i32>() else {
        panic!("expected i32 source");
    };
    let src_view = src.as_view().transpose_view([1, 0]).unwrap();

    gpu.copy_read_into(
        TensorRead::from_view(TensorView::I32(src_view)),
        TensorWrite::from_tensor(&mut gpu_dst),
    )
    .unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_bool_materialization_reports_intentional_erased_limitation() {
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &tensor_bool(vec![2], vec![true, false]));

    let err = gpu
        .to_contiguous_read(TensorRead::from_tensor(&gpu_input))
        .unwrap_err();

    assert_cuda_unsupported_dtype(&err, "CudaBackend::to_contiguous_read", DType::Bool);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_runtime_bool_copy_reports_intentional_erased_limitation() {
    let mut gpu = gpu_backend();
    let gpu_src = upload(&gpu, &tensor_bool(vec![2], vec![true, false]));
    let mut gpu_dst = upload(&gpu, &tensor_bool(vec![2], vec![false, false]));

    let err = gpu
        .copy_read_into(
            TensorRead::from_tensor(&gpu_src),
            TensorWrite::from_tensor(&mut gpu_dst),
        )
        .unwrap_err();

    assert_cuda_unsupported_dtype(&err, "CudaBackend::copy_read_into", DType::Bool);
}

#[test]
#[ignore]
fn cuda_to_contiguous_preserves_negative_stride_view() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![4], vec![1, 2, 3, 4]);
    let gpu_input = upload(&gpu, &input);
    let Some(gpu_tensor) = gpu_input.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };
    let view = gpu_tensor
        .as_view()
        .try_slice_axis(0, StridedSliceSpec::reverse())
        .unwrap();

    let compact = gpu.to_contiguous(&view).unwrap();

    let actual = download(&gpu, &Tensor::from_typed::<i32>(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[4, 3, 2, 1]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_rank_zero_scalar_stays_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![], vec![7]);
    let gpu_input = upload(&gpu, &input);
    let Some(gpu_tensor) = gpu_input.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };

    let compact = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap();

    assert_eq!(compact.shape(), &[] as &[usize]);
    assert_eq!(compact.placement().memory_kind, MemoryKind::Device);
    let actual = download(&gpu, &Tensor::from_typed::<i32>(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[7]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_empty_view_stays_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![0, 3], vec![]);
    let gpu_input = upload(&gpu, &input);
    let Some(gpu_tensor) = gpu_input.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };

    let compact = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap();

    assert_eq!(compact.shape(), &[0, 3]);
    assert_eq!(compact.placement().memory_kind, MemoryKind::Device);
    let actual = download(&gpu, &Tensor::from_typed::<i32>(compact));
    assert_eq!(actual.shape(), &[0, 3]);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[] as &[i32]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_bool_view_returns_unsupported_dtype() {
    let mut gpu = gpu_backend();
    let input = tensor_bool(vec![2], vec![true, false]);
    let gpu_input = upload(&gpu, &input);
    let Some(gpu_tensor) = gpu_input.as_typed::<bool>() else {
        panic!("expected bool tensor");
    };

    let err = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap_err();

    assert_cuda_unsupported_dtype(&err, "CudaBackend::to_contiguous", DType::Bool);
}

#[test]
#[ignore]
fn cuda_to_contiguous_host_view_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let host = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();

    let err = gpu.to_contiguous(&host.as_view()).unwrap_err();

    assert_runtime_state(
        &err,
        "CudaBackend::to_contiguous",
        "expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops.",
    );
}

#[test]
#[ignore]
fn cuda_copy_into_host_source_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let src = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let dst_host = tensor_i32(vec![2], vec![0, 0]);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let Some(dst) = gpu_dst.as_typed_mut::<i32>() else {
        panic!("expected i32 tensor");
    };

    let err = gpu
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert_runtime_state(
        &err,
        "CudaBackend::copy_into",
        "expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops.",
    );
}

#[test]
#[ignore]
fn cuda_copy_into_host_destination_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let src_host = tensor_i32(vec![2], vec![1, 2]);
    let gpu_src = upload(&gpu, &src_host);
    let Some(src) = gpu_src.as_typed::<i32>() else {
        panic!("expected i32 tensor");
    };
    let mut dst = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![0, 0]).unwrap();

    let err = gpu
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert_runtime_state(
        &err,
        "CudaBackend::copy_into",
        "expected CubeCL GPU tensor view, got host tensor. Use upload_tensor() to transfer to GPU before calling GPU ops.",
    );
}

#[test]
#[ignore]
fn cuda_copy_into_updates_strided_view_on_cuda() {
    let mut gpu = gpu_backend();
    let dst_host = tensor_i32(vec![2, 2], vec![0, 0, 0, 0]);
    let src_host = tensor_i32(vec![2, 2], vec![1, 2, 3, 4]);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let gpu_src = upload(&gpu, &src_host);

    let (Some(dst), Some(src)) = (gpu_dst.as_typed_mut::<i32>(), gpu_src.as_typed::<i32>()) else {
        panic!("expected i32 tensors");
    };
    let mut dst_view = dst.as_view_mut().transpose_view([1, 0]).unwrap();

    gpu.copy_into(&src.as_view(), &mut dst_view).unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);
}

/// Issue #1832: an arbitrary-stride source view copies in place, without the
/// caller canonicalizing it first.
#[test]
#[ignore]
fn cuda_copy_into_consumes_arbitrary_stride_source() {
    let mut gpu = gpu_backend();
    let src_host = tensor_i32(vec![2, 2], vec![1, 2, 3, 4]);
    let dst_host = tensor_i32(vec![2, 2], vec![0, 0, 0, 0]);
    let gpu_src = upload(&gpu, &src_host);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let (Some(src), Some(dst)) = (gpu_src.as_typed::<i32>(), gpu_dst.as_typed_mut::<i32>()) else {
        panic!("expected i32 tensors");
    };
    let src_view = src.as_view().transpose_view([1, 0]).unwrap();

    gpu.copy_into(&src_view, &mut dst.as_view_mut()).unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);
}

/// Issue #1832: a strided region at a nonzero offset inside a larger device
/// buffer copies into a strided region of another buffer in one pass, and the
/// destination elements outside the region stay untouched.
///
/// The source region is `dst[1 + 3i + j] = src[2 + i + 4j]` over a 2x3 block,
/// which exercises an offset strided read and an offset non-compact write on
/// the same launch.
macro_rules! region_copy_case {
    ($name:ident, $ty:ty, $tensor:ident, $value:expr) => {
        #[test]
        #[ignore = "requires CUDA 12.8+ GPU"]
        fn $name() {
            let mut gpu = gpu_backend();
            let source_values: Vec<$ty> = (0..16).map($value).collect();
            let gpu_src = upload(&gpu, &$tensor(vec![16], source_values.clone()));
            let mut gpu_dst = upload(&gpu, &$tensor(vec![20], vec![$value(99); 20]));
            let (Some(src), Some(dst)) = (gpu_src.as_typed::<$ty>(), gpu_dst.as_typed_mut::<$ty>())
            else {
                panic!("expected typed tensors");
            };

            let src_view = src.backend_region_view(vec![2, 3], vec![1, 4], 2).unwrap();
            let mut dst_view = dst
                .backend_region_view_mut(vec![2, 3], vec![3, 1], 1)
                .unwrap();
            gpu.copy_into(&src_view, &mut dst_view).unwrap();

            let mut expected = vec![$value(99); 20];
            for row in 0..2usize {
                for column in 0..3usize {
                    expected[1 + 3 * row + column] = source_values[2 + row + 4 * column];
                }
            }
            let actual = download(&gpu, &gpu_dst);
            assert_eq!(actual.as_slice::<$ty>().unwrap(), expected.as_slice());
        }
    };
}

region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_f32,
    f32,
    tensor_f32,
    |value| value as f32
);
region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_f64,
    f64,
    tensor_f64,
    f64::from
);
region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_i32,
    i32,
    tensor_i32,
    |value| value
);
region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_i64,
    i64,
    tensor_i64,
    i64::from
);
region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_c32,
    Complex32,
    tensor_c32,
    |value| Complex32::new(value as f32, -(value as f32))
);
region_copy_case!(
    cuda_copy_into_moves_offset_strided_region_c64,
    Complex64,
    tensor_c64,
    |value| Complex64::new(f64::from(value), -f64::from(value))
);

/// Issue #1832: the erased read-into entry routes F32/F64/C32/C64 through
/// cuTENSOR, so cover an offset strided source there as well, with a rank-3
/// permuted destination.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_copy_read_into_moves_offset_strided_region_through_cutensor() {
    let mut gpu = gpu_backend();
    let source_values: Vec<f64> = (0..48).map(f64::from).collect();
    let gpu_src = upload(&gpu, &tensor_f64(vec![48], source_values.clone()));
    let mut gpu_dst = upload(&gpu, &tensor_f64(vec![64], vec![-1.0; 64]));
    let (Some(src), Some(dst)) = (gpu_src.as_typed::<f64>(), gpu_dst.as_typed_mut::<f64>()) else {
        panic!("expected f64 tensors");
    };

    // 2x3x2 source block with axis strides (1, 4, 16) starting at offset 3.
    let src_view = src
        .backend_region_view(vec![2, 3, 2], vec![1, 4, 16], 3)
        .unwrap();
    // Destination with the two leading axes swapped in memory, at offset 5.
    let dst_view = dst
        .backend_region_view_mut(vec![2, 3, 2], vec![3, 1, 6], 5)
        .unwrap();

    gpu.copy_read_into(
        TensorRead::from_view(TensorView::F64(src_view)),
        TensorWrite::from_view(TensorViewMut::F64(dst_view)),
    )
    .unwrap();

    let mut expected = vec![-1.0f64; 64];
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..2usize {
                expected[5 + 3 * i + j + 6 * k] = source_values[3 + i + 4 * j + 16 * k];
            }
        }
    }
    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<f64>().unwrap(), expected.as_slice());
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_copy_into_rejects_source_on_wrong_device() {
    let mut gpu = gpu_backend();
    let src_host = tensor_i32(vec![2], vec![1, 2]);
    let dst_host = tensor_i32(vec![2], vec![0, 0]);
    let gpu_src = upload(&gpu, &src_host);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let (Ok(src), Some(dst)) = (gpu_src.into_typed::<i32>(), gpu_dst.as_typed_mut::<i32>()) else {
        panic!("expected i32 tensors");
    };
    let wrong_src = with_cuda_ordinal(src, 1);

    let err = gpu
        .copy_into(&wrong_src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CudaBackend::copy_into",
            ref message,
        } if message.contains("cuda:0") && message.contains("Cuda):1")
    ));
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_copy_into_rejects_destination_on_wrong_device() {
    let mut gpu = gpu_backend();
    let src_host = tensor_i32(vec![2], vec![1, 2]);
    let dst_host = tensor_i32(vec![2], vec![0, 0]);
    let gpu_src = upload(&gpu, &src_host);
    let gpu_dst = upload(&gpu, &dst_host);
    let (Some(src), Ok(dst)) = (gpu_src.as_typed::<i32>(), gpu_dst.into_typed::<i32>()) else {
        panic!("expected i32 tensors");
    };
    let mut wrong_dst = with_cuda_ordinal(dst, 1);

    let err = gpu
        .copy_into(&src.as_view(), &mut wrong_dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "CudaBackend::copy_into",
            ref message,
        } if message.contains("cuda:0") && message.contains("Cuda):1")
    ));
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_copy_into_reports_typed_shape_mismatch() {
    let mut gpu = gpu_backend();
    let gpu_src = upload(&gpu, &tensor_i32(vec![2], vec![1, 2]));
    let mut gpu_dst = upload(&gpu, &tensor_i32(vec![3], vec![0, 0, 0]));
    let (Some(src), Some(dst)) = (gpu_src.as_typed::<i32>(), gpu_dst.as_typed_mut::<i32>()) else {
        panic!("expected i32 tensors");
    };

    let err = gpu
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert_shape_mismatch(&err, "CudaBackend::copy_into", &[2], &[3]);
}

/// Issue #1832: materializing an offset strided device region reads the region
/// in place. The vendor path declares the element-size alignment here, so this
/// covers the offset descriptor rather than the alignment regression that
/// `cuda_copy_read_into_moves_offset_strided_region_through_cutensor` pins.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_to_contiguous_read_materializes_offset_strided_region() {
    let mut gpu = gpu_backend();
    let source_values: Vec<f64> = (0..48).map(f64::from).collect();
    let gpu_src = upload(&gpu, &tensor_f64(vec![48], source_values.clone()));
    let Some(src) = gpu_src.as_typed::<f64>() else {
        panic!("expected f64 source");
    };
    let src_view = src.backend_region_view(vec![2, 3], vec![1, 4], 3).unwrap();

    let materialized = gpu
        .to_contiguous_read(TensorRead::from_view(TensorView::F64(src_view)))
        .unwrap();

    let mut expected = Vec::with_capacity(6);
    for column in 0..3usize {
        for row in 0..2usize {
            expected.push(source_values[3 + row + 4 * column]);
        }
    }
    let actual = download(&gpu, &materialized);
    assert_eq!(actual.shape(), &[2, 3]);
    assert_eq!(actual.as_slice::<f64>().unwrap(), expected.as_slice());
}

/// Regression test for issue #1833: the triangular kernels' zero element must
/// be expressible for complex dtypes, which a `u32` cast cannot lower to on the
/// CUDA dialect.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_triangular_ops_match_cpu_for_complex_dtypes() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let matrix_c64 = tensor_c64(
        vec![3, 3],
        (1..=9)
            .map(|value| Complex64::new(f64::from(value), -f64::from(value)))
            .collect(),
    );
    let gpu_c64 = upload(&gpu, &matrix_c64);
    let expected = cpu.tril(&matrix_c64, 0).unwrap();
    let actual = gpu.tril(&gpu_c64, 0).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);
    let expected = cpu.triu(&matrix_c64, -1).unwrap();
    let actual = gpu.triu(&gpu_c64, -1).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);

    let matrix_c32 = tensor_c32(
        vec![2, 2],
        (1..=4)
            .map(|value| Complex32::new(value as f32, 0.5 * value as f32))
            .collect(),
    );
    let gpu_c32 = upload(&gpu, &matrix_c32);
    let expected = cpu.tril(&matrix_c32, 0).unwrap();
    let actual = gpu.tril(&gpu_c32, 0).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);
    let expected = cpu.triu(&matrix_c32, 1).unwrap();
    let actual = gpu.triu(&gpu_c32, 1).unwrap();
    assert_tensor_close(&download(&gpu, &actual), &expected, 0.0);
}
