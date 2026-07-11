// Run with: cargo test --features cuda -- --ignored
use crate::{DType, Error, MemoryKind, Tensor, TypedTensor};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    GpuBackendKind, StridedSliceSpec, TensorIndexing, TensorStructural, TensorViewCanonicalization,
};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_bool, tensor_c32, tensor_c64,
    tensor_f32, tensor_f64, tensor_i32, tensor_i64, upload,
};

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
            assert_eq!($cpu.unwrap_err(), $gpu.unwrap_err());
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
                (Err(expected), Err(actual)) => assert_eq!(actual, expected),
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
        assert_eq!(
            gpu.cast(&gpu_source, target).unwrap_err(),
            cpu.cast(&source, target).unwrap_err()
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
    }
}

#[test]
#[ignore]
fn cuda_to_contiguous_keeps_tensor_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
    let gpu_input = upload(&gpu, &input);
    let Tensor::I32(gpu_tensor) = gpu_input else {
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
    let actual = download(&gpu, &Tensor::I32(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 5, 2, 4, 6]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_preserves_negative_stride_view() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![4], vec![1, 2, 3, 4]);
    let gpu_input = upload(&gpu, &input);
    let Tensor::I32(gpu_tensor) = gpu_input else {
        panic!("expected i32 tensor");
    };
    let view = gpu_tensor
        .as_view()
        .try_slice_axis(0, StridedSliceSpec::reverse())
        .unwrap();

    let compact = gpu.to_contiguous(&view).unwrap();

    let actual = download(&gpu, &Tensor::I32(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[4, 3, 2, 1]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_rank_zero_scalar_stays_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![], vec![7]);
    let gpu_input = upload(&gpu, &input);
    let Tensor::I32(gpu_tensor) = gpu_input else {
        panic!("expected i32 tensor");
    };

    let compact = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap();

    assert_eq!(compact.shape(), &[] as &[usize]);
    assert_eq!(compact.placement().memory_kind, MemoryKind::Device);
    let actual = download(&gpu, &Tensor::I32(compact));
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[7]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_empty_view_stays_on_cuda() {
    let mut gpu = gpu_backend();
    let input = tensor_i32(vec![0, 3], vec![]);
    let gpu_input = upload(&gpu, &input);
    let Tensor::I32(gpu_tensor) = gpu_input else {
        panic!("expected i32 tensor");
    };

    let compact = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap();

    assert_eq!(compact.shape(), &[0, 3]);
    assert_eq!(compact.placement().memory_kind, MemoryKind::Device);
    let actual = download(&gpu, &Tensor::I32(compact));
    assert_eq!(actual.shape(), &[0, 3]);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[] as &[i32]);
}

#[test]
#[ignore]
fn cuda_to_contiguous_bool_view_returns_backend_failure() {
    let mut gpu = gpu_backend();
    let input = tensor_bool(vec![2], vec![true, false]);
    let gpu_input = upload(&gpu, &input);
    let Tensor::Bool(gpu_tensor) = gpu_input else {
        panic!("expected bool tensor");
    };

    let err = gpu.to_contiguous(&gpu_tensor.as_view()).unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CudaBackend::to_contiguous",
            ref message,
        } if message.contains("unsupported dtype")
    ));
}

#[test]
#[ignore]
fn cuda_to_contiguous_host_view_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let host = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();

    let err = gpu.to_contiguous(&host.as_view()).unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CudaBackend::to_contiguous",
            ref message,
        } if message.contains("upload_tensor()")
    ));
}

#[test]
#[ignore]
fn cuda_copy_from_contiguous_host_source_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let src = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let dst_host = tensor_i32(vec![2], vec![0, 0]);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let Tensor::I32(dst) = &mut gpu_dst else {
        panic!("expected i32 tensor");
    };

    let err = gpu
        .copy_from_contiguous(&src, &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CudaBackend::copy_from_contiguous",
            ref message,
        } if message.contains("upload_tensor()")
    ));
}

#[test]
#[ignore]
fn cuda_copy_from_contiguous_host_destination_returns_upload_hint() {
    let mut gpu = gpu_backend();
    let src_host = tensor_i32(vec![2], vec![1, 2]);
    let gpu_src = upload(&gpu, &src_host);
    let Tensor::I32(src) = &gpu_src else {
        panic!("expected i32 tensor");
    };
    let mut dst = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![0, 0]).unwrap();

    let err = gpu
        .copy_from_contiguous(src, &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CudaBackend::copy_from_contiguous",
            ref message,
        } if message.contains("upload_tensor()")
    ));
}

#[test]
#[ignore]
fn cuda_copy_from_contiguous_updates_strided_view_on_cuda() {
    let mut gpu = gpu_backend();
    let dst_host = tensor_i32(vec![2, 2], vec![0, 0, 0, 0]);
    let src_host = tensor_i32(vec![2, 2], vec![1, 2, 3, 4]);
    let mut gpu_dst = upload(&gpu, &dst_host);
    let gpu_src = upload(&gpu, &src_host);

    let (Tensor::I32(dst), Tensor::I32(src)) = (&mut gpu_dst, &gpu_src) else {
        panic!("expected i32 tensors");
    };
    let mut dst_view = dst.as_view_mut().transpose_view([1, 0]).unwrap();

    gpu.copy_from_contiguous(src, &mut dst_view).unwrap();

    let actual = download(&gpu, &gpu_dst);
    assert_eq!(actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);
}
