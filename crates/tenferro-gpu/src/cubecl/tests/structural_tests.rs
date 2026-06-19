// Run with: cargo test --features cuda -- --ignored
use crate::{DType, Error, MemoryKind, Tensor, TypedTensor};
use tenferro_tensor::{
    GpuBackendKind, StridedSliceSpec, TensorIndexing, TensorStructural, TensorViewCanonicalization,
};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_bool, tensor_c64, tensor_f64,
    tensor_i32, tensor_i64, upload,
};

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
