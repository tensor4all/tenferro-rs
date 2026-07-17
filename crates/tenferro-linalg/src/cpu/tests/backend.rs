use super::*;

#[test]
fn test_solve_zero_dim_returns_zeros() {
    let mut backend = CpuBackend::new();
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 0], vec![]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0, 1], vec![]).unwrap());
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[0, 1]);
}

#[test]
fn test_solve_with_1d_vector_rhs() {
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 3.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![5.0, 7.0]).unwrap());
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2]);
    let expected = matmul_f64(
        &[2.0, 1.0, 0.0, 3.0],
        &[get_f64(&x, &[0]), get_f64(&x, &[1])],
        2,
        2,
        1,
    );
    assert_f64_close_tol(expected[0], 5.0, 1e-10);
    assert_f64_close_tol(expected[1], 7.0, 1e-10);
}

#[test]
fn test_solve_with_batched_vector_rhs() {
    let a = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2],
            vec![2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        )
        .unwrap(),
    );
    let b =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![5.0, 7.0, 3.0, 4.0]).unwrap());
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
}

#[test]
fn test_triangular_solve_dtype_mismatch_and_unsupported() {
    let mut backend = CpuBackend::new();
    let a_f32 = Tensor::F32(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0]).unwrap(),
    );
    let b_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap());
    let err = backend
        .triangular_solve(&a_f32, &b_f64, true, true, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::Validation {
            op: "triangular_solve",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));

    let a_i64 =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap());
    let b_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1_i64, 2]).unwrap());
    let err = backend
        .triangular_solve(&a_i64, &b_i64, true, true, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure {
            op: "triangular_solve",
            ..
        }
    ));
}

#[test]
fn test_linalg_returns_errors_for_unsupported_dtypes() {
    let mut backend = CpuBackend::new();
    let i64_matrix =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap());
    let i64_rhs = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1_i64, 2]).unwrap());

    assert!(backend.cholesky(&i64_matrix).is_err());
    assert!(backend.svd(&i64_matrix).is_err());
    assert!(backend.qr(&i64_matrix).is_err());
    assert!(backend.eigh(&i64_matrix).is_err());
    assert!(backend.eig(&i64_matrix).is_err());
    assert!(backend.solve(&i64_matrix, &i64_rhs).is_err());
    assert!(backend
        .triangular_solve(&i64_matrix, &i64_rhs, true, true, false, false)
        .is_err());
}

#[test]
fn test_solve_zero_dim_rhs_returns_zeros() {
    let mut backend = CpuBackend::new();
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![0], vec![]).unwrap());
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[0]);
}

#[test]
fn test_solve_with_regular_matrix_rhs() {
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 3.0]).unwrap());
    let b =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![5.0, 7.0, 3.0, 4.0]).unwrap());
    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&[2.0, 1.0, 0.0, 3.0], &x_data, 2, 2, 2);
    assert_f64_close_tol(recon[0], 5.0, 1e-10);
    assert_f64_close_tol(recon[1], 7.0, 1e-10);
    assert_f64_close_tol(recon[2], 3.0, 1e-10);
    assert_f64_close_tol(recon[3], 4.0, 1e-10);
}

#[test]
fn test_lu_unsupported_dtype_returns_error() {
    let input =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap());
    let mut backend = CpuBackend::new();
    assert!(backend.lu(&input).is_err());
}

#[test]
fn test_lu_zero_sized_batch_outputs_empty_parity() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2, 0], Vec::new()).unwrap());
    let mut backend = CpuBackend::new();
    let outputs = backend.lu(&input).unwrap();

    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].shape(), &[2, 2, 0]);
    assert_eq!(outputs[1].shape(), &[2, 2, 0]);
    assert_eq!(outputs[2].shape(), &[2, 2, 0]);
    assert_eq!(outputs[3].shape(), &[0]);
    for output in outputs {
        match output {
            Tensor::F64(inner) => assert!(inner.host_data().unwrap().is_empty()),
            other => panic!("expected f64 tensor, got {:?}", other.dtype()),
        }
    }
}

#[test]
fn test_svd_unsupported_dtype_returns_error() {
    let input =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap());
    let mut backend = CpuBackend::new();
    assert!(backend.svd(&input).is_err());
}

#[cfg(feature = "cpu-faer")]
#[test]
fn test_faer_svd_decomposition_failure_returns_error() {
    let ctx = CpuContext::with_threads(1).unwrap();
    let mut buffers = BufferPool::new();
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![f64::NAN, 0.0, 0.0, 1.0]).unwrap();

    let err = faer_linalg::svd(&ctx, &mut buffers, &input).unwrap_err();

    assert!(err.to_string().contains("svd"), "unexpected error: {err}");
}

#[cfg(feature = "cpu-faer")]
#[test]
fn test_faer_eig_decomposition_failure_returns_error() {
    let ctx = CpuContext::with_threads(1).unwrap();
    let mut buffers = BufferPool::new();
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![f64::NAN, 0.0, 0.0, 1.0]).unwrap(),
    );

    let err = faer_linalg::eig(&ctx, &mut buffers, &input).unwrap_err();

    assert!(err.to_string().contains("eig"), "unexpected error: {err}");
}

#[test]
fn test_qr_unsupported_dtype_returns_error() {
    let input =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap());
    let mut backend = CpuBackend::new();
    assert!(backend.qr(&input).is_err());
}

#[test]
fn test_eig_returns_complex_outputs_for_real_input() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0, -1.0, 1.0, 0.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let outputs = backend.eig(&input).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
}
