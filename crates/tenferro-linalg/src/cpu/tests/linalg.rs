use super::*;

#[test]
fn svd_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone());
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().svd_view(TensorView::F64(view)).unwrap();

    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);

    let u = matrix_f64_from_tensor(&outputs[0], 3, 2);
    let s = (0..2)
        .map(|i| get_f64(&outputs[1], &[i]))
        .collect::<Vec<_>>();
    let vt = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 3, 2, 2), &vt, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_batched_cholesky() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);

    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input).unwrap();

    assert_eq!(out.shape(), &[3, 3, 2]);
    for batch_idx in 0..2 {
        let l = batch_matrix_f64_from_tensor(&out, 3, 3, batch_idx);
        let recon = matmul_f64(&l, &transpose_f64(&l, 3, 3), 3, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_batched_svd() {
    let a0 = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5, 2.0, 0.0, 1.0, -0.5];
    let a1 = vec![
        2.0, -1.0, 0.5, 3.0, -0.25, 1.5, -2.0, 0.75, 1.0, 2.5, -1.0, 4.0,
    ];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![4, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].shape(), &[4, 3, 2]);
    assert_eq!(out[1].shape(), &[3, 2]);
    assert_eq!(out[2].shape(), &[3, 3, 2]);

    for batch_idx in 0..2 {
        let u = batch_matrix_f64_from_tensor(&out[0], 4, 3, batch_idx);
        let s = batch_vector_f64_from_tensor(&out[1], 3, batch_idx);
        let vt = batch_matrix_f64_from_tensor(&out[2], 3, 3, batch_idx);
        let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 4, 3, 3), &vt, 4, 3, 3);
        let expected = batch_matrix_f64_from_tensor(&input, 4, 3, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_qr() {
    let a0 = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0];
    let a1 = [2.0, -1.0, 0.5, 3.0, -0.25, 1.5];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.qr(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let q = batch_matrix_f64_from_tensor(&out[0], 3, 2, batch_idx);
        let r = batch_matrix_f64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_f64(&q, &r, 3, 2, 2);
        let expected = batch_matrix_f64_from_tensor(&input, 3, 2, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-9);
        }
    }
}

#[test]
fn test_batched_solve() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 3, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3, 1, 2],
        vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.5],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    assert_eq!(x.shape(), &[3, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_f64_from_tensor(&a, 3, 3, batch_idx);
        let x_batch = batch_matrix_f64_from_tensor(&x, 3, 1, batch_idx);
        let recon = matmul_f64(&a_batch, &x_batch, 3, 3, 1);
        let expected = batch_matrix_f64_from_tensor(&b, 3, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_f64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_triangular_solve_lower() {
    let l_data = vec![2.0, 1.0, -0.5, 0.0, 3.0, 1.25, 0.0, 0.0, 1.5];
    let b_data = vec![1.0, -2.0, 0.5];
    let l = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 3], l_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 1], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&l, &b, true, true, false, false)
        .unwrap();

    assert_eq!(x.shape(), &[3, 1]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&l_data, x_data, 3, 3, 1);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![1.0, 2.0, 0.0, 1.0];
    let b_data = vec![7.0, 5.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&x_data, &transpose_f64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_real_branch_combinations() {
    let expected_x = vec![1.0, -2.0, 0.5, 3.0];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (1.0, 1.0)
                    } else {
                        (2.0, 1.5)
                    };
                    let a_data = if lower {
                        vec![diagonal.0, -0.75, 0.0, diagonal.1]
                    } else {
                        vec![diagonal.0, 0.0, 0.5, diagonal.1]
                    };
                    let op_a = if transpose_a {
                        transpose_f64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_f64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_f64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data));
                    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data));
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::F64(inner) => inner.host_data(),
                        _ => panic!("expected f64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_f64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_batched_complex_solve() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 1, 2],
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 2.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(1.5, -0.75),
        ],
    ));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    assert_eq!(x.shape(), &[2, 1, 2]);
    for batch_idx in 0..2 {
        let a_batch = batch_matrix_c64_from_tensor(&a, 2, 2, batch_idx);
        let x_batch = batch_matrix_c64_from_tensor(&x, 2, 1, batch_idx);
        let recon = matmul_c64(&a_batch, &x_batch, 2, 2, 1);
        let expected = batch_matrix_c64_from_tensor(&b, 2, 1, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_solve_non_batched() {
    let a_data = vec![3.0, 1.0, 1.0, 2.0];
    let b_data = vec![5.0, 1.0, -2.0, 4.0];
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&a_data, x_data, 2, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_lu_returns_permutation_factors_and_parity() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0, 1.0, 1.0, 0.0],
    ));
    let mut backend = CpuBackend::new();
    let outputs = backend.lu(&input).unwrap();

    assert_eq!(outputs.len(), 4);
    let p = matrix_f64_from_tensor(&outputs[0], 2, 2);
    let l = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let u = matrix_f64_from_tensor(&outputs[2], 2, 2);
    let parity = get_f64(&outputs[3], &[]);

    let pa = matmul_f64(&p, &matrix_f64_from_tensor(&input, 2, 2), 2, 2, 2);
    let lu = matmul_f64(&l, &u, 2, 2, 2);
    for (actual, expected) in pa.iter().zip(lu.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
    assert_f64_close(parity, -1.0);
}

#[test]
fn test_real_eig_returns_complex_outputs() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 0.0, 0.0, 3.0],
    ));
    let mut backend = CpuBackend::new();
    let outputs = backend.eig(&input).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let mut values = vector_c64_from_tensor(&outputs[0], 2);
    values.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_c64_close(values[0], Complex64::new(1.0, 0.0));
    assert_c64_close(values[1], Complex64::new(3.0, 0.0));
}

#[test]
fn test_batched_complex_eigh() {
    let l0 = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let l1 = vec![
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let a0 = matmul_c64(&l0, &conjugate_transpose_c64(&l0, 2, 2), 2, 2, 2);
    let a1 = matmul_c64(&l1, &conjugate_transpose_c64(&l1, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2, 2],
        a0.iter().chain(a1.iter()).copied().collect(),
    ));

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let values = batch_vector_c64_from_tensor(&out[0], 2, batch_idx);
        let vectors = batch_matrix_c64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_c64(
            &matmul_c64(&vectors, &diag_c64(&values), 2, 2, 2),
            &conjugate_transpose_c64(&vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = batch_matrix_c64_from_tensor(&input, 2, 2, batch_idx);
        for value in &values {
            assert_f64_close_tol(value.im, 0.0, 1.0e-12);
        }
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_eigh() {
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let values = match &out[0] {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected f64 tensor"),
    };
    let vectors = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&values), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    for (actual, expected) in recon.iter().zip(a_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_cholesky_returns_error_for_non_positive_definite_input() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 2.0, 1.0],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.cholesky(&input).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "cholesky", .. }
    ));
}

#[test]
fn test_real_solve_returns_error_for_singular_matrix() {
    let a = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 2.0, 4.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 1.0]));
    let mut backend = CpuBackend::new();
    let err = backend.solve(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "solve", .. }
    ));
}

#[test]
fn test_complex_cholesky() {
    let l = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let a = matmul_c64(&l, &conjugate_transpose_c64(&l, 2, 2), 2, 2, 2);
    let input = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a.clone()));

    let mut backend = CpuBackend::new();
    let out = backend.cholesky(&input).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let l_out = matrix_c64_from_tensor(&out, 2, 2);
    let recon = matmul_c64(&l_out, &conjugate_transpose_c64(&l_out, 2, 2), 2, 2, 2);
    for (actual, expected) in recon.iter().zip(a.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_complex_cholesky_returns_error_for_non_positive_definite_input() {
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.cholesky(&input).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "cholesky", .. }
    ));
}

#[test]
fn test_complex_qr() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![3, 2],
        input_data.clone(),
    ));

    let mut backend = CpuBackend::new();
    let out = backend.qr(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let q = matrix_c64_from_tensor(&out[0], 3, 2);
    let r = matrix_c64_from_tensor(&out[1], 2, 2);
    let recon = matmul_c64(&q, &r, 3, 2, 2);
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_svd() {
    let input_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -0.5),
        Complex64::new(-1.0, 2.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(-0.25, 1.5),
        Complex64::new(3.0, 0.75),
    ];
    let input = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![3, 2],
        input_data.clone(),
    ));
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_c64_from_tensor(&out[0], 3, 2);
    let s = vector_c64_from_tensor(&out[1], 2);
    let vt = matrix_c64_from_tensor(&out[2], 2, 2);
    let recon = matmul_c64(&matmul_c64(&u, &diag_c64(&s), 3, 2, 2), &vt, 3, 2, 2);
    for (actual, expected) in recon.iter().zip(input_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn test_complex_triangular_solve_right_side_unit_transpose() {
    let a_data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let b_data = vec![Complex64::new(2.0, 1.0), Complex64::new(-1.0, 0.5)];
    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()));

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::C64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected c64 tensor"),
    };
    let recon = matmul_c64(&x_data, &transpose_c64(&a_data, 2, 2), 1, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_triangular_solve_covers_all_complex_branch_combinations() {
    let expected_x = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, -0.5),
        Complex64::new(3.0, -1.0),
    ];

    for &left_side in &[true, false] {
        for &lower in &[true, false] {
            for &transpose_a in &[false, true] {
                for &unit_diagonal in &[false, true] {
                    let diagonal = if unit_diagonal {
                        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
                    } else {
                        (Complex64::new(2.0, 0.0), Complex64::new(1.5, 0.0))
                    };
                    let a_data = if lower {
                        vec![
                            diagonal.0,
                            Complex64::new(-0.75, 0.25),
                            Complex64::new(0.0, 0.0),
                            diagonal.1,
                        ]
                    } else {
                        vec![
                            diagonal.0,
                            Complex64::new(0.0, 0.0),
                            Complex64::new(0.5, -0.25),
                            diagonal.1,
                        ]
                    };
                    let op_a = if transpose_a {
                        transpose_c64(&a_data, 2, 2)
                    } else {
                        a_data.clone()
                    };
                    let b_data = if left_side {
                        matmul_c64(&op_a, &expected_x, 2, 2, 2)
                    } else {
                        matmul_c64(&expected_x, &op_a, 2, 2, 2)
                    };

                    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data));
                    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], b_data));
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::C64(inner) => inner.host_data(),
                        _ => panic!("expected c64 tensor"),
                    };
                    for (actual, expected) in x_data.iter().zip(expected_x.iter()) {
                        assert_c64_close_tol(*actual, *expected, 1.0e-10);
                    }
                }
            }
        }
    }
}

#[test]
fn test_complex_solve_returns_error_for_singular_matrix() {
    let a = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    ));
    let b = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    ));
    let mut backend = CpuBackend::new();
    let err = backend.solve(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "solve", .. }
    ));
}
