use super::*;

#[test]
fn svd_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().svd_read(TensorView::F64(view)).unwrap();

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
fn qr_read_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().qr_read(TensorView::F64(view)).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let q = matrix_f64_from_tensor(&outputs[0], 3, 2);
    let r = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_f64(&q, &r, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn qr_read_canonicalizes_transposed_complex_host_view_before_lapack() {
    let data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().qr_read(TensorView::C64(view)).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let q = matrix_c64_from_tensor(&outputs[0], 3, 2);
    let r = matrix_c64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_c64(&q, &r, 3, 2, 2);
    let expected = transpose_c64(&data, 2, 3);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn eigh_read_canonicalizes_transposed_host_view_before_lapack() {
    let data = vec![4.0, 1.0, 1.0, 3.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().eigh_read(TensorView::F64(view)).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let values = match &outputs[0] {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        _ => panic!("expected f64 eigenvalues"),
    };
    let vectors = matrix_f64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&values), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&data, 2, 2);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn eigh_read_canonicalizes_transposed_complex_host_view_before_lapack() {
    let data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, 0.0),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().eigh_read(TensorView::C64(view)).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].dtype(), DType::F64);
    assert_eq!(outputs[1].dtype(), DType::C64);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);

    let values = vector_f64_from_tensor(&outputs[0], 2);
    let vectors = matrix_c64_from_tensor(&outputs[1], 2, 2);
    let recon = matmul_c64(
        &matmul_c64(&vectors, &diag_c64_from_real(&values), 2, 2, 2),
        &conjugate_transpose_c64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c64(&data, 2, 2);

    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn qr_read_accepts_all_supported_linalg_view_dtypes() {
    let f32_input =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0, -2.0, 0.5, 4.0]).unwrap();
    let f64_input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, -2.0, 0.5, 4.0]).unwrap();
    let c32_input = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(-2.0, 1.0),
            Complex32::new(0.5, -0.25),
            Complex32::new(4.0, 1.5),
        ],
    )
    .unwrap();
    let c64_input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(4.0, 1.5),
        ],
    )
    .unwrap();

    let mut backend = CpuBackend::new();
    for (view, dtype) in [
        (
            TensorView::F32(f32_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F32,
        ),
        (
            TensorView::F64(f64_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F64,
        ),
        (
            TensorView::C32(c32_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::C32,
        ),
        (
            TensorView::C64(c64_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::C64,
        ),
    ] {
        let outputs = backend.qr_read(view).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].dtype(), dtype);
        assert_eq!(outputs[1].dtype(), dtype);
        assert_eq!(outputs[0].shape(), &[2, 2]);
        assert_eq!(outputs[1].shape(), &[2, 2]);
    }
}

#[test]
fn eigh_read_accepts_all_supported_linalg_view_dtypes() {
    let f32_input =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let f64_input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let c32_input = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(1.0, 0.5),
            Complex32::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let c64_input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();

    let mut backend = CpuBackend::new();
    for (view, value_dtype, vector_dtype) in [
        (
            TensorView::F32(f32_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F32,
            DType::F32,
        ),
        (
            TensorView::F64(f64_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F64,
            DType::F64,
        ),
        (
            TensorView::C32(c32_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F32,
            DType::C32,
        ),
        (
            TensorView::C64(c64_input.as_view().transpose_view([1, 0]).unwrap()),
            DType::F64,
            DType::C64,
        ),
    ] {
        let outputs = backend.eigh_read(view).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].dtype(), value_dtype);
        assert_eq!(outputs[1].dtype(), vector_dtype);
        assert_eq!(outputs[0].shape(), &[2]);
        assert_eq!(outputs[1].shape(), &[2, 2]);
    }
}

#[test]
fn linalg_read_rejects_non_float_view_dtypes() {
    fn assert_unsupported_dtype(
        result: tenferro_tensor::Result<Vec<Tensor>>,
        expected_op: &'static str,
    ) {
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            tenferro_tensor::Error::BackendFailure {
                op,
                ref message,
            } if op == expected_op && message.contains("unsupported dtype")
        ));
    }

    let i32_input = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![1, 0, 0, 1]).unwrap();
    let i64_input =
        TypedTensor::<i64>::from_vec_col_major(vec![2, 2], vec![1_i64, 0, 0, 1]).unwrap();
    let bool_input =
        TypedTensor::<bool>::from_vec_col_major(vec![2, 2], vec![true, false, false, true])
            .unwrap();

    let mut backend = CpuBackend::new();
    assert_unsupported_dtype(
        backend.svd_read(TensorView::I32(i32_input.as_view())),
        "svd",
    );
    assert_unsupported_dtype(
        backend.svd_read(TensorView::I64(i64_input.as_view())),
        "svd",
    );
    assert_unsupported_dtype(
        backend.svd_read(TensorView::Bool(bool_input.as_view())),
        "svd",
    );
    assert_unsupported_dtype(backend.qr_read(TensorView::I32(i32_input.as_view())), "qr");
    assert_unsupported_dtype(backend.qr_read(TensorView::I64(i64_input.as_view())), "qr");
    assert_unsupported_dtype(
        backend.qr_read(TensorView::Bool(bool_input.as_view())),
        "qr",
    );
    assert_unsupported_dtype(
        backend.eigh_read(TensorView::I32(i32_input.as_view())),
        "eigh",
    );
    assert_unsupported_dtype(
        backend.eigh_read(TensorView::I64(i64_input.as_view())),
        "eigh",
    );
    assert_unsupported_dtype(
        backend.eigh_read(TensorView::Bool(bool_input.as_view())),
        "eigh",
    );
}

#[test]
fn test_batched_cholesky() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);

    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
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
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![4, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
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
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
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
    let a = Tensor::F64(
        TypedTensor::from_vec_col_major(
            vec![3, 3, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let b = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![3, 1, 2], vec![1.0, 2.0, 3.0, -1.0, 4.0, 0.5])
            .unwrap(),
    );

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
    let l = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 3], l_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 1], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&l, &b, true, true, false, false)
        .unwrap();

    assert_eq!(x.shape(), &[3, 1]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap(),
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
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
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

                    let a =
                        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data).unwrap());
                    let b =
                        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data).unwrap());
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::F64(inner) => inner.host_data().unwrap(),
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
    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );
    let b = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 1, 2],
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(0.5, 2.0),
                Complex64::new(-2.0, 0.25),
                Complex64::new(1.5, -0.75),
            ],
        )
        .unwrap(),
    );

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
    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = backend.solve(&a, &b).unwrap();

    let x_data = match &x {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected f64 tensor"),
    };
    let recon = matmul_f64(&a_data, x_data, 2, 2, 2);
    for (actual, expected) in recon.iter().zip(b_data.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn test_real_lu_returns_permutation_factors_and_parity() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]).unwrap());
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
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]).unwrap());
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
    let input = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2, 2],
            a0.iter().chain(a1.iter()).copied().collect(),
        )
        .unwrap(),
    );

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].dtype(), DType::F64);
    assert_eq!(out[1].dtype(), DType::C64);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[1].shape(), &[2, 2, 2]);

    for batch_idx in 0..2 {
        let values = batch_vector_f64_from_tensor(&out[0], 2, batch_idx);
        let vectors = batch_matrix_c64_from_tensor(&out[1], 2, 2, batch_idx);
        let recon = matmul_c64(
            &matmul_c64(&vectors, &diag_c64_from_real(&values), 2, 2, 2),
            &conjugate_transpose_c64(&vectors, 2, 2),
            2,
            2,
            2,
        );
        let expected = batch_matrix_c64_from_tensor(&input, 2, 2, batch_idx);
        for (actual, expected) in recon.iter().zip(expected.iter()) {
            assert_c64_close_tol(*actual, *expected, 1.0e-10);
        }
    }
}

#[test]
fn test_real_eigh() {
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let out = backend.eigh(&input).unwrap();

    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let values = match &out[0] {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
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
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap());
    let mut backend = CpuBackend::new();
    let err = backend.cholesky(&input).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "cholesky", .. }
    ));
}

#[test]
fn test_real_solve_returns_error_for_singular_matrix() {
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 1.0]).unwrap());
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
    let input = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a.clone()).unwrap());

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
    let input = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap(),
    );
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
    let input =
        Tensor::C64(TypedTensor::from_vec_col_major(vec![3, 2], input_data.clone()).unwrap());

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
    let input =
        Tensor::C64(TypedTensor::from_vec_col_major(vec![3, 2], input_data.clone()).unwrap());
    let mut backend = CpuBackend::new();
    let out = backend.svd(&input).unwrap();

    assert_eq!(out.len(), 3);
    assert_eq!(out[0].dtype(), DType::C64);
    assert_eq!(out[1].dtype(), DType::F64);
    assert_eq!(out[2].dtype(), DType::C64);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_c64_from_tensor(&out[0], 3, 2);
    let s = vector_f64_from_tensor(&out[1], 2);
    let vt = matrix_c64_from_tensor(&out[2], 2, 2);
    let recon = matmul_c64(
        &matmul_c64(&u, &diag_c64_from_real(&s), 3, 2, 2),
        &vt,
        3,
        2,
        2,
    );
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
    let a = Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap());
    let b = Tensor::C64(TypedTensor::from_vec_col_major(vec![1, 2], b_data.clone()).unwrap());

    let mut backend = CpuBackend::new();
    let x = backend
        .triangular_solve(&a, &b, false, true, true, true)
        .unwrap();

    assert_eq!(x.shape(), &[1, 2]);
    let x_data = match &x {
        Tensor::C64(inner) => inner.host_data().unwrap().to_vec(),
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

                    let a =
                        Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], a_data).unwrap());
                    let b =
                        Tensor::C64(TypedTensor::from_vec_col_major(vec![2, 2], b_data).unwrap());
                    let mut backend = CpuBackend::new();
                    let x = backend
                        .triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
                        .unwrap();

                    let x_data = match &x {
                        Tensor::C64(inner) => inner.host_data().unwrap(),
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
    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let b = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 1],
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let err = backend.solve(&a, &b).unwrap_err();
    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure { op: "solve", .. }
    ));
}

#[test]
fn svd_read_faer_strided_view_matches_contiguous() {
    // 2x3 matrix stored col-major, then transposed to give a 3x2 strided view.
    let data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0]; // 2x3 col-major
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let out = CpuBackend::new().svd_read(TensorView::F64(view)).unwrap();
    // For 3x2 input (m=3, n=2), thin SVD gives U:[3,2], S:[2], Vt:[2,2].
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let u = matrix_f64_from_tensor(&out[0], 3, 2);
    let s = (0..2).map(|i| get_f64(&out[1], &[i])).collect::<Vec<_>>();
    let vt = matrix_f64_from_tensor(&out[2], 2, 2);
    let recon = matmul_f64(&matmul_f64(&u, &diag_f64(&s), 3, 2, 2), &vt, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3); // A^T is 3x2 col-major
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn svd_read_faer_strided_c64_view() {
    let data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let a = TypedTensor::<Complex64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let out = CpuBackend::new().svd_read(TensorView::C64(view)).unwrap();
    assert_eq!(out[0].shape(), &[3, 2]); // U (thin, complex)
    assert_eq!(out[1].shape(), &[2]); // S (real singular values)
    assert_eq!(out[2].shape(), &[2, 2]); // Vt (thin, complex)

    // Singular values are returned as a real tensor, mirroring the materialized path.
    let s_vals = match &out[1] {
        Tensor::F64(t) => t.host_data().unwrap().to_vec(),
        Tensor::C64(t) => t.host_data().unwrap().iter().map(|c| c.re).collect(),
        _ => panic!("unexpected type for singular values"),
    };
    assert!(s_vals.iter().all(|&v| v.is_finite() && v >= 0.0));
    assert!(s_vals[0] >= s_vals[1]); // singular values descending
}

#[test]
fn qr_read_faer_strided_view_matches_contiguous() {
    let data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 3x2 strided
    let out = CpuBackend::new().qr_read(TensorView::F64(view)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[2, 2]);

    let q = matrix_f64_from_tensor(&out[0], 3, 2);
    let r = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(&q, &r, 3, 2, 2);
    let expected = transpose_f64(&data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-9);
    }
}

#[test]
fn eigh_read_faer_strided_view_matches_contiguous() {
    // Symmetric 2x2 stored col-major, then transposed to a strided view (still symmetric).
    let data = vec![4.0_f64, 1.0, 1.0, 3.0]; // 2x2 symmetric
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], data.clone()).unwrap();
    let view = a.as_view().transpose_view([1, 0]).unwrap(); // 2x2 strided (still symmetric)
    let out = CpuBackend::new().eigh_read(TensorView::F64(view)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].shape(), &[2]); // eigenvalues
    assert_eq!(out[1].shape(), &[2, 2]); // eigenvectors

    let eigenvalues = (0..2).map(|i| get_f64(&out[0], &[i])).collect::<Vec<_>>();
    assert!(eigenvalues[0].is_finite());
    assert!(eigenvalues[1].is_finite());

    // Reconstruct A = V diag(lambda) V^T and compare against the (symmetric) input view.
    let vectors = matrix_f64_from_tensor(&out[1], 2, 2);
    let recon = matmul_f64(
        &matmul_f64(&vectors, &diag_f64(&eigenvalues), 2, 2, 2),
        &transpose_f64(&vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close_tol(*actual, *expected, 1.0e-10);
    }
}
