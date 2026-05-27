use super::*;

fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}

fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c32(
    lhs: &[Complex32],
    rhs: &[Complex32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn transpose_f32(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn conjugate_transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}

fn diag_f32(values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn diag_c32(values: &[Complex32]) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn f32_data(tensor: &Tensor) -> &[f32] {
    match tensor {
        Tensor::F32(inner) => inner.host_data(),
        other => panic!("expected F32 tensor, got {:?}", other.dtype()),
    }
}

fn c32_data(tensor: &Tensor) -> &[Complex32] {
    match tensor {
        Tensor::C32(inner) => inner.host_data(),
        other => panic!("expected C32 tensor, got {:?}", other.dtype()),
    }
}

fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}

fn assert_c32_close(actual: Complex32, expected: Complex32, tol: f32) {
    assert_f32_close(actual.re, expected.re, tol);
    assert_f32_close(actual.im, expected.im, tol);
}

fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_f32_close(*actual, *expected, tol);
    }
}

fn assert_c32_slice_close(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_c32_close(*actual, *expected, tol);
    }
}

#[test]
fn cpu_linalg_accepts_f32_happy_paths() {
    let mut backend = CpuBackend::new();
    let tol = 5.0e-4;

    let lower = vec![2.0, 0.5, 0.0, 1.5];
    let spd = matmul_f32(&lower, &transpose_f32(&lower, 2, 2), 2, 2, 2);
    let chol = backend
        .cholesky(&Tensor::F32(TypedTensor::from_vec_col_major(
            vec![2, 2],
            spd.clone(),
        )))
        .unwrap();
    let chol_recon = matmul_f32(
        f32_data(&chol),
        &transpose_f32(f32_data(&chol), 2, 2),
        2,
        2,
        2,
    );
    assert_f32_slice_close(&chol_recon, &spd, tol);

    let rectangular = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0];
    let input = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![3, 2],
        rectangular.clone(),
    ));
    let qr = backend.qr(&input).unwrap();
    assert_eq!(qr[0].shape(), &[3, 2]);
    assert_eq!(qr[1].shape(), &[2, 2]);
    assert_f32_slice_close(
        &matmul_f32(f32_data(&qr[0]), f32_data(&qr[1]), 3, 2, 2),
        &rectangular,
        tol,
    );

    let svd = backend.svd(&input).unwrap();
    assert_eq!(svd[0].shape(), &[3, 2]);
    assert_eq!(svd[1].shape(), &[2]);
    assert_eq!(svd[2].shape(), &[2, 2]);
    assert_f32_slice_close(
        &matmul_f32(
            &matmul_f32(f32_data(&svd[0]), &diag_f32(f32_data(&svd[1])), 3, 2, 2),
            f32_data(&svd[2]),
            3,
            2,
            2,
        ),
        &rectangular,
        1.0e-3,
    );

    let symmetric = vec![4.0, 1.0, 1.0, 3.0];
    let eigh = backend
        .eigh(&Tensor::F32(TypedTensor::from_vec_col_major(
            vec![2, 2],
            symmetric.clone(),
        )))
        .unwrap();
    assert_f32_slice_close(
        &matmul_f32(
            &matmul_f32(f32_data(&eigh[1]), &diag_f32(f32_data(&eigh[0])), 2, 2, 2),
            &transpose_f32(f32_data(&eigh[1]), 2, 2),
            2,
            2,
            2,
        ),
        &symmetric,
        tol,
    );

    let eig = backend
        .eig(&Tensor::F32(TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 3.0],
        )))
        .unwrap();
    assert_eq!(eig[0].dtype(), DType::C32);
    assert_eq!(eig[1].dtype(), DType::C32);

    let a = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![3.0, 1.0, 1.0, 2.0],
    ));
    let b = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![5.0, 1.0, -2.0, 4.0],
    ));
    let x = backend.solve(&a, &b).unwrap();
    assert_f32_slice_close(
        &matmul_f32(f32_data(&a), f32_data(&x), 2, 2, 2),
        f32_data(&b),
        tol,
    );

    let triangular = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![2.0, 1.0, 0.0, 3.0],
    ));
    let rhs = Tensor::F32(TypedTensor::from_vec_col_major(vec![2, 1], vec![5.0, 7.0]));
    let y = backend
        .triangular_solve(&triangular, &rhs, true, true, false, false)
        .unwrap();
    assert_f32_slice_close(
        &matmul_f32(f32_data(&triangular), f32_data(&y), 2, 2, 1),
        f32_data(&rhs),
        tol,
    );

    let lu_input = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0, 1.0, 1.0, 0.0],
    ));
    let lu = backend.lu(&lu_input).unwrap();
    assert_eq!(lu.len(), 4);
    assert_eq!(lu[3].shape(), &[] as &[usize]);

    let full = backend.full_piv_lu(&lu_input).unwrap();
    assert_eq!(full.len(), 5);
    let full_x = backend.full_piv_lu_solve(&lu_input, &rhs, false).unwrap();
    assert_f32_slice_close(
        &matmul_f32(f32_data(&lu_input), f32_data(&full_x), 2, 2, 1),
        f32_data(&rhs),
        tol,
    );
}

#[test]
fn cpu_linalg_accepts_c32_happy_paths() {
    let mut backend = CpuBackend::new();
    let tol = 1.0e-3;

    let lower = vec![
        Complex32::new(2.0, 0.0),
        Complex32::new(0.5, -1.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.5, 0.0),
    ];
    let spd = matmul_c32(&lower, &conjugate_transpose_c32(&lower, 2, 2), 2, 2, 2);
    let chol = backend
        .cholesky(&Tensor::C32(TypedTensor::from_vec_col_major(
            vec![2, 2],
            spd.clone(),
        )))
        .unwrap();
    assert_c32_slice_close(
        &matmul_c32(
            c32_data(&chol),
            &conjugate_transpose_c32(c32_data(&chol), 2, 2),
            2,
            2,
            2,
        ),
        &spd,
        tol,
    );

    let rectangular = vec![
        Complex32::new(1.0, 1.0),
        Complex32::new(2.0, -0.5),
        Complex32::new(-1.0, 2.0),
        Complex32::new(0.5, -1.0),
        Complex32::new(-0.25, 1.5),
        Complex32::new(3.0, 0.75),
    ];
    let input = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![3, 2],
        rectangular.clone(),
    ));
    let qr = backend.qr(&input).unwrap();
    assert_c32_slice_close(
        &matmul_c32(c32_data(&qr[0]), c32_data(&qr[1]), 3, 2, 2),
        &rectangular,
        2.0e-3,
    );

    let svd = backend.svd(&input).unwrap();
    assert_c32_slice_close(
        &matmul_c32(
            &matmul_c32(c32_data(&svd[0]), &diag_c32(c32_data(&svd[1])), 3, 2, 2),
            c32_data(&svd[2]),
            3,
            2,
            2,
        ),
        &rectangular,
        2.0e-3,
    );

    let eigh = backend
        .eigh(&Tensor::C32(TypedTensor::from_vec_col_major(
            vec![2, 2],
            spd.clone(),
        )))
        .unwrap();
    assert_c32_slice_close(
        &matmul_c32(
            &matmul_c32(c32_data(&eigh[1]), &diag_c32(c32_data(&eigh[0])), 2, 2, 2),
            &conjugate_transpose_c32(c32_data(&eigh[1]), 2, 2),
            2,
            2,
            2,
        ),
        &spd,
        2.0e-3,
    );

    let eig_input = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, 0.0),
        ],
    ));
    let eig = backend.eig(&eig_input).unwrap();
    assert_eq!(eig[0].dtype(), DType::C32);
    assert_eq!(eig[1].dtype(), DType::C32);

    let a = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(3.0, 0.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, 0.0),
        ],
    ));
    let b = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex32::new(5.0, 1.0), Complex32::new(1.0, -2.0)],
    ));
    let x = backend.solve(&a, &b).unwrap();
    assert_c32_slice_close(
        &matmul_c32(c32_data(&a), c32_data(&x), 2, 2, 1),
        c32_data(&b),
        2.0e-3,
    );

    let triangular = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 0.0),
        ],
    ));
    let y = backend
        .triangular_solve(&triangular, &b, true, true, false, false)
        .unwrap();
    assert_c32_slice_close(
        &matmul_c32(c32_data(&triangular), c32_data(&y), 2, 2, 1),
        c32_data(&b),
        2.0e-3,
    );

    let lu = backend.lu(&a).unwrap();
    assert_eq!(lu.len(), 4);
    assert_eq!(lu[3].shape(), &[] as &[usize]);

    let full = backend.full_piv_lu(&a).unwrap();
    assert_eq!(full.len(), 5);
    let full_x = backend.full_piv_lu_solve(&a, &b, false).unwrap();
    assert_c32_slice_close(
        &matmul_c32(c32_data(&a), c32_data(&full_x), 2, 2, 1),
        c32_data(&b),
        2.0e-3,
    );
}
