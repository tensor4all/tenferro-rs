// Run with: cargo test --features cubecl -- --ignored
use num_complex::{Complex32, Complex64};

use crate::TensorBackend;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, upload,
};

fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}

fn transpose_f64(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            out[col_major_index(cols, col, row)] = data[col_major_index(rows, row, col)];
        }
    }
    out
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, col)];
            for row in 0..m {
                out[col_major_index(m, row, col)] += lhs[col_major_index(m, row, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, col)];
            for row in 0..m {
                out[col_major_index(m, row, col)] += lhs[col_major_index(m, row, p)] * rhs_pj;
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
    for col in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, col)];
            for row in 0..m {
                out[col_major_index(m, row, col)] += lhs[col_major_index(m, row, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, col)];
            for row in 0..m {
                out[col_major_index(m, row, col)] += lhs[col_major_index(m, row, p)] * rhs_pj;
            }
        }
    }
    out
}

fn diag_c32(values: &[Complex32]) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
    for (idx, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), idx, idx)] = *value;
    }
    out
}

fn diag_c64(values: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
    for (idx, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), idx, idx)] = *value;
    }
    out
}

fn transpose_c32(data: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            out[col_major_index(cols, col, row)] = data[col_major_index(rows, row, col)];
        }
    }
    out
}

fn conj_transpose_c64(data: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            out[col_major_index(cols, col, row)] = data[col_major_index(rows, row, col)].conj();
        }
    }
    out
}

fn assert_slice_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() <= tol, "lhs={lhs:?} rhs={rhs:?}");
    }
}

fn assert_lu_batch_reconstructs_f32(
    input: &[f32],
    p: &[f32],
    l: &[f32],
    u: &[f32],
    dims: (usize, usize, usize),
    batch: usize,
) {
    let (m, k, n) = dims;
    let a_start = batch * m * n;
    let p_start = batch * m * m;
    let l_start = batch * m * k;
    let u_start = batch * k * n;
    let a_batch = &input[a_start..a_start + m * n];
    let p_batch = &p[p_start..p_start + m * m];
    let l_batch = &l[l_start..l_start + m * k];
    let u_batch = &u[u_start..u_start + k * n];
    let pa = matmul_f32(p_batch, a_batch, m, m, n);
    let lu = matmul_f32(l_batch, u_batch, m, k, n);
    assert_slice_close_f32(&pa, &lu, 1e-4);
}

fn assert_slice_close_c32(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs.re - rhs.re).abs() <= tol, "lhs={lhs:?} rhs={rhs:?}");
        assert!((lhs.im - rhs.im).abs() <= tol, "lhs={lhs:?} rhs={rhs:?}");
    }
}

fn assert_slice_close_c64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs.re - rhs.re).abs() <= tol, "lhs={lhs:?} rhs={rhs:?}");
        assert!((lhs.im - rhs.im).abs() <= tol, "lhs={lhs:?} rhs={rhs:?}");
    }
}

#[test]
#[ignore]
fn test_cubecl_cholesky_batched_f64_matches_cpu() {
    let l0 = vec![2.0, 1.0, 2.0, 0.0, 3.0, -1.0, 0.0, 0.0, 1.5];
    let l1 = vec![1.5, -0.5, 1.0, 0.0, 2.0, 0.75, 0.0, 0.0, 1.25];
    let a0 = matmul_f64(&l0, &transpose_f64(&l0, 3, 3), 3, 3, 3);
    let a1 = matmul_f64(&l1, &transpose_f64(&l1, 3, 3), 3, 3, 3);
    let host = tensor_f64(vec![3, 3, 2], a0.iter().chain(a1.iter()).copied().collect());

    let mut cpu = cpu_backend();
    let expected = cpu.cholesky(&host).unwrap();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &host);
    let gpu_out = gpu.cholesky(&gpu_input).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-9);
}

#[test]
#[ignore]
fn test_cubecl_triangular_solve_c32_reconstructs_rhs() {
    let a = tensor_c32(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
    );
    let b = tensor_c32(
        vec![1, 2],
        vec![Complex32::new(2.0, 1.0), Complex32::new(-1.0, 0.5)],
    );
    let mut gpu = gpu_backend();
    let x_gpu = gpu
        .triangular_solve(
            &upload(&gpu, &a),
            &upload(&gpu, &b),
            false,
            true,
            true,
            true,
        )
        .unwrap();
    let x = download(&gpu, &x_gpu);

    let x_data = x.as_slice::<Complex32>().unwrap().to_vec();
    let a_data = a.as_slice::<Complex32>().unwrap().to_vec();
    let recon = matmul_c32(&x_data, &transpose_c32(&a_data, 2, 2), 1, 2, 2);
    assert_slice_close_c32(&recon, b.as_slice::<Complex32>().unwrap(), 1e-3);
}

#[test]
#[ignore]
fn test_cubecl_lu_f32_reconstructs_pa_equals_lu() {
    let input = tensor_f32(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]);
    let mut gpu = gpu_backend();
    let outputs = gpu.lu(&upload(&gpu, &input)).unwrap();
    let p = download(&gpu, &outputs[0]);
    let l = download(&gpu, &outputs[1]);
    let u = download(&gpu, &outputs[2]);
    let parity = download(&gpu, &outputs[3]);

    let p_data = p.as_slice::<f32>().unwrap().to_vec();
    let l_data = l.as_slice::<f32>().unwrap().to_vec();
    let u_data = u.as_slice::<f32>().unwrap().to_vec();
    let pa = matmul_f32(&p_data, input.as_slice::<f32>().unwrap(), 2, 2, 2);
    let lu = matmul_f32(&l_data, &u_data, 2, 2, 2);
    assert_slice_close_f32(&pa, &lu, 1e-4);
    assert_slice_close_f32(parity.as_slice::<f32>().unwrap(), &[-1.0], 1e-4);
}

#[test]
#[ignore]
fn test_cubecl_lu_f32_handles_rectangular_pivots() {
    let rectangular = tensor_f32(vec![3, 2], vec![0.0, 2.0, 0.0, 1.0, 3.0, 0.0]);
    let mut gpu = gpu_backend();
    let outputs = gpu.lu(&upload(&gpu, &rectangular)).unwrap();
    let p = download(&gpu, &outputs[0]);
    let l = download(&gpu, &outputs[1]);
    let u = download(&gpu, &outputs[2]);
    let parity = download(&gpu, &outputs[3]);
    assert_eq!(p.shape(), &[3, 3]);
    assert_eq!(l.shape(), &[3, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);
    assert_lu_batch_reconstructs_f32(
        rectangular.as_slice::<f32>().unwrap(),
        p.as_slice::<f32>().unwrap(),
        l.as_slice::<f32>().unwrap(),
        u.as_slice::<f32>().unwrap(),
        (3, 2, 2),
        0,
    );
    assert_slice_close_f32(parity.as_slice::<f32>().unwrap(), &[-1.0], 1e-4);
}

#[test]
#[ignore]
fn test_cubecl_lu_f32_handles_batched_pivots() {
    let batched = tensor_f32(vec![2, 2, 2], vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let mut gpu = gpu_backend();
    let outputs = gpu.lu(&upload(&gpu, &batched)).unwrap();
    let p = download(&gpu, &outputs[0]);
    let l = download(&gpu, &outputs[1]);
    let u = download(&gpu, &outputs[2]);
    let parity = download(&gpu, &outputs[3]);
    assert_eq!(p.shape(), &[2, 2, 2]);
    assert_eq!(l.shape(), &[2, 2, 2]);
    assert_eq!(u.shape(), &[2, 2, 2]);
    assert_eq!(parity.shape(), &[2]);
    for batch in 0..2 {
        assert_lu_batch_reconstructs_f32(
            batched.as_slice::<f32>().unwrap(),
            p.as_slice::<f32>().unwrap(),
            l.as_slice::<f32>().unwrap(),
            u.as_slice::<f32>().unwrap(),
            (2, 2, 2),
            batch,
        );
    }
    assert_slice_close_f32(parity.as_slice::<f32>().unwrap(), &[-1.0, 1.0], 1e-4);
}

#[test]
#[ignore]
fn test_cubecl_svd_c32_reconstructs_input() {
    let input = tensor_c32(
        vec![3, 2],
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -0.5),
            Complex32::new(-1.0, 2.0),
            Complex32::new(0.5, -1.0),
            Complex32::new(-0.25, 1.5),
            Complex32::new(3.0, 0.75),
        ],
    );
    let mut gpu = gpu_backend();
    let outputs = gpu.svd(&upload(&gpu, &input)).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let u_data = u.as_slice::<Complex32>().unwrap().to_vec();
    let s_data = s.as_slice::<f32>().unwrap().to_vec();
    let vt_data = vt.as_slice::<Complex32>().unwrap().to_vec();
    let s_complex = s_data
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let recon = matmul_c32(
        &matmul_c32(&u_data, &diag_c32(&s_complex), 3, 2, 2),
        &vt_data,
        3,
        2,
        2,
    );
    assert_slice_close_c32(&recon, input.as_slice::<Complex32>().unwrap(), 3e-3);
}

#[test]
#[ignore]
fn test_cubecl_qr_f32_reconstructs_input() {
    let input = tensor_f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0]);
    let mut gpu = gpu_backend();
    let outputs = gpu.qr(&upload(&gpu, &input)).unwrap();
    let q = download(&gpu, &outputs[0]);
    let r = download(&gpu, &outputs[1]);
    let recon = matmul_f32(
        q.as_slice::<f32>().unwrap(),
        r.as_slice::<f32>().unwrap(),
        3,
        2,
        2,
    );
    assert_slice_close_f32(&recon, input.as_slice::<f32>().unwrap(), 1e-3);
}

#[test]
#[ignore]
fn test_cubecl_eigh_c64_reconstructs_input() {
    let l = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.0),
    ];
    let input = tensor_c64(
        vec![2, 2],
        matmul_c64(&l, &conj_transpose_c64(&l, 2, 2), 2, 2, 2),
    );
    let mut gpu = gpu_backend();
    let outputs = gpu.eigh(&upload(&gpu, &input)).unwrap();
    let values = download(&gpu, &outputs[0]);
    let vectors = download(&gpu, &outputs[1]);
    let values_complex = values
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    let recon = matmul_c64(
        &matmul_c64(
            vectors.as_slice::<Complex64>().unwrap(),
            &diag_c64(&values_complex),
            2,
            2,
            2,
        ),
        &conj_transpose_c64(vectors.as_slice::<Complex64>().unwrap(), 2, 2),
        2,
        2,
        2,
    );
    assert_slice_close_c64(&recon, input.as_slice::<Complex64>().unwrap(), 1e-9);
}

#[test]
#[ignore = "requires CUDA 12+ GPU"]
fn test_gpu_eig_returns_unsupported_error() {
    if !crate::cubecl::gpu_available() {
        eprintln!("skipping test_gpu_eig_returns_unsupported_error — no CUDA device found");
        return;
    }
    let mut backend = super::gpu_backend();
    let cpu = super::tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let gpu = super::upload(&backend, &cpu);

    let err = backend.eig(&gpu).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "eig", .. }
    ));
}

#[test]
#[ignore]
fn test_cubecl_eig_f32_returns_backend_failure() {
    let input = tensor_f32(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]);
    let mut gpu = gpu_backend();
    let err = gpu.eig(&upload(&gpu, &input)).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "eig", .. }
    ));
}

#[test]
#[ignore]
fn test_cubecl_eig_c32_returns_backend_failure() {
    let input = tensor_c32(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 0.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, -0.25),
        ],
    );
    let mut gpu = gpu_backend();
    let err = gpu.eig(&upload(&gpu, &input)).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "eig", .. }
    ));
}

#[test]
#[ignore]
fn test_cubecl_solve_f64_matches_cpu() {
    let a = tensor_f64(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0]);
    let b = tensor_f64(vec![2, 2], vec![5.0, 1.0, -2.0, 4.0]);
    let mut cpu = cpu_backend();
    let expected = cpu.solve(&a, &b).unwrap();
    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);
    let gpu_out = gpu.solve(&gpu_a, &gpu_b).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-9);
}
