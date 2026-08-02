#![cfg(feature = "cuda")]

// Run with: cargo test --features cuda -- --ignored
use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{Error, Tensor, TypedTensor};

fn cpu_backend() -> CpuBackend {
    CpuBackend::new()
}

fn gpu_backend() -> CudaBackend {
    CudaBackend::new(0).unwrap()
}

fn upload(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    upload_tensor(backend.runtime(), tensor).unwrap()
}

fn download(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    download_tensor(backend.runtime(), tensor).unwrap()
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_f64(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_c32(shape: Vec<usize>, data: Vec<Complex32>) -> Tensor {
    Tensor::C32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_c64(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tol: f64) {
    assert_eq!(actual.shape(), expected.shape());
    match (actual, expected) {
        (Tensor::F32(_), Tensor::F32(_)) => {
            let actual = actual.as_slice::<f32>().unwrap();
            let expected = expected.as_slice::<f32>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let diff = (*lhs as f64 - *rhs as f64).abs();
                assert!(
                    diff <= tol,
                    "f32 tensors differ: lhs={lhs:?} rhs={rhs:?} diff={diff}"
                );
            }
        }
        (Tensor::F64(_), Tensor::F64(_)) => {
            let actual = actual.as_slice::<f64>().unwrap();
            let expected = expected.as_slice::<f64>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let diff = (*lhs - *rhs).abs();
                assert!(
                    diff <= tol,
                    "f64 tensors differ: lhs={lhs:?} rhs={rhs:?} diff={diff}"
                );
            }
        }
        (Tensor::C32(_), Tensor::C32(_)) => {
            let actual = actual.as_slice::<Complex32>().unwrap();
            let expected = expected.as_slice::<Complex32>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let real_diff = (lhs.re as f64 - rhs.re as f64).abs();
                let imag_diff = (lhs.im as f64 - rhs.im as f64).abs();
                assert!(
                    real_diff <= tol && imag_diff <= tol,
                    "c32 tensors differ: lhs={lhs:?} rhs={rhs:?}"
                );
            }
        }
        (Tensor::C64(_), Tensor::C64(_)) => {
            let actual = actual.as_slice::<Complex64>().unwrap();
            let expected = expected.as_slice::<Complex64>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let real_diff = (lhs.re - rhs.re).abs();
                let imag_diff = (lhs.im - rhs.im).abs();
                assert!(
                    real_diff <= tol && imag_diff <= tol,
                    "c64 tensors differ: lhs={lhs:?} rhs={rhs:?}"
                );
            }
        }
        _ => panic!(
            "dtype mismatch actual={:?} expected={:?}",
            actual.dtype(),
            expected.dtype()
        ),
    }
}

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

fn assert_relative_error_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    let error_sq = actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>();
    let expected_sq = expected.iter().map(|value| value.powi(2)).sum::<f64>();
    let relative_error = error_sq.sqrt() / expected_sq.sqrt().max(1.0);
    assert!(
        relative_error <= tol,
        "relative error {relative_error:e} exceeds tolerance {tol:e}"
    );
}

fn assert_relative_error_c64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    let error_sq = actual
        .iter()
        .zip(expected)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr())
        .sum::<f64>();
    let expected_sq = expected.iter().map(|value| value.norm_sqr()).sum::<f64>();
    let relative_error = error_sq.sqrt() / expected_sq.sqrt().max(1.0);
    assert!(
        relative_error <= tol,
        "relative error {relative_error:e} exceeds tolerance {tol:e}"
    );
}

fn assert_identity_f64(matrix: &[f64], n: usize, tol: f64) {
    for col in 0..n {
        for row in 0..n {
            let expected = if row == col { 1.0 } else { 0.0 };
            let actual = matrix[col_major_index(n, row, col)];
            assert!(
                (actual - expected).abs() <= tol,
                "isometry residual at ({row}, {col}) is {:e}",
                actual - expected
            );
        }
    }
}

fn assert_identity_c64(matrix: &[Complex64], n: usize, tol: f64) {
    for col in 0..n {
        for row in 0..n {
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let residual = matrix[col_major_index(n, row, col)] - expected;
            assert!(
                residual.norm() <= tol,
                "isometry residual at ({row}, {col}) is {residual:?}"
            );
        }
    }
}

fn assert_singular_values(s: &[f64]) {
    for (index, value) in s.iter().enumerate() {
        assert!(*value >= 0.0, "singular value {index} is negative: {value}");
    }
    for (index, pair) in s.windows(2).enumerate() {
        assert!(
            pair[0] >= pair[1],
            "singular values are not descending at {index}: {:?}",
            pair
        );
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
fn test_cubecl_triangular_solve_batched_f64_matches_cpu() {
    let a = tensor_f64(vec![2, 2, 2], vec![2.0, 0.0, 1.0, 3.0, 4.0, 0.0, -1.0, 2.0]);
    let b = tensor_f64(vec![2, 1, 2], vec![5.0, 9.0, 2.0, 3.0]);
    let mut cpu = cpu_backend();
    let expected = cpu
        .triangular_solve(&a, &b, true, false, false, false)
        .unwrap();
    let mut gpu = gpu_backend();
    let actual = gpu
        .triangular_solve(
            &upload(&gpu, &a),
            &upload(&gpu, &b),
            true,
            false,
            false,
            false,
        )
        .unwrap();
    let actual = download(&gpu, &actual);
    assert_tensor_close(&actual, &expected, 1e-9);
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
fn test_cubecl_svd_values_f64_matches_cpu() {
    let input = tensor_f64(vec![3, 2], vec![1.0, 2.0, -0.5, 0.25, 3.0, -1.0]);
    let mut cpu = cpu_backend();
    let expected = cpu.svd_values(&input).unwrap();
    let mut gpu = gpu_backend();
    let actual = gpu.svd_values(&upload(&gpu, &input)).unwrap();
    let actual = download(&gpu, &actual);
    assert_tensor_close(&actual, &expected, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and exercises legacy gesvd above 1024"]
fn test_cubecl_svd_gesvd_wide_f64_reconstructs_and_matches_values() {
    const M: usize = 8;
    const N: usize = 1025;
    let data = (0..N)
        .flat_map(|col| {
            (0..M).map(move |row| {
                let patterned = ((row * 17 + col * 13 + 3) % 31) as f64 / 31.0 - 0.5;
                patterned + if col == row { 2.0 } else { 0.0 }
            })
        })
        .collect::<Vec<_>>();
    let input = tensor_f64(vec![M, N], data);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = gpu.svd(&gpu_input).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let gpu_values = gpu.svd_values(&gpu_input).unwrap();
    let values = download(&gpu, &gpu_values);

    assert_eq!(u.shape(), &[M, M]);
    assert_eq!(s.shape(), &[M]);
    assert_eq!(vt.shape(), &[M, N]);
    let u_data = u.as_slice::<f64>().unwrap();
    let s_data = s.as_slice::<f64>().unwrap();
    let vt_data = vt.as_slice::<f64>().unwrap();
    assert_singular_values(s_data);
    assert_tensor_close(&values, &s, 1e-10);

    let mut scaled_u = u_data.to_vec();
    for col in 0..M {
        for row in 0..M {
            scaled_u[col_major_index(M, row, col)] *= s_data[col];
        }
    }
    let reconstruction = matmul_f64(&scaled_u, vt_data, M, M, N);
    assert_relative_error_f64(&reconstruction, input.as_slice::<f64>().unwrap(), 1e-10);
    let utu = matmul_f64(&transpose_f64(u_data, M, M), u_data, M, M, M);
    assert_identity_f64(&utu, M, 1e-10);
    let vvt = matmul_f64(vt_data, &transpose_f64(vt_data, M, N), M, N, M);
    assert_identity_f64(&vvt, M, 1e-10);

    let mut cpu = cpu_backend();
    let expected_values = cpu.svd_values(&input).unwrap();
    assert_tensor_close(&values, &expected_values, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and exercises complex legacy gesvd above 1024"]
fn test_cubecl_svd_gesvd_wide_c64_preserves_adjoint_mapping() {
    const M: usize = 8;
    const N: usize = 1025;
    let data = (0..N)
        .flat_map(|col| {
            (0..M).map(move |row| {
                let real = ((row * 11 + col * 7 + 5) % 29) as f64 / 29.0 - 0.5;
                let imag = ((row * 19 + col * 3 + 1) % 23) as f64 / 23.0 - 0.5;
                Complex64::new(real + if col == row { 2.0 } else { 0.0 }, imag)
            })
        })
        .collect::<Vec<_>>();
    let input = tensor_c64(vec![M, N], data);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = gpu.svd(&gpu_input).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let gpu_values = gpu.svd_values(&gpu_input).unwrap();
    let values = download(&gpu, &gpu_values);

    let u_data = u.as_slice::<Complex64>().unwrap();
    let s_data = s.as_slice::<f64>().unwrap();
    let vt_data = vt.as_slice::<Complex64>().unwrap();
    assert_singular_values(s_data);
    assert_tensor_close(&values, &s, 1e-10);

    let mut scaled_u = u_data.to_vec();
    for col in 0..M {
        for row in 0..M {
            scaled_u[col_major_index(M, row, col)] *= s_data[col];
        }
    }
    let reconstruction = matmul_c64(&scaled_u, vt_data, M, M, N);
    assert_relative_error_c64(
        &reconstruction,
        input.as_slice::<Complex64>().unwrap(),
        1e-10,
    );
    let uhu = matmul_c64(&conj_transpose_c64(u_data, M, M), u_data, M, M, M);
    assert_identity_c64(&uhu, M, 1e-10);
    let vvh = matmul_c64(vt_data, &conj_transpose_c64(vt_data, M, N), M, N, M);
    assert_identity_c64(&vvh, M, 1e-10);

    let mut cpu = cpu_backend();
    let expected_values = cpu.svd_values(&input).unwrap();
    assert_tensor_close(&values, &expected_values, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and exercises legacy gesvd above 1024"]
fn test_cubecl_svd_gesvd_tall_f64_retains_direct_route() {
    const M: usize = 1025;
    const N: usize = 8;
    let data = (0..N)
        .flat_map(|col| {
            (0..M).map(move |row| {
                let patterned = ((row * 13 + col * 17 + 3) % 31) as f64 / 31.0 - 0.5;
                patterned + if row == col { 2.0 } else { 0.0 }
            })
        })
        .collect::<Vec<_>>();
    let input = tensor_f64(vec![M, N], data);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = gpu.svd(&gpu_input).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let u_data = u.as_slice::<f64>().unwrap();
    let s_data = s.as_slice::<f64>().unwrap();
    let vt_data = vt.as_slice::<f64>().unwrap();
    assert_singular_values(s_data);

    let mut scaled_u = u_data.to_vec();
    for col in 0..N {
        for row in 0..M {
            scaled_u[col_major_index(M, row, col)] *= s_data[col];
        }
    }
    let reconstruction = matmul_f64(&scaled_u, vt_data, M, N, N);
    assert_relative_error_f64(&reconstruction, input.as_slice::<f64>().unwrap(), 1e-10);

    let mut cpu = cpu_backend();
    let expected_values = cpu.svd_values(&input).unwrap();
    assert_tensor_close(&s, &expected_values, 1e-9);
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
#[ignore]
fn test_cubecl_eigh_values_c64_matches_cpu() {
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
    let mut cpu = cpu_backend();
    let expected = cpu.eigh_values(&input).unwrap();
    let mut gpu = gpu_backend();
    let actual = gpu.eigh_values(&upload(&gpu, &input)).unwrap();
    let actual = download(&gpu, &actual);
    assert_tensor_close(&actual, &expected, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_eig_returns_unsupported_error() {
    if !gpu_available() {
        eprintln!("skipping test_gpu_eig_returns_unsupported_error — no CUDA device found");
        return;
    }
    let mut backend = gpu_backend();
    let cpu = tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let gpu = upload(&backend, &cpu);

    let err = backend.eig(&gpu).unwrap_err();
    assert!(matches!(err, Error::Unsupported { op: "eig", .. }));
}

#[test]
#[ignore]
fn test_cubecl_eig_f32_returns_unsupported_error() {
    let input = tensor_f32(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]);
    let mut gpu = gpu_backend();
    let err = gpu.eig(&upload(&gpu, &input)).unwrap_err();
    assert!(matches!(err, Error::Unsupported { op: "eig", .. }));
}

#[test]
#[ignore]
fn test_cubecl_eig_c32_returns_unsupported_error() {
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
    assert!(matches!(err, Error::Unsupported { op: "eig", .. }));
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
