#![cfg(feature = "cuda")]

// Run with: cargo test --features cuda -- --ignored
use num_complex::{Complex32, Complex64};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
use tenferro_gpu::{
    cuda::download_tensor, cuda::gpu_available, cuda::upload_tensor, cuda::with_cuda_exec_session,
    cuda::CudaBackend, cuda::CudaDeviceId, cuda::CudaExecSession,
};
use tenferro_linalg::{
    EighDriver, EighOptions, HouseholderQr, LinalgBackend, QrGauge, QrOptions,
    RankRevealingQrOptions, SvdDriver, SvdOptions, TensorLinalgExt,
};
use tenferro_tensor::{BackendSessionHost, DType, Error, Tensor, TensorRead, TypedTensor};

fn cpu_backend() -> CpuBackend {
    CpuBackend::new()
}

fn gpu_backend() -> CudaBackend {
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap()
}

fn with_cpu_linalg_session<R>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, f).expect("CPU backend session should be available")
    })
}

fn with_cuda_linalg_session<R>(
    backend: &mut CudaBackend,
    f: impl for<'a> FnOnce(&'a mut CudaExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cuda_exec_session(session, f).expect("CUDA backend session should be available")
    })
}

fn upload(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    upload_tensor(backend.runtime(), tensor).unwrap()
}

fn download(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    download_tensor(backend.runtime(), tensor).unwrap()
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_f64(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_c32(shape: Vec<usize>, data: Vec<Complex32>) -> Tensor {
    Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    )
}

fn tensor_c64(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    )
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tol: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype(), "dtypes must match");
    match actual.dtype() {
        DType::F32 => {
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
        DType::F64 => {
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
        DType::C32 => {
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
        DType::C64 => {
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

fn assert_slice_close_f64(actual: &[f64], expected: &[f64], tol: f64) {
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
    let expected = with_cpu_linalg_session(&mut cpu, |session| session.cholesky(&host)).unwrap();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &host);
    let gpu_out =
        with_cuda_linalg_session(&mut gpu, |session| session.cholesky(&gpu_input)).unwrap();
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
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);
    let x_gpu = with_cuda_linalg_session(&mut gpu, |session| {
        session.triangular_solve(&gpu_a, &gpu_b, false, true, true, true)
    })
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
    let expected = with_cpu_linalg_session(&mut cpu, |session| {
        session.triangular_solve(&a, &b, true, false, false, false)
    })
    .unwrap();
    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);
    let actual = with_cuda_linalg_session(&mut gpu, |session| {
        session.triangular_solve(&gpu_a, &gpu_b, true, false, false, false)
    })
    .unwrap();
    let actual = download(&gpu, &actual);
    assert_tensor_close(&actual, &expected, 1e-9);
}

#[test]
#[ignore]
fn test_cubecl_lu_f32_reconstructs_pa_equals_lu() {
    let input = tensor_f32(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.lu(&gpu_input)).unwrap();
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
    let gpu_input = upload(&gpu, &rectangular);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.lu(&gpu_input)).unwrap();
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
    let gpu_input = upload(&gpu, &batched);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.lu(&gpu_input)).unwrap();
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
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.svd(&gpu_input)).unwrap();
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
    let expected = with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let actual =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_values(&gpu_input)).unwrap();
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
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.svd(&gpu_input)).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let gpu_values =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_values(&gpu_input)).unwrap();
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
    let expected_values =
        with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap();
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
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.svd(&gpu_input)).unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    let gpu_values =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_values(&gpu_input)).unwrap();
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
    let expected_values =
        with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap();
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
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.svd(&gpu_input)).unwrap();
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
    let expected_values =
        with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap();
    assert_tensor_close(&s, &expected_values, 1e-9);
}

/// Patterned `m x n` f64 matrix with a dominant diagonal so every driver
/// sees a well-separated spectrum.
fn patterned_f64_matrix(m: usize, n: usize) -> Tensor {
    let data = (0..n)
        .flat_map(|col| {
            (0..m).map(move |row| {
                let patterned = ((row * 13 + col * 17 + 3) % 31) as f64 / 31.0 - 0.5;
                patterned + if row == col { 2.0 } else { 0.0 }
            })
        })
        .collect::<Vec<_>>();
    tensor_f64(vec![m, n], data)
}

/// Forced-driver SVD through the owned, borrowed, and values-only entry
/// points: factors reconstruct the input and match CPU singular values.
fn check_forced_svd_driver(m: usize, n: usize, driver: SvdDriver) {
    let input = patterned_f64_matrix(m, n);
    let k = m.min(n);
    let options = SvdOptions::default().driver(driver);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_with_options(&gpu_input, options)
    })
    .unwrap();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    assert_eq!(u.shape(), &[m, k]);
    assert_eq!(s.shape(), &[k]);
    assert_eq!(vt.shape(), &[k, n]);
    let u_data = u.as_slice::<f64>().unwrap();
    let s_data = s.as_slice::<f64>().unwrap();
    let vt_data = vt.as_slice::<f64>().unwrap();
    assert_singular_values(s_data);

    let mut scaled_u = u_data.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[col_major_index(m, row, col)] *= s_data[col];
        }
    }
    let reconstruction = matmul_f64(&scaled_u, vt_data, m, k, n);
    assert_relative_error_f64(&reconstruction, input.as_slice::<f64>().unwrap(), 1e-10);

    let read_outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_with_options_read(TensorRead::from_tensor(&gpu_input), options)
    })
    .unwrap();
    let read_s = download(&gpu, &read_outputs[1]);
    assert_tensor_close(&read_s, &s, 1e-10);

    let gpu_values = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_values_with_driver(&gpu_input, driver)
    })
    .unwrap();
    let values = download(&gpu, &gpu_values);
    assert_tensor_close(&values, &s, 1e-10);

    let gpu_read_values = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_values_with_driver_read(TensorRead::from_tensor(&gpu_input), driver)
    })
    .unwrap();
    let read_values = download(&gpu, &gpu_read_values);
    assert_tensor_close(&read_values, &s, 1e-10);

    let mut cpu = cpu_backend();
    let expected_values =
        with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap();
    assert_tensor_close(&s, &expected_values, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and forces legacy gesvd below the 1024 Auto threshold"]
fn test_cubecl_svd_forced_gesvd_below_threshold_reconstructs() {
    check_forced_svd_driver(64, 64, SvdDriver::Gesvd);
    check_forced_svd_driver(8, 64, SvdDriver::Gesvd);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and forces gesvdj above the 1024 Auto threshold"]
fn test_cubecl_svd_forced_gesvdj_above_threshold_reconstructs() {
    check_forced_svd_driver(1025, 8, SvdDriver::Gesvdj);
    check_forced_svd_driver(8, 1025, SvdDriver::Gesvdj);
}

#[test]
#[ignore]
fn test_cubecl_svd_auto_driver_matches_default_svd() {
    let input = patterned_f64_matrix(48, 32);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let default_outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.svd(&gpu_input)).unwrap();
    let auto_outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_with_options(&gpu_input, SvdOptions::default().driver(SvdDriver::Auto))
    })
    .unwrap();
    for (default, auto) in default_outputs.iter().zip(&auto_outputs) {
        assert_tensor_close(&download(&gpu, auto), &download(&gpu, default), 0.0);
    }
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_f64_reconstructs_input() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let input = tensor_f64(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 0.5, 3.0]);
    let mut gpu = gpu_backend();
    let device_input = upload(&gpu, &input);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, r) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = device_input.householder_qr(session).unwrap();
        (
            state.q_columns(0..2, options, session).unwrap(),
            state.r(options, session).unwrap(),
        )
    });
    let q = download(&gpu, &q);
    let r = download(&gpu, &r);
    let reconstructed = matmul_f64(
        q.as_slice::<f64>().unwrap(),
        r.as_slice::<f64>().unwrap(),
        4,
        2,
        2,
    );
    assert_tensor_close(&tensor_f64(vec![4, 2], reconstructed), &input, 1.0e-10);
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_append_and_from_factors_f64() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let a = tensor_f64(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 0.5, 3.0]);
    let b = tensor_f64(vec![4, 2], vec![0.5, 1.0, -2.0, 1.5, 2.0, 0.0, 1.0, -1.0]);
    let c = tensor_f64(vec![4, 1], vec![1.0, -0.5, 2.0, 0.25]);
    let q_factor = tensor_f64(vec![4, 2], vec![1.0, 0.0, 1.0, 2.0, 0.5, 1.0, -1.0, 0.0]);
    let r_factor = tensor_f64(vec![2, 3], vec![2.0, 0.0, -1.0, 3.0, 0.5, 2.0]);
    let invalid_r = tensor_f64(vec![2, 2], vec![2.0, 1.0, 0.5, 3.0]);
    let mut gpu = gpu_backend();
    let a_gpu = upload(&gpu, &a);
    let b_gpu = upload(&gpu, &b);
    let c_gpu = upload(&gpu, &c);
    let q_gpu = upload(&gpu, &q_factor);
    let r_gpu = upload(&gpu, &r_factor);
    let invalid_r_gpu = upload(&gpu, &invalid_r);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (append_q, append_r, factor_q, factor_r) = with_cuda_linalg_session(&mut gpu, |session| {
        let appended = a_gpu
            .householder_qr(session)
            .unwrap()
            .append_columns(&b_gpu, session)
            .unwrap()
            .append_columns(&c_gpu, session)
            .unwrap();
        let invalid =
            HouseholderQr::<Tensor>::from_factors(&q_gpu, &invalid_r_gpu, session).unwrap_err();
        assert!(matches!(invalid, Error::Validation { .. }));
        let imported = HouseholderQr::<Tensor>::from_factors(&q_gpu, &r_gpu, session).unwrap();
        (
            appended.q_columns(0..4, options, session).unwrap(),
            appended.r(options, session).unwrap(),
            imported.q_columns(0..3, options, session).unwrap(),
            imported.r(options, session).unwrap(),
        )
    });
    let append_q = download(&gpu, &append_q);
    let append_r = download(&gpu, &append_r);
    let factor_q = download(&gpu, &factor_q);
    let factor_r = download(&gpu, &factor_r);
    let append_actual = matmul_f64(
        append_q.as_slice::<f64>().unwrap(),
        append_r.as_slice::<f64>().unwrap(),
        4,
        4,
        5,
    );
    let mut combined = a.as_slice::<f64>().unwrap().to_vec();
    combined.extend_from_slice(b.as_slice::<f64>().unwrap());
    combined.extend_from_slice(c.as_slice::<f64>().unwrap());
    assert_tensor_close(
        &tensor_f64(vec![4, 5], append_actual),
        &tensor_f64(vec![4, 5], combined),
        1.0e-9,
    );
    let factor_actual = matmul_f64(
        factor_q.as_slice::<f64>().unwrap(),
        factor_r.as_slice::<f64>().unwrap(),
        4,
        3,
        3,
    );
    let factor_expected = matmul_f64(
        q_factor.as_slice::<f64>().unwrap(),
        r_factor.as_slice::<f64>().unwrap(),
        4,
        2,
        3,
    );
    assert_tensor_close(
        &tensor_f64(vec![4, 3], factor_actual),
        &tensor_f64(vec![4, 3], factor_expected),
        1.0e-9,
    );
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_complex64_append_reconstructs() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let c = |re, im| Complex64::new(re, im);
    let a = tensor_c64(
        vec![3, 2],
        vec![
            c(1.0, 0.2),
            c(2.0, -0.1),
            c(0.5, 0.3),
            c(-1.0, 0.4),
            c(0.2, 0.1),
            c(2.0, -0.5),
        ],
    );
    let b = tensor_c64(vec![3, 1], vec![c(0.5, -0.2), c(1.5, 0.4), c(-0.7, 0.3)]);
    let mut gpu = gpu_backend();
    let a_gpu = upload(&gpu, &a);
    let b_gpu = upload(&gpu, &b);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, selected, r) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = a_gpu
            .householder_qr(session)
            .unwrap()
            .append_columns(&b_gpu, session)
            .unwrap();
        (
            state.q_columns(0..3, options, session).unwrap(),
            state.q_columns(1..2, options, session).unwrap(),
            state.r(options, session).unwrap(),
        )
    });
    let q = download(&gpu, &q);
    let selected = download(&gpu, &selected);
    let r = download(&gpu, &r);
    assert_tensor_close(
        &tensor_c64(
            vec![3, 1],
            q.as_slice::<Complex64>().unwrap()[3..6].to_vec(),
        ),
        &selected,
        1.0e-12,
    );
    let actual = matmul_c64(
        q.as_slice::<Complex64>().unwrap(),
        r.as_slice::<Complex64>().unwrap(),
        3,
        3,
        3,
    );
    let mut expected = a.as_slice::<Complex64>().unwrap().to_vec();
    expected.extend_from_slice(b.as_slice::<Complex64>().unwrap());
    assert_tensor_close(
        &tensor_c64(vec![3, 3], actual),
        &tensor_c64(vec![3, 3], expected),
        1.0e-9,
    );
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_f32_and_c32_reconstruct() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let a = tensor_f32(vec![3, 2], vec![1.0, 2.0, 0.5, -1.0, 0.2, 2.0]);
    let b = tensor_f32(vec![3, 2], vec![0.5, 1.0, -2.0, 2.0, 0.0, 1.0]);
    let c = |re, im| Complex32::new(re, im);
    let complex = tensor_c32(
        vec![3, 2],
        vec![
            c(1.0, 0.2),
            c(2.0, -0.1),
            c(0.5, 0.3),
            c(-1.0, 0.4),
            c(0.2, 0.1),
            c(2.0, -0.5),
        ],
    );
    let complex_r = tensor_c32(
        vec![2, 2],
        vec![c(1.5, 0.2), c(0.0, 0.0), c(-0.5, 0.3), c(2.0, -0.1)],
    );
    let mut gpu = gpu_backend();
    let a_gpu = upload(&gpu, &a);
    let b_gpu = upload(&gpu, &b);
    let complex_gpu = upload(&gpu, &complex);
    let complex_r_gpu = upload(&gpu, &complex_r);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, r, cq, cr, imported_q, imported_r) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = a_gpu
            .householder_qr(session)
            .unwrap()
            .append_columns(&b_gpu, session)
            .unwrap();
        let complex_state = complex_gpu.householder_qr(session).unwrap();
        let imported =
            HouseholderQr::<Tensor>::from_factors(&complex_gpu, &complex_r_gpu, session).unwrap();
        (
            state.q_columns(0..3, options, session).unwrap(),
            state.r(options, session).unwrap(),
            complex_state.q_columns(0..2, options, session).unwrap(),
            complex_state.r(options, session).unwrap(),
            imported.q_columns(0..2, options, session).unwrap(),
            imported.r(options, session).unwrap(),
        )
    });
    let q = download(&gpu, &q);
    let r = download(&gpu, &r);
    let cq = download(&gpu, &cq);
    let cr = download(&gpu, &cr);
    let imported_q = download(&gpu, &imported_q);
    let imported_r = download(&gpu, &imported_r);
    let actual = matmul_f32(
        q.as_slice::<f32>().unwrap(),
        r.as_slice::<f32>().unwrap(),
        3,
        3,
        4,
    );
    let mut expected = a.as_slice::<f32>().unwrap().to_vec();
    expected.extend_from_slice(b.as_slice::<f32>().unwrap());
    assert_tensor_close(
        &tensor_f32(vec![3, 4], actual),
        &tensor_f32(vec![3, 4], expected),
        2.0e-5,
    );
    let complex_actual = matmul_c32(
        cq.as_slice::<Complex32>().unwrap(),
        cr.as_slice::<Complex32>().unwrap(),
        3,
        2,
        2,
    );
    assert_tensor_close(&tensor_c32(vec![3, 2], complex_actual), &complex, 2.0e-5);
    let imported_actual = matmul_c32(
        imported_q.as_slice::<Complex32>().unwrap(),
        imported_r.as_slice::<Complex32>().unwrap(),
        3,
        2,
        2,
    );
    let imported_expected = matmul_c32(
        complex.as_slice::<Complex32>().unwrap(),
        complex_r.as_slice::<Complex32>().unwrap(),
        3,
        2,
        2,
    );
    assert_tensor_close(
        &tensor_c32(vec![3, 2], imported_actual),
        &tensor_c32(vec![3, 2], imported_expected),
        3.0e-5,
    );
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_rank_deficient_zero_append_and_placement() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let input = tensor_f64(vec![3, 2], vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    let empty = tensor_f64(vec![3, 0], vec![]);
    let mut gpu = gpu_backend();
    let input_gpu = upload(&gpu, &input);
    let empty_gpu = upload(&gpu, &empty);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, selected, r) = with_cuda_linalg_session(&mut gpu, |session| {
        let placement_error = input.householder_qr(session).unwrap_err();
        assert!(matches!(placement_error, Error::RuntimeState { .. }));
        let state = input_gpu
            .householder_qr(session)
            .unwrap()
            .append_columns(&empty_gpu, session)
            .unwrap();
        (
            state.q_columns(0..2, options, session).unwrap(),
            state.q_columns(1..2, options, session).unwrap(),
            state.r(options, session).unwrap(),
        )
    });
    let q = download(&gpu, &q);
    let selected = download(&gpu, &selected);
    let r = download(&gpu, &r);
    assert_eq!(selected.shape(), &[3, 1]);
    assert_tensor_close(
        &tensor_f64(vec![3, 1], q.as_slice::<f64>().unwrap()[3..6].to_vec()),
        &selected,
        1.0e-12,
    );
    let actual = matmul_f64(
        q.as_slice::<f64>().unwrap(),
        r.as_slice::<f64>().unwrap(),
        3,
        2,
        2,
    );
    assert_tensor_close(&tensor_f64(vec![3, 2], actual), &input, 1.0e-9);
}

#[test]
#[ignore]
fn test_cuda_qr_positive_diagonal_owned_and_read_paths_stay_on_device() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let c = |re, im| Complex64::new(re, im);
    let input = tensor_c64(
        vec![2, 2],
        vec![c(-1.0, 0.5), c(2.0, 0.1), c(0.3, -0.7), c(-2.0, 0.4)],
    );
    let mut gpu = gpu_backend();
    let input_gpu = upload(&gpu, &input);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (owned, read) = with_cuda_linalg_session(&mut gpu, |session| {
        (
            session.qr_with_options(&input_gpu, options).unwrap(),
            session
                .qr_with_options_read(TensorRead::from_tensor(&input_gpu), options)
                .unwrap(),
        )
    });
    for outputs in [owned, read] {
        let r = download(&gpu, &outputs[1]);
        let values = r.as_slice::<Complex64>().unwrap();
        for index in 0..2 {
            let diagonal = values[index + index * 2];
            assert!(diagonal.re >= 0.0, "negative QR diagonal: {diagonal}");
            assert!(
                diagonal.im.abs() < 1.0e-12,
                "complex QR diagonal: {diagonal}"
            );
        }
    }

    let batched = tensor_c64(
        vec![2, 2, 2],
        vec![
            c(-1.0, 0.5),
            c(2.0, 0.1),
            c(0.3, -0.7),
            c(-2.0, 0.4),
            c(0.2, -1.5),
            c(-0.4, 0.8),
            c(2.0, 0.3),
            c(-0.5, -2.0),
        ],
    );
    let batched = upload(&gpu, &batched);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.qr_with_options(&batched, options).unwrap()
    });
    let r = download(&gpu, &outputs[1]);
    let values = r.as_slice::<Complex64>().unwrap();
    for batch in 0..2 {
        for index in 0..2 {
            let diagonal = values[batch * 4 + index + index * 2];
            assert!(
                diagonal.re >= 0.0,
                "negative batched QR diagonal: {diagonal}"
            );
            assert!(
                diagonal.im.abs() < 1.0e-12,
                "complex batched QR diagonal: {diagonal}"
            );
        }
    }

    let real_batched = tensor_f64(
        vec![2, 2, 2, 2],
        vec![
            -1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, -3.0, -2.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, -0.25,
        ],
    );
    let real_batched = upload(&gpu, &real_batched);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.qr_with_options(&real_batched, options).unwrap()
    });
    let r = download(&gpu, &outputs[1]);
    let values = r.as_slice::<f64>().unwrap();
    for batch in 0..4 {
        for index in 0..2 {
            assert!(values[batch * 4 + index + index * 2] >= 0.0);
        }
    }
}

#[test]
#[ignore]
fn test_cuda_compact_householder_qr_wide_empty_range_and_rank_deficient_import() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let wide = tensor_f64(vec![2, 4], vec![1.0, 2.0, -1.0, 0.5, 2.0, -0.5, 0.25, 1.5]);
    let rank_q = tensor_f64(vec![3, 2], vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    let rank_r = tensor_f64(vec![2, 2], vec![1.0, 0.0, -0.5, 2.0]);
    let mut gpu = gpu_backend();
    let wide_gpu = upload(&gpu, &wide);
    let rank_q_gpu = upload(&gpu, &rank_q);
    let rank_r_gpu = upload(&gpu, &rank_r);
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, empty_q, r, imported_q, imported_r) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = wide_gpu.householder_qr(session).unwrap();
        let imported =
            HouseholderQr::<Tensor>::from_factors(&rank_q_gpu, &rank_r_gpu, session).unwrap();
        (
            state.q_columns(0..2, options, session).unwrap(),
            state.q_columns(0..0, options, session).unwrap(),
            state.r(options, session).unwrap(),
            imported.q_columns(0..2, options, session).unwrap(),
            imported.r(options, session).unwrap(),
        )
    });
    let q = download(&gpu, &q);
    let empty_q = download(&gpu, &empty_q);
    let r = download(&gpu, &r);
    let imported_q = download(&gpu, &imported_q);
    let imported_r = download(&gpu, &imported_r);
    assert_eq!(empty_q.shape(), &[2, 0]);
    let q_data = q.as_slice::<f64>().unwrap();
    for lhs in 0..2 {
        for rhs in 0..2 {
            let dot = (0..2)
                .map(|row| q_data[row + lhs * 2] * q_data[row + rhs * 2])
                .sum::<f64>();
            let expected = if lhs == rhs { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1.0e-10);
        }
    }
    let actual = matmul_f64(q_data, r.as_slice::<f64>().unwrap(), 2, 2, 4);
    assert_tensor_close(&tensor_f64(vec![2, 4], actual), &wide, 1.0e-9);
    let imported_actual = matmul_f64(
        imported_q.as_slice::<f64>().unwrap(),
        imported_r.as_slice::<f64>().unwrap(),
        3,
        2,
        2,
    );
    let imported_expected = matmul_f64(
        rank_q.as_slice::<f64>().unwrap(),
        rank_r.as_slice::<f64>().unwrap(),
        3,
        2,
        2,
    );
    assert_tensor_close(
        &tensor_f64(vec![3, 2], imported_actual),
        &tensor_f64(vec![3, 2], imported_expected),
        1.0e-9,
    );
}

#[test]
#[ignore]
fn test_cubecl_qr_f32_reconstructs_input() {
    let input = tensor_f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0]);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.qr(&gpu_input)).unwrap();
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

/// Symmetric `n x n` block with a well-separated spectrum, repeated over
/// `batch` matrices so the batched loop is exercised too.
fn patterned_symmetric_f64(n: usize, batch: usize) -> Tensor {
    let mut data = Vec::with_capacity(n * n * batch.max(1));
    for b in 0..batch.max(1) {
        for col in 0..n {
            for row in 0..n {
                let (lo, hi) = if row <= col { (row, col) } else { (col, row) };
                let patterned = ((lo * 13 + hi * 17 + 3) % 31) as f64 / 31.0 - 0.5;
                let diagonal = if row == col {
                    2.0 * (row + 1) as f64 + b as f64
                } else {
                    0.0
                };
                data.push(patterned + diagonal);
            }
        }
    }
    let shape = if batch > 1 {
        vec![n, n, batch]
    } else {
        vec![n, n]
    };
    tensor_f64(shape, data)
}

/// Forced-driver eigh through the owned, borrowed, and values-only entry
/// points: `V diag(w) V^T` reconstructs the input and the eigenvalues match
/// the CPU provider.
fn check_forced_eigh_driver(n: usize, batch: usize, driver: EighDriver) {
    let input = patterned_symmetric_f64(n, batch);
    let options = EighOptions::default().driver(driver);
    let batch_total = batch.max(1);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.eigh_with_options(&gpu_input, options)
    })
    .unwrap();
    let values = download(&gpu, &outputs[0]);
    let vectors = download(&gpu, &outputs[1]);
    let expected_values_shape: Vec<usize> = if batch > 1 { vec![n, batch] } else { vec![n] };
    assert_eq!(values.shape(), expected_values_shape.as_slice());
    assert_eq!(vectors.shape(), input.shape());

    let values_data = values.as_slice::<f64>().unwrap();
    let vectors_data = vectors.as_slice::<f64>().unwrap();
    let input_data = input.as_slice::<f64>().unwrap();
    for b in 0..batch_total {
        let v = &vectors_data[b * n * n..(b + 1) * n * n];
        let w = &values_data[b * n..(b + 1) * n];
        // cuSOLVER returns eigenvalues in ascending order for both routines.
        for pair in w.windows(2) {
            assert!(
                pair[0] <= pair[1] + 1e-12,
                "eigenvalues must be ascending, got {pair:?}"
            );
        }
        let mut scaled = v.to_vec();
        for col in 0..n {
            for row in 0..n {
                scaled[col_major_index(n, row, col)] *= w[col];
            }
        }
        let reconstruction = matmul_f64(&scaled, &transpose_f64(v, n, n), n, n, n);
        assert_relative_error_f64(
            &reconstruction,
            &input_data[b * n * n..(b + 1) * n * n],
            1e-9,
        );
    }

    let read_outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.eigh_with_options_read(TensorRead::from_tensor(&gpu_input), options)
    })
    .unwrap();
    assert_tensor_close(&download(&gpu, &read_outputs[0]), &values, 1e-10);

    let gpu_values = with_cuda_linalg_session(&mut gpu, |session| {
        session.eigh_values_with_driver(&gpu_input, driver)
    })
    .unwrap();
    assert_tensor_close(&download(&gpu, &gpu_values), &values, 1e-10);

    let gpu_read_values = with_cuda_linalg_session(&mut gpu, |session| {
        session.eigh_values_with_driver_read(TensorRead::from_tensor(&gpu_input), driver)
    })
    .unwrap();
    assert_tensor_close(&download(&gpu, &gpu_read_values), &values, 1e-10);

    let mut cpu = cpu_backend();
    let expected_values =
        with_cpu_linalg_session(&mut cpu, |session| session.eigh_values(&input)).unwrap();
    assert_tensor_close(&values, &expected_values, 1e-8);
}

#[test]
#[ignore = "requires a CUDA GPU and forces the Jacobi eigensolver that Auto never picks"]
fn test_cubecl_eigh_forced_syevj_matches_reference() {
    check_forced_eigh_driver(8, 1, EighDriver::Syevj);
    check_forced_eigh_driver(64, 1, EighDriver::Syevj);
    check_forced_eigh_driver(6, 4, EighDriver::Syevj);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn test_cubecl_eigh_forced_syevd_matches_reference() {
    check_forced_eigh_driver(8, 1, EighDriver::Syevd);
    check_forced_eigh_driver(6, 4, EighDriver::Syevd);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn test_cubecl_eigh_auto_driver_matches_default_eigh() {
    let input = patterned_symmetric_f64(32, 1);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let default_outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.eigh(&gpu_input)).unwrap();
    let auto_outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.eigh_with_options(&gpu_input, EighOptions::default().driver(EighDriver::Auto))
    })
    .unwrap();
    for (default, auto) in default_outputs.iter().zip(&auto_outputs) {
        assert_tensor_close(&download(&gpu, auto), &download(&gpu, default), 0.0);
    }
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
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| session.eigh(&gpu_input)).unwrap();
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
    let expected =
        with_cpu_linalg_session(&mut cpu, |session| session.eigh_values(&input)).unwrap();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let actual =
        with_cuda_linalg_session(&mut gpu, |session| session.eigh_values(&gpu_input)).unwrap();
    let actual = download(&gpu, &actual);
    assert_tensor_close(&actual, &expected, 1e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_gpu_eig_returns_unsupported_error() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = gpu_backend();
    let cpu = tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let gpu = upload(&backend, &cpu);

    let err = with_cuda_linalg_session(&mut backend, |session| session.eig(&gpu)).unwrap_err();
    assert!(matches!(err, Error::Unsupported { op: "eig", .. }));
}

#[test]
#[ignore = "requires an A100-class CUDA GPU"]
fn test_cuda_rank_revealing_qr_f64_reconstructs_interspersed_rank_deficiency() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let input = tensor_f64(
        vec![4, 5],
        vec![
            1.0, 0.0, 0.0, 0.0, // c0
            0.0, 1.0, 0.0, 0.0, // c1
            1.0, 0.0, 0.0, 0.0, // c2 duplicates c0
            0.0, 0.0, 1.0, 0.0, // c3
            0.0, 0.0, 0.0, 0.0, // c4
        ],
    );
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.rank_revealing_qr(&gpu_input, RankRevealingQrOptions::default().rtol(1.0e-12))
    })
    .unwrap();
    assert_eq!(outputs.len(), 4);
    for output in &outputs {
        assert!(match output.dtype() {
            tenferro_tensor::DType::F64 => output.as_slice::<f64>().is_err(),
            tenferro_tensor::DType::I64 => output.as_slice::<i64>().is_err(),
            dtype => panic!("unexpected RRQR dtype {dtype:?}"),
        });
    }
    let q = download(&gpu, &outputs[0]);
    let r = download(&gpu, &outputs[1]);
    let permutation = download(&gpu, &outputs[2]);
    let rank = download(&gpu, &outputs[3]);
    assert_eq!(permutation.as_slice::<i64>().unwrap(), &[0, 1, 3, 2, 4]);
    assert_eq!(rank.as_slice::<i64>().unwrap(), &[3]);
    let reconstructed = matmul_f64(
        q.as_slice::<f64>().unwrap(),
        r.as_slice::<f64>().unwrap(),
        4,
        4,
        5,
    );
    let p = permutation.as_slice::<i64>().unwrap();
    let expected = p
        .iter()
        .flat_map(|&column| {
            let start = column as usize * 4;
            input.as_slice::<f64>().unwrap()[start..start + 4]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_slice_close_f64(&reconstructed, &expected, 1.0e-10);
    let qtq = matmul_f64(
        &transpose_f64(q.as_slice::<f64>().unwrap(), 4, 4),
        q.as_slice::<f64>().unwrap(),
        4,
        4,
        4,
    );
    let identity = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    assert_slice_close_f64(&qtq, &identity, 1.0e-10);
}

#[test]
#[ignore = "requires an A100-class CUDA GPU"]
fn test_cuda_rank_revealing_qr_complex_dtypes_reconstruct() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = gpu_backend();
    let input_c32 = tensor_c32(
        vec![3, 3],
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ],
    );
    let gpu_input = upload(&gpu, &input_c32);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.rank_revealing_qr(
            &gpu_input,
            RankRevealingQrOptions::default()
                .gauge(tenferro_linalg::QrGauge::PositiveDiagonal)
                .rtol(1.0e-5),
        )
    })
    .unwrap();
    let q = download(&gpu, &outputs[0]);
    let r = download(&gpu, &outputs[1]);
    let permutation = download(&gpu, &outputs[2]);
    let rank = download(&gpu, &outputs[3]);
    assert_eq!(rank.as_slice::<i64>().unwrap(), &[2]);
    let p = permutation.as_slice::<i64>().unwrap();
    let expected = p
        .iter()
        .flat_map(|&column| {
            let start = column as usize * 3;
            input_c32.as_slice::<Complex32>().unwrap()[start..start + 3]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let reconstructed = matmul_c32(
        q.as_slice::<Complex32>().unwrap(),
        r.as_slice::<Complex32>().unwrap(),
        3,
        3,
        3,
    );
    assert_slice_close_c32(&reconstructed, &expected, 2.0e-5);

    let input_c64 = tensor_c64(
        vec![3, 3],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let gpu_input = upload(&gpu, &input_c64);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.rank_revealing_qr(
            &gpu_input,
            RankRevealingQrOptions::default()
                .gauge(tenferro_linalg::QrGauge::PositiveDiagonal)
                .rtol(1.0e-12),
        )
    })
    .unwrap();
    let q = download(&gpu, &outputs[0]);
    let r = download(&gpu, &outputs[1]);
    let permutation = download(&gpu, &outputs[2]);
    let rank = download(&gpu, &outputs[3]);
    assert_eq!(rank.as_slice::<i64>().unwrap(), &[2]);
    let p = permutation.as_slice::<i64>().unwrap();
    let expected = p
        .iter()
        .flat_map(|&column| {
            let start = column as usize * 3;
            input_c64.as_slice::<Complex64>().unwrap()[start..start + 3]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let reconstructed = matmul_c64(
        q.as_slice::<Complex64>().unwrap(),
        r.as_slice::<Complex64>().unwrap(),
        3,
        3,
        3,
    );
    assert_slice_close_c64(&reconstructed, &expected, 1.0e-11);
    for diagonal in 0..3 {
        let value = r.as_slice::<Complex64>().unwrap()[diagonal + diagonal * 3];
        assert!(value.im.abs() <= 1.0e-12 && value.re >= 0.0);
    }
}

#[test]
#[ignore = "requires an A100-class CUDA GPU"]
fn test_cuda_rank_revealing_qr_f32_batched_and_scaled_ranks() {
    assert!(gpu_available(), "CUDA test requires an available device");
    // First 3x2 batch is full rank at large scale; second has duplicate columns.
    let input = tensor_f32(
        vec![3, 2, 2],
        vec![
            1.0e8, 0.0, 0.0, 0.0, 2.0e8, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
        ],
    );
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.rank_revealing_qr(&gpu_input, RankRevealingQrOptions::default().rtol(1.0e-5))
    })
    .unwrap();
    let permutation = download(&gpu, &outputs[2]);
    let rank = download(&gpu, &outputs[3]);
    assert_eq!(outputs[0].shape(), &[3, 2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2, 2]);
    assert_eq!(permutation.shape(), &[2, 2]);
    assert_eq!(rank.shape(), &[2]);
    assert_eq!(rank.as_slice::<i64>().unwrap(), &[2, 1]);
    for batch in permutation.as_slice::<i64>().unwrap().chunks_exact(2) {
        let mut columns = batch.to_vec();
        columns.sort_unstable();
        assert_eq!(columns, vec![0, 1]);
    }
}

#[test]
#[ignore]
fn test_cubecl_eig_f32_returns_unsupported_error() {
    let input = tensor_f32(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let err = with_cuda_linalg_session(&mut gpu, |session| session.eig(&gpu_input)).unwrap_err();
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
    let gpu_input = upload(&gpu, &input);
    let err = with_cuda_linalg_session(&mut gpu, |session| session.eig(&gpu_input)).unwrap_err();
    assert!(matches!(err, Error::Unsupported { op: "eig", .. }));
}

#[test]
#[ignore]
fn test_cubecl_solve_f64_matches_cpu() {
    let a = tensor_f64(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0]);
    let b = tensor_f64(vec![2, 2], vec![5.0, 1.0, -2.0, 4.0]);
    let mut cpu = cpu_backend();
    let expected = with_cpu_linalg_session(&mut cpu, |session| session.solve(&a, &b)).unwrap();
    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);
    let gpu_out =
        with_cuda_linalg_session(&mut gpu, |session| session.solve(&gpu_a, &gpu_b)).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-9);
}

// ---------------------------------------------------------------------------
// Full-matrices SVD on CUDA: `U` is `m x m` and `Vt` is `n x n`, so the
// trailing columns and rows span the left and right nullspaces. The device
// results must satisfy the same identities the CPU providers owe; equality
// across providers is reconstruction, unitarity and spectra, never identical
// basis bytes in degenerate or null subspaces.
// ---------------------------------------------------------------------------

/// Read any supported factor dtype off the host as complex values.
fn complex_values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| Complex64::new(value as f64, 0.0))
            .collect(),
        DType::F64 => tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        DType::C32 => tensor
            .as_slice::<Complex32>()
            .unwrap()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().unwrap().to_vec(),
        other => panic!("unsupported factor dtype {other:?}"),
    }
}

fn real_values(tensor: &Tensor) -> Vec<f64> {
    match tensor.dtype() {
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| value as f64)
            .collect(),
        DType::F64 => tensor.as_slice::<f64>().unwrap().to_vec(),
        other => panic!("singular values must be real, got {other:?}"),
    }
}

/// `matrix^H matrix == I` for a `rows x cols` column-major matrix.
fn assert_isometric(matrix: &[Complex64], rows: usize, cols: usize, tol: f64, label: &str) {
    for left in 0..cols {
        for right in 0..cols {
            let inner: Complex64 = (0..rows)
                .map(|row| matrix[row + left * rows].conj() * matrix[row + right * rows])
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (inner - Complex64::new(expected, 0.0)).norm() < tol,
                "{label}: column pair ({left}, {right}) inner product {inner} is not {expected}"
            );
        }
    }
}

fn adjoint_square(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
    (0..dim * dim)
        .map(|index| matrix[index / dim + (index % dim) * dim].conj())
        .collect()
}

/// Every acceptance identity of the full decomposition, checked on host copies.
fn assert_full_svd_on_host(
    m: usize,
    n: usize,
    source: &[Complex64],
    u: &Tensor,
    s: &Tensor,
    vt: &Tensor,
    tol: f64,
) {
    let k = m.min(n);
    assert_eq!(u.shape(), &[m, m], "U must be square m x m");
    assert_eq!(s.shape(), &[k], "S must hold min(m, n) singular values");
    assert_eq!(vt.shape(), &[n, n], "Vt must be square n x n");

    let u = complex_values(u);
    let vt = complex_values(vt);
    let values = real_values(s);

    assert_isometric(&u, m, m, tol, "U^H U");
    assert_isometric(&adjoint_square(&u, m), m, m, tol, "U U^H");
    assert_isometric(&vt, n, n, tol, "Vt^H Vt");
    assert_isometric(&adjoint_square(&vt, n), n, n, tol, "Vt Vt^H");
    assert!(
        values.windows(2).all(|pair| pair[0] >= pair[1] - tol),
        "singular values must be non-increasing: {values:?}"
    );

    for col in 0..n {
        for row in 0..m {
            let reconstructed: Complex64 = (0..k)
                .map(|index| u[row + index * m] * values[index] * vt[index + col * n])
                .sum();
            assert!(
                (reconstructed - source[row + col * m]).norm() < tol,
                "reconstruction mismatch at ({row}, {col}): {reconstructed} != {}",
                source[row + col * m]
            );
        }
    }
}

/// Deterministic, well-conditioned column-major data with distinct spectra.
fn full_svd_sample(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            1.0 + row * 1.5 - col * 0.75 + row * col * 0.25
        })
        .collect()
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_covers_every_dtype_and_orientation_through_gesvdj() {
    if !gpu_available() {
        return;
    }
    let mut gpu = gpu_backend();
    for (m, n) in [(4_usize, 2_usize), (2, 4), (3, 3)] {
        let real = full_svd_sample(m, n);
        let imaginary: Vec<f64> = (0..m * n)
            .map(|index| 0.5 - (index % m) as f64 * 0.25 + (index / m) as f64 * 0.375)
            .collect();

        let inputs: Vec<(Tensor, f64)> = vec![
            (
                tensor_f32(vec![m, n], real.iter().map(|&value| value as f32).collect()),
                3.0e-4,
            ),
            (tensor_f64(vec![m, n], real.clone()), 1.0e-10),
            (
                tensor_c32(
                    vec![m, n],
                    real.iter()
                        .zip(&imaginary)
                        .map(|(&re, &im)| Complex32::new(re as f32, im as f32))
                        .collect(),
                ),
                3.0e-4,
            ),
            (
                tensor_c64(
                    vec![m, n],
                    real.iter()
                        .zip(&imaginary)
                        .map(|(&re, &im)| Complex64::new(re, im))
                        .collect(),
                ),
                1.0e-10,
            ),
        ];

        for (input, tol) in inputs {
            let source = complex_values(&input);
            let gpu_input = upload(&gpu, &input);
            let outputs =
                with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
            for output in &outputs {
                assert_eq!(
                    output.placement().memory_kind,
                    tenferro_tensor::MemoryKind::Device,
                    "full-SVD outputs must stay resident on the device"
                );
            }
            let u = download(&gpu, &outputs[0]);
            let s = download(&gpu, &outputs[1]);
            let vt = download(&gpu, &outputs[2]);
            assert_full_svd_on_host(m, n, &source, &u, &s, &vt, tol);
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_matches_the_cpu_spectrum() {
    if !gpu_available() {
        return;
    }
    let (m, n) = (5_usize, 3_usize);
    let input = tensor_f64(vec![m, n], full_svd_sample(m, n));
    let mut cpu = cpu_backend();
    let expected = with_cpu_linalg_session(&mut cpu, |session| session.svd_full(&input)).unwrap();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
    let s = download(&gpu, &outputs[1]);
    // Only the spectrum is provider-independent; the bases may differ by phase
    // and in the null subspace.
    for (device, host) in real_values(&s).iter().zip(real_values(&expected[1])) {
        assert!(
            (device - host).abs() < 1.0e-9,
            "device and host spectra disagree: {device} vs {host}"
        );
    }
    assert_eq!(outputs[0].shape(), expected[0].shape());
    assert_eq!(outputs[2].shape(), expected[2].shape());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_handles_batched_inputs() {
    if !gpu_available() {
        return;
    }
    let (m, n, batch) = (3_usize, 2_usize, 2_usize);
    let input = tensor_f64(vec![m, n, batch], full_svd_sample(m, n * batch));
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
    assert_eq!(outputs[0].shape(), &[m, m, batch]);
    assert_eq!(outputs[1].shape(), &[m.min(n), batch]);
    assert_eq!(outputs[2].shape(), &[n, n, batch]);

    let source = complex_values(&input);
    let u = complex_values(&download(&gpu, &outputs[0]));
    let s = real_values(&download(&gpu, &outputs[1]));
    let vt = complex_values(&download(&gpu, &outputs[2]));
    for slab in 0..batch {
        let slab_source = &source[slab * m * n..(slab + 1) * m * n];
        let slab_u = &u[slab * m * m..(slab + 1) * m * m];
        let slab_vt = &vt[slab * n * n..(slab + 1) * n * n];
        let slab_s = &s[slab * m.min(n)..(slab + 1) * m.min(n)];
        assert_isometric(slab_u, m, m, 1.0e-10, "batched U");
        assert_isometric(slab_vt, n, n, 1.0e-10, "batched Vt");
        for col in 0..n {
            for row in 0..m {
                let reconstructed: Complex64 = (0..m.min(n))
                    .map(|index| slab_u[row + index * m] * slab_s[index] * slab_vt[index + col * n])
                    .sum();
                assert!((reconstructed - slab_source[row + col * m]).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_read_accepts_a_strided_device_view() {
    if !gpu_available() {
        return;
    }
    let (m, n) = (2_usize, 3_usize);
    let base = tensor_f64(vec![m, n], full_svd_sample(m, n));
    let mut gpu = gpu_backend();
    let gpu_base = upload(&gpu, &base);
    let Some(typed) = gpu_base.as_typed::<f64>() else {
        panic!("uploaded tensor should keep its f64 tag")
    };
    let transposed = typed.as_view().transpose_view([1, 0]).unwrap();

    let outputs = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_full_read(TensorRead::from_view(tenferro_tensor::TensorView::F64(
            transposed,
        )))
    })
    .unwrap();
    // The view is the transpose, so the decomposed matrix is `n x m`.
    assert_eq!(outputs[0].shape(), &[n, n]);
    assert_eq!(outputs[1].shape(), &[m.min(n)]);
    assert_eq!(outputs[2].shape(), &[m, m]);

    let source_transposed: Vec<Complex64> = (0..m * n)
        .map(|index| {
            let row = index % n;
            let col = index / n;
            Complex64::new(base.as_slice::<f64>().unwrap()[col + row * m], 0.0)
        })
        .collect();
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    assert_full_svd_on_host(n, m, &source_transposed, &u, &s, &vt, 1.0e-10);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_keeps_square_factors_when_a_core_dimension_is_empty() {
    if !gpu_available() {
        return;
    }
    let mut gpu = gpu_backend();
    for (m, n) in [(0_usize, 3_usize), (3, 0)] {
        let input = tensor_f64(vec![m, n], Vec::new());
        let gpu_input = upload(&gpu, &input);
        let outputs =
            with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
        assert_eq!(outputs[0].shape(), &[m, m]);
        assert_eq!(outputs[1].shape(), &[m.min(n)]);
        assert_eq!(outputs[2].shape(), &[n, n]);
        let u = complex_values(&download(&gpu, &outputs[0]));
        let vt = complex_values(&download(&gpu, &outputs[2]));
        assert_isometric(&u, m, m, 1.0e-12, "empty U");
        assert_isometric(&vt, n, n, 1.0e-12, "empty Vt");
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_svd_rejects_unsupported_dtypes() {
    if !gpu_available() {
        return;
    }
    let input = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap(),
    );
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let error =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap_err();
    assert!(matches!(error, Error::Extension { .. }), "got {error:?}");
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and exercises legacy gesvd above 1024"]
fn cuda_full_svd_tall_above_the_gesvdj_threshold_uses_gesvd() {
    if !gpu_available() {
        return;
    }
    // A leading dimension above 1024 selects the legacy gesvd driver, where the
    // full variant asks for `jobu = jobvt = 'A'` instead of `'S'`.
    const M: usize = 1025;
    const N: usize = 4;
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
    let outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
    assert_eq!(outputs[0].shape(), &[M, M]);
    assert_eq!(outputs[1].shape(), &[N]);
    assert_eq!(outputs[2].shape(), &[N, N]);

    let source = complex_values(&input);
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    assert_full_svd_on_host(M, N, &source, &u, &s, &vt, 1.0e-9);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU and exercises legacy gesvd above 1024"]
fn cuda_full_svd_wide_above_the_gesvdj_threshold_reuses_the_adjoint_trick() {
    if !gpu_available() {
        return;
    }
    // `m < n` with `n > 1024`: gesvd factors the adjoint, so the full-`A` outputs
    // come back as `n x n` and `m x m` and are adjointed into place.
    const M: usize = 4;
    const N: usize = 1025;
    let data = (0..N)
        .flat_map(|col| {
            (0..M).map(move |row| {
                let patterned = ((row * 11 + col * 7 + 5) % 29) as f64 / 29.0 - 0.5;
                patterned + if col == row { 2.0 } else { 0.0 }
            })
        })
        .collect::<Vec<_>>();
    let input = tensor_f64(vec![M, N], data);
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let outputs =
        with_cuda_linalg_session(&mut gpu, |session| session.svd_full(&gpu_input)).unwrap();
    assert_eq!(outputs[0].shape(), &[M, M]);
    assert_eq!(outputs[1].shape(), &[M]);
    assert_eq!(outputs[2].shape(), &[N, N]);

    let source = complex_values(&input);
    let u = download(&gpu, &outputs[0]);
    let s = download(&gpu, &outputs[1]);
    let vt = download(&gpu, &outputs[2]);
    assert_full_svd_on_host(M, N, &source, &u, &s, &vt, 1.0e-9);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_q_columns_complete_the_thin_factor_and_span_the_complement() {
    if !gpu_available() {
        return;
    }
    // `q_columns` now reaches the full-Q width `m`, so the columns past
    // `k = min(m, n)` must be orthonormal and orthogonal to the input.
    let (m, n) = (5_usize, 3_usize);
    let data: Vec<f64> = (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            1.0 + row * 1.25 - col * 0.5 + row * col * 0.125
        })
        .collect();
    let input = tensor_f64(vec![m, n], data.clone());
    let mut gpu = gpu_backend();
    let device_input = upload(&gpu, &input);

    let (full, thin, complement) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = device_input.householder_qr(session).unwrap();
        (
            state
                .q_columns(0..m, QrOptions::default(), session)
                .unwrap(),
            state
                .q_columns(0..n, QrOptions::default(), session)
                .unwrap(),
            state
                .q_columns(n..m, QrOptions::default(), session)
                .unwrap(),
        )
    });
    assert_eq!(full.shape(), &[m, m]);
    assert_eq!(thin.shape(), &[m, n]);
    assert_eq!(complement.shape(), &[m, m - n]);

    let full = download(&gpu, &full);
    let full_data = full.as_slice::<f64>().unwrap();
    for left in 0..m {
        for right in 0..m {
            let inner: f64 = (0..m)
                .map(|row| full_data[row + left * m] * full_data[row + right * m])
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (inner - expected).abs() < 1.0e-9,
                "device Q column pair ({left}, {right}) inner product {inner}"
            );
        }
    }
    for col in n..m {
        for input_col in 0..n {
            let inner: f64 = (0..m)
                .map(|row| data[row + input_col * m] * full_data[row + col * m])
                .sum();
            assert!(
                inner.abs() < 1.0e-9,
                "device complement column {col} is not orthogonal to input column {input_col}"
            );
        }
    }

    let thin = download(&gpu, &thin);
    assert_tensor_close(
        &tensor_f64(vec![m, n], full_data[..m * n].to_vec()),
        &thin,
        1.0e-12,
    );
    let complement = download(&gpu, &complement);
    assert_tensor_close(
        &tensor_f64(vec![m, m - n], full_data[n * m..].to_vec()),
        &complement,
        1.0e-12,
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_full_q_positive_diagonal_gauge_leaves_complement_columns_alone() {
    if !gpu_available() {
        return;
    }
    // The gauge comes from R's diagonal, so it has nothing to apply to a
    // complement column; the device kernel must skip it rather than index past
    // the phase vector.
    let (m, n) = (4_usize, 2_usize);
    let input = tensor_f64(vec![m, n], vec![1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 0.5, 3.0]);
    let mut gpu = gpu_backend();
    let device_input = upload(&gpu, &input);

    let (raw, gauged) = with_cuda_linalg_session(&mut gpu, |session| {
        let state = device_input.householder_qr(session).unwrap();
        (
            state
                .q_columns(0..m, QrOptions::default(), session)
                .unwrap(),
            state
                .q_columns(
                    0..m,
                    QrOptions::default().gauge(QrGauge::PositiveDiagonal),
                    session,
                )
                .unwrap(),
        )
    });
    let raw = download(&gpu, &raw);
    let gauged = download(&gpu, &gauged);
    assert_eq!(gauged.shape(), &[m, m]);
    let raw_data = raw.as_slice::<f64>().unwrap();
    let gauged_data = gauged.as_slice::<f64>().unwrap();
    assert_tensor_close(
        &tensor_f64(vec![m, m - n], gauged_data[n * m..].to_vec()),
        &tensor_f64(vec![m, m - n], raw_data[n * m..].to_vec()),
        1.0e-12,
    );
    for col in 0..m {
        let norm: f64 = (0..m).map(|row| gauged_data[row + col * m].powi(2)).sum();
        assert!(
            (norm - 1.0).abs() < 1.0e-9,
            "gauged device column {col} norm {norm}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_q_column_ranges_past_the_full_q_width_are_rejected() {
    if !gpu_available() {
        return;
    }
    let (m, n) = (4_usize, 2_usize);
    let input = tensor_f64(vec![m, n], vec![1.0, 2.0, 3.0, 4.0, 2.0, -1.0, 0.5, 3.0]);
    let mut gpu = gpu_backend();
    let device_input = upload(&gpu, &input);
    let error = with_cuda_linalg_session(&mut gpu, |session| {
        let state = device_input.householder_qr(session).unwrap();
        state
            .q_columns(0..m + 1, QrOptions::default(), session)
            .unwrap_err()
    });
    assert!(
        format!("{error}").contains("range"),
        "expected a range validation error, got {error}"
    );
}
