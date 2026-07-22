//! Full-matrices SVD and least-squares behavior tests.
//!
//! The full-SVD numeric tests exercise the CPU faer provider and are gated on
//! `cpu-faer`; the LAPACK provider is intentionally unsupported for the full
//! variant in this slice and is covered by a separate boundary test. The
//! `lstsq` tests are provider-agnostic because `lstsq` composes `qr` and
//! `triangular_solve`, which both providers implement.

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(Tensor::F64(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    ))
    .unwrap()
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(Tensor::C64(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    ))
    .unwrap()
}

fn run_many(outputs: &[&TracedTensor]) -> Vec<Tensor> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    executor.run_many(&program).unwrap()
}

fn f64_data(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().unwrap().to_vec()
}

fn c64_data(tensor: &Tensor) -> Vec<Complex64> {
    tensor.as_slice::<Complex64>().unwrap().to_vec()
}

/// Column-major reconstruction `A = U[:, :k] diag(S) Vh[:k, :]` for real inputs.
#[cfg(feature = "cpu-faer")]
fn reconstruct_real(m: usize, n: usize, u: &[f64], s: &[f64], vh: &[f64]) -> Vec<f64> {
    let k = m.min(n);
    let mut a = vec![0.0_f64; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0;
            for t in 0..k {
                acc += u[i + t * m] * s[t] * vh[t + j * n];
            }
            a[i + j * m] = acc;
        }
    }
    a
}

#[cfg(feature = "cpu-faer")]
fn reconstruct_complex(
    m: usize,
    n: usize,
    u: &[Complex64],
    s: &[f64],
    vh: &[Complex64],
) -> Vec<Complex64> {
    let k = m.min(n);
    let mut a = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = Complex64::new(0.0, 0.0);
            for t in 0..k {
                acc += u[i + t * m] * s[t] * vh[t + j * n];
            }
            a[i + j * m] = acc;
        }
    }
    a
}

fn max_abs_diff_real(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Full-matrices SVD (faer provider).
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_tall_real_returns_square_factors_and_reconstructs() {
    let m = 3;
    let n = 2;
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    let a = f64_tensor(vec![m, n], data.clone());
    let (u, s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&u, &s, &vh]);
    assert_eq!(out[0].shape(), &[m, m]);
    assert_eq!(out[1].shape(), &[n.min(m)]);
    assert_eq!(out[2].shape(), &[n, n]);

    let recon = reconstruct_real(
        m,
        n,
        &f64_data(&out[0]),
        &f64_data(&out[1]),
        &f64_data(&out[2]),
    );
    assert!(
        max_abs_diff_real(&recon, &data) < 1e-10,
        "reconstruction residual {}",
        max_abs_diff_real(&recon, &data)
    );
}

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_wide_real_reconstructs_and_recovers_nullspace() {
    // rank-2 wide matrix (2 x 3): the trailing Vh row spans the 1-D kernel.
    let m = 2;
    let n = 3;
    let data = vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let a = f64_tensor(vec![m, n], data.clone());
    let (u, s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&u, &s, &vh]);
    assert_eq!(out[0].shape(), &[m, m]);
    assert_eq!(out[2].shape(), &[n, n]);

    let u = f64_data(&out[0]);
    let s = f64_data(&out[1]);
    let vh = f64_data(&out[2]);
    let recon = reconstruct_real(m, n, &u, &s, &vh);
    assert!(max_abs_diff_real(&recon, &data) < 1e-10);

    // Trailing Vh rows (index rank..n) span the right nullspace: A v = 0.
    let rank = 2;
    for t in rank..n {
        // v[j] = Vh[t, j]  (column-major (n x n): index t + j*n)
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..n {
                acc += data[i + j * m] * vh[t + j * n];
            }
            assert!(acc.abs() < 1e-10, "A v residual {acc}");
        }
    }
}

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_1x2_recovers_one_dimensional_nullspace() {
    // Smallest wide system: thin SVD would drop the kernel row entirely.
    let a = f64_tensor(vec![1, 2], vec![1.0_f64, 1.0]);
    let (u, s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&u, &s, &vh]);
    assert_eq!(out[0].shape(), &[1, 1]);
    assert_eq!(out[2].shape(), &[2, 2]);

    let vh = f64_data(&out[2]);
    // Kernel row is index 1: [Vh[1,0], Vh[1,1]] with a . [1,1] == 0.
    // Column-major (2 x 2): Vh[t, j] at index t + j * n.
    let v0 = vh[1]; // t=1, j=0 -> 1
    let v1 = vh[3]; // t=1, j=1 -> 3
    assert!((v0 + v1).abs() < 1e-10, "nullspace dot {}", v0 + v1);
    // It is a genuine unit direction, not the zero vector.
    assert!((v0 * v0 + v1 * v1 - 1.0).abs() < 1e-10);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_random_wide_recovers_nullspace_dimension() {
    // Deterministic rank-3 (3 x 5) matrix: rows are 3 independent vectors, so
    // the trailing two Vh rows span the 2-D kernel.
    let m = 3;
    let n = 5;
    #[rustfmt::skip]
    let data_row_major = [
        [ 2.0, -1.0,  0.0,  3.0,  1.0],
        [ 0.0,  4.0, -2.0,  1.0,  5.0],
        [ 1.0,  1.0,  1.0, -1.0,  2.0],
    ];
    let mut data = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            data[i + j * m] = data_row_major[i][j];
        }
    }
    let a = f64_tensor(vec![m, n], data.clone());
    let (_u, _s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&vh]);
    let vh = f64_data(&out[0]);

    let rank = 3;
    for t in rank..n {
        for i in 0..m {
            let mut acc = 0.0;
            for j in 0..n {
                acc += data[i + j * m] * vh[t + j * n];
            }
            assert!(acc.abs() < 1e-9, "A v residual {acc} for kernel row {t}");
        }
    }
}

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_complex_tall_reconstructs() {
    let m = 3;
    let n = 2;
    let data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(0.0, 3.0),
        Complex64::new(-1.0, 2.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -2.0),
    ];
    let a = c64_tensor(vec![m, n], data.clone());
    let (u, s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&u, &s, &vh]);
    assert_eq!(out[0].shape(), &[m, m]);
    assert_eq!(out[2].shape(), &[n, n]);

    let recon = reconstruct_complex(
        m,
        n,
        &c64_data(&out[0]),
        &f64_data(&out[1]),
        &c64_data(&out[2]),
    );
    let residual = recon
        .iter()
        .zip(&data)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        residual < 1e-10,
        "complex reconstruction residual {residual}"
    );
}

#[cfg(feature = "cpu-faer")]
#[test]
fn svd_full_batch_returns_square_factor_shapes() {
    // Leading matrix dims [m, n], trailing batch dim.
    let m = 3;
    let n = 2;
    let batch = 2;
    let data: Vec<f64> = (0..(m * n * batch))
        .map(|v| (v as f64) * 0.5 + 1.0)
        .collect();
    let a = f64_tensor(vec![m, n, batch], data);
    let (u, s, vh) = a.svd_full().unwrap();
    let out = run_many(&[&u, &s, &vh]);
    assert_eq!(out[0].shape(), &[m, m, batch]);
    assert_eq!(out[1].shape(), &[n.min(m), batch]);
    assert_eq!(out[2].shape(), &[n, n, batch]);
}

// The LAPACK provider does not implement full-matrices SVD in this slice; it
// must surface a typed error rather than silently returning a thin factor.
#[cfg(all(feature = "cpu-blas", not(feature = "cpu-faer")))]
#[test]
fn svd_full_lapack_provider_is_unsupported() {
    let a = f64_tensor(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0]);
    let (u, s, vh) = a.svd_full().unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vh]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let err = executor.run_many(&program).unwrap_err();
    assert!(
        format!("{err}").contains("full-matrices SVD"),
        "expected typed unsupported error, got {err}"
    );
}

// ---------------------------------------------------------------------------
// Least squares (provider-agnostic composition).
// ---------------------------------------------------------------------------

#[test]
fn lstsq_solves_tall_consistent_real_system() {
    // Overdetermined 3 x 2 full-column-rank system with a consistent RHS: the
    // least-squares solution recovers the exact generating x.
    let m = 3;
    let n = 2;
    let a_data = vec![1.0_f64, 1.0, 1.0, 0.0, 1.0, 2.0];
    let x_true = [2.0_f64, -1.0];
    // b = A x_true, column-major A is [m, n] -> A[i,j] = a_data[i + j*m].
    let mut b_data = vec![0.0_f64; m];
    for i in 0..m {
        for (j, xj) in x_true.iter().enumerate() {
            b_data[i] += a_data[i + j * m] * xj;
        }
    }
    let a = f64_tensor(vec![m, n], a_data);
    let b = f64_tensor(vec![m, 1], b_data);
    let x = a.lstsq(&b).unwrap();
    let out = run_many(&[&x]);
    assert_eq!(out[0].shape(), &[n, 1]);
    let x = f64_data(&out[0]);
    assert!(max_abs_diff_real(&x, &x_true) < 1e-9, "lstsq x = {x:?}");
}

#[test]
fn lstsq_solves_square_real_system() {
    let a = f64_tensor(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]);
    let b = f64_tensor(vec![2, 1], vec![4.0_f64, 8.0]);
    let x = a.lstsq(&b).unwrap();
    let out = run_many(&[&x]);
    let x = f64_data(&out[0]);
    assert!(max_abs_diff_real(&x, &[2.0, 2.0]) < 1e-10, "x = {x:?}");
}

#[test]
fn lstsq_solves_tall_consistent_complex_system() {
    let m = 3;
    let n = 2;
    let a_data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(3.0, 0.0),
    ];
    let x_true = [Complex64::new(1.0, -2.0), Complex64::new(-1.0, 1.0)];
    let mut b_data = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        for (j, xj) in x_true.iter().enumerate() {
            b_data[i] += a_data[i + j * m] * xj;
        }
    }
    let a = c64_tensor(vec![m, n], a_data);
    let b = c64_tensor(vec![m, 1], b_data);
    let x = a.lstsq(&b).unwrap();
    let out = run_many(&[&x]);
    let x = c64_data(&out[0]);
    let residual = x
        .iter()
        .zip(&x_true)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(
        residual < 1e-9,
        "complex lstsq residual {residual}, x = {x:?}"
    );
}

#[test]
fn lstsq_rejects_wide_system() {
    let a = f64_tensor(vec![2, 3], vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let b = f64_tensor(vec![2, 1], vec![1.0_f64, 2.0]);
    let err = a.lstsq(&b).unwrap_err();
    assert!(
        format!("{err}").contains("tall or square"),
        "expected wide-system rejection, got {err}"
    );
}
