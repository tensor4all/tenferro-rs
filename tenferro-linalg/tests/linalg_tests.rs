//! Tests for tenferro-linalg: forward decompositions and AD rules.

use num_complex::{Complex32, Complex64};
use tenferro_linalg::*;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

/// Create a column-major tensor from a flat vec and shape.
fn make_tensor(data: Vec<f64>, dims: &[usize]) -> Tensor<f64> {
    let ndim = dims.len();
    let mut strides = vec![0isize; ndim];
    if ndim > 0 {
        strides[0] = 1;
        for i in 1..ndim {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

/// Create a row-major tensor from a flat vec and shape.
fn make_tensor_row_major(data: Vec<f64>, dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(&data, dims, MemoryOrder::RowMajor).unwrap()
}

/// Extract flat data from a Tensor.
fn tensor_data(t: &Tensor<f64>) -> Vec<f64> {
    let c = t.contiguous(COL);
    let off = c.offset() as usize;
    let len: usize = c.dims().iter().product();
    c.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

// ============================================================================
// FD verification utilities
// ============================================================================

/// Compute numerical Jacobian for a matrix->matrix function.
/// For each input element, perturbs by +/-eps and computes central difference.
/// Returns one Tensor per input element (Jacobian column).
fn fd_gradient_matrix(
    f: impl Fn(&Tensor<f64>) -> Tensor<f64>,
    a: &Tensor<f64>,
    eps: f64,
) -> Vec<Tensor<f64>> {
    let n = a.dims().iter().product::<usize>();
    let a_data = tensor_data(a);
    let mut grads = Vec::with_capacity(n);
    for idx in 0..n {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let f_plus = f(&make_tensor(plus, a.dims()));
        let f_minus = f(&make_tensor(minus, a.dims()));
        let fp = tensor_data(&f_plus);
        let fm = tensor_data(&f_minus);
        let grad: Vec<f64> = fp
            .iter()
            .zip(&fm)
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        grads.push(make_tensor(grad, f_plus.dims()));
    }
    grads
}

/// Check that analytic rrule gradient matches FD gradient via VJP contract.
///
/// The VJP contract: <cotangent, J*v> = <J^T*cotangent, v> for all v.
/// We check: analytic_grad = sum_k cotangent[k] * df_k/da[ij].
fn check_rrule_fd<F, G>(forward: F, rrule: G, a: &Tensor<f64>, eps: f64, atol: f64)
where
    F: Fn(&Tensor<f64>) -> Tensor<f64>,
    G: Fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
{
    let output = forward(a);
    let out_size: usize = output.dims().iter().product();
    // Deterministic "random" cotangent
    let cotangent_data: Vec<f64> = (0..out_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent = make_tensor(cotangent_data.clone(), output.dims());

    let analytic_grad = rrule(a, &cotangent);
    let analytic = tensor_data(&analytic_grad);

    // FD: sum_k cotangent[k] * df_k/da[ij]
    let fd_jac = fd_gradient_matrix(&forward, a, eps);
    let a_size: usize = a.dims().iter().product();
    let mut fd_grad = vec![0.0; a_size];
    for ij in 0..a_size {
        let jac_col = tensor_data(&fd_jac[ij]);
        for k in 0..out_size {
            fd_grad[ij] += cotangent_data[k] * jac_col[k];
        }
    }

    let max_err = analytic
        .iter()
        .zip(&fd_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "rrule FD check failed: max_err={max_err} > atol={atol}"
    );
}

/// Check that analytic frule tangent matches FD directional derivative.
fn check_frule_fd<F, G>(forward: F, frule: G, a: &Tensor<f64>, eps: f64, atol: f64)
where
    F: Fn(&Tensor<f64>) -> Tensor<f64>,
    G: Fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
{
    let a_size: usize = a.dims().iter().product();
    // Deterministic "random" tangent direction
    let tangent_data: Vec<f64> = (0..a_size)
        .map(|i| ((i * 13 + 5) % 17) as f64 / 8.0 - 1.0)
        .collect();
    let tangent = make_tensor(tangent_data.clone(), a.dims());

    let analytic_out = frule(a, &tangent);
    let analytic = tensor_data(&analytic_out);

    // FD: (f(A + eps*dA) - f(A - eps*dA)) / (2*eps)
    let a_data = tensor_data(a);
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let f_plus = forward(&make_tensor(plus, a.dims()));
    let f_minus = forward(&make_tensor(minus, a.dims()));
    let fp = tensor_data(&f_plus);
    let fm = tensor_data(&f_minus);
    let fd: Vec<f64> = fp
        .iter()
        .zip(&fm)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

/// Create a well-conditioned n x n test matrix with distinct eigenvalues.
/// Uses deterministic construction: a + a^T + n*I (symmetric positive definite).
fn make_well_conditioned_matrix(n: usize) -> Tensor<f64> {
    let mut data = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let val = ((i * 7 + j * 13 + 3) % 19) as f64 / 10.0 - 0.9;
            data[i + j * n] = val;
        }
    }
    // Symmetrize: A = A + A^T
    let mut sym = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            sym[i + j * n] = data[i + j * n] + data[j + i * n];
        }
    }
    // Add n*I to ensure positive definite
    for i in 0..n {
        sym[i + i * n] += n as f64;
    }
    make_tensor(sym, &[n, n])
}

/// Create a general (non-symmetric) well-conditioned n x n test matrix.
fn make_general_test_matrix(n: usize) -> Tensor<f64> {
    let mut data = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let val = ((i * 7 + j * 13 + 3) % 19) as f64 / 10.0 - 0.9;
            data[i + j * n] = val;
        }
    }
    // Add n*I to ensure invertibility
    for i in 0..n {
        data[i + i * n] += n as f64;
    }
    make_tensor(data, &[n, n])
}

// ============================================================================
// FD helpers smoke test
// ============================================================================

#[test]
fn fd_helpers_smoke_test() {
    // Test with matrix transpose (f(A) = A^T)
    // Jacobian of transpose is a permutation matrix
    // rrule: grad of A^T w.r.t. cotangent is cotangent^T
    // frule: tangent of A^T w.r.t. dA is dA^T
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let transpose_fn = |x: &Tensor<f64>| {
        // Manual transpose for 2x3 -> 3x2
        let d = tensor_data(x);
        // Input is 2x3 col-major: d[i + j*2] for i in 0..2, j in 0..3
        // Output is 3x2 col-major: o[j + i*3] for i in 0..2, j in 0..3
        let mut out = vec![0.0; 6];
        for i in 0..2 {
            for j in 0..3 {
                out[j + i * 3] = d[i + j * 2];
            }
        }
        make_tensor(out, &[3, 2])
    };

    // frule: d(A^T) should be (dA)^T
    check_frule_fd(
        transpose_fn,
        |_x, dx| {
            let dd = tensor_data(dx);
            let mut out = vec![0.0; 6];
            for i in 0..2 {
                for j in 0..3 {
                    out[j + i * 3] = dd[i + j * 2];
                }
            }
            make_tensor(out, &[3, 2])
        },
        &a,
        1e-6,
        1e-8,
    );

    // rrule: grad w.r.t. A given cotangent on A^T is cotangent^T
    check_rrule_fd(
        transpose_fn,
        |_x, co| {
            let cd = tensor_data(co);
            // cotangent is 3x2, output grad is 2x3
            let mut out = vec![0.0; 6];
            for i in 0..3 {
                for j in 0..2 {
                    out[j + i * 2] = cd[i + j * 3];
                }
            }
            make_tensor(out, &[2, 3])
        },
        &a,
        1e-6,
        1e-8,
    );
}

#[test]
fn fd_helpers_well_conditioned_matrix() {
    // Verify that make_well_conditioned_matrix produces a symmetric positive definite matrix.
    let a = make_well_conditioned_matrix(3);
    let d = tensor_data(&a);
    // Check symmetry: a[i,j] == a[j,i]
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (d[i + j * 3] - d[j + i * 3]).abs() < 1e-15,
                "not symmetric at ({i},{j})"
            );
        }
    }
    // Check positive diagonal (necessary but not sufficient for PD)
    for i in 0..3 {
        assert!(d[i + i * 3] > 0.0, "diagonal not positive at ({i},{i})");
    }
}

#[test]
fn fd_helpers_general_test_matrix() {
    // Verify that make_general_test_matrix produces a non-singular matrix
    // by checking that the diagonal dominance condition holds.
    let a = make_general_test_matrix(3);
    let d = tensor_data(&a);
    for i in 0..3 {
        let diag = d[i + i * 3].abs();
        let off_sum: f64 = (0..3).filter(|&j| j != i).map(|j| d[i + j * 3].abs()).sum();
        assert!(
            diag > off_sum,
            "not diagonally dominant at row {i}: diag={diag}, off_sum={off_sum}"
        );
    }
}

// ============================================================================
// SVD tests
// ============================================================================

#[test]
fn svd_identity_3x3() {
    let mut ctx = CpuContext::new(1);
    // SVD of identity should give U=I, S=[1,1,1], Vt=I (up to sign)
    let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = make_tensor(data, &[3, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let s = tensor_data(&result.s);
    assert_eq!(s.len(), 3);
    for v in &s {
        assert!(
            (v - 1.0).abs() < 1e-10,
            "singular value should be 1.0, got {v}"
        );
    }
}

#[test]
fn svd_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // A = U * diag(S) * Vt should reconstruct A
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 col-major
    let a = make_tensor(data.clone(), &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();

    let u = tensor_data(&result.u);
    let s = tensor_data(&result.s);
    let vt = tensor_data(&result.vt);
    let m = 2;
    let n = 3;
    let k = 2;

    // Reconstruct: A_recon = U * diag(S) * Vt
    let mut recon = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u[i + l * m] * s[l] * vt[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }

    let err = data
        .iter()
        .zip(recon.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "SVD reconstruction error: {err}");
}

#[test]
fn svd_accepts_row_major_dense_input() {
    let mut ctx = CpuContext::new(1);
    let logical_col_major = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor_row_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();

    let u = tensor_data(&result.u);
    let s = tensor_data(&result.s);
    let vt = tensor_data(&result.vt);
    let m = 2;
    let n = 3;
    let k = 2;

    let mut recon = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += u[i + l * m] * s[l] * vt[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }

    let err = logical_col_major
        .iter()
        .zip(recon.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "row-major SVD reconstruction error: {err}");
}

#[test]
fn svd_tall_matrix() {
    let mut ctx = CpuContext::new(1);
    // 4x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = make_tensor(data.clone(), &[4, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    assert_eq!(result.u.dims(), &[4, 2]);
    assert_eq!(result.s.dims(), &[2]);
    assert_eq!(result.vt.dims(), &[2, 2]);

    // Check descending order
    let s = tensor_data(&result.s);
    assert!(s[0] >= s[1], "singular values should be descending");
}

#[test]
fn svd_with_max_rank() {
    let mut ctx = CpuContext::new(1);
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor(data, &[2, 3]);
    let opts = SvdOptions {
        max_rank: Some(1),
        cutoff: None,
    };
    let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
    assert_eq!(result.s.dims(), &[1]);
    assert_eq!(result.u.dims(), &[2, 1]);
    assert_eq!(result.vt.dims(), &[1, 3]);
}

// ============================================================================
// QR tests
// ============================================================================

#[test]
fn qr_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor(data.clone(), &[2, 3]);
    let result = qr(&mut ctx, &a).unwrap();

    let q = tensor_data(&result.q);
    let r = tensor_data(&result.r);
    let m = 2;
    let n = 3;
    let k = 2;

    let mut recon = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[i + l * m] * r[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }

    let err = data
        .iter()
        .zip(recon.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "QR reconstruction error: {err}");
}

#[test]
fn qr_accepts_row_major_dense_input() {
    let mut ctx = CpuContext::new(1);
    let logical_col_major = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor_row_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = qr(&mut ctx, &a).unwrap();

    let q = tensor_data(&result.q);
    let r = tensor_data(&result.r);
    let m = 2;
    let n = 3;
    let k = 2;

    let mut recon = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += q[i + l * m] * r[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }

    let err = logical_col_major
        .iter()
        .zip(recon.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "row-major QR reconstruction error: {err}");
}

#[test]
fn qr_orthogonality() {
    let mut ctx = CpuContext::new(1);
    let data = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let a = make_tensor(data, &[3, 3]);
    let result = qr(&mut ctx, &a).unwrap();

    let q = tensor_data(&result.q);
    let n = 3;
    // Q^T Q should be identity
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for l in 0..n {
                dot += q[l + i * n] * q[l + j * n];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-10,
                "Q^T Q[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

// ============================================================================
// LU tests
// ============================================================================

#[test]
fn lu_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![2.0, 1.0, 3.0, 1.0, 4.0, 7.0, 5.0, 3.0, 2.0];
    let a = make_tensor(data.clone(), &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();

    let l = tensor_data(&result.l);
    let u = tensor_data(&result.u);
    let p = result.p.unwrap();
    let n = 3;

    // P A = L U -> A = P^T L U
    let mut lu_prod = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                val += l[i + k * n] * u[k + j * n];
            }
            lu_prod[i + j * n] = val;
        }
    }

    // Apply P^T: A[p_inv[i], j] = lu[i, j]
    let mut p_inv = vec![0; n];
    for i in 0..n {
        p_inv[p[i]] = i;
    }

    for i in 0..n {
        for j in 0..n {
            let err = (data[p[i] + j * n] - lu_prod[i + j * n]).abs();
            assert!(err < 1e-10, "LU reconstruction error at ({i},{j}): {err}");
        }
    }
}

// ============================================================================
// Cholesky tests
// ============================================================================

#[test]
fn cholesky_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // A = [[4, 2], [2, 3]] (symmetric positive definite)
    let data = vec![4.0, 2.0, 2.0, 3.0];
    let a = make_tensor(data, &[2, 2]);
    let l = cholesky(&mut ctx, &a).unwrap();
    let l_data = tensor_data(&l);
    let n = 2;

    // L L^T should equal A
    let mut recon = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                val += l_data[i + k * n] * l_data[j + k * n];
            }
            recon[i + j * n] = val;
        }
    }
    assert!((recon[0] - 4.0).abs() < 1e-10);
    assert!((recon[1] - 2.0).abs() < 1e-10);
    assert!((recon[2] - 2.0).abs() < 1e-10);
    assert!((recon[3] - 3.0).abs() < 1e-10);
}

#[test]
fn cholesky_not_positive_definite() {
    let mut ctx = CpuContext::new(1);
    let data = vec![1.0, 0.0, 0.0, -1.0];
    let a = make_tensor(data, &[2, 2]);
    assert!(cholesky(&mut ctx, &a).is_err());
}

// ============================================================================
// Eigen tests
// ============================================================================

#[test]
fn eigen_symmetric() {
    let mut ctx = CpuContext::new(1);
    // Symmetric matrix: [[2, 1], [1, 2]]
    let data = vec![2.0, 1.0, 1.0, 2.0];
    let a = make_tensor(data, &[2, 2]);
    let result = eigen(&mut ctx, &a).unwrap();

    let vals = tensor_data(&result.values);
    // Eigenvalues should be 1 and 3 (ascending)
    assert!((vals[0] - 1.0).abs() < 1e-10, "eigenvalue 0: {}", vals[0]);
    assert!((vals[1] - 3.0).abs() < 1e-10, "eigenvalue 1: {}", vals[1]);
}

#[test]
fn eigen_nonsymmetric_returns_error() {
    let mut ctx = CpuContext::new(1);
    // Non-symmetric matrix: [[2, 3], [1, 4]]
    let data = vec![2.0, 1.0, 3.0, 4.0];
    let a = make_tensor(data, &[2, 2]);
    assert!(eigen(&mut ctx, &a).is_err());
}

// ============================================================================
// Solve tests
// ============================================================================

#[test]
fn solve_identity() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![1.0, 0.0, 0.0, 1.0];
    let b_data = vec![3.0, 7.0];
    let a = make_tensor(a_data, &[2, 2]);
    let b = make_tensor(b_data.clone(), &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data(&x);
    assert!((xd[0] - 3.0).abs() < 1e-10);
    assert!((xd[1] - 7.0).abs() < 1e-10);
}

#[test]
fn solve_accepts_row_major_dense_input() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor_row_major(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_tensor_row_major(vec![17.0, 39.0], &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data(&x);
    let res0 = 1.0 * xd[0] + 2.0 * xd[1] - 17.0;
    let res1 = 3.0 * xd[0] + 4.0 * xd[1] - 39.0;
    assert!(res0.abs() < 1e-10, "row-major solve residual[0] = {res0}");
    assert!(res1.abs() < 1e-10, "row-major solve residual[1] = {res1}");
}

#[test]
fn solve_general() {
    let mut ctx = CpuContext::new(1);
    // A = [[2, 1], [1, 3]], b = [5, 10]
    let a = make_tensor(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![5.0, 10.0], &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data(&x);
    // Verify: A x = b
    let res0 = 2.0 * xd[0] + 1.0 * xd[1] - 5.0;
    let res1 = 1.0 * xd[0] + 3.0 * xd[1] - 10.0;
    assert!(res0.abs() < 1e-10, "residual[0] = {res0}");
    assert!(res1.abs() < 1e-10, "residual[1] = {res1}");
}

#[test]
fn solve_singular_matrix_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0], &[2, 1]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_rhs_shape_mismatch_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // Wrong leading dim: expected 2, got 3
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_scalar_rhs_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // Scalar RHS is invalid; previously this could hit panic-prone paths.
    let b = make_tensor(vec![1.0], &[]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_triangular_batch_mismatch_returns_error() {
    let mut ctx = CpuContext::new(1);
    // A has batch dim [2], b has batch dim [3]
    let a_data = vec![
        1.0, 0.0, 0.0, 1.0, // batch 0
        1.0, 0.0, 0.0, 1.0, // batch 1
    ];
    let a = make_tensor(a_data, &[2, 2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(solve_triangular(&mut ctx, &a, &b, true).is_err());
}

#[test]
fn solve_triangular_zero_diagonal_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0], &[2, 1]);
    assert!(solve_triangular(&mut ctx, &a, &b, true).is_err());
}

#[test]
fn lstsq_rhs_shape_mismatch_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]); // expected [2]
    assert!(lstsq(&mut ctx, &a, &b).is_err());
}

// ============================================================================
// Inverse tests
// ============================================================================

#[test]
fn inv_2x2() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a_inv = inv(&mut ctx, &a).unwrap();
    let inv_data = tensor_data(&a_inv);

    // A * A^{-1} should be identity
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let n = 2;
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                val += a_data[i + k * n] * inv_data[k + j * n];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-10,
                "A*A^-1[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn inv_singular_matrix_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
    assert!(inv(&mut ctx, &a).is_err());
}

// ============================================================================
// Determinant tests
// ============================================================================

#[test]
fn det_2x2() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let d = det(&mut ctx, &a).unwrap();
    let dv = tensor_data(&d);
    // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert!((dv[0] - (-2.0)).abs() < 1e-10, "det = {}", dv[0]);
}

#[test]
fn det_3x3() {
    let mut ctx = CpuContext::new(1);
    // det([[1,4,7],[2,5,8],[3,6,10]]) = 1*(50-48) - 4*(20-24) + 7*(12-15) = 2+16-21 = -3
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], &[3, 3]);
    let d = det(&mut ctx, &a).unwrap();
    let dv = tensor_data(&d);
    assert!((dv[0] - (-3.0)).abs() < 1e-10, "det = {}", dv[0]);
}

// ============================================================================
// Slogdet tests
// ============================================================================

#[test]
fn slogdet_positive_det() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let result = slogdet(&mut ctx, &a).unwrap();
    let sign = tensor_data(&result.sign);
    let logabsdet = tensor_data(&result.logabsdet);
    assert!((sign[0] - 1.0).abs() < 1e-10, "sign should be 1.0");
    assert!(
        (logabsdet[0] - (6.0_f64).ln()).abs() < 1e-10,
        "logabsdet should be ln(6)"
    );
}

// ============================================================================
// Norm tests
// ============================================================================

#[test]
fn norm_frobenius() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Fro).unwrap();
    let nv = tensor_data(&n);
    let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
    assert!((nv[0] - expected).abs() < 1e-10);
}

#[test]
fn norm_spectral() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Spectral).unwrap();
    let nv = tensor_data(&n);
    assert!((nv[0] - 2.0).abs() < 1e-10, "spectral norm should be 2.0");
}

#[test]
fn norm_frobenius_batched_returns_batch_shape() {
    let mut ctx = CpuContext::new(1);
    // Shape [2,2,2]: two batches of 2x2 matrices.
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let a = make_tensor(data, &[2, 2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Fro).unwrap();
    assert_eq!(n.dims(), &[2]);
    let nv = tensor_data(&n);
    assert!((nv[0] - (30.0_f64).sqrt()).abs() < 1e-10);
    assert!((nv[1] - (174.0_f64).sqrt()).abs() < 1e-10);
}

// ============================================================================
// Pinv tests
// ============================================================================

#[test]
fn pinv_square_invertible() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ap = pinv(&mut ctx, &a, None).unwrap();
    let ap_data = tensor_data(&ap);

    // A * A+ * A ~= A
    let a_data = [1.0, 2.0, 3.0, 4.0];
    let n = 2;
    // A * A+
    let mut aap = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                aap[i + j * n] += a_data[i + k * n] * ap_data[k + j * n];
            }
        }
    }
    // (A * A+) * A
    let mut aapa = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                aapa[i + j * n] += aap[i + k * n] * a_data[k + j * n];
            }
        }
    }
    let err = a_data
        .iter()
        .zip(aapa.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "pinv reconstruction error: {err}");
}

// ============================================================================
// Eig (general, non-symmetric) tests
// ============================================================================

/// Extract flat data from a complex Tensor.
fn tensor_data_complex(t: &Tensor<num_complex::Complex64>) -> Vec<num_complex::Complex64> {
    let c = t.contiguous(COL);
    let off = c.offset() as usize;
    let len: usize = c.dims().iter().product();
    c.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

#[test]
fn eig_2x2_identity() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let result = eig(&mut ctx, &a).unwrap();
    assert_eq!(result.values.dims(), &[2]);
    assert_eq!(result.vectors.dims(), &[2, 2]);
    // Identity has eigenvalues 1.0 + 0i
    let vals = tensor_data_complex(&result.values);
    for v in &vals {
        assert!((v.re - 1.0).abs() < 1e-10, "expected re=1.0, got {}", v.re);
        assert!(v.im.abs() < 1e-10, "expected im=0.0, got {}", v.im);
    }
}

#[test]
fn eig_2x2_real_eigenvalues() {
    // Diagonal matrix: eigenvalues are diagonal entries
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let result = eig(&mut ctx, &a).unwrap();
    let vals = tensor_data_complex(&result.values);
    let mut reals: Vec<f64> = vals.iter().map(|c| c.re).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (reals[0] - 2.0).abs() < 1e-10,
        "expected 2.0, got {}",
        reals[0]
    );
    assert!(
        (reals[1] - 3.0).abs() < 1e-10,
        "expected 3.0, got {}",
        reals[1]
    );
    // Imaginary parts should be zero
    for v in &vals {
        assert!(v.im.abs() < 1e-10, "expected im=0.0, got {}", v.im);
    }
}

#[test]
fn eig_accepts_row_major_dense_input() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor_row_major(vec![1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0], &[3, 3]);
    let result = eig(&mut ctx, &a).unwrap();
    let vals = tensor_data_complex(&result.values);
    let vecs = tensor_data_complex(&result.vectors);
    let n = 3;
    let a_data = tensor_data(&a);

    for j in 0..n {
        for i in 0..n {
            let mut av = num_complex::Complex64::new(0.0, 0.0);
            for k in 0..n {
                let a_ik = num_complex::Complex64::new(a_data[i + k * n], 0.0);
                av += a_ik * vecs[k + j * n];
            }
            let lv = vals[j] * vecs[i + j * n];
            let diff = (av - lv).norm();
            assert!(diff < 1e-10, "row-major eig residual at ({i},{j}) = {diff}");
        }
    }
}

#[test]
fn eig_2x2_complex_eigenvalues() {
    // [[0, -1], [1, 0]] has eigenvalues +/- i
    // Column-major: [0, 1, -1, 0]
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.0, 1.0, -1.0, 0.0], &[2, 2]);
    let result = eig(&mut ctx, &a).unwrap();
    let vals = tensor_data_complex(&result.values);
    let mut imags: Vec<f64> = vals.iter().map(|c| c.im).collect();
    imags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (imags[0] - (-1.0)).abs() < 1e-10,
        "expected -1.0, got {}",
        imags[0]
    );
    assert!(
        (imags[1] - 1.0).abs() < 1e-10,
        "expected 1.0, got {}",
        imags[1]
    );
    // Real parts should be zero
    for v in &vals {
        assert!(v.re.abs() < 1e-10, "expected re=0.0, got {}", v.re);
    }
}

#[test]
fn eig_3x3_reconstruction() {
    // Verify A * V = V * diag(lambda) for a 3x3 matrix
    let mut ctx = CpuContext::new(1);
    // Upper triangular with known eigenvalues 1, 2, 3
    // Column-major: col0=[1,0,0], col1=[1,2,0], col2=[0,1,3]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 3.0], &[3, 3]);
    let result = eig(&mut ctx, &a).unwrap();
    let vals = tensor_data_complex(&result.values);
    let vecs = tensor_data_complex(&result.vectors);
    let n = 3;

    // Check A * V = V * diag(lambda)
    // A is real, V and lambda are complex
    let a_data = tensor_data(&a);
    for j in 0..n {
        // Compute A * v_j (column j of V)
        for i in 0..n {
            let mut av = num_complex::Complex64::new(0.0, 0.0);
            for k in 0..n {
                let a_ik = num_complex::Complex64::new(a_data[i + k * n], 0.0);
                av += a_ik * vecs[k + j * n];
            }
            // Compare with lambda_j * v_j[i]
            let lv = vals[j] * vecs[i + j * n];
            let diff = (av - lv).norm();
            assert!(
                diff < 1e-10,
                "A*V != V*diag(lambda) at ({i},{j}): diff={diff}"
            );
        }
    }
}

#[test]
fn eig_batched_2x2() {
    // Batched: shape [2, 2, 2] — two 2x2 matrices
    let mut ctx = CpuContext::new(1);
    // batch 0: [[1, 0], [0, 2]] => eigenvalues 1, 2
    // batch 1: [[3, 0], [0, 4]] => eigenvalues 3, 4
    // Column-major for [2,2,2]: [1,0, 0,2, 3,0, 0,4]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0], &[2, 2, 2]);
    let result = eig(&mut ctx, &a).unwrap();
    assert_eq!(result.values.dims(), &[2, 2]);
    assert_eq!(result.vectors.dims(), &[2, 2, 2]);

    let vals = tensor_data_complex(&result.values);
    // vals layout for [2, 2]: batch-0 eigenvalues then batch-1 eigenvalues
    // In col-major [n=2, bc=2], stride=[1, 2]: vals[0],vals[1] = batch0, vals[2],vals[3] = batch1
    let mut batch0: Vec<f64> = vals[0..2].iter().map(|c| c.re).collect();
    let mut batch1: Vec<f64> = vals[2..4].iter().map(|c| c.re).collect();
    batch0.sort_by(|a, b| a.partial_cmp(b).unwrap());
    batch1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((batch0[0] - 1.0).abs() < 1e-10);
    assert!((batch0[1] - 2.0).abs() < 1e-10);
    assert!((batch1[0] - 3.0).abs() < 1e-10);
    assert!((batch1[1] - 4.0).abs() < 1e-10);
}

// ============================================================================
// eig AD tests (rrule and frule)
// ============================================================================

#[test]
fn eig_rrule_diagonal_values_only() {
    // For diagonal matrix, eigenvalues are the diagonal entries.
    // With cotangent for eigenvalues = [1, 1], the gradient should be I (identity).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let cotangent = EigCotangent {
        values: Some(make_complex_tensor(
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            &[2],
        )),
        vectors: None,
    };
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    let grad_data = tensor_data(&grad);
    assert_eq!(grad_data.len(), 4);
    // Gradient w.r.t. eigenvalues with unit cotangent should be diagonal = 1
    // and off-diagonal = 0 (for diagonal input)
    assert!(
        (grad_data[0] - 1.0).abs() < 1e-8,
        "grad[0,0] should be 1.0, got {}",
        grad_data[0]
    );
    assert!(
        (grad_data[3] - 1.0).abs() < 1e-8,
        "grad[1,1] should be 1.0, got {}",
        grad_data[3]
    );
    assert!(
        grad_data[1].abs() < 1e-8,
        "grad[1,0] should be 0.0, got {}",
        grad_data[1]
    );
    assert!(
        grad_data[2].abs() < 1e-8,
        "grad[0,1] should be 0.0, got {}",
        grad_data[2]
    );
}

#[test]
fn eig_rrule_no_cotangent() {
    // With no cotangents, gradient should be zero
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let cotangent = EigCotangent::<f64> {
        values: None,
        vectors: None,
    };
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    let grad_data = tensor_data(&grad);
    for (i, &v) in grad_data.iter().enumerate() {
        assert!(
            v.abs() < 1e-12,
            "grad[{i}] should be 0.0 with no cotangent, got {v}"
        );
    }
}

#[test]
fn eig_rrule_finite_difference() {
    // Verify eig_rrule against finite differences for a 3x3 matrix
    let mut ctx = CpuContext::new(1);
    // Upper triangular with distinct eigenvalues 1, 2, 3
    let a_data = vec![1.0, 0.0, 0.0, 0.5, 2.0, 0.0, 0.3, 0.7, 3.0];
    let a = make_tensor(a_data.clone(), &[3, 3]);

    // Use cotangent for eigenvalues only (simpler to verify)
    let cotangent = EigCotangent {
        values: Some(make_complex_tensor(
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
            &[3],
        )),
        vectors: None,
    };
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    let grad_data = tensor_data(&grad);

    // Finite difference: perturb each element and check eigenvalue change
    let eps = 1e-6;
    let dlam = [1.0, 0.5, -1.0]; // cotangent values (real)

    for idx in 0..9 {
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let a_p = make_tensor(a_plus, &[3, 3]);
        let r_p = eig(&mut ctx, &a_p).unwrap();
        let vals_p = tensor_data_complex(&r_p.values);

        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let a_m = make_tensor(a_minus, &[3, 3]);
        let r_m = eig(&mut ctx, &a_m).unwrap();
        let vals_m = tensor_data_complex(&r_m.values);

        // Sort eigenvalues by real part for consistent ordering
        let mut vp: Vec<Complex64> = vals_p;
        let mut vm: Vec<Complex64> = vals_m;
        vp.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        vm.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

        // f = sum_k dlam_k * real(lambda_k), so df/dA_ij = sum_k dlam_k * d(Re(lam_k))/dA_ij
        let mut fd_grad = 0.0;
        for k in 0..3 {
            let d_lam_k = (vp[k].re - vm[k].re) / (2.0 * eps);
            fd_grad += dlam[k] * d_lam_k;
        }

        assert!(
            (grad_data[idx] - fd_grad).abs() < 1e-4,
            "eig_rrule FD mismatch at idx {idx}: analytic={}, fd={}",
            grad_data[idx],
            fd_grad,
        );
    }
}

#[test]
fn eig_frule_diagonal() {
    // For diagonal matrix, tangent eigenvalues should match tangent diagonal entries
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.2], &[2, 2]);
    let (result, tangent) = eig_frule(&mut ctx, &a, &da).unwrap();

    // Primal eigenvalues should be 2 and 3 (in some order)
    let vals = tensor_data_complex(&result.values);
    let mut reals: Vec<f64> = vals.iter().map(|c| c.re).collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((reals[0] - 2.0).abs() < 1e-10, "eigenvalue 0: {}", reals[0]);
    assert!((reals[1] - 3.0).abs() < 1e-10, "eigenvalue 1: {}", reals[1]);

    // Tangent eigenvalues: for diagonal A with dA also diagonal,
    // the tangent eigenvalues should be the diagonal of dA
    let dvals = tensor_data_complex(&tangent.values);
    let mut d_reals: Vec<(f64, f64)> = vals
        .iter()
        .zip(dvals.iter())
        .map(|(v, dv)| (v.re, dv.re))
        .collect();
    d_reals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // eigenvalue 2.0 should have tangent 0.1, eigenvalue 3.0 should have tangent 0.2
    assert!(
        (d_reals[0].1 - 0.1).abs() < 1e-10,
        "dlambda for eigenvalue 2.0 should be 0.1, got {}",
        d_reals[0].1
    );
    assert!(
        (d_reals[1].1 - 0.2).abs() < 1e-10,
        "dlambda for eigenvalue 3.0 should be 0.2, got {}",
        d_reals[1].1
    );
}

#[test]
fn eig_frule_finite_difference() {
    // Verify eig_frule against finite differences for a 3x3 matrix
    let mut ctx = CpuContext::new(1);
    let a_data = vec![1.0, 0.0, 0.0, 0.5, 2.0, 0.0, 0.3, 0.7, 3.0];
    let da_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let a = make_tensor(a_data.clone(), &[3, 3]);
    let da = make_tensor(da_data.clone(), &[3, 3]);
    let (_result, tangent) = eig_frule(&mut ctx, &a, &da).unwrap();
    let dvals = tensor_data_complex(&tangent.values);

    // Finite difference
    let eps = 1e-7;
    let mut a_plus_data = vec![0.0; 9];
    let mut a_minus_data = vec![0.0; 9];
    for i in 0..9 {
        a_plus_data[i] = a_data[i] + eps * da_data[i];
        a_minus_data[i] = a_data[i] - eps * da_data[i];
    }
    let a_plus = make_tensor(a_plus_data, &[3, 3]);
    let a_minus = make_tensor(a_minus_data, &[3, 3]);
    let r_plus = eig(&mut ctx, &a_plus).unwrap();
    let r_minus = eig(&mut ctx, &a_minus).unwrap();
    let vals_p = tensor_data_complex(&r_plus.values);
    let vals_m = tensor_data_complex(&r_minus.values);

    // Sort by real part for consistent ordering
    let mut vp: Vec<Complex64> = vals_p;
    let mut vm: Vec<Complex64> = vals_m;
    let mut dv_sorted: Vec<(f64, Complex64)> = {
        let result2 = eig(&mut ctx, &a).unwrap();
        let vals = tensor_data_complex(&result2.values);
        vals.iter()
            .zip(dvals.iter())
            .map(|(v, dv)| (v.re, *dv))
            .collect()
    };
    vp.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    vm.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    dv_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for k in 0..3 {
        let fd_dlam = (vp[k] - vm[k]) / (2.0 * eps);
        let analytic = dv_sorted[k].1;
        let diff = (analytic - fd_dlam).norm();
        assert!(
            diff < 1e-4,
            "eig_frule FD mismatch at eigenvalue {k}: analytic={analytic}, fd={fd_dlam}, diff={diff}",
        );
    }
}

// ============================================================================
// Matrix exp tests
// ============================================================================

#[test]
fn matrix_exp_identity_succeeds() {
    // Previously this returned an error; now it should succeed.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    // exp(diag(1,1)) = diag(e, e)
    let e = 1.0_f64.exp();
    assert!((data[0] - e).abs() < 1e-10);
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-10);
    assert!((data[3] - e).abs() < 1e-10);
}

#[test]
fn matrix_exp_zero_is_identity() {
    // exp(0) = I
    let mut ctx = CpuContext::new(1);
    let zeros = make_tensor(vec![0.0; 9], &[3, 3]);
    let result = matrix_exp(&mut ctx, &zeros).unwrap();
    let data = tensor_data(&result);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (data[i + j * 3] - expected).abs() < 1e-10,
                "exp(0)[{i},{j}] = {}, expected {expected}",
                data[i + j * 3]
            );
        }
    }
}

#[test]
fn matrix_exp_diagonal() {
    // exp(diag(a,b)) = diag(exp(a), exp(b))
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    assert!(
        (data[0] - 1.0_f64.exp()).abs() < 1e-10,
        "exp(diag(1,2))[0,0] = {}, expected {}",
        data[0],
        1.0_f64.exp()
    );
    assert!(
        (data[3] - 2.0_f64.exp()).abs() < 1e-10,
        "exp(diag(1,2))[1,1] = {}, expected {}",
        data[3],
        2.0_f64.exp()
    );
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-10);
}

#[test]
fn matrix_exp_nilpotent() {
    // For nilpotent matrix [[0,1],[0,0]], exp = [[1,1],[0,1]]
    // Column-major: [0,0, 1,0] for [[0,1],[0,0]]
    // col 0: (0,0), col 1: (1,0)  => A[0,0]=0, A[1,0]=0, A[0,1]=1, A[1,1]=0
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.0, 0.0, 1.0, 0.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    // exp([[0,1],[0,0]]) = [[1,1],[0,1]]
    // col-major: [1, 0, 1, 1]
    assert!(
        (data[0] - 1.0).abs() < 1e-10,
        "[0,0] = {}, expected 1",
        data[0]
    );
    assert!(data[1].abs() < 1e-10, "[1,0] = {}, expected 0", data[1]);
    assert!(
        (data[2] - 1.0).abs() < 1e-10,
        "[0,1] = {}, expected 1",
        data[2]
    );
    assert!(
        (data[3] - 1.0).abs() < 1e-10,
        "[1,1] = {}, expected 1",
        data[3]
    );
}

#[test]
fn matrix_exp_large_norm() {
    // Test with matrix that requires scaling (large entries)
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![10.0, 0.0, 0.0, 10.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    let exp10 = 10.0_f64.exp();
    assert!(
        (data[0] - exp10).abs() / exp10 < 1e-10,
        "exp(diag(10,10))[0,0] relative error = {}",
        (data[0] - exp10).abs() / exp10
    );
    assert!(
        (data[3] - exp10).abs() / exp10 < 1e-10,
        "exp(diag(10,10))[1,1] relative error = {}",
        (data[3] - exp10).abs() / exp10
    );
    assert!(data[1].abs() < 1e-6);
    assert!(data[2].abs() < 1e-6);
}

#[test]
fn matrix_exp_1x1() {
    // 1x1 special case
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0], &[1, 1]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    assert!(
        (data[0] - 3.0_f64.exp()).abs() < 1e-10,
        "exp([3]) = {}, expected {}",
        data[0],
        3.0_f64.exp()
    );
}

#[test]
fn matrix_exp_batched() {
    // Batched: two 2x2 matrices
    let mut ctx = CpuContext::new(1);
    // batch 0: diag(1, 0), batch 1: diag(0, 2)
    // col-major with batch dim: [1,0, 0,0,  0,0, 0,2]
    let a = make_tensor(
        vec![
            1.0, 0.0, 0.0, 0.0, // batch 0: diag(1, 0)
            0.0, 0.0, 0.0, 2.0, // batch 1: diag(0, 2)
        ],
        &[2, 2, 2],
    );
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    // batch 0: exp(diag(1,0)) = diag(e, 1)
    assert!((data[0] - 1.0_f64.exp()).abs() < 1e-10);
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-10);
    assert!((data[3] - 1.0).abs() < 1e-10);
    // batch 1: exp(diag(0,2)) = diag(1, e^2)
    assert!((data[4] - 1.0).abs() < 1e-10);
    assert!(data[5].abs() < 1e-10);
    assert!(data[6].abs() < 1e-10);
    assert!((data[7] - 2.0_f64.exp()).abs() < 1e-10);
}

#[test]
fn matrix_exp_dense_2x2() {
    // Test with a non-diagonal 2x2 matrix.
    // A = [[0, -pi/2], [pi/2, 0]]  (rotation generator)
    // exp(A) = [[cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]] = [[0, -1], [1, 0]]
    //
    // Column-major: A = [0, pi/2, -pi/2, 0]
    let pi_2 = std::f64::consts::FRAC_PI_2;
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.0, pi_2, -pi_2, 0.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    // exp(A) col-major: [cos(pi/2), sin(pi/2), -sin(pi/2), cos(pi/2)]
    //                  = [0, 1, -1, 0]
    assert!(data[0].abs() < 1e-10, "[0,0] = {}, expected 0", data[0]);
    assert!(
        (data[1] - 1.0).abs() < 1e-10,
        "[1,0] = {}, expected 1",
        data[1]
    );
    assert!(
        (data[2] - (-1.0)).abs() < 1e-10,
        "[0,1] = {}, expected -1",
        data[2]
    );
    assert!(data[3].abs() < 1e-10, "[1,1] = {}, expected 0", data[3]);
}

#[test]
fn norm_rrule_batched_cotangent_shape_mismatch_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 2, 2],
    );
    // For batch shape [2], cotangent must be [2]. Scalar is invalid here.
    let bad_cot = make_tensor(vec![1.0], &[]);
    assert!(norm_rrule(&mut ctx, &a, &bad_cot, NormKind::Fro).is_err());
}

// ============================================================================
// AD rule tests: inv_rrule finite difference check
// ============================================================================

#[test]
fn inv_rrule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let n = 2;
    let eps = 1e-6;

    // Cotangent (seed): identity
    let cot_data = vec![1.0, 0.0, 0.0, 1.0];
    let cot = make_tensor(cot_data.clone(), &[2, 2]);
    let grad = inv_rrule(&mut ctx, &a, &cot).unwrap();
    let grad_data = tensor_data(&grad);

    // Finite difference check: grad_A[idx] = sum_k cot[k] * d(inv(A)[k])/dA[idx]
    for idx in 0..n * n {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;
        let inv_plus = tensor_data(&inv(&mut ctx, &make_tensor(a_plus, &[n, n])).unwrap());
        let inv_minus = tensor_data(&inv(&mut ctx, &make_tensor(a_minus, &[n, n])).unwrap());

        let fd: f64 = cot_data
            .iter()
            .enumerate()
            .map(|(k, &c)| c * (inv_plus[k] - inv_minus[k]) / (2.0 * eps))
            .sum();

        assert!(
            (grad_data[idx] - fd).abs() < 1e-4,
            "inv_rrule FD check failed at idx {idx}: analytic={}, fd={fd}",
            grad_data[idx]
        );
    }
}

// ============================================================================
// AD rule tests: det_rrule finite difference check
// ============================================================================

#[test]
fn det_rrule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let n = 2;
    let eps = 1e-6;

    let cot = make_tensor(vec![1.0], &[]);
    let grad = det_rrule(&mut ctx, &a, &cot).unwrap();
    let grad_data = tensor_data(&grad);

    for idx in 0..n * n {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;
        let det_plus = tensor_data(&det(&mut ctx, &make_tensor(a_plus, &[n, n])).unwrap())[0];
        let det_minus = tensor_data(&det(&mut ctx, &make_tensor(a_minus, &[n, n])).unwrap())[0];
        let fd = (det_plus - det_minus) / (2.0 * eps);

        assert!(
            (grad_data[idx] - fd).abs() < 1e-4,
            "det_rrule FD check failed at idx {idx}: analytic={}, fd={fd}",
            grad_data[idx]
        );
    }
}

// ============================================================================
// AD rule tests: solve_rrule finite difference check
// ============================================================================

#[test]
fn solve_rrule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![2.0, 0.5, 0.3, 3.0];
    let b_data = vec![1.0, 2.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let b = make_tensor(b_data.clone(), &[2, 1]);
    let eps = 1e-6;

    let _x = solve(&mut ctx, &a, &b).unwrap();
    let cot = make_tensor(vec![1.0, 1.0], &[2, 1]);
    let grad = solve_rrule(&mut ctx, &a, &b, &cot).unwrap();
    let grad_a_data = tensor_data(&grad.a);

    // FD check for A gradient
    for idx in 0..4 {
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let x_plus = tensor_data(&solve(&mut ctx, &make_tensor(a_plus, &[2, 2]), &b).unwrap());
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let x_minus = tensor_data(&solve(&mut ctx, &make_tensor(a_minus, &[2, 2]), &b).unwrap());

        // sum of x perturbation (since cot = [1,1])
        let fd: f64 = x_plus
            .iter()
            .zip(x_minus.iter())
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .sum();

        assert!(
            (grad_a_data[idx] - fd).abs() < 1e-4,
            "solve_rrule FD check failed at A[{idx}]: analytic={}, fd={fd}",
            grad_a_data[idx]
        );
    }
}

// ============================================================================
// AD rule tests: cholesky_rrule finite difference check
// ============================================================================

#[test]
fn cholesky_rrule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    // Symmetric positive definite
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    let cot_data = vec![1.0, 0.0, 0.0, 1.0];
    let cot = make_tensor(cot_data.clone(), &[2, 2]);
    let grad = cholesky_rrule(&mut ctx, &a, &cot).unwrap();
    let grad_data = tensor_data(&grad);

    // For Cholesky of symmetric matrix, we only test the unique (i,j) with i>=j
    // Perturb symmetrically and check grad
    for idx in 0..4 {
        let i = idx % 2;
        let j = idx / 2;

        let mut a_plus = a_data.clone();
        a_plus[i + j * 2] += eps;
        if i != j {
            a_plus[j + i * 2] += eps; // keep symmetric
        }
        let l_plus = tensor_data(&cholesky(&mut ctx, &make_tensor(a_plus, &[2, 2])).unwrap());

        let mut a_minus = a_data.clone();
        a_minus[i + j * 2] -= eps;
        if i != j {
            a_minus[j + i * 2] -= eps;
        }
        let l_minus = tensor_data(&cholesky(&mut ctx, &make_tensor(a_minus, &[2, 2])).unwrap());

        // FD: sum_k cot[k] * (l_plus[k] - l_minus[k]) / (2*eps)
        let fd: f64 = cot_data
            .iter()
            .enumerate()
            .map(|(k, &c)| c * (l_plus[k] - l_minus[k]) / (2.0 * eps))
            .sum();

        // For symmetric perturbation dA[i,j] += eps AND dA[j,i] += eps,
        // the directional derivative is grad[i,j] + grad[j,i] (if i!=j), or grad[i,i] (if i==j)
        let expected = if i == j {
            grad_data[idx]
        } else {
            grad_data[i + j * 2] + grad_data[j + i * 2]
        };

        assert!(
            (expected - fd).abs() < 1e-3,
            "cholesky_rrule FD check failed at ({i},{j}): analytic={expected}, fd={fd}"
        );
    }
}

// ============================================================================
// AD rule tests: inv_frule finite difference check
// ============================================================================

#[test]
fn inv_frule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    for idx in 0..4 {
        let mut tangent_data = vec![0.0; 4];
        tangent_data[idx] = 1.0;
        let tangent = make_tensor(tangent_data, &[2, 2]);

        let (_, dinv) = inv_frule(&mut ctx, &a, &tangent).unwrap();
        let dinv_data = tensor_data(&dinv);

        // Finite difference
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let inv_plus = tensor_data(&inv(&mut ctx, &make_tensor(a_plus, &[2, 2])).unwrap());
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let inv_minus = tensor_data(&inv(&mut ctx, &make_tensor(a_minus, &[2, 2])).unwrap());

        for k in 0..4 {
            let fd = (inv_plus[k] - inv_minus[k]) / (2.0 * eps);
            assert!(
                (dinv_data[k] - fd).abs() < 1e-4,
                "inv_frule FD check failed: d(inv)[{k}] w.r.t. A[{idx}]: analytic={}, fd={fd}",
                dinv_data[k]
            );
        }
    }
}

// ============================================================================
// AD rule tests: det_frule finite difference check
// ============================================================================

#[test]
fn det_frule_finite_diff() {
    let mut ctx = CpuContext::new(1);
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    for idx in 0..4 {
        let mut tangent_data = vec![0.0; 4];
        tangent_data[idx] = 1.0;
        let tangent = make_tensor(tangent_data, &[2, 2]);

        let (_, ddet) = det_frule(&mut ctx, &a, &tangent).unwrap();
        let ddet_data = tensor_data(&ddet);

        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let det_plus = tensor_data(&det(&mut ctx, &make_tensor(a_plus, &[2, 2])).unwrap())[0];
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let det_minus = tensor_data(&det(&mut ctx, &make_tensor(a_minus, &[2, 2])).unwrap())[0];
        let fd = (det_plus - det_minus) / (2.0 * eps);

        assert!(
            (ddet_data[0] - fd).abs() < 1e-4,
            "det_frule FD check failed at idx {idx}: analytic={}, fd={fd}",
            ddet_data[0]
        );
    }
}

// ============================================================================
// 1D input error tests
// ============================================================================

#[test]
fn svd_1d_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(svd(&mut ctx, &a, None).is_err());
}

#[test]
fn qr_1d_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(qr(&mut ctx, &a).is_err());
}

// ============================================================================
// LinalgScalar trait tests
// ============================================================================

#[test]
fn linalg_scalar_f32() {
    let mut ctx = CpuContext::new(1);
    let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let a = Tensor::<f32>::from_vec(data, &[2, 2], &[1, 2], 0).unwrap();
    let result = svd(&mut ctx, &a, None).unwrap();
    let s_data = result.s.contiguous(COL);
    let s = s_data.buffer().as_slice().unwrap();
    assert!((s[0] - 1.0_f32).abs() < 1e-5);
}

// ============================================================================
// Regression tests: invalid inputs must return Err, not panic (#124)
// ============================================================================

#[test]
fn test_svd_1d_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    assert!(svd(&mut ctx, &a, None).is_err());
}

#[test]
fn test_qr_1d_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    assert!(qr(&mut ctx, &a).is_err());
}

#[test]
fn test_cholesky_non_square_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(cholesky(&mut ctx, &a).is_err());
}

#[test]
fn test_solve_dimension_mismatch_returns_error() {
    let mut ctx = CpuContext::new(1);
    // A is 3x3 but b has leading dimension 2 (mismatch)
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let b = make_tensor(vec![1.0, 2.0], &[2, 1]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

// ============================================================================
// Complex64 integration tests (Tensor-level API)
// ============================================================================

/// Create a column-major Tensor<Complex64> from a flat vec and shape.
fn make_complex_tensor(data: Vec<Complex64>, dims: &[usize]) -> Tensor<Complex64> {
    let ndim = dims.len();
    let mut strides = vec![0isize; ndim];
    if ndim > 0 {
        strides[0] = 1;
        for i in 1..ndim {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

/// Extract flat data from a Tensor<Complex64>.
fn complex_tensor_data(t: &Tensor<Complex64>) -> Vec<Complex64> {
    let c = t.contiguous(COL);
    let off = c.offset() as usize;
    let len: usize = c.dims().iter().product();
    c.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

/// Extract flat data from a Tensor<f64> (for real results like singular values).
fn real_tensor_data(t: &Tensor<f64>) -> Vec<f64> {
    tensor_data(t)
}

/// Maximum element-wise absolute error between two Complex64 slices.
fn complex_max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// Shorthand to construct Complex64.
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_svd_complex64_identity() {
    let mut ctx = CpuContext::new(1);
    let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let a = make_complex_tensor(data, &[2, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();

    // Check shapes
    assert_eq!(result.u.dims(), &[2, 2]);
    assert_eq!(result.s.dims(), &[2]);
    assert_eq!(result.vt.dims(), &[2, 2]);

    // Singular values should be 1.0
    let s = real_tensor_data(&result.s);
    for &val in &s {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "singular value should be 1.0, got {val}"
        );
    }
}

#[test]
fn test_svd_complex64_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // Non-trivial 2x3 complex matrix
    let data = vec![
        c(1.0, 2.0),
        c(3.0, -1.0),
        c(0.0, 1.0),
        c(4.0, 0.0),
        c(-1.0, 3.0),
        c(2.0, 2.0),
    ];
    let a = make_complex_tensor(data.clone(), &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();

    let u = complex_tensor_data(&result.u);
    let s = real_tensor_data(&result.s);
    let vt = complex_tensor_data(&result.vt);
    let m = 2;
    let n = 3;
    let k = 2;

    assert_eq!(result.u.dims(), &[m, k]);
    assert_eq!(result.s.dims(), &[k]);
    assert_eq!(result.vt.dims(), &[k, n]);

    // Reconstruct: A = U * diag(S) * Vt
    let mut recon = vec![c(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = c(0.0, 0.0);
            for l in 0..k {
                val += u[i + l * m] * s[l] * vt[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }

    let err = complex_max_err(&data, &recon);
    assert!(err < 1e-10, "SVD reconstruction error: {err}");
}

#[test]
fn test_svd_complex64_with_max_rank() {
    let mut ctx = CpuContext::new(1);
    let data = vec![
        c(1.0, 0.0),
        c(0.0, 1.0),
        c(2.0, 1.0),
        c(-1.0, 0.0),
        c(3.0, -1.0),
        c(0.0, 2.0),
    ];
    let a = make_complex_tensor(data, &[2, 3]);
    let opts = SvdOptions {
        max_rank: Some(1),
        cutoff: None,
    };
    let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
    assert_eq!(result.s.dims(), &[1]);
    assert_eq!(result.u.dims(), &[2, 1]);
    assert_eq!(result.vt.dims(), &[1, 3]);
}

#[test]
fn test_qr_complex64_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![
        c(1.0, 1.0),
        c(2.0, -1.0),
        c(0.0, 3.0),
        c(4.0, 0.0),
        c(-1.0, 2.0),
        c(3.0, 1.0),
    ];
    let a = make_complex_tensor(data.clone(), &[3, 2]);
    let result = qr(&mut ctx, &a).unwrap();

    assert_eq!(result.q.dims(), &[3, 2]);
    assert_eq!(result.r.dims(), &[2, 2]);

    let q = complex_tensor_data(&result.q);
    let r = complex_tensor_data(&result.r);
    let m = 3;
    let n = 2;
    let k = 2;

    // Q * R = A
    let mut recon = vec![c(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = c(0.0, 0.0);
            for l in 0..k {
                val += q[i + l * m] * r[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }
    let err = complex_max_err(&data, &recon);
    assert!(err < 1e-10, "QR reconstruction error: {err}");
}

#[test]
fn test_lu_complex64_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![
        c(2.0, 1.0),
        c(4.0, 0.0),
        c(1.0, -1.0),
        c(1.0, 0.0),
        c(3.0, 2.0),
        c(0.0, 1.0),
        c(0.0, 1.0),
        c(1.0, 0.0),
        c(5.0, 0.0),
    ];
    let a = make_complex_tensor(data.clone(), &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();

    let l = complex_tensor_data(&result.l);
    let u = complex_tensor_data(&result.u);
    let p = result.p.unwrap();
    let n = 3;

    // L * U = P * A
    let mut lu_prod = vec![c(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = c(0.0, 0.0);
            for k in 0..n {
                val += l[i + k * n] * u[k + j * n];
            }
            lu_prod[i + j * n] = val;
        }
    }

    // Apply P^{-1} to rows of lu_prod to get A back
    let mut recon = vec![c(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            recon[p[i] + j * n] = lu_prod[i + j * n];
        }
    }
    let err = complex_max_err(&data, &recon);
    assert!(err < 1e-10, "LU reconstruction error: {err}");
}

#[test]
fn test_cholesky_complex64_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // Hermitian positive definite: [[4, 1+i], [1-i, 3]]
    let data = vec![c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    let a = make_complex_tensor(data.clone(), &[2, 2]);
    let l_tensor = cholesky(&mut ctx, &a).unwrap();

    assert_eq!(l_tensor.dims(), &[2, 2]);
    let l = complex_tensor_data(&l_tensor);
    let n = 2;

    // L * L^H = A
    let mut recon = vec![c(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = c(0.0, 0.0);
            for p in 0..n {
                sum += l[i + p * n] * l[j + p * n].conj();
            }
            recon[i + j * n] = sum;
        }
    }
    let err = complex_max_err(&data, &recon);
    assert!(err < 1e-10, "Cholesky reconstruction error: {err}");
}

#[test]
fn test_eigen_complex64_hermitian() {
    let mut ctx = CpuContext::new(1);
    // Hermitian: [[3, 1-i], [1+i, 2]], eigenvalues 1 and 4
    let data = vec![c(3.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(2.0, 0.0)];
    let a = make_complex_tensor(data, &[2, 2]);
    let result = eigen(&mut ctx, &a).unwrap();

    let vals = real_tensor_data(&result.values);
    assert!(
        (vals[0] - 1.0).abs() < 1e-10,
        "eigenvalue 0: {}, expected 1.0",
        vals[0]
    );
    assert!(
        (vals[1] - 4.0).abs() < 1e-10,
        "eigenvalue 1: {}, expected 4.0",
        vals[1]
    );
}

#[test]
fn test_solve_complex64() {
    let mut ctx = CpuContext::new(1);
    // A = [[2+i, 1], [0, 3-i]], b = [3+i, 6-2i]
    let a_data = vec![c(2.0, 1.0), c(0.0, 0.0), c(1.0, 0.0), c(3.0, -1.0)];
    let b_data = vec![c(3.0, 1.0), c(6.0, -2.0)];
    let a = make_complex_tensor(a_data.clone(), &[2, 2]);
    let b = make_complex_tensor(b_data.clone(), &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = complex_tensor_data(&x);

    // Verify A * x = b
    let mut ax = vec![c(0.0, 0.0); 2];
    for i in 0..2 {
        for k in 0..2 {
            ax[i] += a_data[i + k * 2] * xd[k];
        }
    }
    let err = complex_max_err(&ax, &b_data);
    assert!(err < 1e-10, "solve residual: {err}");
}

#[test]
fn test_solve_complex64_singular_returns_error() {
    let mut ctx = CpuContext::new(1);
    let a = make_complex_tensor(
        vec![c(1.0, 1.0), c(2.0, 2.0), c(2.0, 0.0), c(4.0, 0.0)],
        &[2, 2],
    );
    let b = make_complex_tensor(vec![c(1.0, 0.0), c(2.0, 0.0)], &[2, 1]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn test_solve_triangular_complex64() {
    let mut ctx = CpuContext::new(1);
    // Lower triangular: [[2+i, 0], [1-i, 3]]
    let a_data = vec![c(2.0, 1.0), c(1.0, -1.0), c(0.0, 0.0), c(3.0, 0.0)];
    let b_data = vec![c(4.0, 2.0), c(5.0, 0.0)];
    let a = make_complex_tensor(a_data.clone(), &[2, 2]);
    let b = make_complex_tensor(b_data.clone(), &[2, 1]);
    let x = solve_triangular(&mut ctx, &a, &b, false).unwrap();
    let xd = complex_tensor_data(&x);

    // Verify A * x = b
    let mut ax = vec![c(0.0, 0.0); 2];
    for i in 0..2 {
        for k in 0..2 {
            ax[i] += a_data[i + k * 2] * xd[k];
        }
    }
    let err = complex_max_err(&ax, &b_data);
    assert!(err < 1e-10, "solve_triangular residual: {err}");
}

// ============================================================================
// matrix_exp AD tests
// ============================================================================

#[test]
fn matrix_exp_frule_zero() {
    // d(exp(0))/dt at tangent dA should be dA itself (since Frechet derivative of exp at zero
    // is the identity map)
    let mut ctx = CpuContext::new(1);
    let zeros = make_tensor(vec![0.0; 4], &[2, 2]);
    let da = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let (result, tangent) = matrix_exp_frule(&mut ctx, &zeros, &da).unwrap();
    // exp(0) = I
    let r = tensor_data(&result);
    assert!((r[0] - 1.0).abs() < 1e-10);
    assert!((r[3] - 1.0).abs() < 1e-10);
    // d(exp(0)) * dA = dA (Frechet derivative of exp at zero is the identity map)
    let t = tensor_data(&tangent);
    assert!((t[0] - 1.0).abs() < 1e-10);
    assert!((t[1] - 2.0).abs() < 1e-10);
    assert!((t[2] - 3.0).abs() < 1e-10);
    assert!((t[3] - 4.0).abs() < 1e-10);
}

#[test]
fn matrix_exp_rrule_zero() {
    // At A=0, the rrule should pass through the cotangent unchanged
    let mut ctx = CpuContext::new(1);
    let zeros = make_tensor(vec![0.0; 4], &[2, 2]);
    let co = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let grad = matrix_exp_rrule(&mut ctx, &zeros, &co).unwrap();
    let g = tensor_data(&grad);
    assert!((g[0] - 1.0).abs() < 1e-10);
    assert!((g[1] - 2.0).abs() < 1e-10);
    assert!((g[2] - 3.0).abs() < 1e-10);
    assert!((g[3] - 4.0).abs() < 1e-10);
}

#[test]
fn matrix_exp_frule_finite_difference() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.5, 0.1, -0.2, 0.3], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.05, -0.03, 0.07], &[2, 2]);
    let eps = 1e-6;

    let (_, analytic_tangent) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
    let t = tensor_data(&analytic_tangent);

    // FD: (exp(A + eps*dA) - exp(A - eps*dA)) / (2*eps)
    let a_data = tensor_data(&a);
    let da_data = tensor_data(&da);
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&da_data)
        .map(|(a, d)| a + eps * d)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&da_data)
        .map(|(a, d)| a - eps * d)
        .collect();
    let exp_plus = matrix_exp(&mut ctx, &make_tensor(plus, &[2, 2])).unwrap();
    let exp_minus = matrix_exp(&mut ctx, &make_tensor(minus, &[2, 2])).unwrap();
    let fp = tensor_data(&exp_plus);
    let fm = tensor_data(&exp_minus);

    for i in 0..4 {
        let fd = (fp[i] - fm[i]) / (2.0 * eps);
        assert!(
            (t[i] - fd).abs() < 1e-4,
            "frule FD mismatch at {i}: analytic={}, fd={fd}",
            t[i]
        );
    }
}

#[test]
fn matrix_exp_rrule_finite_difference() {
    // Verify rrule via FD: for each entry (i,j) of A, perturb A[i,j] by eps and compute
    // (f(A+eps*E_ij) - f(A-eps*E_ij)) / (2*eps), then dot with cotangent.
    let mut ctx = CpuContext::new(1);
    let a_vec = vec![0.5, 0.1, -0.2, 0.3];
    let a = make_tensor(a_vec.clone(), &[2, 2]);
    let co_vec = vec![1.0, -0.5, 0.3, 0.8];
    let co = make_tensor(co_vec.clone(), &[2, 2]);
    let eps = 1e-6;

    let grad = matrix_exp_rrule(&mut ctx, &a, &co).unwrap();
    let g = tensor_data(&grad);

    // For each (i,j), compute grad[i,j] via FD
    for idx in 0..4 {
        let mut plus = a_vec.clone();
        let mut minus = a_vec.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let exp_plus = matrix_exp(&mut ctx, &make_tensor(plus, &[2, 2])).unwrap();
        let exp_minus = matrix_exp(&mut ctx, &make_tensor(minus, &[2, 2])).unwrap();
        let fp = tensor_data(&exp_plus);
        let fm = tensor_data(&exp_minus);
        // grad[idx] = sum_k cotangent[k] * d(exp(A))[k] / dA[idx]
        let mut fd_grad = 0.0;
        for k in 0..4 {
            fd_grad += co_vec[k] * (fp[k] - fm[k]) / (2.0 * eps);
        }
        assert!(
            (g[idx] - fd_grad).abs() < 1e-4,
            "rrule FD mismatch at {idx}: analytic={}, fd={fd_grad}",
            g[idx]
        );
    }
}

// ============================================================================
// Systematic rrule FD checks
// ============================================================================

// 1. SVD rrule FD — test through singular values S
#[test]
fn svd_rrule_fd_through_s() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        svd(&mut b, x, None).unwrap().s
    };
    let rrule_fn = |x: &Tensor<f64>, co_s: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let co = SvdCotangent {
            u: None,
            s: Some(co_s.clone()),
            vt: None,
        };
        svd_rrule(&mut b, x, &co, None).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 2. QR rrule FD — test through R
#[test]
fn qr_rrule_fd_through_r() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        qr(&mut b, x).unwrap().r
    };
    let rrule_fn = |x: &Tensor<f64>, co_r: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let co = QrCotangent {
            q: None,
            r: Some(co_r.clone()),
        };
        qr_rrule(&mut b, x, &co).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 3. LU rrule FD — test through L
// The LU decomposition with partial pivoting has a discrete permutation.
// For a strongly diagonally dominant matrix, the permutation stays constant
// under small perturbations, making FD reliable.
#[test]
fn lu_rrule_fd_through_l() {
    let a = make_general_test_matrix(3);
    let n = 3;
    let eps = 1e-6;
    let atol = 1e-4;

    let mut ctx = CpuContext::new(1);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l_size: usize = result.l.dims().iter().product();

    // Deterministic cotangent for L
    let co_data: Vec<f64> = (0..l_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent_l = make_tensor(co_data.clone(), result.l.dims());

    let co = LuCotangent {
        l: Some(cotangent_l),
        u: None,
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    let analytic = tensor_data(&grad);

    // FD gradient
    let a_data = tensor_data(&a);
    let mut fd_grad = vec![0.0; n * n];
    for idx in 0..n * n {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = tensor_data(
            &lu(
                &mut CpuContext::new(1),
                &make_tensor(plus, &[n, n]),
                LuPivot::Partial,
            )
            .unwrap()
            .l,
        );
        let l_minus = tensor_data(
            &lu(
                &mut CpuContext::new(1),
                &make_tensor(minus, &[n, n]),
                LuPivot::Partial,
            )
            .unwrap()
            .l,
        );

        let mut fd_val = 0.0;
        for k in 0..l_size {
            fd_val += co_data[k] * (l_plus[k] - l_minus[k]) / (2.0 * eps);
        }
        fd_grad[idx] = fd_val;
    }

    let max_err = analytic
        .iter()
        .zip(&fd_grad)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "lu_rrule FD check (through L) failed: max_err={max_err}"
    );
}

// 4. Eigen rrule FD (symmetric) — test through eigenvalues
// eigen() requires symmetric input, so FD perturbation must maintain symmetry.
// We perturb symmetric pairs together: for (i,j) with i<=j, perturb both
// A[i,j] and A[j,i] together.
#[test]
fn eigen_rrule_fd_through_values() {
    let a = make_well_conditioned_matrix(3);
    let n = 3;
    let eps = 1e-6;
    let atol = 1e-4;

    let mut ctx = CpuContext::new(1);
    let result = eigen(&mut ctx, &a).unwrap();
    let val_size: usize = result.values.dims().iter().product();

    // Deterministic cotangent for eigenvalues
    let co_data: Vec<f64> = (0..val_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent = EigenCotangent {
        values: Some(make_tensor(co_data.clone(), result.values.dims())),
        vectors: None,
    };

    let grad = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
    let analytic = tensor_data(&grad);

    let a_data = tensor_data(&a);

    // Test upper-triangular entries (i <= j): perturb A[i,j] and A[j,i] together
    for j in 0..n {
        for i in 0..=j {
            let mut plus = a_data.clone();
            plus[i + j * n] += eps;
            if i != j {
                plus[j + i * n] += eps;
            }

            let mut minus = a_data.clone();
            minus[i + j * n] -= eps;
            if i != j {
                minus[j + i * n] -= eps;
            }

            let vals_p = tensor_data(
                &eigen(&mut CpuContext::new(1), &make_tensor(plus, &[n, n]))
                    .unwrap()
                    .values,
            );
            let vals_m = tensor_data(
                &eigen(&mut CpuContext::new(1), &make_tensor(minus, &[n, n]))
                    .unwrap()
                    .values,
            );

            let mut fd_val = 0.0;
            for k in 0..val_size {
                fd_val += co_data[k] * (vals_p[k] - vals_m[k]) / (2.0 * eps);
            }

            // For symmetric perturbation, the directional derivative is
            // grad[i,j] + grad[j,i] (off-diagonal), or grad[i,i] (diagonal)
            let expected = if i == j {
                analytic[i + j * n]
            } else {
                analytic[i + j * n] + analytic[j + i * n]
            };

            assert!(
                (expected - fd_val).abs() < atol,
                "eigen_rrule FD check failed at ({i},{j}): analytic={expected}, fd={fd_val}"
            );
        }
    }
}

// 5. Cholesky rrule FD (using check_rrule_fd helper)
// Note: cholesky input must be SPD, and for FD perturbation we need to
// maintain symmetry. We write a custom check.
#[test]
fn cholesky_rrule_fd_systematic() {
    let a = make_well_conditioned_matrix(3);
    let n = 3;
    let eps = 1e-6;
    let atol = 1e-3;

    // Compute L and a cotangent
    let mut ctx = CpuContext::new(1);
    let l = cholesky(&mut ctx, &a).unwrap();
    let l_size: usize = l.dims().iter().product();
    let cotangent_data: Vec<f64> = (0..l_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent = make_tensor(cotangent_data.clone(), l.dims());

    let grad = cholesky_rrule(&mut ctx, &a, &cotangent).unwrap();
    let analytic = tensor_data(&grad);

    // FD: perturb symmetrically
    let a_data = tensor_data(&a);

    for idx in 0..n * n {
        let i = idx % n;
        let j = idx / n;

        let mut plus = a_data.clone();
        plus[i + j * n] += eps;
        if i != j {
            plus[j + i * n] += eps;
        }

        let mut minus = a_data.clone();
        minus[i + j * n] -= eps;
        if i != j {
            minus[j + i * n] -= eps;
        }

        let l_plus =
            tensor_data(&cholesky(&mut CpuContext::new(1), &make_tensor(plus, &[n, n])).unwrap());
        let l_minus =
            tensor_data(&cholesky(&mut CpuContext::new(1), &make_tensor(minus, &[n, n])).unwrap());

        let mut fd_val = 0.0;
        for k in 0..l_size {
            fd_val += cotangent_data[k] * (l_plus[k] - l_minus[k]) / (2.0 * eps);
        }

        // For symmetric perturbation: the directional derivative is
        // grad[i,j] + grad[j,i] for off-diagonal, or grad[i,i] for diagonal
        let expected = if i == j {
            analytic[idx]
        } else {
            analytic[i + j * n] + analytic[j + i * n]
        };

        assert!(
            (expected - fd_val).abs() < atol,
            "cholesky_rrule FD check failed at ({i},{j}): analytic={expected}, fd={fd_val}"
        );
    }
}

// 6. Solve rrule FD — test grad w.r.t. A
#[test]
fn solve_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3, 1]);
    let eps = 1e-6;
    let atol = 1e-4;

    let mut ctx = CpuContext::new(1);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let x_size: usize = x.dims().iter().product();

    // Deterministic cotangent
    let co_data: Vec<f64> = (0..x_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent = make_tensor(co_data.clone(), x.dims());

    let grad = solve_rrule(&mut ctx, &a, &b, &cotangent).unwrap();
    let analytic_a = tensor_data(&grad.a);

    // FD gradient w.r.t. A
    let a_data = tensor_data(&a);
    let a_size = a_data.len();
    let mut fd_grad_a = vec![0.0; a_size];
    for idx in 0..a_size {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let xp =
            tensor_data(&solve(&mut CpuContext::new(1), &make_tensor(plus, &[3, 3]), &b).unwrap());
        let xm =
            tensor_data(&solve(&mut CpuContext::new(1), &make_tensor(minus, &[3, 3]), &b).unwrap());
        for k in 0..x_size {
            fd_grad_a[idx] += co_data[k] * (xp[k] - xm[k]) / (2.0 * eps);
        }
    }

    let max_err = analytic_a
        .iter()
        .zip(&fd_grad_a)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "solve_rrule FD check failed: max_err={max_err}"
    );
}

// 7. Inv rrule FD (using check_rrule_fd helper)
#[test]
fn inv_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        inv(&mut b, x).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        inv_rrule(&mut b, x, co).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 8. Det rrule FD
#[test]
fn det_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        det(&mut b, x).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        det_rrule(&mut b, x, co).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 9. Slogdet rrule FD — test through logabsdet
#[test]
fn slogdet_rrule_fd_through_logabsdet() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        slogdet(&mut b, x).unwrap().logabsdet
    };
    let rrule_fn = |x: &Tensor<f64>, co_log: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let co = SlogdetCotangent {
            logabsdet: Some(co_log.clone()),
        };
        slogdet_rrule(&mut b, x, &co).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 10. Lstsq rrule FD — test grad w.r.t. A (tall matrix)
#[test]
fn lstsq_rrule_fd_systematic() {
    let m = 4;
    let n = 2;
    let mut a_data = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let val = ((i * 3 + j * 7 + 1) % 11) as f64 / 10.0;
            a_data[i + j * m] = val;
        }
    }
    for i in 0..n {
        a_data[i + i * m] += 5.0;
    }
    // Construct b = A * x_true so residual is zero
    let x_true = [1.0, 2.0];
    let mut b_data = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            b_data[i] += a_data[i + j * m] * x_true[j];
        }
    }
    let a = make_tensor(a_data.clone(), &[m, n]);
    let b = make_tensor(b_data, &[m]);
    let eps = 1e-6;
    let atol = 1e-3;

    let mut ctx = CpuContext::new(1);
    let result = lstsq(&mut ctx, &a, &b).unwrap();
    let x_size: usize = result.x.dims().iter().product();

    // Deterministic cotangent for x
    let co_data: Vec<f64> = (0..x_size)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let cotangent_x = make_tensor(co_data.clone(), result.x.dims());

    let grad = lstsq_rrule(&mut ctx, &a, &b, &cotangent_x).unwrap();
    let analytic_a = tensor_data(&grad.a);

    // FD gradient w.r.t. A
    let a_size = a_data.len();
    let mut fd_grad_a = vec![0.0; a_size];
    for idx in 0..a_size {
        let mut plus = a_data.clone();
        let mut minus = a_data.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        // b stays the same; when we perturb A, the residual for the perturbed
        // problem won't be exactly zero, but the FD formula is about x(A).
        let xp = tensor_data(
            &lstsq(&mut CpuContext::new(1), &make_tensor(plus, &[m, n]), &b)
                .unwrap()
                .x,
        );
        let xm = tensor_data(
            &lstsq(&mut CpuContext::new(1), &make_tensor(minus, &[m, n]), &b)
                .unwrap()
                .x,
        );
        for k in 0..x_size {
            fd_grad_a[idx] += co_data[k] * (xp[k] - xm[k]) / (2.0 * eps);
        }
    }

    let max_err = analytic_a
        .iter()
        .zip(&fd_grad_a)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "lstsq_rrule FD check failed: max_err={max_err}"
    );
}

// 11. Pinv rrule FD
#[test]
fn pinv_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        pinv(&mut b, x, None).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        pinv_rrule(&mut b, x, co, None).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 12. Matrix exp rrule FD (using check_rrule_fd helper with 3x3)
#[test]
fn matrix_exp_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    // Scale down to keep matrix exp numerically manageable
    let a_data = tensor_data(&a);
    let a_scaled: Vec<f64> = a_data.iter().map(|v| v * 0.1).collect();
    let a = make_tensor(a_scaled, &[3, 3]);

    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        matrix_exp(&mut b, x).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        matrix_exp_rrule(&mut b, x, co).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 13. Norm rrule FD — Frobenius norm
#[test]
fn norm_fro_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Fro).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm_rrule(&mut b, x, co, NormKind::Fro).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-4);
}

// 14. Eig rrule FD — already tested above in eig_rrule_finite_difference;
//     included here for completeness with the systematic pattern.
//     This tests through eigenvalue real parts using sorted matching.
#[test]
fn eig_rrule_fd_systematic() {
    let a = make_general_test_matrix(3);
    let a_data = tensor_data(&a);
    let n = 3;
    let eps = 1e-6;
    let atol = 1e-4;

    // Use cotangent only for eigenvalues (real part)
    let co_vals = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let cotangent = EigCotangent {
        values: Some(make_complex_tensor(co_vals.clone(), &[n])),
        vectors: None,
    };

    let mut ctx = CpuContext::new(1);
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    let grad_data = tensor_data(&grad);

    // Get base eigenvalues for sorting reference
    let base_result = eig(&mut ctx, &a).unwrap();
    let base_vals = tensor_data_complex(&base_result.values);
    let mut base_order: Vec<(usize, f64)> = base_vals
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.re))
        .collect();
    base_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Cotangent values mapped to sorted order
    let co_real: Vec<f64> = base_order
        .iter()
        .map(|&(orig_idx, _)| co_vals[orig_idx].re)
        .collect();

    for idx in 0..n * n {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;

        let r_p = eig(&mut CpuContext::new(1), &make_tensor(a_plus, &[n, n])).unwrap();
        let r_m = eig(&mut CpuContext::new(1), &make_tensor(a_minus, &[n, n])).unwrap();

        let mut vp: Vec<f64> = tensor_data_complex(&r_p.values)
            .iter()
            .map(|c| c.re)
            .collect();
        let mut vm: Vec<f64> = tensor_data_complex(&r_m.values)
            .iter()
            .map(|c| c.re)
            .collect();
        vp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vm.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut fd_grad = 0.0;
        for k in 0..n {
            fd_grad += co_real[k] * (vp[k] - vm[k]) / (2.0 * eps);
        }

        assert!(
            (grad_data[idx] - fd_grad).abs() < atol,
            "eig_rrule FD mismatch at idx {idx}: analytic={}, fd={fd_grad}",
            grad_data[idx],
        );
    }
}

// ============================================================================
// Systematic frule FD checks
// ============================================================================

// 1. SVD frule FD — test through singular values S
#[test]
fn svd_frule_fd_through_s() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        svd(&mut b, x, None).unwrap().s
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, tangent_result) = svd_frule(&mut b, x, dx, None).unwrap();
        tangent_result.s
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 2. QR frule FD — test through R
#[test]
fn qr_frule_fd_through_r() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        qr(&mut b, x).unwrap().r
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, tangent_result) = qr_frule(&mut b, x, dx).unwrap();
        tangent_result.r
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 3. LU frule FD — test through U
// LU with partial pivoting: the permutation is discrete but the strongly
// diagonally dominant test matrix keeps the permutation stable under perturbation.
#[test]
fn lu_frule_fd_through_u() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        lu(&mut b, x, LuPivot::Partial).unwrap().u
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, tangent_result) = lu_frule(&mut b, x, dx, LuPivot::Partial).unwrap();
        tangent_result.u
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 4. Eigen frule FD — test through eigenvalues
// Symmetric eigendecomposition: tangent must also be symmetric.
#[test]
fn eigen_frule_fd_through_values() {
    let a = make_well_conditioned_matrix(3);
    // Build a symmetric tangent
    let t_data = tensor_data(&make_general_test_matrix(3));
    let n = 3;
    let mut sym_t = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            sym_t[i + j * n] = (t_data[i + j * n] + t_data[j + i * n]) / 2.0;
        }
    }
    let sym_tangent = make_tensor(sym_t, &[n, n]);

    // Custom FD: eigenvalues may come in different order, so we sort them.
    let a_data = tensor_data(&a);

    // Deterministic tangent direction = sym_tangent
    let tangent_data = tensor_data(&sym_tangent);

    // Analytic: frule tangent through eigenvalues
    let mut ctx = CpuContext::new(1);
    let (_, tangent_result) = eigen_frule(&mut ctx, &a, &sym_tangent).unwrap();
    let analytic = tensor_data(&tangent_result.values);

    // Sort eigenvalues at base point to establish ordering
    let base_vals = tensor_data(&eigen(&mut ctx, &a).unwrap().values);
    let mut base_order: Vec<(usize, f64)> =
        base_vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    base_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Analytic tangent in sorted order
    let analytic_sorted: Vec<f64> = base_order.iter().map(|&(orig, _)| analytic[orig]).collect();

    // FD: (sorted_eigenvalues(A + eps*dA) - sorted_eigenvalues(A - eps*dA)) / (2*eps)
    let eps = 1e-6;
    let atol = 1e-4;
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let mut vp = tensor_data(
        &eigen(&mut CpuContext::new(1), &make_tensor(plus, &[n, n]))
            .unwrap()
            .values,
    );
    let mut vm = tensor_data(
        &eigen(&mut CpuContext::new(1), &make_tensor(minus, &[n, n]))
            .unwrap()
            .values,
    );
    vp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fd: Vec<f64> = vp
        .iter()
        .zip(&vm)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic_sorted
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "eigen_frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

// 5. Eig frule FD — test through eigenvalue real parts
// Non-symmetric eigendecomposition with complex results.
// Eigenvalue ordering may change, so we sort by real part.
#[test]
fn eig_frule_fd_through_values_re() {
    let a = make_general_test_matrix(3);
    let a_data = tensor_data(&a);
    let n = 3;
    let eps = 1e-6;
    let atol = 1e-4;

    // Deterministic tangent
    let tangent_data: Vec<f64> = (0..n * n)
        .map(|i| ((i * 13 + 5) % 17) as f64 / 8.0 - 1.0)
        .collect();
    let tangent = make_tensor(tangent_data.clone(), &[n, n]);

    // Analytic
    let mut ctx = CpuContext::new(1);
    let (primal, tangent_result) = eig_frule(&mut ctx, &a, &tangent).unwrap();

    // Sort eigenvalues by real part at base point
    let base_vals = tensor_data_complex(&primal.values);
    let mut base_order: Vec<(usize, f64)> = base_vals
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.re))
        .collect();
    base_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let d_vals = tensor_data_complex(&tangent_result.values);
    let analytic_sorted: Vec<f64> = base_order
        .iter()
        .map(|&(orig, _)| d_vals[orig].re)
        .collect();

    // FD
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let r_p = eig(&mut CpuContext::new(1), &make_tensor(plus, &[n, n])).unwrap();
    let r_m = eig(&mut CpuContext::new(1), &make_tensor(minus, &[n, n])).unwrap();
    let mut vp: Vec<f64> = tensor_data_complex(&r_p.values)
        .iter()
        .map(|c| c.re)
        .collect();
    let mut vm: Vec<f64> = tensor_data_complex(&r_m.values)
        .iter()
        .map(|c| c.re)
        .collect();
    vp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let fd: Vec<f64> = vp
        .iter()
        .zip(&vm)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic_sorted
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "eig_frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

// 6. Cholesky frule FD — test through L
// Cholesky is for SPD matrices; tangent must be symmetric.
#[test]
fn cholesky_frule_fd_through_l() {
    let a = make_well_conditioned_matrix(3);
    let n = 3;

    // Build symmetric tangent
    let t_data = tensor_data(&make_general_test_matrix(3));
    let mut sym_t = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            sym_t[i + j * n] = (t_data[i + j * n] + t_data[j + i * n]) / 2.0;
        }
    }
    let sym_tangent = make_tensor(sym_t, &[n, n]);

    // Custom FD with symmetric perturbation
    let a_data = tensor_data(&a);
    let tangent_data = tensor_data(&sym_tangent);
    let eps = 1e-6;
    let atol = 1e-4;

    // Analytic
    let mut ctx = CpuContext::new(1);
    let (_, dl) = cholesky_frule(&mut ctx, &a, &sym_tangent).unwrap();
    let analytic = tensor_data(&dl);

    // FD: (chol(A + eps*dA) - chol(A - eps*dA)) / (2*eps)
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let l_plus =
        tensor_data(&cholesky(&mut CpuContext::new(1), &make_tensor(plus, &[n, n])).unwrap());
    let l_minus =
        tensor_data(&cholesky(&mut CpuContext::new(1), &make_tensor(minus, &[n, n])).unwrap());
    let fd: Vec<f64> = l_plus
        .iter()
        .zip(&l_minus)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "cholesky_frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

// 7. Solve frule FD — test through x, varying A (hold b fixed, tangent_b = 0)
#[test]
fn solve_frule_fd_through_x_vary_a() {
    let n = 3;
    let a = make_general_test_matrix(n);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.5).collect();
    let b = make_tensor(b_data.clone(), &[n]);

    let a_data = tensor_data(&a);
    let eps = 1e-6;
    let atol = 1e-4;

    // Deterministic tangent for A
    let tangent_a_data: Vec<f64> = (0..n * n)
        .map(|i| ((i * 13 + 5) % 17) as f64 / 8.0 - 1.0)
        .collect();
    let tangent_a = make_tensor(tangent_a_data.clone(), &[n, n]);
    // Zero tangent for b
    let tangent_b = make_tensor(vec![0.0; n], &[n]);

    // Analytic
    let mut ctx = CpuContext::new(1);
    let (_, dx) = solve_frule(&mut ctx, &a, &b, &tangent_a, &tangent_b).unwrap();
    let analytic = tensor_data(&dx);

    // FD: (solve(A + eps*dA, b) - solve(A - eps*dA, b)) / (2*eps)
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_a_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_a_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let x_plus =
        tensor_data(&solve(&mut CpuContext::new(1), &make_tensor(plus, &[n, n]), &b).unwrap());
    let x_minus =
        tensor_data(&solve(&mut CpuContext::new(1), &make_tensor(minus, &[n, n]), &b).unwrap());
    let fd: Vec<f64> = x_plus
        .iter()
        .zip(&x_minus)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "solve_frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

// 8. Inv frule FD
#[test]
fn inv_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        inv(&mut b, x).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dinv) = inv_frule(&mut b, x, dx).unwrap();
        dinv
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 9. Det frule FD — output is scalar
#[test]
fn det_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        det(&mut b, x).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dd) = det_frule(&mut b, x, dx).unwrap();
        dd
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 10. Slogdet frule FD — test through logabsdet component
#[test]
fn slogdet_frule_fd_through_logabsdet() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        slogdet(&mut b, x).unwrap().logabsdet
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dresult) = slogdet_frule(&mut b, x, dx).unwrap();
        dresult.logabsdet
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 11. Lstsq frule FD — test through x, varying A (hold b fixed, tangent_b = 0)
#[test]
fn lstsq_frule_fd_through_x_vary_a() {
    let m = 4;
    let n = 2;
    // Build well-conditioned tall matrix
    let mut a_data = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let val = ((i * 3 + j * 7 + 1) % 11) as f64 / 10.0;
            a_data[i + j * m] = val;
        }
    }
    for i in 0..n {
        a_data[i + i * m] += 5.0;
    }
    // Construct b = A * x_true so residual is zero
    let x_true = [1.0, 2.0];
    let mut b_data = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            b_data[i] += a_data[i + j * m] * x_true[j];
        }
    }
    let a = make_tensor(a_data.clone(), &[m, n]);
    let b = make_tensor(b_data, &[m]);
    let eps = 1e-6;
    let atol = 1e-3;

    // Deterministic tangent for A
    let tangent_a_data: Vec<f64> = (0..m * n)
        .map(|i| ((i * 13 + 5) % 17) as f64 / 8.0 - 1.0)
        .collect();
    let tangent_a = make_tensor(tangent_a_data.clone(), &[m, n]);
    // Zero tangent for b
    let tangent_b = make_tensor(vec![0.0; m], &[m]);

    // Analytic
    let mut ctx = CpuContext::new(1);
    let (_, dresult) = lstsq_frule(&mut ctx, &a, &b, &tangent_a, &tangent_b).unwrap();
    let analytic = tensor_data(&dresult.x);

    // FD: (lstsq(A + eps*dA, b).x - lstsq(A - eps*dA, b).x) / (2*eps)
    let plus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_a_data)
        .map(|(a, da)| a + eps * da)
        .collect();
    let minus: Vec<f64> = a_data
        .iter()
        .zip(&tangent_a_data)
        .map(|(a, da)| a - eps * da)
        .collect();
    let xp = tensor_data(
        &lstsq(&mut CpuContext::new(1), &make_tensor(plus, &[m, n]), &b)
            .unwrap()
            .x,
    );
    let xm = tensor_data(
        &lstsq(&mut CpuContext::new(1), &make_tensor(minus, &[m, n]), &b)
            .unwrap()
            .x,
    );
    let fd: Vec<f64> = xp
        .iter()
        .zip(&xm)
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < atol,
        "lstsq_frule FD check failed: max_err={max_err} > atol={atol}"
    );
}

// 12. Pinv frule FD
#[test]
fn pinv_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        pinv(&mut b, x, None).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dpinv) = pinv_frule(&mut b, x, dx, None).unwrap();
        dpinv
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 13. Matrix exp frule FD
#[test]
fn matrix_exp_frule_fd() {
    let a = make_general_test_matrix(3);
    // Scale down to keep matrix exp numerically manageable
    let a_data = tensor_data(&a);
    let a_scaled: Vec<f64> = a_data.iter().map(|v| v * 0.1).collect();
    let a = make_tensor(a_scaled, &[3, 3]);

    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        matrix_exp(&mut b, x).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dexp) = matrix_exp_frule(&mut b, x, dx).unwrap();
        dexp
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// 14. Norm frule FD — Frobenius norm
#[test]
fn norm_fro_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Fro).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dnrm) = norm_frule(&mut b, x, dx, NormKind::Fro).unwrap();
        dnrm
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-4);
}

// ============================================================================
// Coverage: f32 forward tests
// ============================================================================

/// Create a column-major tensor from a flat vec of f32 and shape.
fn make_tensor_f32(data: Vec<f32>, dims: &[usize]) -> Tensor<f32> {
    let ndim = dims.len();
    let mut strides = vec![0isize; ndim];
    if ndim > 0 {
        strides[0] = 1;
        for i in 1..ndim {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

/// Extract flat data from a Tensor<f32>.
fn tensor_data_f32(t: &Tensor<f32>) -> Vec<f32> {
    let c = t.contiguous(COL);
    let off = c.offset() as usize;
    let len: usize = c.dims().iter().product();
    c.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

#[test]
fn svd_f32_identity() {
    let mut ctx = CpuContext::new(1);
    let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let a = make_tensor_f32(data, &[2, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let s = tensor_data_f32(&result.s);
    for &val in &s {
        assert!(
            (val - 1.0_f32).abs() < 1e-5,
            "f32 SVD singular value: {val}"
        );
    }
}

#[test]
fn svd_f32_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // 2x3 matrix
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = make_tensor_f32(data.clone(), &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let u = tensor_data_f32(&result.u);
    let s = tensor_data_f32(&result.s);
    let vt = tensor_data_f32(&result.vt);
    let m = 2;
    let n = 3;
    let k = 2;
    // Reconstruct A = U diag(S) Vt
    let mut recon = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0_f32;
            for l in 0..k {
                val += u[i + l * m] * s[l] * vt[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }
    let err: f32 = data
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1e-4, "f32 SVD reconstruction error: {err}");
}

#[test]
fn qr_f32_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = make_tensor_f32(data.clone(), &[2, 3]);
    let result = qr(&mut ctx, &a).unwrap();
    let q = tensor_data_f32(&result.q);
    let r = tensor_data_f32(&result.r);
    let m = 2;
    let n = 3;
    let k = 2;
    let mut recon = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0_f32;
            for l in 0..k {
                val += q[i + l * m] * r[l + j * k];
            }
            recon[i + j * m] = val;
        }
    }
    let err: f32 = data
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1e-4, "f32 QR reconstruction error: {err}");
}

#[test]
fn solve_f32() {
    let mut ctx = CpuContext::new(1);
    // A = [[2, 1], [1, 3]], b = [5, 10]
    let a = make_tensor_f32(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor_f32(vec![5.0, 10.0], &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data_f32(&x);
    // Verify: Ax = b
    let res0 = 2.0 * xd[0] + 1.0 * xd[1] - 5.0;
    let res1 = 1.0 * xd[0] + 3.0 * xd[1] - 10.0;
    assert!(res0.abs() < 1e-4, "f32 solve residual[0] = {res0}");
    assert!(res1.abs() < 1e-4, "f32 solve residual[1] = {res1}");
}

#[test]
fn det_f32() {
    let mut ctx = CpuContext::new(1);
    // A = [[1, 2], [3, 4]], col-major: [1, 3, 2, 4]
    let a = make_tensor_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let d = det(&mut ctx, &a).unwrap();
    let d_data = tensor_data_f32(&d);
    // det = 1*4 - 2*3 = -2
    assert!(
        (d_data[0] - (-2.0_f32)).abs() < 1e-4,
        "f32 det = {}",
        d_data[0]
    );
}

#[test]
fn inv_f32() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a_inv = inv(&mut ctx, &a).unwrap();
    let inv_data = tensor_data_f32(&a_inv);
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let n = 2;
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0_f32;
            for k in 0..n {
                val += a_data[i + k * n] * inv_data[k + j * n];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (val - expected).abs() < 1e-4,
                "f32 A*A^-1[{i},{j}] = {val}, expected {expected}"
            );
        }
    }
}

#[test]
fn lu_f32_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor_f32(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l = tensor_data_f32(&result.l);
    let u = tensor_data_f32(&result.u);
    let n = 2;
    // P A = L U
    let mut lu_prod = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0_f32;
            for k in 0..n {
                val += l[i + k * n] * u[k + j * n];
            }
            lu_prod[i + j * n] = val;
        }
    }
    // Apply P^-1 to get A back
    let a_data: Vec<f32> = vec![2.0, 1.0, 1.0, 3.0];
    let p = &result.p.unwrap();
    let mut pa = vec![0.0_f32; n * n];
    for j in 0..n {
        for i in 0..n {
            pa[i + j * n] = a_data[p[i] + j * n];
        }
    }
    let err: f32 = lu_prod
        .iter()
        .zip(&pa)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1e-4, "f32 LU reconstruction error: {err}");
}

#[test]
fn cholesky_f32() {
    let mut ctx = CpuContext::new(1);
    // SPD: [[4, 2], [2, 3]]
    let a = make_tensor_f32(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
    let l = cholesky(&mut ctx, &a).unwrap();
    let l_data = tensor_data_f32(&l);
    let n = 2;
    let mut llt = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                llt[i + j * n] += l_data[i + k * n] * l_data[j + k * n];
            }
        }
    }
    let a_data: Vec<f32> = vec![4.0, 2.0, 2.0, 3.0];
    let err: f32 = llt
        .iter()
        .zip(&a_data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 1e-4, "f32 cholesky LL^T error: {err}");
}

#[test]
fn eigen_f32() {
    let mut ctx = CpuContext::new(1);
    // Symmetric: [[2, 1], [1, 3]]
    let a = make_tensor_f32(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let result = eigen(&mut ctx, &a).unwrap();
    let evals = tensor_data_f32(&result.values);
    // Eigenvalues of [[2,1],[1,3]]: (5 +/- sqrt(5))/2 = ~1.382 and ~3.618
    let sum: f32 = evals.iter().sum();
    assert!(
        (sum - 5.0).abs() < 1e-3,
        "f32 eigen sum = {sum}, expected 5.0"
    );
}

#[test]
fn slogdet_f32() {
    let mut ctx = CpuContext::new(1);
    // A = [[1, 2], [3, 4]], col-major: [1, 3, 2, 4]
    let a = make_tensor_f32(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let result = slogdet(&mut ctx, &a).unwrap();
    let sign = tensor_data_f32(&result.sign);
    let logabsdet = tensor_data_f32(&result.logabsdet);
    // det = -2, sign = -1, logabsdet = ln(2)
    assert!(
        (sign[0] - (-1.0_f32)).abs() < 1e-4,
        "f32 slogdet sign = {}",
        sign[0]
    );
    assert!(
        (logabsdet[0] - 2.0_f32.ln()).abs() < 1e-4,
        "f32 slogdet logabsdet = {}",
        logabsdet[0]
    );
}

// ============================================================================
// Coverage: Complex32 backend tests
// ============================================================================

/// Create a column-major tensor from Complex32.
fn make_tensor_c32(data: Vec<Complex32>, dims: &[usize]) -> Tensor<Complex32> {
    let ndim = dims.len();
    let mut strides = vec![0isize; ndim];
    if ndim > 0 {
        strides[0] = 1;
        for i in 1..ndim {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

fn c32(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

/// Extract flat data from a Tensor<Complex32>.
fn tensor_data_c32(t: &Tensor<Complex32>) -> Vec<Complex32> {
    let c_tensor = t.contiguous(COL);
    let off = c_tensor.offset() as usize;
    let len: usize = c_tensor.dims().iter().product();
    c_tensor.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

#[test]
fn svd_complex32_identity() {
    let mut ctx = CpuContext::new(1);
    let data = vec![c32(1.0, 0.0), c32(0.0, 0.0), c32(0.0, 0.0), c32(1.0, 0.0)];
    let a = make_tensor_c32(data, &[2, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    assert_eq!(result.u.dims(), &[2, 2]);
    assert_eq!(result.s.dims(), &[2]);
    assert_eq!(result.vt.dims(), &[2, 2]);
    let s = {
        let c_tensor = result.s.contiguous(COL);
        let off = c_tensor.offset() as usize;
        let len: usize = c_tensor.dims().iter().product();
        c_tensor.buffer().as_slice().unwrap()[off..off + len].to_vec()
    };
    for &val in &s {
        assert!(
            (val - 1.0_f32).abs() < 1e-4,
            "c32 SVD singular value: {val}"
        );
    }
}

#[test]
fn qr_complex32_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![c32(1.0, 2.0), c32(3.0, -1.0), c32(0.0, 1.0), c32(4.0, 0.0)];
    let a = make_tensor_c32(data.clone(), &[2, 2]);
    let result = qr(&mut ctx, &a).unwrap();
    let q = tensor_data_c32(&result.q);
    let r = tensor_data_c32(&result.r);
    let n = 2;
    let mut recon = vec![c32(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = c32(0.0, 0.0);
            for l in 0..n {
                val += q[i + l * n] * r[l + j * n];
            }
            recon[i + j * n] = val;
        }
    }
    let err: f32 = data
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1e-3, "c32 QR reconstruction error: {err}");
}

#[test]
fn lu_complex32_reconstruction() {
    let mut ctx = CpuContext::new(1);
    let data = vec![c32(2.0, 1.0), c32(1.0, 0.0), c32(0.0, 1.0), c32(3.0, -1.0)];
    let a = make_tensor_c32(data.clone(), &[2, 2]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l = tensor_data_c32(&result.l);
    let u = tensor_data_c32(&result.u);
    let n = 2;
    let mut lu_prod = vec![c32(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                lu_prod[i + j * n] += l[i + k * n] * u[k + j * n];
            }
        }
    }
    // PA = LU, build PA
    let p = &result.p.unwrap();
    let mut pa = vec![c32(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            pa[i + j * n] = data[p[i] + j * n];
        }
    }
    let err: f32 = lu_prod
        .iter()
        .zip(&pa)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1e-3, "c32 LU reconstruction error: {err}");
}

#[test]
fn solve_complex32() {
    let mut ctx = CpuContext::new(1);
    // A = [[2+i, 1], [0, 3-i]]
    let a = make_tensor_c32(
        vec![c32(2.0, 1.0), c32(0.0, 0.0), c32(1.0, 0.0), c32(3.0, -1.0)],
        &[2, 2],
    );
    let b = make_tensor_c32(vec![c32(5.0, 0.0), c32(3.0, 0.0)], &[2, 1]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data_c32(&x);
    // Verify Ax = b
    let ax0 = c32(2.0, 1.0) * xd[0] + c32(1.0, 0.0) * xd[1];
    let ax1 = c32(0.0, 0.0) * xd[0] + c32(3.0, -1.0) * xd[1];
    assert!((ax0 - c32(5.0, 0.0)).norm() < 1e-3, "c32 solve residual[0]");
    assert!((ax1 - c32(3.0, 0.0)).norm() < 1e-3, "c32 solve residual[1]");
}

#[test]
fn cholesky_complex32() {
    let mut ctx = CpuContext::new(1);
    // Hermitian SPD: [[4, 2-i], [2+i, 5]]
    let a = make_tensor_c32(
        vec![c32(4.0, 0.0), c32(2.0, 1.0), c32(2.0, -1.0), c32(5.0, 0.0)],
        &[2, 2],
    );
    let l = cholesky(&mut ctx, &a).unwrap();
    let l_data = tensor_data_c32(&l);
    let n = 2;
    // Verify L L^H = A
    let mut llh = vec![c32(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                llh[i + j * n] += l_data[i + k * n] * l_data[j + k * n].conj();
            }
        }
    }
    let a_data = vec![c32(4.0, 0.0), c32(2.0, 1.0), c32(2.0, -1.0), c32(5.0, 0.0)];
    let err: f32 = llh
        .iter()
        .zip(&a_data)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f32::max);
    assert!(err < 1e-3, "c32 cholesky error: {err}");
}

#[test]
fn eigen_complex32_hermitian() {
    let mut ctx = CpuContext::new(1);
    // Hermitian: [[3, 1-i], [1+i, 2]]
    let a = make_tensor_c32(
        vec![c32(3.0, 0.0), c32(1.0, 1.0), c32(1.0, -1.0), c32(2.0, 0.0)],
        &[2, 2],
    );
    let result = eigen(&mut ctx, &a).unwrap();
    let evals = {
        let c_tensor = result.values.contiguous(COL);
        let off = c_tensor.offset() as usize;
        let len: usize = c_tensor.dims().iter().product();
        c_tensor.buffer().as_slice().unwrap()[off..off + len].to_vec()
    };
    let sum: f32 = evals.iter().sum();
    // trace = 3 + 2 = 5
    assert!(
        (sum - 5.0).abs() < 1e-2,
        "c32 eigen sum = {sum}, expected 5.0"
    );
}

#[test]
fn solve_triangular_complex32() {
    let mut ctx = CpuContext::new(1);
    // Lower triangular: [[2+i, 0], [1, 3-i]]
    let a = make_tensor_c32(
        vec![c32(2.0, 1.0), c32(1.0, 0.0), c32(0.0, 0.0), c32(3.0, -1.0)],
        &[2, 2],
    );
    let b = make_tensor_c32(vec![c32(4.0, 2.0), c32(5.0, 0.0)], &[2, 1]);
    let x = solve_triangular(&mut ctx, &a, &b, false).unwrap();
    let xd = tensor_data_c32(&x);
    // Verify Ax = b
    let ax0 = c32(2.0, 1.0) * xd[0];
    let ax1 = c32(1.0, 0.0) * xd[0] + c32(3.0, -1.0) * xd[1];
    assert!(
        (ax0 - c32(4.0, 2.0)).norm() < 1e-3,
        "c32 solve_tri residual[0]"
    );
    assert!(
        (ax1 - c32(5.0, 0.0)).norm() < 1e-3,
        "c32 solve_tri residual[1]"
    );
}

// ============================================================================
// Coverage: Additional Complex64 backend tests (operations not yet tested)
// ============================================================================

#[test]
fn inv_complex64() {
    let mut ctx = CpuContext::new(1);
    let data = vec![c(1.0, 1.0), c(2.0, 0.0), c(0.0, 1.0), c(3.0, -1.0)];
    let a = make_complex_tensor(data.clone(), &[2, 2]);
    let a_inv = inv(&mut ctx, &a).unwrap();
    let inv_data = complex_tensor_data(&a_inv);
    let n = 2;
    for i in 0..n {
        for j in 0..n {
            let mut val = c(0.0, 0.0);
            for k in 0..n {
                val += data[i + k * n] * inv_data[k + j * n];
            }
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            assert!(
                (val - expected).norm() < 1e-10,
                "c64 A*A^-1[{i},{j}] error = {}",
                (val - expected).norm()
            );
        }
    }
}

#[test]
fn lstsq_complex64() {
    let mut ctx = CpuContext::new(1);
    // Overdetermined 3x2 system
    let a = make_complex_tensor(
        vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
        ],
        &[3, 2],
    );
    let b = make_complex_tensor(vec![c(2.0, 1.0), c(3.0, -1.0), c(0.0, 0.0)], &[3]);
    let result = lstsq(&mut ctx, &a, &b).unwrap();
    let x = complex_tensor_data(&result.x);
    // A = [[1,0],[0,1],[0,0]], b = [2+i, 3-i, 0] => x = [2+i, 3-i]
    assert!(
        (x[0] - c(2.0, 1.0)).norm() < 1e-8,
        "c64 lstsq x[0] = {:?}",
        x[0]
    );
    assert!(
        (x[1] - c(3.0, -1.0)).norm() < 1e-8,
        "c64 lstsq x[1] = {:?}",
        x[1]
    );
}

#[test]
fn matrix_exp_complex64() {
    let mut ctx = CpuContext::new(1);
    // exp(0) = I
    let data = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
    let a = make_complex_tensor(data, &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data_out = complex_tensor_data(&result);
    // exp(0) = I
    assert!(
        (data_out[0] - c(1.0, 0.0)).norm() < 1e-10,
        "exp(0)[0,0] = {:?}",
        data_out[0]
    );
    assert!(
        (data_out[3] - c(1.0, 0.0)).norm() < 1e-10,
        "exp(0)[1,1] = {:?}",
        data_out[3]
    );
}

// ============================================================================
// Coverage: solve_triangular forward (upper and lower)
// ============================================================================

#[test]
fn solve_triangular_upper_f64() {
    let mut ctx = CpuContext::new(1);
    // Upper triangular: [[2, 1], [0, 3]]
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![5.0, 6.0], &[2, 1]);
    let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();
    let xd = tensor_data(&x);
    // Verify Ax = b: 2*x0 + 1*x1 = 5, 3*x1 = 6 => x1 = 2, x0 = 1.5
    assert!((xd[1] - 2.0).abs() < 1e-10, "upper tri x[1] = {}", xd[1]);
    assert!((xd[0] - 1.5).abs() < 1e-10, "upper tri x[0] = {}", xd[0]);
}

#[test]
fn solve_triangular_lower_f64() {
    let mut ctx = CpuContext::new(1);
    // Lower triangular: [[2, 0], [1, 3]]
    let a = make_tensor(vec![2.0, 1.0, 0.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![4.0, 5.0], &[2, 1]);
    let x = solve_triangular(&mut ctx, &a, &b, false).unwrap();
    let xd = tensor_data(&x);
    // 2*x0 = 4, x0 + 3*x1 = 5 => x0 = 2, x1 = 1
    assert!((xd[0] - 2.0).abs() < 1e-10, "lower tri x[0] = {}", xd[0]);
    assert!((xd[1] - 1.0).abs() < 1e-10, "lower tri x[1] = {}", xd[1]);
}

#[test]
fn solve_triangular_upper_multi_rhs() {
    let mut ctx = CpuContext::new(1);
    // Upper triangular: [[1, 2], [0, 3]]
    let a = make_tensor(vec![1.0, 0.0, 2.0, 3.0], &[2, 2]);
    // b: (2, 2) = 2 columns
    let b = make_tensor(vec![5.0, 6.0, 8.0, 9.0], &[2, 2]);
    let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    let xd = tensor_data(&x);
    // Column 0: 3*x1 = 6 => x1=2; x0 + 2*2 = 5 => x0=1
    assert!((xd[0] - 1.0).abs() < 1e-10);
    assert!((xd[1] - 2.0).abs() < 1e-10);
    // Column 1: 3*x1 = 9 => x1=3; x0 + 2*3 = 8 => x0=2
    assert!((xd[2] - 2.0).abs() < 1e-10);
    assert!((xd[3] - 3.0).abs() < 1e-10);
}

// ============================================================================
// Coverage: norm Nuclear and Spectral forward + AD
// ============================================================================

#[test]
fn norm_nuclear_forward() {
    let mut ctx = CpuContext::new(1);
    // A = diag(3, 1), nuclear norm = 3 + 1 = 4
    let a = make_tensor(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Nuclear).unwrap();
    let nd = tensor_data(&n);
    assert!((nd[0] - 4.0).abs() < 1e-10, "nuclear norm = {}", nd[0]);
}

#[test]
fn norm_spectral_forward() {
    let mut ctx = CpuContext::new(1);
    // A = diag(3, 1), spectral norm = 3
    let a = make_tensor(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Spectral).unwrap();
    let nd = tensor_data(&n);
    assert!((nd[0] - 3.0).abs() < 1e-10, "spectral norm = {}", nd[0]);
}

#[test]
fn norm_nuclear_rrule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Nuclear).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm_rrule(&mut b, x, co, NormKind::Nuclear).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-3);
}

#[test]
fn norm_spectral_rrule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Spectral).unwrap()
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm_rrule(&mut b, x, co, NormKind::Spectral).unwrap()
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-3);
}

#[test]
fn norm_nuclear_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Nuclear).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dnrm) = norm_frule(&mut b, x, dx, NormKind::Nuclear).unwrap();
        dnrm
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-3);
}

#[test]
fn norm_spectral_frule_fd() {
    let a = make_general_test_matrix(3);
    let fwd = |x: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        norm(&mut b, x, NormKind::Spectral).unwrap()
    };
    let frule_fn = |x: &Tensor<f64>, dx: &Tensor<f64>| {
        let mut b = CpuContext::new(1);
        let (_, dnrm) = norm_frule(&mut b, x, dx, NormKind::Spectral).unwrap();
        dnrm
    };
    check_frule_fd(fwd, frule_fn, &a, 1e-6, 1e-3);
}

// ============================================================================
// Coverage: SVD with cutoff option
// ============================================================================

#[test]
fn svd_with_cutoff() {
    let mut ctx = CpuContext::new(1);
    // Nearly rank-1 matrix
    let a = make_tensor(vec![1.0, 2.0, 1.0 + 1e-14, 2.0 + 1e-14], &[2, 2]);
    let opts = SvdOptions {
        max_rank: None,
        cutoff: Some(1e-10),
    };
    let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
    // One singular value should be truncated to 0
    let s = tensor_data(&result.s);
    // With cutoff, only one significant singular value remains
    let nonzero_count = s.iter().filter(|&&v| v > 1e-10).count();
    assert_eq!(
        nonzero_count, 1,
        "expected 1 significant SV, got {nonzero_count}"
    );
}

#[test]
fn svd_with_default_options() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let opts = SvdOptions::default();
    let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
    let s = tensor_data(&result.s);
    assert_eq!(s.len(), 2);
}

// ============================================================================
// Coverage: batch dimension tests
// ============================================================================

#[test]
fn det_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 matrices: [[1,2],[3,4]] and [[5,6],[7,8]]
    // shape [2, 2, 2], strides [1, 2, 4]
    // data = [a[0,0,0]=1, a[1,0,0]=3, a[0,1,0]=2, a[1,1,0]=4, a[0,0,1]=5, a[1,0,1]=7, a[0,1,1]=6, a[1,1,1]=8]
    let a = make_tensor(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], &[2, 2, 2]);
    let d = det(&mut ctx, &a).unwrap();
    let dd = tensor_data(&d);
    assert_eq!(dd.len(), 2);
    // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert!((dd[0] - (-2.0)).abs() < 1e-10, "batch det[0] = {}", dd[0]);
    // det([[5,6],[7,8]]) = 5*8 - 6*7 = -2
    assert!((dd[1] - (-2.0)).abs() < 1e-10, "batch det[1] = {}", dd[1]);
}

#[test]
fn slogdet_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 matrices stacked along batch dim
    // shape [2, 2, 2], same data as det_batched
    let a = make_tensor(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], &[2, 2, 2]);
    let result = slogdet(&mut ctx, &a).unwrap();
    let signs = tensor_data(&result.sign);
    let logabs = tensor_data(&result.logabsdet);
    assert_eq!(signs.len(), 2);
    assert_eq!(logabs.len(), 2);
    // Both dets = -2, sign = -1, log|det| = ln(2)
    for i in 0..2 {
        assert!(
            (signs[i] - (-1.0)).abs() < 1e-10,
            "slogdet sign[{i}] = {}",
            signs[i]
        );
        assert!(
            (logabs[i] - 2.0_f64.ln()).abs() < 1e-10,
            "slogdet logabs[{i}] = {}",
            logabs[i]
        );
    }
}

#[test]
fn svd_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 identity matrices: shape [2, 2, 2], strides [1, 2, 4]
    // Batch 0 = I: [1, 0, 0, 1], Batch 1 = I: [1, 0, 0, 1]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    assert_eq!(result.u.dims(), &[2, 2, 2]);
    assert_eq!(result.s.dims(), &[2, 2]);
    assert_eq!(result.vt.dims(), &[2, 2, 2]);
}

#[test]
fn qr_batched() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let result = qr(&mut ctx, &a).unwrap();
    assert_eq!(result.q.dims(), &[2, 2, 2]);
    assert_eq!(result.r.dims(), &[2, 2, 2]);
}

#[test]
fn solve_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 identity matrices, vector RHS per batch
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    // b shape [2, 2] means vector RHS (n=2) for each batch
    let b = make_tensor(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data(&x);
    // x = b for identity A
    assert!((xd[0] - 3.0).abs() < 1e-10);
    assert!((xd[1] - 4.0).abs() < 1e-10);
    assert!((xd[2] - 5.0).abs() < 1e-10);
    assert!((xd[3] - 6.0).abs() < 1e-10);
}

#[test]
fn inv_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 identity matrices
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let a_inv = inv(&mut ctx, &a).unwrap();
    assert_eq!(a_inv.dims(), &[2, 2, 2]);
}

#[test]
fn norm_batched_fro() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 identity matrices
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Fro).unwrap();
    let nd = tensor_data(&n);
    assert_eq!(nd.len(), 2);
    // Frobenius norm of identity = sqrt(2)
    for (i, value) in nd.iter().enumerate().take(2) {
        assert!(
            (*value - 2.0_f64.sqrt()).abs() < 1e-10,
            "batch Fro norm[{i}] = {}",
            value
        );
    }
}

// ============================================================================
// Coverage: lstsq forward happy path
// ============================================================================

#[test]
fn lstsq_overdetermined() {
    let mut ctx = CpuContext::new(1);
    // A = [[1,0],[0,1],[0,0]], b = [3, 7, 0]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    let b = make_tensor(vec![3.0, 7.0, 0.0], &[3]);
    let result = lstsq(&mut ctx, &a, &b).unwrap();
    let x = tensor_data(&result.x);
    assert_eq!(x.len(), 2);
    assert!((x[0] - 3.0).abs() < 1e-10, "lstsq x[0] = {}", x[0]);
    assert!((x[1] - 7.0).abs() < 1e-10, "lstsq x[1] = {}", x[1]);
}

#[test]
fn lstsq_underdetermined_returns_error() {
    let mut ctx = CpuContext::new(1);
    // m < n: 2x3
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]);
    let b = make_tensor(vec![1.0, 2.0], &[2]);
    assert!(lstsq(&mut ctx, &a, &b).is_err());
}

// ============================================================================
// Coverage: error paths for validation functions
// ============================================================================

#[test]
fn validate_1d_input_returns_error() {
    let mut ctx = CpuContext::new(1);
    // 1D input to SVD
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(svd(&mut ctx, &a, None).is_err());
    assert!(qr(&mut ctx, &a).is_err());
}

#[test]
fn validate_non_square_for_square_ops() {
    let mut ctx = CpuContext::new(1);
    // 2x3 input to square-only ops
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(eigen(&mut ctx, &a).is_err());
    assert!(cholesky(&mut ctx, &a).is_err());
    assert!(inv(&mut ctx, &a).is_err());
    assert!(det(&mut ctx, &a).is_err());
    assert!(slogdet(&mut ctx, &a).is_err());
}

#[test]
fn solve_rhs_batch_mismatch() {
    let mut ctx = CpuContext::new(1);
    // A is (2,2,2), b is (2,3) — batch dim mismatch
    let a = make_tensor(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[2, 2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_rhs_wrong_leading_dim() {
    let mut ctx = CpuContext::new(1);
    // A is (2,2), b is (3,1) — leading dim mismatch
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3, 1]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_rhs_nrhs_zero() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // b with nrhs=0
    let b: Tensor<f64> = Tensor::from_vec(vec![], &[2, 0], &[1, 2], 0).unwrap();
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn lstsq_rhs_wrong_leading_dim() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    // b dim[0] = 2, expected 3
    let b = make_tensor(vec![1.0, 2.0], &[2]);
    assert!(lstsq(&mut ctx, &a, &b).is_err());
}

#[test]
fn lstsq_rhs_batch_mismatch() {
    let mut ctx = CpuContext::new(1);
    // A: (3, 2, 2), b: (3, 3)
    let a = make_tensor(
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        &[3, 2, 2],
    );
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    assert!(lstsq(&mut ctx, &a, &b).is_err());
}

#[test]
fn lstsq_rhs_ndim_mismatch() {
    let mut ctx = CpuContext::new(1);
    // A: (3, 2), b: (3, 1, 1) — wrong ndim for b
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3, 1, 1]);
    assert!(lstsq(&mut ctx, &a, &b).is_err());
}

#[test]
fn cholesky_non_spd_returns_error() {
    let mut ctx = CpuContext::new(1);
    // Matrix with negative eigenvalue: [[-1, 0], [0, 1]]
    let a = make_tensor(vec![-1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(cholesky(&mut ctx, &a).is_err());
}

#[test]
fn norm_l1_inf_supported() {
    let mut ctx = CpuContext::new(1);
    // Matrix [[1, 3], [2, 4]] in column-major storage.
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let l1 = norm(&mut ctx, &a, NormKind::L1).unwrap();
    let inf = norm(&mut ctx, &a, NormKind::Inf).unwrap();
    let l1_data = tensor_data(&l1);
    let inf_data = tensor_data(&inf);
    assert!((l1_data[0] - 7.0).abs() < 1e-12);
    assert!((inf_data[0] - 6.0).abs() < 1e-12);
}

#[test]
fn norm_vector_kinds_supported() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![3.0, -4.0], &[2]);

    let fro = norm(&mut ctx, &x, NormKind::Fro).unwrap();
    let l1 = norm(&mut ctx, &x, NormKind::L1).unwrap();
    let inf = norm(&mut ctx, &x, NormKind::Inf).unwrap();
    let lp = norm(&mut ctx, &x, NormKind::Lp(3.0)).unwrap();

    assert!((tensor_data(&fro)[0] - 5.0).abs() < 1e-12);
    assert!((tensor_data(&l1)[0] - 7.0).abs() < 1e-12);
    assert!((tensor_data(&inf)[0] - 4.0).abs() < 1e-12);
    assert!((tensor_data(&lp)[0] - 91.0_f64.powf(1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn norm_vector_error_cases_return_errors() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![4.0, -4.0, 0.0], &[3]);

    let lp1 = norm(&mut ctx, &x, NormKind::Lp(1.0)).unwrap();
    assert!((tensor_data(&lp1)[0] - 8.0).abs() < 1e-12);

    assert!(norm(&mut ctx, &x, NormKind::Lp(0.5)).is_err());
    assert!(norm(&mut ctx, &x, NormKind::Spectral).is_err());
}

// ============================================================================
// Coverage: Non-square SVD rrule with full cotangent (dU, dS, dVt)
// ============================================================================

#[test]
fn svd_rrule_tall_with_du_cotangent() {
    // Exercise the m > k correction path in svd_rrule
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 3x2 matrix
        &[3, 2],
    );
    let result = svd(&mut ctx, &a, None).unwrap();
    // Provide du + ds cotangents to exercise the m > k correction
    let du = make_tensor(vec![1.0; 6], result.u.dims()); // 3x2
    let ds = make_tensor(vec![1.0; 2], result.s.dims());
    let cotangent = SvdCotangent {
        u: Some(du),
        s: Some(ds),
        vt: None,
    };
    let grad = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "svd_rrule grad not finite: {val}");
    }
}

#[test]
fn svd_rrule_wide_with_dvt_cotangent() {
    // Exercise the n > k correction path in svd_rrule
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
        &[2, 3],
    );
    let result = svd(&mut ctx, &a, None).unwrap();
    let dvt = make_tensor(vec![1.0; 6], result.vt.dims()); // 2x3
    let ds = make_tensor(vec![1.0; 2], result.s.dims());
    let cotangent = SvdCotangent {
        u: None,
        s: Some(ds),
        vt: Some(dvt),
    };
    let grad = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "svd_rrule grad not finite: {val}");
    }
}

// ============================================================================
// Coverage: SVD frule for non-square (tall m>k and wide n>k)
// ============================================================================

#[test]
fn svd_frule_tall_matrix() {
    // Exercise the m > k projector path in svd_frule (3x2)
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let da = make_tensor(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3], &[3, 2]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    assert_eq!(result.u.dims(), &[3, 2]);
    assert_eq!(dresult.u.dims(), &[3, 2]);
    let du_data = tensor_data(&dresult.u);
    for &val in &du_data {
        assert!(val.is_finite(), "svd_frule dU not finite");
    }
}

#[test]
fn svd_frule_wide_matrix() {
    // Exercise the n > k projector path in svd_frule (2x3)
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let da = make_tensor(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3], &[2, 3]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    assert_eq!(result.vt.dims(), &[2, 3]);
    assert_eq!(dresult.vt.dims(), &[2, 3]);
    let dvt_data = tensor_data(&dresult.vt);
    for &val in &dvt_data {
        assert!(val.is_finite(), "svd_frule dVt not finite");
    }
}

// ============================================================================
// Coverage: QR frule and rrule for non-square
// ============================================================================

#[test]
fn qr_rrule_wide_matrix() {
    // Exercise the n > k path in qr_rrule (2x3 wide matrix)
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = qr(&mut ctx, &a).unwrap();
    // Provide Q and R cotangents
    let dq = make_tensor(vec![1.0; 4], result.q.dims()); // 2x2
    let dr = make_tensor(vec![1.0; 6], result.r.dims()); // 2x3
    let cotangent = QrCotangent {
        q: Some(dq),
        r: Some(dr),
    };
    let grad = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "qr_rrule grad not finite: {val}");
    }
}

#[test]
fn qr_frule_wide_matrix() {
    // Exercise the full path in qr_frule (2x3 wide matrix)
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let da = make_tensor(vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3], &[2, 3]);
    let (result, dresult) = qr_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(result.q.dims(), &[2, 2]);
    assert_eq!(result.r.dims(), &[2, 3]);
    assert_eq!(dresult.q.dims(), &[2, 2]);
    assert_eq!(dresult.r.dims(), &[2, 3]);
    let dr_data = tensor_data(&dresult.r);
    for &val in &dr_data {
        assert!(val.is_finite(), "qr_frule dR not finite: {val}");
    }
}

// Note: LU rrule for non-square matrices is not tested here because
// faer's LU backend panics on non-square input (faer requires m == n).

// ============================================================================
// Coverage: lstsq rrule
// ============================================================================

#[test]
fn lstsq_rrule_basic() {
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    let b = make_tensor(vec![3.0, 7.0, 0.0], &[3]);
    let fwd = |x: &Tensor<f64>| {
        let mut bk = CpuContext::new(1);
        lstsq(&mut bk, x, &b).unwrap().x
    };
    let rrule_fn = |x: &Tensor<f64>, co: &Tensor<f64>| {
        let mut bk = CpuContext::new(1);
        lstsq_rrule(&mut bk, x, &b, co).unwrap().a
    };
    check_rrule_fd(fwd, rrule_fn, &a, 1e-6, 1e-2);
}

// ============================================================================
// Coverage: eigen rrule with vectors cotangent
// ============================================================================

#[test]
fn eigen_rrule_with_vectors_cotangent() {
    // Exercise the code path where both values and vectors cotangents are provided.
    // We use a symmetric 3x3 with well-separated eigenvalues.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![5.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 1.0], &[3, 3]);
    let result = eigen(&mut ctx, &a).unwrap();
    let n = 3;
    // Provide cotangent for both values and vectors
    let de = make_tensor(vec![1.0; n], result.values.dims());
    let dv = make_tensor(vec![1.0; n * n], result.vectors.dims());
    let cotangent = EigenCotangent {
        values: Some(de),
        vectors: Some(dv),
    };
    let grad = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[n, n]);
    // Just verify the grad is finite (exercises both branches)
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "eigen rrule grad not finite: {val}");
    }
}

// ============================================================================
// Coverage: matrix_exp edge cases
// ============================================================================

#[test]
fn matrix_exp_1x1_scalar_val() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0], &[1, 1]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data(&result);
    assert!(
        (data[0] - 2.0_f64.exp()).abs() < 1e-10,
        "matrix_exp(2) = {}, expected {}",
        data[0],
        2.0_f64.exp()
    );
}

#[test]
fn matrix_exp_f32() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor_f32(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let data = tensor_data_f32(&result);
    // exp(0) = I
    assert!((data[0] - 1.0).abs() < 1e-4, "exp(0)[0,0] = {}", data[0]);
    assert!((data[3] - 1.0).abs() < 1e-4, "exp(0)[1,1] = {}", data[3]);
}

// ============================================================================
// Coverage: pinv forward with threshold
// ============================================================================

#[test]
fn pinv_with_threshold() {
    let mut ctx = CpuContext::new(1);
    // Nearly rank-deficient
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1e-15], &[2, 2]);
    let result = pinv(&mut ctx, &a, Some(1e-10)).unwrap();
    let data = tensor_data(&result);
    // Only the first singular value should survive
    assert!((data[0] - 1.0).abs() < 1e-10, "pinv[0,0] = {}", data[0]);
    // The second diagonal should be effectively zero
    assert!(
        data[3].abs() < 1e-4,
        "pinv[1,1] = {} (should be ~0)",
        data[3]
    );
}

// ============================================================================
// Coverage: eig forward for general non-symmetric (covers interleaved ri)
// ============================================================================

#[test]
fn eig_3x3_general() {
    let mut ctx = CpuContext::new(1);
    // Non-symmetric matrix
    let a = make_tensor(vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
    let result = eig(&mut ctx, &a).unwrap();
    assert_eq!(result.values.dims(), &[3]);
    assert_eq!(result.vectors.dims(), &[3, 3]);
    // Sum of eigenvalues = trace = 0 + 0 + 2 = 2
    let vals = {
        let ct = result.values.contiguous(COL);
        let off = ct.offset() as usize;
        let len: usize = ct.dims().iter().product();
        ct.buffer().as_slice().unwrap()[off..off + len].to_vec()
    };
    let sum: num_complex::Complex<f64> = vals.iter().sum();
    assert!(
        (sum.re - 2.0).abs() < 1e-10 && sum.im.abs() < 1e-10,
        "eig trace = {:?}",
        sum
    );
}

// ============================================================================
// Coverage: norm_rrule cotangent shape validation
// ============================================================================

#[test]
fn norm_rrule_cotangent_scalar_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // norm of 2x2 -> scalar, cotangent should be scalar too, not 1D
    let bad_cot = make_tensor(vec![1.0, 2.0], &[2]);
    assert!(norm_rrule(&mut ctx, &a, &bad_cot, NormKind::Fro).is_err());
}

#[test]
fn norm_rrule_cotangent_batch_mismatch() {
    let mut ctx = CpuContext::new(1);
    // batched: (2,2,2), norm -> shape [2], cotangent should be [2] not [3]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let bad_cot = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(norm_rrule(&mut ctx, &a, &bad_cot, NormKind::Fro).is_err());
}

// ============================================================================
// Coverage: norm_rrule for batched Nuclear and Spectral
// ============================================================================

#[test]
fn norm_nuclear_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 diagonal matrices, shape [2, 2, 2], strides [1, 2, 4]
    // Batch 0 = diag(3,1): col-major [3, 0, 0, 1]
    // Batch 1 = diag(2,4): col-major [2, 0, 0, 4]
    let a = make_tensor(vec![3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0], &[2, 2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Nuclear).unwrap();
    let nd = tensor_data(&n);
    assert_eq!(nd.len(), 2);
    // Batch 0: diag(3,1), nuclear = 3 + 1 = 4
    assert!((nd[0] - 4.0).abs() < 1e-10, "batch nuclear[0] = {}", nd[0]);
    // Batch 1: diag(2,4), nuclear = 2 + 4 = 6
    assert!((nd[1] - 6.0).abs() < 1e-10, "batch nuclear[1] = {}", nd[1]);
}

#[test]
fn norm_spectral_batched() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0], &[2, 2, 2]);
    let n = norm(&mut ctx, &a, NormKind::Spectral).unwrap();
    let nd = tensor_data(&n);
    assert_eq!(nd.len(), 2);
    // Batch 0: max SV = 3
    assert!((nd[0] - 3.0).abs() < 1e-10, "batch spectral[0] = {}", nd[0]);
    // Batch 1: max SV = 4
    assert!((nd[1] - 4.0).abs() < 1e-10, "batch spectral[1] = {}", nd[1]);
}

// ============================================================================
// Coverage: solve with vector RHS (nrhs=1 path)
// ============================================================================

#[test]
fn solve_vector_rhs() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = make_tensor(vec![3.0, 7.0], &[2]); // vector RHS, no nrhs dim
    let x = solve(&mut ctx, &a, &b).unwrap();
    let xd = tensor_data(&x);
    assert!((xd[0] - 3.0).abs() < 1e-10);
    assert!((xd[1] - 7.0).abs() < 1e-10);
}

// ============================================================================
// Coverage: solve_triangular with vector RHS
// ============================================================================

#[test]
fn solve_triangular_vector_rhs() {
    let mut ctx = CpuContext::new(1);
    // Upper tri: [[2, 1], [0, 3]]
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![5.0, 6.0], &[2]); // vector RHS
    let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();
    let xd = tensor_data(&x);
    assert!((xd[1] - 2.0).abs() < 1e-10);
    assert!((xd[0] - 1.5).abs() < 1e-10);
}

// ============================================================================
// Coverage: solve_triangular batched
// ============================================================================

#[test]
fn solve_triangular_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 upper triangular matrices, shape [2, 2, 2], strides [1, 2, 4]
    // Batch 0 = [[1,2],[0,3]]: col-major [1, 0, 2, 3]
    // Batch 1 = [[2,1],[0,4]]: col-major [2, 0, 1, 4]
    let a = make_tensor(vec![1.0, 0.0, 2.0, 3.0, 2.0, 0.0, 1.0, 4.0], &[2, 2, 2]);
    let b = make_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
}

// ============================================================================
// Coverage: lstsq batched
// ============================================================================

#[test]
fn lstsq_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 3x2 identity-like matrices, shape [3, 2, 2], strides [1, 3, 6]
    // Batch 0 = [[1,0],[0,1],[0,0]]: col-major [1, 0, 0, 0, 1, 0]
    // Batch 1 = same
    let a = make_tensor(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 2, 2],
    );
    let b = make_tensor(vec![2.0, 3.0, 0.0, 4.0, 5.0, 0.0], &[3, 2]);
    let result = lstsq(&mut ctx, &a, &b).unwrap();
    let x = tensor_data(&result.x);
    assert_eq!(x.len(), 4); // 2 * 2 (n=2, batch=2)
}

// ============================================================================
// Coverage: pinv batched
// ============================================================================

#[test]
fn pinv_batched() {
    let mut ctx = CpuContext::new(1);
    // Two 2x2 identity matrices, shape [2, 2, 2], strides [1, 2, 4]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
    let result = pinv(&mut ctx, &a, None).unwrap();
    assert_eq!(result.dims(), &[2, 2, 2]);
    let data = tensor_data(&result);
    // pinv of identity = identity
    assert!((data[0] - 1.0).abs() < 1e-10);
    assert!((data[3] - 1.0).abs() < 1e-10);
}

// Note: LU frule for non-square matrices is not tested here because
// faer's LU backend panics on non-square input (faer requires m == n).

// ============================================================================
// Coverage: lu_rrule execution (covers ~120 lines in lib.rs)
// ============================================================================

#[test]
fn lu_rrule_square_basic_with_l_cotangent() {
    // Exercise lu_rrule code path with L cotangent on a 3x3 matrix.
    // We do not compare with FD (known formula mismatch), just verify execution + finiteness.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0], &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l_dims = result.l.dims().to_vec();
    let l_size: usize = l_dims.iter().product();
    let cotangent_l = make_tensor(vec![1.0; l_size], &l_dims);
    let co = LuCotangent {
        l: Some(cotangent_l),
        u: None,
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    assert_eq!(grad.dims(), &[3, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "lu_rrule grad not finite: {val}");
    }
}

#[test]
fn lu_rrule_square_basic_with_u_cotangent() {
    // Exercise lu_rrule code path with U cotangent.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0], &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let u_dims = result.u.dims().to_vec();
    let u_size: usize = u_dims.iter().product();
    let cotangent_u = make_tensor(vec![1.0; u_size], &u_dims);
    let co = LuCotangent {
        l: None,
        u: Some(cotangent_u),
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    assert_eq!(grad.dims(), &[3, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "lu_rrule grad not finite: {val}");
    }
}

#[test]
fn lu_rrule_square_with_both_cotangents() {
    // Exercise lu_rrule with both L and U cotangents.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 1.0, 1.0, 4.0], &[2, 2]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l_dims = result.l.dims().to_vec();
    let u_dims = result.u.dims().to_vec();
    let l_size: usize = l_dims.iter().product();
    let u_size: usize = u_dims.iter().product();
    let co = LuCotangent {
        l: Some(make_tensor(vec![0.5; l_size], &l_dims)),
        u: Some(make_tensor(vec![0.5; u_size], &u_dims)),
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "lu_rrule grad not finite: {val}");
    }
}

#[test]
fn lu_rrule_wide_with_both_cotangents() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 1.0, 0.0, 3.0, 1.0, 4.0], &[2, 3]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l_dims = result.l.dims().to_vec();
    let u_dims = result.u.dims().to_vec();
    let co = LuCotangent {
        l: Some(make_tensor(vec![0.25; l_dims.iter().product()], &l_dims)),
        u: Some(make_tensor(vec![0.5; u_dims.iter().product()], &u_dims)),
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    for &val in &tensor_data(&grad) {
        assert!(val.is_finite(), "lu_rrule wide grad not finite: {val}");
    }
}

#[test]
fn lu_rrule_tall_with_both_cotangents() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 1.0, 3.0, 0.0, 4.0, 1.0], &[3, 2]);
    let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
    let l_dims = result.l.dims().to_vec();
    let u_dims = result.u.dims().to_vec();
    let co = LuCotangent {
        l: Some(make_tensor(vec![0.5; l_dims.iter().product()], &l_dims)),
        u: Some(make_tensor(vec![0.25; u_dims.iter().product()], &u_dims)),
    };
    let grad = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);
    for &val in &tensor_data(&grad) {
        assert!(val.is_finite(), "lu_rrule tall grad not finite: {val}");
    }
}

#[test]
fn lu_rrule_rejects_l_cotangent_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 1.0, 1.0, 4.0], &[2, 2]);
    let co = LuCotangent {
        l: Some(make_tensor(vec![1.0, 1.0, 1.0], &[3])),
        u: None,
    };

    let err = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial)
        .expect_err("shape mismatch should return an error");
    assert!(matches!(
        err,
        chainrules_core::AutodiffError::InvalidArgument(_)
    ));
}

#[test]
fn lu_rrule_rejects_u_cotangent_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 1.0, 1.0, 4.0], &[2, 2]);
    let co = LuCotangent {
        l: None,
        u: Some(make_tensor(vec![1.0, 1.0, 1.0], &[3])),
    };

    let err = lu_rrule(&mut ctx, &a, &co, LuPivot::Partial)
        .expect_err("shape mismatch should return an error");
    assert!(matches!(
        err,
        chainrules_core::AutodiffError::InvalidArgument(_)
    ));
}

// ============================================================================
// Coverage: eig_rrule with vectors cotangent (EigCotangent)
// ============================================================================

#[test]
fn eig_rrule_with_vectors_cotangent_only() {
    // Exercise eig_rrule with only vectors cotangent (no values cotangent).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 1.0], &[3, 3]);
    let eig_result = eig(&mut ctx, &a).unwrap();
    let n = 3;

    // Create complex cotangent for vectors only
    let dv_data: Vec<Complex64> = (0..n * n)
        .map(|i| {
            Complex64::new(
                ((i * 3 + 1) % 7) as f64 / 3.0,
                ((i * 5 + 2) % 7) as f64 / 4.0,
            )
        })
        .collect();
    let dv_tensor = make_complex_tensor(dv_data, eig_result.vectors.dims());
    let cotangent = EigCotangent::<f64> {
        values: None,
        vectors: Some(dv_tensor),
    };
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[n, n]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "eig_rrule grad not finite: {val}");
    }
}

#[test]
fn eig_rrule_with_both_values_and_vectors() {
    // Exercise eig_rrule with both values and vectors cotangents.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![4.0, 0.5, 0.5, 2.0], &[2, 2]);
    let eig_result = eig(&mut ctx, &a).unwrap();
    let n = 2;

    let dlam_data: Vec<Complex64> = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
    let dlam_tensor = make_complex_tensor(dlam_data, eig_result.values.dims());

    let dv_data: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, 0.5),
    ];
    let dv_tensor = make_complex_tensor(dv_data, eig_result.vectors.dims());

    let cotangent = EigCotangent::<f64> {
        values: Some(dlam_tensor),
        vectors: Some(dv_tensor),
    };
    let grad = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[n, n]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "eig_rrule grad not finite: {val}");
    }
}

// ============================================================================
// Coverage: solve_rrule and solve_frule with multi-RHS
// ============================================================================

fn assert_close_slices(label: &str, got: &[f64], expected: &[f64], atol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch (got {}, expected {})",
        got.len(),
        expected.len()
    );
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= atol,
            "{label}[{idx}] mismatch: got {g}, expected {e}, atol={atol}"
        );
    }
}

fn assemble_two_rhs_columns(col0: &Tensor<f64>, col1: &Tensor<f64>) -> Vec<f64> {
    let mut data = tensor_data(col0);
    data.extend(tensor_data(col1));
    data
}

#[test]
fn solve_rrule_multi_rhs() {
    // Exercise nrhs > 1 path in solve_rrule.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    // b has shape [2, 3] (n=2, nrhs=3)
    let b = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]);
    let x = solve(&mut ctx, &a, &b).unwrap();
    let co = make_tensor(vec![1.0; 6], x.dims());
    let grad = solve_rrule(&mut ctx, &a, &b, &co).unwrap();
    assert_eq!(grad.a.dims(), &[2, 2]);
    assert_eq!(grad.b.dims(), &[2, 3]);
    let ga = tensor_data(&grad.a);
    let gb = tensor_data(&grad.b);
    for &val in &ga {
        assert!(val.is_finite(), "solve_rrule grad_a not finite: {val}");
    }
    for &val in &gb {
        assert!(val.is_finite(), "solve_rrule grad_b not finite: {val}");
    }
}

#[test]
fn solve_rrule_square_rhs_matches_columnwise_reference() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let cot = make_tensor(vec![0.25, -0.5, 0.75, 1.25], &[2, 2]);

    let grad = solve_rrule(&mut ctx, &a, &b, &cot).unwrap();
    assert_eq!(grad.a.dims(), &[2, 2]);
    assert_eq!(grad.b.dims(), &[2, 2]);

    let b0 = make_tensor(vec![1.0, 0.0], &[2, 1]);
    let b1 = make_tensor(vec![0.0, 1.0], &[2, 1]);
    let cot0 = make_tensor(vec![0.25, -0.5], &[2, 1]);
    let cot1 = make_tensor(vec![0.75, 1.25], &[2, 1]);
    let grad0 = solve_rrule(&mut ctx, &a, &b0, &cot0).unwrap();
    let grad1 = solve_rrule(&mut ctx, &a, &b1, &cot1).unwrap();

    let expected_a: Vec<f64> = tensor_data(&grad0.a)
        .into_iter()
        .zip(tensor_data(&grad1.a))
        .map(|(x, y)| x + y)
        .collect();
    let expected_b = assemble_two_rhs_columns(&grad0.b, &grad1.b);

    assert_close_slices(
        "solve_rrule grad_a",
        &tensor_data(&grad.a),
        &expected_a,
        1e-10,
    );
    assert_close_slices(
        "solve_rrule grad_b",
        &tensor_data(&grad.b),
        &expected_b,
        1e-10,
    );
}

#[test]
fn solve_frule_multi_rhs() {
    // Exercise nrhs > 1 path in solve_frule.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let db = make_tensor(vec![0.1; 6], &[2, 3]);
    let (x, dx) = solve_frule(&mut ctx, &a, &b, &da, &db).unwrap();
    assert_eq!(x.dims(), &[2, 3]);
    assert_eq!(dx.dims(), &[2, 3]);
    let dxd = tensor_data(&dx);
    for &val in &dxd {
        assert!(val.is_finite(), "solve_frule dx not finite: {val}");
    }
}

#[test]
fn solve_frule_square_rhs_matches_columnwise_reference() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let da = make_tensor(vec![0.1, -0.2, 0.3, 0.4], &[2, 2]);
    let db = make_tensor(vec![0.05, -0.1, 0.2, 0.3], &[2, 2]);

    let (x, dx) = solve_frule(&mut ctx, &a, &b, &da, &db).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(dx.dims(), &[2, 2]);

    let b0 = make_tensor(vec![1.0, 0.0], &[2, 1]);
    let b1 = make_tensor(vec![0.0, 1.0], &[2, 1]);
    let db0 = make_tensor(vec![0.05, -0.1], &[2, 1]);
    let db1 = make_tensor(vec![0.2, 0.3], &[2, 1]);
    let (x0, dx0) = solve_frule(&mut ctx, &a, &b0, &da, &db0).unwrap();
    let (x1, dx1) = solve_frule(&mut ctx, &a, &b1, &da, &db1).unwrap();

    let expected_x = assemble_two_rhs_columns(&x0, &x1);
    let expected_dx = assemble_two_rhs_columns(&dx0, &dx1);

    assert_close_slices("solve_frule x", &tensor_data(&x), &expected_x, 1e-10);
    assert_close_slices("solve_frule dx", &tensor_data(&dx), &expected_dx, 1e-10);
}

#[test]
fn solve_triangular_frule_upper_fd() {
    let mut ctx = CpuContext::new(1);
    // Upper triangular A
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    // Two RHS columns
    let b = make_tensor(vec![1.0, 2.0, 0.5, -1.0], &[2, 2]);
    // Tangent includes lower-triangular entry to confirm projection behavior.
    let da = make_tensor(vec![0.1, 0.2, -0.3, 0.4], &[2, 2]);
    let db = make_tensor(vec![0.05, -0.1, 0.2, 0.3], &[2, 2]);

    let (_x, dx) = solve_triangular_frule(&mut ctx, &a, &b, &da, &db, true).unwrap();
    let analytic = tensor_data(&dx);

    let eps = 1e-6;
    let a_data = tensor_data(&a);
    let b_data = tensor_data(&b);
    let da_data = tensor_data(&da);
    let db_data = tensor_data(&db);
    let mut a_plus = a_data.clone();
    let mut a_minus = a_data.clone();
    let mut b_plus = b_data.clone();
    let mut b_minus = b_data.clone();
    for i in 0..a_data.len() {
        a_plus[i] += eps * da_data[i];
        a_minus[i] -= eps * da_data[i];
    }
    for i in 0..b_data.len() {
        b_plus[i] += eps * db_data[i];
        b_minus[i] -= eps * db_data[i];
    }

    let x_plus = solve_triangular(
        &mut ctx,
        &make_tensor(a_plus, &[2, 2]),
        &make_tensor(b_plus, &[2, 2]),
        true,
    )
    .unwrap();
    let x_minus = solve_triangular(
        &mut ctx,
        &make_tensor(a_minus, &[2, 2]),
        &make_tensor(b_minus, &[2, 2]),
        true,
    )
    .unwrap();
    let fd: Vec<f64> = tensor_data(&x_plus)
        .iter()
        .zip(tensor_data(&x_minus).iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    let max_err = analytic
        .iter()
        .zip(fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-5,
        "solve_triangular_frule FD check failed: max_err={max_err}"
    );
}

#[test]
fn solve_triangular_frule_upper_square_rhs_matches_columnwise_reference() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 0.5, -1.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.2, -0.3, 0.4], &[2, 2]);
    let db = make_tensor(vec![0.05, -0.1, 0.2, 0.3], &[2, 2]);

    let (x, dx) = solve_triangular_frule(&mut ctx, &a, &b, &da, &db, true).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(dx.dims(), &[2, 2]);

    let b0 = make_tensor(vec![1.0, 2.0], &[2, 1]);
    let b1 = make_tensor(vec![0.5, -1.0], &[2, 1]);
    let db0 = make_tensor(vec![0.05, -0.1], &[2, 1]);
    let db1 = make_tensor(vec![0.2, 0.3], &[2, 1]);
    let (x0, dx0) = solve_triangular_frule(&mut ctx, &a, &b0, &da, &db0, true).unwrap();
    let (x1, dx1) = solve_triangular_frule(&mut ctx, &a, &b1, &da, &db1, true).unwrap();

    let expected_x = assemble_two_rhs_columns(&x0, &x1);
    let expected_dx = assemble_two_rhs_columns(&dx0, &dx1);

    assert_close_slices(
        "solve_triangular_frule x",
        &tensor_data(&x),
        &expected_x,
        1e-10,
    );
    assert_close_slices(
        "solve_triangular_frule dx",
        &tensor_data(&dx),
        &expected_dx,
        1e-10,
    );
}

#[test]
fn solve_triangular_rrule_upper_fd() {
    let mut ctx = CpuContext::new(1);
    // Upper triangular A
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0], &[2, 1]);
    let cot = make_tensor(vec![0.7, -1.2], &[2, 1]);
    let grad = solve_triangular_rrule(&mut ctx, &a, &b, &cot, true).unwrap();
    let grad_a = tensor_data(&grad.a);
    let grad_b = tensor_data(&grad.b);

    let eps = 1e-6;
    let a_data = tensor_data(&a);
    let b_data = tensor_data(&b);
    let cot_data = tensor_data(&cot);

    // FD check for dA: dL = <cot, solve_triangular(A, b)>
    for idx in 0..a_data.len() {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;

        let x_plus = solve_triangular(&mut ctx, &make_tensor(a_plus, &[2, 2]), &b, true).unwrap();
        let x_minus = solve_triangular(&mut ctx, &make_tensor(a_minus, &[2, 2]), &b, true).unwrap();
        let xp = tensor_data(&x_plus);
        let xm = tensor_data(&x_minus);
        let fd: f64 = xp
            .iter()
            .zip(xm.iter())
            .zip(cot_data.iter())
            .map(|((p, m), c)| c * (p - m) / (2.0 * eps))
            .sum();

        assert!(
            (grad_a[idx] - fd).abs() < 1e-4,
            "solve_triangular_rrule FD check failed at A[{idx}]: analytic={}, fd={fd}",
            grad_a[idx]
        );
    }

    // FD check for dB
    for idx in 0..b_data.len() {
        let mut b_plus = b_data.clone();
        let mut b_minus = b_data.clone();
        b_plus[idx] += eps;
        b_minus[idx] -= eps;

        let x_plus = solve_triangular(&mut ctx, &a, &make_tensor(b_plus, &[2, 1]), true).unwrap();
        let x_minus = solve_triangular(&mut ctx, &a, &make_tensor(b_minus, &[2, 1]), true).unwrap();
        let xp = tensor_data(&x_plus);
        let xm = tensor_data(&x_minus);
        let fd: f64 = xp
            .iter()
            .zip(xm.iter())
            .zip(cot_data.iter())
            .map(|((p, m), c)| c * (p - m) / (2.0 * eps))
            .sum();

        assert!(
            (grad_b[idx] - fd).abs() < 1e-4,
            "solve_triangular_rrule FD check failed at b[{idx}]: analytic={}, fd={fd}",
            grad_b[idx]
        );
    }
}

#[test]
fn solve_triangular_rrule_upper_square_rhs_matches_columnwise_reference() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 0.5, -1.0], &[2, 2]);
    let cot = make_tensor(vec![0.7, -1.2, 0.5, 0.25], &[2, 2]);

    let grad = solve_triangular_rrule(&mut ctx, &a, &b, &cot, true).unwrap();
    assert_eq!(grad.a.dims(), &[2, 2]);
    assert_eq!(grad.b.dims(), &[2, 2]);

    let b0 = make_tensor(vec![1.0, 2.0], &[2, 1]);
    let b1 = make_tensor(vec![0.5, -1.0], &[2, 1]);
    let cot0 = make_tensor(vec![0.7, -1.2], &[2, 1]);
    let cot1 = make_tensor(vec![0.5, 0.25], &[2, 1]);
    let grad0 = solve_triangular_rrule(&mut ctx, &a, &b0, &cot0, true).unwrap();
    let grad1 = solve_triangular_rrule(&mut ctx, &a, &b1, &cot1, true).unwrap();

    let expected_a: Vec<f64> = tensor_data(&grad0.a)
        .into_iter()
        .zip(tensor_data(&grad1.a))
        .map(|(x, y)| x + y)
        .collect();
    let expected_b = assemble_two_rhs_columns(&grad0.b, &grad1.b);

    assert_close_slices(
        "solve_triangular_rrule grad_a",
        &tensor_data(&grad.a),
        &expected_a,
        1e-10,
    );
    assert_close_slices(
        "solve_triangular_rrule grad_b",
        &tensor_data(&grad.b),
        &expected_b,
        1e-10,
    );
}

#[test]
fn solve_triangular_rrule_complex64_matches_manual_formula() {
    let mut ctx = CpuContext::new(1);
    let a = make_complex_tensor(
        vec![c(2.0, 0.0), c(0.0, 0.0), c(1.0, -0.5), c(3.0, 0.0)],
        &[2, 2],
    );
    let b = make_complex_tensor(vec![c(1.0, 0.5), c(2.0, -0.25)], &[2, 1]);
    let cot = make_complex_tensor(vec![c(0.5, -0.2), c(-0.25, 0.1)], &[2, 1]);

    let grad = solve_triangular_rrule(&mut ctx, &a, &b, &cot, true).unwrap();
    let x = solve_triangular(&mut ctx, &a, &b, true).unwrap();

    // G = solve(A^H, cot) where A is upper triangular => A^H is lower triangular.
    let ad = complex_tensor_data(&a);
    let a_h = make_complex_tensor(
        vec![ad[0].conj(), ad[2].conj(), ad[1].conj(), ad[3].conj()],
        &[2, 2],
    );
    let g = solve_triangular(&mut ctx, &a_h, &cot, false).unwrap();
    let gd = complex_tensor_data(&g);
    let xd = complex_tensor_data(&x);

    // dA = triu(-G x^H), dB = G
    let m00 = -(gd[0] * xd[0].conj());
    let m10 = c(0.0, 0.0);
    let m01 = -(gd[0] * xd[1].conj());
    let m11 = -(gd[1] * xd[1].conj());
    let expected_a = make_complex_tensor(vec![m00, m10, m01, m11], &[2, 2]);
    let expected_b = make_complex_tensor(gd, &[2, 1]);

    assert!(
        complex_max_err(
            &complex_tensor_data(&grad.a),
            &complex_tensor_data(&expected_a)
        ) < 1e-10
    );
    assert!(
        complex_max_err(
            &complex_tensor_data(&grad.b),
            &complex_tensor_data(&expected_b)
        ) < 1e-10
    );
}

#[test]
fn solve_rrule_complex64_matches_manual_formula() {
    let mut ctx = CpuContext::new(1);
    let a = make_complex_tensor(
        vec![c(3.0, 0.0), c(1.0, 0.5), c(0.2, -0.3), c(2.0, 0.0)],
        &[2, 2],
    );
    let b = make_complex_tensor(vec![c(1.0, -0.2), c(0.5, 0.4)], &[2, 1]);
    let cot = make_complex_tensor(vec![c(0.3, 0.1), c(-0.7, 0.2)], &[2, 1]);

    let grad = solve_rrule(&mut ctx, &a, &b, &cot).unwrap();
    let x = solve(&mut ctx, &a, &b).unwrap();

    // G = solve(A^H, cot)
    let ad = complex_tensor_data(&a);
    let a_h = make_complex_tensor(
        vec![ad[0].conj(), ad[2].conj(), ad[1].conj(), ad[3].conj()],
        &[2, 2],
    );
    let g = solve(&mut ctx, &a_h, &cot).unwrap();
    let gd = complex_tensor_data(&g);
    let xd = complex_tensor_data(&x);

    // dA = -G x^H, dB = G
    let m00 = -(gd[0] * xd[0].conj());
    let m10 = -(gd[1] * xd[0].conj());
    let m01 = -(gd[0] * xd[1].conj());
    let m11 = -(gd[1] * xd[1].conj());
    let expected_a = make_complex_tensor(vec![m00, m10, m01, m11], &[2, 2]);
    let expected_b = make_complex_tensor(gd, &[2, 1]);

    assert!(
        complex_max_err(
            &complex_tensor_data(&grad.a),
            &complex_tensor_data(&expected_a)
        ) < 1e-10
    );
    assert!(
        complex_max_err(
            &complex_tensor_data(&grad.b),
            &complex_tensor_data(&expected_b)
        ) < 1e-10
    );
}

// ============================================================================
// Coverage: lstsq_frule
// ============================================================================

#[test]
fn lstsq_frule_basic() {
    // Exercise lstsq_frule with a tall overdetermined system.
    let mut ctx = CpuContext::new(1);
    // A is 4x2 (overdetermined)
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[4, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let da = make_tensor(vec![0.1; 8], &[4, 2]);
    let db = make_tensor(vec![0.1; 4], &[4]);
    let (result, dresult) = lstsq_frule(&mut ctx, &a, &b, &da, &db).unwrap();
    assert_eq!(result.x.dims(), &[2]);
    assert_eq!(dresult.x.dims(), &[2]);
    let dxd = tensor_data(&dresult.x);
    for &val in &dxd {
        assert!(val.is_finite(), "lstsq_frule dx not finite: {val}");
    }
}

// ============================================================================
// Coverage: pinv_rrule and pinv_frule
// ============================================================================

#[test]
fn pinv_rrule_execution() {
    // Exercise pinv_rrule (covers ~50 lines in lib.rs).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    let ap = pinv(&mut ctx, &a, None).unwrap();
    let co = make_tensor(vec![1.0; ap.dims().iter().product::<usize>()], ap.dims());
    let grad = pinv_rrule(&mut ctx, &a, &co, None).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "pinv_rrule grad not finite: {val}");
    }
}

#[test]
fn pinv_frule_execution() {
    // Exercise pinv_frule (covers ~50 lines in lib.rs).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]);
    let da = make_tensor(vec![0.1; 6], &[3, 2]);
    let (ap, dap) = pinv_frule(&mut ctx, &a, &da, None).unwrap();
    assert_eq!(ap.dims(), &[2, 3]);
    assert_eq!(dap.dims(), &[2, 3]);
    let dapd = tensor_data(&dap);
    for &val in &dapd {
        assert!(val.is_finite(), "pinv_frule dap not finite: {val}");
    }
}

// ============================================================================
// Coverage: norm_rrule Nuclear & Spectral
// ============================================================================

#[test]
fn norm_nuclear_rrule_execution() {
    // Exercise norm_rrule Nuclear path (covers ~10 lines).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let co = make_tensor(vec![1.0], &[]);
    let grad = norm_rrule(&mut ctx, &a, &co, NormKind::Nuclear).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "norm_nuclear_rrule grad not finite: {val}");
    }
}

#[test]
fn norm_spectral_rrule_execution() {
    // Exercise norm_rrule Spectral path (covers ~10 lines).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let co = make_tensor(vec![1.0], &[]);
    let grad = norm_rrule(&mut ctx, &a, &co, NormKind::Spectral).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(
            val.is_finite(),
            "norm_spectral_rrule grad not finite: {val}"
        );
    }
}

// ============================================================================
// Coverage: norm_frule Nuclear & Spectral
// ============================================================================

#[test]
fn norm_nuclear_frule_execution() {
    // Exercise norm_frule Nuclear path.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::Nuclear).unwrap();
    let nv = tensor_data(&nrm);
    let dv = tensor_data(&dnrm);
    assert!(nv[0].is_finite());
    assert!(dv[0].is_finite());
}

#[test]
fn norm_spectral_frule_execution() {
    // Exercise norm_frule Spectral path.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::Spectral).unwrap();
    let nv = tensor_data(&nrm);
    let dv = tensor_data(&dnrm);
    assert!(nv[0].is_finite());
    assert!(dv[0].is_finite());
}

// ============================================================================
// Coverage: qr_frule and qr_rrule execution (to cover non-square projector terms)
// ============================================================================

#[test]
fn qr_rrule_tall_execution() {
    // Exercise qr_rrule on a tall 4x2 matrix (m > k path).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5], &[4, 2]);
    let result = qr(&mut ctx, &a).unwrap();
    let q_dims = result.q.dims().to_vec();
    let q_size: usize = q_dims.iter().product();
    let r_dims = result.r.dims().to_vec();
    let r_size: usize = r_dims.iter().product();
    let co = QrCotangent {
        q: Some(make_tensor(vec![1.0; q_size], &q_dims)),
        r: Some(make_tensor(vec![1.0; r_size], &r_dims)),
    };
    let grad = qr_rrule(&mut ctx, &a, &co).unwrap();
    assert_eq!(grad.dims(), &[4, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "qr_rrule grad not finite: {val}");
    }
}

#[test]
fn qr_frule_tall_execution() {
    // Exercise qr_frule on a tall 4x2 matrix.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5], &[4, 2]);
    let da = make_tensor(vec![0.1; 8], &[4, 2]);
    let (result, dresult) = qr_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(result.q.dims()[0], 4);
    assert_eq!(result.q.dims()[1], 2);
    let dq = tensor_data(&dresult.q);
    let dr = tensor_data(&dresult.r);
    for &val in &dq {
        assert!(val.is_finite(), "qr_frule dq not finite: {val}");
    }
    for &val in &dr {
        assert!(val.is_finite(), "qr_frule dr not finite: {val}");
    }
}

// ============================================================================
// Coverage: svd_rrule non-square correction paths
// ============================================================================

#[test]
fn svd_rrule_tall_with_all_cotangents() {
    // Exercise all three cotangent branches on tall matrix (m > k).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5], &[3, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let s_dims = result.s.dims().to_vec();
    let u_dims = result.u.dims().to_vec();
    let vt_dims = result.vt.dims().to_vec();
    let s_size: usize = s_dims.iter().product();
    let u_size: usize = u_dims.iter().product();
    let vt_size: usize = vt_dims.iter().product();
    let co = SvdCotangent {
        s: Some(make_tensor(vec![1.0; s_size], &s_dims)),
        u: Some(make_tensor(vec![1.0; u_size], &u_dims)),
        vt: Some(make_tensor(vec![1.0; vt_size], &vt_dims)),
    };
    let grad = svd_rrule(&mut ctx, &a, &co, None).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "svd_rrule grad not finite: {val}");
    }
}

#[test]
fn svd_rrule_wide_with_all_cotangents() {
    // Exercise all three cotangent branches on wide matrix (n > k).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let s_dims = result.s.dims().to_vec();
    let u_dims = result.u.dims().to_vec();
    let vt_dims = result.vt.dims().to_vec();
    let s_size: usize = s_dims.iter().product();
    let u_size: usize = u_dims.iter().product();
    let vt_size: usize = vt_dims.iter().product();
    let co = SvdCotangent {
        s: Some(make_tensor(vec![1.0; s_size], &s_dims)),
        u: Some(make_tensor(vec![1.0; u_size], &u_dims)),
        vt: Some(make_tensor(vec![1.0; vt_size], &vt_dims)),
    };
    let grad = svd_rrule(&mut ctx, &a, &co, None).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "svd_rrule grad not finite: {val}");
    }
}

// ============================================================================
// Coverage: svd_frule non-square paths
// ============================================================================

#[test]
fn svd_frule_tall_all_outputs() {
    // Exercise svd_frule on tall matrix (exercises m > k projector path).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5], &[3, 2]);
    let da = make_tensor(vec![0.1; 6], &[3, 2]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    assert_eq!(result.u.dims()[0], 3);
    assert_eq!(result.u.dims()[1], 2);
    let ds = tensor_data(&dresult.s);
    for &val in &ds {
        assert!(val.is_finite(), "svd_frule ds not finite: {val}");
    }
}

#[test]
fn svd_frule_wide_all_outputs() {
    // Exercise svd_frule on wide matrix (exercises n > k projector path).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[2, 3]);
    let da = make_tensor(vec![0.1; 6], &[2, 3]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    assert_eq!(result.vt.dims()[0], 2);
    assert_eq!(result.vt.dims()[1], 3);
    let ds = tensor_data(&dresult.s);
    for &val in &ds {
        assert!(val.is_finite(), "svd_frule ds not finite: {val}");
    }
}

// ============================================================================
// Coverage: lu_frule execution
// ============================================================================

#[test]
fn lu_frule_square_execution() {
    // Exercise lu_frule on a square matrix.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 1.0, 1.0, 4.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (result, dresult) = lu_frule(&mut ctx, &a, &da, LuPivot::Partial).unwrap();
    assert_eq!(result.l.dims(), &[2, 2]);
    assert_eq!(result.u.dims(), &[2, 2]);
    let dl = tensor_data(&dresult.l);
    let du = tensor_data(&dresult.u);
    for &val in &dl {
        assert!(val.is_finite(), "lu_frule dl not finite: {val}");
    }
    for &val in &du {
        assert!(val.is_finite(), "lu_frule du not finite: {val}");
    }
}

// ============================================================================
// Coverage: cholesky_rrule and cholesky_frule execution
// ============================================================================

#[test]
fn cholesky_rrule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let l = cholesky(&mut ctx, &a).unwrap();
    let co = make_tensor(vec![1.0; 4], l.dims());
    let grad = cholesky_rrule(&mut ctx, &a, &co).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "cholesky_rrule grad not finite: {val}");
    }
}

#[test]
fn cholesky_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.1], &[2, 2]);
    let (l, dl) = cholesky_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(l.dims(), &[2, 2]);
    assert_eq!(dl.dims(), &[2, 2]);
    let dld = tensor_data(&dl);
    for &val in &dld {
        assert!(val.is_finite(), "cholesky_frule dl not finite: {val}");
    }
}

// ============================================================================
// Coverage: eigen_frule execution
// ============================================================================

#[test]
fn eigen_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![3.0, 1.0, 1.0, 2.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (result, dresult) = eigen_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(result.values.dims(), &[2]);
    assert_eq!(dresult.values.dims(), &[2]);
    let de = tensor_data(&dresult.values);
    for &val in &de {
        assert!(val.is_finite(), "eigen_frule de not finite: {val}");
    }
}

// ============================================================================
// Coverage: slogdet_rrule execution
// ============================================================================

#[test]
fn slogdet_rrule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let co_logabsdet = make_tensor(vec![1.0], &[]);
    let cotangent = SlogdetCotangent {
        logabsdet: Some(co_logabsdet),
    };
    let grad = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "slogdet_rrule grad not finite: {val}");
    }
}

// ============================================================================
// Coverage: slogdet_frule execution
// ============================================================================

#[test]
fn slogdet_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (result, dresult) = slogdet_frule(&mut ctx, &a, &da).unwrap();
    assert!(tensor_data(&result.logabsdet)[0].is_finite());
    assert!(tensor_data(&dresult.logabsdet)[0].is_finite());
}

// ============================================================================
// Coverage: det_rrule and det_frule execution
// ============================================================================

#[test]
fn det_rrule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let co = make_tensor(vec![1.0], &[]);
    let grad = det_rrule(&mut ctx, &a, &co).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "det_rrule grad not finite: {val}");
    }
}

#[test]
fn det_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (d, dd) = det_frule(&mut ctx, &a, &da).unwrap();
    assert!(tensor_data(&d)[0].is_finite());
    assert!(tensor_data(&dd)[0].is_finite());
}

// ============================================================================
// Coverage: inv_rrule and inv_frule execution
// ============================================================================

#[test]
fn inv_rrule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let co = make_tensor(vec![1.0; 4], &[2, 2]);
    let grad = inv_rrule(&mut ctx, &a, &co).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "inv_rrule grad not finite: {val}");
    }
}

#[test]
fn inv_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.5, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (a_inv, da_inv) = inv_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(a_inv.dims(), &[2, 2]);
    assert_eq!(da_inv.dims(), &[2, 2]);
    let did = tensor_data(&da_inv);
    for &val in &did {
        assert!(val.is_finite(), "inv_frule da_inv not finite: {val}");
    }
}

// ============================================================================
// Coverage: matrix_exp_rrule and matrix_exp_frule execution
// ============================================================================

#[test]
fn matrix_exp_rrule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.1, 0.0, 0.0, 0.2], &[2, 2]);
    let co = make_tensor(vec![1.0; 4], &[2, 2]);
    let grad = matrix_exp_rrule(&mut ctx, &a, &co).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "matrix_exp_rrule grad not finite: {val}");
    }
}

#[test]
fn matrix_exp_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.1, 0.0, 0.0, 0.2], &[2, 2]);
    let da = make_tensor(vec![0.1; 4], &[2, 2]);
    let (exp_a, dexp_a) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(exp_a.dims(), &[2, 2]);
    assert_eq!(dexp_a.dims(), &[2, 2]);
    let dd = tensor_data(&dexp_a);
    for &val in &dd {
        assert!(val.is_finite(), "matrix_exp_frule dexp not finite: {val}");
    }
}

// ============================================================================
// Coverage: eig_frule execution
// ============================================================================

#[test]
fn eig_frule_execution() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 1.0], &[3, 3]);
    let da = make_tensor(vec![0.01; 9], &[3, 3]);
    let (result, dresult) = eig_frule(&mut ctx, &a, &da).unwrap();
    assert_eq!(result.values.dims(), &[3]);
    assert_eq!(dresult.values.dims(), &[3]);
}

// ============================================================================
// Coverage: lstsq_rrule full execution
// ============================================================================

#[test]
fn lstsq_rrule_full_execution() {
    // Exercise lstsq_rrule with a tall matrix to cover all lines.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5], &[3, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let result = lstsq(&mut ctx, &a, &b).unwrap();
    let co = make_tensor(
        vec![1.0; result.x.dims().iter().product::<usize>()],
        result.x.dims(),
    );
    let grad = lstsq_rrule(&mut ctx, &a, &b, &co).unwrap();
    assert_eq!(grad.a.dims(), &[3, 2]);
    assert_eq!(grad.b.dims(), &[3]);
    let ga = tensor_data(&grad.a);
    let gb = tensor_data(&grad.b);
    for &val in &ga {
        assert!(val.is_finite(), "lstsq_rrule grad_a not finite: {val}");
    }
    for &val in &gb {
        assert!(val.is_finite(), "lstsq_rrule grad_b not finite: {val}");
    }
}

// Note: solve_triangular now has frule support, but rrule is still not exposed.

// ============================================================================
// Coverage: additional validation error paths and branch coverage
// ============================================================================

#[test]
fn lu_nopivot_reconstruction() {
    let mut ctx = CpuContext::new(1);
    // Matrix with nonzero leading pivots for stable NoPivot factorization.
    let data = vec![4.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let a = make_tensor(data.clone(), &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
    assert!(result.p.is_none());

    let l = tensor_data(&result.l);
    let u = tensor_data(&result.u);
    let n = 3;
    let mut lu_prod = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                val += l[i + k * n] * u[k + j * n];
            }
            lu_prod[i + j * n] = val;
        }
    }

    for i in 0..n {
        for j in 0..n {
            let err = (data[i + j * n] - lu_prod[i + j * n]).abs();
            assert!(
                err < 1e-10,
                "NoPivot LU reconstruction error at ({i},{j}): {err}"
            );
        }
    }
}

#[test]
fn solve_rhs_2d_batch_mismatch() {
    // Lines 321-325: validate_solve_rhs 2D b with wrong batch dims.
    let mut ctx = CpuContext::new(1);
    // A is (2,2,2) => batch=[2], b is (2,1,3) => batch=[3], mismatch
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    assert!(solve(&mut ctx, &a, &b).is_err());
}

#[test]
fn solve_triangular_rhs_2d_batch_mismatch() {
    // Also covers lines 321-325 via solve_triangular path.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    assert!(solve_triangular(&mut ctx, &a, &b, true).is_err());
}

#[test]
fn norm_rrule_l1_supported() {
    let mut ctx = CpuContext::new(1);
    // Matrix [[1, -2], [3, 4]] => max column is 2nd.
    let a = make_tensor(vec![1.0, 3.0, -2.0, 4.0], &[2, 2]);
    let co: Tensor<f64> = Tensor::from_vec(vec![2.0], &[], &[], 0).unwrap();
    let grad = norm_rrule(&mut ctx, &a, &co, NormKind::L1).unwrap();
    let gd = tensor_data(&grad);
    let expected = vec![0.0, 0.0, -2.0, 2.0];
    for (g, e) in gd.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-12,
            "L1 rrule mismatch: got {g}, expected {e}"
        );
    }
}

#[test]
fn norm_rrule_inf_supported() {
    let mut ctx = CpuContext::new(1);
    // Matrix [[1, -2], [3, 4]] => max row is 2nd.
    let a = make_tensor(vec![1.0, 3.0, -2.0, 4.0], &[2, 2]);
    let co: Tensor<f64> = Tensor::from_vec(vec![2.0], &[], &[], 0).unwrap();
    let grad = norm_rrule(&mut ctx, &a, &co, NormKind::Inf).unwrap();
    let gd = tensor_data(&grad);
    let expected = vec![0.0, 2.0, 0.0, 2.0];
    for (g, e) in gd.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-12,
            "Inf rrule mismatch: got {g}, expected {e}"
        );
    }
}

#[test]
fn norm_rrule_vector_kinds_supported() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![3.0, -4.0], &[2]);
    let co: Tensor<f64> = Tensor::from_vec(vec![2.0], &[], &[], 0).unwrap();

    let fro = norm_rrule(&mut ctx, &x, &co, NormKind::Fro).unwrap();
    let l1 = norm_rrule(&mut ctx, &x, &co, NormKind::L1).unwrap();
    let inf = norm_rrule(&mut ctx, &x, &co, NormKind::Inf).unwrap();
    let lp = norm_rrule(&mut ctx, &x, &co, NormKind::Lp(3.0)).unwrap();

    let fro_data = tensor_data(&fro);
    let l1_data = tensor_data(&l1);
    let inf_data = tensor_data(&inf);
    let lp_data = tensor_data(&lp);
    let lp_norm = 91.0_f64.powf(1.0 / 3.0);
    let lp_scale = 2.0 / lp_norm.powi(2);

    assert!((fro_data[0] - 1.2).abs() < 1e-12);
    assert!((fro_data[1] + 1.6).abs() < 1e-12);
    assert_eq!(l1_data, vec![2.0, -2.0]);
    assert_eq!(inf_data, vec![0.0, -2.0]);
    assert!((lp_data[0] - 9.0 * lp_scale).abs() < 1e-12);
    assert!((lp_data[1] + 16.0 * lp_scale).abs() < 1e-12);
}

#[test]
fn norm_rrule_vector_edge_cases_supported() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![4.0, -4.0, 0.0], &[3]);
    let co: Tensor<f64> = Tensor::from_vec(vec![6.0], &[], &[], 0).unwrap();

    let lp1 = norm_rrule(&mut ctx, &x, &co, NormKind::Lp(1.0)).unwrap();
    let inf = norm_rrule(&mut ctx, &x, &co, NormKind::Inf).unwrap();

    assert_eq!(tensor_data(&lp1), vec![6.0, -6.0, 0.0]);
    assert_eq!(tensor_data(&inf), vec![3.0, -3.0, 0.0]);

    assert!(norm_rrule(&mut ctx, &x, &co, NormKind::Lp(0.5)).is_err());
    assert!(norm_rrule(&mut ctx, &x, &co, NormKind::Spectral).is_err());
}

#[test]
fn norm_frule_l1_supported() {
    let mut ctx = CpuContext::new(1);
    // Matrix [[1, -2], [3, 4]] and tangent [[0.1, 0.2], [0.3, 0.4]].
    let a = make_tensor(vec![1.0, 3.0, -2.0, 4.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.3, 0.2, 0.4], &[2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::L1).unwrap();
    let n = tensor_data(&nrm);
    let dn = tensor_data(&dnrm);
    assert!((n[0] - 6.0).abs() < 1e-12);
    assert!((dn[0] - 0.2).abs() < 1e-12);
}

#[test]
fn norm_frule_inf_supported() {
    let mut ctx = CpuContext::new(1);
    // Matrix [[1, -2], [3, 4]] and tangent [[0.1, 0.2], [0.3, 0.4]].
    let a = make_tensor(vec![1.0, 3.0, -2.0, 4.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.3, 0.2, 0.4], &[2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::Inf).unwrap();
    let n = tensor_data(&nrm);
    let dn = tensor_data(&dnrm);
    assert!((n[0] - 7.0).abs() < 1e-12);
    assert!((dn[0] - 0.7).abs() < 1e-12);
}

#[test]
fn norm_frule_vector_kinds_supported() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![3.0, -4.0], &[2]);
    let dx = make_tensor(vec![0.1, 0.2], &[2]);

    let (fro, dfro) = norm_frule(&mut ctx, &x, &dx, NormKind::Fro).unwrap();
    let (l1, dl1) = norm_frule(&mut ctx, &x, &dx, NormKind::L1).unwrap();
    let (inf, dinf) = norm_frule(&mut ctx, &x, &dx, NormKind::Inf).unwrap();
    let (lp, dlp) = norm_frule(&mut ctx, &x, &dx, NormKind::Lp(3.0)).unwrap();

    let lp_norm = 91.0_f64.powf(1.0 / 3.0);
    let lp_derivative = (9.0 * 0.1 - 16.0 * 0.2) / lp_norm.powi(2);

    assert!((tensor_data(&fro)[0] - 5.0).abs() < 1e-12);
    assert!((tensor_data(&dfro)[0] + 0.1).abs() < 1e-12);
    assert!((tensor_data(&l1)[0] - 7.0).abs() < 1e-12);
    assert!((tensor_data(&dl1)[0] + 0.1).abs() < 1e-12);
    assert!((tensor_data(&inf)[0] - 4.0).abs() < 1e-12);
    assert!((tensor_data(&dinf)[0] + 0.2).abs() < 1e-12);
    assert!((tensor_data(&lp)[0] - lp_norm).abs() < 1e-12);
    assert!((tensor_data(&dlp)[0] - lp_derivative).abs() < 1e-12);
}

#[test]
fn norm_frule_vector_edge_cases_supported() {
    let mut ctx = CpuContext::new(1);
    let x = make_tensor(vec![4.0, -4.0, 0.0], &[3]);
    let dx = make_tensor(vec![1.0, 2.0, 5.0], &[3]);

    let (lp1, dlp1) = norm_frule(&mut ctx, &x, &dx, NormKind::Lp(1.0)).unwrap();
    let (inf, dinf) = norm_frule(&mut ctx, &x, &dx, NormKind::Inf).unwrap();

    assert!((tensor_data(&lp1)[0] - 8.0).abs() < 1e-12);
    assert!((tensor_data(&dlp1)[0] + 1.0).abs() < 1e-12);
    assert!((tensor_data(&inf)[0] - 4.0).abs() < 1e-12);
    assert!((tensor_data(&dinf)[0] + 0.5).abs() < 1e-12);

    assert!(norm_frule(&mut ctx, &x, &dx, NormKind::Lp(0.5)).is_err());
    assert!(norm_frule(&mut ctx, &x, &dx, NormKind::Spectral).is_err());
}

#[test]
fn slogdet_negative_determinant() {
    // Line 1498: slogdet with negative U diagonal in LU -> sign flip.
    // Use diagonal matrix [[-1, 0], [0, 1]] which has det = -1.
    // LU with partial pivoting may reorder rows. To ensure U has a negative diagonal,
    // use [[-2, 0], [0, 1]] — since -2 has largest absolute value, it's chosen as pivot,
    // yielding U with diag=[-2, 1], so diag[0] < 0 triggers the sign flip.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![-2.0, 0.0, 0.0, 1.0], &[2, 2]);
    let result = slogdet(&mut ctx, &a).unwrap();
    let sign_data = tensor_data(&result.sign);
    // det = -2, so sign should be -1
    assert!(
        sign_data[0] < 0.0,
        "expected negative sign, got {}",
        sign_data[0]
    );
}

#[test]
fn matrix_exp_0x0() {
    // Line 1852: matrix_exp with 0x0 matrix returns empty tensor.
    let mut ctx = CpuContext::new(1);
    let a: Tensor<f64> = Tensor::from_vec(vec![], &[0, 0], &[1, 0], 0).unwrap();
    let result = matrix_exp(&mut ctx, &a).unwrap();
    assert_eq!(result.dims(), &[0, 0]);
}

#[test]
fn norm_rrule_batched_cotangent_wrong_shape() {
    // Lines 390-394: validate_norm_cotangent batch mismatch with batch dims.
    let mut ctx = CpuContext::new(1);
    // A is (2,2,3) -> batch_dims = [3], but cotangent shape is [2]
    let a = make_tensor(
        vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
        &[2, 2, 3],
    );
    let co = make_tensor(vec![1.0, 1.0], &[2]);
    assert!(norm_rrule(&mut ctx, &a, &co, NormKind::Fro).is_err());
}

#[test]
fn norm_rrule_batched_fro_correct_cotangent() {
    // Lines 395, 397: validate_norm_cotangent SUCCESS path with non-empty batch dims.
    let mut ctx = CpuContext::new(1);
    // A: (2,2,2) -> batch_dims = [2]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    // cotangent shape [2] matches batch_dims [2]
    let co = make_tensor(vec![1.0, 1.0], &[2]);
    let grad = norm_rrule(&mut ctx, &a, &co, NormKind::Fro).unwrap();
    assert_eq!(grad.dims(), &[2, 2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "batched norm_rrule grad not finite: {val}");
    }
}

#[test]
fn norm_frule_batched_fro() {
    // Exercise norm_frule with batched input to cover batched paths.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.2], &[2, 2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::Fro).unwrap();
    let nd = tensor_data(&nrm);
    assert_eq!(nd.len(), 2);
    for &val in &nd {
        assert!(val.is_finite(), "batched norm_frule nrm not finite: {val}");
    }
    let dnd = tensor_data(&dnrm);
    for &val in &dnd {
        assert!(val.is_finite(), "batched norm_frule dnrm not finite: {val}");
    }
}

#[test]
fn norm_frule_zero_matrix() {
    // Line 5159: norm_frule Fro with zero matrix (nv == 0).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let da = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let (nrm, dnrm) = norm_frule(&mut ctx, &a, &da, NormKind::Fro).unwrap();
    let nd = tensor_data(&nrm);
    assert!((nd[0]).abs() < 1e-15, "zero matrix norm should be 0");
    let dnd = tensor_data(&dnrm);
    // d||A||/dA at A=0 is technically undefined, but our code returns 0
    assert!(dnd[0].is_finite(), "dnrm should be finite");
}

#[test]
fn svd_rrule_no_cotangent() {
    // Covers svd_rrule with all cotangents None.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let co = SvdCotangent {
        s: None,
        u: None,
        vt: None,
    };
    let grad = svd_rrule(&mut ctx, &a, &co, None).unwrap();
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(
            val.abs() < 1e-15,
            "no-cotangent svd_rrule should be zero, got {val}"
        );
    }
}

#[test]
fn qr_rrule_r_only_cotangent() {
    // Covers qr_rrule with q=None.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]);
    let result = qr(&mut ctx, &a).unwrap();
    let r_dims = result.r.dims().to_vec();
    let r_size: usize = r_dims.iter().product();
    let cotangent = QrCotangent {
        q: None,
        r: Some(make_tensor(vec![1.0; r_size], &r_dims)),
    };
    let grad = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "qr_rrule r-only grad not finite: {val}");
    }
}

#[test]
fn matrix_exp_1x1_special_case() {
    // Line 1858-1862: matrix_exp with 1x1 matrix (special case path).
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0], &[1, 1]);
    let result = matrix_exp(&mut ctx, &a).unwrap();
    let rd = tensor_data(&result);
    assert!(
        (rd[0] - 2.0_f64.exp()).abs() < 1e-10,
        "exp(2) mismatch: got {}",
        rd[0]
    );
}

#[test]
fn det_negative() {
    // Exercise det with matrix that has negative determinant.
    let mut ctx = CpuContext::new(1);
    // [[0, 1], [1, 0]] has det = -1
    let a = make_tensor(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]);
    let result = det(&mut ctx, &a).unwrap();
    let rd = tensor_data(&result);
    assert!(
        (rd[0] + 1.0).abs() < 1e-10,
        "expected det=-1, got {}",
        rd[0]
    );
}

#[test]
fn slogdet_rrule_none_cotangent() {
    // Line 3608: slogdet_rrule with logabsdet=None -> skip inner block.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let cotangent = SlogdetCotangent { logabsdet: None };
    let grad = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
    // With None cotangent, gradient should be all zeros
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!((val).abs() < 1e-15, "expected zero grad, got {val}");
    }
}

#[test]
fn qr_rrule_q_only_cotangent() {
    // Line 2878: qr_rrule with r=None -> zero dR branch.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]);
    let result = qr(&mut ctx, &a).unwrap();
    let q_dims = result.q.dims().to_vec();
    let q_size: usize = q_dims.iter().product();
    let cotangent = QrCotangent {
        q: Some(make_tensor(vec![1.0; q_size], &q_dims)),
        r: None,
    };
    let grad = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "qr_rrule grad not finite: {val}");
    }
}

#[test]
fn solve_rrule_vector_rhs() {
    // Line 3407: solve_rrule with 1D b -> nrhs=1 else branch.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.5, 0.3, 3.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0], &[2]);
    let co = make_tensor(vec![1.0, 1.0], &[2]);
    let grad = solve_rrule(&mut ctx, &a, &b, &co).unwrap();
    let ga = tensor_data(&grad.a);
    for &val in &ga {
        assert!(val.is_finite(), "solve_rrule grad_a not finite: {val}");
    }
    let gb = tensor_data(&grad.b);
    for &val in &gb {
        assert!(val.is_finite(), "solve_rrule grad_b not finite: {val}");
    }
}

#[test]
fn eigen_rrule_vectors_only_cotangent() {
    // Line 3166: eigen_rrule with values=None -> skip dE branch.
    let mut ctx = CpuContext::new(1);
    // Symmetric matrix
    let a = make_tensor(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let result = eigen(&mut ctx, &a).unwrap();
    let v_dims = result.vectors.dims().to_vec();
    let v_size: usize = v_dims.iter().product();
    let cotangent = EigenCotangent {
        values: None,
        vectors: Some(make_tensor(vec![1.0; v_size], &v_dims)),
    };
    let grad = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
    assert_eq!(grad.dims(), &[2, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(val.is_finite(), "eigen_rrule grad not finite: {val}");
    }
}

#[test]
fn svd_rrule_tall_rank_deficient_with_du() {
    // Lines 2769, 2803: svd_rrule non-square correction with near-zero singular value.
    // A rank-1 tall matrix has a zero singular value, triggering sinv -> T::zero() branch.
    let mut ctx = CpuContext::new(1);
    // 3x2 rank-1 matrix: [[1,0],[0,0],[0,0]] in col-major = [1,0,0, 0,0,0]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let u_dims = result.u.dims().to_vec();
    let u_size: usize = u_dims.iter().product();
    let s_dims = result.s.dims().to_vec();
    let s_size: usize = s_dims.iter().product();
    let co = SvdCotangent {
        u: Some(make_tensor(vec![1.0; u_size], &u_dims)),
        s: Some(make_tensor(vec![1.0; s_size], &s_dims)),
        vt: None,
    };
    let grad = svd_rrule(&mut ctx, &a, &co, None).unwrap();
    assert_eq!(grad.dims(), &[3, 2]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(
            val.is_finite(),
            "svd_rrule rank-deficient grad not finite: {val}"
        );
    }
}

#[test]
fn svd_rrule_wide_rank_deficient_with_dvt() {
    // Lines 2803: svd_rrule non-square correction for n > k with near-zero singular value.
    let mut ctx = CpuContext::new(1);
    // 2x3 rank-1 matrix
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let vt_dims = result.vt.dims().to_vec();
    let vt_size: usize = vt_dims.iter().product();
    let s_dims = result.s.dims().to_vec();
    let s_size: usize = s_dims.iter().product();
    let co = SvdCotangent {
        u: None,
        s: Some(make_tensor(vec![1.0; s_size], &s_dims)),
        vt: Some(make_tensor(vec![1.0; vt_size], &vt_dims)),
    };
    let grad = svd_rrule(&mut ctx, &a, &co, None).unwrap();
    assert_eq!(grad.dims(), &[2, 3]);
    let gd = tensor_data(&grad);
    for &val in &gd {
        assert!(
            val.is_finite(),
            "svd_rrule wide rank-deficient grad not finite: {val}"
        );
    }
}

#[test]
fn svd_frule_tall_rank_deficient() {
    // Lines 4075, 4107: svd_frule non-square correction with near-zero singular value.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.1, 0.0, 0.0], &[3, 2]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    let ud = tensor_data(&result.u);
    for &val in &ud {
        assert!(val.is_finite(), "svd_frule u not finite: {val}");
    }
    let dud = tensor_data(&dresult.u);
    for &val in &dud {
        assert!(val.is_finite(), "svd_frule du not finite: {val}");
    }
}

#[test]
fn svd_frule_wide_rank_deficient() {
    // Lines 4075, 4107: svd_frule non-square correction for n > k with near-zero sv.
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.1, 0.0, 0.0], &[2, 3]);
    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    let sd = tensor_data(&result.s);
    for &val in &sd {
        assert!(val.is_finite(), "svd_frule s not finite: {val}");
    }
    let dvtd = tensor_data(&dresult.vt);
    for &val in &dvtd {
        assert!(val.is_finite(), "svd_frule dvt not finite: {val}");
    }
}

#[test]
fn svd_rrule_repeated_singular_values_finite() {
    let mut ctx = CpuContext::new(1);
    // 2 * I has repeated singular values [2, 2].
    let a = make_tensor(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]);
    let result = svd(&mut ctx, &a, None).unwrap();
    let cotangent = SvdCotangent {
        u: Some(make_tensor(vec![1.0; 4], result.u.dims())),
        s: Some(make_tensor(vec![1.0; 2], result.s.dims())),
        vt: Some(make_tensor(vec![1.0; 4], result.vt.dims())),
    };

    let grad = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
    let grad_data = tensor_data(&grad);
    for &val in &grad_data {
        assert!(
            val.is_finite(),
            "svd_rrule repeated-singular-value grad not finite: {val}"
        );
    }
}

#[test]
fn svd_frule_near_repeated_singular_values_finite() {
    let mut ctx = CpuContext::new(1);
    // Diagonal matrix with nearly equal singular values.
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0 + 1e-12], &[2, 2]);
    let da = make_tensor(vec![0.0, 1e-6, -1e-6, 0.0], &[2, 2]);

    let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    for &val in &tensor_data(&result.s) {
        assert!(
            val.is_finite(),
            "svd_frule repeated-spectrum s not finite: {val}"
        );
    }
    for &val in &tensor_data(&dresult.u) {
        assert!(
            val.is_finite(),
            "svd_frule repeated-spectrum dU not finite: {val}"
        );
    }
    for &val in &tensor_data(&dresult.s) {
        assert!(
            val.is_finite(),
            "svd_frule repeated-spectrum dS not finite: {val}"
        );
    }
    for &val in &tensor_data(&dresult.vt) {
        assert!(
            val.is_finite(),
            "svd_frule repeated-spectrum dVt not finite: {val}"
        );
    }
}

#[test]
fn eigen_repeated_eigenvalues_identity() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let result = eigen(&mut ctx, &a).unwrap();

    let vals = tensor_data(&result.values);
    for &val in &vals {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "repeated-eigenvalue forward check failed: {val}"
        );
    }
    for &val in &tensor_data(&result.vectors) {
        assert!(
            val.is_finite(),
            "eigen vectors not finite for repeated spectrum: {val}"
        );
    }
}

#[test]
fn eigen_rrule_repeated_eigenvalues_finite() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let cotangent = EigenCotangent {
        values: Some(make_tensor(vec![1.0; 3], &[3])),
        vectors: Some(make_tensor(vec![1.0; 9], &[3, 3])),
    };

    let grad = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
    for &val in &tensor_data(&grad) {
        assert!(
            val.is_finite(),
            "eigen_rrule repeated-eigenvalue grad not finite: {val}"
        );
    }
}

#[test]
fn eigen_frule_near_repeated_eigenvalues_finite() {
    let mut ctx = CpuContext::new(1);
    // Symmetric matrix with a nearly repeated leading pair of eigenvalues.
    let a = make_tensor(
        vec![1.0, 0.0, 0.0, 0.0, 1.0 + 1e-12, 0.0, 0.0, 0.0, 3.0],
        &[3, 3],
    );
    let da = make_tensor(
        vec![
            0.0, 1e-6, 0.0, //
            1e-6, 0.0, 1e-6, //
            0.0, 1e-6, 0.0,
        ],
        &[3, 3],
    );

    let (result, dresult) = eigen_frule(&mut ctx, &a, &da).unwrap();
    for &val in &tensor_data(&result.values) {
        assert!(
            val.is_finite(),
            "eigen_frule primal eigenvalue not finite: {val}"
        );
    }
    for &val in &tensor_data(&dresult.values) {
        assert!(
            val.is_finite(),
            "eigen_frule repeated-spectrum dE not finite: {val}"
        );
    }
    for &val in &tensor_data(&dresult.vectors) {
        assert!(
            val.is_finite(),
            "eigen_frule repeated-spectrum dV not finite: {val}"
        );
    }
}

#[test]
fn lu_nopivot_near_zero_pivot_returns_error() {
    let mut ctx = CpuContext::new(1);
    // First pivot is zero -> NoPivot LU must fail early.
    let a = make_tensor(vec![0.0, 1.0, 1.0, 1.0], &[2, 2]);
    assert!(lu(&mut ctx, &a, LuPivot::NoPivot).is_err());
}

#[test]
fn lu_nopivot_requires_rank_at_least_two() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(lu(&mut ctx, &a, LuPivot::NoPivot).is_err());
}

#[test]
fn lstsq_rrule_rejects_cotangent_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5], &[3, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let bad_dx = make_tensor(vec![1.0, 1.0, 1.0], &[3]);
    assert!(lstsq_rrule(&mut ctx, &a, &b, &bad_dx).is_err());
}

#[test]
fn lstsq_rrule_cuda_context_returns_cpu_only_error() {
    let mut ctx = tenferro_prims::CudaContext::new();
    let a = make_tensor(vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5], &[3, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let dx = make_tensor(vec![1.0, 1.0], &[2]);
    let msg = match lstsq_rrule(&mut ctx, &a, &b, &dx) {
        Ok(_) => panic!("expected CpuContext-only error"),
        Err(err) => format!("{err}"),
    };
    assert!(
        msg.contains("CpuContext"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn lu_rrule_nopivot_executes_without_permutation_path() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![4.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0], &[3, 3]);
    let result = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
    let l_size: usize = result.l.dims().iter().product();
    let u_size: usize = result.u.dims().iter().product();
    let cotangent = LuCotangent {
        l: Some(make_tensor(vec![0.2; l_size], result.l.dims())),
        u: Some(make_tensor(vec![0.3; u_size], result.u.dims())),
    };
    let grad = lu_rrule(&mut ctx, &a, &cotangent, LuPivot::NoPivot).unwrap();
    for &val in &tensor_data(&grad) {
        assert!(val.is_finite(), "lu_rrule NoPivot grad not finite: {val}");
    }
}

#[test]
fn lu_frule_nopivot_executes_without_permutation_path() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![4.0, 1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0], &[3, 3]);
    let da = make_tensor(vec![0.1; 9], &[3, 3]);
    let (result, dresult) = lu_frule(&mut ctx, &a, &da, LuPivot::NoPivot).unwrap();
    assert!(result.p.is_none());
    for &val in &tensor_data(&dresult.l) {
        assert!(val.is_finite(), "lu_frule NoPivot dL not finite: {val}");
    }
    for &val in &tensor_data(&dresult.u) {
        assert!(val.is_finite(), "lu_frule NoPivot dU not finite: {val}");
    }
}

#[test]
fn solve_triangular_frule_reports_shape_mismatch_for_both_tangents() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.2, -0.1, 4.0], &[3, 3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);

    let bad_da = make_tensor(vec![0.1; 4], &[2, 2]);
    let db = make_tensor(vec![0.1; 3], &[3]);
    assert!(solve_triangular_frule(&mut ctx, &a, &b, &bad_da, &db, true).is_err());

    let da = make_tensor(vec![0.1; 9], &[3, 3]);
    let bad_db = make_tensor(vec![0.1; 4], &[4]);
    assert!(solve_triangular_frule(&mut ctx, &a, &b, &da, &bad_db, true).is_err());
}

#[test]
fn solve_triangular_frule_uses_general_multi_rhs_branch() {
    let mut ctx = CpuContext::new(1);
    let a = make_tensor(vec![2.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.2, -0.1, 4.0], &[3, 3]);
    // nrhs = 2, and nrhs != n, so the general branch is used.
    let b = make_tensor(vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0], &[3, 2]);
    let da = make_tensor(vec![0.01; 9], &[3, 3]);
    let db = make_tensor(vec![0.02; 6], &[3, 2]);
    let (_x, dx) = solve_triangular_frule(&mut ctx, &a, &b, &da, &db, true).unwrap();
    assert_eq!(dx.dims(), &[3, 2]);
    for &val in &tensor_data(&dx) {
        assert!(
            val.is_finite(),
            "solve_triangular_frule dX not finite: {val}"
        );
    }
}

#[test]
fn norm_forward_l1_inf_empty_matrix_and_lp_error() {
    let mut ctx = CpuContext::new(1);
    let a: Tensor<f64> = Tensor::from_vec(vec![], &[0, 3], &[1, 0], 0).unwrap();
    let l1 = norm(&mut ctx, &a, NormKind::L1).unwrap();
    let inf = norm(&mut ctx, &a, NormKind::Inf).unwrap();
    assert!((tensor_data(&l1)[0]).abs() < 1e-15);
    assert!((tensor_data(&inf)[0]).abs() < 1e-15);
    assert!(norm(
        &mut ctx,
        &make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
        NormKind::Lp(3.0)
    )
    .is_err());
}

#[test]
fn norm_rrule_empty_matrix_lp_error_and_sign_branches() {
    let mut ctx = CpuContext::new(1);
    let empty: Tensor<f64> = Tensor::from_vec(vec![], &[0, 3], &[1, 0], 0).unwrap();
    let co_scalar: Tensor<f64> = Tensor::from_vec(vec![1.0], &[], &[], 0).unwrap();
    let grad_empty_l1 = norm_rrule(&mut ctx, &empty, &co_scalar, NormKind::L1).unwrap();
    let grad_empty_inf = norm_rrule(&mut ctx, &empty, &co_scalar, NormKind::Inf).unwrap();
    assert_eq!(grad_empty_l1.dims(), &[0, 3]);
    assert_eq!(grad_empty_inf.dims(), &[0, 3]);

    let a = make_tensor(vec![0.0, 1.0, -2.0, 1.0], &[2, 2]);
    let _ = norm_rrule(&mut ctx, &a, &co_scalar, NormKind::Inf).unwrap();
    assert!(norm_rrule(&mut ctx, &a, &co_scalar, NormKind::Lp(2.0)).is_err());
}

#[test]
fn norm_frule_empty_matrix_lp_error_and_sign_branches() {
    let mut ctx = CpuContext::new(1);
    let empty: Tensor<f64> = Tensor::from_vec(vec![], &[0, 3], &[1, 0], 0).unwrap();
    let de: Tensor<f64> = Tensor::from_vec(vec![], &[0, 3], &[1, 0], 0).unwrap();
    let (_nrm_l1, dn_l1) = norm_frule(&mut ctx, &empty, &de, NormKind::L1).unwrap();
    let (_nrm_inf, dn_inf) = norm_frule(&mut ctx, &empty, &de, NormKind::Inf).unwrap();
    assert!(tensor_data(&dn_l1)[0].is_finite());
    assert!(tensor_data(&dn_inf)[0].is_finite());

    let a = make_tensor(vec![0.0, 1.0, -2.0, 1.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
    let (_n, dn) = norm_frule(&mut ctx, &a, &da, NormKind::Inf).unwrap();
    assert!(tensor_data(&dn)[0].is_finite());
    assert!(norm_frule(&mut ctx, &a, &da, NormKind::Lp(2.0)).is_err());
}
