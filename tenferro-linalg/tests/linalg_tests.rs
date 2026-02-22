//! Tests for tenferro-linalg: forward decompositions and AD rules.

use tenferro_linalg::backend::FaerBackend;
use tenferro_linalg::*;
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

/// Extract flat data from a Tensor.
fn tensor_data(t: &Tensor<f64>) -> Vec<f64> {
    let c = t.contiguous(COL);
    let off = c.offset() as usize;
    let len: usize = c.dims().iter().product();
    c.buffer().as_slice().unwrap()[off..off + len].to_vec()
}

// ============================================================================
// SVD tests
// ============================================================================

#[test]
fn svd_identity_3x3() {
    let mut backend = FaerBackend::new();
    // SVD of identity should give U=I, S=[1,1,1], Vt=I (up to sign)
    let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = make_tensor(data, &[3, 3]);
    let result = svd(&mut backend, &a, None).unwrap();
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
    let mut backend = FaerBackend::new();
    // A = U * diag(S) * Vt should reconstruct A
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 col-major
    let a = make_tensor(data.clone(), &[2, 3]);
    let result = svd(&mut backend, &a, None).unwrap();

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
fn svd_tall_matrix() {
    let mut backend = FaerBackend::new();
    // 4x2 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = make_tensor(data.clone(), &[4, 2]);
    let result = svd(&mut backend, &a, None).unwrap();
    assert_eq!(result.u.dims(), &[4, 2]);
    assert_eq!(result.s.dims(), &[2]);
    assert_eq!(result.vt.dims(), &[2, 2]);

    // Check descending order
    let s = tensor_data(&result.s);
    assert!(s[0] >= s[1], "singular values should be descending");
}

#[test]
fn svd_with_max_rank() {
    let mut backend = FaerBackend::new();
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor(data, &[2, 3]);
    let opts = SvdOptions {
        max_rank: Some(1),
        cutoff: None,
    };
    let result = svd(&mut backend, &a, Some(&opts)).unwrap();
    assert_eq!(result.s.dims(), &[1]);
    assert_eq!(result.u.dims(), &[2, 1]);
    assert_eq!(result.vt.dims(), &[1, 3]);
}

// ============================================================================
// QR tests
// ============================================================================

#[test]
fn qr_reconstruction() {
    let mut backend = FaerBackend::new();
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a = make_tensor(data.clone(), &[2, 3]);
    let result = qr(&mut backend, &a).unwrap();

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
fn qr_orthogonality() {
    let mut backend = FaerBackend::new();
    let data = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let a = make_tensor(data, &[3, 3]);
    let result = qr(&mut backend, &a).unwrap();

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
    let mut backend = FaerBackend::new();
    let data = vec![2.0, 1.0, 3.0, 1.0, 4.0, 7.0, 5.0, 3.0, 2.0];
    let a = make_tensor(data.clone(), &[3, 3]);
    let result = lu(&mut backend, &a, LuPivot::Partial).unwrap();

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
    let mut backend = FaerBackend::new();
    // A = [[4, 2], [2, 3]] (symmetric positive definite)
    let data = vec![4.0, 2.0, 2.0, 3.0];
    let a = make_tensor(data, &[2, 2]);
    let l = cholesky(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let data = vec![1.0, 0.0, 0.0, -1.0];
    let a = make_tensor(data, &[2, 2]);
    assert!(cholesky(&mut backend, &a).is_err());
}

// ============================================================================
// Eigen tests
// ============================================================================

#[test]
fn eigen_symmetric() {
    let mut backend = FaerBackend::new();
    // Symmetric matrix: [[2, 1], [1, 2]]
    let data = vec![2.0, 1.0, 1.0, 2.0];
    let a = make_tensor(data, &[2, 2]);
    let result = eigen(&mut backend, &a).unwrap();

    let vals = tensor_data(&result.values);
    // Eigenvalues should be 1 and 3 (ascending)
    assert!((vals[0] - 1.0).abs() < 1e-10, "eigenvalue 0: {}", vals[0]);
    assert!((vals[1] - 3.0).abs() < 1e-10, "eigenvalue 1: {}", vals[1]);
}

#[test]
fn eigen_nonsymmetric_returns_error() {
    let mut backend = FaerBackend::new();
    // Non-symmetric matrix: [[2, 3], [1, 4]]
    let data = vec![2.0, 1.0, 3.0, 4.0];
    let a = make_tensor(data, &[2, 2]);
    assert!(eigen(&mut backend, &a).is_err());
}

// ============================================================================
// Solve tests
// ============================================================================

#[test]
fn solve_identity() {
    let mut backend = FaerBackend::new();
    let a_data = vec![1.0, 0.0, 0.0, 1.0];
    let b_data = vec![3.0, 7.0];
    let a = make_tensor(a_data, &[2, 2]);
    let b = make_tensor(b_data.clone(), &[2, 1]);
    let x = solve(&mut backend, &a, &b).unwrap();
    let xd = tensor_data(&x);
    assert!((xd[0] - 3.0).abs() < 1e-10);
    assert!((xd[1] - 7.0).abs() < 1e-10);
}

#[test]
fn solve_general() {
    let mut backend = FaerBackend::new();
    // A = [[2, 1], [1, 3]], b = [5, 10]
    let a = make_tensor(vec![2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let b = make_tensor(vec![5.0, 10.0], &[2, 1]);
    let x = solve(&mut backend, &a, &b).unwrap();
    let xd = tensor_data(&x);
    // Verify: A x = b
    let res0 = 2.0 * xd[0] + 1.0 * xd[1] - 5.0;
    let res1 = 1.0 * xd[0] + 3.0 * xd[1] - 10.0;
    assert!(res0.abs() < 1e-10, "residual[0] = {res0}");
    assert!(res1.abs() < 1e-10, "residual[1] = {res1}");
}

#[test]
fn solve_rhs_shape_mismatch_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // Wrong leading dim: expected 2, got 3
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(solve(&mut backend, &a, &b).is_err());
}

#[test]
fn solve_scalar_rhs_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // Scalar RHS is invalid; previously this could hit panic-prone paths.
    let b = make_tensor(vec![1.0], &[]);
    assert!(solve(&mut backend, &a, &b).is_err());
}

#[test]
fn solve_triangular_batch_mismatch_returns_error() {
    let mut backend = FaerBackend::new();
    // A has batch dim [2], b has batch dim [3]
    let a_data = vec![
        1.0, 0.0, 0.0, 1.0, // batch 0
        1.0, 0.0, 0.0, 1.0, // batch 1
    ];
    let a = make_tensor(a_data, &[2, 2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(solve_triangular(&mut backend, &a, &b, true).is_err());
}

#[test]
fn lstsq_rhs_shape_mismatch_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], &[3]); // expected [2]
    assert!(lstsq(&mut backend, &a, &b).is_err());
}

// ============================================================================
// Inverse tests
// ============================================================================

#[test]
fn inv_2x2() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a_inv = inv(&mut backend, &a).unwrap();
    let inv_data = tensor_data(&a_inv);

    // A * A^{-1} should be identity
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
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

// ============================================================================
// Determinant tests
// ============================================================================

#[test]
fn det_2x2() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]);
    let d = det(&mut backend, &a).unwrap();
    let dv = tensor_data(&d);
    // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert!((dv[0] - (-2.0)).abs() < 1e-10, "det = {}", dv[0]);
}

#[test]
fn det_3x3() {
    let mut backend = FaerBackend::new();
    // det([[1,4,7],[2,5,8],[3,6,10]]) = 1*(50-48) - 4*(20-24) + 7*(12-15) = 2+16-21 = -3
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0], &[3, 3]);
    let d = det(&mut backend, &a).unwrap();
    let dv = tensor_data(&d);
    assert!((dv[0] - (-3.0)).abs() < 1e-10, "det = {}", dv[0]);
}

// ============================================================================
// Slogdet tests
// ============================================================================

#[test]
fn slogdet_positive_det() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let result = slogdet(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let n = norm(&mut backend, &a, NormKind::Fro).unwrap();
    let nv = tensor_data(&n);
    let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
    assert!((nv[0] - expected).abs() < 1e-10);
}

#[test]
fn norm_spectral() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let n = norm(&mut backend, &a, NormKind::Spectral).unwrap();
    let nv = tensor_data(&n);
    assert!((nv[0] - 2.0).abs() < 1e-10, "spectral norm should be 2.0");
}

#[test]
fn norm_frobenius_batched_returns_batch_shape() {
    let mut backend = FaerBackend::new();
    // Shape [2,2,2]: two batches of 2x2 matrices.
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let a = make_tensor(data, &[2, 2, 2]);
    let n = norm(&mut backend, &a, NormKind::Fro).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ap = pinv(&mut backend, &a, None).unwrap();
    let ap_data = tensor_data(&ap);

    // A * A+ * A ~= A
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
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
// Eig (non-symmetric) should return error
// ============================================================================

#[test]
fn eig_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(eig(&mut backend, &a).is_err());
}

// ============================================================================
// Matrix exp should return error
// ============================================================================

#[test]
fn matrix_exp_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(matrix_exp(&mut backend, &a).is_err());
}

#[test]
fn norm_rrule_batched_cotangent_shape_mismatch_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 2, 2],
    );
    // For batch shape [2], cotangent must be [2]. Scalar is invalid here.
    let bad_cot = make_tensor(vec![1.0], &[]);
    assert!(norm_rrule(&mut backend, &a, &bad_cot, NormKind::Fro).is_err());
}

// ============================================================================
// AD rule tests: inv_rrule finite difference check
// ============================================================================

#[test]
fn inv_rrule_finite_diff() {
    let mut backend = FaerBackend::new();
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let n = 2;
    let eps = 1e-6;

    // Cotangent (seed): identity
    let cot_data = vec![1.0, 0.0, 0.0, 1.0];
    let cot = make_tensor(cot_data.clone(), &[2, 2]);
    let grad = inv_rrule(&mut backend, &a, &cot).unwrap();
    let grad_data = tensor_data(&grad);

    // Finite difference check: grad_A[idx] = sum_k cot[k] * d(inv(A)[k])/dA[idx]
    for idx in 0..n * n {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;
        let inv_plus = tensor_data(&inv(&mut backend, &make_tensor(a_plus, &[n, n])).unwrap());
        let inv_minus = tensor_data(&inv(&mut backend, &make_tensor(a_minus, &[n, n])).unwrap());

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
    let mut backend = FaerBackend::new();
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let n = 2;
    let eps = 1e-6;

    let cot = make_tensor(vec![1.0], &[]);
    let grad = det_rrule(&mut backend, &a, &cot).unwrap();
    let grad_data = tensor_data(&grad);

    for idx in 0..n * n {
        let mut a_plus = a_data.clone();
        let mut a_minus = a_data.clone();
        a_plus[idx] += eps;
        a_minus[idx] -= eps;
        let det_plus = tensor_data(&det(&mut backend, &make_tensor(a_plus, &[n, n])).unwrap())[0];
        let det_minus = tensor_data(&det(&mut backend, &make_tensor(a_minus, &[n, n])).unwrap())[0];
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
    let mut backend = FaerBackend::new();
    let a_data = vec![2.0, 0.5, 0.3, 3.0];
    let b_data = vec![1.0, 2.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let b = make_tensor(b_data.clone(), &[2, 1]);
    let eps = 1e-6;

    let _x = solve(&mut backend, &a, &b).unwrap();
    let cot = make_tensor(vec![1.0, 1.0], &[2, 1]);
    let grad = solve_rrule(&mut backend, &a, &b, &cot).unwrap();
    let grad_a_data = tensor_data(&grad.a);

    // FD check for A gradient
    for idx in 0..4 {
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let x_plus = tensor_data(&solve(&mut backend, &make_tensor(a_plus, &[2, 2]), &b).unwrap());
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let x_minus =
            tensor_data(&solve(&mut backend, &make_tensor(a_minus, &[2, 2]), &b).unwrap());

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
    let mut backend = FaerBackend::new();
    // Symmetric positive definite
    let a_data = vec![4.0, 1.0, 1.0, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    let cot_data = vec![1.0, 0.0, 0.0, 1.0];
    let cot = make_tensor(cot_data.clone(), &[2, 2]);
    let grad = cholesky_rrule(&mut backend, &a, &cot).unwrap();
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
        let l_plus = tensor_data(&cholesky(&mut backend, &make_tensor(a_plus, &[2, 2])).unwrap());

        let mut a_minus = a_data.clone();
        a_minus[i + j * 2] -= eps;
        if i != j {
            a_minus[j + i * 2] -= eps;
        }
        let l_minus = tensor_data(&cholesky(&mut backend, &make_tensor(a_minus, &[2, 2])).unwrap());

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
    let mut backend = FaerBackend::new();
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    for idx in 0..4 {
        let mut tangent_data = vec![0.0; 4];
        tangent_data[idx] = 1.0;
        let tangent = make_tensor(tangent_data, &[2, 2]);

        let (_, dinv) = inv_frule(&mut backend, &a, &tangent).unwrap();
        let dinv_data = tensor_data(&dinv);

        // Finite difference
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let inv_plus = tensor_data(&inv(&mut backend, &make_tensor(a_plus, &[2, 2])).unwrap());
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let inv_minus = tensor_data(&inv(&mut backend, &make_tensor(a_minus, &[2, 2])).unwrap());

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
    let mut backend = FaerBackend::new();
    let a_data = vec![2.0, 0.5, 0.5, 3.0];
    let a = make_tensor(a_data.clone(), &[2, 2]);
    let eps = 1e-6;

    for idx in 0..4 {
        let mut tangent_data = vec![0.0; 4];
        tangent_data[idx] = 1.0;
        let tangent = make_tensor(tangent_data, &[2, 2]);

        let (_, ddet) = det_frule(&mut backend, &a, &tangent).unwrap();
        let ddet_data = tensor_data(&ddet);

        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let det_plus = tensor_data(&det(&mut backend, &make_tensor(a_plus, &[2, 2])).unwrap())[0];
        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let det_minus = tensor_data(&det(&mut backend, &make_tensor(a_minus, &[2, 2])).unwrap())[0];
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(svd(&mut backend, &a, None).is_err());
}

#[test]
fn qr_1d_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0], &[3]);
    assert!(qr(&mut backend, &a).is_err());
}

// ============================================================================
// LinalgScalar trait tests
// ============================================================================

#[test]
fn linalg_scalar_f32() {
    let mut backend = FaerBackend::new();
    let data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let a = Tensor::<f32>::from_vec(data, &[2, 2], &[1, 2], 0).unwrap();
    let result = svd(&mut backend, &a, None).unwrap();
    let s_data = result.s.contiguous(COL);
    let s = s_data.buffer().as_slice().unwrap();
    assert!((s[0] - 1.0_f32).abs() < 1e-5);
}
