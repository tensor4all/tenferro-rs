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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let result = eig(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let result = eig(&mut backend, &a).unwrap();
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
fn eig_2x2_complex_eigenvalues() {
    // [[0, -1], [1, 0]] has eigenvalues +/- i
    // Column-major: [0, 1, -1, 0]
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![0.0, 1.0, -1.0, 0.0], &[2, 2]);
    let result = eig(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    // Upper triangular with known eigenvalues 1, 2, 3
    // Column-major: col0=[1,0,0], col1=[1,2,0], col2=[0,1,3]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 3.0], &[3, 3]);
    let result = eig(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    // batch 0: [[1, 0], [0, 2]] => eigenvalues 1, 2
    // batch 1: [[3, 0], [0, 4]] => eigenvalues 3, 4
    // Column-major for [2,2,2]: [1,0, 0,2, 3,0, 0,4]
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0], &[2, 2, 2]);
    let result = eig(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let cotangent = EigCotangent {
        values: Some(make_complex_tensor(
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            &[2],
        )),
        vectors: None,
    };
    let grad = eig_rrule(&mut backend, &a, &cotangent).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let cotangent = EigCotangent::<f64> {
        values: None,
        vectors: None,
    };
    let grad = eig_rrule(&mut backend, &a, &cotangent).unwrap();
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
    let mut backend = FaerBackend::new();
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
    let grad = eig_rrule(&mut backend, &a, &cotangent).unwrap();
    let grad_data = tensor_data(&grad);

    // Finite difference: perturb each element and check eigenvalue change
    let eps = 1e-6;
    let dlam = [1.0, 0.5, -1.0]; // cotangent values (real)

    for idx in 0..9 {
        let mut a_plus = a_data.clone();
        a_plus[idx] += eps;
        let a_p = make_tensor(a_plus, &[3, 3]);
        let r_p = eig(&mut backend, &a_p).unwrap();
        let vals_p = tensor_data_complex(&r_p.values);

        let mut a_minus = a_data.clone();
        a_minus[idx] -= eps;
        let a_m = make_tensor(a_minus, &[3, 3]);
        let r_m = eig(&mut backend, &a_m).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.0, 0.0, 0.2], &[2, 2]);
    let (result, tangent) = eig_frule(&mut backend, &a, &da).unwrap();

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
    let mut backend = FaerBackend::new();
    let a_data = vec![1.0, 0.0, 0.0, 0.5, 2.0, 0.0, 0.3, 0.7, 3.0];
    let da_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let a = make_tensor(a_data.clone(), &[3, 3]);
    let da = make_tensor(da_data.clone(), &[3, 3]);
    let (_result, tangent) = eig_frule(&mut backend, &a, &da).unwrap();
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
    let r_plus = eig(&mut backend, &a_plus).unwrap();
    let r_minus = eig(&mut backend, &a_minus).unwrap();
    let vals_p = tensor_data_complex(&r_plus.values);
    let vals_m = tensor_data_complex(&r_minus.values);

    // Sort by real part for consistent ordering
    let mut vp: Vec<Complex64> = vals_p;
    let mut vm: Vec<Complex64> = vals_m;
    let mut dv_sorted: Vec<(f64, Complex64)> = {
        let result2 = eig(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let zeros = make_tensor(vec![0.0; 9], &[3, 3]);
    let result = matrix_exp(&mut backend, &zeros).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![0.0, 0.0, 1.0, 0.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![10.0, 0.0, 0.0, 10.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![3.0], &[1, 1]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    // batch 0: diag(1, 0), batch 1: diag(0, 2)
    // col-major with batch dim: [1,0, 0,0,  0,0, 0,2]
    let a = make_tensor(
        vec![
            1.0, 0.0, 0.0, 0.0, // batch 0: diag(1, 0)
            0.0, 0.0, 0.0, 2.0, // batch 1: diag(0, 2)
        ],
        &[2, 2, 2],
    );
    let result = matrix_exp(&mut backend, &a).unwrap();
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
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![0.0, pi_2, -pi_2, 0.0], &[2, 2]);
    let result = matrix_exp(&mut backend, &a).unwrap();
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

// ============================================================================
// Regression tests: invalid inputs must return Err, not panic (#124)
// ============================================================================

#[test]
fn test_svd_1d_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    assert!(svd(&mut backend, &a, None).is_err());
}

#[test]
fn test_qr_1d_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    assert!(qr(&mut backend, &a).is_err());
}

#[test]
fn test_cholesky_non_square_returns_error() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(cholesky(&mut backend, &a).is_err());
}

#[test]
fn test_solve_dimension_mismatch_returns_error() {
    let mut backend = FaerBackend::new();
    // A is 3x3 but b has leading dimension 2 (mismatch)
    let a = make_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let b = make_tensor(vec![1.0, 2.0], &[2, 1]);
    assert!(solve(&mut backend, &a, &b).is_err());
}

// ============================================================================
// Complex64 integration tests (Tensor-level API)
// ============================================================================

use num_complex::Complex64;

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
    let mut backend = FaerBackend::new();
    let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let a = make_complex_tensor(data, &[2, 2]);
    let result = svd(&mut backend, &a, None).unwrap();

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
    let mut backend = FaerBackend::new();
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
    let result = svd(&mut backend, &a, None).unwrap();

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
    let mut backend = FaerBackend::new();
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
    let result = svd(&mut backend, &a, Some(&opts)).unwrap();
    assert_eq!(result.s.dims(), &[1]);
    assert_eq!(result.u.dims(), &[2, 1]);
    assert_eq!(result.vt.dims(), &[1, 3]);
}

#[test]
fn test_qr_complex64_reconstruction() {
    let mut backend = FaerBackend::new();
    let data = vec![
        c(1.0, 1.0),
        c(2.0, -1.0),
        c(0.0, 3.0),
        c(4.0, 0.0),
        c(-1.0, 2.0),
        c(3.0, 1.0),
    ];
    let a = make_complex_tensor(data.clone(), &[3, 2]);
    let result = qr(&mut backend, &a).unwrap();

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
    let mut backend = FaerBackend::new();
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
    let result = lu(&mut backend, &a, LuPivot::Partial).unwrap();

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
    let mut backend = FaerBackend::new();
    // Hermitian positive definite: [[4, 1+i], [1-i, 3]]
    let data = vec![c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    let a = make_complex_tensor(data.clone(), &[2, 2]);
    let l_tensor = cholesky(&mut backend, &a).unwrap();

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
    let mut backend = FaerBackend::new();
    // Hermitian: [[3, 1-i], [1+i, 2]], eigenvalues 1 and 4
    let data = vec![c(3.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(2.0, 0.0)];
    let a = make_complex_tensor(data, &[2, 2]);
    let result = eigen(&mut backend, &a).unwrap();

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
    let mut backend = FaerBackend::new();
    // A = [[2+i, 1], [0, 3-i]], b = [3+i, 6-2i]
    let a_data = vec![c(2.0, 1.0), c(0.0, 0.0), c(1.0, 0.0), c(3.0, -1.0)];
    let b_data = vec![c(3.0, 1.0), c(6.0, -2.0)];
    let a = make_complex_tensor(a_data.clone(), &[2, 2]);
    let b = make_complex_tensor(b_data.clone(), &[2, 1]);
    let x = solve(&mut backend, &a, &b).unwrap();
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
fn test_solve_triangular_complex64() {
    let mut backend = FaerBackend::new();
    // Lower triangular: [[2+i, 0], [1-i, 3]]
    let a_data = vec![c(2.0, 1.0), c(1.0, -1.0), c(0.0, 0.0), c(3.0, 0.0)];
    let b_data = vec![c(4.0, 2.0), c(5.0, 0.0)];
    let a = make_complex_tensor(a_data.clone(), &[2, 2]);
    let b = make_complex_tensor(b_data.clone(), &[2, 1]);
    let x = solve_triangular(&mut backend, &a, &b, false).unwrap();
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
    let mut backend = FaerBackend::new();
    let zeros = make_tensor(vec![0.0; 4], &[2, 2]);
    let da = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let (result, tangent) = matrix_exp_frule(&mut backend, &zeros, &da).unwrap();
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
    let mut backend = FaerBackend::new();
    let zeros = make_tensor(vec![0.0; 4], &[2, 2]);
    let co = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let grad = matrix_exp_rrule(&mut backend, &zeros, &co).unwrap();
    let g = tensor_data(&grad);
    assert!((g[0] - 1.0).abs() < 1e-10);
    assert!((g[1] - 2.0).abs() < 1e-10);
    assert!((g[2] - 3.0).abs() < 1e-10);
    assert!((g[3] - 4.0).abs() < 1e-10);
}

#[test]
fn matrix_exp_frule_finite_difference() {
    let mut backend = FaerBackend::new();
    let a = make_tensor(vec![0.5, 0.1, -0.2, 0.3], &[2, 2]);
    let da = make_tensor(vec![0.1, 0.05, -0.03, 0.07], &[2, 2]);
    let eps = 1e-6;

    let (_, analytic_tangent) = matrix_exp_frule(&mut backend, &a, &da).unwrap();
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
    let exp_plus = matrix_exp(&mut backend, &make_tensor(plus, &[2, 2])).unwrap();
    let exp_minus = matrix_exp(&mut backend, &make_tensor(minus, &[2, 2])).unwrap();
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
    let mut backend = FaerBackend::new();
    let a_vec = vec![0.5, 0.1, -0.2, 0.3];
    let a = make_tensor(a_vec.clone(), &[2, 2]);
    let co_vec = vec![1.0, -0.5, 0.3, 0.8];
    let co = make_tensor(co_vec.clone(), &[2, 2]);
    let eps = 1e-6;

    let grad = matrix_exp_rrule(&mut backend, &a, &co).unwrap();
    let g = tensor_data(&grad);

    // For each (i,j), compute grad[i,j] via FD
    for idx in 0..4 {
        let mut plus = a_vec.clone();
        let mut minus = a_vec.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let exp_plus = matrix_exp(&mut backend, &make_tensor(plus, &[2, 2])).unwrap();
        let exp_minus = matrix_exp(&mut backend, &make_tensor(minus, &[2, 2])).unwrap();
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
