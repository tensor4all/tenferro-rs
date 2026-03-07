use super::*;

#[test]
fn faer_backend_thin_svd_identity_f64() {
    let mut backend = FaerBackend::new();
    // 3x3 identity matrix, column-major
    let a = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 9]; // 3x3
    let mut s = [0.0_f64; 3];
    let mut vt = [0.0_f64; 9]; // 3x3

    backend.thin_svd(&a, 3, 3, &mut u, &mut s, &mut vt).unwrap();

    // All singular values should be 1.0
    for &val in &s {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "singular value should be 1.0, got {val}"
        );
    }

    // U * diag(S) * Vt should reconstruct the identity
    // For identity: U * Vt should be identity (up to sign)
    let mut recon = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for p in 0..3 {
                sum += u[i + p * 3] * s[p] * vt[p + j * 3];
            }
            recon[i + j * 3] = sum;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((recon[i + j * 3] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn faer_backend_thin_svd_rectangular_f64() {
    let mut backend = FaerBackend::new();
    // 3x2 matrix, column-major: [[1,4],[2,5],[3,6]]
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = 3;
    let n = 2;
    let k = 2; // min(3,2)
    let mut u = vec![0.0_f64; m * k]; // 3x2
    let mut s = vec![0.0_f64; k]; // 2
    let mut vt = vec![0.0_f64; k * n]; // 2x2

    backend.thin_svd(&a, m, n, &mut u, &mut s, &mut vt).unwrap();

    // Singular values should be positive and descending
    assert!(s[0] > 0.0);
    assert!(s[1] > 0.0);
    assert!(s[0] >= s[1]);

    // Reconstruct: U * diag(S) * Vt should give back A
    let mut recon = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += u[i + p * m] * s[p] * vt[p + j * k];
            }
            recon[i + j * m] = sum;
        }
    }
    for idx in 0..a.len() {
        assert!((recon[idx] - a[idx]).abs() < 1e-10);
    }
}

#[test]
fn faer_backend_thin_svd_f32() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mut u = [0.0_f32; 4];
    let mut s = [0.0_f32; 2];
    let mut vt = [0.0_f32; 4];

    backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();

    for &val in &s {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "f32 singular value should be 1.0, got {val}"
        );
    }
}

#[test]
fn faer_backend_qr_f64() {
    let mut backend = FaerBackend::new();
    // 3x2 matrix
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = 3;
    let n = 2;
    let k = 2;
    let mut q = vec![0.0_f64; m * k];
    let mut r = vec![0.0_f64; k * n];

    backend.qr(&a, m, n, &mut q, &mut r).unwrap();

    // Q * R should reconstruct A
    let mut recon = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += q[i + p * m] * r[p + j * k];
            }
            recon[i + j * m] = sum;
        }
    }
    for idx in 0..a.len() {
        assert!((recon[idx] - a[idx]).abs() < 1e-10);
    }
}

#[test]
fn faer_backend_mat_mul_f64() {
    let mut backend = FaerBackend::new();
    // A = 2x2 identity, B = [[1,3],[2,4]], col-major
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64, 2.0, 3.0, 4.0];
    let mut c = [0.0_f64; 4];

    backend.mat_mul(&a, 2, 2, &b, 2, &mut c).unwrap();

    // Identity * B = B
    for idx in 0..4 {
        assert!((c[idx] - b[idx]).abs() < 1e-10);
    }
}

#[test]
fn faer_backend_solve_f64() {
    let mut backend = FaerBackend::new();
    // A = [[2,1],[1,3]], col-major: [2,1,1,3]
    let a = [2.0_f64, 1.0, 1.0, 3.0];
    // b = [5, 7], single RHS
    let b = [5.0_f64, 7.0];
    let mut x = [0.0_f64; 2];

    backend.solve(&a, &b, 2, 1, &mut x).unwrap();

    // Verify: A * x = b
    let ax0 = 2.0 * x[0] + 1.0 * x[1];
    let ax1 = 1.0 * x[0] + 3.0 * x[1];
    assert!((ax0 - 5.0).abs() < 1e-10, "A*x[0] = {ax0}, expected 5.0");
    assert!((ax1 - 7.0).abs() < 1e-10, "A*x[1] = {ax1}, expected 7.0");
}

#[test]
fn faer_backend_cholesky_f64() {
    let mut backend = FaerBackend::new();
    // Symmetric positive definite: [[4,2],[2,3]], col-major: [4,2,2,3]
    let a = [4.0_f64, 2.0, 2.0, 3.0];
    let mut l = [0.0_f64; 4];

    backend.cholesky(&a, 2, &mut l).unwrap();

    // L * L^T should reconstruct A
    let mut recon = [0.0_f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = 0.0;
            for p in 0..2 {
                sum += l[i + p * 2] * l[j + p * 2];
            }
            recon[i + j * 2] = sum;
        }
    }
    for idx in 0..4 {
        assert!((recon[idx] - a[idx]).abs() < 1e-10);
    }
}

#[test]
fn faer_backend_eigen_sym_f64() {
    let mut backend = FaerBackend::new();
    // Symmetric: [[2,1],[1,2]], col-major: [2,1,1,2]
    let a = [2.0_f64, 1.0, 1.0, 2.0];
    let mut values = [0.0_f64; 2];
    let mut vectors = [0.0_f64; 4];

    backend.eigen_sym(&a, 2, &mut values, &mut vectors).unwrap();

    // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    assert!((values[0] - 1.0).abs() < 1e-10);
    assert!((values[1] - 3.0).abs() < 1e-10);
}

#[test]
fn faer_backend_solve_triangular_f64() {
    let mut backend = FaerBackend::new();
    // Lower triangular: [[2,0],[1,3]], col-major: [2,1,0,3]
    let a = [2.0_f64, 1.0, 0.0, 3.0];
    let b = [4.0_f64, 5.0];
    let mut x = [0.0_f64; 2];

    backend
        .solve_triangular(&a, &b, 2, 1, false, &mut x)
        .unwrap();

    // Verify: A * x = b
    let ax0 = 2.0 * x[0] + 0.0 * x[1];
    let ax1 = 1.0 * x[0] + 3.0 * x[1];
    assert!((ax0 - 4.0).abs() < 1e-10, "A*x[0] = {ax0}, expected 4.0");
    assert!((ax1 - 5.0).abs() < 1e-10, "A*x[1] = {ax1}, expected 5.0");
}

#[test]
fn faer_backend_lu_f64() {
    let mut backend = FaerBackend::new();
    // 3x3 matrix: [[2,1,1],[4,3,3],[8,7,9]], col-major
    let a = [2.0_f64, 4.0, 8.0, 1.0, 3.0, 7.0, 1.0, 3.0, 9.0];
    let m = 3;
    let n = 3;
    let k = 3;
    let mut perm = vec![0usize; m];
    let mut l = vec![0.0_f64; m * k];
    let mut u_out = vec![0.0_f64; k * n];

    backend.lu(&a, m, n, &mut perm, &mut l, &mut u_out).unwrap();

    // P * A = L * U: reconstruct L * U then apply inverse permutation
    let mut lu_prod = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += l[i + p * m] * u_out[p + j * k];
            }
            lu_prod[i + j * m] = sum;
        }
    }

    // Apply P^{-1} to rows of lu_prod to get A back.
    // perm[i] = j means row i of P*A comes from row j of A.
    // So: A[perm[i], col] = lu_prod[i, col]
    let mut recon = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            recon[perm[i] + j * m] = lu_prod[i + j * m];
        }
    }
    for idx in 0..a.len() {
        assert!((recon[idx] - a[idx]).abs() < 1e-10);
    }
}

#[test]
fn faer_backend_default_trait() {
    let backend = FaerBackend;
    // Just verify it can be created
    let _ = backend;
}

#[test]
fn faer_backend_thin_svd_invalid_input() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short for 2x2
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];

    let result = backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt);
    assert!(result.is_err());
}

// ========================================================================
// f64 error path tests (cover slice-length validation in real backend)
// ========================================================================

#[test]
fn faer_backend_thin_svd_f64_invalid_u() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2
    let mut u = [0.0_f64; 1]; // too short (need 4)
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_thin_svd_f64_invalid_s() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 1]; // too short (need 2)
    let mut vt = [0.0_f64; 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_thin_svd_f64_invalid_vt() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 1]; // too short (need 4)
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_qr_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short for 2x2
    let mut q = [0.0_f64; 4];
    let mut r = [0.0_f64; 4];
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_qr_f64_invalid_q() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut q = [0.0_f64; 1]; // too short
    let mut r = [0.0_f64; 4];
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_qr_f64_invalid_r() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut q = [0.0_f64; 4];
    let mut r = [0.0_f64; 1]; // too short
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_lu_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short
    let mut perm = [0usize; 2];
    let mut l = [0.0_f64; 4];
    let mut u_out = [0.0_f64; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_f64_invalid_perm() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut perm = [0usize; 1]; // too short
    let mut l = [0.0_f64; 4];
    let mut u_out = [0.0_f64; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_f64_invalid_l() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut perm = [0usize; 2];
    let mut l = [0.0_f64; 1]; // too short
    let mut u_out = [0.0_f64; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_f64_invalid_u() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut perm = [0usize; 2];
    let mut l = [0.0_f64; 4];
    let mut u_out = [0.0_f64; 1]; // too short
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_cholesky_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short for n=2
    let mut l = [0.0_f64; 4];
    assert!(backend.cholesky(&a, 2, &mut l).is_err());
}

#[test]
fn faer_backend_cholesky_f64_invalid_l() {
    let mut backend = FaerBackend::new();
    let a = [4.0_f64, 0.0, 0.0, 4.0]; // SPD
    let mut l = [0.0_f64; 1]; // too short
    assert!(backend.cholesky(&a, 2, &mut l).is_err());
}

#[test]
fn faer_backend_eigen_sym_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short for n=2
    let mut values = [0.0_f64; 2];
    let mut vectors = [0.0_f64; 4];
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eigen_sym_f64_invalid_values() {
    let mut backend = FaerBackend::new();
    let a = [2.0_f64, 1.0, 1.0, 2.0]; // 2x2 symmetric
    let mut values = [0.0_f64; 1]; // too short
    let mut vectors = [0.0_f64; 4];
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eigen_sym_f64_invalid_vectors() {
    let mut backend = FaerBackend::new();
    let a = [2.0_f64, 1.0, 1.0, 2.0];
    let mut values = [0.0_f64; 2];
    let mut vectors = [0.0_f64; 1]; // too short
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_mat_mul_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short for 2x2
    let b = [1.0_f64, 0.0, 0.0, 1.0];
    let mut c = [0.0_f64; 4];
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_mat_mul_f64_invalid_b() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64]; // too short
    let mut c = [0.0_f64; 4];
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_mat_mul_f64_invalid_c() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64, 0.0, 0.0, 1.0];
    let mut c = [0.0_f64; 1]; // too short
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_solve_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short
    let b = [1.0_f64, 0.0];
    let mut x = [0.0_f64; 2];
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_f64_invalid_b() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64]; // too short
    let mut x = [0.0_f64; 2];
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_f64_invalid_x() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64, 0.0];
    let mut x = [0.0_f64]; // too short
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_tri_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short
    let b = [1.0_f64, 0.0];
    let mut x = [0.0_f64; 2];
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_solve_tri_f64_invalid_b() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64]; // too short
    let mut x = [0.0_f64; 2];
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_solve_tri_f64_invalid_x() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let b = [1.0_f64, 0.0];
    let mut x = [0.0_f64]; // too short
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_eig_general_f64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64]; // too short
    let mut values_ri = [0.0_f64; 4];
    let mut vectors_ri = [0.0_f64; 8];
    assert!(backend
        .eig_general(&a, 2, &mut values_ri, &mut vectors_ri)
        .is_err());
}

#[test]
fn faer_backend_eig_general_f64_invalid_values() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut values_ri = [0.0_f64; 1]; // too short (need 2*2=4)
    let mut vectors_ri = [0.0_f64; 8];
    assert!(backend
        .eig_general(&a, 2, &mut values_ri, &mut vectors_ri)
        .is_err());
}

#[test]
fn faer_backend_eig_general_f64_invalid_vectors() {
    let mut backend = FaerBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut values_ri = [0.0_f64; 4];
    let mut vectors_ri = [0.0_f64; 1]; // too short (need 2*2*2=8)
    assert!(backend
        .eig_general(&a, 2, &mut values_ri, &mut vectors_ri)
        .is_err());
}

#[test]
fn faer_backend_thin_svd_f64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [f64::NAN, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_eigen_sym_f64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [1.0, f64::NAN, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let mut values = [0.0_f64; 3];
    let mut vectors = [0.0_f64; 9];
    assert!(backend.eigen_sym(&a, 3, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eig_general_f64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [f64::NAN, 0.0, 0.0, 1.0];
    let mut values_ri = [0.0_f64; 4];
    let mut vectors_ri = [0.0_f64; 8];
    assert!(backend
        .eig_general(&a, 2, &mut values_ri, &mut vectors_ri)
        .is_err());
}

// ========================================================================
// Complex64 backend tests
// ========================================================================

/// Helper: complex matrix multiplication C = A * B (col-major, m x k times k x n).
fn complex_mat_mul(
    a: &[Complex64],
    m: usize,
    k: usize,
    b: &[Complex64],
    n: usize,
) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..k {
                sum += a[i + p * m] * b[p + j * k];
            }
            c[i + j * m] = sum;
        }
    }
    c
}

/// Helper: maximum element-wise absolute difference between two Complex64 slices.
fn complex_max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn faer_backend_thin_svd_complex64_identity() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // 2x2 complex identity, col-major
    let a = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let mut u = vec![Complex64::new(0.0, 0.0); 4];
    let mut s = [0.0_f64; 2];
    let mut vt = vec![Complex64::new(0.0, 0.0); 4];

    backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();

    // Singular values should be 1.0
    for &val in &s {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "singular value should be 1.0, got {val}"
        );
    }

    // Reconstruct: U * diag(S) * Vt should equal A
    let mut recon = vec![Complex64::new(0.0, 0.0); 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..2 {
                sum += u[i + p * 2] * s[p] * vt[p + j * 2];
            }
            recon[i + j * 2] = sum;
        }
    }
    assert!(
        complex_max_err(&recon, &a) < 1e-10,
        "SVD reconstruction of complex identity failed"
    );
}

#[test]
fn faer_backend_thin_svd_complex64_hermitian() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // Hermitian matrix: [[2, 1+i], [1-i, 3]], col-major: [2, 1-i, 1+i, 3]
    let a = [c(2.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    let m = 2;
    let n = 2;
    let k = 2;
    let mut u = vec![Complex64::new(0.0, 0.0); m * k];
    let mut s = vec![0.0_f64; k];
    let mut vt = vec![Complex64::new(0.0, 0.0); k * n];

    backend.thin_svd(&a, m, n, &mut u, &mut s, &mut vt).unwrap();

    // Singular values should be positive
    assert!(s[0] > 0.0);
    assert!(s[1] > 0.0);
    assert!(s[0] >= s[1]);

    // Reconstruct: U * diag(S) * Vt = A
    let mut recon = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..k {
                sum += u[i + p * m] * s[p] * vt[p + j * k];
            }
            recon[i + j * m] = sum;
        }
    }
    assert!(
        complex_max_err(&recon, &a) < 1e-10,
        "SVD reconstruction of Hermitian complex matrix failed"
    );
}

#[test]
fn faer_backend_qr_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // 3x2 complex matrix, col-major
    let a = [
        c(1.0, 1.0),
        c(2.0, -1.0),
        c(0.0, 3.0),
        c(4.0, 0.0),
        c(-1.0, 2.0),
        c(3.0, 1.0),
    ];
    let m = 3;
    let n = 2;
    let k = 2;
    let mut q = vec![Complex64::new(0.0, 0.0); m * k];
    let mut r = vec![Complex64::new(0.0, 0.0); k * n];

    backend.qr(&a, m, n, &mut q, &mut r).unwrap();

    // Q * R should reconstruct A
    let recon = complex_mat_mul(&q, m, k, &r, n);
    assert!(
        complex_max_err(&recon, &a) < 1e-10,
        "QR reconstruction of complex matrix failed"
    );
}

#[test]
fn faer_backend_lu_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // 3x3 complex matrix, col-major
    let a = [
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
    let m = 3;
    let n = 3;
    let k = 3;
    let mut perm = vec![0usize; m];
    let mut l = vec![Complex64::new(0.0, 0.0); m * k];
    let mut u_out = vec![Complex64::new(0.0, 0.0); k * n];

    backend.lu(&a, m, n, &mut perm, &mut l, &mut u_out).unwrap();

    // L * U = P * A -> reconstruct by applying P^{-1}
    let lu_prod = complex_mat_mul(&l, m, k, &u_out, n);

    // Apply P^{-1} to rows of lu_prod to get A back.
    // perm[i] = j means row i of P*A comes from row j of A.
    let mut recon = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            recon[perm[i] + j * m] = lu_prod[i + j * m];
        }
    }
    assert!(
        complex_max_err(&recon, &a) < 1e-10,
        "LU reconstruction of complex matrix failed"
    );
}

#[test]
fn faer_backend_cholesky_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // Hermitian positive definite: [[4, 1+i], [1-i, 3]], col-major: [4, 1-i, 1+i, 3]
    let a = [c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    let n = 2;
    let mut l = vec![Complex64::new(0.0, 0.0); n * n];

    backend.cholesky(&a, n, &mut l).unwrap();

    // L * L^H should reconstruct A
    let mut recon = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..n {
                // L^H[p,j] = conj(L[j,p])
                sum += l[i + p * n] * l[j + p * n].conj();
            }
            recon[i + j * n] = sum;
        }
    }
    assert!(
        complex_max_err(&recon, &a) < 1e-10,
        "Cholesky reconstruction of complex HPD matrix failed"
    );
}

#[test]
fn faer_backend_eigen_sym_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // Hermitian: [[3, 1-i], [1+i, 2]], col-major: [3, 1+i, 1-i, 2]
    // Eigenvalues: tr=5, det=3*2-(1-i)(1+i)=6-2=4, disc=sqrt(25-16)=3
    // lambda = (5 +/- 3)/2 = 4, 1
    let a = [c(3.0, 0.0), c(1.0, 1.0), c(1.0, -1.0), c(2.0, 0.0)];
    let n = 2;
    let mut values = vec![0.0_f64; n];
    let mut vectors = vec![Complex64::new(0.0, 0.0); n * n];

    backend.eigen_sym(&a, n, &mut values, &mut vectors).unwrap();

    // Eigenvalues should be 1.0 and 4.0 (ascending)
    assert!((values[0] - 1.0).abs() < 1e-10);
    assert!((values[1] - 4.0).abs() < 1e-10);

    // Verify A * v = lambda * v for each eigenvector
    for col in 0..n {
        let lambda = Complex64::new(values[col], 0.0);
        for row in 0..n {
            let mut av = Complex64::new(0.0, 0.0);
            for p in 0..n {
                av += a[row + p * n] * vectors[p + col * n];
            }
            let lv = lambda * vectors[row + col * n];
            assert!(
                (av - lv).norm() < 1e-10,
                "A*v != lambda*v at ({row},{col}): av={av}, lv={lv}"
            );
        }
    }
}

#[test]
fn faer_backend_mat_mul_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // A = 2x2 identity, B = [[1+i, 3], [2, 4-i]]
    let a = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let b = [c(1.0, 1.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, -1.0)];
    let mut result = vec![Complex64::new(0.0, 0.0); 4];

    backend.mat_mul(&a, 2, 2, &b, 2, &mut result).unwrap();

    // Identity * B = B
    assert!(complex_max_err(&result, &b) < 1e-10, "mat_mul: I * B != B");
}

#[test]
fn faer_backend_solve_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // A = [[2+i, 1], [0, 3-i]], b = [3+i, 6-2i]
    let a = [c(2.0, 1.0), c(0.0, 0.0), c(1.0, 0.0), c(3.0, -1.0)];
    let b = [c(3.0, 1.0), c(6.0, -2.0)];
    let mut x = vec![Complex64::new(0.0, 0.0); 2];

    backend.solve(&a, &b, 2, 1, &mut x).unwrap();

    // Verify A * x = b
    let ax = complex_mat_mul(&a, 2, 2, &x, 1);
    assert!(
        complex_max_err(&ax, &b) < 1e-10,
        "solve: A*x != b, got A*x = {:?}",
        ax
    );
}

#[test]
fn faer_backend_solve_triangular_complex64_lower() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // Lower triangular: [[2+i, 0], [1-i, 3]], col-major: [2+i, 1-i, 0, 3]
    let a = [c(2.0, 1.0), c(1.0, -1.0), c(0.0, 0.0), c(3.0, 0.0)];
    let b = [c(4.0, 2.0), c(5.0, 0.0)];
    let mut x = vec![Complex64::new(0.0, 0.0); 2];

    backend
        .solve_triangular(&a, &b, 2, 1, false, &mut x)
        .unwrap();

    // Verify A * x = b
    let ax = complex_mat_mul(&a, 2, 2, &x, 1);
    assert!(
        complex_max_err(&ax, &b) < 1e-10,
        "solve_triangular(lower): A*x != b"
    );
}

#[test]
fn faer_backend_solve_triangular_complex64_upper() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    // Upper triangular: [[3, 1+2i], [0, 2-i]], col-major: [3, 0, 1+2i, 2-i]
    let a = [c(3.0, 0.0), c(0.0, 0.0), c(1.0, 2.0), c(2.0, -1.0)];
    let b = [c(7.0, 2.0), c(4.0, -2.0)];
    let mut x = vec![Complex64::new(0.0, 0.0); 2];

    backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .unwrap();

    // Verify A * x = b
    let ax = complex_mat_mul(&a, 2, 2, &x, 1);
    assert!(
        complex_max_err(&ax, &b) < 1e-10,
        "solve_triangular(upper): A*x != b"
    );
}

#[test]
fn faer_backend_thin_svd_complex64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let mut u = vec![Complex64::new(0.0, 0.0); 4];
    let mut s = [0.0_f64; 2];
    let mut vt = vec![Complex64::new(0.0, 0.0); 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_eigen_sym_complex64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let mut values = [0.0_f64; 3];
    let mut vectors = vec![Complex64::new(0.0, 0.0); 9];
    assert!(backend.eigen_sym(&a, 3, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eig_general_complex64_nan_returns_error() {
    let mut backend = FaerBackend::new();
    let a = [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let mut values = vec![Complex64::new(0.0, 0.0); 2];
    let mut vectors = vec![Complex64::new(0.0, 0.0); 4];
    assert!(backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .is_err());
}

// ========================================================================
// Complex32 coverage and additional complex error paths migrated from
// integration tests so the public API does not need to expose these cases.
// ========================================================================

#[test]
fn faer_backend_mat_mul_complex32() {
    let mut backend = FaerBackend::new();
    let c = |re: f32, im: f32| Complex32::new(re, im);
    let a = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let b = [c(1.0, 1.0), c(3.0, 0.0), c(2.0, -1.0), c(4.0, 2.0)];
    let mut out = [Complex32::new(0.0, 0.0); 4];

    backend.mat_mul(&a, 2, 2, &b, 2, &mut out).unwrap();

    for i in 0..4 {
        assert!(
            (out[i].re - b[i].re).abs() < 1e-5 && (out[i].im - b[i].im).abs() < 1e-5,
            "C32 mat_mul[{i}] = {:?}, expected {:?}",
            out[i],
            b[i]
        );
    }
}

#[test]
fn faer_backend_solve_complex32() {
    let mut backend = FaerBackend::new();
    let c = |re: f32, im: f32| Complex32::new(re, im);
    let a = [c(2.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)];
    let b_rhs = [c(1.0, 1.0), c(2.0, 0.0)];
    let mut x = [Complex32::new(0.0, 0.0); 2];

    backend.solve(&a, &b_rhs, 2, 1, &mut x).unwrap();

    let ax0 = a[0] * x[0] + a[2] * x[1];
    let ax1 = a[1] * x[0] + a[3] * x[1];
    assert!((ax0 - b_rhs[0]).norm() < 1e-3, "C32 solve Ax[0] mismatch");
    assert!((ax1 - b_rhs[1]).norm() < 1e-3, "C32 solve Ax[1] mismatch");
}

#[test]
fn faer_backend_eig_general_complex32() {
    let mut backend = FaerBackend::new();
    let c = |re: f32, im: f32| Complex32::new(re, im);
    let a = [c(1.0, 0.0), c(2.0, 0.0), c(0.0, 1.0), c(3.0, 0.0)];
    let mut values = [Complex32::new(0.0, 0.0); 2];
    let mut vectors = [Complex32::new(0.0, 0.0); 4];

    backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .unwrap();

    for &v in &values {
        assert!(v.re.is_finite() && v.im.is_finite());
    }
}

#[test]
fn faer_backend_eig_general_complex64() {
    let mut backend = FaerBackend::new();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let a = [c(1.0, 0.0), c(2.0, 0.0), c(0.0, 1.0), c(3.0, 0.0)];
    let mut values = [Complex64::new(0.0, 0.0); 2];
    let mut vectors = [Complex64::new(0.0, 0.0); 4];

    backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .unwrap();

    for &v in &values {
        assert!(v.re.is_finite() && v.im.is_finite());
    }
}

#[test]
fn faer_backend_thin_svd_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [Complex64::new(1.0, 0.0)];
    let mut u = [Complex64::new(0.0, 0.0); 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [Complex64::new(0.0, 0.0); 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_thin_svd_complex64_invalid_u() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut u = [z; 1];
    let mut s = [0.0_f64; 2];
    let mut vt = [z; 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_thin_svd_complex64_invalid_s() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut u = [z; 4];
    let mut s = [0.0_f64; 1];
    let mut vt = [z; 4];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_thin_svd_complex64_invalid_vt() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut u = [z; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [z; 1];
    assert!(backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).is_err());
}

#[test]
fn faer_backend_qr_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let mut q = [z; 4];
    let mut r = [z; 4];
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_qr_complex64_invalid_q() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut q = [z; 1];
    let mut r = [z; 4];
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_qr_complex64_invalid_r() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut q = [z; 4];
    let mut r = [z; 1];
    assert!(backend.qr(&a, 2, 2, &mut q, &mut r).is_err());
}

#[test]
fn faer_backend_lu_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let mut perm = [0usize; 2];
    let mut l = [z; 4];
    let mut u_out = [z; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_complex64_invalid_perm() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut perm = [0usize; 1];
    let mut l = [z; 4];
    let mut u_out = [z; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_complex64_invalid_l() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut perm = [0usize; 2];
    let mut l = [z; 1];
    let mut u_out = [z; 4];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_lu_complex64_invalid_u() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut perm = [0usize; 2];
    let mut l = [z; 4];
    let mut u_out = [z; 1];
    assert!(backend.lu(&a, 2, 2, &mut perm, &mut l, &mut u_out).is_err());
}

#[test]
fn faer_backend_cholesky_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let mut l = [z; 4];
    assert!(backend.cholesky(&a, 2, &mut l).is_err());
}

#[test]
fn faer_backend_cholesky_complex64_invalid_l() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(4.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut l = [z; 1];
    assert!(backend.cholesky(&a, 2, &mut l).is_err());
}

#[test]
fn faer_backend_cholesky_complex64_not_pd() {
    let mut backend = FaerBackend::new();
    let c = |re, im| Complex64::new(re, im);
    let a = [c(-1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)];
    let mut l = [Complex64::new(0.0, 0.0); 4];
    assert!(backend.cholesky(&a, 2, &mut l).is_err());
}

#[test]
fn faer_backend_eigen_sym_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let a = [Complex64::new(1.0, 0.0)];
    let mut values = [0.0_f64; 2];
    let mut vectors = [Complex64::new(0.0, 0.0); 4];
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eigen_sym_complex64_invalid_values() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut values = [0.0_f64; 1];
    let mut vectors = [z; 4];
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_eigen_sym_complex64_invalid_vectors() {
    let mut backend = FaerBackend::new();
    let c = Complex64::new(1.0, 0.0);
    let z = Complex64::new(0.0, 0.0);
    let a = [c, z, z, c];
    let mut values = [0.0_f64; 2];
    let mut vectors = [z; 1];
    assert!(backend.eigen_sym(&a, 2, &mut values, &mut vectors).is_err());
}

#[test]
fn faer_backend_mat_mul_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let b = [z; 4];
    let mut c = [z; 4];
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_mat_mul_complex64_invalid_b() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [z];
    let mut c = [z; 4];
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_mat_mul_complex64_invalid_c() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [c64, z, z, c64];
    let mut c = [z; 1];
    assert!(backend.mat_mul(&a, 2, 2, &b, 2, &mut c).is_err());
}

#[test]
fn faer_backend_solve_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let b = [z; 2];
    let mut x = [z; 2];
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_complex64_invalid_b() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [z];
    let mut x = [z; 2];
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_complex64_invalid_x() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [c64, z];
    let mut x = [z];
    assert!(backend.solve(&a, &b, 2, 1, &mut x).is_err());
}

#[test]
fn faer_backend_solve_triangular_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let b = [z; 2];
    let mut x = [z; 2];
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_solve_triangular_complex64_invalid_b() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [z];
    let mut x = [z; 2];
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_solve_triangular_complex64_invalid_x() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let b = [c64, z];
    let mut x = [z];
    assert!(backend
        .solve_triangular(&a, &b, 2, 1, true, &mut x)
        .is_err());
}

#[test]
fn faer_backend_eig_general_complex64_invalid_a() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let a = [z];
    let mut values = [z; 2];
    let mut vectors = [z; 4];
    assert!(backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .is_err());
}

#[test]
fn faer_backend_eig_general_complex64_invalid_values() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let mut values = [z];
    let mut vectors = [z; 4];
    assert!(backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .is_err());
}

#[test]
fn faer_backend_eig_general_complex64_invalid_vectors() {
    let mut backend = FaerBackend::new();
    let z = Complex64::new(0.0, 0.0);
    let c64 = Complex64::new(1.0, 0.0);
    let a = [c64, z, z, c64];
    let mut values = [z; 2];
    let mut vectors = [z; 1];
    assert!(backend
        .eig_general(&a, 2, &mut values, &mut vectors)
        .is_err());
}
