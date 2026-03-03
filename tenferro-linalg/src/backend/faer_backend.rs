//! Faer backend for linear algebra operations.
//!
//! This module provides the [`FaerBackend`] struct implementing
//! [`LinalgBackend`] for `f64`, `f32`, `Complex64`, and `Complex32`.

use num_complex::{Complex32, Complex64};

// ============================================================================
// FaerBackend: LinalgBackend implementation using faer
// ============================================================================

use super::LinalgBackend;
use faer::linalg::solvers::Solve;
use tenferro_device::{Error, Result};

/// Pure-Rust linear algebra backend powered by [faer](https://crates.io/crates/faer).
///
/// This struct is stateless; `&mut self` is accepted for future workspace reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{FaerBackend, LinalgBackend};
///
/// let mut backend = FaerBackend::new();
/// let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity, col-major
/// let mut u = [0.0; 4];
/// let mut s = [0.0; 2];
/// let mut vt = [0.0; 4];
/// backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FaerBackend;

impl FaerBackend {
    /// Create a new `FaerBackend`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FaerBackend {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_linalg_backend {
    ($ty:ty) => {
        impl LinalgBackend<$ty> for FaerBackend {
            type Real = $ty;

            fn thin_svd(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                u: &mut [$ty],
                s: &mut [Self::Real],
                vt: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if u.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: u slice length {} < m*k = {}",
                        u.len(),
                        m * k
                    )));
                }
                if s.len() < k {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: s slice length {} < k = {}",
                        s.len(),
                        k
                    )));
                }
                if vt.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: vt slice length {} < k*n = {}",
                        vt.len(),
                        k * n
                    )));
                }

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let svd = mat.thin_svd().map_err(|_| {
                    Error::InvalidArgument("thin_svd: SVD computation failed".into())
                })?;

                let u_ref = svd.U();
                let v_ref = svd.V();
                let s_diag = svd.S();

                for j in 0..k {
                    for i in 0..m {
                        u[i + j * m] = u_ref[(i, j)];
                    }
                }

                let s_col = s_diag.column_vector();
                for i in 0..k {
                    s[i] = s_col[i];
                }

                // Write Vt (k x n) directly: vt[i + j*k] = V[j, i]
                for j in 0..n {
                    for i in 0..k {
                        vt[i + j * k] = v_ref[(j, i)];
                    }
                }

                Ok(())
            }

            fn qr(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                q: &mut [$ty],
                r: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "qr: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if q.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "qr: q slice length {} < m*k = {}",
                        q.len(),
                        m * k
                    )));
                }
                if r.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "qr: r slice length {} < k*n = {}",
                        r.len(),
                        k * n
                    )));
                }

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_Q();
                let r_mat = qr_result.thin_R();

                for j in 0..k {
                    for i in 0..m {
                        q[i + j * m] = q_mat[(i, j)];
                    }
                }

                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = r_mat[(i, j)];
                    }
                }

                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$ty],
                u_out: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "lu: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if perm.len() < m {
                    return Err(Error::InvalidArgument(format!(
                        "lu: perm slice length {} < m = {}",
                        perm.len(),
                        m
                    )));
                }
                if l.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "lu: l slice length {} < m*k = {}",
                        l.len(),
                        m * k
                    )));
                }
                if u_out.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "lu: u_out slice length {} < k*n = {}",
                        u_out.len(),
                        k * n
                    )));
                }

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.L();
                let u_mat = lu_result.U();

                for j in 0..k {
                    for i in 0..m {
                        l[i + j * m] = l_mat[(i, j)];
                    }
                }

                for j in 0..n {
                    for i in 0..k {
                        u_out[i + j * k] = u_mat[(i, j)];
                    }
                }

                let perm_ref = lu_result.P();
                let (fwd, _inv) = perm_ref.arrays();
                perm[..m].copy_from_slice(fwd);

                Ok(())
            }

            fn cholesky(&mut self, a: &[$ty], n: usize, l: &mut [$ty]) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "cholesky: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if l.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "cholesky: l slice length {} < n*n = {}",
                        l.len(),
                        n * n
                    )));
                }

                let mat = faer::MatRef::from_column_major_slice(a, n, n);
                match mat.llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.L();
                        for j in 0..n {
                            for i in 0..n {
                                l[i + j * n] = l_mat[(i, j)];
                            }
                        }
                        Ok(())
                    }
                    Err(_) => Err(Error::InvalidArgument(
                        "cholesky: matrix is not positive definite".to_string(),
                    )),
                }
            }

            fn eigen_sym(
                &mut self,
                a: &[$ty],
                n: usize,
                values: &mut [Self::Real],
                vectors: &mut [$ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if values.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: values slice length {} < n = {}",
                        values.len(),
                        n
                    )));
                }
                if vectors.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: vectors slice length {} < n*n = {}",
                        vectors.len(),
                        n * n
                    )));
                }

                let mat = faer::MatRef::from_column_major_slice(a, n, n);
                let eig = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|_| {
                    Error::InvalidArgument("eigen_sym: eigendecomposition failed".into())
                })?;

                let u_ref = eig.U();
                let s_diag = eig.S();

                for j in 0..n {
                    for i in 0..n {
                        vectors[i + j * n] = u_ref[(i, j)];
                    }
                }

                let s_col = s_diag.column_vector();
                for i in 0..n {
                    values[i] = s_col[i];
                }

                Ok(())
            }

            fn mat_mul(
                &mut self,
                a: &[$ty],
                m: usize,
                k: usize,
                b: &[$ty],
                n: usize,
                c: &mut [$ty],
            ) -> Result<()> {
                if a.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: a slice length {} < m*k = {}",
                        a.len(),
                        m * k
                    )));
                }
                if b.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: b slice length {} < k*n = {}",
                        b.len(),
                        k * n
                    )));
                }
                if c.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: c slice length {} < m*n = {}",
                        c.len(),
                        m * n
                    )));
                }

                let a_mat = faer::MatRef::from_column_major_slice(a, m, k);
                let b_mat = faer::MatRef::from_column_major_slice(b, k, n);
                let result = &a_mat * &b_mat;

                for j in 0..n {
                    for i in 0..m {
                        c[i + j * m] = result[(i, j)];
                    }
                }

                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                x: &mut [$ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "solve: a slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if b.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve: b slice length {} < n*nrhs = {}",
                        b.len(),
                        n * nrhs
                    )));
                }
                if x.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve: x slice length {} < n*nrhs = {}",
                        x.len(),
                        n * nrhs
                    )));
                }

                let a_mat = faer::MatRef::from_column_major_slice(a, n, n);
                let b_mat = faer::MatRef::from_column_major_slice(b, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let result = lu.solve(&b_mat);

                for j in 0..nrhs {
                    for i in 0..n {
                        x[i + j * n] = result[(i, j)];
                    }
                }

                Ok(())
            }

            fn solve_triangular(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: a slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if b.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: b slice length {} < n*nrhs = {}",
                        b.len(),
                        n * nrhs
                    )));
                }
                if x.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: x slice length {} < n*nrhs = {}",
                        x.len(),
                        n * nrhs
                    )));
                }

                let a_mat = faer::MatRef::from_column_major_slice(a, n, n);
                for col in 0..nrhs {
                    let b_col = &b[col * n..(col + 1) * n];
                    let x_col = &mut x[col * n..(col + 1) * n];

                    if upper {
                        // Back substitution for upper triangular
                        for i in (0..n).rev() {
                            let mut sum = b_col[i];
                            for j in (i + 1)..n {
                                sum = sum - a_mat[(i, j)] * x_col[j];
                            }
                            x_col[i] = sum / a_mat[(i, i)];
                        }
                    } else {
                        // Forward substitution for lower triangular
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum = sum - a_mat[(i, j)] * x_col[j];
                            }
                            x_col[i] = sum / a_mat[(i, i)];
                        }
                    }
                }

                Ok(())
            }

            fn eig_general(
                &mut self,
                a: &[$ty],
                n: usize,
                values_ri: &mut [$ty],
                vectors_ri: &mut [$ty],
            ) -> Result<()> {
                use faer::c64;

                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if values_ri.len() < 2 * n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: values_ri slice length {} < 2*n = {}",
                        values_ri.len(),
                        2 * n
                    )));
                }
                if vectors_ri.len() < 2 * n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: vectors_ri slice length {} < 2*n*n = {}",
                        vectors_ri.len(),
                        2 * n * n
                    )));
                }

                // Convert real input to complex for faer eigendecomposition
                let a_complex: Vec<c64> = a[..n * n]
                    .iter()
                    .map(|&v| c64::new(v as f64, 0.0))
                    .collect();
                let mat = faer::MatRef::from_column_major_slice(&a_complex, n, n);
                let eig = mat.eigen().map_err(|e| {
                    Error::InvalidArgument(format!("eigendecomposition failed: {e:?}"))
                })?;

                let s_diag = eig.S();
                let s_col = s_diag.column_vector();
                let u_ref = eig.U();

                // Write eigenvalues as interleaved [re, im, re, im, ...]
                for i in 0..n {
                    let val = s_col[i];
                    values_ri[2 * i] = val.re as $ty;
                    values_ri[2 * i + 1] = val.im as $ty;
                }

                // Write eigenvectors as interleaved column-major
                for j in 0..n {
                    for i in 0..n {
                        let val = u_ref[(i, j)];
                        vectors_ri[2 * (i + j * n)] = val.re as $ty;
                        vectors_ri[2 * (i + j * n) + 1] = val.im as $ty;
                    }
                }

                Ok(())
            }
        }
    };
}

impl_linalg_backend!(f64);
impl_linalg_backend!(f32);

// ============================================================================
// Complex type conversion helpers
// ============================================================================

/// Convert a slice of `Complex64` to a `Vec` of faer `c64` (safe element-wise copy).
fn to_faer_c64(src: &[Complex64]) -> Vec<faer::c64> {
    src.iter().map(|c| faer::c64::new(c.re, c.im)).collect()
}

/// Copy a faer `c64` matrix (column-major) into a `Complex64` output slice.
fn from_faer_c64_mat(
    mat: faer::MatRef<'_, faer::c64>,
    out: &mut [Complex64],
    rows: usize,
    cols: usize,
) {
    for j in 0..cols {
        for i in 0..rows {
            let v = mat[(i, j)];
            out[i + j * rows] = Complex64::new(v.re, v.im);
        }
    }
}

/// Convert a slice of `Complex32` to a `Vec` of faer `c32` (safe element-wise copy).
fn to_faer_c32(src: &[Complex32]) -> Vec<faer::c32> {
    src.iter().map(|c| faer::c32::new(c.re, c.im)).collect()
}

/// Copy a faer `c32` matrix (column-major) into a `Complex32` output slice.
fn from_faer_c32_mat(
    mat: faer::MatRef<'_, faer::c32>,
    out: &mut [Complex32],
    rows: usize,
    cols: usize,
) {
    for j in 0..cols {
        for i in 0..rows {
            let v = mat[(i, j)];
            out[i + j * rows] = Complex32::new(v.re, v.im);
        }
    }
}

// ============================================================================
// Complex LinalgBackend implementations
// ============================================================================

/// Macro for implementing `LinalgBackend` for complex types (`Complex64`, `Complex32`).
///
/// Unlike the real-valued macro, this one converts between `num_complex` types
/// and faer's native complex types (`c64`, `c32`) via safe element-wise copy.
macro_rules! impl_complex_linalg_backend {
    ($complex_ty:ty, $real_ty:ty, $to_faer:ident, $from_faer_mat:ident) => {
        impl LinalgBackend<$complex_ty> for FaerBackend {
            type Real = $real_ty;

            fn thin_svd(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                u: &mut [$complex_ty],
                s: &mut [Self::Real],
                vt: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if u.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: u slice length {} < m*k = {}",
                        u.len(),
                        m * k
                    )));
                }
                if s.len() < k {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: s slice length {} < k = {}",
                        s.len(),
                        k
                    )));
                }
                if vt.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "thin_svd: vt slice length {} < k*n = {}",
                        vt.len(),
                        k * n
                    )));
                }

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let svd = mat.thin_svd().map_err(|_| {
                    Error::InvalidArgument("thin_svd: SVD computation failed".into())
                })?;

                let u_ref = svd.U();
                let v_ref = svd.V();
                let s_diag = svd.S();

                // Copy U (m x k)
                $from_faer_mat(u_ref, u, m, k);

                // Singular values are real
                let s_col = s_diag.column_vector();
                for i in 0..k {
                    s[i] = s_col[i].re;
                }

                // Vt = conjugate transpose of V: vt[i + j*k] = conj(V[j, i])
                for j in 0..n {
                    for i in 0..k {
                        let v = v_ref[(j, i)];
                        vt[i + j * k] = <$complex_ty>::new(v.re, -v.im);
                    }
                }

                Ok(())
            }

            fn qr(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                q: &mut [$complex_ty],
                r: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "qr: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if q.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "qr: q slice length {} < m*k = {}",
                        q.len(),
                        m * k
                    )));
                }
                if r.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "qr: r slice length {} < k*n = {}",
                        r.len(),
                        k * n
                    )));
                }

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_Q();
                let r_mat = qr_result.thin_R();

                $from_faer_mat(q_mat.as_ref(), q, m, k);
                $from_faer_mat(r_mat, r, k, n);

                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$complex_ty],
                u_out: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                if a.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "lu: input slice length {} < m*n = {}",
                        a.len(),
                        m * n
                    )));
                }
                if perm.len() < m {
                    return Err(Error::InvalidArgument(format!(
                        "lu: perm slice length {} < m = {}",
                        perm.len(),
                        m
                    )));
                }
                if l.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "lu: l slice length {} < m*k = {}",
                        l.len(),
                        m * k
                    )));
                }
                if u_out.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "lu: u_out slice length {} < k*n = {}",
                        u_out.len(),
                        k * n
                    )));
                }

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.L();
                let u_mat = lu_result.U();

                $from_faer_mat(l_mat, l, m, k);
                $from_faer_mat(u_mat, u_out, k, n);

                let perm_ref = lu_result.P();
                let (fwd, _inv) = perm_ref.arrays();
                perm[..m].copy_from_slice(fwd);

                Ok(())
            }

            fn cholesky(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                l: &mut [$complex_ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "cholesky: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if l.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "cholesky: l slice length {} < n*n = {}",
                        l.len(),
                        n * n
                    )));
                }

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                match mat.llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.L();
                        $from_faer_mat(l_mat, l, n, n);
                        Ok(())
                    }
                    Err(_) => Err(Error::InvalidArgument(
                        "cholesky: matrix is not positive definite".to_string(),
                    )),
                }
            }

            fn eigen_sym(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                values: &mut [Self::Real],
                vectors: &mut [$complex_ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if values.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: values slice length {} < n = {}",
                        values.len(),
                        n
                    )));
                }
                if vectors.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eigen_sym: vectors slice length {} < n*n = {}",
                        vectors.len(),
                        n * n
                    )));
                }

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                let eig = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|_| {
                    Error::InvalidArgument("eigen_sym: eigendecomposition failed".into())
                })?;

                let u_ref = eig.U();
                let s_diag = eig.S();

                $from_faer_mat(u_ref, vectors, n, n);

                // Eigenvalues of a Hermitian matrix are real
                let s_col = s_diag.column_vector();
                for i in 0..n {
                    values[i] = s_col[i].re;
                }

                Ok(())
            }

            fn mat_mul(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                k: usize,
                b: &[$complex_ty],
                n: usize,
                c: &mut [$complex_ty],
            ) -> Result<()> {
                if a.len() < m * k {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: a slice length {} < m*k = {}",
                        a.len(),
                        m * k
                    )));
                }
                if b.len() < k * n {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: b slice length {} < k*n = {}",
                        b.len(),
                        k * n
                    )));
                }
                if c.len() < m * n {
                    return Err(Error::InvalidArgument(format!(
                        "mat_mul: c slice length {} < m*n = {}",
                        c.len(),
                        m * n
                    )));
                }

                let a_faer = $to_faer(a);
                let b_faer = $to_faer(b);
                let a_mat = faer::MatRef::from_column_major_slice(&a_faer, m, k);
                let b_mat = faer::MatRef::from_column_major_slice(&b_faer, k, n);
                let result = &a_mat * &b_mat;

                $from_faer_mat(result.as_ref(), c, m, n);

                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$complex_ty],
                b: &[$complex_ty],
                n: usize,
                nrhs: usize,
                x: &mut [$complex_ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "solve: a slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if b.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve: b slice length {} < n*nrhs = {}",
                        b.len(),
                        n * nrhs
                    )));
                }
                if x.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve: x slice length {} < n*nrhs = {}",
                        x.len(),
                        n * nrhs
                    )));
                }

                let a_faer = $to_faer(a);
                let b_faer = $to_faer(b);
                let a_mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                let b_mat = faer::MatRef::from_column_major_slice(&b_faer, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let result = lu.solve(&b_mat);

                $from_faer_mat(result.as_ref(), x, n, nrhs);

                Ok(())
            }

            fn solve_triangular(
                &mut self,
                a: &[$complex_ty],
                b: &[$complex_ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$complex_ty],
            ) -> Result<()> {
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: a slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if b.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: b slice length {} < n*nrhs = {}",
                        b.len(),
                        n * nrhs
                    )));
                }
                if x.len() < n * nrhs {
                    return Err(Error::InvalidArgument(format!(
                        "solve_triangular: x slice length {} < n*nrhs = {}",
                        x.len(),
                        n * nrhs
                    )));
                }

                // Perform forward/back substitution directly on Complex slices.
                // Use column-major indexing: a[(i,j)] = a[i + j*n].
                for col in 0..nrhs {
                    let b_col = &b[col * n..(col + 1) * n];
                    let x_col = &mut x[col * n..(col + 1) * n];

                    if upper {
                        // Back substitution for upper triangular
                        for i in (0..n).rev() {
                            let mut sum = b_col[i];
                            for j in (i + 1)..n {
                                sum = sum - a[i + j * n] * x_col[j];
                            }
                            x_col[i] = sum / a[i + i * n];
                        }
                    } else {
                        // Forward substitution for lower triangular
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum = sum - a[i + j * n] * x_col[j];
                            }
                            x_col[i] = sum / a[i + i * n];
                        }
                    }
                }

                Ok(())
            }

            fn eig_general(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                values_ri: &mut [$complex_ty],
                vectors_ri: &mut [$complex_ty],
            ) -> Result<()> {
                use faer::c64;

                // For complex T, each element already holds re+im, so
                // values_ri has length n (not 2*n) and vectors_ri has length n*n.
                if a.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: input slice length {} < n*n = {}",
                        a.len(),
                        n * n
                    )));
                }
                if values_ri.len() < n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: values_ri slice length {} < n = {}",
                        values_ri.len(),
                        n
                    )));
                }
                if vectors_ri.len() < n * n {
                    return Err(Error::InvalidArgument(format!(
                        "eig_general: vectors_ri slice length {} < n*n = {}",
                        vectors_ri.len(),
                        n * n
                    )));
                }

                // Always convert to c64 for eigendecomposition (works for both
                // Complex64 and Complex32 input, avoiding potential c32 limitations)
                let a_c64: Vec<c64> = a[..n * n]
                    .iter()
                    .map(|c| c64::new(c.re as f64, c.im as f64))
                    .collect();
                let mat = faer::MatRef::from_column_major_slice(&a_c64, n, n);
                let eig = mat.eigen().map_err(|e| {
                    Error::InvalidArgument(format!("eigendecomposition failed: {e:?}"))
                })?;

                let s_diag = eig.S();
                let s_col = s_diag.column_vector();
                let u_ref = eig.U();

                for i in 0..n {
                    let val = s_col[i];
                    values_ri[i] = <$complex_ty>::new(val.re as $real_ty, val.im as $real_ty);
                }

                for j in 0..n {
                    for i in 0..n {
                        let val = u_ref[(i, j)];
                        vectors_ri[i + j * n] =
                            <$complex_ty>::new(val.re as $real_ty, val.im as $real_ty);
                    }
                }

                Ok(())
            }
        }
    };
}

impl_complex_linalg_backend!(Complex64, f64, to_faer_c64, from_faer_c64_mat);
impl_complex_linalg_backend!(Complex32, f32, to_faer_c32, from_faer_c32_mat);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
        let backend = FaerBackend::default();
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
}
