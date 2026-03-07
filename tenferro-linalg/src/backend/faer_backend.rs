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
                                sum -= a_mat[(i, j)] * x_col[j];
                            }
                            x_col[i] = sum / a_mat[(i, i)];
                        }
                    } else {
                        // Forward substitution for lower triangular
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum -= a_mat[(i, j)] * x_col[j];
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
                                sum -= a[i + j * n] * x_col[j];
                            }
                            x_col[i] = sum / a[i + i * n];
                        }
                    } else {
                        // Forward substitution for lower triangular
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum -= a[i + j * n] * x_col[j];
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
mod tests;
