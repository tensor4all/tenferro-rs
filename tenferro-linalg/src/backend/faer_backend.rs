//! Faer backend for linear algebra operations.
//!
//! This module provides the [`FaerBackend`] struct implementing
//! [`LinalgBackend`] for `f64`, `f32`, `Complex64`, and `Complex32`.

use faer::linalg::solvers::SpSolver;
use num_complex::{Complex32, Complex64};

// ============================================================================
// FaerBackend: LinalgBackend implementation using faer
// ============================================================================

use super::LinalgBackend;
use tenferro_device::{Error, Result};

/// Pure-Rust linear algebra backend powered by [faer](https://crates.io/crates/faer).
///
/// This struct is stateless; `&mut self` is accepted for future workspace reuse.
///
/// # Examples
///
/// ```
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

                let mat = faer::mat::from_column_major_slice(a, m, n);
                let svd = mat.thin_svd();

                let u_ref = svd.u();
                let v_ref = svd.v();
                let s_col = svd.s_diagonal();

                for j in 0..k {
                    for i in 0..m {
                        u[i + j * m] = u_ref[(i, j)];
                    }
                }

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

                let mat = faer::mat::from_column_major_slice(a, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_q();
                let r_mat = qr_result.compute_thin_r();

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

                let mat = faer::mat::from_column_major_slice(a, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.compute_l();
                let u_mat = lu_result.compute_u();

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

                let perm_ref = lu_result.row_permutation();
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

                let mat = faer::mat::from_column_major_slice(a, n, n);
                match mat.cholesky(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.compute_l();
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

                let mat = faer::mat::from_column_major_slice(a, n, n);
                let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);

                let u_ref = eig.u();
                let s_diag = eig.s();

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

                let a_mat = faer::mat::from_column_major_slice(a, m, k);
                let b_mat = faer::mat::from_column_major_slice(b, k, n);
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

                let a_mat = faer::mat::from_column_major_slice(a, n, n);
                let b_mat = faer::mat::from_column_major_slice(b, n, nrhs);
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

                let a_mat = faer::mat::from_column_major_slice(a, n, n);
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
        }
    };
}

impl_linalg_backend!(f64);
impl_linalg_backend!(f32);

// ============================================================================
// Complex type conversion helpers
// ============================================================================

/// Convert a slice of `Complex64` to a `Vec` of faer `c64` (safe element-wise copy).
fn to_faer_c64(src: &[Complex64]) -> Vec<faer::complex_native::c64> {
    src.iter()
        .map(|c| faer::complex_native::c64::new(c.re, c.im))
        .collect()
}

/// Copy a faer `c64` matrix (column-major) into a `Complex64` output slice.
fn from_faer_c64_mat(
    mat: faer::MatRef<'_, faer::complex_native::c64>,
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
fn to_faer_c32(src: &[Complex32]) -> Vec<faer::complex_native::c32> {
    src.iter()
        .map(|c| faer::complex_native::c32::new(c.re, c.im))
        .collect()
}

/// Copy a faer `c32` matrix (column-major) into a `Complex32` output slice.
fn from_faer_c32_mat(
    mat: faer::MatRef<'_, faer::complex_native::c32>,
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
                let mat = faer::mat::from_column_major_slice(&a_faer, m, n);
                let svd = mat.thin_svd();

                let u_ref = svd.u();
                let v_ref = svd.v();
                let s_col = svd.s_diagonal();

                // Copy U (m x k)
                $from_faer_mat(u_ref, u, m, k);

                // Singular values are real (stored as complex with zero imag)
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
                let mat = faer::mat::from_column_major_slice(&a_faer, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_q();
                let r_mat = qr_result.compute_thin_r();

                $from_faer_mat(q_mat.as_ref(), q, m, k);
                $from_faer_mat(r_mat.as_ref(), r, k, n);

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
                let mat = faer::mat::from_column_major_slice(&a_faer, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.compute_l();
                let u_mat = lu_result.compute_u();

                $from_faer_mat(l_mat.as_ref(), l, m, k);
                $from_faer_mat(u_mat.as_ref(), u_out, k, n);

                let perm_ref = lu_result.row_permutation();
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
                let mat = faer::mat::from_column_major_slice(&a_faer, n, n);
                match mat.cholesky(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.compute_l();
                        $from_faer_mat(l_mat.as_ref(), l, n, n);
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
                let mat = faer::mat::from_column_major_slice(&a_faer, n, n);
                let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);

                let u_ref = eig.u();
                let s_diag = eig.s();

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
                let a_mat = faer::mat::from_column_major_slice(&a_faer, m, k);
                let b_mat = faer::mat::from_column_major_slice(&b_faer, k, n);
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
                let a_mat = faer::mat::from_column_major_slice(&a_faer, n, n);
                let b_mat = faer::mat::from_column_major_slice(&b_faer, n, nrhs);
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
                assert!(
                    (recon[i + j * 3] - expected).abs() < 1e-10,
                    "reconstruction[{i},{j}] = {}, expected {expected}",
                    recon[i + j * 3]
                );
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
            assert!(
                (recon[idx] - a[idx]).abs() < 1e-10,
                "reconstruction[{idx}] = {}, expected {}",
                recon[idx],
                a[idx]
            );
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
            assert!(
                (recon[idx] - a[idx]).abs() < 1e-10,
                "QR reconstruction[{idx}] = {}, expected {}",
                recon[idx],
                a[idx]
            );
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
            assert!(
                (c[idx] - b[idx]).abs() < 1e-10,
                "mat_mul[{idx}] = {}, expected {}",
                c[idx],
                b[idx]
            );
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
            assert!(
                (recon[idx] - a[idx]).abs() < 1e-10,
                "Cholesky reconstruction[{idx}] = {}, expected {}",
                recon[idx],
                a[idx]
            );
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
        assert!(
            (values[0] - 1.0).abs() < 1e-10,
            "eigenvalue[0] = {}, expected 1.0",
            values[0]
        );
        assert!(
            (values[1] - 3.0).abs() < 1e-10,
            "eigenvalue[1] = {}, expected 3.0",
            values[1]
        );
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
            assert!(
                (recon[idx] - a[idx]).abs() < 1e-10,
                "LU reconstruction[{idx}] = {}, expected {}",
                recon[idx],
                a[idx]
            );
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
}
