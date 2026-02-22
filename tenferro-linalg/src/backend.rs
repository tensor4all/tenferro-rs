//! Private faer backend dispatch for linear algebra operations.
//!
//! This module provides type-erased dispatch to faer decompositions
//! via the `FaerOps` trait, implemented for `f64` and `f32`.

use faer::linalg::solvers::SpSolver;
use num_traits::Zero;

/// Backend dispatch trait for faer linear algebra operations.
///
/// Each method takes raw column-major data and returns raw results.
/// The public API handles Tensor wrapping/unwrapping and batching.
pub(crate) trait FaerOps:
    Copy
    + Zero
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + 'static
{
    /// Thin SVD: A = U diag(S) V^H.
    /// Returns (U col-major m×k, S vector k, V col-major n×k) where k=min(m,n).
    /// Singular values are in descending order.
    fn thin_svd(data: &[Self], m: usize, n: usize) -> (Vec<Self>, Vec<Self>, Vec<Self>);

    /// Thin QR: A = Q R.
    /// Returns (Q col-major m×k, R col-major k×n) where k=min(m,n).
    fn qr_decomp(data: &[Self], m: usize, n: usize) -> (Vec<Self>, Vec<Self>);

    /// LU with partial pivoting: P A = L U.
    /// Returns (perm forward m, L col-major m×k, U col-major k×n) where k=min(m,n).
    fn lu_decomp(data: &[Self], m: usize, n: usize) -> (Vec<usize>, Vec<Self>, Vec<Self>);

    /// Cholesky: A = L L^H.
    /// Returns L col-major n×n, or error message.
    fn cholesky_decomp(data: &[Self], n: usize) -> std::result::Result<Vec<Self>, String>;

    /// Symmetric eigendecomposition: A = V diag(λ) V^H.
    /// Returns (eigenvalues ascending n, eigenvectors col-major n×n).
    fn eigen_sym(data: &[Self], n: usize) -> (Vec<Self>, Vec<Self>);

    /// Matrix multiply: C = A * B, all column-major.
    /// A is m×k, B is k×n, C is m×n.
    fn mat_mul(a: &[Self], m: usize, k: usize, b: &[Self], n: usize) -> Vec<Self>;

    /// Solve linear system: A x = b.
    /// A is n×n col-major, b is n×nrhs col-major.
    /// Returns x as n×nrhs col-major.
    fn mat_solve(a: &[Self], b: &[Self], n: usize, nrhs: usize) -> Vec<Self>;

    /// Solve triangular system.
    /// A is n×n col-major (upper or lower triangular), b is n×nrhs col-major.
    /// Returns x as n×nrhs col-major.
    fn mat_solve_triangular(
        a: &[Self],
        b: &[Self],
        n: usize,
        nrhs: usize,
        upper: bool,
    ) -> Vec<Self>;
}

macro_rules! impl_faer_ops {
    ($ty:ty) => {
        impl FaerOps for $ty {
            fn thin_svd(data: &[Self], m: usize, n: usize) -> (Vec<Self>, Vec<Self>, Vec<Self>) {
                let mat = faer::mat::from_column_major_slice(data, m, n);
                let svd = mat.thin_svd();
                let k = m.min(n);

                let u_ref = svd.u();
                let v_ref = svd.v();
                let s_col = svd.s_diagonal();

                let mut u = vec![<$ty as Zero>::zero(); m * k];
                for j in 0..k {
                    for i in 0..m {
                        u[i + j * m] = u_ref[(i, j)];
                    }
                }

                let mut s = vec![<$ty as Zero>::zero(); k];
                for i in 0..k {
                    s[i] = s_col[i];
                }

                let mut v = vec![<$ty as Zero>::zero(); n * k];
                for j in 0..k {
                    for i in 0..n {
                        v[i + j * n] = v_ref[(i, j)];
                    }
                }

                (u, s, v)
            }

            fn qr_decomp(data: &[Self], m: usize, n: usize) -> (Vec<Self>, Vec<Self>) {
                let mat = faer::mat::from_column_major_slice(data, m, n);
                let qr = mat.qr();
                let k = m.min(n);

                let q_mat = qr.compute_thin_q();
                let r_mat = qr.compute_thin_r();

                let mut q = vec![<$ty as Zero>::zero(); m * k];
                for j in 0..k {
                    for i in 0..m {
                        q[i + j * m] = q_mat[(i, j)];
                    }
                }

                let mut r = vec![<$ty as Zero>::zero(); k * n];
                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = r_mat[(i, j)];
                    }
                }

                (q, r)
            }

            fn lu_decomp(data: &[Self], m: usize, n: usize) -> (Vec<usize>, Vec<Self>, Vec<Self>) {
                let mat = faer::mat::from_column_major_slice(data, m, n);
                let lu = mat.partial_piv_lu();
                let k = m.min(n);

                let l_mat = lu.compute_l();
                let u_mat = lu.compute_u();

                let mut l = vec![<$ty as Zero>::zero(); m * k];
                for j in 0..k {
                    for i in 0..m {
                        l[i + j * m] = l_mat[(i, j)];
                    }
                }

                let mut u = vec![<$ty as Zero>::zero(); k * n];
                for j in 0..n {
                    for i in 0..k {
                        u[i + j * k] = u_mat[(i, j)];
                    }
                }

                let perm_ref = lu.row_permutation();
                let (fwd, _inv) = perm_ref.arrays();
                let p: Vec<usize> = fwd.to_vec();

                (p, l, u)
            }

            fn cholesky_decomp(data: &[Self], n: usize) -> std::result::Result<Vec<Self>, String> {
                let mat = faer::mat::from_column_major_slice(data, n, n);
                match mat.cholesky(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.compute_l();
                        let mut l = vec![<$ty as Zero>::zero(); n * n];
                        for j in 0..n {
                            for i in 0..n {
                                l[i + j * n] = l_mat[(i, j)];
                            }
                        }
                        Ok(l)
                    }
                    Err(_) => Err("matrix is not positive definite".to_string()),
                }
            }

            fn eigen_sym(data: &[Self], n: usize) -> (Vec<Self>, Vec<Self>) {
                let mat = faer::mat::from_column_major_slice(data, n, n);
                let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);

                let u_ref = eig.u();
                let s_diag = eig.s();

                let mut vectors = vec![<$ty as Zero>::zero(); n * n];
                for j in 0..n {
                    for i in 0..n {
                        vectors[i + j * n] = u_ref[(i, j)];
                    }
                }

                let s_col = s_diag.column_vector();
                let mut values = vec![<$ty as Zero>::zero(); n];
                for i in 0..n {
                    values[i] = s_col[i];
                }

                (values, vectors)
            }

            fn mat_mul(a: &[Self], m: usize, k: usize, b: &[Self], n: usize) -> Vec<Self> {
                let a_mat = faer::mat::from_column_major_slice(a, m, k);
                let b_mat = faer::mat::from_column_major_slice(b, k, n);
                let c = &a_mat * &b_mat;
                let mut result = vec![<$ty as Zero>::zero(); m * n];
                for j in 0..n {
                    for i in 0..m {
                        result[i + j * m] = c[(i, j)];
                    }
                }
                result
            }

            fn mat_solve(a: &[Self], b: &[Self], n: usize, nrhs: usize) -> Vec<Self> {
                let a_mat = faer::mat::from_column_major_slice(a, n, n);
                let b_mat = faer::mat::from_column_major_slice(b, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let x = lu.solve(&b_mat);
                let mut result = vec![<$ty as Zero>::zero(); n * nrhs];
                for j in 0..nrhs {
                    for i in 0..n {
                        result[i + j * n] = x[(i, j)];
                    }
                }
                result
            }

            fn mat_solve_triangular(
                a: &[Self],
                b: &[Self],
                n: usize,
                nrhs: usize,
                upper: bool,
            ) -> Vec<Self> {
                let a_mat = faer::mat::from_column_major_slice(a, n, n);
                let mut x = vec![<$ty as Zero>::zero(); n * nrhs];
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
                x
            }
        }
    };
}

impl_faer_ops!(f64);
impl_faer_ops!(f32);

// ============================================================================
// Batch helpers
// ============================================================================

/// Compute column-major strides for given dimensions.
pub(crate) fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    strides[0] = 1;
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1] as isize;
    }
    strides
}
