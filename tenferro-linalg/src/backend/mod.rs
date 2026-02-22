//! Backend abstraction for linear algebra operations.
//!
//! This module defines the [`LinalgBackend`] trait, which provides a
//! backend-agnostic interface for matrix decompositions and solvers.
//! Implementations write results into caller-provided output buffers
//! (pre-allocated slices), avoiding internal allocations.
//!
//! All matrices use **column-major** (Fortran) layout: element `(i, j)` of
//! an `m x n` matrix is stored at index `i + j * m`.
//!
//! # Available backends
//!
//! - **faer** (feature `faer`, enabled by default): Pure-Rust linear algebra
//!   via the [`faer`](https://crates.io/crates/faer) crate.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_linalg::backend::{LinalgBackend, FaerBackend};
//!
//! let mut backend = FaerBackend::new();
//! let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity, col-major
//! let mut u = [0.0; 4];
//! let mut s = [0.0; 2];
//! let mut vt = [0.0; 4];
//! backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
//! ```

#[cfg(feature = "faer")]
pub mod faer_backend;

#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;

use tenferro_device::Result;

/// Backend-agnostic interface for matrix linear algebra operations.
///
/// All input/output slices use **column-major** layout. The trait is
/// parameterized by scalar type `T` (e.g., `f64`, `f32`).
///
/// Implementations take `&mut self` to allow internal workspace reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::LinalgBackend;
///
/// fn do_svd<B: LinalgBackend<f64>>(backend: &mut B) {
///     let a = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
///     let mut u = [0.0; 4];
///     let mut s = [0.0; 2];
///     let mut vt = [0.0; 4];
///     backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// }
/// ```
pub trait LinalgBackend<T: Copy + 'static> {
    /// The real-valued scalar type for singular/eigenvalues.
    /// For real `T`, this is `T` itself. For complex `T`, this would be the real part type.
    type Real: Copy + 'static;

    /// Thin SVD: `A = U diag(S) Vt`.
    ///
    /// - `a`: input matrix, column-major `m x n`
    /// - `u`: output, column-major `m x k` where `k = min(m, n)`
    /// - `s`: output, vector of length `k` (singular values, descending)
    /// - `vt`: output, column-major `k x n` (conjugate transpose of V)
    fn thin_svd(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        u: &mut [T],
        s: &mut [Self::Real],
        vt: &mut [T],
    ) -> Result<()>;

    /// Thin QR decomposition: `A = Q R`.
    ///
    /// - `a`: input matrix, column-major `m x n`
    /// - `q`: output, column-major `m x k` where `k = min(m, n)`
    /// - `r`: output, column-major `k x n`
    fn qr(&mut self, a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) -> Result<()>;

    /// LU decomposition with partial pivoting: `P A = L U`.
    ///
    /// - `a`: input matrix, column-major `m x n`
    /// - `perm`: output, forward permutation vector of length `m`
    /// - `l`: output, column-major `m x k` where `k = min(m, n)`
    /// - `u_out`: output, column-major `k x n`
    fn lu(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        perm: &mut [usize],
        l: &mut [T],
        u_out: &mut [T],
    ) -> Result<()>;

    /// Cholesky decomposition: `A = L L^H`.
    ///
    /// - `a`: input matrix, column-major `n x n` (must be symmetric positive definite)
    /// - `l`: output, column-major `n x n` (lower triangular)
    fn cholesky(&mut self, a: &[T], n: usize, l: &mut [T]) -> Result<()>;

    /// Symmetric eigendecomposition: `A = V diag(lambda) V^H`.
    ///
    /// - `a`: input matrix, column-major `n x n` (must be symmetric/Hermitian)
    /// - `values`: output, eigenvalues in ascending order, length `n`
    /// - `vectors`: output, eigenvectors column-major `n x n`
    fn eigen_sym(
        &mut self,
        a: &[T],
        n: usize,
        values: &mut [Self::Real],
        vectors: &mut [T],
    ) -> Result<()>;

    /// Matrix multiplication: `C = A * B`.
    ///
    /// - `a`: input, column-major `m x k`
    /// - `b`: input, column-major `k x n`
    /// - `c`: output, column-major `m x n`
    fn mat_mul(
        &mut self,
        a: &[T],
        m: usize,
        k: usize,
        b: &[T],
        n: usize,
        c: &mut [T],
    ) -> Result<()>;

    /// Solve linear system: `A x = b`.
    ///
    /// - `a`: input, column-major `n x n`
    /// - `b`: input, column-major `n x nrhs`
    /// - `x`: output, column-major `n x nrhs`
    fn solve(&mut self, a: &[T], b: &[T], n: usize, nrhs: usize, x: &mut [T]) -> Result<()>;

    /// Solve triangular system: `A x = b`.
    ///
    /// - `a`: input, column-major `n x n` (upper or lower triangular)
    /// - `b`: input, column-major `n x nrhs`
    /// - `upper`: if `true`, `A` is upper triangular; otherwise lower triangular
    /// - `x`: output, column-major `n x nrhs`
    fn solve_triangular(
        &mut self,
        a: &[T],
        b: &[T],
        n: usize,
        nrhs: usize,
        upper: bool,
        x: &mut [T],
    ) -> Result<()>;
}

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
