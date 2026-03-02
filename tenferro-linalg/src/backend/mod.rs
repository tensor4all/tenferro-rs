//! Backend abstraction for linear algebra operations.
//!
//! This module provides both the slice-level [`LinalgBackend`] trait (used
//! internally by the CPU provider) and the tensor-level
//! [`TensorLinalgBackend`] trait (the public backend boundary).
//!
//! # CPU provider selection
//!
//! Exactly one of the following features must be enabled:
//!
//! - `linalg-faer`: Pure-Rust via [`faer`](https://crates.io/crates/faer) (default)
//! - `linalg-lapack`: External LAPACK binding (placeholder)
//!
//! Enabling both or neither is a compile error.
//!
//! # Device backends
//!
//! - **CPU**: [`CpuTensorLinalgBackend`] / [`CpuTensorLinalgContext`]
//! - **CUDA**: [`CudaTensorLinalgBackend`] / [`CudaTensorLinalgContext`] (stub)
//! - **HIP**: [`HipTensorLinalgBackend`] / [`HipTensorLinalgContext`] (stub)
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_linalg::backend::{CpuTensorLinalgContext, TensorLinalgBackend, CpuTensorLinalgBackend};
//! use tenferro_tensor::Tensor;
//!
//! let mut ctx = CpuTensorLinalgContext::new();
//! let a: Tensor<f64> = todo!();
//! let b: Tensor<f64> = todo!();
//! let _x = <CpuTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(&mut ctx, &a, &b).unwrap();
//! ```

// ============================================================================
// Feature policy: exactly one CPU linalg provider must be enabled
// ============================================================================

#[cfg(all(feature = "linalg-faer", feature = "linalg-lapack"))]
compile_error!(
    "Features `linalg-faer` and `linalg-lapack` are mutually exclusive. Enable exactly one."
);

#[cfg(not(any(feature = "linalg-faer", feature = "linalg-lapack")))]
compile_error!("No CPU linalg provider selected. Enable `linalg-faer` or `linalg-lapack`.");

// ============================================================================
// Submodules
// ============================================================================

// Slice-level backend (internal implementation detail)
#[cfg(feature = "linalg-faer")]
pub mod faer_backend;

// Tensor-level API and types
pub mod tensor_api;
pub mod tensor_context;
pub(crate) mod tensor_helpers;

// Device backends
pub mod cpu;
#[cfg(feature = "linalg-faer")]
pub(crate) mod cpu_faer;
#[cfg(feature = "linalg-lapack")]
pub(crate) mod cpu_lapack;
pub mod cuda;
pub mod hip;

// ============================================================================
// Re-exports
// ============================================================================

// Tensor-level API (public)
pub use tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgBackend,
};
pub use tensor_context::TensorLinalgContextFor;

// CPU backend (public)
pub use cpu::{CpuTensorLinalgBackend, CpuTensorLinalgContext};

// GPU backend stubs (public)
pub use cuda::{CudaTensorLinalgBackend, CudaTensorLinalgContext};
pub use hip::{HipTensorLinalgBackend, HipTensorLinalgContext};

// Slice-level backend (still public for backward compatibility during migration)
#[cfg(feature = "linalg-faer")]
pub use faer_backend::FaerBackend;

use tenferro_device::Result;

/// Slice-level backend interface for matrix linear algebra operations.
///
/// All input/output slices use **column-major** layout. The trait is
/// parameterized by scalar type `T` (e.g., `f64`, `f32`).
///
/// Implementations take `&mut self` to allow internal workspace reuse.
///
/// This trait is used internally by CPU provider implementations.
/// The public API boundary is [`TensorLinalgBackend`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::backend::LinalgBackend;
///
/// fn do_svd<B: LinalgBackend<f64, Real = f64>>(backend: &mut B) {
///     let a = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
///     let mut u = [0.0; 4];
///     let mut s = [0.0; 2];
///     let mut vt = [0.0; 4];
///     backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// }
/// ```
pub trait LinalgBackend<T: Copy + 'static> {
    /// The real-valued scalar type for singular/eigenvalues.
    type Real: Copy + 'static;

    /// Thin SVD: `A = U diag(S) Vt`.
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
    fn qr(&mut self, a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) -> Result<()>;

    /// LU decomposition with partial pivoting: `P A = L U`.
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
    fn cholesky(&mut self, a: &[T], n: usize, l: &mut [T]) -> Result<()>;

    /// Symmetric eigendecomposition: `A = V diag(lambda) V^H`.
    fn eigen_sym(
        &mut self,
        a: &[T],
        n: usize,
        values: &mut [Self::Real],
        vectors: &mut [T],
    ) -> Result<()>;

    /// Matrix multiplication: `C = A * B`.
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
    fn solve(&mut self, a: &[T], b: &[T], n: usize, nrhs: usize, x: &mut [T]) -> Result<()>;

    /// Solve triangular system: `A x = b`.
    fn solve_triangular(
        &mut self,
        a: &[T],
        b: &[T],
        n: usize,
        nrhs: usize,
        upper: bool,
        x: &mut [T],
    ) -> Result<()>;

    /// General eigendecomposition: `A V = V diag(lambda)`.
    ///
    /// For real `T`: output uses interleaved re/im pairs (`2*n` values, `2*n*n` vectors).
    /// For complex `T`: output uses direct complex elements (`n` values, `n*n` vectors).
    fn eig_general(
        &mut self,
        a: &[T],
        n: usize,
        values_ri: &mut [T],
        vectors_ri: &mut [T],
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
