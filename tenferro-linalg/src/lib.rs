//! Batched matrix linear algebra decompositions with AD rules.
//!
//! This crate provides SVD, QR, LU, eigendecomposition, Cholesky, least squares,
//! linear solve, matrix inverse, determinant, pseudoinverse, matrix exponential,
//! triangular solve, and norms for tensors
//! with shape `(m, n, *)`, adapted from PyTorch's `torch.linalg` for
//! column-major layout:
//!
//! - **First 2 dimensions** are the matrix (`m × n`).
//! - **All following dimensions** (`*`) are independent batch dimensions.
//! - Input must be **column-major contiguous** (LAPACK/cuSOLVER native).
//!
//! This convention mirrors PyTorch's `(*, m, n)` but is flipped for
//! col-major: in col-major the first dimensions are contiguous, so
//! placing the matrix there ensures LAPACK can operate directly without
//! transposition.
//!
//! This module is **context-agnostic**: it does not know about tensor
//! networks, MPS, or any specific application. If you need to decompose
//! a tensor along arbitrary legs, `permute` + `reshape` +
//! `contiguous(ColumnMajor)` before calling these functions.
//!
//! # AD rules
//!
//! Each decomposition has stateless `_rrule` (reverse-mode / VJP) and
//! `_frule` (forward-mode / JVP) functions. These implement matrix-level
//! AD formulas (Mathieu 2019 et al.) using batched operations that
//! naturally broadcast over batch dimensions `*`.
//!
//! There are no `tracked_*` / `dual_*` functions — the chainrules tape
//! engine composes `permute_backward` + `reshape_backward` + `svd_rrule`
//! via the standard chain rule automatically.
//!
//! # Examples
//!
//! ## SVD of a matrix
//!
//! ```ignore
//! use tenferro_linalg::{svd, SvdOptions};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! // 2D matrix: shape [3, 4]
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&a, None).unwrap();
//! // result.u:  shape [3, 3]  (m × k, k = min(m,n) = 3)
//! // result.s:  shape [3]     (singular values)
//! // result.vt: shape [3, 4]  (k × n)
//! ```
//!
//! ## Batched SVD
//!
//! ```ignore
//! use tenferro_linalg::svd;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! // Batched: shape [m, n, batch] = [3, 4, 10]
//! let a = Tensor::<f64>::zeros(&[3, 4, 10], mem, col);
//! let result = svd(&a, None).unwrap();
//! // result.u:  shape [3, 3, 10]
//! // result.s:  shape [3, 10]
//! // result.vt: shape [3, 4, 10]
//! ```
//!
//! ## Decomposing a 4D tensor along specific legs
//!
//! ```ignore
//! use tenferro_linalg::svd;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! // 4D tensor [2, 3, 4, 5] — want SVD with left=[0,1], right=[2,3]
//! let t = Tensor::<f64>::zeros(&[2, 3, 4, 5], mem, col);
//!
//! // User's responsibility: permute + reshape + contiguous
//! let mat = t.permute(&[0, 1, 2, 3])   // already in order
//!            .reshape(&[6, 20]).unwrap() // m = 2*3 = 6, n = 4*5 = 20
//!            .contiguous(col);
//! let result = svd(&mat, None).unwrap();
//! // Then reshape result.u, result.vt back to desired tensor shape
//! ```
//!
//! ## Reverse-mode AD (stateless rrule)
//!
//! ```ignore
//! use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&a, None).unwrap();
//!
//! // Full cotangent: gradient through U, S, and Vt
//! let cotangent = SvdCotangent {
//!     u: Some(Tensor::ones(&[3, 3], mem, col)),
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: Some(Tensor::ones(&[3, 4], mem, col)),
//! };
//! let grad_a = svd_rrule(&a, &cotangent, None).unwrap();
//! // grad_a has same shape as a: [3, 4]
//!
//! // Partial cotangent: gradient only through singular values (always stable)
//! let cotangent_s_only = SvdCotangent {
//!     u: None,
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: None,
//! };
//! let grad_a2 = svd_rrule(&a, &cotangent_s_only, None).unwrap();
//! ```

use chainrules_core::AdResult;
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_prims::UnaryOp;
use tenferro_tensor::Tensor;

// ============================================================================
// Result types
// ============================================================================

/// SVD result: `A = U * diag(S) * Vt`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `u`: shape `(m, k, *)`
/// - `s`: shape `(k, *)` (singular values, descending order)
/// - `vt`: shape `(k, n, *)`
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::svd;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = svd(&a, None).unwrap();
/// assert_eq!(result.s.ndim(), 1);
/// ```
pub struct SvdResult<T: Scalar> {
    /// Left singular vectors. Shape: `(m, k, *)`.
    pub u: Tensor<T>,
    /// Singular values (descending order). Shape: `(k, *)`.
    pub s: Tensor<T>,
    /// Right singular vectors (conjugate-transposed). Shape: `(k, n, *)`.
    pub vt: Tensor<T>,
}

/// Options for truncated SVD.
///
/// When both `max_rank` and `cutoff` are specified, the more restrictive
/// constraint applies.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SvdOptions;
///
/// // Keep at most 10 singular values above 1e-12
/// let opts = SvdOptions {
///     max_rank: Some(10),
///     cutoff: Some(1e-12),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SvdOptions {
    /// Maximum number of singular values to keep. `None` means no limit.
    pub max_rank: Option<usize>,
    /// Discard singular values below this threshold. `None` means no cutoff.
    pub cutoff: Option<f64>,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self {
            max_rank: None,
            cutoff: None,
        }
    }
}

/// QR decomposition result: `A = Q * R`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `q`: shape `(m, k, *)` (orthonormal columns)
/// - `r`: shape `(k, n, *)` (upper triangular)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::qr;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = qr(&a).unwrap();
/// assert_eq!(result.q.dims(), &[4, 3]);
/// assert_eq!(result.r.dims(), &[3, 3]);
/// ```
pub struct QrResult<T: Scalar> {
    /// Orthonormal factor. Shape: `(m, k, *)`.
    pub q: Tensor<T>,
    /// Upper triangular factor. Shape: `(k, n, *)`.
    pub r: Tensor<T>,
}

/// Pivoting strategy for LU decomposition.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LuPivot;
///
/// let pivot = LuPivot::Partial; // default, uses LAPACK dgetrf
/// let no_pivot = LuPivot::NoPivot; // no pivoting, numerically unstable
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LuPivot {
    /// Partial (row) pivoting (default). Uses LAPACK `?getrf` / cuSOLVER `Xgetrf`.
    /// Numerically stable for general matrices.
    #[default]
    Partial,
    /// No pivoting. Faster but numerically unstable unless the matrix is
    /// known to be well-conditioned (e.g., diagonally dominant). The
    /// permutation field `p` in [`LuResult`] will be `None`.
    NoPivot,
}

/// LU decomposition result: `A = P * L * U`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `p`: permutation indices, shape `(m, *)` — `None` when [`LuPivot::NoPivot`]
/// - `l`: shape `(m, k, *)` (unit lower triangular)
/// - `u`: shape `(k, n, *)` (upper triangular)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
///
/// // With partial pivoting (default)
/// let result = lu(&a, LuPivot::Partial).unwrap();
/// assert!(result.p.is_some());
///
/// // Without pivoting
/// let result = lu(&a, LuPivot::NoPivot).unwrap();
/// assert!(result.p.is_none());
/// ```
pub struct LuResult<T: Scalar> {
    /// Row permutation indices. `Some` for [`LuPivot::Partial`], `None` for
    /// [`LuPivot::NoPivot`]. Shape: `(m, *)`.
    pub p: Option<Vec<usize>>,
    /// Unit lower triangular factor. Shape: `(m, k, *)`.
    pub l: Tensor<T>,
    /// Upper triangular factor. Shape: `(k, n, *)`.
    pub u: Tensor<T>,
}

/// Eigendecomposition result: `A * V = V * diag(values)`.
///
/// Only valid for square matrices (`m == n`).
///
/// - `values`: shape `(n, *)` (eigenvalues)
/// - `vectors`: shape `(n, n, *)` (right eigenvectors as columns)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::eigen;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eigen(&a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
pub struct EigenResult<T: Scalar> {
    /// Eigenvalues. Shape: `(n, *)`.
    pub values: Tensor<T>,
    /// Right eigenvectors (columns). Shape: `(n, n, *)`.
    pub vectors: Tensor<T>,
}

/// Sign-and-log-determinant result: `det(A) = sign * exp(logabsdet)`.
///
/// For an input of shape `(n, n, *)`:
///
/// - `sign`: shape `(*)` (sign of the determinant, ±1 for real, unit complex for complex)
/// - `logabsdet`: shape `(*)` (log of absolute value of determinant)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::slogdet;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = slogdet(&a).unwrap();
/// ```
pub struct SlogdetResult<T: Scalar> {
    /// Sign of determinant. Shape: `(*)`.
    pub sign: Tensor<T>,
    /// Log of absolute value of determinant. Shape: `(*)`.
    pub logabsdet: Tensor<T>,
}

/// Gradient result for `solve_rrule`: cotangents for both `A` and `b`.
pub struct SolveGrad<T: Scalar> {
    /// Cotangent for A. Same shape as `A`.
    pub a: Tensor<T>,
    /// Cotangent for b. Same shape as `b`.
    pub b: Tensor<T>,
}

/// Norm kind for [`norm`].
#[derive(Debug, Clone, Copy)]
pub enum NormKind {
    /// Frobenius norm (matrix) or L2 norm (vector).
    Fro,
    /// Nuclear norm (sum of singular values).
    Nuclear,
    /// Spectral norm (largest singular value) / operator 2-norm.
    Spectral,
    /// L1 norm (max absolute column sum for matrices, sum of abs for vectors).
    L1,
    /// L-infinity norm (max absolute row sum for matrices, max abs for vectors).
    Inf,
    /// General Lp norm for vectors. `p` must be >= 1.
    Lp(f64),
}

// ============================================================================
// Primary decomposition functions
// ============================================================================

/// Compute the SVD of a batched matrix.
///
/// Input shape: `(m, n, *)`. Must be column-major contiguous.
///
/// # Arguments
///
/// * `tensor` — Input tensor of shape `(m, n, *)`
/// * `options` — Optional truncation parameters
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{svd, SvdOptions};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, col);
///
/// // Full SVD
/// let result = svd(&a, None).unwrap();
///
/// // Truncated SVD
/// let opts = SvdOptions { max_rank: Some(2), cutoff: None };
/// let result = svd(&a, Some(&opts)).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn svd<T: Scalar>(_tensor: &Tensor<T>, _options: Option<&SvdOptions>) -> Result<SvdResult<T>> {
    todo!()
}

/// Compute the QR decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`. Must be column-major contiguous.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::qr;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = qr(&a).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn qr<T: Scalar>(_tensor: &Tensor<T>) -> Result<QrResult<T>> {
    todo!()
}

/// Compute the LU decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`. Must be column-major contiguous.
///
/// # Arguments
///
/// * `tensor` — Input tensor of shape `(m, n, *)`
/// * `pivot` — Pivoting strategy: [`LuPivot::Partial`] (default, stable)
///   or [`LuPivot::NoPivot`] (faster, unstable)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
///
/// // Partial pivoting (default)
/// let result = lu(&a, LuPivot::Partial).unwrap();
///
/// // No pivoting
/// let result = lu(&a, LuPivot::NoPivot).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn lu<T: Scalar>(_tensor: &Tensor<T>, _pivot: LuPivot) -> Result<LuResult<T>> {
    todo!()
}

/// Compute the eigendecomposition of a batched square matrix.
///
/// Input shape: `(n, n, *)`. Must be column-major contiguous.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::eigen;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eigen(&a).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions or
/// the first two dimensions are not equal.
pub fn eigen<T: Scalar>(_tensor: &Tensor<T>) -> Result<EigenResult<T>> {
    todo!()
}

/// Solve the least squares problem: `x = argmin ||Ax - b||²`.
///
/// Input shapes: `A` is `(m, n, *)`, `b` is `(m, *)`, with `m >= n`.
/// Both must be column-major contiguous.
///
/// Internally computes `x = R⁻¹ Q† b` via thin QR decomposition of `A`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lstsq;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[10, 5], mem, col);
/// let b = Tensor::<f64>::zeros(&[10], mem, col);
/// let result = lstsq(&a, &b).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `A` has fewer than 2 dimensions, `b` has fewer
/// than 1 dimension, or `m < n`.
pub fn lstsq<T: Scalar>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<LstsqResult<T>> {
    todo!()
}

/// Compute the Cholesky decomposition of a Hermitian positive-definite matrix.
///
/// Input shape: `(n, n, *)`. Returns lower triangular `L` such that `A = L L†`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::cholesky;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let l = cholesky(&a).unwrap();
/// ```
pub fn cholesky<T: Scalar>(_tensor: &Tensor<T>) -> Result<Tensor<T>> {
    todo!()
}

/// Solve a square linear system `A x = b`.
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::solve;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let b = Tensor::<f64>::zeros(&[3], mem, col);
/// let x = solve(&a, &b).unwrap();
/// ```
pub fn solve<T: Scalar>(_a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> {
    todo!()
}

/// Compute the inverse of a square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::inv;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_inv = inv(&a).unwrap();
/// ```
pub fn inv<T: Scalar>(_tensor: &Tensor<T>) -> Result<Tensor<T>> {
    todo!()
}

/// Compute the determinant of a square matrix.
///
/// Input shape: `(n, n, *)`. Returns shape `(*)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::det;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let d = det(&a).unwrap();
/// ```
pub fn det<T: Scalar>(_tensor: &Tensor<T>) -> Result<Tensor<T>> {
    todo!()
}

/// Compute sign and log-absolute-determinant of a square matrix.
///
/// Numerically stable alternative to [`det`]. `det(A) = sign * exp(logabsdet)`.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::slogdet;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = slogdet(&a).unwrap();
/// // det(A) ≈ result.sign * exp(result.logabsdet)
/// ```
pub fn slogdet<T: Scalar>(_tensor: &Tensor<T>) -> Result<SlogdetResult<T>> {
    todo!()
}

/// Compute the eigendecomposition of a general (non-symmetric) square matrix.
///
/// Unlike [`eigen`] (which requires Hermitian/symmetric input and returns
/// real eigenvalues), this function handles general matrices. Eigenvalues
/// may be complex even for real input.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::eig;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eig(&a).unwrap();
/// ```
pub fn eig<T: Scalar>(_tensor: &Tensor<T>) -> Result<EigenResult<T>> {
    todo!()
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// Computed via SVD: `pinv(A) = V diag(1/S) U†`, with singular values
/// below a threshold treated as zero.
///
/// Input shape: `(m, n, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::pinv;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_pinv = pinv(&a, None).unwrap();
/// ```
pub fn pinv<T: Scalar>(_tensor: &Tensor<T>, _rcond: Option<f64>) -> Result<Tensor<T>> {
    todo!()
}

/// Compute the matrix exponential `exp(A)` of a square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::matrix_exp;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let exp_a = matrix_exp(&a).unwrap();
/// ```
pub fn matrix_exp<T: Scalar>(_tensor: &Tensor<T>) -> Result<Tensor<T>> {
    todo!()
}

/// Solve a triangular linear system `A x = b`.
///
/// `A` must be upper or lower triangular (specified by `upper`).
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::solve_triangular;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let b = Tensor::<f64>::zeros(&[3], mem, col);
/// let x = solve_triangular(&a, &b, true).unwrap(); // upper=true
/// ```
pub fn solve_triangular<T: Scalar>(
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _upper: bool,
) -> Result<Tensor<T>> {
    todo!()
}

/// Compute a matrix or vector norm.
///
/// Input shape: `(m, n, *)` for matrix norms, `(n, *)` for vector norms.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{norm, NormKind};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let fro = norm(&a, NormKind::Fro).unwrap();
/// ```
pub fn norm<T: Scalar>(_tensor: &Tensor<T>, _kind: NormKind) -> Result<Tensor<T>> {
    todo!()
}

/// Least squares result: `x = argmin ||Ax - b||²`.
///
/// For an input `A` of shape `(m, n, *)` and `b` of shape `(m, *)` with `m >= n`:
///
/// - `x`: shape `(n, *)` (least-squares solution)
/// - `residual`: shape `(m, *)` (residual `b - Ax`)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lstsq;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[10, 5], mem, col);
/// let b = Tensor::<f64>::zeros(&[10], mem, col);
/// let result = lstsq(&a, &b).unwrap();
/// assert_eq!(result.x.dims(), &[5]);
/// ```
pub struct LstsqResult<T: Scalar> {
    /// Least-squares solution. Shape: `(n, *)`.
    pub x: Tensor<T>,
    /// Residual `b - Ax`. Shape: `(m, *)`.
    pub residual: Tensor<T>,
}

/// Gradient result for `lstsq_rrule`: cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lstsq, lstsq_rrule};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[10, 5], mem, col);
/// let b = Tensor::<f64>::zeros(&[10], mem, col);
/// let dx = Tensor::<f64>::ones(&[5], mem, col);
/// let grad = lstsq_rrule(&a, &b, &dx).unwrap();
/// // grad.a: shape [10, 5], grad.b: shape [10]
/// ```
pub struct LstsqGrad<T: Scalar> {
    /// Cotangent for A. Same shape as `A`.
    pub a: Tensor<T>,
    /// Cotangent for b. Same shape as `b`.
    pub b: Tensor<T>,
}

// ============================================================================
// AD cotangent types
// ============================================================================

/// Cotangent (adjoint) for SVD outputs.
///
/// Each field is `Option` because the user may not need gradients for
/// all outputs (e.g., only `s` for singular value optimization).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::SvdCotangent;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
///
/// // Only need gradient through singular values
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::<f64>::ones(&[3], mem, col)),
///     vt: None,
/// };
/// ```
pub struct SvdCotangent<T: Scalar> {
    /// Cotangent for U. Shape must match `SvdResult::u`.
    pub u: Option<Tensor<T>>,
    /// Cotangent for S. Shape must match `SvdResult::s`.
    pub s: Option<Tensor<T>>,
    /// Cotangent for Vt. Shape must match `SvdResult::vt`.
    pub vt: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for QR outputs.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::QrCotangent;
///
/// let cotangent = QrCotangent::<f64> { q: None, r: None };
/// ```
pub struct QrCotangent<T: Scalar> {
    /// Cotangent for Q. Shape must match `QrResult::q`.
    pub q: Option<Tensor<T>>,
    /// Cotangent for R. Shape must match `QrResult::r`.
    pub r: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for LU outputs.
///
/// Note: the permutation `p` is discrete and has no gradient.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::LuCotangent;
///
/// let cotangent = LuCotangent::<f64> { l: None, u: None };
/// ```
pub struct LuCotangent<T: Scalar> {
    /// Cotangent for L. Shape must match `LuResult::l`.
    pub l: Option<Tensor<T>>,
    /// Cotangent for U. Shape must match `LuResult::u`.
    pub u: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for eigendecomposition outputs.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::EigenCotangent;
///
/// let cotangent = EigenCotangent::<f64> { values: None, vectors: None };
/// ```
pub struct EigenCotangent<T: Scalar> {
    /// Cotangent for eigenvalues. Shape must match `EigenResult::values`.
    pub values: Option<Tensor<T>>,
    /// Cotangent for eigenvectors. Shape must match `EigenResult::vectors`.
    pub vectors: Option<Tensor<T>>,
}

/// Cotangent (adjoint) for slogdet outputs.
///
/// Note: `sign` is piecewise constant and not differentiable.
/// Gradient flows only through `logabsdet`.
pub struct SlogdetCotangent<T: Scalar> {
    /// Cotangent for logabsdet. Shape must match `SlogdetResult::logabsdet`.
    pub logabsdet: Option<Tensor<T>>,
}

// ============================================================================
// AD functions: rrule (reverse-mode, stateless)
// ============================================================================

/// Reverse-mode AD rule for SVD (VJP / pullback).
///
/// Computes the gradient of the input given cotangents for the SVD outputs.
/// Uses batched matrix operations (Mathieu 2019) that broadcast over `*`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
///
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::ones(&[3], mem, col)),
///     vt: None,
/// };
/// let grad_a = svd_rrule(&a, &cotangent, None).unwrap();
/// ```
pub fn svd_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &SvdCotangent<T>,
    _options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>> {
    // SVD reverse-mode AD (Mathieu 2019).
    //
    // Given: A = U · diag(S) · Vt, with cotangents dU, dS, dVt.
    //
    // Algorithm (all operations batched over `*` dims):
    //
    // 1. Forward pass: (U, S, Vt) = svd(A)
    //    → Already computed by the caller; recompute or cache as needed.
    //
    // 2. Build F-matrix: F_ij = 1/(σ_j² - σ_i²) for i≠j, 0 for i=j.
    //    Ops: ElementwiseMul(S, S) → S², broadcast, subtract, Reciprocal,
    //         zero diagonal.
    //    Prims used: ElementwiseMul, ElementwiseUnary(Reciprocal).
    //
    // 3. Compute Ut·dU (k×k batched):
    //    Ops: BatchedGemm(Ut, dU)
    //    Prims used: BatchedGemm.
    //
    // 4. Symmetrize: M = Ut·dU - (Ut·dU)^T via permute.
    //    Ops: permute (zero-copy), alpha/beta subtraction.
    //    Prims used: Permute (metadata only), BatchedGemm with beta=-1.
    //
    // 5. Hadamard product: F ⊙ M
    //    Prims used: ElementwiseMul.
    //
    // 6. Add diagonal dS: F⊙M + diag(dS)
    //    Prims used: AntiTrace (embed 1D → diagonal of 2D).
    //
    // 7. Assemble: dA = U · (F⊙M + diag(dS)) · Vt
    //    Prims used: BatchedGemm (two multiplications).
    //
    // 8. (Full-rank case, m > n) Add projector term:
    //    dA += (I - U·Ut) · dU · diag(1/S) · Vt
    //    Prims used: eye, BatchedGemm, ElementwiseUnary(Reciprocal).

    // Suppress unused import warning in skeleton.
    let _ = UnaryOp::Reciprocal;

    todo!("SVD rrule: implement steps 1-8 using tenferro-prims operations")
}

/// Reverse-mode AD rule for QR (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{qr_rrule, QrCotangent};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[4, 3], mem, col);
/// let cotangent = QrCotangent {
///     q: Some(Tensor::ones(&[4, 3], mem, col)),
///     r: None,
/// };
/// let grad_a = qr_rrule(&a, &cotangent).unwrap();
/// ```
pub fn qr_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &QrCotangent<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for LU (VJP / pullback).
///
/// The `pivot` argument must match the pivoting strategy used in the forward pass.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lu_rrule, LuCotangent, LuPivot};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = LuCotangent {
///     l: Some(Tensor::ones(&[3, 3], mem, col)),
///     u: None,
/// };
/// let grad_a = lu_rrule(&a, &cotangent, LuPivot::Partial).unwrap();
/// ```
pub fn lu_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &LuCotangent<T>,
    _pivot: LuPivot,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for eigendecomposition (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{eigen_rrule, EigenCotangent};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigenCotangent {
///     values: Some(Tensor::ones(&[3], mem, col)),
///     vectors: None,
/// };
/// let grad_a = eigen_rrule(&a, &cotangent).unwrap();
/// ```
pub fn eigen_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for least squares (VJP / pullback).
///
/// Returns cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lstsq_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[10, 5], mem, col);
/// let b = Tensor::<f64>::zeros(&[10], mem, col);
/// let dx = Tensor::<f64>::ones(&[5], mem, col);
/// let grad = lstsq_rrule(&a, &b, &dx).unwrap();
/// // grad.a: cotangent for A, grad.b: cotangent for b
/// ```
pub fn lstsq_rrule<T: Scalar>(
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _cotangent: &Tensor<T>,
) -> AdResult<LstsqGrad<T>> {
    todo!()
}

/// Reverse-mode AD rule for Cholesky (VJP / pullback).
///
/// Given `A = L L†` and cotangent `L̄`, computes `Ā`.
pub fn cholesky_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for linear solve (VJP / pullback).
///
/// Given `Ax = b` and cotangent `x̄`, computes `(Ā, b̄)`.
pub fn solve_rrule<T: Scalar>(
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>> {
    todo!()
}

/// Reverse-mode AD rule for matrix inverse (VJP / pullback).
///
/// `Ā = -A⁻ᴴ · cotangent · A⁻ᴴ`.
pub fn inv_rrule<T: Scalar>(_tensor: &Tensor<T>, _cotangent: &Tensor<T>) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for determinant (VJP / pullback).
///
/// `Ā = det(A) · cotangent · A⁻ᵀ`.
pub fn det_rrule<T: Scalar>(_tensor: &Tensor<T>, _cotangent: &Tensor<T>) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for slogdet (VJP / pullback).
///
/// `Ā = cotangent_logabsdet · A⁻ᵀ`.
pub fn slogdet_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &SlogdetCotangent<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for general eigendecomposition (VJP / pullback).
pub fn eig_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for pseudoinverse (VJP / pullback).
pub fn pinv_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &Tensor<T>,
    _rcond: Option<f64>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for matrix exponential (VJP / pullback).
pub fn matrix_exp_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    todo!()
}

/// Reverse-mode AD rule for norm (VJP / pullback).
pub fn norm_rrule<T: Scalar>(
    _tensor: &Tensor<T>,
    _cotangent: &Tensor<T>,
    _kind: NormKind,
) -> AdResult<Tensor<T>> {
    todo!()
}

// ============================================================================
// AD functions: frule (forward-mode, stateless)
// ============================================================================

/// Forward-mode AD rule for SVD (JVP / pushforward).
///
/// Computes the JVP of all SVD outputs given a tangent for the input.
/// Uses batched matrix operations that broadcast over `*`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::svd_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (result, dresult) = svd_frule(&a, &da, None).unwrap();
/// ```
pub fn svd_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
    _options: Option<&SvdOptions>,
) -> AdResult<(SvdResult<T>, SvdResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for QR (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::qr_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[4, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let (result, dresult) = qr_frule(&a, &da).unwrap();
/// ```
pub fn qr_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(QrResult<T>, QrResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for LU (JVP / pushforward).
///
/// The `pivot` argument must match the pivoting strategy used in the forward pass.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lu_frule, LuPivot};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = lu_frule(&a, &da, LuPivot::Partial).unwrap();
/// ```
pub fn lu_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
    _pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for eigendecomposition (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::eigen_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eigen_frule(&a, &da).unwrap();
/// ```
pub fn eigen_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for least squares (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lstsq_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[10, 5], mem, col);
/// let b = Tensor::<f64>::zeros(&[10], mem, col);
/// let da = Tensor::<f64>::ones(&[10, 5], mem, col);
/// let db = Tensor::<f64>::ones(&[10], mem, col);
/// let (result, dresult) = lstsq_frule(&a, &b, &da, &db).unwrap();
/// ```
pub fn lstsq_frule<T: Scalar>(
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _tangent_a: &Tensor<T>,
    _tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T>, LstsqResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for Cholesky (JVP / pushforward).
pub fn cholesky_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for linear solve (JVP / pushforward).
pub fn solve_frule<T: Scalar>(
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _tangent_a: &Tensor<T>,
    _tangent_b: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
pub fn inv_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for determinant (JVP / pushforward).
pub fn det_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for slogdet (JVP / pushforward).
pub fn slogdet_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T>, SlogdetResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for general eigendecomposition (JVP / pushforward).
pub fn eig_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)> {
    todo!()
}

/// Forward-mode AD rule for pseudoinverse (JVP / pushforward).
pub fn pinv_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
    _rcond: Option<f64>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for matrix exponential (JVP / pushforward).
pub fn matrix_exp_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}

/// Forward-mode AD rule for norm (JVP / pushforward).
pub fn norm_frule<T: Scalar>(
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
    _kind: NormKind,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    todo!()
}
