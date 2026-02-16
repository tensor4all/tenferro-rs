//! Matrix-level linear algebra decompositions with AD rules.
//!
//! This crate provides 2D matrix SVD, QR, LU, and eigendecomposition along
//! with their reverse-mode (rrule/pullback) and forward-mode (frule/pushforward)
//! differentiation rules, following Mathieu (2019) and related literature.
//!
//! The API uses [`dlpack_core::MatrixView`] for inputs and [`dlpack_core::Matrix`]
//! for outputs, enabling GPU-transparent operation. An [`Alloc`](dlpack_core::Alloc)
//! trait object is passed to each function for device-aware memory allocation.
//!
//! Higher-level crates (e.g., `tenferro-linalg`) handle N-dimensional tensor
//! matricization/unmatricization and delegate to this crate for the pure
//! 2D matrix AD math.
//!
//! # Examples
//!
//! ## SVD of a column-major matrix
//!
//! ```ignore
//! use dlpack_core::{Matrix, Alloc};
//! use chainrules_linalg::mat_svd;
//!
//! let data = vec![1.0f64; 12];
//! let a = Matrix::from_vec(data, 3, 4, [1, 3]);
//!
//! let alloc = CpuAlloc; // implements Alloc
//! let result = mat_svd(&a.as_view(), None, &alloc);
//! assert_eq!(result.k, 3); // k = min(m, n)
//! ```
//!
//! ## Reverse-mode AD through SVD
//!
//! ```ignore
//! use dlpack_core::Matrix;
//! use chainrules_linalg::{mat_svd, mat_svd_rrule, MatSvdCotangent};
//!
//! let a = Matrix::from_vec(vec![1.0f64; 12], 3, 4, [1, 3]);
//! let result = mat_svd(&a.as_view(), None, &alloc);
//!
//! // Cotangent: only backprop through singular values
//! let ds = vec![1.0; result.k];
//! let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
//! let grad_a = mat_svd_rrule(&a.as_view(), &cotangent, None, &alloc);
//! assert_eq!(grad_a.nrows(), 3);
//! assert_eq!(grad_a.ncols(), 4);
//! ```

use dlpack_core::{Alloc, Matrix, MatrixView};

// ============================================================================
// Result types (using dlpack_core::Matrix<T>)
// ============================================================================

/// SVD result: `A = U * diag(S) * Vt`.
///
/// All matrices are stored in column-major order.
///
/// - `u`: shape `m × k` (left singular vectors)
/// - `s`: singular values, length `k`
/// - `vt`: shape `k × n` (right singular vectors transposed)
///
/// where `k = min(m, n)` (or truncated rank if options were applied).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_svd;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 3, 4, [1, 3]);
/// let result = mat_svd(&a.as_view(), None, &alloc);
/// assert_eq!(result.u.nrows(), 3);
/// assert_eq!(result.s.len(), result.k);
/// ```
pub struct MatSvdResult<T> {
    /// Left singular vectors (column-major, m × k).
    pub u: Matrix<T>,
    /// Singular values (descending order), length k.
    pub s: Vec<T>,
    /// Right singular vectors transposed (column-major, k × n).
    pub vt: Matrix<T>,
    /// Number of rows of the input matrix.
    pub m: usize,
    /// Number of columns of the input matrix.
    pub n: usize,
    /// Number of singular values kept.
    pub k: usize,
}

/// Options for truncated SVD.
///
/// When both `max_rank` and `cutoff` are specified, the more restrictive
/// constraint applies.
///
/// # Examples
///
/// ```
/// use chainrules_linalg::MatSvdOptions;
///
/// let opts = MatSvdOptions {
///     max_rank: Some(10),
///     cutoff: Some(1e-12),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MatSvdOptions {
    /// Maximum number of singular values to keep. `None` means no limit.
    pub max_rank: Option<usize>,
    /// Discard singular values below this threshold. `None` means no cutoff.
    pub cutoff: Option<f64>,
}

impl Default for MatSvdOptions {
    fn default() -> Self {
        Self {
            max_rank: None,
            cutoff: None,
        }
    }
}

/// QR decomposition result: `A = Q * R`.
///
/// - `q`: shape `m × k` (orthonormal columns)
/// - `r`: shape `k × n` (upper triangular)
///
/// where `k = min(m, n)`.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_qr;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 4, 3, [1, 4]);
/// let result = mat_qr(&a.as_view(), &alloc);
/// assert_eq!(result.q.nrows(), 4);
/// ```
pub struct MatQrResult<T> {
    /// Orthonormal factor Q (column-major, m × k).
    pub q: Matrix<T>,
    /// Upper triangular factor R (column-major, k × n).
    pub r: Matrix<T>,
    /// Number of rows of the input matrix.
    pub m: usize,
    /// Number of columns of the input matrix.
    pub n: usize,
    /// Rank k = min(m, n).
    pub k: usize,
}

/// LU decomposition result: `A = P * L * U` (partial pivoting).
///
/// - `p`: row permutation vector of length `m`
/// - `l`: shape `m × k` (unit lower triangular)
/// - `u`: shape `k × n` (upper triangular)
///
/// where `k = min(m, n)`.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_lu;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let result = mat_lu(&a.as_view(), &alloc);
/// assert_eq!(result.p.len(), 3);
/// ```
pub struct MatLuResult<T> {
    /// Row permutation vector (partial pivoting), length m.
    pub p: Vec<usize>,
    /// Unit lower triangular factor L (column-major, m × k).
    pub l: Matrix<T>,
    /// Upper triangular factor U (column-major, k × n).
    pub u: Matrix<T>,
    /// Number of rows of the input matrix.
    pub m: usize,
    /// Number of columns of the input matrix.
    pub n: usize,
    /// Rank k = min(m, n).
    pub k: usize,
}

/// Eigendecomposition result: `A * V = V * diag(values)`.
///
/// Only valid for square matrices.
///
/// - `values`: eigenvalues, length `n`
/// - `vectors`: right eigenvectors (column-major, n × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_eigen;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let result = mat_eigen(&a.as_view(), &alloc);
/// assert_eq!(result.values.len(), 3);
/// ```
pub struct MatEigenResult<T> {
    /// Eigenvalues, length n.
    pub values: Vec<T>,
    /// Right eigenvectors (column-major, n × n).
    pub vectors: Matrix<T>,
    /// Matrix dimension n.
    pub n: usize,
}

// ============================================================================
// Cotangent types
// ============================================================================

/// Cotangent (adjoint) for SVD outputs.
///
/// Each field is `Option` because the user may not need gradients for
/// all outputs (e.g., only `s` for singular value optimization).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::MatSvdCotangent;
///
/// let ds = vec![1.0f64; 3];
/// let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
/// ```
pub struct MatSvdCotangent<'a, T> {
    /// Cotangent for U (m × k). `None` if not needed.
    pub u: Option<MatrixView<'a, T>>,
    /// Cotangent for S (length k). `None` if not needed.
    pub s: Option<&'a [T]>,
    /// Cotangent for Vt (k × n). `None` if not needed.
    pub vt: Option<MatrixView<'a, T>>,
}

/// Cotangent (adjoint) for QR outputs.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::MatQrCotangent;
///
/// let cotangent = MatQrCotangent::<f64> { q: None, r: None };
/// ```
pub struct MatQrCotangent<'a, T> {
    /// Cotangent for Q (m × k). `None` if not needed.
    pub q: Option<MatrixView<'a, T>>,
    /// Cotangent for R (k × n). `None` if not needed.
    pub r: Option<MatrixView<'a, T>>,
}

/// Cotangent (adjoint) for LU outputs.
///
/// The permutation `p` is discrete and has no gradient.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::MatLuCotangent;
///
/// let cotangent = MatLuCotangent::<f64> { l: None, u: None };
/// ```
pub struct MatLuCotangent<'a, T> {
    /// Cotangent for L (m × k). `None` if not needed.
    pub l: Option<MatrixView<'a, T>>,
    /// Cotangent for U (k × n). `None` if not needed.
    pub u: Option<MatrixView<'a, T>>,
}

/// Cotangent (adjoint) for eigendecomposition outputs.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::MatEigenCotangent;
///
/// let cotangent = MatEigenCotangent::<f64> { values: None, vectors: None };
/// ```
pub struct MatEigenCotangent<'a, T> {
    /// Cotangent for eigenvalues (length n). `None` if not needed.
    pub values: Option<&'a [T]>,
    /// Cotangent for eigenvectors (n × n). `None` if not needed.
    pub vectors: Option<MatrixView<'a, T>>,
}

// ============================================================================
// SVD functions
// ============================================================================

/// Compute the SVD of a 2D matrix.
///
/// Decomposes `A` into `U * diag(S) * Vt`.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `options` — Optional truncation parameters
/// * `alloc` — Device-aware allocator for output matrices
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_svd;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 3, 4, [1, 3]);
/// let result = mat_svd(&a.as_view(), None, &alloc);
/// ```
pub fn mat_svd<T: Clone>(
    _a: &MatrixView<'_, T>,
    _options: Option<&MatSvdOptions>,
    _alloc: &dyn Alloc,
) -> MatSvdResult<T> {
    todo!()
}

/// Reverse-mode AD rule for SVD (pullback).
///
/// Given the input matrix and cotangents for the SVD outputs, computes
/// the gradient of the input matrix. Implements Mathieu (2019).
///
/// # Arguments
///
/// * `a` — Input matrix (m × n) used in the forward pass
/// * `cotangent` — Cotangents for SVD outputs (U, S, Vt)
/// * `options` — Same truncation options used in the forward pass
/// * `alloc` — Device-aware allocator for the output gradient matrix
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{mat_svd_rrule, MatSvdCotangent};
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 3, 4, [1, 3]);
/// let ds = vec![1.0; 3];
/// let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
/// let grad = mat_svd_rrule(&a.as_view(), &cotangent, None, &alloc);
/// assert_eq!(grad.nrows(), 3);
/// ```
pub fn mat_svd_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatSvdCotangent<'_, T>,
    _options: Option<&MatSvdOptions>,
    _alloc: &dyn Alloc,
) -> Matrix<T> {
    todo!()
}

/// Forward-mode AD rule for SVD (pushforward / JVP).
///
/// Returns a pair of (primal result, tangent result).
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `tangent` — Tangent of the input matrix (m × n)
/// * `options` — Optional truncation parameters
/// * `alloc` — Device-aware allocator for output matrices
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_svd_frule;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 3, 4, [1, 3]);
/// let da = Matrix::from_vec(vec![0.1f64; 12], 3, 4, [1, 3]);
/// let (result, dresult) = mat_svd_frule(&a.as_view(), &da.as_view(), None, &alloc);
/// ```
pub fn mat_svd_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
    _options: Option<&MatSvdOptions>,
    _alloc: &dyn Alloc,
) -> (MatSvdResult<T>, MatSvdResult<T>) {
    todo!()
}

// ============================================================================
// QR functions
// ============================================================================

/// Compute the thin QR decomposition of a 2D matrix.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_qr;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 4, 3, [1, 4]);
/// let result = mat_qr(&a.as_view(), &alloc);
/// ```
pub fn mat_qr<T: Clone>(_a: &MatrixView<'_, T>, _alloc: &dyn Alloc) -> MatQrResult<T> {
    todo!()
}

/// Reverse-mode AD rule for QR (pullback).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{mat_qr_rrule, MatQrCotangent};
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 4, 3, [1, 4]);
/// let cotangent = MatQrCotangent { q: None, r: None };
/// let grad = mat_qr_rrule(&a.as_view(), &cotangent, &alloc);
/// ```
pub fn mat_qr_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatQrCotangent<'_, T>,
    _alloc: &dyn Alloc,
) -> Matrix<T> {
    todo!()
}

/// Forward-mode AD rule for QR (pushforward / JVP).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_qr_frule;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 12], 4, 3, [1, 4]);
/// let da = Matrix::from_vec(vec![0.1f64; 12], 4, 3, [1, 4]);
/// let (result, dresult) = mat_qr_frule(&a.as_view(), &da.as_view(), &alloc);
/// ```
pub fn mat_qr_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
    _alloc: &dyn Alloc,
) -> (MatQrResult<T>, MatQrResult<T>) {
    todo!()
}

// ============================================================================
// LU functions
// ============================================================================

/// Compute the LU decomposition with partial pivoting.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_lu;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let result = mat_lu(&a.as_view(), &alloc);
/// ```
pub fn mat_lu<T: Clone>(_a: &MatrixView<'_, T>, _alloc: &dyn Alloc) -> MatLuResult<T> {
    todo!()
}

/// Reverse-mode AD rule for LU (pullback).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{mat_lu_rrule, MatLuCotangent};
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let cotangent = MatLuCotangent { l: None, u: None };
/// let grad = mat_lu_rrule(&a.as_view(), &cotangent, &alloc);
/// ```
pub fn mat_lu_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatLuCotangent<'_, T>,
    _alloc: &dyn Alloc,
) -> Matrix<T> {
    todo!()
}

/// Forward-mode AD rule for LU (pushforward / JVP).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_lu_frule;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let da = Matrix::from_vec(vec![0.1f64; 9], 3, 3, [1, 3]);
/// let (result, dresult) = mat_lu_frule(&a.as_view(), &da.as_view(), &alloc);
/// ```
pub fn mat_lu_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
    _alloc: &dyn Alloc,
) -> (MatLuResult<T>, MatLuResult<T>) {
    todo!()
}

// ============================================================================
// Eigen functions
// ============================================================================

/// Compute the eigendecomposition of a square matrix.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_eigen;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let result = mat_eigen(&a.as_view(), &alloc);
/// ```
pub fn mat_eigen<T: Clone>(_a: &MatrixView<'_, T>, _alloc: &dyn Alloc) -> MatEigenResult<T> {
    todo!()
}

/// Reverse-mode AD rule for eigendecomposition (pullback).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{mat_eigen_rrule, MatEigenCotangent};
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let cotangent = MatEigenCotangent { values: None, vectors: None };
/// let grad = mat_eigen_rrule(&a.as_view(), &cotangent, &alloc);
/// ```
pub fn mat_eigen_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatEigenCotangent<'_, T>,
    _alloc: &dyn Alloc,
) -> Matrix<T> {
    todo!()
}

/// Forward-mode AD rule for eigendecomposition (pushforward / JVP).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::mat_eigen_frule;
/// use dlpack_core::Matrix;
///
/// let a = Matrix::from_vec(vec![1.0f64; 9], 3, 3, [1, 3]);
/// let da = Matrix::from_vec(vec![0.1f64; 9], 3, 3, [1, 3]);
/// let (result, dresult) = mat_eigen_frule(&a.as_view(), &da.as_view(), &alloc);
/// ```
pub fn mat_eigen_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
    _alloc: &dyn Alloc,
) -> (MatEigenResult<T>, MatEigenResult<T>) {
    todo!()
}
