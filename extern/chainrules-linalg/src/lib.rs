//! Matrix-level linear algebra decompositions with AD rules.
//!
//! This crate provides 2D matrix SVD, QR, LU, and eigendecomposition along
//! with their reverse-mode (rrule/pullback) and forward-mode (frule/pushforward)
//! differentiation rules, following Mathieu (2019) and related literature.
//!
//! The API uses [`MatrixView`] — a zero-dependency 2D view over `&[T]` with
//! shape and strides — so the crate can be used standalone without any tensor
//! library. Higher-level crates (e.g., `tenferro-linalg`) handle N-dimensional
//! tensor matricization/unmatricization and delegate to this crate for the
//! pure matrix AD math.
//!
//! # Examples
//!
//! ## SVD of a column-major matrix
//!
//! ```ignore
//! use chainrules_linalg::{MatrixView, mat_svd, MatSvdOptions};
//!
//! // 3×4 column-major matrix (row_stride=1, col_stride=3)
//! let data = vec![0.0_f64; 12];
//! let a = MatrixView::new(&data, 3, 4, 1, 3);
//!
//! let result = mat_svd(&a, None);
//! assert_eq!(result.k, 3); // k = min(m, n)
//! ```
//!
//! ## Reverse-mode AD through SVD
//!
//! ```ignore
//! use chainrules_linalg::{MatrixView, MatSvdCotangent, mat_svd, mat_svd_rrule};
//!
//! let data = vec![1.0_f64; 12];
//! let a = MatrixView::new(&data, 3, 4, 1, 3);
//!
//! let result = mat_svd(&a, None);
//!
//! // Cotangent: only backprop through singular values
//! let ds = vec![1.0; result.k];
//! let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
//! let grad_a = mat_svd_rrule(&a, &cotangent, None);
//! assert_eq!(grad_a.len(), 3 * 4); // m × n
//! ```

// ============================================================================
// Matrix view types
// ============================================================================

/// Immutable 2D matrix view over a contiguous slice.
///
/// Provides a zero-dependency way to pass 2D matrix data with arbitrary
/// row and column strides. This allows both row-major and column-major
/// layouts, as well as views into sub-matrices.
///
/// # Layout
///
/// Element `(i, j)` is at index `i * row_stride + j * col_stride` in `data`.
///
/// - Column-major (Fortran order): `row_stride = 1`, `col_stride = nrows`
/// - Row-major (C order): `row_stride = ncols`, `col_stride = 1`
///
/// # Examples
///
/// ```
/// use chainrules_linalg::MatrixView;
///
/// // 2×3 column-major matrix
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let m = MatrixView::new(&data, 2, 3, 1, 2);
/// assert_eq!(m.nrows, 2);
/// assert_eq!(m.ncols, 3);
/// ```
pub struct MatrixView<'a, T> {
    /// Underlying data slice.
    pub data: &'a [T],
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Stride between consecutive rows.
    pub row_stride: usize,
    /// Stride between consecutive columns.
    pub col_stride: usize,
}

impl<'a, T> MatrixView<'a, T> {
    /// Creates a new matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules_linalg::MatrixView;
    ///
    /// let data = vec![0.0_f64; 6];
    /// let m = MatrixView::new(&data, 2, 3, 1, 2);
    /// assert_eq!(m.nrows, 2);
    /// ```
    pub fn new(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            data,
            nrows,
            ncols,
            row_stride,
            col_stride,
        }
    }
}

/// Mutable 2D matrix view over a contiguous slice.
///
/// Same layout semantics as [`MatrixView`] but allows mutation.
///
/// # Examples
///
/// ```
/// use chainrules_linalg::MatrixViewMut;
///
/// let mut data = vec![0.0_f64; 6];
/// let m = MatrixViewMut::new(&mut data, 2, 3, 1, 2);
/// assert_eq!(m.nrows, 2);
/// ```
pub struct MatrixViewMut<'a, T> {
    /// Underlying mutable data slice.
    pub data: &'a mut [T],
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Stride between consecutive rows.
    pub row_stride: usize,
    /// Stride between consecutive columns.
    pub col_stride: usize,
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// Creates a new mutable matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules_linalg::MatrixViewMut;
    ///
    /// let mut data = vec![0.0_f64; 6];
    /// let m = MatrixViewMut::new(&mut data, 2, 3, 1, 2);
    /// assert_eq!(m.nrows, 2);
    /// ```
    pub fn new(
        data: &'a mut [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            data,
            nrows,
            ncols,
            row_stride,
            col_stride,
        }
    }
}

// ============================================================================
// Result types (owned, using Vec<T>)
// ============================================================================

/// SVD result: `A = U * diag(S) * Vt`.
///
/// All data is stored in column-major order.
///
/// - `u`: shape `m × k`, stored as `Vec<T>` of length `m * k`
/// - `s`: singular values, `Vec<T>` of length `k`
/// - `vt`: shape `k × n`, stored as `Vec<T>` of length `k * n`
///
/// where `k = min(m, n)` (or truncated rank if options were applied).
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_svd};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 3, 4, 1, 3);
/// let result = mat_svd(&a, None);
/// assert_eq!(result.u.len(), 3 * result.k);
/// assert_eq!(result.s.len(), result.k);
/// assert_eq!(result.vt.len(), result.k * 4);
/// ```
pub struct MatSvdResult<T> {
    /// Left singular vectors (column-major, m × k).
    pub u: Vec<T>,
    /// Singular values (descending order), length k.
    pub s: Vec<T>,
    /// Right singular vectors transposed (column-major, k × n).
    pub vt: Vec<T>,
    /// Number of rows of the input matrix.
    pub m: usize,
    /// Number of columns of the input matrix.
    pub n: usize,
    /// Number of singular values kept (min(m, n) or truncated).
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
/// All data is stored in column-major order.
///
/// - `q`: shape `m × k`, stored as `Vec<T>` of length `m * k`
/// - `r`: shape `k × n`, stored as `Vec<T>` of length `k * n`
///
/// where `k = min(m, n)`.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_qr};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 4, 3, 1, 4);
/// let result = mat_qr(&a);
/// assert_eq!(result.q.len(), 4 * result.k);
/// assert_eq!(result.r.len(), result.k * 3);
/// ```
pub struct MatQrResult<T> {
    /// Orthonormal factor Q (column-major, m × k).
    pub q: Vec<T>,
    /// Upper triangular factor R (column-major, k × n).
    pub r: Vec<T>,
    /// Number of rows of the input matrix.
    pub m: usize,
    /// Number of columns of the input matrix.
    pub n: usize,
    /// Rank k = min(m, n).
    pub k: usize,
}

/// LU decomposition result: `A = P * L * U` (partial pivoting).
///
/// All data is stored in column-major order.
///
/// - `p`: row permutation vector of length `m`
/// - `l`: shape `m × k`, stored as `Vec<T>` of length `m * k` (unit lower triangular)
/// - `u`: shape `k × n`, stored as `Vec<T>` of length `k * n` (upper triangular)
///
/// where `k = min(m, n)`.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_lu};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let result = mat_lu(&a);
/// assert_eq!(result.p.len(), 3);
/// assert_eq!(result.l.len(), 3 * result.k);
/// assert_eq!(result.u.len(), result.k * 3);
/// ```
pub struct MatLuResult<T> {
    /// Row permutation vector (partial pivoting), length m.
    pub p: Vec<usize>,
    /// Unit lower triangular factor L (column-major, m × k).
    pub l: Vec<T>,
    /// Upper triangular factor U (column-major, k × n).
    pub u: Vec<T>,
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
/// - `values`: eigenvalues, `Vec<T>` of length `n`
/// - `vectors`: right eigenvectors (column-major, n × n), `Vec<T>` of length `n * n`
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_eigen};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let result = mat_eigen(&a);
/// assert_eq!(result.values.len(), 3);
/// assert_eq!(result.vectors.len(), 9);
/// ```
pub struct MatEigenResult<T> {
    /// Eigenvalues, length n.
    pub values: Vec<T>,
    /// Right eigenvectors (column-major, n × n).
    pub vectors: Vec<T>,
    /// Matrix dimension n.
    pub n: usize,
}

// ============================================================================
// Cotangent types (for AD rules)
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
/// // Only backpropagate through singular values
/// let ds = vec![1.0_f64; 3];
/// let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
/// ```
pub struct MatSvdCotangent<'a, T> {
    /// Cotangent for U (column-major, m × k). `None` if not needed.
    pub u: Option<MatrixView<'a, T>>,
    /// Cotangent for S (length k). `None` if not needed.
    pub s: Option<&'a [T]>,
    /// Cotangent for Vt (column-major, k × n). `None` if not needed.
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
    /// Cotangent for Q (column-major, m × k). `None` if not needed.
    pub q: Option<MatrixView<'a, T>>,
    /// Cotangent for R (column-major, k × n). `None` if not needed.
    pub r: Option<MatrixView<'a, T>>,
}

/// Cotangent (adjoint) for LU outputs.
///
/// Note: the permutation `p` is discrete and has no gradient.
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::MatLuCotangent;
///
/// let cotangent = MatLuCotangent::<f64> { l: None, u: None };
/// ```
pub struct MatLuCotangent<'a, T> {
    /// Cotangent for L (column-major, m × k). `None` if not needed.
    pub l: Option<MatrixView<'a, T>>,
    /// Cotangent for U (column-major, k × n). `None` if not needed.
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
    /// Cotangent for eigenvectors (column-major, n × n). `None` if not needed.
    pub vectors: Option<MatrixView<'a, T>>,
}

// ============================================================================
// SVD functions
// ============================================================================

/// Compute the SVD of a 2D matrix.
///
/// Decomposes `A` into `U * diag(S) * Vt` where `U` and `Vt` are
/// (semi-)orthogonal and `S` contains non-negative singular values
/// in descending order.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `options` — Optional truncation parameters
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_svd, MatSvdOptions};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 3, 4, 1, 3);
///
/// // Full SVD
/// let result = mat_svd(&a, None);
///
/// // Truncated SVD
/// let opts = MatSvdOptions { max_rank: Some(2), cutoff: None };
/// let result = mat_svd(&a, Some(&opts));
/// ```
pub fn mat_svd<T: Clone>(
    _a: &MatrixView<'_, T>,
    _options: Option<&MatSvdOptions>,
) -> MatSvdResult<T> {
    todo!()
}

/// Reverse-mode AD rule for SVD (pullback).
///
/// Given the input matrix and cotangents for the SVD outputs, computes
/// the gradient of the input matrix. Implements the formulas from
/// Mathieu (2019).
///
/// Returns the gradient as a column-major `Vec<T>` of length `m * n`.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n) used in the forward pass
/// * `cotangent` — Cotangents for SVD outputs (U, S, Vt)
/// * `options` — Same truncation options used in the forward pass
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, MatSvdCotangent, mat_svd_rrule};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 3, 4, 1, 3);
/// let ds = vec![1.0; 3];
/// let cotangent = MatSvdCotangent { u: None, s: Some(&ds), vt: None };
/// let grad = mat_svd_rrule(&a, &cotangent, None);
/// assert_eq!(grad.len(), 12);
/// ```
pub fn mat_svd_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatSvdCotangent<'_, T>,
    _options: Option<&MatSvdOptions>,
) -> Vec<T> {
    todo!()
}

/// Forward-mode AD rule for SVD (pushforward / JVP).
///
/// Given the input matrix and its tangent, computes both the SVD result
/// and the tangents of all SVD outputs (dU, dS, dVt).
///
/// Returns a pair of (primal result, tangent result).
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `tangent` — Tangent of the input matrix (m × n)
/// * `options` — Optional truncation parameters
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_svd_frule};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 3, 4, 1, 3);
/// let da = MatrixView::new(&data, 3, 4, 1, 3);
/// let (result, dresult) = mat_svd_frule(&a, &da, None);
/// ```
pub fn mat_svd_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
    _options: Option<&MatSvdOptions>,
) -> (MatSvdResult<T>, MatSvdResult<T>) {
    todo!()
}

// ============================================================================
// QR functions
// ============================================================================

/// Compute the thin QR decomposition of a 2D matrix.
///
/// Decomposes `A` into `Q * R` where `Q` has orthonormal columns and
/// `R` is upper triangular.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_qr};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 4, 3, 1, 4);
/// let result = mat_qr(&a);
/// assert_eq!(result.k, 3); // min(4, 3)
/// ```
pub fn mat_qr<T: Clone>(_a: &MatrixView<'_, T>) -> MatQrResult<T> {
    todo!()
}

/// Reverse-mode AD rule for QR (pullback).
///
/// Returns the gradient of the input matrix as a column-major `Vec<T>`.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n) used in the forward pass
/// * `cotangent` — Cotangents for QR outputs (Q, R)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, MatQrCotangent, mat_qr_rrule};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 4, 3, 1, 4);
/// let cotangent = MatQrCotangent { q: None, r: None };
/// let grad = mat_qr_rrule(&a, &cotangent);
/// assert_eq!(grad.len(), 12);
/// ```
pub fn mat_qr_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatQrCotangent<'_, T>,
) -> Vec<T> {
    todo!()
}

/// Forward-mode AD rule for QR (pushforward / JVP).
///
/// Returns a pair of (primal result, tangent result).
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `tangent` — Tangent of the input matrix (m × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_qr_frule};
///
/// let data = vec![1.0_f64; 12];
/// let a = MatrixView::new(&data, 4, 3, 1, 4);
/// let da = MatrixView::new(&data, 4, 3, 1, 4);
/// let (result, dresult) = mat_qr_frule(&a, &da);
/// ```
pub fn mat_qr_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
) -> (MatQrResult<T>, MatQrResult<T>) {
    todo!()
}

// ============================================================================
// LU functions
// ============================================================================

/// Compute the LU decomposition of a 2D matrix with partial pivoting.
///
/// Decomposes `A` into `P * L * U` where `P` is a permutation matrix,
/// `L` is unit lower triangular, and `U` is upper triangular.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_lu};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let result = mat_lu(&a);
/// assert_eq!(result.p.len(), 3);
/// ```
pub fn mat_lu<T: Clone>(_a: &MatrixView<'_, T>) -> MatLuResult<T> {
    todo!()
}

/// Reverse-mode AD rule for LU (pullback).
///
/// Returns the gradient of the input matrix as a column-major `Vec<T>`.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n) used in the forward pass
/// * `cotangent` — Cotangents for LU outputs (L, U)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, MatLuCotangent, mat_lu_rrule};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let cotangent = MatLuCotangent { l: None, u: None };
/// let grad = mat_lu_rrule(&a, &cotangent);
/// assert_eq!(grad.len(), 9);
/// ```
pub fn mat_lu_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatLuCotangent<'_, T>,
) -> Vec<T> {
    todo!()
}

/// Forward-mode AD rule for LU (pushforward / JVP).
///
/// Returns a pair of (primal result, tangent result).
///
/// # Arguments
///
/// * `a` — Input matrix (m × n)
/// * `tangent` — Tangent of the input matrix (m × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_lu_frule};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let da = MatrixView::new(&data, 3, 3, 1, 3);
/// let (result, dresult) = mat_lu_frule(&a, &da);
/// ```
pub fn mat_lu_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
) -> (MatLuResult<T>, MatLuResult<T>) {
    todo!()
}

// ============================================================================
// Eigen functions
// ============================================================================

/// Compute the eigendecomposition of a square 2D matrix.
///
/// Decomposes `A` into `V * diag(values) * V^{-1}` where `values`
/// are the eigenvalues and `V` contains the right eigenvectors.
///
/// # Arguments
///
/// * `a` — Input square matrix (n × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_eigen};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let result = mat_eigen(&a);
/// assert_eq!(result.values.len(), 3);
/// assert_eq!(result.vectors.len(), 9);
/// ```
pub fn mat_eigen<T: Clone>(_a: &MatrixView<'_, T>) -> MatEigenResult<T> {
    todo!()
}

/// Reverse-mode AD rule for eigendecomposition (pullback).
///
/// Returns the gradient of the input matrix as a column-major `Vec<T>`.
///
/// # Arguments
///
/// * `a` — Input square matrix (n × n) used in the forward pass
/// * `cotangent` — Cotangents for eigen outputs (values, vectors)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, MatEigenCotangent, mat_eigen_rrule};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let cotangent = MatEigenCotangent { values: None, vectors: None };
/// let grad = mat_eigen_rrule(&a, &cotangent);
/// assert_eq!(grad.len(), 9);
/// ```
pub fn mat_eigen_rrule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _cotangent: &MatEigenCotangent<'_, T>,
) -> Vec<T> {
    todo!()
}

/// Forward-mode AD rule for eigendecomposition (pushforward / JVP).
///
/// Returns a pair of (primal result, tangent result).
///
/// # Arguments
///
/// * `a` — Input square matrix (n × n)
/// * `tangent` — Tangent of the input matrix (n × n)
///
/// # Examples
///
/// ```ignore
/// use chainrules_linalg::{MatrixView, mat_eigen_frule};
///
/// let data = vec![1.0_f64; 9];
/// let a = MatrixView::new(&data, 3, 3, 1, 3);
/// let da = MatrixView::new(&data, 3, 3, 1, 3);
/// let (result, dresult) = mat_eigen_frule(&a, &da);
/// ```
pub fn mat_eigen_frule<T: Clone>(
    _a: &MatrixView<'_, T>,
    _tangent: &MatrixView<'_, T>,
) -> (MatEigenResult<T>, MatEigenResult<T>) {
    todo!()
}
