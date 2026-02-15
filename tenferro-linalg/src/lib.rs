//! Tensor-level linear algebra decompositions.
//!
//! This crate provides SVD, QR, LU, and eigendecomposition for
//! N-dimensional dense tensors. The user specifies which dimensions
//! form the "row" (left) and "column" (right) sides of a matrix.
//! Internally, the tensor is permuted and reshaped into a 2D matrix,
//! the decomposition is applied via an external backend (faer for CPU,
//! cuSOLVER for GPU), and the results are reshaped back to tensor form.
//!
//! This follows the **matricize -> decompose -> unmatricize** pattern
//! from TensorAlgebra.jl, using numeric dimension indices only.
//!
//! # Dimension specification
//!
//! All decomposition functions take `left` and `right` dimension index
//! slices. These must be disjoint and their union must cover `0..ndim()`.
//! The tensor is permuted so that `left` dims come first, then `right`
//! dims, and reshaped to `[m, n]` where `m = product(left dim sizes)`
//! and `n = product(right dim sizes)`.
//!
//! # Examples
//!
//! ## SVD of a 4D tensor
//!
//! ```ignore
//! use tenferro_linalg::{svd, SvdOptions};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! // 4D tensor with shape [2, 3, 4, 5]
//! let t = Tensor::<f64>::zeros(&[2, 3, 4, 5], mem, col);
//!
//! // Group dims 0,1 as rows (m=6) and dims 2,3 as cols (n=20)
//! let result = svd(&t, &[0, 1], &[2, 3], None).unwrap();
//! // result.u:  shape [2, 3, 6]  (left_dims... × k)
//! // result.s:  shape [6]        (singular values)
//! // result.vt: shape [6, 4, 5]  (k × right_dims...)
//!
//! // Truncated SVD: keep at most 3 singular values
//! let opts = SvdOptions { max_rank: Some(3), cutoff: None };
//! let result = svd(&t, &[0, 1], &[2, 3], Some(&opts)).unwrap();
//! // result.u:  shape [2, 3, 3]
//! // result.s:  shape [3]
//! // result.vt: shape [3, 4, 5]
//! ```
//!
//! ## QR of a matrix
//!
//! ```ignore
//! use tenferro_linalg::qr;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[4, 3], mem, col);
//!
//! let result = qr(&a, &[0], &[1]).unwrap();
//! // result.q: shape [4, 3]  (m × k)
//! // result.r: shape [3, 3]  (k × n)
//! ```
//!
//! ## Reverse-mode AD through SVD
//!
//! ```ignore
//! use chainrules::Tape;
//! use tenferro_linalg::tracked_svd;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//!
//! let tape = Tape::<Tensor<f64>>::new();
//! let a = tape.leaf(Tensor::zeros(&[3, 4], mem, col));
//! let result = tracked_svd(&a, &[0], &[1], None).unwrap();
//! // Use result.s to form a scalar loss, then pullback...
//! ```

use chainrules::{AdResult, Differentiable, DualTensor, TrackedTensor};
use strided_traits::ScalarBase;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

// ============================================================================
// Result types
// ============================================================================

/// SVD result: `A = U * diag(S) * Vt`.
///
/// For a tensor with left dimensions of total size `m` and right dimensions
/// of total size `n`, with `k = min(m, n)` (or truncated rank):
///
/// - `u`: shape `left_dims... × k`
/// - `s`: shape `[k]` (singular values, always real)
/// - `vt`: shape `k × right_dims...`
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
/// let result = svd(&a, &[0], &[1], None).unwrap();
/// assert_eq!(result.s.ndim(), 1);
/// ```
pub struct SvdResult<T: ScalarBase> {
    /// Left singular vectors. Shape: `left_dims... × k`.
    pub u: Tensor<T>,
    /// Singular values (descending order). Shape: `[k]`.
    pub s: Tensor<T>,
    /// Right singular vectors (conjugate-transposed). Shape: `k × right_dims...`.
    pub vt: Tensor<T>,
}

/// Options for truncated SVD.
///
/// When both `max_rank` and `cutoff` are specified, the more restrictive
/// constraint applies (i.e., the result has at most `max_rank` singular
/// values, all of which are above `cutoff`).
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
/// For a tensor with left dimensions of total size `m` and right dimensions
/// of total size `n`, with `k = min(m, n)`:
///
/// - `q`: shape `left_dims... × k` (orthonormal columns)
/// - `r`: shape `k × right_dims...` (upper triangular)
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
/// let result = qr(&a, &[0], &[1]).unwrap();
/// assert_eq!(result.q.dims(), &[4, 3]);
/// assert_eq!(result.r.dims(), &[3, 3]);
/// ```
pub struct QrResult<T: ScalarBase> {
    /// Orthonormal factor. Shape: `left_dims... × k`.
    pub q: Tensor<T>,
    /// Upper triangular factor. Shape: `k × right_dims...`.
    pub r: Tensor<T>,
}

/// LU decomposition result: `A = P * L * U` (partial pivoting).
///
/// For a tensor with left dimensions of total size `m` and right dimensions
/// of total size `n`, with `k = min(m, n)`:
///
/// - `p`: row permutation vector of length `m`
/// - `l`: shape `left_dims... × k` (unit lower triangular)
/// - `u`: shape `k × right_dims...` (upper triangular)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lu;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = lu(&a, &[0], &[1]).unwrap();
/// assert_eq!(result.p.len(), 3);
/// ```
pub struct LuResult<T: ScalarBase> {
    /// Row permutation vector (partial pivoting). Length: `m`.
    pub p: Vec<usize>,
    /// Unit lower triangular factor. Shape: `left_dims... × k`.
    pub l: Tensor<T>,
    /// Upper triangular factor. Shape: `k × right_dims...`.
    pub u: Tensor<T>,
}

/// Eigendecomposition result: `A * V = V * diag(values)`.
///
/// Only valid for square matrices (left dims product == right dims product).
///
/// - `values`: shape `[n]` (eigenvalues)
/// - `vectors`: shape `left_dims... × n` (right eigenvectors as columns)
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
/// let result = eigen(&a, &[0], &[1]).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
pub struct EigenResult<T: ScalarBase> {
    /// Eigenvalues. Shape: `[n]`.
    pub values: Tensor<T>,
    /// Right eigenvectors (columns). Shape: `left_dims... × n`.
    pub vectors: Tensor<T>,
}

// ============================================================================
// Primary decomposition functions
// ============================================================================

/// Compute the SVD of a tensor.
///
/// Matricizes the tensor according to `left`/`right` dimension indices,
/// computes the SVD of the resulting matrix, and reshapes the factors
/// back to tensor form.
///
/// # Arguments
///
/// * `tensor` — Input tensor
/// * `left` — Dimension indices forming the row space
/// * `right` — Dimension indices forming the column space
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
/// let result = svd(&a, &[0], &[1], None).unwrap();
///
/// // Truncated SVD
/// let opts = SvdOptions { max_rank: Some(2), cutoff: None };
/// let result = svd(&a, &[0], &[1], Some(&opts)).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `left` and `right` do not form a valid partition
/// of `0..tensor.ndim()`.
pub fn svd<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _options: Option<&SvdOptions>,
) -> Result<SvdResult<T>> {
    todo!()
}

/// Compute the QR decomposition of a tensor.
///
/// Matricizes the tensor according to `left`/`right` dimension indices,
/// computes the thin QR of the resulting matrix, and reshapes the factors
/// back to tensor form.
///
/// # Arguments
///
/// * `tensor` — Input tensor
/// * `left` — Dimension indices forming the row space
/// * `right` — Dimension indices forming the column space
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
/// let result = qr(&a, &[0], &[1]).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `left` and `right` do not form a valid partition
/// of `0..tensor.ndim()`.
pub fn qr<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
) -> Result<QrResult<T>> {
    todo!()
}

/// Compute the LU decomposition of a tensor (partial pivoting).
///
/// Matricizes the tensor according to `left`/`right` dimension indices,
/// computes the LU factorization with partial pivoting, and reshapes
/// the factors back to tensor form.
///
/// # Arguments
///
/// * `tensor` — Input tensor
/// * `left` — Dimension indices forming the row space
/// * `right` — Dimension indices forming the column space
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lu;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = lu(&a, &[0], &[1]).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `left` and `right` do not form a valid partition
/// of `0..tensor.ndim()`.
pub fn lu<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
) -> Result<LuResult<T>> {
    todo!()
}

/// Compute the eigendecomposition of a tensor.
///
/// Matricizes the tensor according to `left`/`right` dimension indices,
/// computes the eigendecomposition of the resulting square matrix, and
/// reshapes the eigenvectors back to tensor form.
///
/// # Arguments
///
/// * `tensor` — Input tensor
/// * `left` — Dimension indices forming the row space
/// * `right` — Dimension indices forming the column space
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
/// let result = eigen(&a, &[0], &[1]).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `left` and `right` do not form a valid partition
/// of `0..tensor.ndim()`, or if the resulting matrix is not square.
pub fn eigen<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
) -> Result<EigenResult<T>> {
    todo!()
}

// ============================================================================
// AD cotangent types (bundle cotangents for multi-output decompositions)
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
pub struct SvdCotangent<T: ScalarBase> {
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
pub struct QrCotangent<T: ScalarBase> {
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
pub struct LuCotangent<T: ScalarBase> {
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
pub struct EigenCotangent<T: ScalarBase> {
    /// Cotangent for eigenvalues. Shape must match `EigenResult::values`.
    pub values: Option<Tensor<T>>,
    /// Cotangent for eigenvectors. Shape must match `EigenResult::vectors`.
    pub vectors: Option<Tensor<T>>,
}

// ============================================================================
// AD tracked result types (reverse-mode)
// ============================================================================

/// Tracked SVD result for reverse-mode AD.
///
/// Each output component is wrapped in [`TrackedTensor`] so gradients
/// can flow back through the decomposition.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::tracked_svd;
/// use chainrules::Tape;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_svd(&a, &[0], &[1], None).unwrap();
/// // result.u, result.s, result.vt are TrackedTensor values
/// ```
pub struct TrackedSvdResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Tracked left singular vectors.
    pub u: TrackedTensor<Tensor<T>>,
    /// Tracked singular values.
    pub s: TrackedTensor<Tensor<T>>,
    /// Tracked right singular vectors.
    pub vt: TrackedTensor<Tensor<T>>,
}

/// Tracked QR result for reverse-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::tracked_qr;
/// use chainrules::Tape;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_qr(&a, &[0], &[1]).unwrap();
/// ```
pub struct TrackedQrResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Tracked Q factor.
    pub q: TrackedTensor<Tensor<T>>,
    /// Tracked R factor.
    pub r: TrackedTensor<Tensor<T>>,
}

/// Tracked LU result for reverse-mode AD.
///
/// The permutation `p` is not tracked (discrete, non-differentiable).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::tracked_lu;
/// use chainrules::Tape;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_lu(&a, &[0], &[1]).unwrap();
/// ```
pub struct TrackedLuResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Row permutation (not tracked).
    pub p: Vec<usize>,
    /// Tracked L factor.
    pub l: TrackedTensor<Tensor<T>>,
    /// Tracked U factor.
    pub u: TrackedTensor<Tensor<T>>,
}

/// Tracked eigendecomposition result for reverse-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::tracked_eigen;
/// use chainrules::Tape;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_eigen(&a, &[0], &[1]).unwrap();
/// ```
pub struct TrackedEigenResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Tracked eigenvalues.
    pub values: TrackedTensor<Tensor<T>>,
    /// Tracked eigenvectors.
    pub vectors: TrackedTensor<Tensor<T>>,
}

// ============================================================================
// AD dual result types (forward-mode)
// ============================================================================

/// Dual SVD result for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::dual_svd;
/// use chainrules::DualTensor;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_svd(&a_dual, &[0], &[1], None).unwrap();
/// ```
pub struct DualSvdResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Dual left singular vectors.
    pub u: DualTensor<Tensor<T>>,
    /// Dual singular values.
    pub s: DualTensor<Tensor<T>>,
    /// Dual right singular vectors.
    pub vt: DualTensor<Tensor<T>>,
}

/// Dual QR result for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::dual_qr;
/// use chainrules::DualTensor;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[4, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_qr(&a_dual, &[0], &[1]).unwrap();
/// ```
pub struct DualQrResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Dual Q factor.
    pub q: DualTensor<Tensor<T>>,
    /// Dual R factor.
    pub r: DualTensor<Tensor<T>>,
}

/// Dual LU result for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::dual_lu;
/// use chainrules::DualTensor;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_lu(&a_dual, &[0], &[1]).unwrap();
/// ```
pub struct DualLuResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Row permutation (not differentiable).
    pub p: Vec<usize>,
    /// Dual L factor.
    pub l: DualTensor<Tensor<T>>,
    /// Dual U factor.
    pub u: DualTensor<Tensor<T>>,
}

/// Dual eigendecomposition result for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::dual_eigen;
/// use chainrules::DualTensor;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_eigen(&a_dual, &[0], &[1]).unwrap();
/// ```
pub struct DualEigenResult<T: ScalarBase>
where
    Tensor<T>: Differentiable,
{
    /// Dual eigenvalues.
    pub values: DualTensor<Tensor<T>>,
    /// Dual eigenvectors.
    pub vectors: DualTensor<Tensor<T>>,
}

// ============================================================================
// AD functions: tracked (reverse-mode)
// ============================================================================

/// Tracked SVD (reverse-mode AD).
///
/// Records the SVD operation on the reverse-mode tape so that
/// [`Tape::pullback`](chainrules::Tape::pullback) can compute gradients.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_linalg::tracked_svd;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 4], mem, col));
/// let result = tracked_svd(&a, &[0], &[1], None).unwrap();
/// ```
pub fn tracked_svd<T: ScalarBase>(
    _tensor: &TrackedTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
    _options: Option<&SvdOptions>,
) -> AdResult<TrackedSvdResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Tracked QR (reverse-mode AD).
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_linalg::tracked_qr;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_qr(&a, &[0], &[1]).unwrap();
/// ```
pub fn tracked_qr<T: ScalarBase>(
    _tensor: &TrackedTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<TrackedQrResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Tracked LU (reverse-mode AD).
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_linalg::tracked_lu;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_lu(&a, &[0], &[1]).unwrap();
/// ```
pub fn tracked_lu<T: ScalarBase>(
    _tensor: &TrackedTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<TrackedLuResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Tracked eigen (reverse-mode AD).
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_linalg::tracked_eigen;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
/// let result = tracked_eigen(&a, &[0], &[1]).unwrap();
/// ```
pub fn tracked_eigen<T: ScalarBase>(
    _tensor: &TrackedTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<TrackedEigenResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

// ============================================================================
// AD functions: dual (forward-mode)
// ============================================================================

/// Dual SVD (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_linalg::dual_svd;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_svd(&a_dual, &[0], &[1], None).unwrap();
/// ```
pub fn dual_svd<T: ScalarBase>(
    _tensor: &DualTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
    _options: Option<&SvdOptions>,
) -> AdResult<DualSvdResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Dual QR (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_linalg::dual_qr;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[4, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_qr(&a_dual, &[0], &[1]).unwrap();
/// ```
pub fn dual_qr<T: ScalarBase>(
    _tensor: &DualTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<DualQrResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Dual LU (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_linalg::dual_lu;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_lu(&a_dual, &[0], &[1]).unwrap();
/// ```
pub fn dual_lu<T: ScalarBase>(
    _tensor: &DualTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<DualLuResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Dual eigen (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_linalg::dual_eigen;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let result = dual_eigen(&a_dual, &[0], &[1]).unwrap();
/// ```
pub fn dual_eigen<T: ScalarBase>(
    _tensor: &DualTensor<Tensor<T>>,
    _left: &[usize],
    _right: &[usize],
) -> AdResult<DualEigenResult<T>>
where
    Tensor<T>: Differentiable,
{
    todo!()
}

// ============================================================================
// AD functions: rrule (reverse-mode rule, without tape)
// ============================================================================

/// Reverse-mode rule for SVD (pullback without tape).
///
/// Computes the gradient of the input tensor given cotangents for
/// the SVD outputs. Intended for FFI and manual AD.
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
/// let result = svd(&a, &[0], &[1], None).unwrap();
///
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::ones(&[3], mem, col)),
///     vt: None,
/// };
/// let grad_a = svd_rrule(&a, &[0], &[1], None, &cotangent).unwrap();
/// ```
pub fn svd_rrule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _options: Option<&SvdOptions>,
    _cotangent: &SvdCotangent<T>,
) -> Result<Tensor<T>> {
    todo!()
}

/// Reverse-mode rule for QR (pullback without tape).
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
/// let grad_a = qr_rrule(&a, &[0], &[1], &cotangent).unwrap();
/// ```
pub fn qr_rrule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _cotangent: &QrCotangent<T>,
) -> Result<Tensor<T>> {
    todo!()
}

/// Reverse-mode rule for LU (pullback without tape).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{lu_rrule, LuCotangent};
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
/// let grad_a = lu_rrule(&a, &[0], &[1], &cotangent).unwrap();
/// ```
pub fn lu_rrule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _cotangent: &LuCotangent<T>,
) -> Result<Tensor<T>> {
    todo!()
}

/// Reverse-mode rule for eigendecomposition (pullback without tape).
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
/// let grad_a = eigen_rrule(&a, &[0], &[1], &cotangent).unwrap();
/// ```
pub fn eigen_rrule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _cotangent: &EigenCotangent<T>,
) -> Result<Tensor<T>> {
    todo!()
}

// ============================================================================
// AD functions: frule (forward-mode rule, without tape)
// ============================================================================

/// Forward-mode rule for SVD (pushforward without tape).
///
/// Computes the JVP of all SVD outputs given a tangent for the input tensor.
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
/// let result = svd_frule(&a, &[0], &[1], None, Some(&da)).unwrap();
/// ```
pub fn svd_frule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _options: Option<&SvdOptions>,
    _tangent: Option<&Tensor<T>>,
) -> Result<SvdResult<T>> {
    todo!()
}

/// Forward-mode rule for QR (pushforward without tape).
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
/// let result = qr_frule(&a, &[0], &[1], Some(&da)).unwrap();
/// ```
pub fn qr_frule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _tangent: Option<&Tensor<T>>,
) -> Result<QrResult<T>> {
    todo!()
}

/// Forward-mode rule for LU (pushforward without tape).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::lu_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let result = lu_frule(&a, &[0], &[1], Some(&da)).unwrap();
/// ```
pub fn lu_frule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _tangent: Option<&Tensor<T>>,
) -> Result<LuResult<T>> {
    todo!()
}

/// Forward-mode rule for eigendecomposition (pushforward without tape).
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
/// let result = eigen_frule(&a, &[0], &[1], Some(&da)).unwrap();
/// ```
pub fn eigen_frule<T: ScalarBase>(
    _tensor: &Tensor<T>,
    _left: &[usize],
    _right: &[usize],
    _tangent: Option<&Tensor<T>>,
) -> Result<EigenResult<T>> {
    todo!()
}
