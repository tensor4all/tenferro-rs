use super::*;

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
/// ```
/// use tenferro_linalg::svd;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = svd(&mut ctx, &a, None).unwrap();
/// assert_eq!(result.s.ndim(), 1);
/// ```
pub struct SvdResult<T: Scalar, R: Scalar = T> {
    /// Left singular vectors. Shape: `(m, k, *)`.
    pub u: Tensor<T>,
    /// Singular values (descending order, always real). Shape: `(k, *)`.
    pub s: Tensor<R>,
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
#[derive(Debug, Clone, Default)]
pub struct SvdOptions {
    /// Maximum number of singular values to keep. `None` means no limit.
    pub max_rank: Option<usize>,
    /// Discard singular values below this threshold. `None` means no cutoff.
    pub cutoff: Option<f64>,
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
/// ```
/// use tenferro_linalg::qr;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = qr(&mut ctx, &a).unwrap();
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
/// ```
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor
/// ).unwrap();
///
/// // With partial pivoting (default)
/// let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
/// assert!(result.p.is_some());
///
/// // NoPivot is also supported.
/// let no_pivot = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
/// assert!(no_pivot.p.is_none());
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

/// Structured Cholesky result with numerical status information.
///
/// `info` contains one entry per batch matrix. A zero entry indicates success.
/// A positive entry indicates that the corresponding batch matrix was not
/// positive definite.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cholesky_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[4.0_f64, 2.0, 2.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = cholesky_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.l.dims(), &[2, 2]);
/// assert_eq!(result.info, vec![0]);
/// ```
#[derive(Debug)]
pub struct CholeskyExResult<T: Scalar> {
    /// Lower-triangular Cholesky factor. Shape: `(n, n, *)`.
    pub l: Tensor<T>,
    /// Per-batch numerical status, flattened over batch dimensions.
    pub info: Vec<i32>,
}

/// Structured inverse result with numerical status information.
///
/// `info` contains one entry per batch matrix. A zero entry indicates success.
/// A positive entry indicates that the corresponding batch matrix was singular
/// or numerically unstable for inversion.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = inv_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.inverse.dims(), &[2, 2]);
/// assert_eq!(result.info, vec![0]);
/// ```
#[derive(Debug)]
pub struct InvExResult<T: Scalar> {
    /// Inverse matrix. Shape: `(n, n, *)`.
    pub inverse: Tensor<T>,
    /// Per-batch numerical status, flattened over batch dimensions.
    pub info: Vec<i32>,
}

/// Structured solve result with numerical status information.
///
/// `info` contains one entry per batch matrix. A zero entry indicates success.
/// A positive entry indicates that the corresponding batch matrix was singular
/// or numerically unstable for the solve.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let b = Tensor::from_slice(&[2.0_f64, -1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let result = solve_ex(&mut ctx, &a, &b).unwrap();
/// assert_eq!(result.solution.dims(), &[2]);
/// assert_eq!(result.info, vec![0]);
/// ```
#[derive(Debug)]
pub struct SolveExResult<T: Scalar> {
    /// Solution tensor. Shape matches the input `b`.
    pub solution: Tensor<T>,
    /// Per-batch numerical status, flattened over batch dimensions.
    pub info: Vec<i32>,
}

/// Packed LU factorization result.
///
/// `factors` stores the strict lower-triangular multipliers in its lower part
/// and the upper-triangular factor in its upper part, using the same packed
/// layout as LAPACK `getrf`.
///
/// `pivots` contains the forward row-permutation indices for each batch matrix,
/// flattened as `m * batch_count`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lu_factor;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = lu_factor(&mut ctx, &a).unwrap();
/// assert_eq!(result.factors.dims(), &[2, 2]);
/// assert_eq!(result.pivots.len(), 2);
/// ```
#[derive(Debug)]
pub struct LuFactorResult<T: Scalar> {
    /// Packed LU factors with the same shape as the input matrix.
    pub factors: Tensor<T>,
    /// Forward row-permutation indices, flattened over batch dimensions.
    pub pivots: Vec<usize>,
}

/// Packed LU factorization result with numerical status information.
///
/// `info` contains one entry per batch matrix. A zero entry indicates success.
/// A positive entry indicates that the corresponding `U(info, info)` diagonal
/// entry was numerically zero after factorization.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lu_factor_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = lu_factor_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.factors.dims(), &[2, 2]);
/// assert_eq!(result.pivots.len(), 2);
/// assert_eq!(result.info, vec![0]);
/// ```
#[derive(Debug)]
pub struct LuFactorExResult<T: Scalar> {
    /// Packed LU factors with the same shape as the input matrix.
    pub factors: Tensor<T>,
    /// Forward row-permutation indices, flattened over batch dimensions.
    pub pivots: Vec<usize>,
    /// Per-batch numerical status, flattened over batch dimensions.
    pub info: Vec<i32>,
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
/// ```
/// use tenferro_linalg::eigen;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eigen(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
pub struct EigenResult<T: Scalar, R: Scalar = T> {
    /// Eigenvalues (real for Hermitian/symmetric). Shape: `(n, *)`.
    pub values: Tensor<R>,
    /// Right eigenvectors (columns). Shape: `(n, n, *)`.
    pub vectors: Tensor<T>,
}

/// Result of general eigendecomposition (always complex-valued).
///
/// Unlike [`EigenResult`] (which is for symmetric/Hermitian matrices with real eigenvalues),
/// this type always returns complex eigenvalues and eigenvectors, since a general
/// (non-symmetric) real matrix can have complex eigenvalues.
///
/// For an input of shape `(n, n, *)`:
///
/// - `values`: shape `(n, *)` — complex eigenvalues
/// - `vectors`: shape `(n, n, *)` — complex right eigenvectors (columns)
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{eig, EigResult};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result: EigResult<f64> = eig(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
pub struct EigResult<R: LinalgScalar<Real = R> + num_traits::Float> {
    /// Complex eigenvalues. Shape: `(n, *)`.
    pub values: Tensor<num_complex::Complex<R>>,
    /// Complex right eigenvectors (columns). Shape: `(n, n, *)`.
    pub vectors: Tensor<num_complex::Complex<R>>,
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
/// ```
/// use tenferro_linalg::slogdet;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = slogdet(&mut ctx, &a).unwrap();
/// ```
pub struct SlogdetResult<T: Scalar, R: Scalar = T> {
    /// Sign of determinant. Shape: `(*)`.
    pub sign: Tensor<T>,
    /// Log of absolute value of determinant (always real). Shape: `(*)`.
    pub logabsdet: Tensor<R>,
}

/// Gradient result for `solve_rrule`: cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col);
/// let grad = solve_rrule(&mut ctx, &a, &b, &cotangent).unwrap();
/// // grad.a: shape [3, 3], grad.b: shape [3]
/// ```
pub struct SolveGrad<T: Scalar> {
    /// Cotangent for A. Same shape as `A`.
    pub a: Tensor<T>,
    /// Cotangent for b. Same shape as `b`.
    pub b: Tensor<T>,
}

/// Norm kind for [`norm`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::NormKind;
///
/// let kind = NormKind::Fro;
/// let lp = NormKind::Lp(3.0);
/// ```
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
/// ```
/// use tenferro_linalg::{lstsq, lstsq_rrule};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let dx = Tensor::<f64>::ones(&[2], mem, col);
/// let grad = lstsq_rrule(&mut ctx, &a, &b, &dx).unwrap();
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
/// ```
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
/// ```
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
/// ```
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
/// ```
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

/// Cotangent (adjoint) for general eigendecomposition outputs.
///
/// Unlike [`EigenCotangent`] (used for symmetric `eigen`), this cotangent
/// carries complex-valued tensors because `eig()` returns complex
/// eigenvalues and eigenvectors even for real inputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::EigCotangent;
/// use num_complex::Complex64;
///
/// let cotangent = EigCotangent::<f64> { values: None, vectors: None };
/// ```
pub struct EigCotangent<R: LinalgScalar<Real = R> + num_traits::Float> {
    /// Cotangent for eigenvalues. Shape: `(n, *)`. Complex-valued.
    pub values: Option<Tensor<num_complex::Complex<R>>>,
    /// Cotangent for eigenvectors. Shape: `(n, n, *)`. Complex-valued.
    pub vectors: Option<Tensor<num_complex::Complex<R>>>,
}

/// Cotangent (adjoint) for slogdet outputs.
///
/// Note: `sign` is piecewise constant and not differentiable.
/// Gradient flows only through `logabsdet`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SlogdetCotangent;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::<f64>::ones(&[], mem, col)),
/// };
/// ```
pub struct SlogdetCotangent<T: Scalar> {
    /// Cotangent for logabsdet. Shape must match `SlogdetResult::logabsdet`.
    pub logabsdet: Option<Tensor<T>>,
}
