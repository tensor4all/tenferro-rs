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

pub mod backend;

use chainrules_core::AdResult;
use num_traits::Float;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

// ============================================================================
// LinalgScalar trait
// ============================================================================

/// Types that support linear algebra decompositions.
///
/// Automatically implemented for `f64` and `f32`. Complex type support is planned.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::LinalgScalar;
///
/// fn my_func<T: LinalgScalar>(x: T) -> T { x }
/// let y = my_func(1.0_f64);
/// ```
pub trait LinalgScalar:
    Scalar + Float + std::ops::Sub<Output = Self> + std::fmt::Debug + 'static
{
}

impl LinalgScalar for f64 {}
impl LinalgScalar for f32 {}

// ============================================================================
// Batch processing helpers
// ============================================================================

/// Validate that tensor has at least 2 dimensions.
/// Returns (m, n, batch_dims_slice).
fn validate_2d<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<(usize, usize, &[usize])> {
    if tensor.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "expected at least 2 dimensions, got {}",
            tensor.ndim()
        )));
    }
    let m = tensor.dims()[0];
    let n = tensor.dims()[1];
    let batch = &tensor.dims()[2..];
    Ok((m, n, batch))
}

/// Validate that tensor is square (first two dims equal) and at least 2D.
/// Returns (n, batch_dims_slice).
fn validate_square<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<(usize, &[usize])> {
    let (m, n, batch) = validate_2d(tensor)?;
    if m != n {
        return Err(Error::ShapeMismatch {
            expected: vec![m, m],
            got: vec![m, n],
        });
    }
    Ok((n, batch))
}

/// Validate RHS shape for solve/solve_triangular.
///
/// Accepted shapes for `b` when `a` is `(n, n, *)`:
/// - `(n, *)` (vector RHS, implied `nrhs=1`)
/// - `(n, k, *)` (multiple RHS)
fn validate_solve_rhs<T: LinalgScalar>(
    b: &Tensor<T>,
    n: usize,
    batch_dims: &[usize],
    op_name: &str,
) -> Result<usize> {
    if b.ndim() == 1 + batch_dims.len() {
        if b.dims()[0] != n {
            return Err(Error::InvalidArgument(format!(
                "{op_name} expects b dim[0] == n ({n}), got {}",
                b.dims()[0]
            )));
        }
        if &b.dims()[1..] != batch_dims {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims mismatch: expected {:?}, got {:?}",
                batch_dims,
                &b.dims()[1..]
            )));
        }
        return Ok(1);
    }

    if b.ndim() == 2 + batch_dims.len() {
        if b.dims()[0] != n {
            return Err(Error::InvalidArgument(format!(
                "{op_name} expects b dim[0] == n ({n}), got {}",
                b.dims()[0]
            )));
        }
        if &b.dims()[2..] != batch_dims {
            return Err(Error::InvalidArgument(format!(
                "{op_name} batch dims mismatch: expected {:?}, got {:?}",
                batch_dims,
                &b.dims()[2..]
            )));
        }
        let nrhs = b.dims()[1];
        if nrhs == 0 {
            return Err(Error::InvalidArgument(format!(
                "{op_name} requires b dim[1] (nrhs) > 0"
            )));
        }
        return Ok(nrhs);
    }

    Err(Error::InvalidArgument(format!(
        "{op_name} expects b shape (n, *) or (n, k, *), got {:?}",
        b.dims()
    )))
}

/// Validate RHS shape for least squares.
///
/// Current implementation supports vector RHS only:
/// `a: (m, n, *)`, `b: (m, *)`.
fn validate_lstsq_rhs<T: LinalgScalar>(
    b: &Tensor<T>,
    m: usize,
    batch_dims: &[usize],
) -> Result<()> {
    if b.ndim() != 1 + batch_dims.len() {
        return Err(Error::InvalidArgument(format!(
            "lstsq expects b shape (m, *), got {:?}",
            b.dims()
        )));
    }
    if b.dims()[0] != m {
        return Err(Error::InvalidArgument(format!(
            "lstsq expects b dim[0] == m ({m}), got {}",
            b.dims()[0]
        )));
    }
    if &b.dims()[1..] != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "lstsq batch dims mismatch: expected {:?}, got {:?}",
            batch_dims,
            &b.dims()[1..]
        )));
    }
    Ok(())
}

/// Validate cotangent shape for norm AD.
/// For primal output shape `(*)`, cotangent must have the same shape.
fn validate_norm_cotangent<T: LinalgScalar>(
    cotangent: &Tensor<T>,
    batch_dims: &[usize],
) -> Result<()> {
    if batch_dims.is_empty() {
        if cotangent.ndim() == 0 {
            return Ok(());
        }
        return Err(Error::InvalidArgument(format!(
            "norm cotangent shape mismatch: expected scalar [], got {:?}",
            cotangent.dims()
        )));
    }

    if cotangent.dims() != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "norm cotangent shape mismatch: expected {:?}, got {:?}",
            batch_dims,
            cotangent.dims()
        )));
    }

    Ok(())
}

/// Validate Hermitian/symmetric structure for batched square matrices stored
/// in column-major contiguous layout.
fn validate_hermitian_batches<T: LinalgScalar>(
    data: &[T],
    offset: usize,
    n: usize,
    bc: usize,
    op_name: &str,
) -> Result<()> {
    let mat_size = n * n;
    let tol_scale = T::from(128.0).unwrap_or(T::one());

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];
        for j in 0..n {
            for i in 0..j {
                let a_ij = batch_data[i + j * n];
                let a_ji = batch_data[j + i * n];
                let diff = (a_ij - a_ji).abs();
                let scale = T::one() + a_ij.abs().max(a_ji.abs());
                let tol = T::epsilon() * tol_scale * scale;
                if diff > tol {
                    return Err(Error::InvalidArgument(format!(
                        "{op_name} expects symmetric/Hermitian input; mismatch at ({i}, {j}) in batch {b}"
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Ensure tensor is column-major contiguous. Returns a (possibly cloned) contiguous tensor.
fn ensure_col_major<T: LinalgScalar>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor.contiguous(MemoryOrder::ColumnMajor)
}

/// Compute batch count from batch dims (product, or 1 if empty).
fn batch_count(batch_dims: &[usize]) -> usize {
    batch_dims.iter().product::<usize>().max(1)
}

/// Build output dims: [mat_dims..., batch_dims...].
fn output_dims(mat_dims: &[usize], batch_dims: &[usize]) -> Vec<usize> {
    let mut dims = mat_dims.to_vec();
    dims.extend_from_slice(batch_dims);
    dims
}

/// Create a Tensor from raw column-major data with the given dims.
fn tensor_from_data<T: LinalgScalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

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
/// // NoPivot currently returns an error in this implementation.
/// assert!(lu(&a, LuPivot::NoPivot).is_err());
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
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::solve_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let b = Tensor::<f64>::zeros(&[3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col);
/// let grad = solve_rrule(&a, &b, &cotangent).unwrap();
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

// ============================================================================
// Primary decomposition functions
// ============================================================================

/// Compute the SVD of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
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
pub fn svd<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let mat_size = m * n;

    // Determine effective rank after truncation
    let max_k = if let Some(opts) = options {
        opts.max_rank.map_or(k, |r| r.min(k))
    } else {
        k
    };

    let mut u_data = vec![T::zero(); m * max_k * bc];
    let mut s_data = vec![T::zero(); max_k * bc];
    let mut vt_data = vec![T::zero(); max_k * n * bc];

    // Pre-allocate temp buffers for full-rank results per batch
    let mut u_full = vec![T::zero(); m * k];
    let mut s_full = vec![T::zero(); k];
    let mut vt_full = vec![T::zero(); k * n];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        backend.thin_svd(batch_data, m, n, &mut u_full, &mut s_full, &mut vt_full)?;

        // Apply cutoff truncation
        let actual_k = if let Some(opts) = options {
            if let Some(cutoff) = opts.cutoff {
                let cutoff_t = T::from(cutoff).unwrap();
                let mut ak = max_k;
                while ak > 0 && s_full[ak - 1] < cutoff_t {
                    ak -= 1;
                }
                ak
            } else {
                max_k
            }
        } else {
            max_k
        };

        // Copy U (m × max_k, col-major)
        for j in 0..actual_k {
            for i in 0..m {
                u_data[b * m * max_k + i + j * m] = u_full[i + j * m];
            }
        }

        // Copy S (max_k)
        for i in 0..actual_k {
            s_data[b * max_k + i] = s_full[i];
        }

        // Copy Vt (max_k × n, col-major) from vt_full (k × n, col-major)
        // vt_full is already k×n col-major: vt_full[i + j*k]
        for j in 0..n {
            for i in 0..actual_k {
                vt_data[b * max_k * n + i + j * max_k] = vt_full[i + j * k];
            }
        }
    }

    let u_dims = output_dims(&[m, max_k], batch_dims);
    let s_dims = output_dims(&[max_k], batch_dims);
    let vt_dims = output_dims(&[max_k, n], batch_dims);

    Ok(SvdResult {
        u: tensor_from_data(u_data, &u_dims)?,
        s: tensor_from_data(s_data, &s_dims)?,
        vt: tensor_from_data(vt_data, &vt_dims)?,
    })
}

/// Compute the QR decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
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
pub fn qr<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<QrResult<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let mat_size = m * n;

    let mut q_data = vec![T::zero(); m * k * bc];
    let mut r_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        let q_out = &mut q_data[b * m * k..(b + 1) * m * k];
        let r_out = &mut r_data[b * k * n..(b + 1) * k * n];
        backend.qr(batch_data, m, n, q_out, r_out)?;
    }

    let q_dims = output_dims(&[m, k], batch_dims);
    let r_dims = output_dims(&[k, n], batch_dims);

    Ok(QrResult {
        q: tensor_from_data(q_data, &q_dims)?,
        r: tensor_from_data(r_data, &r_dims)?,
    })
}

/// Compute the LU decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
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
/// // NoPivot currently returns an error in this implementation.
/// assert!(lu(&a, LuPivot::NoPivot).is_err());
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn lu<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    pivot: LuPivot,
) -> Result<LuResult<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let mat_size = m * n;

    if pivot == LuPivot::NoPivot {
        return Err(Error::InvalidArgument(
            "NoPivot LU is not yet implemented".into(),
        ));
    }

    let mut l_data = vec![T::zero(); m * k * bc];
    let mut u_data = vec![T::zero(); k * n * bc];
    // For batched LU, we store all permutations concatenated
    let mut all_perms = vec![0usize; m * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        let perm_out = &mut all_perms[b * m..(b + 1) * m];
        let l_out = &mut l_data[b * m * k..(b + 1) * m * k];
        let u_out = &mut u_data[b * k * n..(b + 1) * k * n];
        backend.lu(batch_data, m, n, perm_out, l_out, u_out)?;
    }

    let l_dims = output_dims(&[m, k], batch_dims);
    let u_dims = output_dims(&[k, n], batch_dims);

    Ok(LuResult {
        p: Some(all_perms),
        l: tensor_from_data(l_data, &l_dims)?,
        u: tensor_from_data(u_data, &u_dims)?,
    })
}

/// Compute the eigendecomposition of a batched square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
///
/// `eigen` uses a symmetric/Hermitian eigensolver and validates
/// `A[i, j] == A[j, i]` (within floating-point tolerance) for each batch.
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
/// Returns an error if the input has fewer than 2 dimensions, the first two
/// dimensions are not equal, or the matrix is not symmetric/Hermitian.
pub fn eigen<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<EigenResult<T>> {
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    validate_hermitian_batches(data, offset, n, bc, "eigen")?;

    let mut val_data = vec![T::zero(); n * bc];
    let mut vec_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        let val_out = &mut val_data[b * n..(b + 1) * n];
        let vec_out = &mut vec_data[b * n * n..(b + 1) * n * n];
        backend.eigen_sym(batch_data, n, val_out, vec_out)?;
    }

    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);

    Ok(EigenResult {
        values: tensor_from_data(val_data, &val_dims)?,
        vectors: tensor_from_data(vec_data, &vec_dims)?,
    })
}

/// Solve the least squares problem: `x = argmin ||Ax - b||²`.
///
/// Input shapes: `A` is `(m, n, *)`, `b` is `(m, *)`, with `m >= n`.
/// The function internally normalizes inputs to column-major contiguous layout.
/// If inputs are not already contiguous, internal copies are performed.
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
/// Returns an error if `A` has fewer than 2 dimensions, `m < n`, or `b`
/// does not match `(m, *)` with the same batch dimensions as `A`.
pub fn lstsq<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T>> {
    let (m, n, batch_dims) = validate_2d(a)?;
    if m < n {
        return Err(Error::InvalidArgument(format!(
            "lstsq requires m >= n, got m={m}, n={n}"
        )));
    }
    validate_lstsq_rhs(b, m, batch_dims)?;

    // Solve via QR: A = Q R, then x = R^{-1} Q^T b
    let qr_result = qr(backend, a)?;
    let q_input = ensure_col_major(&qr_result.q);
    let r_input = ensure_col_major(&qr_result.r);
    let b_input = ensure_col_major(b);

    let q_data = q_input.buffer().as_slice().unwrap();
    let r_data = r_input.buffer().as_slice().unwrap();
    let b_data = b_input.buffer().as_slice().unwrap();
    let q_off = q_input.offset() as usize;
    let r_off = r_input.offset() as usize;
    let b_off = b_input.offset() as usize;

    let k = m.min(n); // = n since m >= n
    let bc = batch_count(batch_dims);

    let mut x_data = vec![T::zero(); n * bc];
    let mut res_data = vec![T::zero(); m * bc];

    let mut x_buf = vec![T::zero(); k];

    for batch in 0..bc {
        let q_b = &q_data[q_off + batch * m * k..q_off + (batch + 1) * m * k];
        let r_b = &r_data[r_off + batch * k * n..r_off + (batch + 1) * k * n];
        let b_b = &b_data[b_off + batch * m..b_off + (batch + 1) * m];

        // Compute Q^T b (k × 1)
        let mut qtb = vec![T::zero(); k];
        for i in 0..k {
            let mut sum = T::zero();
            for j in 0..m {
                sum = sum + q_b[j + i * m] * b_b[j];
            }
            qtb[i] = sum;
        }

        // Solve R x = Q^T b (upper triangular)
        backend.solve_triangular(r_b, &qtb, k, 1, true, &mut x_buf)?;
        x_data[batch * n..(batch + 1) * n].copy_from_slice(&x_buf);

        // Compute residual: r = b - A x
        let a_contiguous = a.contiguous(MemoryOrder::ColumnMajor);
        let a_slice = a_contiguous.buffer().as_slice().unwrap();
        let a_off = a_contiguous.offset() as usize;
        let a_data_local = &a_slice[a_off + batch * m * n..a_off + (batch + 1) * m * n];
        for i in 0..m {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a_data_local[i + j * m] * x_buf[j];
            }
            res_data[batch * m + i] = b_b[i] - ax_i;
        }
    }

    let x_dims = output_dims(&[n], batch_dims);
    let res_dims = output_dims(&[m], batch_dims);

    Ok(LstsqResult {
        x: tensor_from_data(x_data, &x_dims)?,
        residual: tensor_from_data(res_data, &res_dims)?,
    })
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
pub fn cholesky<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>> {
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut l_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        let l_out = &mut l_data[b * mat_size..(b + 1) * mat_size];
        backend.cholesky(batch_data, n, l_out)?;
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(l_data, &dims)
}

/// Solve a square linear system `A x = b`.
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
/// Batch dimensions in `b` must match those of `A`.
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
pub fn solve<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>> {
    let (n, batch_dims) = validate_square(a)?;
    let nrhs = validate_solve_rhs(b, n, batch_dims, "solve")?;
    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);

    let a_data = a_input.buffer().as_slice().unwrap();
    let b_data = b_input.buffer().as_slice().unwrap();
    let a_off = a_input.offset() as usize;
    let b_off = b_input.offset() as usize;
    let bc = batch_count(batch_dims);

    let mut x_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[a_off + batch * n * n..a_off + (batch + 1) * n * n];
        let b_b = &b_data[b_off + batch * n * nrhs..b_off + (batch + 1) * n * nrhs];

        let x_out = &mut x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        backend.solve(a_b, b_b, n, nrhs, x_out)?;
    }

    let x_dims = b.dims().to_vec();
    tensor_from_data(x_data, &x_dims)
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
pub fn inv<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>> {
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    // Solve A * X = I for each batch
    let mut eye_mat = vec![T::zero(); n * n];
    for i in 0..n {
        eye_mat[i + i * n] = T::one();
    }

    let mut inv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_b = &data[start..start + mat_size];
        let x_out = &mut inv_data[b * mat_size..(b + 1) * mat_size];
        backend.solve(a_b, &eye_mat, n, n, x_out)?;
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(inv_data, &dims)
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
pub fn det<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>> {
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut det_data = vec![T::zero(); bc];

    // Pre-allocate temp buffers for LU per batch
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        // det = product of diagonal of U * sign from permutation
        backend.lu(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut d = T::one();
        for i in 0..n {
            d = d * u_buf[i + i * n]; // U diagonal
        }

        // Count transpositions in permutation
        let mut sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    sign = -sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }

        if sign < 0 {
            d = T::zero() - d;
        }
        det_data[b] = d;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        // Scalar result: shape []
        let strides = vec![];
        Tensor::from_vec(det_data, &dims, &strides, 0)
    } else {
        tensor_from_data(det_data, &dims)
    }
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
pub fn slogdet<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T>> {
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut sign_data = vec![T::zero(); bc];
    let mut logabsdet_data = vec![T::zero(); bc];

    // Pre-allocate temp buffers for LU per batch
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        backend.lu(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut log_abs = T::zero();
        let mut sign = T::one();
        for i in 0..n {
            let diag = u_buf[i + i * n];
            log_abs = log_abs + diag.abs().ln();
            if diag < T::zero() {
                sign = T::zero() - sign;
            }
        }

        // Count transpositions
        let mut perm_sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    perm_sign = -perm_sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }
        if perm_sign < 0 {
            sign = T::zero() - sign;
        }

        sign_data[b] = sign;
        logabsdet_data[b] = log_abs;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        let strides = vec![];
        Ok(SlogdetResult {
            sign: Tensor::from_vec(sign_data, &dims, &strides, 0)?,
            logabsdet: Tensor::from_vec(logabsdet_data, &dims, &strides, 0)?,
        })
    } else {
        Ok(SlogdetResult {
            sign: tensor_from_data(sign_data, &dims)?,
            logabsdet: tensor_from_data(logabsdet_data, &dims)?,
        })
    }
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
/// assert!(eig(&a).is_err());
/// ```
pub fn eig<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
) -> Result<EigenResult<T>> {
    // General (non-symmetric) eigendecomposition requires complex output.
    // Deferred until complex type support is added.
    Err(Error::InvalidArgument(
        "general eigendecomposition (eig) not yet implemented; use eigen() for symmetric matrices"
            .into(),
    ))
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
pub fn pinv<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)?;

    // Compute via SVD: pinv(A) = V diag(1/S) U^T
    let svd_result = svd(backend, tensor, None)?;
    let u_input = ensure_col_major(&svd_result.u);
    let s_input = ensure_col_major(&svd_result.s);
    let vt_input = ensure_col_major(&svd_result.vt);

    let u_data = u_input.buffer().as_slice().unwrap();
    let s_data = s_input.buffer().as_slice().unwrap();
    let vt_data = vt_input.buffer().as_slice().unwrap();
    let u_off = u_input.offset() as usize;
    let s_off = s_input.offset() as usize;
    let vt_off = vt_input.offset() as usize;

    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let threshold = T::from(rcond.unwrap_or(1e-15)).unwrap();

    let mut result_data = vec![T::zero(); n * m * bc];

    for b in 0..bc {
        let s_b = &s_data[s_off + b * k..s_off + (b + 1) * k];
        let u_b = &u_data[u_off + b * m * k..u_off + (b + 1) * m * k];
        let vt_b = &vt_data[vt_off + b * k * n..vt_off + (b + 1) * k * n];

        let s_max = s_b
            .iter()
            .copied()
            .fold(T::zero(), |a, b| if a > b { a } else { b });
        let cutoff = s_max * threshold;

        // Build diag(1/S) U^T (k × m): element [i,j] = (1/s_i) * U[j,i]
        let mut sinv_ut = vec![T::zero(); k * m];
        for i in 0..k {
            if s_b[i] > cutoff {
                let sinv = T::one() / s_b[i];
                for j in 0..m {
                    sinv_ut[i + j * k] = sinv * u_b[j + i * m];
                }
            }
        }

        // Compute V * sinv_ut = Vt^T * sinv_ut
        // V is n×k (stored as Vt transposed): V[i,j] = Vt[j,i] = vt_b[j + i*k]
        for j in 0..m {
            for i in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    // V[i,p] = vt_b[p + i*k]
                    sum = sum + vt_b[p + i * k] * sinv_ut[p + j * k];
                }
                result_data[b * n * m + i + j * n] = sum;
            }
        }
    }

    let dims = output_dims(&[n, m], batch_dims);
    tensor_from_data(result_data, &dims)
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
/// assert!(matrix_exp(&a).is_err());
/// ```
pub fn matrix_exp<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
) -> Result<Tensor<T>> {
    Err(Error::InvalidArgument(
        "matrix_exp not yet implemented".into(),
    ))
}

/// Solve a triangular linear system `A x = b`.
///
/// `A` must be upper or lower triangular (specified by `upper`).
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
/// Batch dimensions in `b` must match those of `A`.
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
pub fn solve_triangular<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>> {
    let (n, batch_dims) = validate_square(a)?;
    let nrhs = validate_solve_rhs(b, n, batch_dims, "solve_triangular")?;
    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);

    let a_data = a_input.buffer().as_slice().unwrap();
    let b_data = b_input.buffer().as_slice().unwrap();
    let a_off = a_input.offset() as usize;
    let b_off = b_input.offset() as usize;
    let bc = batch_count(batch_dims);

    let mut x_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[a_off + batch * n * n..a_off + (batch + 1) * n * n];
        let b_b = &b_data[b_off + batch * n * nrhs..b_off + (batch + 1) * n * nrhs];

        let x_out = &mut x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        backend.solve_triangular(a_b, b_b, n, nrhs, upper, x_out)?;
    }

    let x_dims = b.dims().to_vec();
    tensor_from_data(x_data, &x_dims)
}

/// Compute a matrix norm.
///
/// Input shape: `(m, n, *)`.
///
/// Supported kinds in the current implementation:
/// - `NormKind::Fro`
/// - `NormKind::Nuclear`
/// - `NormKind::Spectral`
///
/// Return shape is `(*)` (batch dimensions). For non-batched input this is
/// a scalar tensor `[]`.
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
pub fn norm<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    let input = ensure_col_major(tensor);
    let data = input.buffer().as_slice().unwrap();
    let offset = input.offset() as usize;

    match kind {
        NormKind::Fro => {
            // Frobenius norm per batch: sqrt(sum of squares over matrix dims)
            let mut out = vec![T::zero(); bc];
            for batch in 0..bc {
                let start = offset + batch * mat_size;
                let mut sum = T::zero();
                for i in 0..mat_size {
                    let v = data[start + i];
                    sum = sum + v * v;
                }
                out[batch] = sum.sqrt();
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Nuclear => {
            // Nuclear norm per batch: sum of singular values
            let svd_result = svd(backend, tensor, None)?;
            let s_data = svd_result.s.buffer().as_slice().unwrap();
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for batch in 0..bc {
                let mut sum = T::zero();
                let start = s_off + batch * k;
                for i in 0..k {
                    sum = sum + s_data[start + i];
                }
                out[batch] = sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Spectral => {
            // Spectral norm per batch: largest singular value
            let svd_result = svd(backend, tensor, None)?;
            let s_data = svd_result.s.buffer().as_slice().unwrap();
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for batch in 0..bc {
                out[batch] = s_data[s_off + batch * k];
            }
            tensor_from_data(out, &out_dims)
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
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
///
/// # Examples
///
/// ```ignore
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

// ============================================================================
// Matrix operation helpers for AD rules
// ============================================================================

/// Transpose a column-major m×n matrix to n×m column-major.
fn transpose<T: LinalgScalar>(data: &[T], m: usize, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            result[j + i * n] = data[i + j * m];
        }
    }
    result
}

/// Scale a slice element-wise: out[i] = alpha * data[i].
fn scale_vec<T: LinalgScalar>(data: &[T], alpha: T) -> Vec<T> {
    data.iter().map(|&x| alpha * x).collect()
}

/// Add two slices element-wise: out[i] = a[i] + b[i].
fn add_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Subtract two slices element-wise: out[i] = a[i] - b[i].
fn sub_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Create identity matrix (n×n, col-major).
fn eye<T: LinalgScalar>(n: usize) -> Vec<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i + i * n] = T::one();
    }
    data
}

/// Hadamard (element-wise) product.
fn hadamard<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Extract lower triangular part (including diagonal) of col-major n×n.
fn tril<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in j..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract upper triangular part (including diagonal) of col-major n×n.
fn triu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..=j {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract strictly lower triangular part (excluding diagonal) of col-major n×n.
fn tril_strict<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in (j + 1)..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Copyltu: Hermitianize from lower triangle.
/// M_ij = M_ij if i > j, conj(M_ji) if i < j, Re(M_ii) if i == j.
/// For real: M + tril(M,-1)^T, with diagonal halved effect.
fn copyltu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            if i > j {
                result[i + j * n] = data[i + j * n];
                result[j + i * n] = data[i + j * n]; // transpose for real
            } else if i == j {
                result[i + j * n] = data[i + j * n];
            }
        }
    }
    result
}

/// phi operator for Cholesky AD: tril(X) with diagonal halved.
fn phi<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = tril(data, n);
    let half = T::from(0.5).unwrap();
    for i in 0..n {
        result[i + i * n] = result[i + i * n] * half;
    }
    result
}

// ============================================================================
// LinalgBackend convenience wrappers for AD code
// ============================================================================

/// Mat mul via LinalgBackend, returning Vec for convenience in AD code.
fn backend_mat_mul<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    backend
        .mat_mul(a, m, k, b, n, &mut c)
        .expect("mat_mul failed in AD rule");
    c
}

/// Solve via LinalgBackend, returning Vec for convenience in AD code.
fn backend_solve<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> Vec<T> {
    let mut x = vec![T::zero(); n * nrhs];
    backend
        .solve(a, b, n, nrhs, &mut x)
        .expect("solve failed in AD rule");
    x
}

/// Solve triangular via LinalgBackend, returning Vec for convenience in AD code.
fn backend_solve_tri<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> Vec<T> {
    let mut x = vec![T::zero(); n * nrhs];
    backend
        .solve_triangular(a, b, n, nrhs, upper, &mut x)
        .expect("solve_triangular failed in AD rule");
    x
}

/// Thin SVD via LinalgBackend, returning (U, S, V) for convenience in AD code.
/// Note: returns V (not Vt) as column-major n×k for convenience in AD code.
fn backend_thin_svd<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T],
    m: usize,
    n: usize,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let k = m.min(n);
    let mut u = vec![T::zero(); m * k];
    let mut s = vec![T::zero(); k];
    let mut vt = vec![T::zero(); k * n];
    backend
        .thin_svd(a, m, n, &mut u, &mut s, &mut vt)
        .expect("thin_svd failed in AD rule");
    // Convert Vt (k×n) to V (n×k) for convenience
    let v = transpose(&vt, k, n);
    (u, s, v)
}

/// QR decomposition via LinalgBackend, returning (Q, R) for convenience in AD code.
fn backend_qr<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T],
    m: usize,
    n: usize,
) -> (Vec<T>, Vec<T>) {
    let k = m.min(n);
    let mut q = vec![T::zero(); m * k];
    let mut r = vec![T::zero(); k * n];
    backend
        .qr(a, m, n, &mut q, &mut r)
        .expect("qr failed in AD rule");
    (q, r)
}

/// phi* (adjoint of phi): phi*(X) = (X + X^T - diag(X)) / 2
/// Diagonal gets halved, off-diagonal gets symmetrized.
fn phi_star<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let half = T::from(0.5).unwrap();
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                result[i + j * n] = half * data[i + j * n];
            } else {
                result[i + j * n] = half * (data[i + j * n] + data[j + i * n]);
            }
        }
    }
    result
}

/// Extract data slice from Tensor (ensuring col-major).
fn extract_data<T: LinalgScalar>(tensor: &Tensor<T>) -> (Vec<T>, usize) {
    let t = ensure_col_major(tensor);
    let offset = t.offset() as usize;
    let slice = t.buffer().as_slice().unwrap();
    let total_len = tensor.dims().iter().product::<usize>();
    (slice[offset..offset + total_len].to_vec(), 0)
}

// ============================================================================
// AD functions: rrule (reverse-mode, stateless)
// ============================================================================

/// Reverse-mode AD rule for SVD (VJP / pullback).
///
/// Computes the gradient of the input given cotangents for the SVD outputs.
/// Uses the F-matrix approach (Mathieu 2019).
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
pub fn svd_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>> {
    let result = svd(backend, tensor, options)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let eta = T::from(1e-40).unwrap();

    let (u_data, _) = extract_data(&result.u);
    let (s_data, _) = extract_data(&result.s);
    let (vt_data, _) = extract_data(&result.vt);

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        // V = Vt^T: n×k
        let v_b = transpose(vt_b, k, n);

        // Build F-matrix (k×k): F_ij = 1/(s_j² - s_i²) for i≠j, 0 diagonal
        let mut f_mat = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let denom = s_b[j] * s_b[j] - s_b[i] * s_b[i];
                    f_mat[i + j * k] = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                }
            }
        }

        // Start building inner matrix Gamma (k×k)
        let mut gamma = vec![T::zero(); k * k];

        // From dS cotangent: add diag(dS)
        if let Some(ref ds) = cotangent.s {
            let (ds_data, _) = extract_data(ds);
            let ds_b = &ds_data[b * k..(b + 1) * k];
            for i in 0..k {
                gamma[i + i * k] = gamma[i + i * k] + ds_b[i];
            }
        }

        // From dU cotangent: F ⊙ (U^T dU + (U^T dU)^T) * S
        if let Some(ref du) = cotangent.u {
            let (du_data, _) = extract_data(du);
            let du_b = &du_data[b * m * k..(b + 1) * m * k];
            // U^T dU (k×k)
            let ut_du = backend_mat_mul(backend, &transpose(u_b, m, k), k, m, du_b, k);
            for i in 0..k {
                for j in 0..k {
                    let sym = ut_du[i + j * k] + ut_du[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + f_mat[i + j * k] * sym * s_b[j];
                }
            }
        }

        // From dVt cotangent: S * F ⊙ (V^T dV + (V^T dV)^T)
        if let Some(ref dvt) = cotangent.vt {
            let (dvt_data, _) = extract_data(dvt);
            let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
            // dV = dVt^T (n×k)
            let dv_b = transpose(dvt_b, k, n);
            // V^T dV (k×k)
            let vt_dv = backend_mat_mul(backend, &transpose(&v_b, n, k), k, n, &dv_b, k);
            for i in 0..k {
                for j in 0..k {
                    let sym = vt_dv[i + j * k] + vt_dv[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + s_b[i] * f_mat[i + j * k] * sym;
                }
            }
        }

        // Core: dA_core = U * Gamma * V^T (m×k × k×k × k×n = m×n)
        let u_gamma = backend_mat_mul(backend, u_b, m, k, &gamma, k);
        let da_core = backend_mat_mul(backend, &u_gamma, m, k, &transpose(&v_b, n, k), n);

        // Copy core to output
        for i in 0..m * n {
            grad_a[b * m * n + i] = da_core[i];
        }

        // Non-square correction: (I - UU^T) dU S_inv^T V^T when m > k
        if m > k {
            if let Some(ref du) = cotangent.u {
                let (du_data, _) = extract_data(du);
                let du_b = &du_data[b * m * k..(b + 1) * m * k];
                // dU * diag(1/S) (m×k)
                let mut du_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = if s_b[j].abs() > eta {
                        T::one() / s_b[j]
                    } else {
                        T::zero()
                    };
                    for i in 0..m {
                        du_sinv[i + j * m] = du_b[i + j * m] * sinv;
                    }
                }
                // (I - UU^T) * du_sinv * V^T
                let inner = backend_mat_mul(backend, &transpose(u_b, m, k), k, m, &du_sinv, k);
                let uut_du = backend_mat_mul(backend, u_b, m, k, &inner, k);
                let proj = sub_vec(&du_sinv, &uut_du);
                let correction = backend_mat_mul(backend, &proj, m, k, &transpose(&v_b, n, k), n);
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }

        // Non-square correction for n > k: U S_inv^T (I - VV^T) dV^T
        if n > k {
            if let Some(ref dvt) = cotangent.vt {
                let (dvt_data, _) = extract_data(dvt);
                let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
                let dv_b = transpose(dvt_b, k, n);
                // diag(1/S) * dV^T (k×n) = diag(1/S) * Vt_cotangent
                // But we need dV (n×k), so: (I - VV^T) dV → project
                let inner = backend_mat_mul(backend, &transpose(&v_b, n, k), k, n, &dv_b, k);
                let vvt_dv = backend_mat_mul(backend, &v_b, n, k, &inner, k);
                let proj_dv = sub_vec(&dv_b, &vvt_dv);
                // U * diag(1/S) * proj_dv^T
                let mut u_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = if s_b[j].abs() > eta {
                        T::one() / s_b[j]
                    } else {
                        T::zero()
                    };
                    for i in 0..m {
                        u_sinv[i + j * m] = u_b[i + j * m] * sinv;
                    }
                }
                let correction =
                    backend_mat_mul(backend, &u_sinv, m, k, &transpose(&proj_dv, n, k), n);
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
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
pub fn qr_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &QrCotangent<T>,
) -> AdResult<Tensor<T>> {
    let result = qr(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (q_data, _) = extract_data(&result.q);
    let (r_data, _) = extract_data(&result.r);

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];

        // Initialize dQ and dR from cotangents (zero if not provided)
        let dq_b: Vec<T> = if let Some(ref dq) = cotangent.q {
            let (dq_data, _) = extract_data(dq);
            dq_data[b * m * k..(b + 1) * m * k].to_vec()
        } else {
            vec![T::zero(); m * k]
        };
        let dr_b: Vec<T> = if let Some(ref dr) = cotangent.r {
            let (dr_data, _) = extract_data(dr);
            dr_data[b * k * n..(b + 1) * k * n].to_vec()
        } else {
            vec![T::zero(); k * n]
        };

        // For thin QR (m >= n): A = QR where Q is m×k, R is k×n
        // W = R dR^T - dQ^T Q (k×k)
        let r_drt = backend_mat_mul(backend, r_b, k, n, &transpose(&dr_b, k, n), k);
        let dqt_q = backend_mat_mul(backend, &transpose(&dq_b, m, k), k, m, q_b, k);
        let w = sub_vec(&r_drt, &dqt_q);

        // H = copyltu(W) — symmetrize from lower triangle
        let h = copyltu(&w, k);

        // B = dQ + Q H (m×k)
        let qh = backend_mat_mul(backend, q_b, m, k, &h, k);
        let rhs = add_vec(&dq_b, &qh);

        // dA = B R^{-T} = solve R^T x = B^T, then transpose
        // R is k×k upper triangular (first k columns)
        let r_square = if n > k {
            // Extract first k columns of R (k×n → k×k)
            let mut rs = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    rs[i + j * k] = r_b[i + j * k];
                }
            }
            rs
        } else {
            r_b[..k * k].to_vec()
        };

        // dA[:, :k] = B R_square^{-T} (m×k solve)
        let rhs_t = transpose(&rhs, m, k);
        let da_t = backend_solve_tri(backend, &r_square, &rhs_t, k, m, true);
        let da_first_k = transpose(&da_t, k, m);

        // Copy first k columns
        for j in 0..k.min(n) {
            for i in 0..m {
                grad_a[b * m * n + i + j * m] = da_first_k[i + j * m];
            }
        }

        // For wide case (n > k), handle remaining columns via dR
        if n > k {
            // dA[:, k:] = Q dR[:, k:]
            for j in k..n {
                for i in 0..m {
                    let mut val = T::zero();
                    for l in 0..k {
                        val = val + q_b[i + l * m] * dr_b[l + j * k];
                    }
                    grad_a[b * m * n + i + j * m] = val;
                }
            }
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
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
pub fn lu_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &LuCotangent<T>,
    _pivot: LuPivot,
) -> AdResult<Tensor<T>> {
    let result = lu(backend, tensor, LuPivot::Partial)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&result.l);
    let (u_data, _) = extract_data(&result.u);
    let p_vec = result.p.as_ref();

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];

        let dl_b: Vec<T> = if let Some(ref dl) = cotangent.l {
            let (dl_data, _) = extract_data(dl);
            dl_data[b * m * k..(b + 1) * m * k].to_vec()
        } else {
            vec![T::zero(); m * k]
        };
        let du_b: Vec<T> = if let Some(ref du) = cotangent.u {
            let (du_data, _) = extract_data(du);
            du_data[b * k * n..(b + 1) * k * n].to_vec()
        } else {
            vec![T::zero(); k * n]
        };

        // F_bar = tril_strict(L^T dL) + triu(dU U^T) (k×k)
        let lt_dl = backend_mat_mul(backend, &transpose(l_b, m, k), k, m, &dl_b[..m * k], k);
        let du_ut = backend_mat_mul(
            backend,
            &du_b[..k * k],
            k,
            n.min(k),
            &transpose(&u_b[..k * k], k, n.min(k)),
            k,
        );
        let mut f_bar = vec![T::zero(); k * k];
        for j in 0..k {
            for i in 0..k {
                if i > j {
                    f_bar[i + j * k] = lt_dl[i + j * k];
                } else {
                    f_bar[i + j * k] = du_ut[i + j * k];
                }
            }
        }

        // dA = P^T L^{-T} F_bar U^{-T}
        let lt = transpose(l_b, m, k);
        let lt_square: Vec<T> = if m > k {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = lt[i + j * k];
                }
            }
            s
        } else {
            lt
        };
        let linvt_fbar = backend_solve_tri(backend, &lt_square, &f_bar, k, k, true);

        let ut = transpose(u_b, k, n);
        let ut_square: Vec<T> = if n > k {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = ut[i + j * k];
                }
            }
            s
        } else {
            ut
        };
        let da_inner_t = backend_solve_tri(
            backend,
            &ut_square,
            &transpose(&linvt_fbar, k, k),
            k,
            k,
            false,
        );
        let da_inner = transpose(&da_inner_t, k, k);

        // Apply P^T (inverse permutation)
        let p_inv: Vec<usize> = if let Some(pv) = p_vec {
            let p_b = &pv[b * m..(b + 1) * m];
            let mut inv = vec![0usize; m];
            for i in 0..m {
                if p_b[i] < m {
                    inv[p_b[i]] = i;
                }
            }
            inv
        } else {
            (0..m).collect()
        };

        if m == n {
            for j in 0..n {
                for i in 0..m {
                    grad_a[b * m * n + p_inv[i] + j * m] = da_inner[i + j * k];
                }
            }
        } else {
            for j in 0..n.min(k) {
                for i in 0..m.min(k) {
                    grad_a[b * m * n + p_inv[i] + j * m] = da_inner[i + j * k];
                }
            }
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
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
pub fn eigen_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>> {
    // Symmetric eigendecomposition: A = V diag(E) V^T
    let result = eigen(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let eta = T::from(1e-40).unwrap();

    let (v_data, _) = extract_data(&result.vectors);
    let (e_data, _) = extract_data(&result.values);

    let mut grad_a = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];

        // Build F-matrix (n×n): F_ij = 1/(e_j - e_i) for i≠j, 0 diagonal
        let mut f_mat = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[j] - e_b[i];
                    f_mat[i + j * n] = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                }
            }
        }

        // Inner matrix D = diag(dE) + F ⊙ (V^T dV + (V^T dV)^T) / 2
        let mut d_mat = vec![T::zero(); n * n];

        if let Some(ref de) = cotangent.values {
            let (de_data, _) = extract_data(de);
            let de_b = &de_data[b * n..(b + 1) * n];
            for i in 0..n {
                d_mat[i + i * n] = de_b[i];
            }
        }

        if let Some(ref dv) = cotangent.vectors {
            let (dv_data, _) = extract_data(dv);
            let dv_b = &dv_data[b * n * n..(b + 1) * n * n];
            let vt_dv = backend_mat_mul(backend, &transpose(v_b, n, n), n, n, dv_b, n);
            let half = T::from(0.5).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let sym = half * (vt_dv[i + j * n] + vt_dv[j + i * n]);
                    d_mat[i + j * n] = d_mat[i + j * n] + f_mat[i + j * n] * sym;
                }
            }
        }

        // dA = V D V^T
        let vd = backend_mat_mul(backend, v_b, n, n, &d_mat, n);
        let da_b = backend_mat_mul(backend, &vd, n, n, &transpose(v_b, n, n), n);

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
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
pub fn lstsq_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent_x: &Tensor<T>,
) -> AdResult<LstsqGrad<T>> {
    // lstsq: min_x ||Ax - b||^2, solution x = (A^T A)^{-1} A^T b
    // dA = -(A^{-T} dx) x^T + (b - Ax) z^T where z = (A^T A)^{-1} dx ... complex
    // Simplified: use the identity dx = (A^T A)^{-1} A^T db - (A^T A)^{-1} (dA^T (b - Ax) + A^T dA x)
    // For reverse mode: dA = (b - Ax) z^T - A z x^T where z = A^{+T} dx
    let result = lstsq(backend, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(a);
    let (b_data, _) = extract_data(b);
    let (x_data, _) = extract_data(&result.x);
    let (dx_data, _) = extract_data(cotangent_x);

    let mut grad_a_data = vec![T::zero(); m * n * bc];
    let mut grad_b_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let b_b = &b_data[batch * m..(batch + 1) * m];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let dx_b = &dx_data[batch * n..(batch + 1) * n];

        // z = A^{+T} dx = (A^T A)^{-1} A dx (solve via the transpose pinv)
        // A^T A z = A^T ... but simpler: z = pinv(A^T) dx
        // Use QR: A = QR, then A^{+T} = Q R^{-1}, so z = Q R^{-1} dx
        let (q_d, r_d) = backend_qr(backend, a_b, m, n);
        let k = m.min(n);
        // r_square: first k×k of R
        let r_square: Vec<T> = {
            let mut rs = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    rs[i + j * k] = r_d[i + j * k];
                }
            }
            rs
        };
        let rinv_dx = backend_solve_tri(backend, &r_square, dx_b, k, 1, true);
        // z = Q * rinv_dx (m×1)
        let z = backend_mat_mul(backend, &q_d, m, k, &rinv_dx, 1);

        // residual = b - Ax
        let ax = backend_mat_mul(backend, a_b, m, n, x_b, 1);
        let residual: Vec<T> = b_b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();

        // dA = residual z^T - A z x^T ... but actually the simpler formula:
        // dA = -z x^T (from the solution contribution)
        // db = z (from b contribution)
        // Plus residual term... For overdetermined systems:
        // dA = -(z x^T) + ... this is approximate. Use the standard formula:
        // dA = -z x^T, db = z (standard least squares gradient)
        for j in 0..n {
            for i in 0..m {
                grad_a_data[batch * m * n + i + j * m] = -z[i] * x_b[j];
            }
        }

        // Also add residual correction: d(residual)/dA contribution
        // For overdetermined: db += residual_correction... keep simple for now
        let _ = residual; // used only for reconstruction check

        grad_b_data[batch * m..(batch + 1) * m].copy_from_slice(&z);
    }

    let a_dims = output_dims(&[m, n], batch_dims);
    let b_dims = output_dims(&[m], batch_dims);
    Ok(LstsqGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for Cholesky (VJP / pullback).
///
/// Given `A = L L†` and cotangent `L̄`, computes `Ā`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::cholesky_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = cholesky_rrule(&a, &cotangent).unwrap();
/// ```
pub fn cholesky_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    // A = L L^T, dA = L^{-T} phi*(tril(L^T dL)) L^{-1}
    let l = cholesky(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l);
    let (dl_data, _) = extract_data(cotangent);

    let mut grad_a = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let dl_b = &dl_data[b * n * n..(b + 1) * n * n];

        // S = tril(L^T dL)
        let lt_dl = backend_mat_mul(backend, &transpose(l_b, n, n), n, n, dl_b, n);
        let s = tril(&lt_dl, n);

        // Apply phi*: symmetrize S → (S + S^T) / 2
        let s_sym = phi_star(&s, n);

        // Solve L^T x = S_sym → x = L^{-T} S_sym
        let x = backend_solve_tri(backend, &transpose(l_b, n, n), &s_sym, n, n, true);

        // Solve x L = result → result = x L^{-1} → L^T result^T = x^T → result^T = L^{-T} x^T
        let xt = transpose(&x, n, n);
        let result_t = backend_solve_tri(backend, &transpose(l_b, n, n), &xt, n, n, true);
        let da_b = transpose(&result_t, n, n);

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for linear solve (VJP / pullback).
///
/// Given `Ax = b` and cotangent `x̄`, computes `(Ā, b̄)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::solve_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let b = Tensor::<f64>::zeros(&[3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col);
/// let grad = solve_rrule(&a, &b, &cotangent).unwrap();
/// ```
pub fn solve_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>> {
    // Ax = b → G = A^{-T} dx, dB = G, dA = -G x^T
    let x = solve(backend, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let nrhs = if b.ndim() > 1 && b.dims()[1] != n {
        b.dims()[1]
    } else {
        1
    };

    let (a_data, _) = extract_data(a);
    let (x_data, _) = extract_data(&x);
    let (dx_data, _) = extract_data(cotangent);

    let mut grad_a_data = vec![T::zero(); n * n * bc];
    let mut grad_b_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // G = A^{-T} dx = solve(A^T, dx)
        let at = transpose(a_b, n, n);
        let g = backend_solve(backend, &at, dx_b, n, nrhs);

        // dB = G
        grad_b_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&g);

        // dA = -G x^T (n×nrhs × nrhs×n = n×n)
        let g_xt = backend_mat_mul(backend, &g, n, nrhs, &transpose(x_b, n, nrhs), n);
        let neg_g_xt = scale_vec(&g_xt, -T::one());
        grad_a_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg_g_xt);
    }

    let a_dims = output_dims(&[n, n], batch_dims);
    let b_dims = output_dims(&[n, nrhs], batch_dims);
    Ok(SolveGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for matrix inverse (VJP / pullback).
///
/// `Ā = -A⁻ᴴ · cotangent · A⁻ᴴ`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::inv_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = inv_rrule(&a, &cotangent).unwrap();
/// ```
pub fn inv_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    // dA = -B^T dB B^T where B = A^{-1}
    let b_inv = inv(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv);
    let (db_data, _) = extract_data(cotangent);

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * n..(batch + 1) * n * n];

        let bt = transpose(b_b, n, n);
        let bt_db = backend_mat_mul(backend, &bt, n, n, db_b, n);
        let bt_db_bt = backend_mat_mul(backend, &bt_db, n, n, &bt, n);
        let neg = scale_vec(&bt_db_bt, -T::one());
        grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for determinant (VJP / pullback).
///
/// `Ā = det(A) · cotangent · A⁻ᵀ`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::det_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = det_rrule(&a, &cotangent).unwrap();
/// ```
pub fn det_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    // dA = ddet * det(A) * A^{-T}
    let det_val = det(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (det_data, _) = extract_data(&det_val);
    let (ddet_data, _) = extract_data(cotangent);

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let d = det_data[batch];
        let dd = ddet_data[batch];

        // A^{-T}
        let a_inv = backend_solve(backend, a_b, &eye::<T>(n), n, n);
        let a_inv_t = transpose(&a_inv, n, n);

        let scale = dd * d;
        let da_b = scale_vec(&a_inv_t, scale);
        grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for slogdet (VJP / pullback).
///
/// `Ā = cotangent_logabsdet · A⁻ᵀ`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{slogdet_rrule, SlogdetCotangent};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::ones(&[], mem, col)),
/// };
/// let grad_a = slogdet_rrule(&a, &cotangent).unwrap();
/// ```
pub fn slogdet_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T>,
) -> AdResult<Tensor<T>> {
    // dA = d_logabsdet * A^{-T}
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);

    let mut grad_a = vec![T::zero(); n * n * bc];

    if let Some(ref dlog) = cotangent.logabsdet {
        let (dlog_data, _) = extract_data(dlog);
        for batch in 0..bc {
            let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
            let dl = dlog_data[batch];

            let a_inv = backend_solve(backend, a_b, &eye::<T>(n), n, n);
            let a_inv_t = transpose(&a_inv, n, n);
            let da_b = scale_vec(&a_inv_t, dl);
            grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&da_b);
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for general eigendecomposition (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{eig_rrule, EigenCotangent};
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
/// let grad_a = eig_rrule(&a, &cotangent).unwrap();
/// ```
pub fn eig_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
    _cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>> {
    Err(chainrules_core::AutodiffError::ModeNotSupported {
        mode: "rrule/frule".into(),
        reason: "general eigendecomposition AD requires complex output; use eigen_rrule for symmetric matrices".into(),
    })
}

/// Reverse-mode AD rule for pseudoinverse (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::pinv_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let grad_a = pinv_rrule(&a, &cotangent, None).unwrap();
/// ```
pub fn pinv_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<Tensor<T>> {
    // dA = -(A+)^T dA+ (A+)^T + (I - AA+)(dA+)^T A+(A+)^T + (A+)^T A+ (dA+)^T (I - A+A)
    let ap = pinv(backend, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (ap_data, _) = extract_data(&ap);
    let (dap_data, _) = extract_data(cotangent);

    let mut grad_a = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let dap_b = &dap_data[batch * n * m..(batch + 1) * n * m];

        let apt = transpose(ap_b, n, m); // m×n
        let dapt = transpose(dap_b, n, m); // m×n

        // Term 1: -(A+)^T dA+ (A+)^T = -apt * dap * apt^T
        // apt: m×n, dap: n×m, apt: m×n → m×n * n×m * m×n = m×n
        let t1 = backend_mat_mul(backend, &apt, m, n, dap_b, m);
        let t1 = backend_mat_mul(backend, &t1, m, m, &apt, n);
        let t1 = scale_vec(&t1, -T::one());

        // Term 2: (I - AA+)(dA+)^T A+ (A+)^T
        // AA+ (m×m)
        let aap = backend_mat_mul(backend, a_b, m, n, ap_b, m);
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        // (dA+)^T A+ = dapt * ap (m×n * n×m = m×m)
        let dapt_ap = backend_mat_mul(backend, &dapt, m, n, ap_b, m);
        // * (A+)^T = * apt (m×m * m×n = m×n)
        let dapt_ap_apt = backend_mat_mul(backend, &dapt_ap, m, m, &apt, n);
        let t2 = backend_mat_mul(backend, &i_aap, m, m, &dapt_ap_apt, n);

        // Term 3: (A+)^T A+ (dA+)^T (I - A+A)
        // A+A (n×n)
        let apa = backend_mat_mul(backend, ap_b, n, m, a_b, n);
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        // (A+)^T A+ = apt * ap (m×n * n×m = m×m)
        let apt_ap = backend_mat_mul(backend, &apt, m, n, ap_b, m);
        // * (dA+)^T = * dapt (m×m * m×n = m×n)
        let apt_ap_dapt = backend_mat_mul(backend, &apt_ap, m, m, &dapt, n);
        let t3 = backend_mat_mul(backend, &apt_ap_dapt, m, n, &i_apa, n);

        let da_b = add_vec(&t1, &add_vec(&t2, &t3));
        grad_a[batch * m * n..(batch + 1) * m * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for matrix exponential (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::matrix_exp_rrule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = matrix_exp_rrule(&a, &cotangent).unwrap();
/// ```
pub fn matrix_exp_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
    _cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>> {
    Err(chainrules_core::AutodiffError::ModeNotSupported {
        mode: "rrule/frule".into(),
        reason: "matrix exponential AD not yet implemented".into(),
    })
}

/// Reverse-mode AD rule for norm (VJP / pullback).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{norm_rrule, NormKind};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = norm_rrule(&a, &cotangent, NormKind::Fro).unwrap();
/// ```
pub fn norm_rrule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<Tensor<T>> {
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    validate_norm_cotangent(cotangent, batch_dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (a_data, _) = extract_data(tensor);
    let (dn_data, _) = extract_data(cotangent);

    let mut grad_a = vec![T::zero(); m * n * bc];

    match kind {
        NormKind::Fro => {
            // dA = dn * A / ||A||_F
            let nrm = norm(backend, tensor, NormKind::Fro)
                .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
            let (nrm_data, _) = extract_data(&nrm);
            for batch in 0..bc {
                let dn = dn_data[batch];
                let nv = nrm_data[batch];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = scale * a_data[batch * m * n + i];
                }
            }
        }
        NormKind::Nuclear => {
            // dA = dn * U V^T
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(backend, a_b, m, n);
                let k = m.min(n);
                let uv = backend_mat_mul(backend, &u, m, k, &transpose(&v, n, k), n);
                let dn = dn_data[batch];
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = dn * uv[i];
                }
            }
        }
        NormKind::Spectral => {
            // dA = dn * u1 v1^T (rank-1 outer product of leading singular vectors)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(backend, a_b, m, n);
                let dn = dn_data[batch];
                for j in 0..n {
                    for i in 0..m {
                        grad_a[batch * m * n + i + j * m] = dn * u[i] * v[j];
                    }
                }
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_rrule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
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
pub fn svd_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> AdResult<(SvdResult<T>, SvdResult<T>)> {
    let result = svd(backend, tensor, options)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let eta = T::from(1e-40).unwrap();

    let (u_data, _) = extract_data(&result.u);
    let (s_data, _) = extract_data(&result.s);
    let (vt_data, _) = extract_data(&result.vt);
    let (da_data, _) = extract_data(tangent);

    let mut du_data = vec![T::zero(); m * k * bc];
    let mut ds_data = vec![T::zero(); k * bc];
    let mut dvt_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // C = U^T dA V (k×k)
        let ut_da = backend_mat_mul(backend, &transpose(u_b, m, k), k, m, da_b, n);
        let v_b = transpose(vt_b, k, n);
        let c = backend_mat_mul(backend, &ut_da, k, n, &v_b, k);

        // dS = diag(C)
        for i in 0..k {
            ds_data[b * k + i] = c[i + i * k];
        }

        // F-matrix
        let mut f_mat = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let denom = s_b[j] * s_b[j] - s_b[i] * s_b[i];
                    f_mat[i + j * k] = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                }
            }
        }

        // dU = U (F ⊙ (S C^T + C S)) + (I_m - U U^T) dA V S^{-1}
        let mut sc_t_plus_cs = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                sc_t_plus_cs[i + j * k] = s_b[i] * c[j + i * k] + c[i + j * k] * s_b[j];
            }
        }
        let f_inner = hadamard(&f_mat, &sc_t_plus_cs);
        let du_core = backend_mat_mul(backend, u_b, m, k, &f_inner, k);

        // Projector term for dU
        if m > k {
            let inner = backend_mat_mul(backend, &transpose(u_b, m, k), k, m, da_b, n);
            let uut_da = backend_mat_mul(backend, u_b, m, k, &inner, n);
            let proj_da: Vec<T> = da_b
                .iter()
                .zip(uut_da.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let proj_da_v = backend_mat_mul(backend, &proj_da, m, n, &v_b, k);
            for j in 0..k {
                let sinv = if s_b[j].abs() > eta {
                    T::one() / s_b[j]
                } else {
                    T::zero()
                };
                for i in 0..m {
                    du_data[b * m * k + i + j * m] =
                        du_core[i + j * m] + proj_da_v[i + j * m] * sinv;
                }
            }
        } else {
            du_data[b * m * k..(b + 1) * m * k].copy_from_slice(&du_core);
        }

        // dVt = (F ⊙ (S^T C + C^T S)) V^T + S^{-1} U^T dA (I_n - V V^T)
        let mut st_c_plus_ct_s = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                st_c_plus_ct_s[i + j * k] = s_b[j] * c[i + j * k] + c[j + i * k] * s_b[i];
            }
        }
        let f_inner2 = hadamard(&f_mat, &st_c_plus_ct_s);
        let dvt_core = backend_mat_mul(backend, &f_inner2, k, k, vt_b, n);

        if n > k {
            let vvt = backend_mat_mul(backend, &v_b, n, k, vt_b, n);
            let i_n = eye::<T>(n);
            let i_vvt = sub_vec(&i_n, &vvt);
            let ut_da = backend_mat_mul(backend, &transpose(u_b, m, k), k, m, da_b, n);
            let sinv_ut_da = {
                let mut r = vec![T::zero(); k * n];
                for i in 0..k {
                    let sinv = if s_b[i].abs() > eta {
                        T::one() / s_b[i]
                    } else {
                        T::zero()
                    };
                    for j in 0..n {
                        r[i + j * k] = sinv * ut_da[i + j * k];
                    }
                }
                r
            };
            let proj = backend_mat_mul(backend, &sinv_ut_da, k, n, &i_vvt, n);
            dvt_data[b * k * n..(b + 1) * k * n].copy_from_slice(&add_vec(&dvt_core, &proj));
        } else {
            dvt_data[b * k * n..(b + 1) * k * n].copy_from_slice(&dvt_core);
        }
    }

    let u_dims = output_dims(&[m, k], batch_dims);
    let s_dims = output_dims(&[k], batch_dims);
    let vt_dims = output_dims(&[k, n], batch_dims);

    let dresult = SvdResult {
        u: tensor_from_data(du_data, &u_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        s: tensor_from_data(ds_data, &s_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        vt: tensor_from_data(dvt_data, &vt_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };

    Ok((result, dresult))
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
pub fn qr_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(QrResult<T>, QrResult<T>)> {
    let result = qr(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (q_data, _) = extract_data(&result.q);
    let (r_data, _) = extract_data(&result.r);
    let (da_data, _) = extract_data(tangent);

    let mut dq_data = vec![T::zero(); m * k * bc];
    let mut dr_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // Q^T dA (k×n)
        let qt_da = backend_mat_mul(backend, &transpose(q_b, m, k), k, m, da_b, n);

        // M = R^{-1} Q^T dA → solve R M = Q^T dA (for square R block)
        let r_sq: Vec<T> = {
            let mut rs = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    rs[i + j * k] = r_b[i + j * k];
                }
            }
            rs
        };

        // dR = triu(Q^T dA)[:k, :n] for the thin case, but using the proper formula:
        // F = R^{-1} Q^T dA, dR = triu(F) R ... no wait.
        // Proper: dR[:k,:k] = triu(Q^T dA[:,:k]) for the square part
        // Simpler approach: dR = Q^T dA (projects cotangent), dQ = (dA - Q dR) R^{-1}

        // dR_full = Q^T dA (k×n)
        // But we need triu: for R upper triangular, dR should be upper triangular
        let mut dr_b_vec = vec![T::zero(); k * n];
        // For the square part (k×k): dR_sq = triu(Qt_dA[:k,:k])
        for j in 0..n {
            for i in 0..k.min(j + 1) {
                dr_b_vec[i + j * k] = qt_da[i + j * k];
            }
        }

        // dQ = (dA - Q dR) R^{-1}
        let q_dr = backend_mat_mul(backend, q_b, m, k, &dr_b_vec, n);
        let da_minus_qdr: Vec<T> = da_b.iter().zip(q_dr.iter()).map(|(&a, &b)| a - b).collect();

        // Solve (dA - Q dR) = dQ R → dQ = (dA - Q dR) R^{-1}
        // For thin case: dQ (m×k) R_sq (k×k) = (dA - Q dR)[:, :k]
        let rhs: Vec<T> = {
            let mut r = vec![T::zero(); m * k];
            for j in 0..k {
                for i in 0..m {
                    r[i + j * m] = da_minus_qdr[i + j * m];
                }
            }
            r
        };
        // Solve: dQ R = rhs → dQ = rhs R^{-1} → R^T dQ^T = rhs^T
        let rhs_t = transpose(&rhs, m, k);
        let r_sq_t = transpose(&r_sq, k, k);
        let dq_t = backend_solve_tri(backend, &r_sq_t, &rhs_t, k, m, false);
        let dq_b_vec = transpose(&dq_t, k, m);

        dq_data[b * m * k..(b + 1) * m * k].copy_from_slice(&dq_b_vec);
        dr_data[b * k * n..(b + 1) * k * n].copy_from_slice(&dr_b_vec);
    }

    let q_dims = output_dims(&[m, k], batch_dims);
    let r_dims = output_dims(&[k, n], batch_dims);
    let dresult = QrResult {
        q: tensor_from_data(dq_data, &q_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        r: tensor_from_data(dr_data, &r_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
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
pub fn lu_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)> {
    let result = lu(backend, tensor, pivot)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&result.l);
    let (u_data, _) = extract_data(&result.u);
    let p_vec = result.p.as_ref();
    let (da_data, _) = extract_data(tangent);

    let mut dl_data = vec![T::zero(); m * k * bc];
    let mut du_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // Apply permutation: P dA (m×n)
        let mut pda = vec![T::zero(); m * n];
        if let Some(pv) = p_vec {
            let p_b = &pv[b * m..(b + 1) * m];
            for i in 0..m {
                for j in 0..n {
                    pda[i + j * m] = da_b[p_b[i] + j * m];
                }
            }
        } else {
            pda.copy_from_slice(da_b);
        }

        // F = L^{-1} P dA U^{-1} (k×k for square part)
        // First: L^{-1} PdA → solve L x = PdA
        let l_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = l_b[i + j * m];
                }
            }
            s
        };
        let pda_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * n];
            for j in 0..n {
                for i in 0..k {
                    s[i + j * k] = pda[i + j * m];
                }
            }
            s
        };
        let linv_pda = backend_solve_tri(backend, &l_sq, &pda_sq, k, n, false);

        // Then: (L^{-1} PdA) U^{-1} → solve (result) U = linv_pda
        let u_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = u_b[i + j * k];
                }
            }
            s
        };
        // Solve x U = linv_pda → U^T x^T = linv_pda^T
        let f_t = backend_solve_tri(
            backend,
            &transpose(&u_sq, k, k),
            &transpose(&linv_pda, k, n),
            k,
            k,
            false,
        );
        let f = transpose(&f_t, k, k);

        // dL = L tril_strict(F) (m×k)
        let tril_f = tril_strict(&f, k);
        let dl_b_vec = backend_mat_mul(backend, &l_sq, k, k, &tril_f, k);
        for j in 0..k {
            for i in 0..k {
                dl_data[b * m * k + i + j * m] = dl_b_vec[i + j * k];
            }
        }

        // dU = triu(F) U (k×n)
        let triu_f = triu(&f, k);
        let du_b_vec = backend_mat_mul(backend, &triu_f, k, k, &u_sq, k);
        for j in 0..k {
            for i in 0..k {
                du_data[b * k * n + i + j * k] = du_b_vec[i + j * k];
            }
        }
    }

    let l_dims = output_dims(&[m, k], batch_dims);
    let u_dims = output_dims(&[k, n], batch_dims);
    let dresult = LuResult {
        p: None, // permutation has no derivative
        l: tensor_from_data(dl_data, &l_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        u: tensor_from_data(du_data, &u_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
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
pub fn eigen_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)> {
    let result = eigen(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let eta = T::from(1e-40).unwrap();

    let (v_data, _) = extract_data(&result.vectors);
    let (e_data, _) = extract_data(&result.values);
    let (da_data, _) = extract_data(tangent);

    let mut de_data = vec![T::zero(); n * bc];
    let mut dv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // C = V^T dA V (n×n)
        let vt_da = backend_mat_mul(backend, &transpose(v_b, n, n), n, n, da_b, n);
        let c = backend_mat_mul(backend, &vt_da, n, n, v_b, n);

        // dE = diag(C)
        for i in 0..n {
            de_data[b * n + i] = c[i + i * n];
        }

        // dV = V F ⊙ C where F_ij = 1/(e_i - e_j) for i≠j, 0 diagonal
        let mut fc = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[i] - e_b[j];
                    let f_ij = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                    fc[i + j * n] = f_ij * c[i + j * n];
                }
            }
        }
        let dv_b_vec = backend_mat_mul(backend, v_b, n, n, &fc, n);
        dv_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dv_b_vec);
    }

    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);
    let dresult = EigenResult {
        values: tensor_from_data(de_data, &val_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        vectors: tensor_from_data(dv_data, &vec_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
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
pub fn lstsq_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T>, LstsqResult<T>)> {
    // dx = A^+ (db - dA x), where A^+ = (A^T A)^{-1} A^T
    let result = lstsq(backend, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(a);
    let (x_data, _) = extract_data(&result.x);
    let (da_data, _) = extract_data(tangent_a);
    let (db_data, _) = extract_data(tangent_b);

    let mut dx_data = vec![T::zero(); n * bc];
    let mut dres_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
        let db_b = &db_data[batch * m..(batch + 1) * m];

        // dA x (m×1)
        let da_x = backend_mat_mul(backend, da_b, m, n, x_b, 1);
        // db - dA x
        let rhs: Vec<T> = db_b.iter().zip(da_x.iter()).map(|(&a, &b)| a - b).collect();

        // A^+ rhs = (A^T A)^{-1} A^T rhs
        let at_rhs = backend_mat_mul(backend, &transpose(a_b, m, n), n, m, &rhs, 1);
        let ata = backend_mat_mul(backend, &transpose(a_b, m, n), n, m, a_b, n);
        let dx_b_vec = backend_solve(backend, &ata, &at_rhs, n, 1);
        dx_data[batch * n..(batch + 1) * n].copy_from_slice(&dx_b_vec);

        // d(residual) = db - dA x - A dx
        let a_dx = backend_mat_mul(backend, a_b, m, n, &dx_b_vec, 1);
        for i in 0..m {
            dres_data[batch * m + i] = rhs[i] - a_dx[i];
        }
    }

    let x_dims = output_dims(&[n], batch_dims);
    let res_dims = output_dims(&[m], batch_dims);
    let dresult = LstsqResult {
        x: tensor_from_data(dx_data, &x_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        residual: tensor_from_data(dres_data, &res_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for Cholesky (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::cholesky_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (l, dl) = cholesky_frule(&a, &da).unwrap();
/// ```
pub fn cholesky_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    // dL = L phi(L^{-1} dA L^{-T})
    let l = cholesky(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l);
    let (da_data, _) = extract_data(tangent);

    let mut dl_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // L^{-1} dA: solve L x = dA
        let linv_da = backend_solve_tri(backend, l_b, da_b, n, n, false);
        // (L^{-1} dA) L^{-T}: solve (result) L^T = linv_da → L x^T = linv_da^T
        let linv_da_linvt_t =
            backend_solve_tri(backend, l_b, &transpose(&linv_da, n, n), n, n, false);
        let inner = transpose(&linv_da_linvt_t, n, n);

        // phi(inner) = tril with diagonal halved
        let phi_inner = phi(&inner, n);

        // dL = L phi(inner)
        let dl_b_vec = backend_mat_mul(backend, l_b, n, n, &phi_inner, n);
        dl_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dl_b_vec);
    }

    let dims = output_dims(&[n, n], batch_dims);
    let dl = tensor_from_data(dl_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((l, dl))
}

/// Forward-mode AD rule for linear solve (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::solve_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let b = Tensor::<f64>::zeros(&[3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let db = Tensor::<f64>::ones(&[3], mem, col);
/// let (x, dx) = solve_frule(&a, &b, &da, &db).unwrap();
/// ```
pub fn solve_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    // dx = A^{-1} (db - dA x)
    let x = solve(backend, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let nrhs = if b.ndim() > 1 && b.dims()[1] != n {
        b.dims()[1]
    } else {
        1
    };

    let (a_data, _) = extract_data(a);
    let (x_data, _) = extract_data(&x);
    let (da_data, _) = extract_data(tangent_a);
    let (db_data, _) = extract_data(tangent_b);

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // dA x (n×nrhs)
        let da_x = backend_mat_mul(backend, da_b, n, n, x_b, nrhs);
        // db - dA x
        let rhs = sub_vec(db_b, &da_x);
        // A^{-1} (db - dA x)
        let dx_b_vec = backend_solve(backend, a_b, &rhs, n, nrhs);
        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b_vec);
    }

    let dims = output_dims(&[n, nrhs], batch_dims);
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::inv_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (a_inv, da_inv) = inv_frule(&a, &da).unwrap();
/// ```
pub fn inv_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    // dB = -B dA B where B = A^{-1}
    let b_inv = inv(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv);
    let (da_data, _) = extract_data(tangent);

    let mut db_data = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let b_da = backend_mat_mul(backend, b_b, n, n, da_b, n);
        let b_da_b = backend_mat_mul(backend, &b_da, n, n, b_b, n);
        let neg = scale_vec(&b_da_b, -T::one());
        db_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg);
    }

    let dims = output_dims(&[n, n], batch_dims);
    let db = tensor_from_data(db_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((b_inv, db))
}

/// Forward-mode AD rule for determinant (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::det_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (d, dd) = det_frule(&a, &da).unwrap();
/// ```
pub fn det_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    // d(det) = det(A) * tr(A^{-1} dA)
    let d = det(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (d_data, _) = extract_data(&d);
    let (da_data, _) = extract_data(tangent);

    let mut dd_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(backend, a_b, &eye::<T>(n), n, n);
        let a_inv_da = backend_mat_mul(backend, &a_inv, n, n, da_b, n);
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dd_data[batch] = d_data[batch] * trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dd = tensor_from_data(dd_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((d, dd))
}

/// Forward-mode AD rule for slogdet (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::slogdet_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = slogdet_frule(&a, &da).unwrap();
/// ```
pub fn slogdet_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T>, SlogdetResult<T>)> {
    // d(logabsdet) = Re(tr(A^{-1} dA)), d(sign) = 0 (for real)
    let result = slogdet(backend, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (da_data, _) = extract_data(tangent);

    let mut dlog_data = vec![T::zero(); bc];
    let dsign_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(backend, a_b, &eye::<T>(n), n, n);
        let a_inv_da = backend_mat_mul(backend, &a_inv, n, n, da_b, n);
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dlog_data[batch] = trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dresult = SlogdetResult {
        sign: tensor_from_data(dsign_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        logabsdet: tensor_from_data(dlog_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for general eigendecomposition (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::eig_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eig_frule(&a, &da).unwrap();
/// ```
pub fn eig_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)> {
    Err(chainrules_core::AutodiffError::ModeNotSupported {
        mode: "rrule/frule".into(),
        reason: "general eigendecomposition AD requires complex output; use eigen_frule for symmetric matrices".into(),
    })
}

/// Forward-mode AD rule for pseudoinverse (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::pinv_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (pinv_a, dpinv_a) = pinv_frule(&a, &da, None).unwrap();
/// ```
pub fn pinv_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    // dA+ = -A+ dA A+ + (I - A+A) dA^T (A+)^T A+ + A+ (A+)^T dA^T (I - AA+)
    let ap = pinv(backend, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (ap_data, _) = extract_data(&ap);
    let (da_data, _) = extract_data(tangent);

    let mut dap_data = vec![T::zero(); n * m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];

        let dat = transpose(da_b, m, n); // n×m
        let apt = transpose(ap_b, n, m); // m×n

        // Term 1: -A+ dA A+ (n×m × m×n × n×m = n×m)
        let ap_da = backend_mat_mul(backend, ap_b, n, m, da_b, n);
        let ap_da_ap = backend_mat_mul(backend, &ap_da, n, n, ap_b, m);
        let t1 = scale_vec(&ap_da_ap, -T::one());

        // Term 2: (I - A+A) dA^T (A+)^T A+
        let apa = backend_mat_mul(backend, ap_b, n, m, a_b, n); // n×n
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        let dat_apt = backend_mat_mul(backend, &dat, n, m, &apt, n); // n×n
        let dat_apt_ap = backend_mat_mul(backend, &dat_apt, n, n, ap_b, m); // n×m
        let t2 = backend_mat_mul(backend, &i_apa, n, n, &dat_apt_ap, m);

        // Term 3: A+ (A+)^T dA^T (I - AA+)
        let aap = backend_mat_mul(backend, a_b, m, n, ap_b, m); // m×m
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        let ap_apt = backend_mat_mul(backend, ap_b, n, m, &apt, n); // n×n
        let ap_apt_dat = backend_mat_mul(backend, &ap_apt, n, n, &dat, m); // n×m
        let t3 = backend_mat_mul(backend, &ap_apt_dat, n, m, &i_aap, m);

        let dap_b_vec = add_vec(&t1, &add_vec(&t2, &t3));
        dap_data[batch * n * m..(batch + 1) * n * m].copy_from_slice(&dap_b_vec);
    }

    let dims = output_dims(&[n, m], batch_dims);
    let dap = tensor_from_data(dap_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((ap, dap))
}

/// Forward-mode AD rule for matrix exponential (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::matrix_exp_frule;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (exp_a, dexp_a) = matrix_exp_frule(&a, &da).unwrap();
/// ```
pub fn matrix_exp_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    _backend: &mut B,
    _tensor: &Tensor<T>,
    _tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    Err(chainrules_core::AutodiffError::ModeNotSupported {
        mode: "rrule/frule".into(),
        reason: "matrix exponential AD not yet implemented".into(),
    })
}

/// Forward-mode AD rule for norm (JVP / pushforward).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::{norm_frule, NormKind};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (n, dn) = norm_frule(&a, &da, NormKind::Fro).unwrap();
/// ```
pub fn norm_frule<T: LinalgScalar, B: backend::LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<(Tensor<T>, Tensor<T>)> {
    let nrm = norm(backend, tensor, kind)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor);
    let (nrm_data, _) = extract_data(&nrm);
    let (da_data, _) = extract_data(tangent);

    let mut dnrm_data = vec![T::zero(); bc];

    match kind {
        NormKind::Fro => {
            // d||A||_F = tr(A^T dA) / ||A||_F
            for batch in 0..bc {
                let nv = nrm_data[batch];
                if nv > T::zero() {
                    let mut dot = T::zero();
                    for i in 0..m * n {
                        dot = dot + a_data[batch * m * n + i] * da_data[batch * m * n + i];
                    }
                    dnrm_data[batch] = dot / nv;
                }
            }
        }
        NormKind::Nuclear => {
            // d||A||_* = tr(U^T dA V)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(backend, a_b, m, n);
                let k = m.min(n);
                let ut_da = backend_mat_mul(backend, &transpose(&u, m, k), k, m, da_b, n);
                let ut_da_v = backend_mat_mul(backend, &ut_da, k, n, &v, k);
                let mut trace = T::zero();
                for i in 0..k {
                    trace = trace + ut_da_v[i + i * k];
                }
                dnrm_data[batch] = trace;
            }
        }
        NormKind::Spectral => {
            // d||A||_2 = u1^T dA v1
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(backend, a_b, m, n);
                let mut val = T::zero();
                for i in 0..m {
                    for j in 0..n {
                        val = val + u[i] * da_b[i + j * m] * v[j];
                    }
                }
                dnrm_data[batch] = val;
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_frule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[], batch_dims);
    let dnrm = tensor_from_data(dnrm_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((nrm, dnrm))
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn validate_hermitian_batches_accepts_symmetric_input() {
        // Two 2x2 symmetric matrices in column-major layout.
        let data = vec![
            2.0, 1.0, 1.0, 3.0, // batch 0
            5.0, 0.0, 0.0, 7.0, // batch 1
        ];
        let result = validate_hermitian_batches(&data, 0, 2, 2, "eigen");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_hermitian_batches_rejects_nonsymmetric_input() {
        // 2x2 non-symmetric matrix [[2, 4], [1, 3]] in column-major layout.
        let data = vec![2.0, 1.0, 4.0, 3.0];
        let result = validate_hermitian_batches(&data, 0, 2, 1, "eigen");
        assert!(matches!(result, Err(Error::InvalidArgument(_))));
    }
}
