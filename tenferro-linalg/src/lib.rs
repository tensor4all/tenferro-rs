#![allow(clippy::multiple_bound_locations)]

//! Batched matrix linear algebra decompositions with AD rules.
//!
//! CPU decompositions and solvers are fully implemented via the
//! [`faer`](https://crates.io/crates/faer) backend. GPU backends are planned.
//!
//! This crate provides SVD, QR, LU, eigendecomposition, Cholesky, least squares,
//! linear solve, matrix inverse, determinant, pseudoinverse, matrix exponential,
//! triangular solve, and norms for tensors
//! with shape `(m, n, *)`, adapted from PyTorch's `torch.linalg` for
//! column-major layout:
//!
//! - **First 2 dimensions** are the matrix (`m × n`).
//! - **All following dimensions** (`*`) are independent batch dimensions.
//! - Inputs are **internally normalized** to column-major contiguous layout.
//!   If an input is not already contiguous, an internal copy is performed.
//!   Calling `.contiguous(ColumnMajor)` explicitly is optional but useful
//!   when you want to control exactly where copies happen.
//!
//! This convention mirrors PyTorch's `(*, m, n)` but is flipped for
//! col-major: in col-major the first dimensions are contiguous, so
//! placing the matrix there ensures LAPACK can operate directly without
//! transposition.
//!
//! This module is **context-agnostic**: it does not know about tensor
//! networks, MPS, or any specific application. If you need to decompose
//! a tensor along arbitrary legs, `permute` + `reshape` before calling
//! these functions.
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
//! ```
//! use tenferro_linalg::{svd, SvdOptions};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // 2D matrix: shape [3, 4]
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//! // result.u:  shape [3, 3]  (m × k, k = min(m,n) = 3)
//! // result.s:  shape [3]     (singular values)
//! // result.vt: shape [3, 4]  (k × n)
//! ```
//!
//! ## Batched SVD
//!
//! ```
//! use tenferro_linalg::svd;
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // Batched: shape [m, n, batch] = [3, 4, 10]
//! let a = Tensor::<f64>::zeros(&[3, 4, 10], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//! // result.u:  shape [3, 3, 10]
//! // result.s:  shape [3, 10]
//! // result.vt: shape [3, 4, 10]
//! ```
//!
//! ## Decomposing a 4D tensor along specific legs
//!
//! ```
//! use tenferro_linalg::svd;
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // 4D tensor [2, 3, 4, 5] — want SVD with left=[0,1], right=[2,3]
//! let t = Tensor::<f64>::zeros(&[2, 3, 4, 5], mem, col);
//!
//! // permute + reshape (contiguous is handled internally, but can be called explicitly)
//! let mat = t.permute(&[0, 1, 2, 3]).unwrap()  // already in order
//!            .reshape(&[6, 20]).unwrap();        // m = 2*3 = 6, n = 4*5 = 20
//! let result = svd(&mut ctx, &mat, None).unwrap();
//! // Then reshape result.u, result.vt back to desired tensor shape
//! ```
//!
//! ## Reverse-mode AD (stateless rrule)
//!
//! ```
//! use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//!
//! // Full cotangent: gradient through U, S, and Vt
//! let cotangent = SvdCotangent {
//!     u: Some(Tensor::ones(&[3, 3], mem, col)),
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: Some(Tensor::ones(&[3, 4], mem, col)),
//! };
//! let grad_a = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
//! // grad_a has same shape as a: [3, 4]
//!
//! // Partial cotangent: gradient only through singular values (always stable)
//! let cotangent_s_only = SvdCotangent {
//!     u: None,
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: None,
//! };
//! let grad_a2 = svd_rrule(&mut ctx, &a, &cotangent_s_only, None).unwrap();
//! ```

#[cfg(all(feature = "provider-src", not(feature = "linalg-lapack")))]
compile_error!("provider-src requires linalg-lapack");
#[cfg(all(feature = "provider-inject", not(feature = "linalg-lapack")))]
compile_error!("provider-inject requires linalg-lapack");
#[cfg(all(
    any(
        feature = "src-openblas",
        feature = "src-netlib",
        feature = "src-accelerate",
        feature = "src-r",
        feature = "src-intel-mkl-dynamic-sequential",
        feature = "src-intel-mkl-dynamic-parallel",
        feature = "src-intel-mkl-static-sequential",
        feature = "src-intel-mkl-static-parallel"
    ),
    not(feature = "linalg-lapack")
))]
compile_error!("src-* features require linalg-lapack and provider-src");

#[cfg(feature = "linalg-lapack")]
const _: () = {
    let provider_count =
        (cfg!(feature = "provider-src") as usize) + (cfg!(feature = "provider-inject") as usize);
    assert!(
        provider_count == 1,
        "linalg-lapack requires exactly one provider: provider-src or provider-inject"
    );

    let src_count = (cfg!(feature = "src-openblas") as usize)
        + (cfg!(feature = "src-netlib") as usize)
        + (cfg!(feature = "src-accelerate") as usize)
        + (cfg!(feature = "src-r") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-parallel") as usize)
        + (cfg!(feature = "src-intel-mkl-static-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-static-parallel") as usize);

    if cfg!(feature = "provider-src") {
        assert!(
            src_count == 1,
            "provider-src requires exactly one src-* feature"
        );
    }
    if cfg!(feature = "provider-inject") {
        assert!(src_count == 0, "provider-inject forbids src-* features");
    }
};

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;

pub mod backend;
#[cfg(all(feature = "linalg-lapack", feature = "provider-inject"))]
pub mod inject;
mod prims_bridge;

use std::any::type_name;

use chainrules_core::AdResult;
use num_traits::Zero;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

// ============================================================================
// LinalgScalar trait
// ============================================================================

#[doc(inline)]
pub use tenferro_linalg_prims::LinalgScalar;

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

fn invalid_vector_lp_exponent_error(p: f64) -> Error {
    Error::InvalidArgument(format!("vector Lp norm requires p >= 1, got {p}"))
}

fn matrix_only_norm_kind_error(kind: NormKind) -> Error {
    Error::InvalidArgument(format!("norm kind {kind:?} expects matrix input"))
}

fn invalid_vector_lp_exponent_ad_error(p: f64) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(format!(
        "vector Lp norm requires p >= 1, got {p}"
    ))
}

fn matrix_only_norm_kind_ad_error(kind: NormKind) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(format!(
        "norm kind {kind:?} expects matrix input"
    ))
}

/// Validate Hermitian/symmetric structure for batched square matrices stored
/// in column-major contiguous layout.
///
/// For complex types, checks `A[i,j] == conj(A[j,i])`.
/// For real types, checks `A[i,j] == A[j,i]`.
fn validate_hermitian_batches<T: LinalgScalar>(
    data: &[T],
    offset: usize,
    n: usize,
    bc: usize,
    op_name: &str,
) -> Result<()> {
    let mat_size = n * n;
    let tol_scale = <T::Real as num_traits::NumCast>::from(128.0).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{op_name}: cannot convert tolerance scale 128.0 to real type"
        ))
    })?;

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];
        for j in 0..n {
            for i in 0..j {
                let a_ij = batch_data[i + j * n];
                let a_ji = batch_data[j + i * n];
                // Hermitian check: a_ij should equal conj(a_ji)
                let diff = (a_ij - a_ji.conj()).abs_real();
                let scale = <T::Real as num_traits::One>::one()
                    + num_traits::Float::max(a_ij.abs_real(), a_ji.abs_real());
                let tol = T::real_epsilon() * tol_scale * scale;
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

/// Extract the raw data slice from a tensor.
///
/// Returns an error if the tensor buffer cannot be viewed as a contiguous slice
/// (e.g., non-CPU buffer or unexpected memory layout).
fn extract_slice<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<&[T]> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

/// Convert an f64 constant to scalar type T.
///
/// Returns an error if the conversion is not supported by the scalar type.
fn scalar_from<T: LinalgScalar>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| {
        Error::InvalidArgument(format!("cannot convert {val} to target scalar type"))
    })
}

/// Convert a tenferro_device::Error into an AutodiffError for use in AD functions.
fn to_ad_err(e: Error) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(e.to_string())
}

/// Compute batch count from batch dims (product, or 1 if empty).
fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

/// Build output dims: [mat_dims..., batch_dims...].
fn output_dims(mat_dims: &[usize], batch_dims: &[usize]) -> Vec<usize> {
    let mut dims = mat_dims.to_vec();
    dims.extend_from_slice(batch_dims);
    dims
}

fn is_identity_permutation(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(idx, &axis)| idx == axis)
}

fn axes_to_end_permutation(rank: usize, axes: &[usize]) -> Vec<usize> {
    let mut is_solution_axis = vec![false; rank];
    for &axis in axes {
        is_solution_axis[axis] = true;
    }

    let mut perm = Vec::with_capacity(rank);
    for (axis, selected) in is_solution_axis.iter().enumerate() {
        if !selected {
            perm.push(axis);
        }
    }
    perm.extend_from_slice(axes);
    perm
}

fn validate_tensor_solve_axes(
    rank: usize,
    expected_len: usize,
    dims: Option<&[usize]>,
) -> Result<Vec<usize>> {
    let axes = if let Some(dims) = dims {
        if dims.len() != expected_len {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve expects {} solution axes, got {}",
                expected_len,
                dims.len()
            )));
        }
        dims.to_vec()
    } else {
        (rank - expected_len..rank).collect()
    };

    let mut seen = vec![false; rank];
    for &axis in &axes {
        if axis >= rank {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve axis {} is out of bounds for rank {}",
                axis, rank
            )));
        }
        if std::mem::replace(&mut seen[axis], true) {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve axes must be unique, got {:?}",
                axes
            )));
        }
    }
    Ok(axes)
}

/// Create a Tensor from raw column-major data with the given dims.
fn tensor_from_data<T: LinalgScalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

/// Create a Tensor from raw column-major data with the given dims.
///
/// Like [`tensor_from_data`] but only requires `Scalar`, so it works for
/// `Complex<R>` types that are not `LinalgScalar`.
fn tensor_from_data_scalar<T: Scalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
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
/// ```
/// use tenferro_linalg::{svd, SvdOptions};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, col);
///
/// // Full SVD
/// let result = svd(&mut ctx, &a, None).unwrap();
///
/// // Truncated SVD
/// let opts = SvdOptions { max_rank: Some(2), cutoff: None };
/// let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn svd<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::thin_svd(ctx, tensor)?;
    if options.is_none() {
        return Ok(SvdResult {
            u: result.u,
            s: result.s,
            vt: result.vt,
        });
    }

    let u_input = ensure_col_major(&result.u);
    let s_input = ensure_col_major(&result.s);
    let vt_input = ensure_col_major(&result.vt);
    let u_data_in = extract_slice(&u_input)?;
    let s_data_in = extract_slice(&s_input)?;
    let vt_data_in = extract_slice(&vt_input)?;
    let u_offset = u_input.offset() as usize;
    let s_offset = s_input.offset() as usize;
    let vt_offset = vt_input.offset() as usize;

    let m = result.u.dims()[0];
    let k = result.s.dims()[0];
    let n = result.vt.dims()[1];
    let batch_dims = &result.s.dims()[1..];
    let bc = batch_count(batch_dims);

    // Determine effective rank after truncation
    let opts = options.expect("checked above");
    let max_k = opts.max_rank.map_or(k, |r| r.min(k));

    let mut u_data = vec![T::zero(); m * max_k * bc];
    let mut s_data = vec![<T::Real>::zero(); max_k * bc];
    let mut vt_data = vec![T::zero(); max_k * n * bc];

    for b in 0..bc {
        let u_full = &u_data_in[u_offset + b * m * k..u_offset + (b + 1) * m * k];
        let s_full = &s_data_in[s_offset + b * k..s_offset + (b + 1) * k];
        let vt_full = &vt_data_in[vt_offset + b * k * n..vt_offset + (b + 1) * k * n];

        // Apply cutoff truncation
        let actual_k = if let Some(cutoff) = opts.cutoff {
            let cutoff_r: T::Real = scalar_from(cutoff)?;
            let mut ak = max_k;
            while ak > 0 && s_full[ak - 1] < cutoff_r {
                ak -= 1;
            }
            ak
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
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn qr<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<QrResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::qr(ctx, tensor)?;

    Ok(QrResult {
        q: result.q,
        r: result.r,
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
/// // Partial pivoting (default)
/// let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
///
/// // NoPivot is supported (no permutation output).
/// let no_pivot = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
/// assert!(no_pivot.p.is_none());
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn lu<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    pivot: LuPivot,
) -> Result<LuResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    if pivot == LuPivot::NoPivot {
        let (m, n, batch_dims) = validate_2d(tensor)?;
        let bc = batch_count(batch_dims);
        let k = m.min(n);
        let mat_size = m * n;

        let input = ensure_col_major(tensor);
        let data = extract_slice(&input)?;
        let offset = input.offset() as usize;

        let mut all_l = vec![T::zero(); m * k * bc];
        let mut all_u = vec![T::zero(); k * n * bc];

        for batch in 0..bc {
            let start = offset + batch * mat_size;
            let mut lu_data = data[start..start + mat_size].to_vec();

            // Doolittle LU without pivoting.
            for p in 0..k {
                let pivot_val = lu_data[p + p * m];
                if pivot_val.abs_real() <= T::real_epsilon() {
                    return Err(Error::InvalidArgument(format!(
                        "NoPivot LU encountered near-zero pivot at row {p} in batch {batch}"
                    )));
                }

                for i in (p + 1)..m {
                    lu_data[i + p * m] = lu_data[i + p * m] / pivot_val;
                }
                for j in (p + 1)..n {
                    let up = lu_data[p + j * m];
                    for i in (p + 1)..m {
                        let idx = i + j * m;
                        lu_data[idx] = lu_data[idx] - lu_data[i + p * m] * up;
                    }
                }
            }

            for j in 0..k {
                for i in 0..m {
                    let val = if i < j {
                        T::zero()
                    } else if i == j {
                        T::one()
                    } else {
                        lu_data[i + j * m]
                    };
                    all_l[batch * m * k + i + j * m] = val;
                }
            }
            for j in 0..n {
                for i in 0..k {
                    let val = if i <= j {
                        lu_data[i + j * m]
                    } else {
                        T::zero()
                    };
                    all_u[batch * k * n + i + j * k] = val;
                }
            }
        }

        let l_dims = output_dims(&[m, k], batch_dims);
        let u_dims = output_dims(&[k, n], batch_dims);
        return Ok(LuResult {
            p: None,
            l: tensor_from_data(all_l, &l_dims)?,
            u: tensor_from_data(all_u, &u_dims)?,
        });
    }

    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;

    Ok(LuResult {
        p: Some(result.pivots.into_iter().map(|p| p as usize).collect()),
        l: result.l,
        u: result.u,
    })
}

/// Compute the packed LU factorization of a batched matrix.
///
/// The returned `factors` tensor has the same shape as the input. Its strict
/// lower-triangular part stores the multipliers for `L`, and its diagonal plus
/// upper-triangular part stores `U`.
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
pub fn lu_factor<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<LuFactorResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let result = lu_factor_impl(ctx, tensor)?;
    Ok(LuFactorResult {
        factors: result.factors,
        pivots: result.pivots,
    })
}

/// Compute the packed LU factorization with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
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
/// assert_eq!(result.info, vec![0]);
/// ```
pub fn lu_factor_ex<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    lu_factor_impl(ctx, tensor)
}

/// Solve `A x = b` from a packed LU factorization.
///
/// `factors` and `pivots` should come from [`lu_factor`] or [`lu_factor_ex`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu_factor, lu_solve};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[3.0_f64, 1.0, 1.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let b = Tensor::from_slice(&[9.0_f64, 8.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let lu = lu_factor(&mut ctx, &a).unwrap();
/// let x = lu_solve(&mut ctx, &lu.factors, &lu.pivots, &b).unwrap();
/// assert_eq!(x.dims(), &[2]);
/// ```
pub fn lu_solve<T: LinalgScalar, C>(
    ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &[usize],
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("lu_solve")?;
    lu_solve_impl(ctx, factors, pivots, b)
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
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions, the first two
/// dimensions are not equal, or the matrix is not symmetric/Hermitian.
pub fn eigen<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<EigenResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    validate_hermitian_batches(data, offset, n, bc, "eigen")?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eigen_sym(ctx, tensor)?;

    Ok(EigenResult {
        values: result.values,
        vectors: result.vectors,
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
/// ```
/// use tenferro_linalg::lstsq;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let result = lstsq(&mut ctx, &a, &b).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `A` has fewer than 2 dimensions, `m < n`, or `b`
/// does not match `(m, *)` with the same batch dimensions as `A`.
pub fn lstsq<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("lstsq")?;

    let (m, n, batch_dims) = validate_2d(a)?;
    if m < n {
        return Err(Error::InvalidArgument(format!(
            "lstsq requires m >= n, got m={m}, n={n}"
        )));
    }
    validate_lstsq_rhs(b, m, batch_dims)?;

    // Solve via QR: A = Q R, then x = R^{-1} Q^T b
    let qr_result = qr(ctx, a)?;
    let q_input = ensure_col_major(&qr_result.q);
    let r_input = ensure_col_major(&qr_result.r);
    let b_input = ensure_col_major(b);

    let q_data = extract_slice(&q_input)?;
    let r_data = extract_slice(&r_input)?;
    let b_data = extract_slice(&b_input)?;
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
        backend::cpu::solve_triangular_slices(r_b, &qtb, k, 1, true, &mut x_buf)?;
        x_data[batch * n..(batch + 1) * n].copy_from_slice(&x_buf);

        // Compute residual: r = b - A x
        let a_contiguous = a.contiguous(MemoryOrder::ColumnMajor);
        let a_slice = extract_slice(&a_contiguous)?;
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
/// ```no_run
/// use tenferro_linalg::cholesky;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let l = cholesky(&mut ctx, &a).unwrap();
/// ```
pub fn cholesky<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::cholesky(ctx, tensor)
}

/// Compute the Cholesky decomposition with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
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
pub fn cholesky_ex<T: LinalgScalar, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<CholeskyExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("cholesky_ex")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;

    let mut factors = vec![T::zero(); mat_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let l_out = &mut factors[batch * mat_size..(batch + 1) * mat_size];
        if backend::cpu::cholesky_slices(a_slice, n, l_out).is_err() {
            l_out.fill(T::zero());
            info[batch] = 1;
        }
    }

    Ok(CholeskyExResult {
        l: tensor_from_data(factors, &output_dims(&[n, n], batch_dims))?,
        info,
    })
}

/// Solve a square linear system `A x = b`.
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
/// Batch dimensions in `b` must match those of `A`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let x = solve(&mut ctx, &a, &b).unwrap();
/// ```
pub fn solve<T: LinalgScalar, C>(ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}

/// Solve a square linear system with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
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
pub fn solve_ex<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("solve_ex")?;

    let (n, batch_dims) = validate_square(a)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_ex")?;
    let bc = batch_count(batch_dims);

    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);
    let a_data = extract_slice(&a_input)?;
    let b_data = extract_slice(&b_input)?;
    let a_offset = a_input.offset() as usize;
    let b_offset = b_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut solution = vec![T::zero(); rhs_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let a_start = a_offset + batch * mat_size;
        let b_start = b_offset + batch * rhs_size;
        let a_slice = &a_data[a_start..a_start + mat_size];
        let b_slice = &b_data[b_start..b_start + rhs_size];
        let x_out = &mut solution[batch * rhs_size..(batch + 1) * rhs_size];
        if backend::cpu::solve_slices(a_slice, b_slice, n, rhs.nrhs, x_out).is_err() {
            x_out.fill(T::zero());
            info[batch] = 1;
        }
    }

    Ok(SolveExResult {
        solution: tensor_from_data(solution, &rhs.output_dims)?,
        info,
    })
}

/// Compute the inverse of a square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3,
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_inv = inv(&mut ctx, &a).unwrap();
/// ```
pub fn inv<T: LinalgScalar, C>(_ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("inv")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
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
        backend::cpu::solve_slices(a_b, &eye_mat, n, n, x_out)?;
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(inv_data, &dims)
}

/// Compute the inverse with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
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
pub fn inv_ex<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<InvExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let mut eye_data = vec![T::zero(); n * n * bc];
    let eye = identity_matrix::<T>(n);
    for batch in 0..bc {
        eye_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
    }
    let rhs = tensor_from_data(eye_data, &output_dims(&[n, n], batch_dims))?;
    let result = solve_ex(ctx, tensor, &rhs)?;
    Ok(InvExResult {
        inverse: result.solution,
        info: result.info,
    })
}

/// Compute the determinant of a square matrix.
///
/// Input shape: `(n, n, *)`. Returns shape `(*)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let d = det(&mut ctx, &a).unwrap();
/// ```
pub fn det<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("det")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut det_data = vec![T::zero(); bc];

    // Pre-allocate temp buffers for LU per batch
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for (b, det_slot) in det_data.iter_mut().enumerate().take(bc) {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        // det = product of diagonal of U * sign from permutation
        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

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
        *det_slot = d;
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
/// // det(A) ≈ result.sign * exp(result.logabsdet)
/// ```
pub fn slogdet<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("slogdet")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
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

        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

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
/// and eigenvectors are always returned as complex, since a general real
/// matrix can have complex eigenvalue pairs.
///
/// Input shape: `(n, n, *)`.
///
/// Returns [`EigResult`] with complex eigenvalues (shape `(n, *)`) and
/// complex right eigenvectors (shape `(n, n, *)`).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eig;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eig(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions or the first
/// two dimensions are not equal.
pub fn eig<T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<EigResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eig(ctx, tensor)?;

    Ok(EigResult {
        values: result.values,
        vectors: result.vectors,
    })
}

fn ensure_cpu_backend<T: LinalgScalar, C>(op: &str) -> Result<()>
where
    C: backend::TensorLinalgContextFor<T>,
{
    if type_name::<C::Backend>() == type_name::<backend::CpuTensorLinalgBackend>() {
        return Ok(());
    }

    Err(Error::DeviceError(format!(
        "{op} is currently supported only on CpuContext"
    )))
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
/// ```
/// use tenferro_linalg::pinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_pinv = pinv(&mut ctx, &a, None).unwrap();
/// ```
pub fn pinv<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("pinv")?;

    let (m, n, batch_dims) = validate_2d(tensor)?;

    // Compute via SVD: pinv(A) = V diag(1/S) U^T
    let svd_result = svd(ctx, tensor, None)?;
    let u_input = ensure_col_major(&svd_result.u);
    let s_input = ensure_col_major(&svd_result.s);
    let vt_input = ensure_col_major(&svd_result.vt);

    let u_data = extract_slice(&u_input)?;
    let s_data = extract_slice(&s_input)?;
    let vt_data = extract_slice(&vt_input)?;
    let u_off = u_input.offset() as usize;
    let s_off = s_input.offset() as usize;
    let vt_off = vt_input.offset() as usize;

    let k = m.min(n);
    let bc = batch_count(batch_dims);
    // Default threshold: 1e-15 matches NumPy/Julia convention for f64
    // (approximately 4.5 × machine epsilon). Singular values below
    // `s_max * threshold` are treated as zero.
    let threshold: T = scalar_from(rcond.unwrap_or(1e-15))?;

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
            // Division is safe: s_b[i] > cutoff > 0 guarantees s_b[i] is
            // bounded away from zero by at least `s_max * threshold`.
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
/// Uses the scaling-and-squaring method with Pad\u{e9}\[13/13\] approximation
/// (Al-Mohy & Higham, 2010), following the PyTorch approach.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let exp_a = matrix_exp(&mut ctx, &a).unwrap();
/// // exp(0) = I
/// ```
pub fn matrix_exp<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("matrix_exp")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut result_data = vec![T::zero(); mat_size * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_slice = &data[start..start + mat_size];
        let exp_a = matrix_exp_single(ctx, a_slice, n)?;
        result_data[b * mat_size..(b + 1) * mat_size].copy_from_slice(&exp_a);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(result_data, &dims)
}

/// Raise a square matrix to an integer power.
///
/// Negative exponents are supported for invertible matrices.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_power;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let a3 = matrix_power(&mut ctx, &a, 3).unwrap();
/// assert_eq!(a3.dims(), &[2, 2]);
/// ```
pub fn matrix_power<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    exponent: i64,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("matrix_power")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let dims = output_dims(&[n, n], batch_dims);

    if exponent == 0 {
        let eye = identity_matrix::<T>(n);
        let mut data = vec![T::zero(); n * n * bc];
        for batch in 0..bc {
            data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
        }
        return tensor_from_data(data, &dims);
    }

    let positive_exponent = if exponent < 0 {
        let abs = exponent.checked_abs().ok_or_else(|| {
            Error::InvalidArgument("matrix_power does not support i64::MIN exponent".into())
        })?;
        let inverse = inv(ctx, tensor)?;
        return matrix_power(ctx, &inverse, abs);
    } else {
        exponent as u64
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;
    let mut out = vec![T::zero(); mat_size * bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let powered = matrix_power_single(ctx, a_slice, n, positive_exponent)?;
        out[batch * mat_size..(batch + 1) * mat_size].copy_from_slice(&powered);
    }

    tensor_from_data(out, &dims)
}

/// Compute the cross product along the leading vector axis.
///
/// Inputs must have shape `(3, *)` and identical dimensions. The cross product
/// is evaluated independently over every trailing index.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cross;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
/// let c = cross(&mut ctx, &a, &b).unwrap();
/// assert_eq!(c.dims(), &[3]);
/// ```
pub fn cross<T: LinalgScalar, C>(_ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("cross")?;

    if a.ndim() != b.ndim() {
        return Err(Error::InvalidArgument(format!(
            "cross expects matching ranks, got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }
    if a.ndim() == 0 || a.dims()[0] != 3 {
        return Err(Error::InvalidArgument(format!(
            "cross expects leading vector dimension of size 3, got {:?}",
            a.dims()
        )));
    }
    if b.ndim() == 0 || b.dims()[0] != 3 {
        return Err(Error::InvalidArgument(format!(
            "cross expects leading vector dimension of size 3, got {:?}",
            b.dims()
        )));
    }
    let mut out_dims = vec![3];
    for axis in 1..a.ndim() {
        let lhs = a.dims()[axis];
        let rhs = b.dims()[axis];
        if lhs != rhs && lhs != 1 && rhs != 1 {
            return Err(Error::InvalidArgument(format!(
                "cross broadcast mismatch on axis {axis}: left={}, right={}",
                lhs, rhs
            )));
        }
        out_dims.push(lhs.max(rhs));
    }

    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);
    let a_data = extract_slice(&a_input)?;
    let b_data = extract_slice(&b_input)?;
    let a_offset = a_input.offset() as usize;
    let b_offset = b_input.offset() as usize;
    let lanes = out_dims[1..].iter().product::<usize>().max(1);
    let out_strides = backend::col_major_strides(&out_dims);
    let a_strides = backend::col_major_strides(a.dims());
    let b_strides = backend::col_major_strides(b.dims());
    let mut out = vec![T::zero(); out_dims.iter().product()];
    let mut index = vec![0usize; out_dims.len().saturating_sub(1)];

    for _lane in 0..lanes {
        let mut a_tail_offset = 0isize;
        let mut b_tail_offset = 0isize;
        let mut out_tail_offset = 0isize;
        for axis in 1..out_dims.len() {
            let coord = index[axis - 1];
            out_tail_offset += coord as isize * out_strides[axis];
            let a_coord = if a.dims()[axis] == 1 { 0 } else { coord };
            let b_coord = if b.dims()[axis] == 1 { 0 } else { coord };
            a_tail_offset += a_coord as isize * a_strides[axis];
            b_tail_offset += b_coord as isize * b_strides[axis];
        }

        let a_base = (a_offset as isize + a_tail_offset) as usize;
        let b_base = (b_offset as isize + b_tail_offset) as usize;
        let o_base = out_tail_offset as usize;
        let ax = a_data[a_base];
        let ay = a_data[a_base + 1];
        let az = a_data[a_base + 2];
        let bx = b_data[b_base];
        let by = b_data[b_base + 1];
        let bz = b_data[b_base + 2];
        out[o_base] = ay * bz - az * by;
        out[o_base + 1] = az * bx - ax * bz;
        out[o_base + 2] = ax * by - ay * bx;

        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_dims[axis + 1] {
                break;
            }
            index[axis] = 0;
        }
    }

    tensor_from_data(out, &out_dims)
}

/// Form the explicit product of Householder reflectors.
///
/// `a` stores reflector vectors in the standard QR compact format with shape
/// `(m, n, *)`. `tau` stores the reflector coefficients with shape `(k, *)`,
/// where `k <= min(m, n)`.
///
/// The result has shape `(m, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::householder_product;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///     &[4, 2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let q = householder_product(&mut ctx, &a, &tau).unwrap();
/// assert_eq!(q.dims(), &[4, 2]);
/// ```
pub fn householder_product<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    tau: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("householder_product")?;

    let (m, n, batch_dims) = validate_2d(a)?;
    if tau.ndim() != 1 + batch_dims.len() {
        return Err(Error::InvalidArgument(format!(
            "householder_product expects tau shape (k, *), got {:?}",
            tau.dims()
        )));
    }
    if &tau.dims()[1..] != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "householder_product batch dims mismatch: expected {:?}, got {:?}",
            batch_dims,
            &tau.dims()[1..]
        )));
    }

    let k = tau.dims()[0];
    if k > m.min(n) {
        return Err(Error::InvalidArgument(format!(
            "householder_product expects tau length <= min(m, n) = {}, got {}",
            m.min(n),
            k
        )));
    }

    let a_input = ensure_col_major(a);
    let tau_input = ensure_col_major(tau);
    let a_data = extract_slice(&a_input)?;
    let tau_data = extract_slice(&tau_input)?;
    let a_offset = a_input.offset() as usize;
    let tau_offset = tau_input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let mut out = vec![T::zero(); mat_size * bc];

    for batch in 0..bc {
        let a_start = a_offset + batch * mat_size;
        let tau_start = tau_offset + batch * k;
        let a_batch = &a_data[a_start..a_start + mat_size];
        let tau_batch = &tau_data[tau_start..tau_start + k];
        let q_batch = &mut out[batch * mat_size..(batch + 1) * mat_size];

        for col in 0..n {
            if col < m {
                q_batch[col * m + col] = T::one();
            }
        }

        for reflector in (0..k).rev() {
            let tau_i = tau_batch[reflector];
            if tau_i == T::zero() {
                continue;
            }
            for col in 0..n {
                let mut proj = q_batch[reflector + col * m];
                for row in (reflector + 1)..m {
                    proj = proj + a_batch[row + reflector * m].conj() * q_batch[row + col * m];
                }
                proj = tau_i * proj;
                q_batch[reflector + col * m] = q_batch[reflector + col * m] - proj;
                for row in (reflector + 1)..m {
                    q_batch[row + col * m] =
                        q_batch[row + col * m] - a_batch[row + reflector * m] * proj;
                }
            }
        }
    }

    tensor_from_data(out, &output_dims(&[m, n], batch_dims))
}

/// Build a Vandermonde matrix from leading-dimension vectors.
///
/// If `columns` is `None`, the output uses as many columns as the input vector
/// length. For scalar input, the leading vector length is treated as `1`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::vander;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let v = vander(&mut ctx, &x, Some(3), true).unwrap();
/// assert_eq!(v.dims(), &[2, 3]);
/// ```
pub fn vander<T: LinalgScalar, C>(
    _ctx: &mut C,
    x: &Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("vander")?;

    let (vector_len, batch_dims): (usize, &[usize]) = if x.ndim() == 0 {
        (1, &[])
    } else {
        (x.dims()[0], &x.dims()[1..])
    };
    let columns = columns.unwrap_or(vector_len);

    let x_input = ensure_col_major(x);
    let x_data = extract_slice(&x_input)?;
    let x_offset = x_input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mut out = vec![T::zero(); vector_len * columns * bc];

    for batch in 0..bc {
        let vector = if x.ndim() == 0 {
            &x_data[x_offset..x_offset + 1]
        } else {
            let start = x_offset + batch * vector_len;
            &x_data[start..start + vector_len]
        };
        for row in 0..vector_len {
            let value = vector[row];
            let mut powers = vec![T::one(); columns];
            for col in 1..columns {
                powers[col] = powers[col - 1] * value;
            }
            for col in 0..columns {
                let power = if increasing {
                    powers[col]
                } else {
                    powers[columns.saturating_sub(col + 1)]
                };
                out[batch * vector_len * columns + row + col * vector_len] = power;
            }
        }
    }

    tensor_from_data(out, &output_dims(&[vector_len, columns], batch_dims))
}

/// Invert a tensorized square operator.
///
/// `ind` splits the tensor shape into `(left_dims, right_dims)` and requires
/// `prod(left_dims) == prod(right_dims)`. The output shape is
/// `(right_dims..., left_dims...)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let eye = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let a = eye.reshape(&[1, 2, 1, 2]).unwrap();
/// let inv = tensorinv(&mut ctx, &a, 2).unwrap();
/// assert_eq!(inv.dims(), &[1, 2, 1, 2]);
/// ```
pub fn tensorinv<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    ind: usize,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("tensorinv")?;

    if ind == 0 || ind >= tensor.ndim() {
        return Err(Error::InvalidArgument(format!(
            "tensorinv expects 0 < ind < rank, got ind={ind} for shape {:?}",
            tensor.dims()
        )));
    }

    let left_dims = &tensor.dims()[..ind];
    let right_dims = &tensor.dims()[ind..];
    let left_prod = left_dims.iter().product::<usize>();
    let right_prod = right_dims.iter().product::<usize>();
    if left_prod != right_prod {
        return Err(Error::InvalidArgument(format!(
            "tensorinv requires prod(shape[..ind]) == prod(shape[ind..]); got {} and {} for {:?}",
            left_prod,
            right_prod,
            tensor.dims()
        )));
    }

    let input = ensure_col_major(tensor);
    let matrix = input.reshape(&[left_prod, right_prod])?;
    let inverse = inv(ctx, &matrix)?;

    let mut out_dims = right_dims.to_vec();
    out_dims.extend_from_slice(left_dims);
    inverse.reshape(&out_dims)
}

/// Solve a tensorized linear system.
///
/// By default the solution uses the trailing `a.ndim() - b.ndim()` axes of `a`.
/// If `dims` is provided, those axes are moved to the end in the given order
/// before solving, and the solution shape follows that order.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorsolve;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let eye = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///     &[4, 4],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
/// let b = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let x = tensorsolve(&mut ctx, &a, &b, None).unwrap();
/// assert_eq!(x.dims(), &[2, 2]);
/// ```
pub fn tensorsolve<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dims: Option<&[usize]>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("tensorsolve")?;

    if b.ndim() > a.ndim() {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve expects b rank <= a rank, got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }

    let solution_rank = a.ndim() - b.ndim();
    let solution_axes = validate_tensor_solve_axes(a.ndim(), solution_rank, dims)?;
    let perm = axes_to_end_permutation(a.ndim(), &solution_axes);
    let a_permuted = if is_identity_permutation(&perm) {
        a.clone()
    } else {
        a.permute(&perm)?
    };

    if &a_permuted.dims()[..b.ndim()] != b.dims() {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve leading dims of permuted a must match b; got {:?} and {:?}",
            a_permuted.dims(),
            b.dims()
        )));
    }

    let lhs_prod = b.dims().iter().product::<usize>();
    let rhs_dims = &a_permuted.dims()[b.ndim()..];
    let rhs_prod = rhs_dims.iter().product::<usize>();
    if lhs_prod != rhs_prod {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve requires matching flattened system size, got {} and {}",
            lhs_prod, rhs_prod
        )));
    }

    let a_contiguous = ensure_col_major(&a_permuted);
    let a_matrix = a_contiguous.reshape(&[lhs_prod, rhs_prod])?;
    let b_contiguous = ensure_col_major(b);
    let b_vector = b_contiguous.reshape(&[lhs_prod])?;
    let x = solve(ctx, &a_matrix, &b_vector)?;
    x.reshape(rhs_dims)
}

// ============================================================================
// matrix_exp helpers (private)
// ============================================================================

/// Pad\u{e9}\[13/13\] coefficients b\[0\]..b\[13\] (integer values as f64).
const PADE13_COEFFS: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// Theta threshold for order-13 Pad\u{e9} (f64).
const THETA_13: f64 = 5.371920351148152;

/// Compute the matrix 1-norm (max column sum of absolute values).
///
/// `a` is stored column-major as a flat slice of length `n*n`.
fn matrix_1_norm<T: LinalgScalar>(a: &[T], n: usize) -> T::Real {
    let mut max_col_sum = <T::Real as num_traits::Zero>::zero();
    for j in 0..n {
        let mut col_sum = <T::Real as num_traits::Zero>::zero();
        for i in 0..n {
            col_sum = col_sum + a[i + j * n].abs_real();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

/// Multiply two n x n column-major matrices using the backend.
fn backend_mat_mul_nn<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    prims_bridge::batched_gemm_via_prims(a, n, n, b, n)
}

/// Compute `result = alpha * a + beta * b` element-wise for flat slices.
fn mat_linear_combine<T: LinalgScalar>(alpha: T, a: &[T], beta: T, b: &[T], result: &mut [T]) {
    for i in 0..result.len() {
        result[i] = alpha * a[i] + beta * b[i];
    }
}

/// Build an n x n identity matrix in column-major flat layout.
fn identity_matrix<T: LinalgScalar>(n: usize) -> Vec<T> {
    let mut eye = vec![T::zero(); n * n];
    for i in 0..n {
        eye[i + i * n] = T::one();
    }
    eye
}

/// Scale a flat matrix slice by a scalar, returning a new vector.
fn mat_scale<T: LinalgScalar>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|&x| x * s).collect()
}

/// Add two flat matrix slices element-wise, returning a new vector.
fn mat_add<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Compute matrix exponential of a single n x n column-major matrix.
///
/// Uses scaling-and-squaring with Pad\u{e9}\[13/13\] approximation.
fn matrix_exp_single<T: LinalgScalar, C>(ctx: &mut C, a: &[T], n: usize) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    // Special case: 0x0 matrix
    if n == 0 {
        return Ok(Vec::new());
    }

    // Special case: 1x1 matrix
    if n == 1 {
        let a_f64: f64 = num_traits::NumCast::from(a[0]).ok_or_else(|| {
            Error::InvalidArgument("matrix_exp: cannot convert 1×1 element to f64".into())
        })?;
        let exp_val = a_f64.exp();
        let result_val = T::from(exp_val).ok_or_else(|| {
            Error::InvalidArgument("cannot convert exp result to target type".into())
        })?;
        return Ok(vec![result_val]);
    }

    // 1. Compute ||A||_1
    let norm_a = matrix_1_norm(a, n);
    let norm_f64: f64 = num_traits::NumCast::from(norm_a)
        .ok_or_else(|| Error::InvalidArgument("matrix_exp: cannot convert 1-norm to f64".into()))?;

    // 2. Determine scaling factor s
    let s: usize = if norm_f64 <= THETA_13 {
        0
    } else {
        (norm_f64 / THETA_13).log2().ceil().max(0.0) as usize
    };

    // 3. Scale A: a_scaled = A / 2^s
    let scale_denom = (1u64 << s.min(63)) as f64;
    let scale_inv = T::from(1.0 / scale_denom).ok_or_else(|| {
        Error::InvalidArgument("cannot convert scale factor to target type".into())
    })?;
    let a_scaled = mat_scale(a, scale_inv);

    // 4. Compute matrix powers: A2, A4, A6
    let a2 = backend_mat_mul_nn(ctx, &a_scaled, &a_scaled, n)?;
    let a4 = backend_mat_mul_nn(ctx, &a2, &a2, n)?;
    let a6 = backend_mat_mul_nn(ctx, &a4, &a2, n)?;

    // Convert Pade coefficients to type T
    let b: Vec<T> = PADE13_COEFFS
        .iter()
        .map(|&c| {
            T::from(c).ok_or_else(|| {
                Error::InvalidArgument("cannot convert Pade coefficient to target type".into())
            })
        })
        .collect::<Result<Vec<T>>>()?;

    let eye = identity_matrix::<T>(n);
    let nn = n * n;

    // 5. Compute U and V for Pade[13/13]:
    //
    //   inner_u = b[13]*A6 + b[11]*A4 + b[9]*A2
    //   U = A * (A6 * inner_u + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    //
    //   inner_v = b[12]*A6 + b[10]*A4 + b[8]*A2
    //   V = A6 * inner_v + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I

    // Compute inner_u = b[13]*A6 + b[11]*A4 + b[9]*A2
    let mut inner_u = vec![T::zero(); nn];
    for i in 0..nn {
        inner_u[i] = b[13] * a6[i] + b[11] * a4[i] + b[9] * a2[i];
    }

    // a6_inner_u = A6 * inner_u
    let a6_inner_u = backend_mat_mul_nn(ctx, &a6, &inner_u, n)?;

    // u_inner = a6_inner_u + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I
    let mut u_inner = vec![T::zero(); nn];
    for i in 0..nn {
        u_inner[i] = a6_inner_u[i] + b[7] * a6[i] + b[5] * a4[i] + b[3] * a2[i] + b[1] * eye[i];
    }

    // U = A_scaled * u_inner
    let u = backend_mat_mul_nn(ctx, &a_scaled, &u_inner, n)?;

    // Compute inner_v = b[12]*A6 + b[10]*A4 + b[8]*A2
    let mut inner_v = vec![T::zero(); nn];
    for i in 0..nn {
        inner_v[i] = b[12] * a6[i] + b[10] * a4[i] + b[8] * a2[i];
    }

    // a6_inner_v = A6 * inner_v
    let a6_inner_v = backend_mat_mul_nn(ctx, &a6, &inner_v, n)?;

    // V = a6_inner_v + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    let mut v = vec![T::zero(); nn];
    for i in 0..nn {
        v[i] = a6_inner_v[i] + b[6] * a6[i] + b[4] * a4[i] + b[2] * a2[i] + b[0] * eye[i];
    }

    // 6. Solve (-U + V) * X = (U + V)  =>  X = exp(A_scaled)
    let neg_one = T::from(-1.0)
        .ok_or_else(|| Error::InvalidArgument("cannot convert -1 to target type".into()))?;
    // lhs = V - U = -U + V
    let mut lhs = vec![T::zero(); nn];
    mat_linear_combine(neg_one, &u, T::one(), &v, &mut lhs);

    // rhs = U + V
    let rhs = mat_add(&u, &v);

    // Solve lhs * X = rhs  (nrhs = n for matrix RHS)
    let mut result = vec![T::zero(); nn];
    backend::cpu::solve_slices(&lhs, &rhs, n, n, &mut result)?;

    // 7. Repeated squaring: result = result^(2^s)
    for _ in 0..s {
        result = backend_mat_mul_nn(ctx, &result, &result, n)?;
    }

    Ok(result)
}

fn matrix_power_single<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
    exponent: u64,
) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    if exponent == 1 {
        return Ok(a.to_vec());
    }

    let mut result = identity_matrix::<T>(n);
    let mut base = a.to_vec();
    let mut power = exponent;

    while power > 0 {
        if power & 1 == 1 {
            result = backend_mat_mul_nn(ctx, &result, &base, n)?;
        }
        power >>= 1;
        if power > 0 {
            base = backend_mat_mul_nn(ctx, &base, &base, n)?;
        }
    }

    Ok(result)
}

fn lu_factor_impl<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let factors = pack_lu_factors(&result.l, &result.u, m, n, batch_dims)?;

    let u_input = ensure_col_major(&result.u);
    let u_data = extract_slice(&u_input)?;
    let u_offset = u_input.offset() as usize;
    let mut info = vec![0_i32; bc];

    for (batch, info_slot) in info.iter_mut().enumerate().take(bc) {
        let start = u_offset + batch * k * n;
        let u_slice = &u_data[start..start + k * n];
        for i in 0..k {
            if u_slice[i + i * k].abs_real() <= T::real_epsilon() {
                *info_slot = (i + 1) as i32;
                break;
            }
        }
    }

    Ok(LuFactorExResult {
        factors,
        pivots: result
            .pivots
            .into_iter()
            .map(|pivot| pivot as usize)
            .collect(),
        info,
    })
}

fn pack_lu_factors<T: LinalgScalar>(
    l: &Tensor<T>,
    u: &Tensor<T>,
    m: usize,
    n: usize,
    batch_dims: &[usize],
) -> Result<Tensor<T>> {
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let l_input = ensure_col_major(l);
    let u_input = ensure_col_major(u);
    let l_data = extract_slice(&l_input)?;
    let u_data = extract_slice(&u_input)?;
    let l_offset = l_input.offset() as usize;
    let u_offset = u_input.offset() as usize;
    let mut packed = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let l_start = l_offset + batch * m * k;
        let u_start = u_offset + batch * k * n;
        let l_slice = &l_data[l_start..l_start + m * k];
        let u_slice = &u_data[u_start..u_start + k * n];
        let packed_slice = &mut packed[batch * m * n..(batch + 1) * m * n];
        for j in 0..n {
            for i in 0..m {
                packed_slice[i + j * m] = if i > j {
                    l_slice[i + j * m]
                } else {
                    u_slice[i + j * k]
                };
            }
        }
    }

    tensor_from_data(packed, &output_dims(&[m, n], batch_dims))
}

fn lu_solve_impl<T: LinalgScalar, C>(
    _ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &[usize],
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(factors)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "lu_solve")?;
    let bc = batch_count(batch_dims);
    let expected_pivots = n * bc;
    if pivots.len() != expected_pivots {
        return Err(Error::InvalidArgument(format!(
            "lu_solve expects pivots.len() == {expected_pivots}, got {}",
            pivots.len()
        )));
    }

    let factors_input = ensure_col_major(factors);
    let rhs_input = ensure_col_major(b);
    let factors_data = extract_slice(&factors_input)?;
    let rhs_data = extract_slice(&rhs_input)?;
    let factors_offset = factors_input.offset() as usize;
    let rhs_offset = rhs_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut out = vec![T::zero(); rhs_size * bc];
    let mut lower = vec![T::zero(); mat_size];
    let mut upper = vec![T::zero(); mat_size];
    let mut permuted_rhs = vec![T::zero(); rhs_size];
    let mut tmp = vec![T::zero(); rhs_size];

    for batch in 0..bc {
        let factor_start = factors_offset + batch * mat_size;
        let rhs_start = rhs_offset + batch * rhs_size;
        let factor_slice = &factors_data[factor_start..factor_start + mat_size];
        let rhs_slice = &rhs_data[rhs_start..rhs_start + rhs_size];
        let pivot_slice = &pivots[batch * n..(batch + 1) * n];

        unpack_packed_lu_square(factor_slice, n, &mut lower, &mut upper);
        apply_lu_permutation(pivot_slice, rhs_slice, n, rhs.nrhs, &mut permuted_rhs)?;
        backend::cpu::solve_triangular_slices(&lower, &permuted_rhs, n, rhs.nrhs, false, &mut tmp)?;
        backend::cpu::solve_triangular_slices(
            &upper,
            &tmp,
            n,
            rhs.nrhs,
            true,
            &mut out[batch * rhs_size..(batch + 1) * rhs_size],
        )?;
    }

    tensor_from_data(out, &rhs.output_dims)
}

fn unpack_packed_lu_square<T: LinalgScalar>(
    factors: &[T],
    n: usize,
    lower: &mut [T],
    upper: &mut [T],
) {
    lower.fill(T::zero());
    upper.fill(T::zero());
    for j in 0..n {
        for i in 0..n {
            let value = factors[i + j * n];
            if i > j {
                lower[i + j * n] = value;
            } else {
                upper[i + j * n] = value;
                if i == j {
                    lower[i + j * n] = T::one();
                }
            }
        }
    }
}

fn apply_lu_permutation<T: LinalgScalar>(
    pivots: &[usize],
    rhs: &[T],
    n: usize,
    nrhs: usize,
    out: &mut [T],
) -> Result<()> {
    for &pivot in pivots {
        if pivot >= n {
            return Err(Error::InvalidArgument(format!(
                "lu_solve pivot index {pivot} is out of range for n={n}"
            )));
        }
    }

    for col in 0..nrhs {
        let col_offset = col * n;
        for row in 0..n {
            out[row + col_offset] = rhs[pivots[row] + col_offset];
        }
    }

    Ok(())
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
/// ```
/// use tenferro_linalg::solve_triangular;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0],
///     &[3, 3],
///     col,
/// ).unwrap();
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let x = solve_triangular(&mut ctx, &a, &b, true).unwrap(); // upper=true
/// ```
pub fn solve_triangular<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve_triangular(ctx, a, b, upper)
}

/// Compute a norm.
///
/// Supported input shapes:
/// - rank-1 vectors `(n)` for `NormKind::Fro`, `NormKind::L1`, `NormKind::Inf`,
///   and `NormKind::Lp(p)`
/// - matrices `(m, n, *)` for all currently implemented matrix norms
///
/// Supported kinds in the current implementation:
/// - `NormKind::Fro`
/// - `NormKind::Nuclear`
/// - `NormKind::Spectral`
/// - `NormKind::L1` (max absolute column sum)
/// - `NormKind::Inf` (max absolute row sum)
/// - `NormKind::Lp(p)` for vectors
///
/// Return shape is `(*)` (batch dimensions) for matrices. For vectors and
/// non-batched matrices, the result is a scalar tensor `[]`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let fro = norm(&mut ctx, &a, NormKind::Fro).unwrap();
/// ```
pub fn norm<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("norm")?;

    if tensor.ndim() == 1 {
        let input = ensure_col_major(tensor);
        let offset = input.offset() as usize;
        let len = tensor.dims()[0];
        let vec_data = &extract_slice(&input)?[offset..offset + len];

        let value = match kind {
            NormKind::Fro => {
                let mut sum = T::zero();
                for &v in vec_data {
                    sum = sum + v * v;
                }
                sum.sqrt()
            }
            NormKind::L1 => vec_data.iter().fold(T::zero(), |acc, &v| acc + v.abs()),
            NormKind::Inf => vec_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs())),
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_error(p));
                }
                let (p_t, mut sum) = (scalar_from::<T>(p)?, T::zero());
                for &v in vec_data {
                    sum = sum + v.abs().powf(p_t);
                }
                sum.powf(T::one() / p_t)
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_error(kind));
            }
        };

        return tensor_from_data(vec![value], &[]);
    }

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;

    match kind {
        NormKind::Fro => {
            // Frobenius norm per batch: sqrt(sum of squares over matrix dims)
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let start = offset + batch * mat_size;
                let mut sum = T::zero();
                for i in 0..mat_size {
                    let v = data[start + i];
                    sum = sum + v * v;
                }
                *out_slot = sum.sqrt();
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Nuclear => {
            // Nuclear norm per batch: sum of singular values
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let mut sum = T::zero();
                let start = s_off + batch * k;
                for i in 0..k {
                    sum = sum + s_data[start + i];
                }
                *out_slot = sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Spectral => {
            // Spectral norm per batch: largest singular value
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                *out_slot = s_data[s_off + batch * k];
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::L1 => {
            // Matrix L1 norm per batch: max absolute column sum.
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_col_sum = T::zero();
                for j in 0..n {
                    let mut col_sum = T::zero();
                    for i in 0..m {
                        col_sum = col_sum + data[start + i + j * m].abs();
                    }
                    if j == 0 || col_sum > max_col_sum {
                        max_col_sum = col_sum;
                    }
                }
                *out_slot = max_col_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Inf => {
            // Matrix infinity norm per batch: max absolute row sum.
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_row_sum = T::zero();
                for i in 0..m {
                    let mut row_sum = T::zero();
                    for j in 0..n {
                        row_sum = row_sum + data[start + i + j * m].abs();
                    }
                    if i == 0 || row_sum > max_row_sum {
                        max_row_sum = row_sum;
                    }
                }
                *out_slot = max_row_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
}

/// Compute the matrix condition number with a selected norm convention.
///
/// Currently supported for square matrices with `NormKind::Fro`,
/// `NormKind::L1`, `NormKind::Inf`, and `NormKind::Spectral`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{cond, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 0.5], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let value = cond(&mut ctx, &a, NormKind::Fro).unwrap();
/// assert_eq!(value.dims(), &[]);
/// ```
pub fn cond<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    match kind {
        NormKind::Fro | NormKind::L1 | NormKind::Inf | NormKind::Spectral => {}
        _ => {
            return Err(Error::InvalidArgument(format!(
                "cond only supports Fro, L1, Inf, and Spectral norms, got {kind:?}"
            )));
        }
    }

    validate_square(tensor)?;
    let lhs = norm(ctx, tensor, kind)?;
    let inverse = inv(ctx, tensor)?;
    let rhs = norm(ctx, &inverse, kind)?;
    let lhs_data = extract_slice(&lhs)?;
    let rhs_data = extract_slice(&rhs)?;
    let lhs_offset = lhs.offset() as usize;
    let rhs_offset = rhs.offset() as usize;
    let len = lhs.dims().iter().product::<usize>().max(1);
    let mut out = vec![T::zero(); len];
    for i in 0..len {
        out[i] = lhs_data[lhs_offset + i] * rhs_data[rhs_offset + i];
    }
    tensor_from_data(out, lhs.dims())
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
/// ```
/// use tenferro_linalg::lstsq;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let result = lstsq(&mut ctx, &a, &b).unwrap();
/// assert_eq!(result.x.dims(), &[2]);
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

/// Conjugate transpose (adjoint) of a column-major m×n matrix to n×m.
///
/// For real types this is equivalent to [`transpose`].
fn adjoint_transpose<T: LinalgScalar>(data: &[T], m: usize, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            result[j + i * n] = data[i + j * m].conj();
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
fn phi<T: LinalgScalar>(data: &[T], n: usize) -> AdResult<Vec<T>> {
    let mut result = tril(data, n);
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;
    for i in 0..n {
        result[i + i * n] = result[i + i * n] * half;
    }
    Ok(result)
}

// ============================================================================
// Complex matrix helpers for eig AD rules
// ============================================================================

/// Complex type alias parameterized by real scalar.
type Cx<R> = num_complex::Complex<R>;

/// Extract data slice from a Tensor whose element type implements `Scalar`
/// (but not necessarily `LinalgScalar`). Used for `Tensor<Complex<R>>` in eig AD.
fn extract_data_scalar<T: Scalar>(tensor: &Tensor<T>) -> AdResult<Vec<T>> {
    let t = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = t.offset() as usize;
    let slice = t.buffer().as_slice().ok_or_else(|| {
        chainrules_core::AutodiffError::InvalidArgument(
            "tensor buffer is not a contiguous CPU slice".into(),
        )
    })?;
    let total_len: usize = tensor.dims().iter().product();
    Ok(slice[offset..offset + total_len].to_vec())
}

/// Complex matrix multiply: C = A * B  (all n*n, column-major flat slices).
fn complex_mat_mul_nn<R>(a: &[Cx<R>], b: &[Cx<R>], n: usize) -> Vec<Cx<R>>
where
    R: num_traits::Float + num_traits::NumCast,
{
    let zero = Cx::new(R::zero(), R::zero());
    let mut c = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = zero;
            for k in 0..n {
                sum = sum + a[i + k * n] * b[k + j * n];
            }
            c[i + j * n] = sum;
        }
    }
    c
}

/// Conjugate transpose of n*n complex matrix (column-major).
fn complex_conj_transpose<R>(a: &[Cx<R>], n: usize) -> Vec<Cx<R>>
where
    R: num_traits::Float + num_traits::NumCast,
{
    let zero = Cx::new(R::zero(), R::zero());
    let mut result = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            result[i + j * n] = a[j + i * n].conj();
        }
    }
    result
}

/// Solve A X = B for X, where A and B are n*n complex matrices.
///
/// Converts the complex n*n system to a real 2n*2n system and
/// delegates to `backend::cpu::solve_slices()`.
fn complex_solve_nn<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[Cx<T>],
    b: &[Cx<T>],
    n: usize,
) -> AdResult<Vec<Cx<T>>>
where
    T: backend::CpuLinalgScalar,
{
    let nn = 2 * n;
    let mut a_real = vec![T::zero(); nn * nn];
    let mut b_real = vec![T::zero(); nn * nn];

    for j in 0..n {
        for i in 0..n {
            let aij = a[i + j * n];
            // Top-left: Re(A)
            a_real[i + j * nn] = aij.re;
            // Top-right: -Im(A)
            a_real[i + (j + n) * nn] = T::zero() - aij.im;
            // Bottom-left: Im(A)
            a_real[(i + n) + j * nn] = aij.im;
            // Bottom-right: Re(A)
            a_real[(i + n) + (j + n) * nn] = aij.re;

            let bij = b[i + j * n];
            b_real[i + j * nn] = bij.re;
            b_real[(i + n) + j * nn] = bij.im;
        }
    }

    let mut x_real = vec![T::zero(); nn * nn];
    backend::cpu::solve_slices(&a_real, &b_real, nn, nn, &mut x_real).map_err(to_ad_err)?;

    let zero = Cx::new(T::zero(), T::zero());
    let mut result = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            result[i + j * n] = Cx::new(x_real[i + j * nn], x_real[(i + n) + j * nn]);
        }
    }
    Ok(result)
}

// ============================================================================
// LinalgBackend convenience wrappers for AD code
// ============================================================================

/// Mat mul via LinalgBackend, returning Vec for convenience in AD code.
fn backend_mat_mul<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    prims_bridge::batched_gemm_via_prims(a, m, k, b, n).map_err(to_ad_err)
}

/// Solve via LinalgBackend, returning Vec for convenience in AD code.
fn backend_solve<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_slices(a, b, n, nrhs, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Solve triangular via LinalgBackend, returning Vec for convenience in AD code.
fn backend_solve_tri<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_triangular_slices(a, b, n, nrhs, upper, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Thin SVD via LinalgBackend, returning (U, S, V) for convenience in AD code.
/// Note: returns V (not Vt) as column-major n×k for convenience in AD code.
fn backend_thin_svd<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>, Vec<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let k = m.min(n);
    let mut u = vec![T::zero(); m * k];
    let mut s = vec![T::zero(); k];
    let mut vt = vec![T::zero(); k * n];
    backend::cpu::thin_svd_slices(a, m, n, &mut u, &mut s, &mut vt).map_err(to_ad_err)?;
    // Convert Vt (k×n) to V (n×k) for convenience
    let v = transpose(&vt, k, n);
    Ok((u, s, v))
}

/// QR decomposition via LinalgBackend, returning (Q, R) for convenience in AD code.
fn backend_qr<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let k = m.min(n);
    let mut q = vec![T::zero(); m * k];
    let mut r = vec![T::zero(); k * n];
    backend::cpu::qr_slices(a, m, n, &mut q, &mut r).map_err(to_ad_err)?;
    Ok((q, r))
}

/// phi* (adjoint of phi): phi*(X) = (X + X^T - diag(X)) / 2
/// Diagonal gets halved, off-diagonal gets symmetrized.
fn phi_star<T: LinalgScalar>(data: &[T], n: usize) -> AdResult<Vec<T>> {
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;
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
    Ok(result)
}

/// Extract data slice from Tensor (ensuring col-major).
fn extract_data<T: LinalgScalar>(tensor: &Tensor<T>) -> AdResult<(Vec<T>, usize)> {
    let t = ensure_col_major(tensor);
    let offset = t.offset() as usize;
    let slice = extract_slice(&t).map_err(to_ad_err)?;
    let total_len = tensor.dims().iter().product::<usize>();
    Ok((slice[offset..offset + total_len].to_vec(), 0))
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
/// ```
/// use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
///
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::ones(&[3], mem, col)),
///     vt: None,
/// };
/// let grad_a = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn svd_rrule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    let result = svd(ctx, tensor, options)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T = {
        let raw: T = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (u_data, _) = extract_data(&result.u)?;
    let (s_data, _) = extract_data(&result.s)?;
    let (vt_data, _) = extract_data(&result.vt)?;

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
            let (ds_data, _) = extract_data(ds)?;
            let ds_b = &ds_data[b * k..(b + 1) * k];
            for i in 0..k {
                gamma[i + i * k] = gamma[i + i * k] + ds_b[i];
            }
        }

        // From dU cotangent: F ⊙ (U^T dU + (U^T dU)^T) * S
        if let Some(ref du) = cotangent.u {
            let (du_data, _) = extract_data(du)?;
            let du_b = &du_data[b * m * k..(b + 1) * m * k];
            // U^T dU (k×k)
            let ut_du = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, du_b, k)?;
            for i in 0..k {
                for j in 0..k {
                    let skew = ut_du[i + j * k] - ut_du[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + f_mat[i + j * k] * skew * s_b[j];
                }
            }
        }

        // From dVt cotangent: S * F ⊙ (V^T dV + (V^T dV)^T)
        if let Some(ref dvt) = cotangent.vt {
            let (dvt_data, _) = extract_data(dvt)?;
            let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
            // dV = dVt^T (n×k)
            let dv_b = transpose(dvt_b, k, n);
            // V^T dV (k×k)
            let vt_dv = backend_mat_mul(ctx, &transpose(&v_b, n, k), k, n, &dv_b, k)?;
            for i in 0..k {
                for j in 0..k {
                    let skew = vt_dv[i + j * k] - vt_dv[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + s_b[i] * f_mat[i + j * k] * skew;
                }
            }
        }

        // Core: dA_core = U * Gamma * V^T (m×k × k×k × k×n = m×n)
        let u_gamma = backend_mat_mul(ctx, u_b, m, k, &gamma, k)?;
        let da_core = backend_mat_mul(ctx, &u_gamma, m, k, &transpose(&v_b, n, k), n)?;

        // Copy core to output
        for i in 0..m * n {
            grad_a[b * m * n + i] = da_core[i];
        }

        // Non-square correction: (I - UU^T) dU S_inv^T V^T when m > k
        if m > k {
            if let Some(ref du) = cotangent.u {
                let (du_data, _) = extract_data(du)?;
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
                let inner = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, &du_sinv, k)?;
                let uut_du = backend_mat_mul(ctx, u_b, m, k, &inner, k)?;
                let proj = sub_vec(&du_sinv, &uut_du);
                let correction = backend_mat_mul(ctx, &proj, m, k, &transpose(&v_b, n, k), n)?;
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }

        // Non-square correction for n > k: U S_inv^T (I - VV^T) dV^T
        if n > k {
            if let Some(ref dvt) = cotangent.vt {
                let (dvt_data, _) = extract_data(dvt)?;
                let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
                let dv_b = transpose(dvt_b, k, n);
                // diag(1/S) * dV^T (k×n) = diag(1/S) * Vt_cotangent
                // But we need dV (n×k), so: (I - VV^T) dV → project
                let inner = backend_mat_mul(ctx, &transpose(&v_b, n, k), k, n, &dv_b, k)?;
                let vvt_dv = backend_mat_mul(ctx, &v_b, n, k, &inner, k)?;
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
                    backend_mat_mul(ctx, &u_sinv, m, k, &transpose(&proj_dv, n, k), n)?;
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
/// ```
/// use tenferro_linalg::{qr_rrule, QrCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
///     &[4, 3],
///     col,
/// ).unwrap();
/// let cotangent = QrCotangent {
///     q: Some(Tensor::ones(&[4, 3], mem, col)),
///     r: None,
/// };
/// let grad_a = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn qr_rrule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &QrCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    let result = qr(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (q_data, _) = extract_data(&result.q)?;
    let (r_data, _) = extract_data(&result.r)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];

        // Initialize dQ and dR from cotangents (zero if not provided)
        let dq_b: Vec<T> = if let Some(ref dq) = cotangent.q {
            let (dq_data, _) = extract_data(dq)?;
            dq_data[b * m * k..(b + 1) * m * k].to_vec()
        } else {
            vec![T::zero(); m * k]
        };
        let dr_b: Vec<T> = if let Some(ref dr) = cotangent.r {
            let (dr_data, _) = extract_data(dr)?;
            dr_data[b * k * n..(b + 1) * k * n].to_vec()
        } else {
            vec![T::zero(); k * n]
        };

        if m >= n {
            // For thin QR (m >= n): A = QR where Q is m×k, R is k×n.
            // Match PyTorch's reduced-QR backward for the real case.
            let r_drt = backend_mat_mul(ctx, r_b, k, n, &transpose(&dr_b, k, n), k)?;
            let dqt_q = backend_mat_mul(ctx, &transpose(&dq_b, m, k), k, m, q_b, k)?;
            let w = sub_vec(&r_drt, &dqt_q);

            let h = copyltu(&w, k);
            let qh = backend_mat_mul(ctx, q_b, m, k, &h, k)?;
            let rhs = add_vec(&dq_b, &qh);

            let r_square = r_b[..k * n].to_vec();
            let rhs_t = transpose(&rhs, m, k);
            let da_t = backend_solve_tri(ctx, &r_square, &rhs_t, k, m, true)?;
            let da_first_k = transpose(&da_t, k, m);

            for j in 0..k.min(n) {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = da_first_k[i + j * m];
                }
            }
        } else {
            // Wide reduced QR follows the PyTorch backward:
            // gA = pi*(Q trilImInvAdjSkew(Q^T gQ - gR R^T) R1^{-T}) + Q gR
            let qtgq = backend_mat_mul(ctx, &transpose(q_b, m, k), k, m, &dq_b, k)?;
            let gr_rt = backend_mat_mul(ctx, &dr_b, k, n, &transpose(r_b, k, n), k)?;
            let wide_inner = sub_vec(&qtgq, &gr_rt);

            let mut lower_skew = vec![T::zero(); k * k];
            for j in 0..k {
                for i in j..k {
                    lower_skew[i + j * k] = wide_inner[i + j * k] - wide_inner[j + i * k];
                }
            }

            let q_lower = backend_mat_mul(ctx, q_b, m, k, &lower_skew, k)?;
            let q_lower_t = transpose(&q_lower, m, k);
            let mut r1 = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    r1[i + j * k] = r_b[i + j * k];
                }
            }
            let leading_t = backend_solve_tri(ctx, &r1, &q_lower_t, k, m, true)?;
            let leading = transpose(&leading_t, k, m);

            for j in 0..k {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = leading[i + j * m];
                }
            }

            let qgr = backend_mat_mul(ctx, q_b, m, k, &dr_b, n)?;
            for j in 0..n {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = grad_a[b * m * n + i + j * m] + qgr[i + j * m];
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
/// ```
/// use tenferro_linalg::{lu_rrule, LuCotangent, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3], col)
///     .unwrap();
/// let cotangent = LuCotangent {
///     l: Some(Tensor::ones(&[3, 3], mem, col)),
///     u: None,
/// };
/// let grad_a = lu_rrule(&mut ctx, &a, &cotangent, LuPivot::Partial).unwrap();
/// ```
pub fn lu_rrule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &LuCotangent<T>,
    pivot: LuPivot,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    let result = lu(ctx, tensor, pivot)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    if let Some(ref dl) = cotangent.l {
        if dl.dims() != result.l.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lu_rrule L cotangent shape mismatch: expected {:?}, got {:?}",
                result.l.dims(),
                dl.dims()
            ))));
        }
    }
    if let Some(ref du) = cotangent.u {
        if du.dims() != result.u.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lu_rrule U cotangent shape mismatch: expected {:?}, got {:?}",
                result.u.dims(),
                du.dims()
            ))));
        }
    }

    let (l_data, _) = extract_data(&result.l)?;
    let (u_data, _) = extract_data(&result.u)?;
    let dl_data = if let Some(ref dl) = cotangent.l {
        Some(extract_data(dl)?.0)
    } else {
        None
    };
    let du_data = if let Some(ref du) = cotangent.u {
        Some(extract_data(du)?.0)
    } else {
        None
    };
    let p_vec = result.p.as_ref();

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];
        let dl_b = dl_data
            .as_ref()
            .map(|data| &data[b * m * k..(b + 1) * m * k]);
        let du_b = du_data
            .as_ref()
            .map(|data| &data[b * k * n..(b + 1) * k * n]);

        let batch_grad = if m == n {
            let l_t = transpose(l_b, k, k);
            let mut inner = vec![T::zero(); k * k];

            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_t, k, k, dl_b, k)?;
                inner = add_vec(&inner, &tril_strict(&lt_dl, k));
            }
            if let Some(du_b) = du_b {
                let du_ut = backend_mat_mul(ctx, du_b, k, k, &transpose(u_b, k, k), k)?;
                inner = add_vec(&inner, &triu(&du_ut, k));
            }

            let right_t = backend_solve_tri(ctx, u_b, &transpose(&inner, k, k), k, k, true)?;
            let right = transpose(&right_t, k, k);
            backend_solve_tri(ctx, &l_t, &right, k, k, true)?
        } else if m < n {
            let l_t = transpose(l_b, k, k);
            let u1: Vec<T> = {
                let mut out = vec![T::zero(); k * k];
                for j in 0..k {
                    for i in 0..k {
                        out[i + j * k] = u_b[i + j * k];
                    }
                }
                out
            };

            let mut core = vec![T::zero(); k * k];
            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_t, k, k, dl_b, k)?;
                core = add_vec(&core, &lt_dl);
            }
            if let Some(du_b) = du_b {
                let mut du_triu = vec![T::zero(); k * n];
                for j in 0..n {
                    for i in 0..k {
                        if i <= j {
                            du_triu[i + j * k] = du_b[i + j * k];
                        }
                    }
                }
                let du_term = backend_mat_mul(ctx, &du_triu, k, n, &transpose(u_b, k, n), k)?;
                core = sub_vec(&core, &du_term);
            }

            let lower = tril_strict(&core, k);
            let lower_t = backend_solve_tri(ctx, &u1, &transpose(&lower, k, k), k, k, true)?;
            let leading = transpose(&lower_t, k, k);

            let mut pre_left = vec![T::zero(); k * n];
            for j in 0..k {
                for i in 0..k {
                    pre_left[i + j * k] = leading[i + j * k];
                }
            }
            if let Some(du_b) = du_b {
                for j in 0..k {
                    for i in 0..=j {
                        pre_left[i + j * k] = pre_left[i + j * k] + du_b[i + j * k];
                    }
                }
                for j in k..n {
                    for i in 0..k {
                        pre_left[i + j * k] = du_b[i + j * k];
                    }
                }
            }

            backend_solve_tri(ctx, &l_t, &pre_left, k, n, true)?
        } else {
            let l1: Vec<T> = {
                let mut out = vec![T::zero(); k * k];
                for j in 0..k {
                    for i in 0..k {
                        out[i + j * k] = l_b[i + j * m];
                    }
                }
                out
            };
            let l1_t = transpose(&l1, k, k);

            let mut core = vec![T::zero(); k * k];
            if let Some(du_b) = du_b {
                let du_term = backend_mat_mul(ctx, du_b, k, k, &transpose(u_b, k, k), k)?;
                core = add_vec(&core, &du_term);
            }
            if let Some(dl_b) = dl_b {
                let mut dl_tril = vec![T::zero(); m * k];
                for j in 0..k {
                    for i in (j + 1)..m {
                        dl_tril[i + j * m] = dl_b[i + j * m];
                    }
                }
                let lt_dl = backend_mat_mul(ctx, &transpose(l_b, m, k), k, m, &dl_tril, k)?;
                core = sub_vec(&core, &lt_dl);
            }

            let upper = triu(&core, k);
            let leading = backend_solve_tri(ctx, &l1_t, &upper, k, k, true)?;

            let mut pre_right = vec![T::zero(); m * k];
            for j in 0..k {
                for i in 0..k {
                    pre_right[i + j * m] = leading[i + j * k];
                }
            }
            if let Some(dl_b) = dl_b {
                for j in 0..k {
                    for i in (j + 1)..k {
                        pre_right[i + j * m] = pre_right[i + j * m] + dl_b[i + j * m];
                    }
                    for i in k..m {
                        pre_right[i + j * m] = dl_b[i + j * m];
                    }
                }
            }

            let batch_grad_t =
                backend_solve_tri(ctx, u_b, &transpose(&pre_right, m, k), k, m, true)?;
            transpose(&batch_grad_t, k, m)
        };

        let out = &mut grad_a[b * m * n..(b + 1) * m * n];
        if let Some(pv) = p_vec {
            let p_b = &pv[b * m..(b + 1) * m];
            for j in 0..n {
                for i in 0..m {
                    out[p_b[i] + j * m] = batch_grad[i + j * m];
                }
            }
        } else {
            out.copy_from_slice(&batch_grad);
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
/// ```
/// use tenferro_linalg::{eigen_rrule, EigenCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigenCotangent {
///     values: Some(Tensor::ones(&[3], mem, col)),
///     vectors: None,
/// };
/// let grad_a = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eigen_rrule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    // Symmetric eigendecomposition: A = V diag(E) V^T
    let result = eigen(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T = {
        let raw: T = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (v_data, _) = extract_data(&result.vectors)?;
    let (e_data, _) = extract_data(&result.values)?;

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
            let (de_data, _) = extract_data(de)?;
            let de_b = &de_data[b * n..(b + 1) * n];
            for i in 0..n {
                d_mat[i + i * n] = de_b[i];
            }
        }

        if let Some(ref dv) = cotangent.vectors {
            let (dv_data, _) = extract_data(dv)?;
            let dv_b = &dv_data[b * n * n..(b + 1) * n * n];
            let vt_dv = backend_mat_mul(ctx, &transpose(v_b, n, n), n, n, dv_b, n)?;
            let half: T = scalar_from(0.5).map_err(to_ad_err)?;
            for i in 0..n {
                for j in 0..n {
                    let skew = half * (vt_dv[i + j * n] - vt_dv[j + i * n]);
                    d_mat[i + j * n] = d_mat[i + j * n] + f_mat[i + j * n] * skew;
                }
            }
        }

        // dA = V D V^T
        let vd = backend_mat_mul(ctx, v_b, n, n, &d_mat, n)?;
        let da_b = backend_mat_mul(ctx, &vd, n, n, &transpose(v_b, n, n), n)?;

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
/// ```
/// use tenferro_linalg::lstsq_rrule;
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
/// // grad.a: cotangent for A, grad.b: cotangent for b
/// ```
pub fn lstsq_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent_x: &Tensor<T>,
) -> AdResult<LstsqGrad<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("lstsq_rrule").map_err(to_ad_err)?;

    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    if cotangent_x.dims() != result.x.dims() {
        return Err(to_ad_err(Error::InvalidArgument(format!(
            "lstsq_rrule cotangent shape mismatch: expected {:?}, got {:?}",
            result.x.dims(),
            cotangent_x.dims()
        ))));
    }

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&result.x)?;
    let (r_data, _) = extract_data(&result.residual)?;
    let (dx_data, _) = extract_data(cotangent_x)?;

    let mut grad_a_data = vec![T::zero(); m * n * bc];
    let mut grad_b_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let r_b = &r_data[batch * m..(batch + 1) * m];
        let dx_b = &dx_data[batch * n..(batch + 1) * n];

        let (q_d, r_d) = backend_qr(ctx, a_b, m, n)?;
        let y = backend_solve_tri(ctx, &transpose(&r_d, n, n), dx_b, n, 1, false)?;
        let z = backend_solve_tri(ctx, &r_d, &y, n, 1, true)?;
        let grad_b = backend_mat_mul(ctx, &q_d, m, n, &y, 1)?;

        for j in 0..n {
            for i in 0..m {
                grad_a_data[batch * m * n + i + j * m] = r_b[i] * z[j] - grad_b[i] * x_b[j];
            }
        }
        grad_b_data[batch * m..(batch + 1) * m].copy_from_slice(&grad_b);
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
/// ```no_run
/// use tenferro_linalg::cholesky_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = cholesky_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn cholesky_rrule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    // A = L L^T, dA = L^{-T} phi*(tril(L^T dL)) L^{-1}
    let l = cholesky(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l)?;
    let (dl_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let dl_b = &dl_data[b * n * n..(b + 1) * n * n];

        // S = tril(L^T dL)
        let lt_dl = backend_mat_mul(ctx, &transpose(l_b, n, n), n, n, dl_b, n)?;
        let s = tril(&lt_dl, n);

        // Apply phi*: symmetrize S → (S + S^T) / 2
        let s_sym = phi_star(&s, n)?;

        // Solve L^T x = S_sym → x = L^{-T} S_sym
        let x = backend_solve_tri(ctx, &transpose(l_b, n, n), &s_sym, n, n, true)?;

        // Solve x L = result → result = x L^{-1} → L^T result^T = x^T → result^T = L^{-T} x^T
        let xt = transpose(&x, n, n);
        let result_t = backend_solve_tri(ctx, &transpose(l_b, n, n), &xt, n, n, true)?;
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
/// ```
pub fn solve_rrule<T: LinalgScalar>(
    ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>>
where
    T: backend::CpuLinalgScalar,
{
    // Ax = b → G = A^{-H} dx, dB = G, dA = -G x^H
    let x = solve(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_rrule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (dx_data, _) = extract_data(cotangent)?;

    let mut grad_a_data = vec![T::zero(); n * n * bc];
    let mut grad_b_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // G = A^{-H} dx = solve(A^H, dx)
        let at = adjoint_transpose(a_b, n, n);
        let g = backend_solve(ctx, &at, dx_b, n, nrhs)?;

        // dB = G
        grad_b_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&g);

        // dA = -G x^H (n×nrhs × nrhs×n = n×n)
        let x_h = adjoint_transpose(x_b, n, nrhs);
        let g_xh = backend_mat_mul(ctx, &g, n, nrhs, &x_h, n)?;
        let neg_g_xh = scale_vec(&g_xh, -T::one());
        grad_a_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg_g_xh);
    }

    let a_dims = output_dims(&[n, n], batch_dims);
    let b_dims = rhs.output_dims;
    Ok(SolveGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for triangular solve (VJP / pullback).
///
/// Given `A x = b` with triangular `A` and cotangent `x̄`, computes `(Ā, b̄)`.
///
/// - `G = A^{-H} x̄` solved with conjugate-transposed triangular structure
/// - `b̄ = G`
/// - `Ā = proj(-G x^H)` where `proj = triu` for upper, `tril` for lower
pub fn solve_triangular_rrule<T: LinalgScalar>(
    ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
    upper: bool,
) -> AdResult<SolveGrad<T>>
where
    T: backend::CpuLinalgScalar,
{
    let x = solve_triangular(ctx, a, b, upper)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        batch_dims,
        "solve_triangular_rrule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (dx_data, _) = extract_data(cotangent)?;

    let mut grad_a_data = vec![T::zero(); n * n * bc];
    let mut grad_b_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // G = A^{-H} dX, where A^H flips upper/lower.
        let at = adjoint_transpose(a_b, n, n);
        let g = backend_solve_tri(ctx, &at, dx_b, n, nrhs, !upper)?;

        // dB = G
        grad_b_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&g);

        // dA = proj(-G x^H)
        let x_h = adjoint_transpose(x_b, n, nrhs);
        let g_xh = backend_mat_mul(ctx, &g, n, nrhs, &x_h, n)?;
        let neg_g_xh = scale_vec(&g_xh, -T::one());
        let projected = if upper {
            triu(&neg_g_xh, n)
        } else {
            tril(&neg_g_xh, n)
        };
        grad_a_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&projected);
    }

    let a_dims = output_dims(&[n, n], batch_dims);
    let b_dims = rhs.output_dims;
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
/// ```
/// use tenferro_linalg::inv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = inv_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn inv_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("inv_rrule").map_err(to_ad_err)?;

    // dA = -B^T dB B^T where B = A^{-1}
    let b_inv = inv(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv)?;
    let (db_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * n..(batch + 1) * n * n];

        let bt = transpose(b_b, n, n);
        let bt_db = backend_mat_mul(ctx, &bt, n, n, db_b, n)?;
        let bt_db_bt = backend_mat_mul(ctx, &bt_db, n, n, &bt, n)?;
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
/// ```
/// use tenferro_linalg::det_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = det_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn det_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("det_rrule").map_err(to_ad_err)?;

    // dA = ddet * det(A) * A^{-T}
    let det_val = det(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (det_data, _) = extract_data(&det_val)?;
    let (ddet_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let d = det_data[batch];
        let dd = ddet_data[batch];

        // A^{-T}
        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
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
/// ```
/// use tenferro_linalg::{slogdet_rrule, SlogdetCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::ones(&[], mem, col)),
/// };
/// let grad_a = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn slogdet_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("slogdet_rrule").map_err(to_ad_err)?;

    // dA = d_logabsdet * A^{-T}
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    if let Some(ref dlog) = cotangent.logabsdet {
        let (dlog_data, _) = extract_data(dlog)?;
        for batch in 0..bc {
            let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
            let dl = dlog_data[batch];

            let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
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
/// Given eigendecomposition `A V = V diag(lambda)`, computes the gradient
/// of the input `A` from complex-valued cotangents for eigenvalues and
/// eigenvectors using the Mike Giles formulas.
///
/// The cotangent uses [`EigCotangent`] with complex-valued tensors
/// because `eig()` returns complex output even for real inputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{eig_rrule, EigCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use num_complex::Complex64;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigCotangent::<f64> {
///     values: None,
///     vectors: None,
/// };
/// let grad_a = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eig_rrule<T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    cotangent: &EigCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
{
    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    // Compute eigendecomposition
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;
    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];

        // Compute F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // V^H (conjugate transpose of V)
        let vh = complex_conj_transpose(v, n);

        // Build M_bar = diag(d_bar_lambda) + F .* (V^H d_bar_V)
        let mut m_bar = vec![zero_c; n * n];

        if let Some(ref dv_bar) = cotangent.vectors {
            let dv_bar_data = extract_data_scalar(dv_bar)?;
            let dv_bar_b = &dv_bar_data[b * n * n..(b + 1) * n * n];
            let vh_dv = complex_mat_mul_nn(&vh, dv_bar_b, n);
            for k in 0..n * n {
                m_bar[k] = f_mat[k] * vh_dv[k];
            }
        }

        if let Some(ref dlam) = cotangent.values {
            let dlam_data = extract_data_scalar(dlam)?;
            for i in 0..n {
                m_bar[i + i * n] = m_bar[i + i * n] + dlam_data[b * n + i];
            }
        }

        // d_bar_A = V^{-H} M_bar V^H = solve(V^H, M_bar @ V^H)
        let m_vh = complex_mat_mul_nn(&m_bar, &vh, n);
        let da_complex = complex_solve_nn(ctx, &vh, &m_vh, n)?;

        // Take real part (since input A was real)
        for k in 0..n * n {
            grad_data[b * n * n + k] = da_complex[k].re;
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}

/// Reverse-mode AD rule for pseudoinverse (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let grad_a = pinv_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn pinv_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("pinv_rrule").map_err(to_ad_err)?;

    // dA = -(A+)^T dA+ (A+)^T + (I - AA+)(dA+)^T A+(A+)^T + (A+)^T A+ (dA+)^T (I - A+A)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (dap_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let dap_b = &dap_data[batch * n * m..(batch + 1) * n * m];

        let apt = transpose(ap_b, n, m); // m×n
        let dapt = transpose(dap_b, n, m); // m×n

        // Term 1: -(A+)^T dA+ (A+)^T = -apt * dap * apt^T
        // apt: m×n, dap: n×m, apt: m×n → m×n * n×m * m×n = m×n
        let t1 = backend_mat_mul(ctx, &apt, m, n, dap_b, m)?;
        let t1 = backend_mat_mul(ctx, &t1, m, m, &apt, n)?;
        let t1 = scale_vec(&t1, -T::one());

        // Term 2: (I - AA+)(dA+)^T A+ (A+)^T
        // AA+ (m×m)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?;
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        // (dA+)^T A+ = dapt * ap (m×n * n×m = m×m)
        let dapt_ap = backend_mat_mul(ctx, &dapt, m, n, ap_b, m)?;
        // * (A+)^T = * apt (m×m * m×n = m×n)
        let dapt_ap_apt = backend_mat_mul(ctx, &dapt_ap, m, m, &apt, n)?;
        let t2 = backend_mat_mul(ctx, &i_aap, m, m, &dapt_ap_apt, n)?;

        // Term 3: (A+)^T A+ (dA+)^T (I - A+A)
        // A+A (n×n)
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?;
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        // (A+)^T A+ = apt * ap (m×n * n×m = m×m)
        let apt_ap = backend_mat_mul(ctx, &apt, m, n, ap_b, m)?;
        // * (dA+)^T = * dapt (m×m * m×n = m×n)
        let apt_ap_dapt = backend_mat_mul(ctx, &apt_ap, m, m, &dapt, n)?;
        let t3 = backend_mat_mul(ctx, &apt_ap_dapt, m, n, &i_apa, n)?;

        let da_b = add_vec(&t1, &add_vec(&t2, &t3));
        grad_a[batch * m * n..(batch + 1) * m * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for matrix exponential (VJP / pullback).
///
/// Computes the gradient of the input given a cotangent for `exp(A)`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A^T, cotangent], [0, A^T]]
/// grad_A = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = matrix_exp_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn matrix_exp_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("matrix_exp_rrule").map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (co_data, _) = extract_data(cotangent)?;

    let nn = 2 * n;
    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let co = &co_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A^T, cotangent], [0, A^T]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // A^T: transpose of A — a^T[i,j] = a[j,i] = a[j + i*n]
                let a_t_ij = a[j + i * n];
                // Top-left: A^T
                m[i + j * nn] = a_t_ij;
                // Top-right: cotangent
                m[i + (j + n) * nn] = co[i + j * n];
                // Bottom-right: A^T
                m[(i + n) + (j + n) * nn] = a_t_ij;
                // Bottom-left: already zero
            }
        }

        // Compute exp(M)
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-right block → gradient d̄A
        for j in 0..n {
            for i in 0..n {
                grad_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}

/// Reverse-mode AD rule for norm (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_rrule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = norm_rrule(&mut ctx, &a, &cotangent, NormKind::Fro).unwrap();
/// ```
pub fn norm_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("norm_rrule").map_err(to_ad_err)?;

    if tensor.ndim() == 1 {
        validate_norm_cotangent(cotangent, &[]).map_err(to_ad_err)?;
        let (a_data, _) = extract_data(tensor)?;
        let (dn_data, _) = extract_data(cotangent)?;
        let dn = dn_data[0];
        let len = tensor.dims()[0];
        let mut grad_a = vec![T::zero(); len];

        match kind {
            NormKind::Fro => {
                let nrm = norm(ctx, tensor, NormKind::Fro).map_err(to_ad_err)?;
                let (nrm_data, _) = extract_data(&nrm)?;
                let nv = nrm_data[0];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..len {
                    grad_a[i] = scale * a_data[i];
                }
            }
            NormKind::L1 => {
                for i in 0..len {
                    let v = a_data[i];
                    let sign = if v > T::zero() {
                        T::one()
                    } else if v < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    };
                    grad_a[i] = dn * sign;
                }
            }
            NormKind::Inf => {
                let max_abs = a_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs()));
                let active: Vec<usize> = a_data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v.abs() == max_abs { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    let scale = dn / active_count;
                    for i in active {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[i] = scale * sign;
                    }
                }
            }
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_ad_error(p));
                }
                if p == 1.0 {
                    for i in 0..len {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[i] = dn * sign;
                    }
                } else {
                    let nrm = norm(ctx, tensor, kind).map_err(to_ad_err)?;
                    let (nrm_data, _) = extract_data(&nrm)?;
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        let scale = dn / nv.powf(p_minus_one);
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            grad_a[i] = scale * sign * v.abs().powf(p_minus_one);
                        }
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        return tensor_from_data(grad_a, &[len]).map_err(to_ad_err);
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    validate_norm_cotangent(cotangent, batch_dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (a_data, _) = extract_data(tensor)?;
    let (dn_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    match kind {
        NormKind::Fro => {
            // dA = dn * A / ||A||_F
            let nrm = norm(ctx, tensor, NormKind::Fro)
                .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
            let (nrm_data, _) = extract_data(&nrm)?;
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
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let uv = backend_mat_mul(ctx, &u, m, k, &transpose(&v, n, k), n)?;
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
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let dn = dn_data[batch];
                for j in 0..n {
                    for i in 0..m {
                        grad_a[batch * m * n + i + j * m] = dn * u[i] * v[j];
                    }
                }
            }
        }
        NormKind::L1 => {
            // dA = dn * sign(A) on columns that attain max absolute column sum.
            // At ties, average uniformly over active columns.
            for batch in 0..bc {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut col_sums = vec![T::zero(); n];
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in 0..m {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    col_sums[j] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &col_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_cols: Vec<usize> = col_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &sum)| if sum == max_sum { Some(j) } else { None })
                    .collect();
                if active_cols.is_empty() {
                    continue;
                }
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for j in active_cols {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
                    }
                }
            }
        }
        NormKind::Inf => {
            // dA = dn * sign(A) on rows that attain max absolute row sum.
            // At ties, average uniformly over active rows.
            for batch in 0..bc {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut row_sums = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    row_sums[i] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &row_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_rows: Vec<usize> = row_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &sum)| if sum == max_sum { Some(i) } else { None })
                    .collect();
                if active_rows.is_empty() {
                    continue;
                }
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for i in active_rows {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
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
/// ```
/// use tenferro_linalg::svd_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn svd_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> AdResult<(SvdResult<T>, SvdResult<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let result = svd(ctx, tensor, options)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T = {
        let raw: T = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (u_data, _) = extract_data(&result.u)?;
    let (s_data, _) = extract_data(&result.s)?;
    let (vt_data, _) = extract_data(&result.vt)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut du_data = vec![T::zero(); m * k * bc];
    let mut ds_data = vec![T::zero(); k * bc];
    let mut dvt_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // C = U^T dA V (k×k)
        let ut_da = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
        let v_b = transpose(vt_b, k, n);
        let c = backend_mat_mul(ctx, &ut_da, k, n, &v_b, k)?;

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
        let du_core = backend_mat_mul(ctx, u_b, m, k, &f_inner, k)?;

        // Projector term for dU
        if m > k {
            let inner = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
            let uut_da = backend_mat_mul(ctx, u_b, m, k, &inner, n)?;
            let proj_da: Vec<T> = da_b
                .iter()
                .zip(uut_da.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let proj_da_v = backend_mat_mul(ctx, &proj_da, m, n, &v_b, k)?;
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
                st_c_plus_ct_s[i + j * k] = -(s_b[i] * c[i + j * k] + c[j + i * k] * s_b[j]);
            }
        }
        let f_inner2 = hadamard(&f_mat, &st_c_plus_ct_s);
        let dvt_core = backend_mat_mul(ctx, &f_inner2, k, k, vt_b, n)?;

        if n > k {
            let vvt = backend_mat_mul(ctx, &v_b, n, k, vt_b, n)?;
            let i_n = eye::<T>(n);
            let i_vvt = sub_vec(&i_n, &vvt);
            let ut_da = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
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
            let proj = backend_mat_mul(ctx, &sinv_ut_da, k, n, &i_vvt, n)?;
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
/// ```
/// use tenferro_linalg::qr_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
///     &[4, 3],
///     col,
/// ).unwrap();
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let (result, dresult) = qr_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn qr_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(QrResult<T>, QrResult<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let result = qr(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;

    let (q_data, _) = extract_data(&result.q)?;
    let (r_data, _) = extract_data(&result.r)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dq_data = vec![T::zero(); m * k * bc];
    let mut dr_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        let (dq_b_vec, dr_b_vec) = if m >= n {
            let r_sq = r_b[..n * n].to_vec();
            let darinv_t = backend_solve_tri(
                ctx,
                &transpose(&r_sq, n, n),
                &transpose(da_b, m, n),
                n,
                m,
                false,
            )?;
            let darinv = transpose(&darinv_t, n, m);
            let qhdarinv = backend_mat_mul(ctx, &transpose(q_b, m, n), n, m, &darinv, n)?;
            let sym = add_vec(&qhdarinv, &transpose(&qhdarinv, n, n));

            let mut dr_hat = vec![T::zero(); n * n];
            for j in 0..n {
                for i in 0..=j {
                    let mut val = sym[i + j * n];
                    if i == j {
                        val = val * half;
                    }
                    dr_hat[i + j * n] = val;
                }
            }

            let dq = sub_vec(&darinv, &backend_mat_mul(ctx, q_b, m, n, &dr_hat, n)?);
            let dr = backend_mat_mul(ctx, &dr_hat, n, n, &r_sq, n)?;
            (dq, dr)
        } else {
            let qhda = backend_mat_mul(ctx, &transpose(q_b, m, k), k, m, da_b, n)?;
            // k = min(m,n) so k*n == k*k when n == k (the only case reaching here)
            let r1 = r_b[..k * n].to_vec();

            let mut qhda1 = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    qhda1[i + j * k] = qhda[i + j * k];
                }
            }
            let qhda1_rinv_t = backend_solve_tri(
                ctx,
                &transpose(&r1, k, k),
                &transpose(&qhda1, k, k),
                k,
                k,
                false,
            )?;
            let qhda1_rinv = transpose(&qhda1_rinv_t, k, k);
            let lower = tril_strict(&qhda1_rinv, k);
            let dq_hat = sub_vec(&lower, &transpose(&lower, k, k));

            let dr = sub_vec(&qhda, &backend_mat_mul(ctx, &dq_hat, k, k, r_b, n)?);
            let dq = backend_mat_mul(ctx, q_b, m, k, &dq_hat, k)?;
            (dq, dr)
        };

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
/// ```
/// use tenferro_linalg::{lu_frule, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3], col)
///     .unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = lu_frule(&mut ctx, &a, &da, LuPivot::Partial).unwrap();
/// ```
pub fn lu_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let result = lu(ctx, tensor, pivot)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&result.l)?;
    let (u_data, _) = extract_data(&result.u)?;
    let p_vec = result.p.as_ref();
    let (da_data, _) = extract_data(tangent)?;

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
        let linv_pda = backend_solve_tri(ctx, &l_sq, &pda_sq, k, n, false)?;

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
            ctx,
            &transpose(&u_sq, k, k),
            &transpose(&linv_pda, k, n),
            k,
            k,
            false,
        )?;
        let f = transpose(&f_t, k, k);

        // dL = L tril_strict(F) (m×k)
        let tril_f = tril_strict(&f, k);
        let dl_b_vec = backend_mat_mul(ctx, &l_sq, k, k, &tril_f, k)?;
        for j in 0..k {
            for i in 0..k {
                dl_data[b * m * k + i + j * m] = dl_b_vec[i + j * k];
            }
        }

        // dU = triu(F) U (k×n)
        let triu_f = triu(&f, k);
        let du_b_vec = backend_mat_mul(ctx, &triu_f, k, k, &u_sq, k)?;
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
/// ```
/// use tenferro_linalg::eigen_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eigen_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eigen_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let result = eigen(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T = {
        let raw: T = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (v_data, _) = extract_data(&result.vectors)?;
    let (e_data, _) = extract_data(&result.values)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut de_data = vec![T::zero(); n * bc];
    let mut dv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // C = V^T dA V (n×n)
        let vt_da = backend_mat_mul(ctx, &transpose(v_b, n, n), n, n, da_b, n)?;
        let c = backend_mat_mul(ctx, &vt_da, n, n, v_b, n)?;

        // dE = diag(C)
        for i in 0..n {
            de_data[b * n + i] = c[i + i * n];
        }

        // dV = V F ⊙ C where F_ij = 1/(e_i - e_j) for i≠j, 0 diagonal
        let mut fc = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[j] - e_b[i];
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
        let dv_b_vec = backend_mat_mul(ctx, v_b, n, n, &fc, n)?;
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
/// ```
/// use tenferro_linalg::lstsq_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 2], mem, col);
/// let db = Tensor::<f64>::ones(&[3], mem, col);
/// let (result, dresult) = lstsq_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn lstsq_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T>, LstsqResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("lstsq_frule").map_err(to_ad_err)?;

    // dx = A^+ (db - dA x), where A^+ = (A^T A)^{-1} A^T
    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&result.x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * bc];
    let mut dres_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
        let db_b = &db_data[batch * m..(batch + 1) * m];

        // dA x (m×1)
        let da_x = backend_mat_mul(ctx, da_b, m, n, x_b, 1)?;
        // db - dA x
        let rhs: Vec<T> = db_b.iter().zip(da_x.iter()).map(|(&a, &b)| a - b).collect();

        // A^+ rhs = (A^T A)^{-1} A^T rhs
        let at_rhs = backend_mat_mul(ctx, &transpose(a_b, m, n), n, m, &rhs, 1)?;
        let ata = backend_mat_mul(ctx, &transpose(a_b, m, n), n, m, a_b, n)?;
        let dx_b_vec = backend_solve(ctx, &ata, &at_rhs, n, 1)?;
        dx_data[batch * n..(batch + 1) * n].copy_from_slice(&dx_b_vec);

        // d(residual) = db - dA x - A dx
        let a_dx = backend_mat_mul(ctx, a_b, m, n, &dx_b_vec, 1)?;
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
/// ```no_run
/// use tenferro_linalg::cholesky_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (l, dl) = cholesky_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn cholesky_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
{
    // dL = L phi(L^{-1} dA L^{-T})
    let l = cholesky(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dl_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // L^{-1} dA: solve L x = dA
        let linv_da = backend_solve_tri(ctx, l_b, da_b, n, n, false)?;
        // (L^{-1} dA) L^{-T}: solve (result) L^T = linv_da → L x^T = linv_da^T
        let linv_da_linvt_t = backend_solve_tri(ctx, l_b, &transpose(&linv_da, n, n), n, n, false)?;
        let inner = transpose(&linv_da_linvt_t, n, n);

        // phi(inner) = tril with diagonal halved
        let phi_inner = phi(&inner, n)?;

        // dL = L phi(inner)
        let dl_b_vec = backend_mat_mul(ctx, l_b, n, n, &phi_inner, n)?;
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
/// ```
/// use tenferro_linalg::solve_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let db = Tensor::<f64>::ones(&[3], mem, col);
/// let (x, dx) = solve_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn solve_frule<T: LinalgScalar<Real = T> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
{
    // dx = A^{-1} (db - dA x)
    let x = solve(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_frule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // dA x (n×nrhs)
        let da_x = backend_mat_mul(ctx, da_b, n, n, x_b, nrhs)?;
        // db - dA x
        let rhs = sub_vec(db_b, &da_x);
        // A^{-1} (db - dA x)
        let dx_b_vec = backend_solve(ctx, a_b, &rhs, n, nrhs)?;
        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b_vec);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for triangular solve (JVP / pushforward).
///
/// Computes:
/// - `x = solve_triangular(a, b, upper)`
/// - `dx = solve_triangular(a, db - proj(dA) * x, upper)`
///
/// where `proj(dA)` keeps only the active triangular part
/// (`triu` when `upper=true`, `tril` when `upper=false`).
pub fn solve_triangular_frule<T: LinalgScalar>(
    ctx: &mut tenferro_prims::CpuContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
    upper: bool,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
{
    if tangent_a.dims() != a.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_a shape mismatch: expected {:?}, got {:?}",
            a.dims(),
            tangent_a.dims()
        )));
    }
    if tangent_b.dims() != b.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_b shape mismatch: expected {:?}, got {:?}",
            b.dims(),
            tangent_b.dims()
        )));
    }

    // dX = A^{-1} (dB - proj(dA) X), with projection to the triangular tangent space.
    let x = solve_triangular(ctx, a, b, upper)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        batch_dims,
        "solve_triangular_frule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // Project dA onto the same triangular structure as A.
        let da_proj = if upper { triu(da_b, n) } else { tril(da_b, n) };

        // dA * x, treating x as n x nrhs in column-major layout.
        let da_x =
            prims_bridge::batched_gemm_via_prims(&da_proj, n, n, x_b, nrhs).map_err(to_ad_err)?;

        // RHS tangent: dB - dA * x
        let rhs = sub_vec(db_b, &da_x);

        // dX from triangular solve with the same structure.
        let mut dx_b = vec![T::zero(); n * nrhs];
        backend::cpu::solve_triangular_slices(a_b, &rhs, n, nrhs, upper, &mut dx_b)
            .map_err(to_ad_err)?;

        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (a_inv, da_inv) = inv_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn inv_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("inv_frule").map_err(to_ad_err)?;

    // dB = -B dA B where B = A^{-1}
    let b_inv = inv(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut db_data = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let b_da = backend_mat_mul(ctx, b_b, n, n, da_b, n)?;
        let b_da_b = backend_mat_mul(ctx, &b_da, n, n, b_b, n)?;
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
/// ```
/// use tenferro_linalg::det_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (d, dd) = det_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn det_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("det_frule").map_err(to_ad_err)?;

    // d(det) = det(A) * tr(A^{-1} dA)
    let d = det(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (d_data, _) = extract_data(&d)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dd_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
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
/// ```
/// use tenferro_linalg::slogdet_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = slogdet_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn slogdet_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T>, SlogdetResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("slogdet_frule").map_err(to_ad_err)?;

    // d(logabsdet) = Re(tr(A^{-1} dA)), d(sign) = 0 (for real)
    let result = slogdet(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dlog_data = vec![T::zero(); bc];
    let dsign_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
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
/// Given eigendecomposition `A V = V diag(lambda)`, computes the tangents
/// of eigenvalues and eigenvectors from a real tangent `dA` using the
/// Mike Giles formulas.
///
/// Returns `(primal, tangent)` where both are [`EigResult`] with complex
/// eigenvalues and eigenvectors.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eig_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eig_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eig_frule<T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float>(
    ctx: &mut tenferro_prims::CpuContext,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigResult<T>, EigResult<T>)>
where
    T: backend::CpuLinalgScalar,
{
    // Forward pass
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;
    let (tang_data, _) = extract_data(tangent)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut dval_data = vec![zero_c; n * bc];
    let mut dvec_data = vec![zero_c; n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];
        let da = &tang_data[b * n * n..(b + 1) * n * n];

        // Convert real dA to complex
        let da_complex: Vec<Cx<T>> = da.iter().map(|&x| Cx::new(x, T::zero())).collect();

        // W = V^{-1} dA V = solve(V, dA_c @ V)
        let da_v = complex_mat_mul_nn(&da_complex, v, n);
        let w = complex_solve_nn(ctx, v, &da_v, n)?;

        // d_lambda = diag(W)
        for i in 0..n {
            dval_data[b * n + i] = w[i + i * n];
        }

        // F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // dV = V * (F .* W)
        let mut fw = vec![zero_c; n * n];
        for k in 0..n * n {
            fw[k] = f_mat[k] * w[k];
        }
        let dv = complex_mat_mul_nn(v, &fw, n);
        dvec_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dv);
    }

    // Build tangent EigResult
    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);

    let d_result = EigResult {
        values: tensor_from_data_scalar(dval_data, &val_dims).map_err(to_ad_err)?,
        vectors: tensor_from_data_scalar(dvec_data, &vec_dims).map_err(to_ad_err)?,
    };

    Ok((eig_result, d_result))
}

/// Forward-mode AD rule for pseudoinverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (pinv_a, dpinv_a) = pinv_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn pinv_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("pinv_frule").map_err(to_ad_err)?;

    // dA+ = -A+ dA A+ + (I - A+A) dA^T (A+)^T A+ + A+ (A+)^T dA^T (I - AA+)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dap_data = vec![T::zero(); n * m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];

        let dat = transpose(da_b, m, n); // n×m
        let apt = transpose(ap_b, n, m); // m×n

        // Term 1: -A+ dA A+ (n×m × m×n × n×m = n×m)
        let ap_da = backend_mat_mul(ctx, ap_b, n, m, da_b, n)?;
        let ap_da_ap = backend_mat_mul(ctx, &ap_da, n, n, ap_b, m)?;
        let t1 = scale_vec(&ap_da_ap, -T::one());

        // Term 2: (I - A+A) dA^T (A+)^T A+
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?; // n×n
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        let dat_apt = backend_mat_mul(ctx, &dat, n, m, &apt, n)?; // n×n
        let dat_apt_ap = backend_mat_mul(ctx, &dat_apt, n, n, ap_b, m)?; // n×m
        let t2 = backend_mat_mul(ctx, &i_apa, n, n, &dat_apt_ap, m)?;

        // Term 3: A+ (A+)^T dA^T (I - AA+)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?; // m×m
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        let ap_apt = backend_mat_mul(ctx, ap_b, n, m, &apt, n)?; // n×n
        let ap_apt_dat = backend_mat_mul(ctx, &ap_apt, n, n, &dat, m)?; // n×m
        let t3 = backend_mat_mul(ctx, &ap_apt_dat, n, m, &i_aap, m)?;

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
/// Computes `exp(A)` and the Frechet derivative `d(exp(A))` in the direction `dA`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A, dA], [0, A]]
/// exp(A)    = top-left  n×n block of exp(M)
/// d(exp(A)) = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (exp_a, dexp_a) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn matrix_exp_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("matrix_exp_frule").map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let nn = 2 * n;
    let mut result_data = vec![T::zero(); n * n * bc];
    let mut tangent_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let da = &da_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A, dA], [0, A]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // Top-left: A
                m[i + j * nn] = a[i + j * n];
                // Top-right: dA
                m[i + (j + n) * nn] = da[i + j * n];
                // Bottom-right: A
                m[(i + n) + (j + n) * nn] = a[i + j * n];
                // Bottom-left: already zero
            }
        }

        // Compute exp(M) — call matrix_exp_single with the 2n×2n matrix
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-left block → exp(A)
        for j in 0..n {
            for i in 0..n {
                result_data[b * n * n + i + j * n] = exp_m[i + j * nn];
            }
        }

        // Extract top-right block → d(exp(A))
        for j in 0..n {
            for i in 0..n {
                tangent_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    let result = tensor_from_data(result_data, &dims).map_err(to_ad_err)?;
    let tang = tensor_from_data(tangent_data, &dims).map_err(to_ad_err)?;
    Ok((result, tang))
}

/// Forward-mode AD rule for norm (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_frule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (n, dn) = norm_frule(&mut ctx, &a, &da, NormKind::Fro).unwrap();
/// ```
pub fn norm_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    ensure_cpu_backend::<T, C>("norm_frule").map_err(to_ad_err)?;

    let nrm = norm(ctx, tensor, kind)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    if tensor.ndim() == 1 {
        let (a_data, _) = extract_data(tensor)?;
        let (nrm_data, _) = extract_data(&nrm)?;
        let (da_data, _) = extract_data(tangent)?;
        let len = tensor.dims()[0];
        let mut dnrm = T::zero();

        match kind {
            NormKind::Fro => {
                let nv = nrm_data[0];
                if nv > T::zero() {
                    for i in 0..len {
                        dnrm = dnrm + a_data[i] * da_data[i];
                    }
                    dnrm = dnrm / nv;
                }
            }
            NormKind::L1 => {
                for i in 0..len {
                    let v = a_data[i];
                    let sign = if v > T::zero() {
                        T::one()
                    } else if v < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    };
                    dnrm = dnrm + sign * da_data[i];
                }
            }
            NormKind::Inf => {
                let max_abs = a_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs()));
                let active: Vec<usize> = a_data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v.abs() == max_abs { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    for i in active.iter().copied() {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    dnrm = dnrm / active_count;
                }
            }
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_ad_error(p));
                }
                if p == 1.0 {
                    for i in 0..len {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                } else {
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            dnrm = dnrm + sign * v.abs().powf(p_minus_one) * da_data[i];
                        }
                        dnrm = dnrm / nv.powf(p_minus_one);
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        let dnrm = tensor_from_data(vec![dnrm], &[]).map_err(to_ad_err)?;
        return Ok((nrm, dnrm));
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (nrm_data, _) = extract_data(&nrm)?;
    let (da_data, _) = extract_data(tangent)?;

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
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let ut_da = backend_mat_mul(ctx, &transpose(&u, m, k), k, m, da_b, n)?;
                let ut_da_v = backend_mat_mul(ctx, &ut_da, k, n, &v, k)?;
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
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let mut val = T::zero();
                for i in 0..m {
                    for j in 0..n {
                        val = val + u[i] * da_b[i + j * m] * v[j];
                    }
                }
                dnrm_data[batch] = val;
            }
        }
        NormKind::L1 => {
            // d||A||_1 = sum_i sign(A_ij) dA_ij on active max columns.
            // At ties, average uniformly over active columns.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut col_sums = vec![T::zero(); n];
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in 0..m {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    col_sums[j] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &col_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_cols: Vec<usize> = col_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &sum)| if sum == max_sum { Some(j) } else { None })
                    .collect();
                if active_cols.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for j in active_cols.iter().copied() {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
            }
        }
        NormKind::Inf => {
            // d||A||_inf = sum_j sign(A_ij) dA_ij on active max rows.
            // At ties, average uniformly over active rows.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut row_sums = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    row_sums[i] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &row_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_rows: Vec<usize> = row_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &sum)| if sum == max_sum { Some(i) } else { None })
                    .collect();
                if active_rows.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for i in active_rows.iter().copied() {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
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
mod tests;

#[cfg(test)]
mod eig_scalar_tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn eig_buffer_sizes_f32() {
        let (vals, vecs) = f32::eig_buffer_sizes(3);
        assert_eq!(vals, 6); // 2*n
        assert_eq!(vecs, 18); // 2*n*n
    }

    #[test]
    fn eig_ri_to_complex_f32() {
        let val_ri = [1.0_f32, 0.5, 2.0, -0.5];
        let vec_ri = [1.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut vals = [Complex32::new(0.0, 0.0); 2];
        let mut vecs = [Complex32::new(0.0, 0.0); 4];
        f32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
        assert!((vals[0].re - 1.0).abs() < 1e-6);
        assert!((vals[0].im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn eig_buffer_sizes_complex64() {
        let (vals, vecs) = Complex64::eig_buffer_sizes(3);
        assert_eq!(vals, 3); // n
        assert_eq!(vecs, 9); // n*n
    }

    #[test]
    fn eig_ri_to_complex_complex64() {
        let c = |re, im| Complex64::new(re, im);
        let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
        let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
        let mut vals = [Complex64::new(0.0, 0.0); 2];
        let mut vecs = [Complex64::new(0.0, 0.0); 4];
        Complex64::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
        assert!((vals[0].re - 1.0).abs() < 1e-12);
        assert!((vals[1].im + 0.5).abs() < 1e-12);
    }

    #[test]
    fn eig_buffer_sizes_complex32() {
        let (vals, vecs) = Complex32::eig_buffer_sizes(2);
        assert_eq!(vals, 2); // n
        assert_eq!(vecs, 4); // n*n
    }

    #[test]
    fn eig_ri_to_complex_complex32() {
        let c = |re, im| Complex32::new(re, im);
        let val_ri = [c(1.0, 0.5), c(2.0, -0.5)];
        let vec_ri = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(1.0, 0.0)];
        let mut vals = [Complex32::new(0.0, 0.0); 2];
        let mut vecs = [Complex32::new(0.0, 0.0); 4];
        Complex32::eig_ri_to_complex(2, &val_ri, &vec_ri, &mut vals, &mut vecs);
        assert!((vals[0].re - 1.0).abs() < 1e-6);
    }
}

#[allow(unexpected_cfgs)]
#[cfg(all(test, not(coverage)))]
mod internal_tests {
    use super::*;
    use num_complex::Complex64;

    // These tests assert that backend_mat_mul/backend_mat_mul_nn stay on the
    // prims bridge instead of calling backend-local matrix multiplication.

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

    #[test]
    fn validate_2d_and_validate_square_cover_error_and_batch_paths() {
        let vector =
            Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let err = validate_2d(&vector).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));

        let nonsquare = Tensor::from_slice(
            &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let err = validate_square(&nonsquare).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));

        let batched_square = Tensor::from_slice(
            &[1.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let (m, n, batch) = validate_2d(&batched_square).unwrap();
        assert_eq!((m, n), (2, 2));
        assert_eq!(batch, &[2]);
        let (n_square, batch_square) = validate_square(&batched_square).unwrap();
        assert_eq!(n_square, 2);
        assert_eq!(batch_square, &[2]);
    }

    #[test]
    fn backend_mat_mul_uses_prims_for_real_scalars() {
        let mut ctx = tenferro_prims::CpuContext::new(1);
        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0];

        let c = backend_mat_mul(&mut ctx, &a, 2, 3, &b, 2).unwrap();

        assert_eq!(c, vec![76.0, 100.0, 103.0, 136.0]);
    }

    #[test]
    fn backend_mat_mul_nn_uses_prims_for_real_scalars() {
        let mut ctx = tenferro_prims::CpuContext::new(1);
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];

        let c = backend_mat_mul_nn(&mut ctx, &a, &b, 2).unwrap();

        assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn backend_mat_mul_nn_uses_prims_for_complex_scalars() {
        let mut ctx = tenferro_prims::CpuContext::new(1);
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let b = vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ];

        let c = backend_mat_mul_nn(&mut ctx, &a, &b, 2).unwrap();

        assert_eq!(
            c,
            vec![
                Complex64::new(23.0, 0.0),
                Complex64::new(34.0, 0.0),
                Complex64::new(31.0, 0.0),
                Complex64::new(46.0, 0.0),
            ]
        );
    }

    trait TestScalar: LinalgScalar {
        fn from_f64(v: f64) -> Self;
    }

    impl TestScalar for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }

    impl TestScalar for Complex64 {
        fn from_f64(v: f64) -> Self {
            Self::new(v, 0.0)
        }
    }

    fn make<T: TestScalar>(data: &[f64], dims: &[usize]) -> Tensor<T> {
        let typed: Vec<T> = data.iter().map(|&v| T::from_f64(v)).collect();
        Tensor::from_slice(&typed, dims, MemoryOrder::ColumnMajor).unwrap()
    }

    fn run_generic_context_solve_dispatch_smoke<T, C>(mut ctx: C, expect_device_error: bool)
    where
        T: TestScalar,
        C: backend::TensorLinalgContextFor<T>,
    {
        let a = make::<T>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let b = make::<T>(&[4.0, 7.0], &[2]);
        match solve(&mut ctx, &a, &b) {
            Ok(x) => {
                assert!(!expect_device_error);
                assert_eq!(x.dims(), &[2]);
            }
            Err(err) => {
                assert!(expect_device_error);
                assert!(matches!(err, Error::DeviceError(_)));
            }
        }
    }

    #[cfg(feature = "linalg-faer")]
    #[test]
    fn solve_dispatch_is_generic_over_cpu_context_and_scalar() {
        run_generic_context_solve_dispatch_smoke::<f64, _>(
            tenferro_prims::CpuContext::new(1),
            false,
        );
        run_generic_context_solve_dispatch_smoke::<Complex64, _>(
            tenferro_prims::CpuContext::new(1),
            false,
        );
    }

    #[cfg(all(feature = "linalg-lapack", feature = "provider-src"))]
    #[test]
    fn solve_dispatch_is_generic_over_cpu_context_and_scalar_with_lapack_provider() {
        run_generic_context_solve_dispatch_smoke::<f64, _>(
            tenferro_prims::CpuContext::new(1),
            false,
        );
        run_generic_context_solve_dispatch_smoke::<Complex64, _>(
            tenferro_prims::CpuContext::new(1),
            false,
        );
    }

    #[test]
    fn solve_dispatch_is_generic_over_cuda_context_and_scalar() {
        run_generic_context_solve_dispatch_smoke::<f64, _>(
            tenferro_prims::CudaContext::new(),
            true,
        );
        run_generic_context_solve_dispatch_smoke::<Complex64, _>(
            tenferro_prims::CudaContext::new(),
            true,
        );
    }

    #[test]
    fn solve_dispatch_is_generic_over_hip_context_and_scalar() {
        run_generic_context_solve_dispatch_smoke::<f64, _>(
            tenferro_prims::RocmContext::new(),
            true,
        );
        run_generic_context_solve_dispatch_smoke::<Complex64, _>(
            tenferro_prims::RocmContext::new(),
            true,
        );
    }

    #[test]
    fn inv_dispatch_with_cuda_context_returns_device_error() {
        let mut ctx = tenferro_prims::CudaContext::new();
        let a = make::<f64>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let err = inv(&mut ctx, &a).unwrap_err();
        assert!(matches!(err, Error::DeviceError(_)));
    }
}
