use super::*;

/// SVD result: `A = U * diag(S) * Vt`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `u`: shape `(m, k, *)`
/// - `s`: shape `(k, *)` in descending order
/// - `vt`: shape `(k, n, *)`
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::svd;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let result = svd(&mut ctx, &a, None).unwrap();
/// assert_eq!(result.s.ndim(), 1);
/// ```
#[derive(Debug)]
pub struct SvdResult<T: Scalar, R: Scalar = T> {
    /// Left singular vectors. Shape: `(m, k, *)`.
    pub u: Tensor<T>,
    /// Singular values. Shape: `(k, *)`.
    pub s: Tensor<R>,
    /// Right singular vectors (conjugate-transposed). Shape: `(k, n, *)`.
    pub vt: Tensor<T>,
}

/// Options for truncated SVD.
///
/// When both fields are specified, the more restrictive condition applies.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::SvdOptions;
///
/// let opts = SvdOptions {
///     max_rank: Some(8),
///     cutoff: Some(1e-12),
/// };
/// assert_eq!(opts.max_rank, Some(8));
/// ```
#[derive(Debug, Clone, Default)]
pub struct SvdOptions {
    /// Maximum number of singular values to keep.
    pub max_rank: Option<usize>,
    /// Discard singular values below this threshold.
    pub cutoff: Option<f64>,
}

/// QR decomposition result: `A = Q * R`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `q`: shape `(m, k, *)`
/// - `r`: shape `(k, n, *)`
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::qr;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[4, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let result = qr(&mut ctx, &a).unwrap();
/// assert_eq!(result.q.dims(), &[4, 3]);
/// assert_eq!(result.r.dims(), &[3, 3]);
/// ```
#[derive(Debug)]
pub struct QrResult<T: Scalar> {
    /// Orthonormal factor. Shape: `(m, k, *)`.
    pub q: Tensor<T>,
    /// Upper-triangular factor. Shape: `(k, n, *)`.
    pub r: Tensor<T>,
}

/// Pivoting strategy for LU decomposition.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LuPivot;
///
/// assert_eq!(LuPivot::default(), LuPivot::Partial);
/// assert_eq!(LuPivot::NoPivot, LuPivot::NoPivot);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LuPivot {
    /// Partial row pivoting.
    #[default]
    Partial,
    /// No pivoting.
    NoPivot,
}

/// LU decomposition result: `A = P * L * U`.
///
/// For an input of shape `(m, n, *)` with `k = min(m, n)`:
///
/// - `p`: permutation matrix tensor of shape `(m, m, *)`
///   or an empty tensor of shape `[0]` when pivoting is disabled
/// - `l`: shape `(m, k, *)`
/// - `u`: shape `(k, n, *)`
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
/// assert_eq!(result.p.dims(), &[2, 2]);
/// ```
#[derive(Debug)]
pub struct LuResult<T: Scalar> {
    /// Row permutation matrix tensor.
    pub p: Tensor<T>,
    /// Unit lower-triangular factor.
    pub l: Tensor<T>,
    /// Upper-triangular factor.
    pub u: Tensor<T>,
}

/// Gradient result for `solve_rrule`: cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::solve_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mem = LogicalMemorySpace::MainMemory;
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col);
/// let grad = solve_rrule(&mut ctx, &a, &b, &cotangent).unwrap();
/// assert_eq!(grad.a.dims(), &[3, 3]);
/// assert_eq!(grad.b.dims(), &[3]);
/// ```
#[derive(Debug)]
pub struct SolveGrad<T: Scalar> {
    /// Cotangent for the coefficient matrix `A`.
    pub a: Tensor<T>,
    /// Cotangent for the right-hand side `b`.
    pub b: Tensor<T>,
}

/// Least-squares result.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lstsq;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 1.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let b = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let result = lstsq(&mut ctx, &a, &b).unwrap();
/// assert_eq!(result.x.dims(), &[2]);
/// ```
#[derive(Debug)]
pub struct LstsqResult<T: Scalar> {
    /// Least-squares solution.
    pub x: Tensor<T>,
    /// Residual tensor `b - Ax`.
    pub residual: Tensor<T>,
}

/// Gradient result for `lstsq_rrule`: cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LstsqGrad;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let grad = LstsqGrad {
///     a: Tensor::<f64>::zeros(&[2, 2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     b: Tensor::<f64>::zeros(&[2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
/// };
/// assert_eq!(grad.a.ndim(), 2);
/// assert_eq!(grad.b.ndim(), 1);
/// ```
#[derive(Debug)]
pub struct LstsqGrad<T: Scalar> {
    /// Cotangent for the system matrix `A`.
    pub a: Tensor<T>,
    /// Cotangent for the right-hand side `b`.
    pub b: Tensor<T>,
}
