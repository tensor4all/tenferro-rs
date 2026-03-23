use super::*;

/// Structured Cholesky result with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero means success.
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
/// assert_eq!(result.info.len(), 1);
/// ```
#[derive(Debug)]
pub struct CholeskyExResult<T: Scalar> {
    /// Lower-triangular Cholesky factor.
    pub l: Tensor<T>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Structured inverse result with numerical status information.
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
/// assert_eq!(result.info.len(), 1);
/// ```
#[derive(Debug)]
pub struct InvExResult<T: Scalar> {
    /// Inverse matrix.
    pub inverse: Tensor<T>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Structured solve result with numerical status information.
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
/// assert_eq!(result.info.len(), 1);
/// ```
#[derive(Debug)]
pub struct SolveExResult<T: Scalar> {
    /// Solution tensor.
    pub solution: Tensor<T>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Packed LU factorization result.
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
/// assert_eq!(result.pivots.len(), 2);
/// ```
#[derive(Debug)]
pub struct LuFactorResult<T: Scalar> {
    /// Packed LU factors with the same shape as the input.
    pub factors: Tensor<T>,
    /// Backend pivot tensor in 1-indexed step-pivot form.
    pub pivots: Tensor<i32>,
}

/// Packed LU factorization result with numerical status information.
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
/// assert_eq!(result.info.len(), 1);
/// ```
#[derive(Debug)]
pub struct LuFactorExResult<T: Scalar> {
    /// Packed LU factors with the same shape as the input.
    pub factors: Tensor<T>,
    /// Backend pivot tensor in 1-indexed step-pivot form.
    pub pivots: Tensor<i32>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Sign-and-log-determinant result: `det(A) = sign * exp(logabsdet)`.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::slogdet;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[3, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let result = slogdet(&mut ctx, &a).unwrap();
/// assert_eq!(result.sign.ndim(), 0);
/// ```
#[derive(Debug)]
pub struct SlogdetResult<T: Scalar, R: Scalar = T> {
    /// Sign of the determinant.
    pub sign: Tensor<T>,
    /// Log of the absolute determinant.
    pub logabsdet: Tensor<R>,
}
