use super::*;

/// Eigendecomposition result: `A * V = V * diag(values)`.
///
/// Only valid for square matrices.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::eigen;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[3, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let result = eigen(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// ```
#[derive(Debug)]
pub struct EigenResult<T: Scalar, R: Scalar = T> {
    /// Eigenvalues.
    pub values: Tensor<R>,
    /// Right eigenvectors stored by columns.
    pub vectors: Tensor<T>,
}

/// Result of general eigendecomposition (always complex-valued).
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::{eig, EigResult};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[3, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let result: EigResult<f64> = eig(&mut ctx, &a).unwrap();
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
#[derive(Debug)]
pub struct EigResult<R: LinalgScalar<Real = R> + num_traits::Float> {
    /// Complex eigenvalues.
    pub values: Tensor<num_complex::Complex<R>>,
    /// Complex right eigenvectors.
    pub vectors: Tensor<num_complex::Complex<R>>,
}

/// Norm kind for [`norm`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::NormKind;
///
/// let fro = NormKind::Fro;
/// let lp = NormKind::Lp(3.0);
/// assert!(matches!(fro, NormKind::Fro));
/// assert!(matches!(lp, NormKind::Lp(_)));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum NormKind {
    /// Frobenius norm or vector L2 norm.
    Fro,
    /// Nuclear norm.
    Nuclear,
    /// Spectral norm.
    Spectral,
    /// L1 norm.
    L1,
    /// Infinity norm.
    Inf,
    /// General Lp norm.
    Lp(f64),
}
