//! Tensor-level backend trait and result types for linalg decompositions.
//!
//! This module defines [`TensorLinalgBackend<T>`], the tensor-aware backend
//! interface that accepts [`Tensor<T>`] at the boundary instead of raw slices.
//! Result types ([`QrTensorResult`], [`SvdTensorResult`], etc.) are also
//! defined here.

use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::LinalgScalar;

// ============================================================================
// Result types
// ============================================================================

/// Result of a tensor-level QR decomposition.
///
/// Holds the thin factors `Q` and `R` such that `A = Q R`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::QrTensorResult;
/// use tenferro_tensor::Tensor;
///
/// let result: QrTensorResult<f64> = todo!();
/// let _q: &Tensor<f64> = &result.q;
/// let _r: &Tensor<f64> = &result.r;
/// ```
#[derive(Clone)]
pub struct QrTensorResult<T: LinalgScalar> {
    /// The thin orthonormal factor.
    pub q: Tensor<T>,
    /// The thin upper-triangular factor.
    pub r: Tensor<T>,
}

/// Result of a tensor-level thin SVD.
///
/// Holds `U`, singular values `S`, and `Vt` such that `A = U diag(S) Vt`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::SvdTensorResult;
/// use tenferro_tensor::Tensor;
///
/// let result: SvdTensorResult<f64> = todo!();
/// let _u: &Tensor<f64> = &result.u;
/// let _s: &Tensor<f64> = &result.s;
/// let _vt: &Tensor<f64> = &result.vt;
/// ```
#[derive(Clone)]
pub struct SvdTensorResult<T: LinalgScalar> {
    /// The thin left singular vectors.
    pub u: Tensor<T>,
    /// Singular values in descending order.
    pub s: Tensor<T::Real>,
    /// The conjugate transpose of the right singular vectors.
    pub vt: Tensor<T>,
}

/// Result of a tensor-level LU factorization.
///
/// Holds `L`, `U`, and the pivot vector from partial pivoting.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::LuTensorResult;
///
/// let result: LuTensorResult<f64> = todo!();
/// let _pivots: &[i32] = &result.pivots;
/// ```
#[derive(Clone)]
pub struct LuTensorResult<T: LinalgScalar> {
    /// The unit-lower factor.
    pub l: Tensor<T>,
    /// The upper factor.
    pub u: Tensor<T>,
    /// Host-side LU pivots.
    pub pivots: Vec<i32>,
}

/// Result of a tensor-level Hermitian eigendecomposition.
///
/// Holds real-valued eigenvalues and eigenvectors such that
/// `A = V diag(lambda) V^H`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::EigenTensorResult;
/// use tenferro_tensor::Tensor;
///
/// let result: EigenTensorResult<f64> = todo!();
/// let _values: &Tensor<f64> = &result.values;
/// let _vectors: &Tensor<f64> = &result.vectors;
/// ```
#[derive(Clone)]
pub struct EigenTensorResult<T: LinalgScalar> {
    /// Eigenvalues in ascending order.
    pub values: Tensor<T::Real>,
    /// Eigenvectors stored as columns.
    pub vectors: Tensor<T>,
}

/// Result of a tensor-level general eigendecomposition.
///
/// The general eigendecomposition always returns complex-valued eigenpairs,
/// even for real input matrices.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::EigTensorResult;
/// use tenferro_tensor::Tensor;
///
/// let result: EigTensorResult<f64> = todo!();
/// let _values: &Tensor<num_complex::Complex64> = &result.values;
/// let _vectors: &Tensor<num_complex::Complex64> = &result.vectors;
/// ```
#[derive(Clone)]
pub struct EigTensorResult<T>
where
    T: LinalgScalar,
{
    /// Complex eigenvalues.
    pub values: Tensor<T::Complex>,
    /// Complex eigenvectors stored as columns.
    pub vectors: Tensor<T::Complex>,
}

// ============================================================================
// TensorLinalgBackend trait
// ============================================================================

/// Tensor-level backend interface for linalg decompositions and solves.
///
/// This trait mirrors the structure of `tenferro-prims::TensorPrims`:
/// - the backend type defines capability
/// - the associated context owns execution resources
/// - callers pass `&mut Context` explicitly
///
/// The trait is op-specific rather than using a descriptor/plan pattern,
/// because linalg ops need multi-output results, mixed dtypes, and pivot
/// metadata.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{CpuTensorLinalgBackend, TensorLinalgBackend};
/// use tenferro_tensor::Tensor;
///
/// let mut ctx = tenferro_prims::CpuContext::new(1);
/// let a: Tensor<f64> = todo!();
/// let b: Tensor<f64> = todo!();
/// let _x = <CpuTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(&mut ctx, &a, &b).unwrap();
/// ```
pub trait TensorLinalgBackend<T: LinalgScalar> {
    /// Backend-specific execution context.
    type Context;

    /// Solve a dense linear system `A x = b`.
    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Solve a triangular system `A x = b`.
    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;

    /// Compute the thin QR decomposition.
    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>>;

    /// Compute the thin singular value decomposition.
    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>>;

    /// Compute the LU factorization with pivots.
    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>>;

    /// Compute the Cholesky factor.
    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>>;

    /// Compute the Hermitian eigendecomposition.
    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>>;

    /// Compute the general eigendecomposition.
    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>>;
}

#[cfg(test)]
mod tests;
