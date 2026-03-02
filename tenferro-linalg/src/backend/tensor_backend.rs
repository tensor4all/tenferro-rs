//! Tensor-level backend API for linalg decompositions and solves.
//!
//! This module defines the proposed tensor-level backend surface that sits
//! above primitive tensor ops and below the public `tenferro-linalg` APIs.
//! The API is intentionally operation-specific, matching the structure of
//! decomposition and solve kernels more closely than `TensorPrims`.

use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::LinalgScalar;

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
/// The first concrete implementation will keep pivots on the host as
/// `Vec<i32>` rather than introducing a tensor metadata abstraction up front.
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

/// Tensor-level backend interface for linalg decompositions and solves.
///
/// This trait is the planned tensor-aware counterpart to the existing
/// slice-based [`crate::backend::LinalgBackend`] trait.
///
/// Implementations should be device-aware and should operate directly on
/// `Tensor<T>` values rather than on extracted CPU slices.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{
///     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
/// };
/// use tenferro_tensor::Tensor;
///
/// let mut ctx = FaerTensorLinalgContext::new();
/// let a: Tensor<f64> = todo!();
/// let b: Tensor<f64> = todo!();
/// let _x = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(&mut ctx, &a, &b).unwrap();
/// ```
pub trait TensorLinalgBackend<T: LinalgScalar> {
    /// Backend-specific execution context.
    type Context;

    /// Solve a dense linear system `A x = b`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let b: Tensor<f64> = todo!();
    /// let _x = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::solve(&mut ctx, &a, &b).unwrap();
    /// ```
    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Solve a triangular system `A x = b`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let b: Tensor<f64> = todo!();
    /// let _x = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::solve_triangular(&mut ctx, &a, &b, true).unwrap();
    /// ```
    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;

    /// Compute the thin QR decomposition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _qr = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::qr(&mut ctx, &a).unwrap();
    /// ```
    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>>;

    /// Compute the thin singular value decomposition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _svd = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::thin_svd(&mut ctx, &a).unwrap();
    /// ```
    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>>;

    /// Compute the LU factorization with pivots.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _lu = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::lu_factor(&mut ctx, &a).unwrap();
    /// ```
    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>>;

    /// Compute the Cholesky factor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _l = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::cholesky(&mut ctx, &a).unwrap();
    /// ```
    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>>;

    /// Compute the Hermitian eigendecomposition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _eig = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::eigen_sym(&mut ctx, &a).unwrap();
    /// ```
    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>>;

    /// Compute the general eigendecomposition.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tenferro_linalg::backend::{
    /// #     FaerTensorLinalgBackend, FaerTensorLinalgContext, TensorLinalgBackend,
    /// # };
    /// # use tenferro_tensor::Tensor;
    /// let mut ctx = FaerTensorLinalgContext::new();
    /// let a: Tensor<f64> = todo!();
    /// let _eig = <FaerTensorLinalgBackend as TensorLinalgBackend<f64>>::eig(&mut ctx, &a).unwrap();
    /// ```
    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>>;
}

/// CPU execution context for the future faer-backed tensor linalg adapter.
///
/// This type is intentionally API-only for now. The first implementation will
/// own reusable faer-side scratch state here instead of using the backend type
/// itself as the context object.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::FaerTensorLinalgContext;
///
/// let mut ctx = FaerTensorLinalgContext::new();
/// let _ = &mut ctx;
/// ```
#[cfg(feature = "faer")]
#[derive(Debug)]
pub struct FaerTensorLinalgContext {
    _inner: super::FaerBackend,
}

#[cfg(feature = "faer")]
impl FaerTensorLinalgContext {
    /// Create a new tensor-level faer execution context.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg::backend::FaerTensorLinalgContext;
    ///
    /// let _ctx = FaerTensorLinalgContext::new();
    /// ```
    pub fn new() -> Self {
        todo!("FaerTensorLinalgContext is an API skeleton; implementation will follow issue #246")
    }
}

/// Marker type for the future faer-backed tensor linalg adapter.
///
/// The backend type provides the trait implementation, while
/// [`FaerTensorLinalgContext`] owns backend-local execution state.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::FaerTensorLinalgBackend;
///
/// let _backend = FaerTensorLinalgBackend;
/// ```
#[cfg(feature = "faer")]
#[derive(Debug, Default, Clone, Copy)]
pub struct FaerTensorLinalgBackend;

#[cfg(feature = "faer")]
impl<T> TensorLinalgBackend<T> for FaerTensorLinalgBackend
where
    T: LinalgScalar,
    super::FaerBackend: super::LinalgBackend<T, Real = T::Real>,
{
    type Context = FaerTensorLinalgContext;

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let _ = (ctx, a, b);
        todo!("FaerTensorLinalgBackend::solve is an API skeleton")
    }

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>> {
        let _ = (ctx, a, b, upper);
        todo!("FaerTensorLinalgBackend::solve_triangular is an API skeleton")
    }

    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::qr is an API skeleton")
    }

    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::thin_svd is an API skeleton")
    }

    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::lu_factor is an API skeleton")
    }

    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::cholesky is an API skeleton")
    }

    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::eigen_sym is an API skeleton")
    }

    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>> {
        let _ = (ctx, a);
        todo!("FaerTensorLinalgBackend::eig is an API skeleton")
    }
}
