//! CPU tensor linalg backend and context.
//!
//! The actual provider implementation is selected at compile time via
//! `linalg-faer` or `linalg-lapack` features.

use super::tensor_api::TensorLinalgBackend;
use super::tensor_context::TensorLinalgContextFor;
use crate::LinalgScalar;

#[cfg(feature = "linalg-faer")]
use super::cpu_faer;

/// CPU execution context for tensor linalg operations.
///
/// Owns reusable workspace state for the selected CPU provider
/// (faer or LAPACK). Create one per thread or pass explicitly.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::CpuTensorLinalgContext;
///
/// let mut ctx = CpuTensorLinalgContext::new();
/// ```
#[derive(Debug)]
pub struct CpuTensorLinalgContext {
    #[cfg(feature = "linalg-faer")]
    pub(crate) faer_backend: crate::backend::faer_backend::FaerBackend,
}

impl CpuTensorLinalgContext {
    /// Create a new CPU tensor linalg context.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg::backend::CpuTensorLinalgContext;
    ///
    /// let _ctx = CpuTensorLinalgContext::new();
    /// ```
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "linalg-faer")]
            faer_backend: crate::backend::faer_backend::FaerBackend::new(),
        }
    }
}

impl Default for CpuTensorLinalgContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker type for the CPU tensor linalg backend.
///
/// The backend type provides the trait implementation, while
/// [`CpuTensorLinalgContext`] owns backend-local execution state.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::CpuTensorLinalgBackend;
///
/// let _backend = CpuTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuTensorLinalgBackend;

#[cfg(feature = "linalg-faer")]
impl<T> TensorLinalgBackend<T> for CpuTensorLinalgBackend
where
    T: LinalgScalar,
    crate::backend::faer_backend::FaerBackend: crate::backend::LinalgBackend<T, Real = T::Real>,
{
    type Context = CpuTensorLinalgContext;

    fn solve(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::solve(ctx, a, b)
    }

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
        upper: bool,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::solve_triangular(ctx, a, b, upper)
    }

    fn qr(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::QrTensorResult<T>> {
        cpu_faer::qr(ctx, a)
    }

    fn thin_svd(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::SvdTensorResult<T>> {
        cpu_faer::thin_svd(ctx, a)
    }

    fn lu_factor(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorResult<T>> {
        cpu_faer::lu_factor(ctx, a)
    }

    fn cholesky(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::cholesky(ctx, a)
    }

    fn eigen_sym(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigenTensorResult<T>> {
        cpu_faer::eigen_sym(ctx, a)
    }

    fn eig(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigTensorResult<T>> {
        cpu_faer::eig(ctx, a)
    }
}

#[cfg(feature = "linalg-faer")]
impl<T> TensorLinalgContextFor<T> for CpuTensorLinalgContext
where
    T: LinalgScalar,
    crate::backend::faer_backend::FaerBackend: crate::backend::LinalgBackend<T, Real = T::Real>,
{
    type Backend = CpuTensorLinalgBackend;
}
