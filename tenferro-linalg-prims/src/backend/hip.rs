use tenferro_device::Result;
use tenferro_tensor::Tensor;

use super::TensorLinalgContextFor;
use crate::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, KernelLinalgScalar,
    LinalgCapabilityOp, LuTensorExResult, LuTensorResult, QrTensorResult, SolveTensorExResult,
    SvdTensorResult, TensorLinalgPrims,
};

mod cholesky;
mod common;
mod eig;
mod lu;
mod qr;
mod solve;
mod solve_triangular;
mod svdvals;
mod thin_svd;

/// Marker type for the HIP tensor linalg backend.
///
/// # Examples
///
/// ```ignore
/// let _backend = tenferro_linalg_prims::backend::HipTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct HipTensorLinalgBackend;

impl<T: KernelLinalgScalar> TensorLinalgPrims<T> for HipTensorLinalgBackend {
    type Context = tenferro_prims::RocmContext;

    fn has_linalg_support(_op: LinalgCapabilityOp) -> bool {
        false
    }

    fn solve_ex(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
    ) -> Result<SolveTensorExResult<T>> {
        solve::solve_ex(_ctx, _a, _b)
    }

    fn solve(_ctx: &mut Self::Context, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> {
        solve::solve(_ctx, _a, _b)
    }

    fn lu_solve(
        _ctx: &mut Self::Context,
        _factors: &Tensor<T>,
        _pivots: &Tensor<i32>,
        _b: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        solve::lu_solve(_ctx, _factors, _pivots, _b)
    }

    fn solve_triangular(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
        _upper: bool,
    ) -> Result<Tensor<T>> {
        solve_triangular::solve_triangular(_ctx, _a, _b, _upper)
    }

    fn qr(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<QrTensorResult<T>> {
        qr::qr(_ctx, _a)
    }

    fn thin_svd(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<SvdTensorResult<T>> {
        thin_svd::thin_svd(_ctx, _a)
    }

    fn svdvals(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T::Real>> {
        svdvals::svdvals(_ctx, _a)
    }

    fn lu_factor_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorExResult<T>> {
        lu::lu_factor_ex(_ctx, _a)
    }

    fn lu_factor(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        lu::lu_factor(_ctx, _a)
    }

    fn lu_factor_no_pivot(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        lu::lu_factor_no_pivot(_ctx, _a)
    }

    fn cholesky_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>> {
        cholesky::cholesky_ex(_ctx, _a)
    }

    fn cholesky(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T>> {
        cholesky::cholesky(_ctx, _a)
    }

    fn eigen_sym(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigenTensorResult<T>> {
        eig::eigen_sym(_ctx, _a)
    }

    fn eig(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigTensorResult<T>> {
        eig::eig(_ctx, _a)
    }
}

impl<T: KernelLinalgScalar> TensorLinalgContextFor<T> for tenferro_prims::RocmContext {
    type Backend = HipTensorLinalgBackend;
}
