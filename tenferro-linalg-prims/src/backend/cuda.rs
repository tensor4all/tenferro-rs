mod cholesky;
mod lu;
mod qr;
mod runtime;
mod scalar_type;
mod solve;
mod wrappers;

use tenferro_device::Result;
use tenferro_tensor::Tensor;

use super::TensorLinalgContextFor;
use crate::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, LinalgCapabilityOp,
    LuTensorExResult, LuTensorResult, QrTensorResult, SolveTensorExResult, SvdTensorResult,
    TensorLinalgPrims,
};
use scalar_type::CudaLinalgScalar;

/// Marker type for the CUDA tensor linalg backend.
///
/// # Examples
///
/// ```ignore
/// let _backend = tenferro_linalg_prims::backend::CudaTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaTensorLinalgBackend;

fn unsupported<T, S: CudaLinalgScalar>(op: &str) -> Result<T> {
    let _ = S::cuda_data_type();
    runtime::unsupported(op)
}

impl<T: CudaLinalgScalar> TensorLinalgPrims<T> for CudaTensorLinalgBackend {
    type Context = tenferro_prims::CudaContext;

    fn has_linalg_support(op: LinalgCapabilityOp) -> bool {
        matches!(
            op,
            LinalgCapabilityOp::Solve
                | LinalgCapabilityOp::Qr
                | LinalgCapabilityOp::LuFactor
                | LinalgCapabilityOp::LuFactorEx
                | LinalgCapabilityOp::Cholesky
                | LinalgCapabilityOp::CholeskyEx
        ) && match op {
            LinalgCapabilityOp::Solve => solve::has_solve_support::<T>(),
            LinalgCapabilityOp::Qr => qr::has_qr_support::<T>(),
            LinalgCapabilityOp::LuFactor | LinalgCapabilityOp::LuFactorEx => {
                lu::has_lu_support::<T>()
            }
            LinalgCapabilityOp::Cholesky | LinalgCapabilityOp::CholeskyEx => {
                cholesky::has_cholesky_support::<T>()
            }
            _ => false,
        }
    }

    fn solve_ex(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
    ) -> Result<SolveTensorExResult<T>> {
        unsupported::<SolveTensorExResult<T>, T>("solve_ex")
    }

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        solve::solve(ctx, a, b)
    }

    fn solve_triangular(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
        _upper: bool,
    ) -> Result<Tensor<T>> {
        unsupported::<Tensor<T>, T>("solve_triangular")
    }

    fn qr(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<QrTensorResult<T>> {
        qr::qr(_ctx, _a)
    }

    fn thin_svd(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<SvdTensorResult<T>> {
        unsupported::<SvdTensorResult<T>, T>("thin_svd")
    }

    fn svdvals(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T::Real>> {
        unsupported::<Tensor<T::Real>, T>("svdvals")
    }

    fn lu_factor_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorExResult<T>> {
        lu::lu_factor_ex(_ctx, _a)
    }

    fn lu_factor(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        lu::lu_factor(_ctx, _a)
    }

    fn cholesky_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>> {
        cholesky::cholesky_ex(_ctx, _a)
    }

    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>> {
        cholesky::cholesky(ctx, a)
    }

    fn eigen_sym(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigenTensorResult<T>> {
        unsupported::<EigenTensorResult<T>, T>("eigen_sym")
    }

    fn eig(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigTensorResult<T>> {
        unsupported::<EigTensorResult<T>, T>("eig")
    }
}

impl<T: CudaLinalgScalar> TensorLinalgContextFor<T> for tenferro_prims::CudaContext {
    type Backend = CudaTensorLinalgBackend;
}

#[cfg(test)]
mod tests;
