//! CUDA tensor linalg backend stub.
//!
//! This module defines the future CUDA backend type.
//! All methods currently return `Error::DeviceError`.

use super::tensor_api::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, LuTensorExResult, LuTensorResult,
    QrTensorResult, SolveTensorExResult, SvdTensorResult, TensorLinalgBackend,
};
use crate::KernelLinalgScalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

/// Marker type for the CUDA tensor linalg backend (stub).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::CudaTensorLinalgBackend;
///
/// let _backend = CudaTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaTensorLinalgBackend;

fn unsupported<T>() -> Result<T> {
    Err(Error::DeviceError(
        "CUDA linalg backend is not yet implemented".into(),
    ))
}

impl<T: KernelLinalgScalar> TensorLinalgBackend<T> for CudaTensorLinalgBackend {
    type Context = tenferro_prims::CudaContext;

    fn has_linalg_support(_op: super::tensor_api::LinalgCapabilityOp) -> bool {
        false
    }

    fn solve_ex(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
    ) -> Result<SolveTensorExResult<T>> {
        unsupported()
    }

    fn solve(_ctx: &mut Self::Context, _a: &Tensor<T>, _b: &Tensor<T>) -> Result<Tensor<T>> {
        unsupported()
    }
    fn solve_triangular(
        _ctx: &mut Self::Context,
        _a: &Tensor<T>,
        _b: &Tensor<T>,
        _upper: bool,
    ) -> Result<Tensor<T>> {
        unsupported()
    }
    fn qr(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<QrTensorResult<T>> {
        unsupported()
    }
    fn thin_svd(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<SvdTensorResult<T>> {
        unsupported()
    }
    fn svdvals(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T::Real>> {
        unsupported()
    }
    fn lu_factor_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorExResult<T>> {
        unsupported()
    }
    fn lu_factor(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        unsupported()
    }
    fn cholesky_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>> {
        unsupported()
    }
    fn cholesky(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T>> {
        unsupported()
    }
    fn eigen_sym(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigenTensorResult<T>> {
        unsupported()
    }
    fn eig(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigTensorResult<T>> {
        unsupported()
    }
}

#[cfg(test)]
mod tests;
