//! HIP tensor linalg backend stub.
//!
//! This module defines the future HIP backend type.
//! All methods currently return `Error::DeviceError`.

use super::tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgBackend,
};
use super::tensor_context::TensorLinalgContextFor;
use crate::KernelLinalgScalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

/// Marker type for the HIP tensor linalg backend (stub).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::HipTensorLinalgBackend;
///
/// let _backend = HipTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct HipTensorLinalgBackend;

fn unsupported<T>() -> Result<T> {
    Err(Error::DeviceError(
        "HIP linalg backend is not yet implemented".into(),
    ))
}

impl<T: KernelLinalgScalar> TensorLinalgBackend<T> for HipTensorLinalgBackend {
    type Context = tenferro_prims::RocmContext;

    fn has_linalg_support(_op: super::tensor_api::LinalgCapabilityOp) -> bool {
        false
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
    fn lu_factor(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
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

impl<T: KernelLinalgScalar> TensorLinalgContextFor<T> for tenferro_prims::RocmContext {
    type Backend = HipTensorLinalgBackend;
}

#[cfg(test)]
mod tests;
