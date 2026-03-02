//! CUDA tensor linalg backend stub.
//!
//! This module defines the future CUDA backend and context types.
//! All methods currently return `Error::UnsupportedDevice`.

use super::tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgBackend,
};
use crate::LinalgScalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

/// CUDA execution context for tensor linalg operations (stub).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::CudaTensorLinalgContext;
///
/// let _ctx = CudaTensorLinalgContext::new();
/// ```
#[derive(Debug)]
pub struct CudaTensorLinalgContext;

impl CudaTensorLinalgContext {
    /// Create a new CUDA tensor linalg context (stub).
    pub fn new() -> Self {
        Self
    }
}

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

impl<T: LinalgScalar> TensorLinalgBackend<T> for CudaTensorLinalgBackend {
    type Context = CudaTensorLinalgContext;

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
