//! HIP tensor linalg backend stub.
//!
//! This module defines the future HIP backend and context types.
//! All methods currently return `Error::UnsupportedDevice`.

use super::tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgBackend,
};
use crate::LinalgScalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

/// HIP execution context for tensor linalg operations (stub).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::HipTensorLinalgContext;
///
/// let _ctx = HipTensorLinalgContext::new();
/// ```
#[derive(Debug)]
pub struct HipTensorLinalgContext;

impl HipTensorLinalgContext {
    /// Create a new HIP tensor linalg context (stub).
    pub fn new() -> Self {
        Self
    }
}

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

impl<T: LinalgScalar> TensorLinalgBackend<T> for HipTensorLinalgBackend {
    type Context = HipTensorLinalgContext;

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

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::MemoryOrder;

    fn dummy_tensor() -> Tensor<f64> {
        Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap()
    }

    #[test]
    fn hip_stubs_return_device_error() {
        let mut ctx = HipTensorLinalgContext::new();
        let a = dummy_tensor();
        let b = dummy_tensor();
        assert!(HipTensorLinalgBackend::solve(&mut ctx, &a, &b).is_err());
        assert!(HipTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).is_err());
        assert!(HipTensorLinalgBackend::qr(&mut ctx, &a).is_err());
        assert!(HipTensorLinalgBackend::thin_svd(&mut ctx, &a).is_err());
        assert!(HipTensorLinalgBackend::lu_factor(&mut ctx, &a).is_err());
        assert!(HipTensorLinalgBackend::cholesky(&mut ctx, &a).is_err());
        assert!(HipTensorLinalgBackend::eigen_sym(&mut ctx, &a).is_err());
        assert!(HipTensorLinalgBackend::eig(&mut ctx, &a).is_err());
    }
}
