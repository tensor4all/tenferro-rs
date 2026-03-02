//! CUDA tensor linalg backend stub.
//!
//! This module defines the future CUDA backend type.
//! All methods currently return `Error::DeviceError`.

use super::tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgBackend,
};
use crate::LinalgScalar;
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

impl<T: LinalgScalar> TensorLinalgBackend<T> for CudaTensorLinalgBackend {
    type Context = tenferro_prims::CudaContext;

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
    fn cuda_stubs_return_device_error() {
        let mut ctx = tenferro_prims::CudaContext::new();
        let a = dummy_tensor();
        let b = dummy_tensor();
        assert!(CudaTensorLinalgBackend::solve(&mut ctx, &a, &b).is_err());
        assert!(CudaTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).is_err());
        assert!(CudaTensorLinalgBackend::qr(&mut ctx, &a).is_err());
        assert!(CudaTensorLinalgBackend::thin_svd(&mut ctx, &a).is_err());
        assert!(CudaTensorLinalgBackend::lu_factor(&mut ctx, &a).is_err());
        assert!(CudaTensorLinalgBackend::cholesky(&mut ctx, &a).is_err());
        assert!(CudaTensorLinalgBackend::eigen_sym(&mut ctx, &a).is_err());
        assert!(CudaTensorLinalgBackend::eig(&mut ctx, &a).is_err());
    }
}
