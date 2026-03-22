use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use super::TensorLinalgContextFor;
use crate::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, KernelLinalgScalar,
    LinalgCapabilityOp, LuTensorExResult, LuTensorResult, QrTensorResult, SolveTensorExResult,
    SvdTensorResult, TensorLinalgPrims,
};

/// Marker type for the HIP tensor linalg backend.
///
/// # Examples
///
/// ```ignore
/// let _backend = tenferro_linalg_prims::backend::HipTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct HipTensorLinalgBackend;

fn unsupported<T>() -> Result<T> {
    Err(Error::DeviceError(
        "HIP linalg backend is not yet implemented".into(),
    ))
}

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

impl<T: KernelLinalgScalar> TensorLinalgContextFor<T> for tenferro_prims::RocmContext {
    type Backend = HipTensorLinalgBackend;
}
