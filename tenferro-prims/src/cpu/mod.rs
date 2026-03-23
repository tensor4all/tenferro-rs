use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

mod analytic;
mod batched_gemm;
mod common;
mod complex_real;
mod complex_scale;
mod context;
mod contract;
mod execution;
mod family_reduction;
mod gemm_support;
mod layout_fusion;
mod metadata;
mod metadata_cast;
mod plan;
mod planning;
mod reduction;
mod scalar;
#[cfg(feature = "gemm-blas")]
mod scratch;

pub use analytic::CpuAnalyticPlan;
pub use complex_real::CpuComplexRealPlan;
pub use complex_scale::CpuComplexScalePlan;
pub use context::{CpuBackend, CpuContext};
pub use plan::CpuPlan;
pub use scalar::CpuScalarPlan;

/// Convert a CPU tensor to an immutable strided view.
pub(crate) fn tensor_to_view<T: Scalar>(t: &Tensor<T>) -> Result<StridedView<'_, T>> {
    let data = t
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedView::new(data, t.dims(), t.strides(), t.offset())
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Convert a CPU tensor to a mutable strided view.
pub(crate) fn tensor_to_view_mut<T: Scalar>(t: &mut Tensor<T>) -> Result<StridedViewMut<'_, T>> {
    let dims = t.dims().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.offset();
    let data = t
        .buffer_mut()
        .as_mut_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedViewMut::new(data, &dims, &strides, offset)
        .map_err(|e| Error::StrideError(format!("{e}")))
}

#[cfg(test)]
mod tests;
