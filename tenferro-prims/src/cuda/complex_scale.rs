use std::marker::PhantomData;

use num_complex::{Complex32, Complex64, ComplexFloat};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use crate::cuda::CudaContext;
use crate::{
    validate_shape_count, validate_shape_eq, ComplexScalePrimsDescriptor, CudaBackend,
    TensorComplexScalePrims,
};

/// CUDA execution plan for the complex-by-real pointwise protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CudaComplexScalePlan;
/// let _ = std::mem::size_of::<CudaComplexScalePlan<Complex64>>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaComplexScalePlan<T: Scalar> {
    kind: CudaComplexScalePlanKind,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CudaComplexScalePlanKind {
    PointwiseMul,
}

trait RuntimeComplexScaleScalar: ComplexFloat + Scalar + 'static
where
    Self::Real: Scalar + Send + Sync,
{
    unsafe fn pointwise_scale_raw(
        runtime: &tenferro_device::cuda::runtime::CudaRuntime,
        alpha: Self,
        lhs: *const Self,
        rhs: *const Self::Real,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: Self,
    ) -> Result<()>;
}

impl RuntimeComplexScaleScalar for Complex32 {
    unsafe fn pointwise_scale_raw(
        runtime: &tenferro_device::cuda::runtime::CudaRuntime,
        alpha: Self,
        lhs: *const Self,
        rhs: *const Self::Real,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: Self,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_mul_complex32_real_f32_raw(
                alpha,
                lhs,
                dims,
                lhs_strides,
                lhs_offset,
                rhs,
                rhs_strides,
                rhs_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }
}

impl RuntimeComplexScaleScalar for Complex64 {
    unsafe fn pointwise_scale_raw(
        runtime: &tenferro_device::cuda::runtime::CudaRuntime,
        alpha: Self,
        lhs: *const Self,
        rhs: *const Self::Real,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: Self,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_mul_complex64_real_f64_raw(
                alpha,
                lhs,
                dims,
                lhs_strides,
                lhs_offset,
                rhs,
                rhs_strides,
                rhs_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }
}

fn ensure_cuda_tensor<T: Scalar>(tensor: &Tensor<T>, device_id: usize, label: &str) -> Result<()> {
    match tensor.logical_memory_space() {
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } if tensor_device == device_id => Ok(()),
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } => Err(Error::DeviceError(format!(
            "{label} is on CUDA device {tensor_device}, expected device {device_id}"
        ))),
        other => Err(Error::DeviceError(format!(
            "{label} is not resident on CUDA device {device_id}: {other:?}"
        ))),
    }
}

fn plan_complex_scale<Input>(
    desc: &ComplexScalePrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaComplexScalePlan<Input>>
where
    Input: RuntimeComplexScaleScalar,
    Input::Real: Scalar + Send + Sync,
{
    validate_shape_count(shapes, 3, "CudaComplexScalePointwiseMul")?;
    validate_shape_eq(shapes[0], shapes[1], "CudaComplexScalePointwiseMul lhs/rhs")?;
    validate_shape_eq(
        shapes[0],
        shapes[2],
        "CudaComplexScalePointwiseMul lhs/output",
    )?;
    match desc {
        ComplexScalePrimsDescriptor::PointwiseMul => Ok(CudaComplexScalePlan {
            kind: CudaComplexScalePlanKind::PointwiseMul,
            _marker: PhantomData,
        }),
    }
}

fn execute_complex_scale<Input>(
    ctx: &mut CudaContext,
    plan: &CudaComplexScalePlan<Input>,
    alpha: Input,
    lhs: &Tensor<Input>,
    rhs: &Tensor<Input::Real>,
    beta: Input,
    output: &mut Tensor<Input>,
) -> Result<()>
where
    Input: RuntimeComplexScaleScalar,
    Input::Real: Scalar + Send + Sync,
{
    let _ = &plan.kind;
    ensure_cuda_tensor(lhs, ctx.device_id(), "CudaComplexScalePointwiseMul lhs")?;
    ensure_cuda_tensor(rhs, ctx.device_id(), "CudaComplexScalePointwiseMul rhs")?;
    ensure_cuda_tensor(
        output,
        ctx.device_id(),
        "CudaComplexScalePointwiseMul output",
    )?;
    validate_shape_eq(lhs.dims(), rhs.dims(), "CudaComplexScalePointwiseMul rhs")?;
    validate_shape_eq(
        lhs.dims(),
        output.dims(),
        "CudaComplexScalePointwiseMul output",
    )?;

    let runtime = ctx.shared_runtime();
    let lhs_ptr = lhs
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("CUDA lhs buffer is not on GPU".into()))?
        as *const Input;
    let rhs_ptr = rhs
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("CUDA rhs buffer is not on GPU".into()))?
        as *const Input::Real;
    let dst_ptr = output
        .buffer_mut()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("CUDA output buffer is not on GPU".into()))?
        as *mut Input;

    unsafe {
        Input::pointwise_scale_raw(
            runtime.as_ref(),
            alpha,
            lhs_ptr,
            rhs_ptr,
            lhs.dims(),
            lhs.strides(),
            lhs.offset(),
            rhs.strides(),
            rhs.offset(),
            dst_ptr,
            output.strides(),
            output.offset(),
            beta,
        )
    }
}

impl<Input> TensorComplexScalePrims<Input> for CudaBackend
where
    Input: RuntimeComplexScaleScalar,
    Input::Real: Scalar + Send + Sync,
{
    type Plan = CudaComplexScalePlan<Input>;
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexScalePrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        plan_complex_scale(desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Input,
        lhs: &Tensor<Input>,
        rhs: &Tensor<Input::Real>,
        beta: Input,
        output: &mut Tensor<Input>,
    ) -> Result<()> {
        execute_complex_scale(ctx, plan, alpha, lhs, rhs, beta, output)
    }

    fn has_complex_scale_support(desc: ComplexScalePrimsDescriptor) -> bool {
        matches!(desc, ComplexScalePrimsDescriptor::PointwiseMul)
    }
}
