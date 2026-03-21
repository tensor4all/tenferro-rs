use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{
    cuda::runtime::{CudaRuntime, RealUnaryOp},
    Error, LogicalMemorySpace, Result,
};
use tenferro_tensor::Tensor;

use crate::{
    validate_execute_inputs, validate_shape_count, validate_shape_eq, AnalyticPrimsDescriptor,
    AnalyticUnaryOp, CudaBackend, TensorAnalyticPrims,
};

use super::CudaContext;

/// CUDA execution plan for the analytic protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CudaAnalyticPlan;
/// let _ = std::mem::size_of::<CudaAnalyticPlan<f64>>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaAnalyticPlan<T: Scalar> {
    op: AnalyticUnaryOp,
    _marker: PhantomData<T>,
}

trait RuntimeRealAnalyticScalar: Scalar + 'static {
    unsafe fn pointwise_unary_raw(
        runtime: &CudaRuntime,
        op: RealUnaryOp,
        alpha: Self,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>;
}

trait CudaRealAnalyticScalar: Scalar + Conjugate + RuntimeRealAnalyticScalar + 'static {}

impl CudaRealAnalyticScalar for f32 {}
impl CudaRealAnalyticScalar for f64 {}

impl RuntimeRealAnalyticScalar for f32 {
    unsafe fn pointwise_unary_raw(
        runtime: &CudaRuntime,
        op: RealUnaryOp,
        alpha: Self,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_unary_real_f32_raw(
                op,
                alpha,
                src,
                dims,
                src_strides,
                src_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }
}

impl RuntimeRealAnalyticScalar for f64 {
    unsafe fn pointwise_unary_raw(
        runtime: &CudaRuntime,
        op: RealUnaryOp,
        alpha: Self,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_unary_real_f64_raw(
                op,
                alpha,
                src,
                dims,
                src_strides,
                src_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }
}

fn validate_pointwise_shapes(shapes: &[&[usize]], arity: usize, op_name: &str) -> Result<()> {
    validate_shape_count(shapes, arity + 1, op_name)?;
    let output_shape = shapes[arity];
    for (idx, shape) in shapes[..arity].iter().enumerate() {
        validate_shape_eq(shape, output_shape, &format!("{op_name} input {idx}"))?;
    }
    Ok(())
}

fn supports_analytic_unary(op: AnalyticUnaryOp) -> bool {
    matches!(op, AnalyticUnaryOp::Log)
}

fn to_runtime_unary(op: AnalyticUnaryOp) -> Result<RealUnaryOp> {
    match op {
        AnalyticUnaryOp::Log => Ok(RealUnaryOp::Log),
        _ => Err(Error::InvalidArgument(format!(
            "analytic unary operation {op:?} is not implemented on CudaBackend"
        ))),
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

fn tensor_device_ptr<T: Scalar>(tensor: &Tensor<T>, label: &str) -> Result<*const T> {
    tensor
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn tensor_device_mut_ptr<T: Scalar>(tensor: &mut Tensor<T>, label: &str) -> Result<*mut T> {
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut T)
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn resolved_input<S>(ctx: &mut CudaContext, input: &Tensor<S>) -> Tensor<S>
where
    S: Scalar + Conjugate + 'static,
{
    if input.is_conjugated() {
        CudaBackend::resolve_conj(ctx, input)
    } else {
        input.clone()
    }
}

fn plan_real_analytic<S>(
    desc: &AnalyticPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaAnalyticPlan<S>>
where
    S: CudaRealAnalyticScalar,
{
    match desc {
        AnalyticPrimsDescriptor::PointwiseUnary { op } => {
            validate_pointwise_shapes(shapes, 1, "CudaAnalyticPointwiseUnary")?;
            if !supports_analytic_unary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "analytic unary operation {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<S>()
                )));
            }
            Ok(CudaAnalyticPlan {
                op: *op,
                _marker: PhantomData,
            })
        }
        _ => Err(Error::InvalidArgument(format!(
            "analytic descriptor {desc:?} is not implemented on CudaBackend in phase 1"
        ))),
    }
}

fn execute_real_analytic<S>(
    ctx: &mut CudaContext,
    plan: &CudaAnalyticPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: CudaRealAnalyticScalar,
{
    if output.is_conjugated() {
        return Err(Error::InvalidArgument(
            "CUDA analytic family does not support conjugated outputs".into(),
        ));
    }

    validate_execute_inputs(inputs, 1, "CudaAnalyticPointwiseUnary")?;
    let input = resolved_input(ctx, inputs[0]);
    ensure_cuda_tensor(&input, ctx.device_id(), "CUDA analytic input")?;
    ensure_cuda_tensor(output, ctx.device_id(), "CUDA analytic output")?;

    let runtime = Arc::clone(ctx.shared_runtime());
    let runtime_op = to_runtime_unary(plan.op)?;
    let output_dims = output.dims().to_vec();
    let output_strides = output.strides().to_vec();
    let output_offset = output.offset();
    let output_ptr = tensor_device_mut_ptr(output, "CUDA analytic output")?;

    unsafe {
        S::pointwise_unary_raw(
            runtime.as_ref(),
            runtime_op,
            alpha,
            tensor_device_ptr(&input, "CUDA analytic input")?,
            &output_dims,
            input.strides(),
            input.offset(),
            beta,
            output_ptr,
            &output_strides,
            output_offset,
        )
    }
}

fn unsupported_complex_analytic<S>(desc: &AnalyticPrimsDescriptor) -> Result<CudaAnalyticPlan<S>>
where
    S: Scalar,
{
    Err(Error::InvalidArgument(format!(
        "CUDA analytic family is not implemented for {}: {desc:?}",
        std::any::type_name::<S>()
    )))
}

macro_rules! impl_cuda_analytic_prims_real {
    ($scalar:ty) => {
        impl TensorAnalyticPrims<Standard<$scalar>> for CudaBackend {
            type Plan = CudaAnalyticPlan<$scalar>;
            type Context = CudaContext;

            fn plan(
                _ctx: &mut Self::Context,
                desc: &AnalyticPrimsDescriptor,
                shapes: &[&[usize]],
            ) -> Result<Self::Plan> {
                plan_real_analytic::<$scalar>(desc, shapes)
            }

            fn execute(
                ctx: &mut Self::Context,
                plan: &Self::Plan,
                alpha: $scalar,
                inputs: &[&Tensor<$scalar>],
                beta: $scalar,
                output: &mut Tensor<$scalar>,
            ) -> Result<()> {
                execute_real_analytic(ctx, plan, alpha, inputs, beta, output)
            }

            fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool {
                matches!(
                    desc,
                    AnalyticPrimsDescriptor::PointwiseUnary {
                        op: AnalyticUnaryOp::Log
                    }
                )
            }
        }
    };
}

macro_rules! impl_cuda_analytic_prims_unsupported {
    ($scalar:ty) => {
        impl TensorAnalyticPrims<Standard<$scalar>> for CudaBackend {
            type Plan = CudaAnalyticPlan<$scalar>;
            type Context = CudaContext;

            fn plan(
                _ctx: &mut Self::Context,
                desc: &AnalyticPrimsDescriptor,
                _shapes: &[&[usize]],
            ) -> Result<Self::Plan> {
                unsupported_complex_analytic::<$scalar>(desc)
            }

            fn execute(
                _ctx: &mut Self::Context,
                _plan: &Self::Plan,
                _alpha: $scalar,
                _inputs: &[&Tensor<$scalar>],
                _beta: $scalar,
                _output: &mut Tensor<$scalar>,
            ) -> Result<()> {
                Err(Error::InvalidArgument(format!(
                    "CUDA analytic family is not implemented for {}",
                    std::any::type_name::<$scalar>()
                )))
            }

            fn has_analytic_support(_desc: AnalyticPrimsDescriptor) -> bool {
                false
            }
        }
    };
}

impl_cuda_analytic_prims_real!(f32);
impl_cuda_analytic_prims_real!(f64);
impl_cuda_analytic_prims_unsupported!(Complex32);
impl_cuda_analytic_prims_unsupported!(Complex64);
