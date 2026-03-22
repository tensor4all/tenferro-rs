use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::{Complex32, Complex64, ComplexFloat};
use tenferro_algebra::Scalar;
use tenferro_device::{
    cuda::runtime::{ComplexRealUnaryOp as RuntimeComplexRealUnaryOp, CudaRuntime},
    Error, LogicalMemorySpace, Result,
};
use tenferro_tensor::Tensor;

use crate::cuda::CudaContext;
use crate::{
    validate_execute_inputs, validate_shape_count, validate_shape_eq, ComplexRealPrimsDescriptor,
    ComplexRealUnaryOp, CudaBackend, TensorComplexRealPrims,
};

/// CUDA execution plan for the complex-to-real unary protocol family.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::CudaComplexRealPlan;
/// let _ = std::mem::size_of::<CudaComplexRealPlan<Complex64>>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaComplexRealPlan<T: Scalar> {
    kind: ComplexRealUnaryOp,
    _marker: PhantomData<T>,
}

trait RuntimeComplexRealScalar: Scalar + ComplexFloat + 'static {
    unsafe fn pointwise_unary_complex_real_raw(
        runtime: &CudaRuntime,
        op: RuntimeComplexRealUnaryOp,
        alpha: Self::Real,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self::Real,
        dst: *mut Self::Real,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>;
}

impl RuntimeComplexRealScalar for Complex32 {
    unsafe fn pointwise_unary_complex_real_raw(
        runtime: &CudaRuntime,
        op: RuntimeComplexRealUnaryOp,
        alpha: Self::Real,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self::Real,
        dst: *mut Self::Real,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_unary_complex32_to_real_f32_raw(
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

impl RuntimeComplexRealScalar for Complex64 {
    unsafe fn pointwise_unary_complex_real_raw(
        runtime: &CudaRuntime,
        op: RuntimeComplexRealUnaryOp,
        alpha: Self::Real,
        src: *const Self,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Self::Real,
        dst: *mut Self::Real,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_unary_complex64_to_real_f64_raw(
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

fn tensor_device_mut_ptr<T: Scalar>(tensor: &Tensor<T>, label: &str) -> Result<*mut T> {
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut T)
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn supports_complex_real_unary(op: ComplexRealUnaryOp) -> bool {
    matches!(
        op,
        ComplexRealUnaryOp::Abs | ComplexRealUnaryOp::Real | ComplexRealUnaryOp::Imag
    )
}

fn plan_complex_real_unary<T>(
    desc: &ComplexRealPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaComplexRealPlan<T>>
where
    T: RuntimeComplexRealScalar,
    T::Real: Scalar + Send + Sync,
{
    validate_shape_count(shapes, 2, "CudaComplexRealPointwiseUnary")?;
    validate_shape_eq(shapes[0], shapes[1], "CudaComplexRealPointwiseUnary")?;
    match desc {
        ComplexRealPrimsDescriptor::PointwiseUnary { op } => {
            if !supports_complex_real_unary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "complex-real unary operation {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<T>()
                )));
            }
            Ok(CudaComplexRealPlan {
                kind: *op,
                _marker: PhantomData,
            })
        }
    }
}

fn execute_complex_real_unary<T>(
    ctx: &mut CudaContext,
    plan: &CudaComplexRealPlan<T>,
    alpha: T::Real,
    inputs: &[&Tensor<T>],
    beta: T::Real,
    output: &mut Tensor<T::Real>,
) -> Result<()>
where
    T: RuntimeComplexRealScalar,
    T::Real: Scalar + Send + Sync,
{
    validate_execute_inputs(inputs, 1, "CudaComplexRealPointwiseUnary")?;
    ensure_cuda_tensor(inputs[0], ctx.device_id(), "CUDA complex-real input")?;
    ensure_cuda_tensor(output, ctx.device_id(), "CUDA complex-real output")?;

    let runtime = Arc::clone(ctx.shared_runtime());
    let input = inputs[0];
    let output_dims = output.dims().to_vec();
    let output_strides = output.strides().to_vec();
    let input_ptr = tensor_device_ptr(input, "CUDA complex-real input")?;
    let output_ptr = tensor_device_mut_ptr(output, "CUDA complex-real output")?;
    let runtime_op = match plan.kind {
        ComplexRealUnaryOp::Abs => RuntimeComplexRealUnaryOp::Abs,
        ComplexRealUnaryOp::Real => RuntimeComplexRealUnaryOp::Real,
        ComplexRealUnaryOp::Imag => RuntimeComplexRealUnaryOp::Imag,
    };

    unsafe {
        T::pointwise_unary_complex_real_raw(
            runtime.as_ref(),
            runtime_op,
            alpha,
            input_ptr,
            &output_dims,
            input.strides(),
            input.offset(),
            beta,
            output_ptr,
            &output_strides,
            output.offset(),
        )
    }
}

macro_rules! impl_cuda_complex_real_prims {
    ($scalar:ty) => {
        impl TensorComplexRealPrims<$scalar> for CudaBackend {
            type Real = <$scalar as ComplexFloat>::Real;
            type Plan = CudaComplexRealPlan<$scalar>;
            type Context = CudaContext;

            fn plan(
                _ctx: &mut Self::Context,
                desc: &ComplexRealPrimsDescriptor,
                shapes: &[&[usize]],
            ) -> Result<Self::Plan> {
                plan_complex_real_unary::<$scalar>(desc, shapes)
            }

            fn execute(
                ctx: &mut Self::Context,
                plan: &Self::Plan,
                alpha: Self::Real,
                inputs: &[&Tensor<$scalar>],
                beta: Self::Real,
                output: &mut Tensor<Self::Real>,
            ) -> Result<()> {
                execute_complex_real_unary::<$scalar>(ctx, plan, alpha, inputs, beta, output)
            }

            fn has_complex_real_support(desc: ComplexRealPrimsDescriptor) -> bool {
                matches!(
                    desc,
                    ComplexRealPrimsDescriptor::PointwiseUnary {
                        op: ComplexRealUnaryOp::Abs
                            | ComplexRealUnaryOp::Real
                            | ComplexRealUnaryOp::Imag
                    }
                )
            }
        }
    };
}

impl_cuda_complex_real_prims!(Complex32);
impl_cuda_complex_real_prims!(Complex64);
