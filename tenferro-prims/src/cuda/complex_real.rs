use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{NumCast, One, Zero};
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::cuda::runtime::{
    ComplexRealUnaryOp as RuntimeComplexRealUnaryOp, CudaRuntime,
};
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::cuda::CudaContext;
use crate::{
    validate_execute_inputs, validate_rank, validate_shape_count, validate_shape_eq,
    ComplexRealPrimsDescriptor, ComplexRealUnaryOp, CudaBackend, ScalarPrimsDescriptor,
    ScalarReductionOp, TensorComplexRealPrims, TensorScalarPrims,
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
    kind: CudaComplexRealPlanKind,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CudaComplexRealPlanKind {
    PointwiseUnary {
        op: ComplexRealUnaryOp,
    },
    Reduction {
        unary_op: ComplexRealUnaryOp,
        reduction_op: ScalarReductionOp,
        reduced_axes: Vec<usize>,
        reduced_total: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReductionPlanSpec {
    reduced_axes: Vec<usize>,
    reduced_total: usize,
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
                kind: CudaComplexRealPlanKind::PointwiseUnary { op: *op },
                _marker: PhantomData,
            })
        }
        _ => Err(Error::InvalidArgument(
            "expected complex-real unary descriptor".into(),
        )),
    }
}

fn plan_complex_real_reduction<T>(
    desc: &ComplexRealPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaComplexRealPlan<T>>
where
    T: RuntimeComplexRealScalar,
    T::Real: Scalar + Send + Sync,
{
    match desc {
        ComplexRealPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            unary_op,
            reduction_op,
        } => {
            if !supports_complex_real_unary(*unary_op) {
                return Err(Error::InvalidArgument(format!(
                    "complex-real unary operation {unary_op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<T>()
                )));
            }
            let reduction =
                plan_reduction_axes(modes_a, modes_c, shapes, "CudaComplexRealReduction")?;
            Ok(CudaComplexRealPlan {
                kind: CudaComplexRealPlanKind::Reduction {
                    unary_op: *unary_op,
                    reduction_op: *reduction_op,
                    reduced_axes: reduction.reduced_axes,
                    reduced_total: reduction.reduced_total,
                },
                _marker: PhantomData,
            })
        }
        _ => Err(Error::InvalidArgument(
            "expected complex-real reduction descriptor".into(),
        )),
    }
}

fn plan_reduction_axes(
    modes_a: &[u32],
    modes_c: &[u32],
    shapes: &[&[usize]],
    op_name: &str,
) -> Result<ReductionPlanSpec> {
    validate_shape_count(shapes, 2, op_name)?;
    validate_rank(shapes[0], modes_a.len(), &format!("{op_name} input"))?;
    validate_rank(shapes[1], modes_c.len(), &format!("{op_name} output"))?;

    let mut expected_output = Vec::with_capacity(modes_c.len());
    for &mode in modes_c {
        let Some(axis) = modes_a.iter().position(|&candidate| candidate == mode) else {
            return Err(Error::InvalidArgument(format!(
                "{op_name}: output mode {mode} not found in input modes {modes_a:?}"
            )));
        };
        expected_output.push(shapes[0][axis]);
    }
    validate_shape_eq(shapes[1], &expected_output, &format!("{op_name} output"))?;

    let reduced_axes: Vec<usize> = modes_a
        .iter()
        .enumerate()
        .filter(|(_, mode)| !modes_c.contains(mode))
        .map(|(axis, _)| axis)
        .collect();
    let reduced_total = reduced_axes
        .iter()
        .map(|&axis| shapes[0][axis])
        .product::<usize>();

    Ok(ReductionPlanSpec {
        reduced_axes,
        reduced_total,
    })
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
    T::Real: Scalar + Send + Sync + NumCast,
    CudaBackend: TensorScalarPrims<Standard<T::Real>, Context = CudaContext>,
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
    match &plan.kind {
        CudaComplexRealPlanKind::PointwiseUnary { op } => {
            let runtime_op = match op {
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
        CudaComplexRealPlanKind::Reduction {
            unary_op,
            reduction_op,
            reduced_axes,
            reduced_total,
        } => {
            let temp = Tensor::<T::Real>::zeros(
                input.dims(),
                output.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let temp_strides = temp.strides().to_vec();
            let temp_ptr = tensor_device_mut_ptr(&temp, "CUDA complex-real temporary")?;
            let runtime_op = match unary_op {
                ComplexRealUnaryOp::Abs => RuntimeComplexRealUnaryOp::Abs,
                ComplexRealUnaryOp::Real => RuntimeComplexRealUnaryOp::Real,
                ComplexRealUnaryOp::Imag => RuntimeComplexRealUnaryOp::Imag,
            };
            unsafe {
                T::pointwise_unary_complex_real_raw(
                    runtime.as_ref(),
                    runtime_op,
                    T::Real::one(),
                    input_ptr,
                    input.dims(),
                    input.strides(),
                    input.offset(),
                    T::Real::zero(),
                    temp_ptr,
                    &temp_strides,
                    temp.offset(),
                )?;
            }

            let modes_a: Vec<u32> = (0..temp.ndim())
                .map(|axis| {
                    u32::try_from(axis).map_err(|_| {
                        Error::InvalidArgument(format!("axis {axis} exceeds u32 range"))
                    })
                })
                .collect::<Result<_>>()?;
            let modes_c: Vec<u32> = modes_a
                .iter()
                .copied()
                .enumerate()
                .filter(|(axis, _)| !reduced_axes.contains(axis))
                .map(|(_, mode)| mode)
                .collect();
            let desc = ScalarPrimsDescriptor::Reduction {
                modes_a,
                modes_c,
                op: if matches!(reduction_op, ScalarReductionOp::Mean) {
                    ScalarReductionOp::Sum
                } else {
                    *reduction_op
                },
            };
            let reduction_alpha = if matches!(reduction_op, ScalarReductionOp::Mean) {
                let Some(scale) = <T::Real as NumCast>::from(*reduced_total) else {
                    return Err(Error::InvalidArgument(format!(
                        "cannot represent CUDA mean reduction size {} in {}",
                        reduced_total,
                        std::any::type_name::<T::Real>()
                    )));
                };
                alpha / scale
            } else {
                alpha
            };
            let plan = <CudaBackend as TensorScalarPrims<Standard<T::Real>>>::plan(
                ctx,
                &desc,
                &[temp.dims(), output.dims()],
            )?;
            <CudaBackend as TensorScalarPrims<Standard<T::Real>>>::execute(
                ctx,
                &plan,
                reduction_alpha,
                &[&temp],
                beta,
                output,
            )?;
            Ok(())
        }
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
                match desc {
                    ComplexRealPrimsDescriptor::PointwiseUnary { .. } => {
                        plan_complex_real_unary::<$scalar>(desc, shapes)
                    }
                    ComplexRealPrimsDescriptor::Reduction { .. } => {
                        plan_complex_real_reduction::<$scalar>(desc, shapes)
                    }
                }
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
                    } | ComplexRealPrimsDescriptor::Reduction {
                        unary_op: ComplexRealUnaryOp::Abs
                            | ComplexRealUnaryOp::Real
                            | ComplexRealUnaryOp::Imag,
                        reduction_op: ScalarReductionOp::Sum
                            | ScalarReductionOp::Prod
                            | ScalarReductionOp::Mean
                            | ScalarReductionOp::Max
                            | ScalarReductionOp::Min,
                        ..
                    }
                )
            }
        }
    };
}

impl_cuda_complex_real_prims!(Complex32);
impl_cuda_complex_real_prims!(Complex64);
