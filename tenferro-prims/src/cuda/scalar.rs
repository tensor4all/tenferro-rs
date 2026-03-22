use std::marker::PhantomData;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{
    cuda::runtime::{CudaRuntime, RealBinaryOp, RealReductionOp, RealTernaryOp, RealUnaryOp},
    Error, LogicalMemorySpace, Result,
};
use tenferro_tensor::Tensor;

use crate::{
    validate_execute_inputs, validate_rank, validate_shape_count, validate_shape_eq, CudaBackend,
    ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp, ScalarTernaryOp, ScalarUnaryOp,
    TensorScalarPrims,
};

use super::CudaContext;

/// CUDA execution plan for the scalar protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CudaScalarPlan;
/// let _ = std::mem::size_of::<CudaScalarPlan<f64>>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaScalarPlan<T: Scalar> {
    kind: CudaScalarPlanKind,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CudaScalarPlanKind {
    PointwiseUnary {
        op: ScalarUnaryOp,
    },
    PointwiseBinary {
        op: ScalarBinaryOp,
    },
    PointwiseTernary {
        op: ScalarTernaryOp,
    },
    Reduction {
        kept_axes: Vec<usize>,
        reduced_axes: Vec<usize>,
        op: ScalarReductionOp,
    },
}

trait RuntimeRealScalarOps: Scalar + 'static {
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

    unsafe fn pointwise_binary_raw(
        runtime: &CudaRuntime,
        op: RealBinaryOp,
        alpha: Self,
        lhs: *const Self,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const Self,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>;

    unsafe fn pointwise_ternary_raw(
        runtime: &CudaRuntime,
        op: RealTernaryOp,
        alpha: Self,
        cond: *const Self,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const Self,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const Self,
        false_strides: &[isize],
        false_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>;

    unsafe fn reduce_raw(
        runtime: &CudaRuntime,
        op: RealReductionOp,
        alpha: Self,
        input: *const Self,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: Self,
        output: *mut Self,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()>;
}

trait CudaRealScalar: Scalar + Conjugate + RuntimeRealScalarOps + 'static {}

impl CudaRealScalar for f32 {}
impl CudaRealScalar for f64 {}

impl RuntimeRealScalarOps for f32 {
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

    unsafe fn pointwise_binary_raw(
        runtime: &CudaRuntime,
        op: RealBinaryOp,
        alpha: Self,
        lhs: *const Self,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const Self,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_binary_real_f32_raw(
                op,
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

    unsafe fn pointwise_ternary_raw(
        runtime: &CudaRuntime,
        op: RealTernaryOp,
        alpha: Self,
        cond: *const Self,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const Self,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const Self,
        false_strides: &[isize],
        false_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_ternary_real_f32_raw(
                op,
                alpha,
                cond,
                dims,
                cond_strides,
                cond_offset,
                on_true,
                true_strides,
                true_offset,
                on_false,
                false_strides,
                false_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }

    unsafe fn reduce_raw(
        runtime: &CudaRuntime,
        op: RealReductionOp,
        alpha: Self,
        input: *const Self,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: Self,
        output: *mut Self,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        unsafe {
            runtime.reduce_real_f32_raw(
                op,
                alpha,
                input,
                input_dims,
                input_strides,
                input_offset,
                beta,
                output,
                output_dims,
                output_strides,
                output_offset,
                kept_axes,
                reduced_axes,
            )
        }
    }
}
impl RuntimeRealScalarOps for f64 {
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

    unsafe fn pointwise_binary_raw(
        runtime: &CudaRuntime,
        op: RealBinaryOp,
        alpha: Self,
        lhs: *const Self,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const Self,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_binary_real_f64_raw(
                op,
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

    unsafe fn pointwise_ternary_raw(
        runtime: &CudaRuntime,
        op: RealTernaryOp,
        alpha: Self,
        cond: *const Self,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const Self,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const Self,
        false_strides: &[isize],
        false_offset: isize,
        beta: Self,
        dst: *mut Self,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        unsafe {
            runtime.pointwise_ternary_real_f64_raw(
                op,
                alpha,
                cond,
                dims,
                cond_strides,
                cond_offset,
                on_true,
                true_strides,
                true_offset,
                on_false,
                false_strides,
                false_offset,
                beta,
                dst,
                dst_strides,
                dst_offset,
            )
        }
    }

    unsafe fn reduce_raw(
        runtime: &CudaRuntime,
        op: RealReductionOp,
        alpha: Self,
        input: *const Self,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: Self,
        output: *mut Self,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        unsafe {
            runtime.reduce_real_f64_raw(
                op,
                alpha,
                input,
                input_dims,
                input_strides,
                input_offset,
                beta,
                output,
                output_dims,
                output_strides,
                output_offset,
                kept_axes,
                reduced_axes,
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

fn plan_reduction_axes(
    modes_a: &[u32],
    modes_c: &[u32],
    shapes: &[&[usize]],
    op_name: &str,
) -> Result<Vec<usize>> {
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

    Ok(modes_a
        .iter()
        .enumerate()
        .filter(|(_, mode)| !modes_c.contains(mode))
        .map(|(idx, _)| idx)
        .collect())
}

fn supports_scalar_unary(op: ScalarUnaryOp) -> bool {
    matches!(
        op,
        ScalarUnaryOp::Conj | ScalarUnaryOp::Abs | ScalarUnaryOp::Reciprocal
    )
}

fn supports_scalar_binary(op: ScalarBinaryOp) -> bool {
    matches!(
        op,
        ScalarBinaryOp::Add
            | ScalarBinaryOp::Sub
            | ScalarBinaryOp::Mul
            | ScalarBinaryOp::Div
            | ScalarBinaryOp::Maximum
            | ScalarBinaryOp::Greater
            | ScalarBinaryOp::GreaterEqual
            | ScalarBinaryOp::Minimum
    )
}

fn supports_scalar_ternary(op: ScalarTernaryOp) -> bool {
    matches!(op, ScalarTernaryOp::Where)
}

fn supports_scalar_reduction(op: ScalarReductionOp) -> bool {
    matches!(
        op,
        ScalarReductionOp::Sum
            | ScalarReductionOp::Prod
            | ScalarReductionOp::Max
            | ScalarReductionOp::Min
    )
}

fn to_runtime_unary(op: ScalarUnaryOp) -> Result<RealUnaryOp> {
    match op {
        ScalarUnaryOp::Conj => Ok(RealUnaryOp::Conj),
        ScalarUnaryOp::Abs => Ok(RealUnaryOp::Abs),
        ScalarUnaryOp::Reciprocal => Ok(RealUnaryOp::Reciprocal),
        _ => Err(Error::InvalidArgument(format!(
            "scalar unary operation {op:?} is not implemented on CudaBackend"
        ))),
    }
}

fn to_runtime_binary(op: ScalarBinaryOp) -> Result<RealBinaryOp> {
    match op {
        ScalarBinaryOp::Add => Ok(RealBinaryOp::Add),
        ScalarBinaryOp::Sub => Ok(RealBinaryOp::Sub),
        ScalarBinaryOp::Mul => Ok(RealBinaryOp::Mul),
        ScalarBinaryOp::Div => Ok(RealBinaryOp::Div),
        ScalarBinaryOp::Maximum => Ok(RealBinaryOp::Maximum),
        ScalarBinaryOp::Minimum => Ok(RealBinaryOp::Minimum),
        ScalarBinaryOp::Greater => Ok(RealBinaryOp::Greater),
        ScalarBinaryOp::GreaterEqual => Ok(RealBinaryOp::GreaterEqual),
        _ => Err(Error::InvalidArgument(format!(
            "scalar binary operation {op:?} is not implemented on CudaBackend"
        ))),
    }
}

fn to_runtime_ternary(op: ScalarTernaryOp) -> Result<RealTernaryOp> {
    match op {
        ScalarTernaryOp::Where => Ok(RealTernaryOp::Where),
    }
}

fn to_runtime_reduction(op: ScalarReductionOp) -> Result<RealReductionOp> {
    match op {
        ScalarReductionOp::Sum => Ok(RealReductionOp::Sum),
        ScalarReductionOp::Max => Ok(RealReductionOp::Max),
        ScalarReductionOp::Min => Ok(RealReductionOp::Min),
        ScalarReductionOp::Prod => Ok(RealReductionOp::Prod),
        _ => Err(Error::InvalidArgument(format!(
            "scalar reduction {op:?} is not implemented on CudaBackend"
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

fn plan_real_scalar<S>(
    desc: &ScalarPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaScalarPlan<S>>
where
    S: CudaRealScalar,
{
    let kind = match desc {
        ScalarPrimsDescriptor::PointwiseUnary { op } => {
            validate_pointwise_shapes(shapes, 1, "CudaScalarPointwiseUnary")?;
            if !supports_scalar_unary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "scalar unary operation {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<S>()
                )));
            }
            CudaScalarPlanKind::PointwiseUnary { op: *op }
        }
        ScalarPrimsDescriptor::PointwiseBinary { op } => {
            validate_pointwise_shapes(shapes, 2, "CudaScalarPointwiseBinary")?;
            if !supports_scalar_binary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "scalar binary operation {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<S>()
                )));
            }
            CudaScalarPlanKind::PointwiseBinary { op: *op }
        }
        ScalarPrimsDescriptor::PointwiseTernary { op } => {
            validate_pointwise_shapes(shapes, 3, "CudaScalarPointwiseTernary")?;
            if !supports_scalar_ternary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "scalar ternary operation {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<S>()
                )));
            }
            CudaScalarPlanKind::PointwiseTernary { op: *op }
        }
        ScalarPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op,
        } => {
            if !supports_scalar_reduction(*op) {
                return Err(Error::InvalidArgument(format!(
                    "scalar reduction {op:?} is not supported on CudaBackend for {}",
                    std::any::type_name::<S>()
                )));
            }
            let reduced_axes =
                plan_reduction_axes(modes_a, modes_c, shapes, "CudaScalarReduction")?;
            let kept_axes: Vec<usize> = modes_c
                .iter()
                .map(|mode| {
                    modes_a.iter().position(|candidate| candidate == mode).ok_or_else(|| {
                        Error::InvalidArgument(format!(
                            "CudaScalarReduction output mode {mode} not found in input modes {modes_a:?}"
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            CudaScalarPlanKind::Reduction {
                kept_axes,
                reduced_axes,
                op: *op,
            }
        }
    };

    Ok(CudaScalarPlan {
        kind,
        _marker: PhantomData,
    })
}

fn execute_real_scalar<S>(
    ctx: &mut CudaContext,
    plan: &CudaScalarPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: CudaRealScalar,
{
    if output.is_conjugated() {
        return Err(Error::InvalidArgument(
            "CUDA scalar family does not support conjugated outputs".into(),
        ));
    }

    ensure_cuda_tensor(output, ctx.device_id(), "CUDA scalar output")?;
    match &plan.kind {
        CudaScalarPlanKind::PointwiseUnary { op } => {
            validate_execute_inputs(inputs, 1, "CudaScalarPointwiseUnary")?;
            let input = resolved_input(ctx, inputs[0]);
            ensure_cuda_tensor(&input, ctx.device_id(), "CUDA scalar unary input")?;
            let runtime = Arc::clone(ctx.shared_runtime());
            let runtime_op = to_runtime_unary(*op)?;
            let output_dims = output.dims().to_vec();
            let output_strides = output.strides().to_vec();
            let output_offset = output.offset();
            let output_ptr = tensor_device_mut_ptr(output, "CUDA scalar unary output")?;
            unsafe {
                S::pointwise_unary_raw(
                    runtime.as_ref(),
                    runtime_op,
                    alpha,
                    tensor_device_ptr(&input, "CUDA scalar unary input")?,
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
        CudaScalarPlanKind::PointwiseBinary { op } => {
            validate_execute_inputs(inputs, 2, "CudaScalarPointwiseBinary")?;
            let lhs = resolved_input(ctx, inputs[0]);
            let rhs = resolved_input(ctx, inputs[1]);
            ensure_cuda_tensor(&lhs, ctx.device_id(), "CUDA scalar binary lhs")?;
            ensure_cuda_tensor(&rhs, ctx.device_id(), "CUDA scalar binary rhs")?;
            let runtime = Arc::clone(ctx.shared_runtime());
            let runtime_op = to_runtime_binary(*op)?;
            let output_dims = output.dims().to_vec();
            let output_strides = output.strides().to_vec();
            let output_offset = output.offset();
            let output_ptr = tensor_device_mut_ptr(output, "CUDA scalar binary output")?;
            unsafe {
                S::pointwise_binary_raw(
                    runtime.as_ref(),
                    runtime_op,
                    alpha,
                    tensor_device_ptr(&lhs, "CUDA scalar binary lhs")?,
                    &output_dims,
                    lhs.strides(),
                    lhs.offset(),
                    tensor_device_ptr(&rhs, "CUDA scalar binary rhs")?,
                    rhs.strides(),
                    rhs.offset(),
                    beta,
                    output_ptr,
                    &output_strides,
                    output_offset,
                )
            }
        }
        CudaScalarPlanKind::PointwiseTernary { op } => {
            validate_execute_inputs(inputs, 3, "CudaScalarPointwiseTernary")?;
            let cond = resolved_input(ctx, inputs[0]);
            let on_true = resolved_input(ctx, inputs[1]);
            let on_false = resolved_input(ctx, inputs[2]);
            ensure_cuda_tensor(&cond, ctx.device_id(), "CUDA scalar ternary condition")?;
            ensure_cuda_tensor(&on_true, ctx.device_id(), "CUDA scalar ternary true branch")?;
            ensure_cuda_tensor(
                &on_false,
                ctx.device_id(),
                "CUDA scalar ternary false branch",
            )?;
            let runtime = Arc::clone(ctx.shared_runtime());
            let runtime_op = to_runtime_ternary(*op)?;
            let output_dims = output.dims().to_vec();
            let output_strides = output.strides().to_vec();
            let output_offset = output.offset();
            let output_ptr = tensor_device_mut_ptr(output, "CUDA scalar ternary output")?;
            unsafe {
                S::pointwise_ternary_raw(
                    runtime.as_ref(),
                    runtime_op,
                    alpha,
                    tensor_device_ptr(&cond, "CUDA scalar ternary condition")?,
                    &output_dims,
                    cond.strides(),
                    cond.offset(),
                    tensor_device_ptr(&on_true, "CUDA scalar ternary true branch")?,
                    on_true.strides(),
                    on_true.offset(),
                    tensor_device_ptr(&on_false, "CUDA scalar ternary false branch")?,
                    on_false.strides(),
                    on_false.offset(),
                    beta,
                    output_ptr,
                    &output_strides,
                    output_offset,
                )
            }
        }
        CudaScalarPlanKind::Reduction {
            kept_axes,
            reduced_axes,
            op,
        } => {
            validate_execute_inputs(inputs, 1, "CudaScalarReduction")?;
            let input = resolved_input(ctx, inputs[0]);
            ensure_cuda_tensor(&input, ctx.device_id(), "CUDA scalar reduction input")?;
            let runtime = Arc::clone(ctx.shared_runtime());
            let runtime_op = to_runtime_reduction(*op)?;
            let output_dims = output.dims().to_vec();
            let output_strides = output.strides().to_vec();
            let output_offset = output.offset();
            let output_ptr = tensor_device_mut_ptr(output, "CUDA scalar reduction output")?;
            unsafe {
                S::reduce_raw(
                    runtime.as_ref(),
                    runtime_op,
                    alpha,
                    tensor_device_ptr(&input, "CUDA scalar reduction input")?,
                    input.dims(),
                    input.strides(),
                    input.offset(),
                    beta,
                    output_ptr,
                    &output_dims,
                    &output_strides,
                    output_offset,
                    kept_axes,
                    reduced_axes,
                )
            }
        }
    }
}

fn unsupported_complex_scalar<S>(desc: &ScalarPrimsDescriptor) -> Result<CudaScalarPlan<S>>
where
    S: Scalar,
{
    Err(Error::InvalidArgument(format!(
        "CUDA scalar family is not implemented for {}: {desc:?}",
        std::any::type_name::<S>()
    )))
}

macro_rules! impl_cuda_scalar_prims_real {
    ($scalar:ty) => {
        impl TensorScalarPrims<Standard<$scalar>> for CudaBackend {
            type Plan = CudaScalarPlan<$scalar>;
            type Context = CudaContext;

            fn plan(
                _ctx: &mut Self::Context,
                desc: &ScalarPrimsDescriptor,
                shapes: &[&[usize]],
            ) -> Result<Self::Plan> {
                plan_real_scalar::<$scalar>(desc, shapes)
            }

            fn execute(
                ctx: &mut Self::Context,
                plan: &Self::Plan,
                alpha: $scalar,
                inputs: &[&Tensor<$scalar>],
                beta: $scalar,
                output: &mut Tensor<$scalar>,
            ) -> Result<()> {
                execute_real_scalar(ctx, plan, alpha, inputs, beta, output)
            }

            fn has_scalar_support(desc: ScalarPrimsDescriptor) -> bool {
                match desc {
                    ScalarPrimsDescriptor::PointwiseUnary { op } => supports_scalar_unary(op),
                    ScalarPrimsDescriptor::PointwiseBinary { op } => supports_scalar_binary(op),
                    ScalarPrimsDescriptor::PointwiseTernary { op } => supports_scalar_ternary(op),
                    ScalarPrimsDescriptor::Reduction { op, .. } => supports_scalar_reduction(op),
                }
            }
        }
    };
}

macro_rules! impl_cuda_scalar_prims_unsupported {
    ($scalar:ty) => {
        impl TensorScalarPrims<Standard<$scalar>> for CudaBackend {
            type Plan = CudaScalarPlan<$scalar>;
            type Context = CudaContext;

            fn plan(
                _ctx: &mut Self::Context,
                desc: &ScalarPrimsDescriptor,
                _shapes: &[&[usize]],
            ) -> Result<Self::Plan> {
                unsupported_complex_scalar::<$scalar>(desc)
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
                    "CUDA scalar family is not implemented for {}",
                    std::any::type_name::<$scalar>()
                )))
            }

            fn has_scalar_support(_desc: ScalarPrimsDescriptor) -> bool {
                false
            }
        }
    };
}

impl_cuda_scalar_prims_real!(f32);
impl_cuda_scalar_prims_real!(f64);
impl_cuda_scalar_prims_unsupported!(Complex32);
impl_cuda_scalar_prims_unsupported!(Complex64);
