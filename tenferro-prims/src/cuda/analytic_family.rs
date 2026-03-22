use std::ffi::c_void;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::{
    CutensorOperator, CUTENSOR_OP_ACOS, CUTENSOR_OP_ACOSH, CUTENSOR_OP_ADD, CUTENSOR_OP_ASIN,
    CUTENSOR_OP_ASINH, CUTENSOR_OP_ATAN, CUTENSOR_OP_ATANH, CUTENSOR_OP_COS, CUTENSOR_OP_COSH,
    CUTENSOR_OP_EXP, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_LOG, CUTENSOR_OP_MUL, CUTENSOR_OP_NEG,
    CUTENSOR_OP_SIN, CUTENSOR_OP_SINH, CUTENSOR_OP_SQRT, CUTENSOR_OP_TAN, CUTENSOR_OP_TANH,
};
use crate::{
    validate_execute_inputs, AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp,
    AnalyticUnaryOp,
};

use super::custom::{
    ComplexBinaryKernelOp, ComplexUnaryKernelOp, RealBinaryKernelOp, RealUnaryKernelOp,
};
use super::family_common::{
    plan_reduction_shapes, scale_real_alpha, supports_complex_scalar_type,
    supports_real_scalar_type, validate_pointwise_shapes,
};
use super::planning::plan_reduction;
use super::pointwise_ops::{
    execute_binary_trinary, execute_copy_with_accum, execute_custom_complex_binary,
    execute_custom_complex_unary, execute_custom_real_binary, execute_custom_real_unary,
    execute_direct_unary, execute_square,
};
use super::runtime::{
    allocate_workspace, ensure_device_tensor, new_gpu_tensor, null_stream,
    tensor_device_ptr_with_offset, validate_runtime_shape,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::{CudaAnalyticPlan, CudaContext, CudaPlan, CudaPlanDescriptor};

pub(super) fn plan_analytic_descriptor<S: Scalar + 'static>(
    _ctx: &mut CudaContext,
    desc: &AnalyticPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    if !has_analytic_support::<S>(desc.clone()) {
        return Err(Error::InvalidArgument(format!(
            "analytic family descriptor {desc:?} is not supported on CudaBackend for {}",
            std::any::type_name::<S>()
        )));
    }

    let desc = match desc {
        AnalyticPrimsDescriptor::PointwiseUnary { op } => {
            validate_pointwise_shapes(shapes, 1, "AnalyticPointwiseUnary")?;
            CudaAnalyticPlan::PointwiseUnary { op: *op }
        }
        AnalyticPrimsDescriptor::PointwiseBinary { op } => {
            validate_pointwise_shapes(shapes, 2, "AnalyticPointwiseBinary")?;
            CudaAnalyticPlan::PointwiseBinary { op: *op }
        }
        AnalyticPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op,
        } => {
            let spec = plan_reduction_shapes(modes_a, modes_c, shapes, "AnalyticReduction")?;
            CudaAnalyticPlan::Reduction {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                op: *op,
                reduced_total: spec.reduced_total,
            }
        }
    };

    Ok(CudaPlan {
        desc: CudaPlanDescriptor::Analytic(desc),
        shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
        _marker: std::marker::PhantomData,
    })
}

pub(super) fn has_analytic_support<S: Scalar + 'static>(desc: AnalyticPrimsDescriptor) -> bool {
    if supports_real_scalar_type::<S>() {
        return true;
    }
    if supports_complex_scalar_type::<S>() {
        return match desc {
            AnalyticPrimsDescriptor::PointwiseUnary { op } => supports_complex_analytic_unary(op),
            AnalyticPrimsDescriptor::PointwiseBinary { op } => supports_complex_analytic_binary(op),
            AnalyticPrimsDescriptor::Reduction { .. } => false,
        };
    }
    false
}

pub(super) fn execute_analytic_plan<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let CudaPlanDescriptor::Analytic(desc) = &plan.desc else {
        return Err(Error::InvalidArgument(
            "expected CUDA analytic-family plan".into(),
        ));
    };

    match desc {
        CudaAnalyticPlan::PointwiseUnary { op } => {
            execute_analytic_unary(ctx, plan, *op, alpha, inputs, beta, output)
        }
        CudaAnalyticPlan::PointwiseBinary { op } => {
            execute_analytic_binary(ctx, plan, *op, alpha, inputs, beta, output)
        }
        CudaAnalyticPlan::Reduction {
            modes_a,
            modes_c,
            op,
            reduced_total,
        } => execute_analytic_reduction(
            ctx,
            plan,
            modes_a,
            modes_c,
            *op,
            *reduced_total,
            alpha,
            inputs,
            beta,
            output,
        ),
    }
}

fn execute_analytic_unary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    op: AnalyticUnaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 1, "AnalyticPointwiseUnary")?;
    ensure_device_tensor("input", inputs[0], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("input", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

    if supports_complex_scalar_type::<S>() {
        return execute_complex_analytic_unary(ctx, op, alpha, inputs[0], beta, output);
    }

    if let Some(op_a) = direct_real_analytic_unary(op) {
        return execute_direct_unary(ctx, op_a, alpha, inputs[0], beta, output);
    }

    let custom_op = match op {
        AnalyticUnaryOp::Rsqrt => RealUnaryKernelOp::Rsqrt,
        AnalyticUnaryOp::Expm1 => RealUnaryKernelOp::Expm1,
        AnalyticUnaryOp::Log1p => RealUnaryKernelOp::Log1p,
        _ => {
            return Err(Error::InvalidArgument(format!(
                "analytic unary operation {op:?} is not supported on CUDA for {}",
                std::any::type_name::<S>()
            )))
        }
    };
    execute_custom_real_unary(ctx, custom_op, alpha, inputs[0], beta, output)
}

fn execute_complex_analytic_unary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op: AnalyticUnaryOp,
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let custom_op = match op {
        AnalyticUnaryOp::Sqrt => ComplexUnaryKernelOp::Sqrt,
        AnalyticUnaryOp::Rsqrt => ComplexUnaryKernelOp::Rsqrt,
        AnalyticUnaryOp::Exp => ComplexUnaryKernelOp::Exp,
        AnalyticUnaryOp::Expm1 => ComplexUnaryKernelOp::Expm1,
        AnalyticUnaryOp::Log => ComplexUnaryKernelOp::Log,
        AnalyticUnaryOp::Log1p => ComplexUnaryKernelOp::Log1p,
        AnalyticUnaryOp::Sin => ComplexUnaryKernelOp::Sin,
        AnalyticUnaryOp::Cos => ComplexUnaryKernelOp::Cos,
        AnalyticUnaryOp::Tan => ComplexUnaryKernelOp::Tan,
        AnalyticUnaryOp::Tanh => ComplexUnaryKernelOp::Tanh,
        AnalyticUnaryOp::Asin => ComplexUnaryKernelOp::Asin,
        AnalyticUnaryOp::Acos => ComplexUnaryKernelOp::Acos,
        AnalyticUnaryOp::Atan => ComplexUnaryKernelOp::Atan,
        AnalyticUnaryOp::Sinh => ComplexUnaryKernelOp::Sinh,
        AnalyticUnaryOp::Cosh => ComplexUnaryKernelOp::Cosh,
        AnalyticUnaryOp::Asinh => ComplexUnaryKernelOp::Asinh,
        AnalyticUnaryOp::Acosh => ComplexUnaryKernelOp::Acosh,
        AnalyticUnaryOp::Atanh => ComplexUnaryKernelOp::Atanh,
    };
    execute_custom_complex_unary(ctx, custom_op, alpha, input, beta, output)
}

fn execute_analytic_binary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    op: AnalyticBinaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 2, "AnalyticPointwiseBinary")?;
    ensure_device_tensor("lhs", inputs[0], ctx.device_id)?;
    ensure_device_tensor("rhs", inputs[1], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("lhs", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("rhs", inputs[1].dims(), &plan.shapes[1])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[2])?;

    if supports_complex_scalar_type::<S>() {
        return execute_complex_analytic_binary(ctx, op, alpha, inputs, beta, output);
    }

    let custom_op = match op {
        AnalyticBinaryOp::Pow => RealBinaryKernelOp::Pow,
        AnalyticBinaryOp::Atan2 => RealBinaryKernelOp::Atan2,
        AnalyticBinaryOp::Hypot => RealBinaryKernelOp::Hypot,
        AnalyticBinaryOp::Xlogy => RealBinaryKernelOp::Xlogy,
    };
    execute_custom_real_binary(ctx, custom_op, alpha, inputs[0], inputs[1], beta, output)
}

fn execute_complex_analytic_binary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op: AnalyticBinaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let custom_op = match op {
        AnalyticBinaryOp::Pow => ComplexBinaryKernelOp::Pow,
        AnalyticBinaryOp::Xlogy => ComplexBinaryKernelOp::Xlogy,
        _ => {
            return Err(Error::InvalidArgument(format!(
                "analytic binary operation {op:?} is not supported on CUDA for {}",
                std::any::type_name::<S>()
            )))
        }
    };
    execute_custom_complex_binary(ctx, custom_op, alpha, inputs[0], inputs[1], beta, output)
}

fn execute_analytic_reduction<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    modes_a: &[u32],
    modes_c: &[u32],
    op: AnalyticReductionOp,
    reduced_total: usize,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 1, "AnalyticReduction")?;
    ensure_device_tensor("input", inputs[0], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("input", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

    let mean = reduce_mean(ctx, inputs[0], modes_a, modes_c, reduced_total)?;
    let mut square = new_gpu_tensor::<S>(inputs[0].dims(), ctx.device_id);
    execute_square(ctx, inputs[0], &mut square)?;
    let mean_square = reduce_mean(ctx, &square, modes_a, modes_c, reduced_total)?;
    let mut mean_squared = new_gpu_tensor::<S>(output.dims(), ctx.device_id);
    execute_binary_trinary(
        ctx,
        &mean,
        &mean,
        &mut mean_squared,
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_MUL,
        S::one(),
        S::one(),
        S::zero(),
    )?;
    let mut variance = new_gpu_tensor::<S>(output.dims(), ctx.device_id);
    execute_binary_trinary(
        ctx,
        &mean_square,
        &mean_squared,
        &mut variance,
        CUTENSOR_OP_NEG,
        CUTENSOR_OP_ADD,
        S::one(),
        S::one(),
        S::zero(),
    )?;

    match op {
        AnalyticReductionOp::Var => execute_copy_with_accum(ctx, alpha, &variance, beta, output),
        AnalyticReductionOp::Std => {
            execute_direct_unary(ctx, CUTENSOR_OP_SQRT, alpha, &variance, beta, output)
        }
    }
}

fn reduce_mean<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    input: &Tensor<S>,
    modes_a: &[u32],
    modes_c: &[u32],
    reduced_total: usize,
) -> Result<Tensor<S>> {
    let output_dims = reduction_output_dims(input.dims(), modes_a, modes_c)?;
    let output = new_gpu_tensor::<S>(&output_dims, ctx.device_id);
    let scaled_alpha = scale_real_alpha(S::one(), reduced_total)?;
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes_a_i32: Vec<i32> = modes_a.iter().map(|&mode| mode as i32).collect();
    let modes_c_i32: Vec<i32> = modes_c.iter().map(|&mode| mode as i32).collect();
    let native = plan_reduction(
        ctx,
        data_type,
        compute,
        &modes_a_i32,
        input.dims(),
        input.strides(),
        &modes_c_i32,
        output.dims(),
        output.strides(),
        CUTENSOR_OP_ADD,
    )?;
    let workspace = allocate_workspace(native.workspace_size)?;
    let ws_ptr = workspace.as_ref().map_or(std::ptr::null_mut(), |ws| ws.ptr);
    let input_ptr = tensor_device_ptr_with_offset("mean input", input)? as *const c_void;
    let out_ptr = tensor_device_ptr_with_offset("mean output", &output)?;
    let zero = S::zero();
    let status = unsafe {
        (ctx.vtable.reduce)(
            ctx.handle.raw,
            native.plan.raw,
            &scaled_alpha as *const S as *const c_void,
            input_ptr,
            &zero as *const S as *const c_void,
            out_ptr as *const c_void,
            out_ptr,
            ws_ptr,
            native.workspace_size,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorReduce")?;
    Ok(output)
}

fn reduction_output_dims(
    input_dims: &[usize],
    modes_a: &[u32],
    modes_c: &[u32],
) -> Result<Vec<usize>> {
    modes_c
        .iter()
        .map(|mode| {
            let axis = modes_a
                .iter()
                .position(|candidate| candidate == mode)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "reduction output mode {mode} is not present in input modes {modes_a:?}"
                    ))
                })?;
            input_dims.get(axis).copied().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "reduction axis {axis} is out of bounds for input rank {}",
                    input_dims.len()
                ))
            })
        })
        .collect()
}

fn supports_complex_analytic_unary(op: AnalyticUnaryOp) -> bool {
    matches!(
        op,
        AnalyticUnaryOp::Sqrt
            | AnalyticUnaryOp::Rsqrt
            | AnalyticUnaryOp::Exp
            | AnalyticUnaryOp::Expm1
            | AnalyticUnaryOp::Log
            | AnalyticUnaryOp::Log1p
            | AnalyticUnaryOp::Sin
            | AnalyticUnaryOp::Cos
            | AnalyticUnaryOp::Tan
            | AnalyticUnaryOp::Tanh
            | AnalyticUnaryOp::Asin
            | AnalyticUnaryOp::Acos
            | AnalyticUnaryOp::Atan
            | AnalyticUnaryOp::Sinh
            | AnalyticUnaryOp::Cosh
            | AnalyticUnaryOp::Asinh
            | AnalyticUnaryOp::Acosh
            | AnalyticUnaryOp::Atanh
    )
}

fn supports_complex_analytic_binary(op: AnalyticBinaryOp) -> bool {
    matches!(op, AnalyticBinaryOp::Pow | AnalyticBinaryOp::Xlogy)
}

fn direct_real_analytic_unary(op: AnalyticUnaryOp) -> Option<CutensorOperator> {
    match op {
        AnalyticUnaryOp::Sqrt => Some(CUTENSOR_OP_SQRT),
        AnalyticUnaryOp::Rsqrt => None,
        AnalyticUnaryOp::Exp => Some(CUTENSOR_OP_EXP),
        AnalyticUnaryOp::Expm1 => None,
        AnalyticUnaryOp::Log => Some(CUTENSOR_OP_LOG),
        AnalyticUnaryOp::Log1p => None,
        AnalyticUnaryOp::Sin => Some(CUTENSOR_OP_SIN),
        AnalyticUnaryOp::Cos => Some(CUTENSOR_OP_COS),
        AnalyticUnaryOp::Tan => Some(CUTENSOR_OP_TAN),
        AnalyticUnaryOp::Tanh => Some(CUTENSOR_OP_TANH),
        AnalyticUnaryOp::Asin => Some(CUTENSOR_OP_ASIN),
        AnalyticUnaryOp::Acos => Some(CUTENSOR_OP_ACOS),
        AnalyticUnaryOp::Atan => Some(CUTENSOR_OP_ATAN),
        AnalyticUnaryOp::Sinh => Some(CUTENSOR_OP_SINH),
        AnalyticUnaryOp::Cosh => Some(CUTENSOR_OP_COSH),
        AnalyticUnaryOp::Asinh => Some(CUTENSOR_OP_ASINH),
        AnalyticUnaryOp::Acosh => Some(CUTENSOR_OP_ACOSH),
        AnalyticUnaryOp::Atanh => Some(CUTENSOR_OP_ATANH),
    }
}
