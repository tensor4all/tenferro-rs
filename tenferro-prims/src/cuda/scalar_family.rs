use std::ffi::c_void;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::{
    CutensorOperator, CUTENSOR_OP_ABS, CUTENSOR_OP_ADD, CUTENSOR_OP_CONJ, CUTENSOR_OP_IDENTITY,
    CUTENSOR_OP_MAX, CUTENSOR_OP_MIN, CUTENSOR_OP_MUL, CUTENSOR_OP_NEG, CUTENSOR_OP_RCP,
};
use crate::{
    validate_execute_inputs, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp,
    ScalarUnaryOp,
};

use super::custom::RealUnaryKernelOp;
use super::family_common::{
    plan_reduction_shapes, scale_real_alpha, supports_real_scalar_type, validate_pointwise_shapes,
};
use super::planning::plan_reduction;
use super::pointwise_ops::{
    execute_binary_trinary, execute_copy_with_accum, execute_custom_real_unary,
    execute_direct_unary,
};
use super::runtime::{
    allocate_workspace, ensure_device_tensor, new_gpu_tensor, null_stream,
    tensor_device_ptr_with_offset, validate_runtime_shape,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::{CudaContext, CudaPlan, CudaPlanDescriptor, CudaScalarPlan};

pub(super) fn plan_scalar_descriptor<S: Scalar + 'static>(
    _ctx: &mut CudaContext,
    desc: &ScalarPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    if !has_scalar_support::<S>(desc.clone()) {
        return Err(Error::InvalidArgument(format!(
            "scalar family descriptor {desc:?} is not supported on CudaBackend for {}",
            std::any::type_name::<S>()
        )));
    }

    let desc = match desc {
        ScalarPrimsDescriptor::PointwiseUnary { op } => {
            validate_pointwise_shapes(shapes, 1, "ScalarPointwiseUnary")?;
            CudaScalarPlan::PointwiseUnary { op: *op }
        }
        ScalarPrimsDescriptor::PointwiseBinary { op } => {
            validate_pointwise_shapes(shapes, 2, "ScalarPointwiseBinary")?;
            CudaScalarPlan::PointwiseBinary { op: *op }
        }
        ScalarPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op,
        } => {
            let spec = plan_reduction_shapes(modes_a, modes_c, shapes, "ScalarReduction")?;
            CudaScalarPlan::Reduction {
                modes_a: modes_a.clone(),
                modes_c: modes_c.clone(),
                op: *op,
                reduced_total: spec.reduced_total,
            }
        }
    };

    Ok(CudaPlan {
        desc: CudaPlanDescriptor::Scalar(desc),
        shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
        _marker: std::marker::PhantomData,
    })
}

pub(super) fn has_scalar_support<S: Scalar + 'static>(desc: ScalarPrimsDescriptor) -> bool {
    if !supports_real_scalar_type::<S>() {
        return false;
    }

    match desc {
        ScalarPrimsDescriptor::PointwiseUnary { .. } => true,
        ScalarPrimsDescriptor::PointwiseBinary { .. } => true,
        ScalarPrimsDescriptor::Reduction { .. } => true,
    }
}

pub(super) fn execute_scalar_plan<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let CudaPlanDescriptor::Scalar(desc) = &plan.desc else {
        return Err(Error::InvalidArgument(
            "expected CUDA scalar-family plan".into(),
        ));
    };

    match desc {
        CudaScalarPlan::PointwiseUnary { op } => {
            execute_scalar_unary(ctx, plan, *op, alpha, inputs, beta, output)
        }
        CudaScalarPlan::PointwiseBinary { op } => {
            execute_scalar_binary(ctx, plan, *op, alpha, inputs, beta, output)
        }
        CudaScalarPlan::Reduction {
            modes_a,
            modes_c,
            op,
            reduced_total,
        } => execute_scalar_reduction(
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

fn execute_scalar_unary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    op: ScalarUnaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 1, "ScalarPointwiseUnary")?;
    ensure_device_tensor("input", inputs[0], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("input", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

    if let Some(op_a) = direct_scalar_unary(op) {
        return execute_direct_unary(ctx, op_a, alpha, inputs[0], beta, output);
    }

    match op {
        ScalarUnaryOp::Square => execute_binary_trinary(
            ctx,
            inputs[0],
            inputs[0],
            output,
            CUTENSOR_OP_IDENTITY,
            CUTENSOR_OP_MUL,
            alpha,
            S::one(),
            beta,
        ),
        ScalarUnaryOp::Imag => execute_custom_real_unary(
            ctx,
            RealUnaryKernelOp::ImagZero,
            alpha,
            inputs[0],
            beta,
            output,
        ),
        _ => Err(Error::InvalidArgument(format!(
            "scalar unary operation {op:?} is not supported on CUDA for {}",
            std::any::type_name::<S>()
        ))),
    }
}

fn execute_scalar_binary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    op: ScalarBinaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 2, "ScalarPointwiseBinary")?;
    ensure_device_tensor("lhs", inputs[0], ctx.device_id)?;
    ensure_device_tensor("rhs", inputs[1], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("lhs", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("rhs", inputs[1].dims(), &plan.shapes[1])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[2])?;

    if matches!(
        op,
        ScalarBinaryOp::Maximum
            | ScalarBinaryOp::Minimum
            | ScalarBinaryOp::ClampMin
            | ScalarBinaryOp::ClampMax
    ) {
        let mut temp = new_gpu_tensor::<S>(output.dims(), ctx.device_id);
        execute_scalar_binary_homogeneous(ctx, op, S::one(), inputs, S::zero(), &mut temp)?;
        return execute_copy_with_accum(ctx, alpha, &temp, beta, output);
    }

    execute_scalar_binary_homogeneous(ctx, op, alpha, inputs, beta, output)
}

fn execute_scalar_binary_homogeneous<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op: ScalarBinaryOp,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let (op_b, op_ab, rhs_scale) = match op {
        ScalarBinaryOp::Add => (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_ADD, alpha),
        ScalarBinaryOp::Sub => (CUTENSOR_OP_NEG, CUTENSOR_OP_ADD, alpha),
        ScalarBinaryOp::Mul => (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL, S::one()),
        ScalarBinaryOp::Div => (CUTENSOR_OP_RCP, CUTENSOR_OP_MUL, S::one()),
        ScalarBinaryOp::Maximum | ScalarBinaryOp::ClampMin => {
            (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MAX, S::one())
        }
        ScalarBinaryOp::Minimum | ScalarBinaryOp::ClampMax => {
            (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MIN, S::one())
        }
    };
    execute_binary_trinary(
        ctx, inputs[0], inputs[1], output, op_b, op_ab, alpha, rhs_scale, beta,
    )
}

fn execute_scalar_reduction<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    modes_a: &[u32],
    modes_c: &[u32],
    op: ScalarReductionOp,
    reduced_total: usize,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    validate_execute_inputs(inputs, 1, "ScalarReduction")?;
    ensure_device_tensor("input", inputs[0], ctx.device_id)?;
    ensure_device_tensor("output", output, ctx.device_id)?;
    validate_runtime_shape("input", inputs[0].dims(), &plan.shapes[0])?;
    validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let reduce_op = match op {
        ScalarReductionOp::Sum | ScalarReductionOp::Mean => CUTENSOR_OP_ADD,
        ScalarReductionOp::Prod => CUTENSOR_OP_MUL,
        ScalarReductionOp::Max => CUTENSOR_OP_MAX,
        ScalarReductionOp::Min => CUTENSOR_OP_MIN,
    };
    let scaled_alpha = if matches!(op, ScalarReductionOp::Mean) {
        scale_real_alpha(alpha, reduced_total)?
    } else {
        alpha
    };
    let modes_a_i32: Vec<i32> = modes_a.iter().map(|&mode| mode as i32).collect();
    let modes_c_i32: Vec<i32> = modes_c.iter().map(|&mode| mode as i32).collect();
    let native = plan_reduction(
        ctx,
        data_type,
        compute,
        &modes_a_i32,
        inputs[0].dims(),
        inputs[0].strides(),
        &modes_c_i32,
        output.dims(),
        output.strides(),
        reduce_op,
    )?;
    let workspace = allocate_workspace(native.workspace_size)?;
    let ws_ptr = workspace.as_ref().map_or(std::ptr::null_mut(), |ws| ws.ptr);
    let input_ptr = tensor_device_ptr_with_offset("input", inputs[0])? as *const c_void;
    let out_ptr = tensor_device_ptr_with_offset("output", output)?;
    let status = unsafe {
        (ctx.vtable.reduce)(
            ctx.handle.raw,
            native.plan.raw,
            &scaled_alpha as *const S as *const c_void,
            input_ptr,
            &beta as *const S as *const c_void,
            out_ptr as *const c_void,
            out_ptr,
            ws_ptr,
            native.workspace_size,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorReduce")
}

fn direct_scalar_unary(op: ScalarUnaryOp) -> Option<CutensorOperator> {
    match op {
        ScalarUnaryOp::Neg => Some(CUTENSOR_OP_NEG),
        ScalarUnaryOp::Conj => Some(CUTENSOR_OP_CONJ),
        ScalarUnaryOp::Abs => Some(CUTENSOR_OP_ABS),
        ScalarUnaryOp::Reciprocal => Some(CUTENSOR_OP_RCP),
        ScalarUnaryOp::Real => Some(CUTENSOR_OP_IDENTITY),
        ScalarUnaryOp::Imag | ScalarUnaryOp::Square => None,
    }
}
