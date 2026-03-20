use std::ffi::c_void;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::{CutensorOperator, CUTENSOR_OP_ADD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL};
use crate::infra::typed_dispatch::{cast_scalar_value, dispatch_real_scalar_type};

use super::custom::{RealBinaryKernelOp, RealUnaryKernelOp};
use super::planning::{plan_elementwise_binary, plan_elementwise_trinary, TrinaryPlanSpec};
use super::runtime::{
    make_contiguous_on_cuda, null_stream, prepare_custom_output, tensor_device_addr_with_offset,
    tensor_device_ptr_with_offset, write_custom_output_back,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::CudaContext;

pub(super) fn execute_direct_unary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op_a: CutensorOperator,
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes: Vec<i32> = (0..input.ndim() as i32).collect();
    let native = plan_elementwise_binary(
        ctx,
        data_type,
        compute,
        &modes,
        input.dims(),
        input.strides(),
        output.strides(),
        output.strides(),
        op_a,
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_ADD,
    )?;
    let input_ptr = tensor_device_ptr_with_offset("input", input)? as *const c_void;
    let output_ptr = tensor_device_ptr_with_offset("output", output)?;
    let status = unsafe {
        (ctx.vtable.elementwise_binary_execute)(
            ctx.handle.raw,
            native.plan.raw,
            &alpha as *const S as *const c_void,
            input_ptr,
            &beta as *const S as *const c_void,
            output_ptr as *const c_void,
            output_ptr,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorElementwiseBinaryExecute")
}

pub(super) fn execute_copy_with_accum<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    execute_direct_unary(ctx, CUTENSOR_OP_IDENTITY, alpha, input, beta, output)
}

pub(super) fn execute_square<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    input: &Tensor<S>,
    output: &mut Tensor<S>,
) -> Result<()> {
    execute_binary_trinary(
        ctx,
        input,
        input,
        output,
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_MUL,
        S::one(),
        S::one(),
        S::zero(),
    )
}

pub(super) fn execute_binary_trinary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    lhs: &Tensor<S>,
    rhs: &Tensor<S>,
    output: &mut Tensor<S>,
    op_b: CutensorOperator,
    op_ab: CutensorOperator,
    lhs_scale: S,
    rhs_scale: S,
    output_scale: S,
) -> Result<()> {
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
    let modes: Vec<i32> = (0..lhs.ndim() as i32).collect();
    let native = plan_elementwise_trinary(
        ctx,
        data_type,
        compute,
        TrinaryPlanSpec {
            modes_a: &modes,
            shape_a: lhs.dims(),
            strides_a: lhs.strides(),
            op_a: CUTENSOR_OP_IDENTITY,
            modes_b: &modes,
            shape_b: rhs.dims(),
            strides_b: rhs.strides(),
            op_b,
            modes_c: &modes,
            shape_c: output.dims(),
            strides_c: output.strides(),
            op_c: CUTENSOR_OP_IDENTITY,
            shape_d: output.dims(),
            strides_d: output.strides(),
            op_ab,
            op_abc: CUTENSOR_OP_ADD,
        },
    )?;
    let lhs_ptr = tensor_device_ptr_with_offset("lhs", lhs)? as *const c_void;
    let rhs_ptr = tensor_device_ptr_with_offset("rhs", rhs)? as *const c_void;
    let output_ptr = tensor_device_ptr_with_offset("output", output)?;
    let status = unsafe {
        (ctx.vtable.elementwise_trinary_execute)(
            ctx.handle.raw,
            native.plan.raw,
            &lhs_scale as *const S as *const c_void,
            lhs_ptr,
            &rhs_scale as *const S as *const c_void,
            rhs_ptr,
            &output_scale as *const S as *const c_void,
            output_ptr as *const c_void,
            output_ptr,
            null_stream(),
        )
    };
    super::planning::check_status(status, "cutensorElementwiseTrinaryExecute")
}

pub(super) fn execute_custom_real_unary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op: RealUnaryKernelOp,
    alpha: S,
    input: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let contiguous_input = make_contiguous_on_cuda(ctx, input)?;
    let (contiguous_output, needs_copy_back) = prepare_custom_output(ctx, output)?;
    let input_addr = tensor_device_addr_with_offset("custom unary input", &contiguous_input)?;
    let output_addr = tensor_device_addr_with_offset("custom unary output", &contiguous_output)?;

    dispatch_real_scalar_type!(S, Concrete, {
        let alpha_concrete = cast_scalar_value!(alpha, S, Concrete);
        let beta_concrete = cast_scalar_value!(beta, S, Concrete);
        if std::mem::size_of::<Concrete>() == std::mem::size_of::<f32>() {
            ctx.custom.launch_pointwise_unary_f32(
                op,
                input_addr,
                output_addr,
                contiguous_input.len(),
                cast_scalar_value!(alpha_concrete, Concrete, f32),
                cast_scalar_value!(beta_concrete, Concrete, f32),
            )?;
        } else {
            ctx.custom.launch_pointwise_unary_f64(
                op,
                input_addr,
                output_addr,
                contiguous_input.len(),
                cast_scalar_value!(alpha_concrete, Concrete, f64),
                cast_scalar_value!(beta_concrete, Concrete, f64),
            )?;
        }
        return if needs_copy_back {
            write_custom_output_back(ctx, &contiguous_output, output)
        } else {
            Ok(())
        };
    });

    Err(Error::InvalidArgument(format!(
        "custom CUDA unary path is not supported for {}",
        std::any::type_name::<S>()
    )))
}

pub(super) fn execute_custom_real_binary<S: Scalar + 'static>(
    ctx: &mut CudaContext,
    op: RealBinaryKernelOp,
    alpha: S,
    lhs: &Tensor<S>,
    rhs: &Tensor<S>,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    let contiguous_lhs = make_contiguous_on_cuda(ctx, lhs)?;
    let contiguous_rhs = make_contiguous_on_cuda(ctx, rhs)?;
    let (contiguous_output, needs_copy_back) = prepare_custom_output(ctx, output)?;
    let lhs_addr = tensor_device_addr_with_offset("custom binary lhs", &contiguous_lhs)?;
    let rhs_addr = tensor_device_addr_with_offset("custom binary rhs", &contiguous_rhs)?;
    let output_addr = tensor_device_addr_with_offset("custom binary output", &contiguous_output)?;

    dispatch_real_scalar_type!(S, Concrete, {
        let alpha_concrete = cast_scalar_value!(alpha, S, Concrete);
        let beta_concrete = cast_scalar_value!(beta, S, Concrete);
        if std::mem::size_of::<Concrete>() == std::mem::size_of::<f32>() {
            ctx.custom.launch_pointwise_binary_f32(
                op,
                lhs_addr,
                rhs_addr,
                output_addr,
                contiguous_lhs.len(),
                cast_scalar_value!(alpha_concrete, Concrete, f32),
                cast_scalar_value!(beta_concrete, Concrete, f32),
            )?;
        } else {
            ctx.custom.launch_pointwise_binary_f64(
                op,
                lhs_addr,
                rhs_addr,
                output_addr,
                contiguous_lhs.len(),
                cast_scalar_value!(alpha_concrete, Concrete, f64),
                cast_scalar_value!(beta_concrete, Concrete, f64),
            )?;
        }
        return if needs_copy_back {
            write_custom_output_back(ctx, &contiguous_output, output)
        } else {
            Ok(())
        };
    });

    Err(Error::InvalidArgument(format!(
        "custom CUDA binary path is not supported for {}",
        std::any::type_name::<S>()
    )))
}
