use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{
    validate_execute_inputs, validate_shape_count, validate_shape_eq, SemiringBinaryOp,
    SemiringCoreDescriptor, SemiringFastPathDescriptor,
};

use super::diagonal::{
    execute_anti_diag, execute_anti_trace, execute_trace, validate_diagonal_plan,
};
use super::planning::{
    check_status, default_col_major_strides, plan_contraction, plan_elementwise_binary,
    plan_permutation, plan_reduction,
};
use super::runtime::allocate_workspace;
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::{CudaContext, CudaPlan, CudaPlanDescriptor, CudaPlanStorage};
use crate::cuda_ffi::{CUTENSOR_OP_ADD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL};

pub(super) fn plan_core_descriptor<S: Scalar>(
    ctx: &mut CudaContext,
    desc: &SemiringCoreDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    cudarc::runtime::result::device::set(ctx.device_id as i32)
        .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;

    match desc {
        SemiringCoreDescriptor::BatchedGemm {
            batch_dims,
            m: _,
            n: _,
            k: _,
        } => {
            validate_shape_count(shapes, 3, "BatchedGemm")?;
            let nb = batch_dims.len() as u32;
            let m_mode = 0i32;
            let k_mode = 1i32;
            let n_mode = 2i32;
            let batch_modes: Vec<i32> = (0..nb).map(|i| (i + 3) as i32).collect();
            let mut modes_a = Vec::with_capacity(2 + batch_modes.len());
            let mut modes_b = Vec::with_capacity(2 + batch_modes.len());
            let mut modes_c = Vec::with_capacity(2 + batch_modes.len());
            modes_a.extend([m_mode, k_mode]);
            modes_b.extend([k_mode, n_mode]);
            modes_c.extend([m_mode, n_mode]);
            modes_a.extend(batch_modes.iter().copied());
            modes_b.extend(batch_modes.iter().copied());
            modes_c.extend(batch_modes.iter().copied());
            let strides_a = default_col_major_strides(shapes[0]);
            let strides_b = default_col_major_strides(shapes[1]);
            let strides_c = default_col_major_strides(shapes[2]);
            let plan = plan_contraction(
                ctx, data_type, compute, &modes_a, shapes[0], &strides_a, &modes_b, shapes[1],
                &strides_b, &modes_c, shapes[2], &strides_c,
            )?;
            Ok(CudaPlan {
                plan: CudaPlanStorage::Compiled(plan),
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::Trace { .. }
        | SemiringCoreDescriptor::AntiTrace { .. }
        | SemiringCoreDescriptor::AntiDiag { .. } => {
            validate_diagonal_plan(desc, shapes)?;
            let _ = (data_type, compute);
            Ok(CudaPlan {
                plan: CudaPlanStorage::DeferredDiagonal,
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::ReduceAdd { modes_a, modes_c } => {
            validate_shape_count(shapes, 2, "ReduceAdd")?;
            let modes_a_i32: Vec<i32> = modes_a.iter().map(|&m| m as i32).collect();
            let modes_c_i32: Vec<i32> = modes_c.iter().map(|&m| m as i32).collect();
            let strides_a = default_col_major_strides(shapes[0]);
            let strides_c = default_col_major_strides(shapes[1]);
            let plan = plan_reduction(
                ctx,
                data_type,
                compute,
                &modes_a_i32,
                shapes[0],
                &strides_a,
                &modes_c_i32,
                shapes[1],
                &strides_c,
                CUTENSOR_OP_ADD,
            )?;
            Ok(CudaPlan {
                plan: CudaPlanStorage::Compiled(plan),
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::MakeContiguous => {
            validate_shape_count(shapes, 2, "MakeContiguous")?;
            validate_shape_eq(shapes[1], shapes[0], "MakeContiguous output")?;
            let _ = (data_type, compute);
            Ok(CudaPlan {
                plan: CudaPlanStorage::DeferredMakeContiguous,
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
    }
}

pub(super) fn plan_fast_descriptor<S: Scalar>(
    ctx: &mut CudaContext,
    desc: &SemiringFastPathDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    cudarc::runtime::result::device::set(ctx.device_id as i32)
        .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;

    match desc {
        SemiringFastPathDescriptor::Contract {
            modes_a,
            modes_b,
            modes_c,
        } => {
            validate_shape_count(shapes, 3, "Contract")?;
            let modes_a_i32: Vec<i32> = modes_a.iter().map(|&m| m as i32).collect();
            let modes_b_i32: Vec<i32> = modes_b.iter().map(|&m| m as i32).collect();
            let modes_c_i32: Vec<i32> = modes_c.iter().map(|&m| m as i32).collect();
            let strides_a = default_col_major_strides(shapes[0]);
            let strides_b = default_col_major_strides(shapes[1]);
            let strides_c = default_col_major_strides(shapes[2]);
            let plan = plan_contraction(
                ctx,
                data_type,
                compute,
                &modes_a_i32,
                shapes[0],
                &strides_a,
                &modes_b_i32,
                shapes[1],
                &strides_b,
                &modes_c_i32,
                shapes[2],
                &strides_c,
            )?;
            Ok(CudaPlan {
                plan: CudaPlanStorage::Compiled(plan),
                desc: CudaPlanDescriptor::Fast(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringFastPathDescriptor::ElementwiseBinary { op } => {
            validate_shape_count(shapes, 3, "ElementwiseBinary")?;
            let ndim = shapes[0].len();
            let modes: Vec<i32> = (0..ndim as i32).collect();
            let strides = default_col_major_strides(shapes[0]);
            let cutensor_op = match op {
                SemiringBinaryOp::Add => CUTENSOR_OP_ADD,
                SemiringBinaryOp::Mul => CUTENSOR_OP_MUL,
            };
            let plan = plan_elementwise_binary(
                ctx,
                data_type,
                compute,
                &modes,
                shapes[0],
                &strides,
                &strides,
                &strides,
                CUTENSOR_OP_IDENTITY,
                CUTENSOR_OP_IDENTITY,
                cutensor_op,
            )?;
            Ok(CudaPlan {
                plan: CudaPlanStorage::Compiled(plan),
                desc: CudaPlanDescriptor::Fast(desc.clone()),
                _marker: PhantomData,
            })
        }
    }
}

pub(super) fn execute_plan<S: Scalar>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<S>,
    alpha: S,
    inputs: &[&Tensor<S>],
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()> {
    cudarc::runtime::result::device::set(ctx.device_id as i32)
        .map_err(|e| Error::DeviceError(format!("CUDA runtime set-device failed: {e:?}")))?;
    let handle = ctx.handle.raw;
    let stream: *mut c_void = ptr::null_mut();

    let alpha_ptr = &alpha as *const S as *const c_void;
    let beta_ptr = &beta as *const S as *const c_void;

    match &plan.desc {
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::BatchedGemm { .. })
        | CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::Contract { .. }) => {
            validate_execute_inputs(inputs, 2, "Contraction")?;
            let CudaPlanStorage::Compiled(plan_handle) = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA contraction plan was not compiled".into(),
                ));
            };
            let workspace = allocate_workspace(plan_handle.workspace_size)?;
            let ws_ptr = workspace.as_ref().map_or(ptr::null_mut(), |ws| ws.ptr);
            let a_ptr = inputs[0]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                as *const c_void;
            let b_ptr = inputs[1]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input B not on GPU".into()))?
                as *const c_void;
            let c_ptr = output
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                as *const c_void;
            let d_ptr = c_ptr as *mut c_void;

            let status = unsafe {
                (ctx.vtable.contract)(
                    handle,
                    plan_handle.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    b_ptr,
                    beta_ptr,
                    c_ptr,
                    d_ptr,
                    ws_ptr,
                    plan_handle.workspace_size,
                    stream,
                )
            };
            check_status(status, "cutensorContract")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::MakeContiguous) => {
            validate_execute_inputs(inputs, 1, "MakeContiguous")?;
            validate_shape_eq(output.dims(), inputs[0].dims(), "MakeContiguous output")?;
            let data_type = scalar_data_type::<S>()?;
            let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;
            let ndim = inputs[0].dims().len();
            let modes: Vec<i32> = (0..ndim as i32).collect();
            let plan_handle = plan_permutation(
                ctx,
                data_type,
                compute,
                &modes,
                inputs[0].dims(),
                inputs[0].strides(),
                &modes,
                output.dims(),
                output.strides(),
            )?;
            let a_ptr = inputs[0]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                as *const c_void;
            let b_ptr = output
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                as *const c_void as *mut c_void;

            let status = unsafe {
                (ctx.vtable.permute)(
                    handle,
                    plan_handle.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    b_ptr,
                    stream,
                )
            };
            check_status(status, "cutensorPermute")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::ReduceAdd { .. }) => {
            validate_execute_inputs(inputs, 1, "ReduceAdd")?;
            let CudaPlanStorage::Compiled(plan_handle) = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA reduction plan was not compiled".into(),
                ));
            };
            let workspace = allocate_workspace(plan_handle.workspace_size)?;
            let ws_ptr = workspace.as_ref().map_or(ptr::null_mut(), |ws| ws.ptr);
            let a_ptr = inputs[0]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                as *const c_void;
            let c_ptr = output
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                as *const c_void;
            let d_ptr = c_ptr as *mut c_void;

            let status = unsafe {
                (ctx.vtable.reduce)(
                    handle,
                    plan_handle.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    beta_ptr,
                    c_ptr,
                    d_ptr,
                    ws_ptr,
                    plan_handle.workspace_size,
                    stream,
                )
            };
            check_status(status, "cutensorReduce")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::Trace {
            modes_a,
            modes_c,
            paired,
        }) => {
            validate_execute_inputs(inputs, 1, "Trace")?;
            let CudaPlanStorage::DeferredDiagonal = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA trace plan was expected to be deferred".into(),
                ));
            };
            execute_trace(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            )
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::AntiTrace {
            modes_a,
            modes_c,
            paired,
        }) => {
            validate_execute_inputs(inputs, 1, "AntiTrace")?;
            let CudaPlanStorage::DeferredDiagonal = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA anti-trace plan was expected to be deferred".into(),
                ));
            };
            execute_anti_trace(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            )
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::AntiDiag {
            modes_a,
            modes_c,
            paired,
        }) => {
            validate_execute_inputs(inputs, 1, "AntiDiag")?;
            let CudaPlanStorage::DeferredDiagonal = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA anti-diag plan was expected to be deferred".into(),
                ));
            };
            execute_anti_diag(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            )
        }
        CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::ElementwiseBinary { .. }) => {
            validate_execute_inputs(inputs, 2, "ElementwiseBinary")?;
            let CudaPlanStorage::Compiled(plan_handle) = &plan.plan else {
                return Err(Error::DeviceError(
                    "CUDA elementwise-binary plan was not compiled".into(),
                ));
            };
            let a_ptr = inputs[0]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))?
                as *const c_void;
            let c_ptr = inputs[1]
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input C not on GPU".into()))?
                as *const c_void;
            let d_ptr = output
                .buffer()
                .as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))?
                as *const c_void as *mut c_void;
            let gamma_ptr = beta_ptr;

            let status = unsafe {
                (ctx.vtable.elementwise_binary_execute)(
                    handle,
                    plan_handle.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    gamma_ptr,
                    c_ptr,
                    d_ptr,
                    stream,
                )
            };
            check_status(status, "cutensorElementwiseBinaryExecute")
        }
        _ => Err(Error::DeviceError(
            "CUDA execution for this semiring descriptor is not implemented yet".into(),
        )),
    }
}

pub(super) fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool {
    matches!(
        desc,
        SemiringFastPathDescriptor::Contract { .. }
            | SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Add | SemiringBinaryOp::Mul,
            }
    )
}
