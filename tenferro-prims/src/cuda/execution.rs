use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{
    validate_execute_inputs, validate_shape_count, SemiringBinaryOp, SemiringCoreDescriptor,
    SemiringFastPathDescriptor,
};

use super::planning::{
    check_status, default_col_major_strides, plan_contraction, plan_elementwise_binary,
    plan_permutation, plan_reduction,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::{CudaContext, CudaPlan, CudaPlanDescriptor};
use crate::cuda_ffi::{CUTENSOR_OP_ADD, CUTENSOR_OP_MUL};

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
            let mut modes_a = Vec::new();
            let mut modes_b = Vec::new();
            let mut modes_c = Vec::new();
            for i in 0..nb {
                modes_a.push(i as i32);
                modes_b.push(i as i32);
                modes_c.push(i as i32);
            }
            let m_mode = nb as i32;
            let k_mode = (nb + 1) as i32;
            let n_mode = (nb + 2) as i32;
            modes_a.extend([m_mode, k_mode]);
            modes_b.extend([k_mode, n_mode]);
            modes_c.extend([m_mode, n_mode]);
            let strides_a = default_col_major_strides(shapes[0]);
            let strides_b = default_col_major_strides(shapes[1]);
            let strides_c = default_col_major_strides(shapes[2]);
            let plan = plan_contraction(
                ctx, data_type, compute, &modes_a, shapes[0], &strides_a, &modes_b, shapes[1],
                &strides_b, &modes_c, shapes[2], &strides_c,
            )?;
            Ok(CudaPlan {
                plan,
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::Trace { .. }
        | SemiringCoreDescriptor::AntiTrace { .. }
        | SemiringCoreDescriptor::AntiDiag { .. } => Err(Error::DeviceError(
            "Trace/AntiTrace/AntiDiag not yet supported on CUDA backend".into(),
        )),
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
                plan,
                desc: CudaPlanDescriptor::Core(desc.clone()),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::MakeContiguous => {
            validate_shape_count(shapes, 2, "MakeContiguous")?;
            let ndim = shapes[0].len();
            let modes: Vec<i32> = (0..ndim as i32).collect();
            let strides_a = default_col_major_strides(shapes[0]);
            let strides_b = default_col_major_strides(shapes[1]);
            let plan = plan_permutation(
                ctx, data_type, compute, &modes, shapes[0], &strides_a, &modes, shapes[1],
                &strides_b,
            )?;
            Ok(CudaPlan {
                plan,
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
                plan,
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
                cutensor_op,
            )?;
            Ok(CudaPlan {
                plan,
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
    let ws_ptr: *mut c_void = ptr::null_mut();
    let ws_size: u64 = 0;

    let alpha_ptr = &alpha as *const S as *const c_void;
    let beta_ptr = &beta as *const S as *const c_void;

    match &plan.desc {
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::BatchedGemm { .. })
        | CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::Contract { .. }) => {
            validate_execute_inputs(inputs, 2, "Contraction")?;
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
                    plan.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    b_ptr,
                    beta_ptr,
                    c_ptr,
                    d_ptr,
                    ws_ptr,
                    ws_size,
                    stream,
                )
            };
            check_status(status, "cutensorContract")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::MakeContiguous) => {
            validate_execute_inputs(inputs, 1, "MakeContiguous")?;
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
                (ctx.vtable.permute)(handle, plan.plan.raw, alpha_ptr, a_ptr, b_ptr, stream)
            };
            check_status(status, "cutensorPermute")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::ReduceAdd { .. }) => {
            validate_execute_inputs(inputs, 1, "ReduceAdd")?;
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
                    plan.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    beta_ptr,
                    c_ptr,
                    d_ptr,
                    ws_ptr,
                    ws_size,
                    stream,
                )
            };
            check_status(status, "cutensorReduce")
        }
        CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::ElementwiseBinary { .. }) => {
            validate_execute_inputs(inputs, 2, "ElementwiseBinary")?;
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
                    plan.plan.raw,
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
