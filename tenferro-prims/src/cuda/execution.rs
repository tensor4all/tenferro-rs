use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use crate::{
    validate_execute_inputs, validate_shape_count, SemiringBinaryOp, SemiringCoreDescriptor,
    SemiringFastPathDescriptor,
};

use super::planning::{
    check_status, plan_contraction, plan_elementwise_trinary, plan_permutation, plan_reduction,
    TrinaryPlanSpec,
};
use super::scalar_type::{scalar_compute_descriptor, scalar_data_type};
use super::{CudaContext, CudaPlan, CudaPlanDescriptor};
use crate::cuda_ffi::{CUTENSOR_OP_ADD, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_MUL};

pub(super) fn plan_core_descriptor<S: Scalar>(
    _ctx: &mut CudaContext,
    desc: &SemiringCoreDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    match desc {
        SemiringCoreDescriptor::BatchedGemm { .. } => {
            validate_shape_count(shapes, 3, "BatchedGemm")?;
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Core(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::Trace { .. }
        | SemiringCoreDescriptor::AntiTrace { .. }
        | SemiringCoreDescriptor::AntiDiag { .. } => {
            super::diagonal::validate_diagonal_plan(desc, shapes)?;
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Core(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::ReduceAdd { modes_a, modes_c } => {
            validate_shape_count(shapes, 2, "ReduceAdd")?;
            let _ = (modes_a, modes_c);
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Core(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
        SemiringCoreDescriptor::MakeContiguous => {
            validate_shape_count(shapes, 2, "MakeContiguous")?;
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Core(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
    }
}

pub(super) fn plan_fast_descriptor<S: Scalar>(
    _ctx: &mut CudaContext,
    desc: &SemiringFastPathDescriptor,
    shapes: &[&[usize]],
) -> Result<CudaPlan<S>> {
    match desc {
        SemiringFastPathDescriptor::Contract { .. } => {
            validate_shape_count(shapes, 3, "Contract")?;
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Fast(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
        SemiringFastPathDescriptor::ElementwiseBinary { op } => {
            validate_shape_count(shapes, 3, "ElementwiseBinary")?;
            let _ = op;
            Ok(CudaPlan {
                desc: CudaPlanDescriptor::Fast(desc.clone()),
                shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
                _marker: PhantomData,
            })
        }
    }
}

struct WorkspaceBuffer {
    ptr: *mut c_void,
}

impl Drop for WorkspaceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudarc::runtime::result::free_sync(self.ptr) };
        }
    }
}

fn allocate_workspace(size: u64) -> Result<Option<WorkspaceBuffer>> {
    if size == 0 {
        return Ok(None);
    }
    let ptr = unsafe { cudarc::runtime::result::malloc_sync(size as usize) }
        .map_err(|e| Error::DeviceError(format!("cudaMalloc workspace failed: {e:?}")))?;
    Ok(Some(WorkspaceBuffer { ptr }))
}

fn validate_runtime_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "{name} shape mismatch: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn ensure_device_tensor<S: Scalar>(name: &str, tensor: &Tensor<S>, device_id: usize) -> Result<()> {
    match tensor.logical_memory_space() {
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } if tensor_device == device_id => Ok(()),
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } => Err(Error::DeviceError(format!(
            "{name} is on GPU device {tensor_device}, expected {device_id}"
        ))),
        space => Err(Error::DeviceError(format!(
            "{name} is not resident on GPU device {device_id}: {space:?}"
        ))),
    }
}

fn tensor_device_ptr_with_offset<S: Scalar>(name: &str, tensor: &Tensor<S>) -> Result<*mut c_void> {
    let ptr = tensor
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError(format!("{name} not on GPU")))? as *mut S;
    Ok(unsafe { ptr.offset(tensor.offset()) } as *mut c_void)
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
    let data_type = scalar_data_type::<S>()?;
    let compute = scalar_compute_descriptor::<S>(&ctx.vtable)?;

    match &plan.desc {
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::BatchedGemm { .. })
        | CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::Contract { .. }) => {
            validate_execute_inputs(inputs, 2, "Contraction")?;
            ensure_device_tensor("input A", inputs[0], ctx.device_id)?;
            ensure_device_tensor("input B", inputs[1], ctx.device_id)?;
            ensure_device_tensor("output", output, ctx.device_id)?;
            validate_runtime_shape("input A", inputs[0].dims(), &plan.shapes[0])?;
            validate_runtime_shape("input B", inputs[1].dims(), &plan.shapes[1])?;
            validate_runtime_shape("output", output.dims(), &plan.shapes[2])?;

            let desc = match &plan.desc {
                CudaPlanDescriptor::Core(SemiringCoreDescriptor::BatchedGemm {
                    batch_dims,
                    ..
                }) => {
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
                    (modes_a, modes_b, modes_c)
                }
                CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::Contract {
                    modes_a,
                    modes_b,
                    modes_c,
                }) => (
                    modes_a.iter().map(|&m| m as i32).collect(),
                    modes_b.iter().map(|&m| m as i32).collect(),
                    modes_c.iter().map(|&m| m as i32).collect(),
                ),
                _ => unreachable!(),
            };

            let native = plan_contraction(
                ctx,
                data_type,
                compute,
                &desc.0,
                inputs[0].dims(),
                inputs[0].strides(),
                &desc.1,
                inputs[1].dims(),
                inputs[1].strides(),
                &desc.2,
                output.dims(),
                output.strides(),
            )?;
            let workspace = allocate_workspace(native.workspace_size)?;
            let ws_ptr = workspace
                .as_ref()
                .map_or(ptr::null_mut(), |buffer| buffer.ptr);
            let ws_size = native.workspace_size;

            let a_ptr = tensor_device_ptr_with_offset("input A", inputs[0])? as *const c_void;
            let b_ptr = tensor_device_ptr_with_offset("input B", inputs[1])? as *const c_void;
            let c_ptr = tensor_device_ptr_with_offset("output", output)? as *const c_void;
            let d_ptr = c_ptr as *mut c_void;

            let status = unsafe {
                (ctx.vtable.contract)(
                    handle,
                    native.plan.raw,
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
            ensure_device_tensor("input A", inputs[0], ctx.device_id)?;
            ensure_device_tensor("output", output, ctx.device_id)?;
            validate_runtime_shape("input A", inputs[0].dims(), &plan.shapes[0])?;
            validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

            if beta != S::zero() {
                return Err(Error::DeviceError(
                    "MakeContiguous on CUDA currently requires beta == 0".into(),
                ));
            }

            let ndim = inputs[0].dims().len();
            let modes: Vec<i32> = (0..ndim as i32).collect();
            let native = plan_permutation(
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
            let a_ptr = tensor_device_ptr_with_offset("input A", inputs[0])? as *const c_void;
            let b_ptr = tensor_device_ptr_with_offset("output", output)?;

            let status = unsafe {
                (ctx.vtable.permute)(handle, native.plan.raw, alpha_ptr, a_ptr, b_ptr, stream)
            };
            check_status(status, "cutensorPermute")
        }
        CudaPlanDescriptor::Core(SemiringCoreDescriptor::ReduceAdd { .. }) => {
            validate_execute_inputs(inputs, 1, "ReduceAdd")?;
            ensure_device_tensor("input A", inputs[0], ctx.device_id)?;
            ensure_device_tensor("output", output, ctx.device_id)?;
            validate_runtime_shape("input A", inputs[0].dims(), &plan.shapes[0])?;
            validate_runtime_shape("output", output.dims(), &plan.shapes[1])?;

            let (modes_a, modes_c) = match &plan.desc {
                CudaPlanDescriptor::Core(SemiringCoreDescriptor::ReduceAdd {
                    modes_a,
                    modes_c,
                }) => (
                    modes_a.iter().map(|&m| m as i32).collect::<Vec<_>>(),
                    modes_c.iter().map(|&m| m as i32).collect::<Vec<_>>(),
                ),
                _ => unreachable!(),
            };
            let native = plan_reduction(
                ctx,
                data_type,
                compute,
                &modes_a,
                inputs[0].dims(),
                inputs[0].strides(),
                &modes_c,
                output.dims(),
                output.strides(),
                CUTENSOR_OP_ADD,
            )?;
            let workspace = allocate_workspace(native.workspace_size)?;
            let ws_ptr = workspace
                .as_ref()
                .map_or(ptr::null_mut(), |buffer| buffer.ptr);
            let ws_size = native.workspace_size;

            let a_ptr = tensor_device_ptr_with_offset("input A", inputs[0])? as *const c_void;
            let c_ptr = tensor_device_ptr_with_offset("output", output)? as *const c_void;
            let d_ptr = c_ptr as *mut c_void;

            let status = unsafe {
                (ctx.vtable.reduce)(
                    handle,
                    native.plan.raw,
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
            ensure_device_tensor("input A", inputs[0], ctx.device_id)?;
            ensure_device_tensor("input B", inputs[1], ctx.device_id)?;
            ensure_device_tensor("output", output, ctx.device_id)?;
            validate_runtime_shape("input A", inputs[0].dims(), &plan.shapes[0])?;
            validate_runtime_shape("input B", inputs[1].dims(), &plan.shapes[1])?;
            validate_runtime_shape("output", output.dims(), &plan.shapes[2])?;

            let cutensor_op = match &plan.desc {
                CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::ElementwiseBinary { op }) => {
                    match op {
                        SemiringBinaryOp::Add => CUTENSOR_OP_ADD,
                        SemiringBinaryOp::Mul => CUTENSOR_OP_MUL,
                    }
                }
                _ => unreachable!(),
            };
            let ndim = inputs[0].dims().len();
            let modes: Vec<i32> = (0..ndim as i32).collect();
            let native = plan_elementwise_trinary(
                ctx,
                data_type,
                compute,
                TrinaryPlanSpec {
                    modes_a: &modes,
                    shape_a: inputs[0].dims(),
                    strides_a: inputs[0].strides(),
                    op_a: CUTENSOR_OP_IDENTITY,
                    modes_b: &modes,
                    shape_b: inputs[1].dims(),
                    strides_b: inputs[1].strides(),
                    op_b: CUTENSOR_OP_IDENTITY,
                    modes_c: &modes,
                    shape_c: output.dims(),
                    strides_c: output.strides(),
                    op_c: CUTENSOR_OP_IDENTITY,
                    shape_d: output.dims(),
                    strides_d: output.strides(),
                    op_ab: cutensor_op,
                    op_abc: CUTENSOR_OP_ADD,
                },
            )?;

            let a_ptr = tensor_device_ptr_with_offset("input A", inputs[0])? as *const c_void;
            let b_ptr = tensor_device_ptr_with_offset("input B", inputs[1])? as *const c_void;
            let d_ptr = tensor_device_ptr_with_offset("output", output)?;
            let rhs_scale = match &plan.desc {
                CudaPlanDescriptor::Fast(SemiringFastPathDescriptor::ElementwiseBinary { op }) => {
                    match op {
                        SemiringBinaryOp::Add => alpha,
                        SemiringBinaryOp::Mul => S::one(),
                    }
                }
                _ => unreachable!(),
            };

            let status = unsafe {
                (ctx.vtable.elementwise_trinary_execute)(
                    handle,
                    native.plan.raw,
                    alpha_ptr,
                    a_ptr,
                    &rhs_scale as *const S as *const c_void,
                    b_ptr,
                    beta_ptr,
                    d_ptr as *const c_void,
                    d_ptr,
                    stream,
                )
            };
            check_status(status, "cutensorElementwiseTrinaryExecute")
        }
        CudaPlanDescriptor::Core(
            SemiringCoreDescriptor::Trace { .. }
            | SemiringCoreDescriptor::AntiTrace { .. }
            | SemiringCoreDescriptor::AntiDiag { .. },
        ) => match &plan.desc {
            CudaPlanDescriptor::Core(SemiringCoreDescriptor::Trace {
                modes_a,
                modes_c,
                paired,
            }) => super::diagonal::execute_trace(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            ),
            CudaPlanDescriptor::Core(SemiringCoreDescriptor::AntiTrace {
                modes_a,
                modes_c,
                paired,
            }) => super::diagonal::execute_anti_trace(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            ),
            CudaPlanDescriptor::Core(SemiringCoreDescriptor::AntiDiag {
                modes_a,
                modes_c,
                paired,
            }) => super::diagonal::execute_anti_diag(
                ctx, modes_a, modes_c, paired, alpha, inputs[0], beta, output,
            ),
            _ => unreachable!(),
        },
        CudaPlanDescriptor::Scalar(_) | CudaPlanDescriptor::Analytic(_) => Err(
            Error::InvalidArgument("execute_plan received a non-semiring CUDA plan".into()),
        ),
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
