use std::ptr;
use std::sync::Arc;

use tenferro_device::{Error, Result};

use crate::cuda_ffi::*;

use super::wrappers::{OpDescWrapper, PlanPrefWrapper, PlanWrapper, TensorDescWrapper};
use super::CudaContext;

#[derive(Clone, Debug)]
pub(super) struct NativeCutensorPlan {
    pub(super) plan: PlanWrapper,
    pub(super) workspace_size: u64,
}

pub(super) fn default_col_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim as isize);
    }
    strides
}

/// Check a cuTENSOR status code, converting non-success to Error.
pub(super) fn check_status(status: cutensorStatus_t, context: &str) -> Result<()> {
    if status == CUTENSOR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "cuTENSOR error {status} in {context}"
        )))
    }
}

fn create_tensor_desc(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    shape: &[usize],
    strides: &[isize],
    data_type: CutensorDataType,
) -> Result<TensorDescWrapper> {
    let num_modes = shape.len() as u32;
    let extent: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let stride: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
    let mut raw: cutensorTensorDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_tensor_descriptor)(
            handle,
            &mut raw,
            num_modes,
            extent.as_ptr(),
            stride.as_ptr(),
            data_type,
            128,
        )
    };
    check_status(status, "cutensorCreateTensorDescriptor")?;
    Ok(TensorDescWrapper {
        raw,
        vtable: Arc::clone(vtable),
    })
}

fn build_cutensor_plan(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    op_desc: &OpDescWrapper,
) -> Result<NativeCutensorPlan> {
    let mut pref_raw: cutensorPlanPreference_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan_preference)(
            handle,
            &mut pref_raw,
            CUTENSOR_ALGO_DEFAULT,
            CUTENSOR_JIT_MODE_NONE,
        )
    };
    check_status(status, "cutensorCreatePlanPreference")?;
    let pref = PlanPrefWrapper {
        raw: pref_raw,
        vtable: Arc::clone(vtable),
    };

    let mut workspace_size: u64 = 0;
    let status = unsafe {
        (vtable.estimate_workspace_size)(
            handle,
            op_desc.raw,
            pref.raw,
            CUTENSOR_WORKSPACE_DEFAULT,
            &mut workspace_size,
        )
    };
    check_status(status, "cutensorEstimateWorkspaceSize")?;

    let mut plan_raw: cutensorPlan_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan)(handle, &mut plan_raw, op_desc.raw, pref.raw, workspace_size)
    };
    check_status(status, "cutensorCreatePlan")?;

    Ok(NativeCutensorPlan {
        plan: PlanWrapper {
            raw: plan_raw,
            vtable: Arc::clone(vtable),
        },
        workspace_size,
    })
}

pub(super) fn plan_contraction(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
) -> Result<NativeCutensorPlan> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;

    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_contraction)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_b.raw,
            modes_b.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_c.raw,
            modes_c.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_d.raw,
            modes_c.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreateContraction")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

pub(super) fn plan_permutation(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
) -> Result<NativeCutensorPlan> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_permutation)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_b.raw,
            modes_b.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreatePermutation")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

pub(super) fn plan_reduction(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
    reduce_op: CutensorOperator,
) -> Result<NativeCutensorPlan> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_reduction)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes_a.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_c.raw,
            modes_c.as_ptr(),
            CUTENSOR_OP_IDENTITY,
            desc_d.raw,
            modes_c.as_ptr(),
            reduce_op,
            compute,
        )
    };
    check_status(status, "cutensorCreateReduction")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

pub(super) fn plan_elementwise_binary(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    modes: &[i32],
    shape: &[usize],
    strides_a: &[isize],
    strides_c: &[isize],
    strides_d: &[isize],
    op_a: CutensorOperator,
    op_c: CutensorOperator,
    op: CutensorOperator,
) -> Result<NativeCutensorPlan> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape, strides_a, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, shape, strides_c, data_type)?;
    let desc_d = create_tensor_desc(handle, vtable, shape, strides_d, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_elementwise_binary)(
            handle,
            &mut op_raw,
            desc_a.raw,
            modes.as_ptr(),
            op_a,
            desc_c.raw,
            modes.as_ptr(),
            op_c,
            desc_d.raw,
            modes.as_ptr(),
            op,
            compute,
        )
    };
    check_status(status, "cutensorCreateElementwiseBinary")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}

pub(super) struct TrinaryPlanSpec<'a> {
    pub(super) modes_a: &'a [i32],
    pub(super) shape_a: &'a [usize],
    pub(super) strides_a: &'a [isize],
    pub(super) op_a: CutensorOperator,
    pub(super) modes_b: &'a [i32],
    pub(super) shape_b: &'a [usize],
    pub(super) strides_b: &'a [isize],
    pub(super) op_b: CutensorOperator,
    pub(super) modes_c: &'a [i32],
    pub(super) shape_c: &'a [usize],
    pub(super) strides_c: &'a [isize],
    pub(super) op_c: CutensorOperator,
    pub(super) shape_d: &'a [usize],
    pub(super) strides_d: &'a [isize],
    pub(super) op_ab: CutensorOperator,
    pub(super) op_abc: CutensorOperator,
}

pub(super) fn plan_elementwise_trinary(
    ctx: &mut CudaContext,
    data_type: CutensorDataType,
    compute: cutensorComputeDescriptor_t,
    spec: TrinaryPlanSpec<'_>,
) -> Result<NativeCutensorPlan> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, spec.shape_a, spec.strides_a, data_type)?;
    let desc_b = create_tensor_desc(handle, vtable, spec.shape_b, spec.strides_b, data_type)?;
    let desc_c = create_tensor_desc(handle, vtable, spec.shape_c, spec.strides_c, data_type)?;
    let desc_d = create_tensor_desc(handle, vtable, spec.shape_d, spec.strides_d, data_type)?;

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_elementwise_trinary)(
            handle,
            &mut op_raw,
            desc_a.raw,
            spec.modes_a.as_ptr(),
            spec.op_a,
            desc_b.raw,
            spec.modes_b.as_ptr(),
            spec.op_b,
            desc_c.raw,
            spec.modes_c.as_ptr(),
            spec.op_c,
            desc_d.raw,
            spec.modes_c.as_ptr(),
            spec.op_ab,
            spec.op_abc,
            compute,
        )
    };
    check_status(status, "cutensorCreateElementwiseTrinary")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };
    build_cutensor_plan(handle, vtable, &op_desc)
}
