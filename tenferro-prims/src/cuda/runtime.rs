use std::ffi::c_void;
use std::ptr;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::SemiringCoreDescriptor;

use super::execution::{execute_plan, plan_core_descriptor};

pub(super) struct WorkspaceBuffer {
    pub(super) ptr: *mut c_void,
}

impl Drop for WorkspaceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudarc::runtime::result::free_sync(self.ptr) };
        }
    }
}

pub(super) fn allocate_workspace(size: u64) -> Result<Option<WorkspaceBuffer>> {
    if size == 0 {
        return Ok(None);
    }
    let ptr = unsafe { cudarc::runtime::result::malloc_sync(size as usize) }
        .map_err(|e| Error::DeviceError(format!("cudaMalloc workspace failed: {e:?}")))?;
    Ok(Some(WorkspaceBuffer { ptr }))
}

pub(super) fn validate_runtime_shape(
    name: &str,
    actual: &[usize],
    expected: &[usize],
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "{name} shape mismatch: expected {expected:?}, got {actual:?}"
        )))
    }
}

pub(super) fn ensure_device_tensor<S: Scalar>(
    name: &str,
    tensor: &Tensor<S>,
    device_id: usize,
) -> Result<()> {
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

pub(super) fn tensor_device_ptr_with_offset<S: Scalar>(
    name: &str,
    tensor: &Tensor<S>,
) -> Result<*mut c_void> {
    let ptr = tensor
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError(format!("{name} not on GPU")))? as *mut S;
    Ok(unsafe { ptr.offset(tensor.offset()) } as *mut c_void)
}

pub(super) fn tensor_device_addr_with_offset<S: Scalar>(
    name: &str,
    tensor: &Tensor<S>,
) -> Result<u64> {
    Ok(tensor_device_ptr_with_offset(name, tensor)? as u64)
}

pub(super) fn new_gpu_tensor<S: Scalar>(dims: &[usize], device_id: usize) -> Result<Tensor<S>> {
    Tensor::<S>::zeros(
        dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
}

pub(super) fn null_stream() -> *mut c_void {
    ptr::null_mut()
}

pub(super) fn make_contiguous_on_cuda<S: Scalar>(
    ctx: &mut super::CudaContext,
    input: &Tensor<S>,
) -> Result<Tensor<S>> {
    if input.is_col_major_contiguous() && input.offset() == 0 {
        return Ok(input.clone());
    }
    let plan = plan_core_descriptor::<S>(
        ctx,
        &SemiringCoreDescriptor::MakeContiguous,
        &[input.dims(), input.dims()],
    )?;
    let mut output = new_gpu_tensor::<S>(input.dims(), ctx.device_id)?;
    execute_plan(ctx, &plan, S::one(), &[input], S::zero(), &mut output)?;
    Ok(output)
}

pub(super) fn prepare_custom_output<S: Scalar>(
    ctx: &mut super::CudaContext,
    output: &Tensor<S>,
) -> Result<(Tensor<S>, bool)> {
    if output.is_col_major_contiguous() && output.offset() == 0 {
        return Ok((output.clone(), false));
    }
    Ok((make_contiguous_on_cuda(ctx, output)?, true))
}

pub(super) fn write_custom_output_back<S: Scalar>(
    ctx: &mut super::CudaContext,
    contiguous_output: &Tensor<S>,
    output: &mut Tensor<S>,
) -> Result<()> {
    if output.is_col_major_contiguous() && output.offset() == 0 {
        return Ok(());
    }
    let plan = plan_core_descriptor::<S>(
        ctx,
        &SemiringCoreDescriptor::MakeContiguous,
        &[contiguous_output.dims(), output.dims()],
    )?;
    execute_plan(
        ctx,
        &plan,
        S::one(),
        &[contiguous_output],
        S::zero(),
        output,
    )
}
