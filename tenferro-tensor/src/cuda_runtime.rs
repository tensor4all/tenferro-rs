use std::sync::Arc;

use cudarc::driver::{result as cuda_result, CudaContext};

use super::*;

fn cuda_error(operation: &str, err: impl std::fmt::Debug) -> Error {
    Error::DeviceError(format!("{operation} failed: {err:?}"))
}

fn init_cuda_context(device_id: usize) -> Result<Arc<CudaContext>> {
    std::panic::catch_unwind(|| CudaContext::new(device_id))
        .map_err(|_| Error::DeviceError("CUDA driver initialization panicked".into()))?
        .map_err(|err| cuda_error("CUDA device init", err))
}

fn copy_host_buffer_to_gpu<T: Scalar>(
    host_data: &[T],
    target: LogicalMemorySpace,
) -> Result<DataBuffer<T>> {
    let LogicalMemorySpace::GpuMemory { device_id } = target else {
        return Err(Error::DeviceError(format!(
            "unsupported CUDA allocation target: {target:?}"
        )));
    };

    let ctx = init_cuda_context(device_id)?;
    ctx.bind_to_thread()
        .map_err(|err| cuda_error("CUDA context bind", err))?;

    let num_bytes = std::mem::size_of_val(host_data);
    let device_ptr = unsafe { cuda_result::malloc_sync(num_bytes) }
        .map_err(|err| cuda_error("cudaMalloc", err))?;

    if !host_data.is_empty() {
        if let Err(err) = unsafe { cuda_result::memcpy_htod_sync(device_ptr, host_data) } {
            let _ = unsafe { cuda_result::free_sync(device_ptr) };
            return Err(cuda_error("cudaMemcpyHtoD", err));
        }
    }

    let release_ctx = Arc::clone(&ctx);
    let release = move || {
        let _ = release_ctx.bind_to_thread();
        let _ = unsafe { cuda_result::free_sync(device_ptr) };
    };

    Ok(DataBuffer {
        inner: Arc::new(BufferInner::Gpu {
            device_ptr: device_ptr as *mut T,
            len: host_data.len(),
            space: target,
            release: Some(Box::new(release)),
        }),
    })
}

fn copy_gpu_buffer_to_host<T: Scalar>(
    buffer: &DataBuffer<T>,
    source: LogicalMemorySpace,
) -> Result<Vec<T>> {
    let LogicalMemorySpace::GpuMemory { device_id } = source else {
        return Err(Error::DeviceError(format!(
            "unsupported CUDA transfer source: {source:?}"
        )));
    };

    let ctx = init_cuda_context(device_id)?;
    ctx.bind_to_thread()
        .map_err(|err| cuda_error("CUDA context bind", err))?;

    let mut host_data = vec![T::zero(); buffer.len()];
    if host_data.is_empty() {
        return Ok(host_data);
    }

    let device_ptr = buffer
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("tensor buffer is not resident on GPU".into()))?;

    unsafe {
        cuda_result::memcpy_dtoh_sync(&mut host_data, device_ptr as usize as _)
            .map_err(|err| cuda_error("cudaMemcpyDtoH", err))?;
    }

    Ok(host_data)
}

fn transferred_fw_grad<T: Scalar>(
    source: &Option<Box<Tensor<T>>>,
    target: LogicalMemorySpace,
) -> Result<Option<Box<Tensor<T>>>> {
    source
        .as_ref()
        .map(|fw_grad| fw_grad.to_memory_space_async(target).map(Box::new))
        .transpose()
}

fn rebuild_tensor_in_space<T: Scalar>(
    source: &Tensor<T>,
    buffer: DataBuffer<T>,
    target: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    Ok(Tensor {
        buffer,
        dims: source.dims.clone(),
        strides: source.strides.clone(),
        offset: source.offset,
        logical_memory_space: target,
        preferred_compute_device: source.preferred_compute_device,
        event: None,
        conjugated: source.conjugated,
        fw_grad: transferred_fw_grad(&source.fw_grad, target)?,
    })
}

pub(super) fn transfer_tensor<T: Scalar>(
    source: &Tensor<T>,
    target: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    match (source.logical_memory_space, target) {
        (LogicalMemorySpace::MainMemory, LogicalMemorySpace::GpuMemory { .. }) => {
            let host_data = source
                .buffer
                .as_slice()
                .ok_or_else(|| Error::DeviceError("CPU tensor is not host-accessible".into()))?;
            let buffer = copy_host_buffer_to_gpu(host_data, target)?;
            rebuild_tensor_in_space(source, buffer, target)
        }
        (LogicalMemorySpace::GpuMemory { .. }, LogicalMemorySpace::MainMemory) => {
            let host_data = copy_gpu_buffer_to_host(&source.buffer, source.logical_memory_space)?;
            rebuild_tensor_in_space(source, DataBuffer::from_vec(host_data), target)
        }
        (LogicalMemorySpace::GpuMemory { .. }, LogicalMemorySpace::GpuMemory { .. }) => Err(
            Error::DeviceError("GPU-to-GPU transfer not yet implemented".into()),
        ),
        (_, LogicalMemorySpace::PinnedMemory) => Err(Error::DeviceError(
            "pinned-memory transfer not yet implemented".into(),
        )),
        (_, LogicalMemorySpace::ManagedMemory) => Err(Error::DeviceError(
            "managed-memory transfer not yet implemented".into(),
        )),
        (_, LogicalMemorySpace::MainMemory) => Err(Error::DeviceError(
            "unsupported transfer source memory space".into(),
        )),
        (_, LogicalMemorySpace::GpuMemory { .. }) => Err(Error::DeviceError(
            "unsupported transfer source memory space".into(),
        )),
    }
}
