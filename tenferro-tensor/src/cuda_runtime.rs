use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::cuda::runtime::{
    self as device_cuda, ContiguousOrder, CudaBuffer, StridedCopySpec, ZeroTrailingByCountsSpec,
};
use tenferro_device::{Error, LogicalMemorySpace, Result};

use crate::{layout::compute_contiguous_strides, DataBuffer, MemoryOrder, Tensor};

fn gpu_runtime(space: LogicalMemorySpace) -> Result<Arc<device_cuda::CudaRuntime>> {
    let LogicalMemorySpace::GpuMemory { device_id } = space else {
        return Err(Error::DeviceError(format!(
            "unsupported CUDA memory space: {space:?}"
        )));
    };
    device_cuda::get_or_init(device_id)
}

fn buffer_from_cuda_allocation<T: Scalar>(
    allocation: CudaBuffer<T>,
    target: LogicalMemorySpace,
) -> DataBuffer<T> {
    let device_ptr = allocation.device_ptr();
    let len = allocation.len();
    unsafe { DataBuffer::from_gpu_parts(device_ptr, len, target, move || drop(allocation)) }
}

fn alloc_gpu_buffer<T: Scalar>(len: usize, target: LogicalMemorySpace) -> Result<DataBuffer<T>> {
    let runtime = gpu_runtime(target)?;
    let allocation = runtime.alloc::<T>(len)?;
    Ok(buffer_from_cuda_allocation(allocation, target))
}

fn copy_host_buffer_to_gpu<T: Scalar>(
    host_data: &[T],
    target: LogicalMemorySpace,
) -> Result<DataBuffer<T>> {
    let runtime = gpu_runtime(target)?;
    let allocation = runtime.alloc::<T>(host_data.len())?;
    if !host_data.is_empty() {
        unsafe {
            runtime.copy_htod_raw(host_data, allocation.device_ptr(), allocation.len())?;
        }
    }

    Ok(buffer_from_cuda_allocation(allocation, target))
}

fn copy_gpu_buffer_to_host<T: Scalar>(
    buffer: &DataBuffer<T>,
    source: LogicalMemorySpace,
) -> Result<Vec<T>> {
    let runtime = gpu_runtime(source)?;
    let device_ptr = buffer
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("tensor buffer is not resident on GPU".into()))?;

    unsafe { runtime.copy_dtoh_raw(device_ptr, buffer.len()) }
}

fn transferred_fw_grad<T: Scalar>(
    source: Option<&Tensor<T>>,
    target: LogicalMemorySpace,
) -> Result<Option<Box<Tensor<T>>>> {
    source
        .map(|fw_grad| fw_grad.to_memory_space_async(target).map(Box::new))
        .transpose()
}

fn rebuild_tensor_in_space<T: Scalar>(
    source: &Tensor<T>,
    buffer: DataBuffer<T>,
    target: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    Ok(Tensor::from_parts(
        buffer,
        Arc::from(source.dims()),
        Arc::from(source.strides()),
        source.offset(),
        target,
        source.preferred_compute_device(),
        None,
        source.is_conjugated(),
        transferred_fw_grad(source.fw_grad(), target)?,
    ))
}

fn contiguous_order(order: MemoryOrder) -> ContiguousOrder {
    match order {
        MemoryOrder::ColumnMajor => ContiguousOrder::ColumnMajor,
        MemoryOrder::RowMajor => ContiguousOrder::RowMajor,
    }
}

pub(super) fn contiguous_tensor<T: Scalar>(
    source: &Tensor<T>,
    order: MemoryOrder,
) -> Result<Tensor<T>> {
    let space = source.logical_memory_space();
    let runtime = gpu_runtime(space)?;
    let out_strides = compute_contiguous_strides(source.dims(), order);
    let output = Tensor::from_parts(
        alloc_gpu_buffer(source.len(), space)?,
        Arc::from(source.dims()),
        Arc::from(out_strides),
        0,
        space,
        source.preferred_compute_device(),
        None,
        source.is_conjugated(),
        None,
    );
    if source.is_empty() {
        return Ok(output);
    }

    let src_ptr = source
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("source tensor buffer is not on GPU".into()))?;
    let dst_ptr = output
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("output tensor buffer is not on GPU".into()))?
        as *mut T;
    let spec = StridedCopySpec::to_contiguous(
        source.dims(),
        source.strides(),
        source.offset(),
        contiguous_order(order),
    )?;

    unsafe {
        runtime.copy_strided_raw(src_ptr, dst_ptr, &spec)?;
    }

    Ok(output)
}

pub(super) fn zero_trailing_by_counts_tensor<T, R>(
    source: &Tensor<T>,
    keep_counts: &Tensor<R>,
    axis: usize,
    structural_rank: usize,
) -> Result<Tensor<T>>
where
    T: Scalar,
    R: Scalar + device_cuda::RuntimeKeepCountScalar,
{
    let space = source.logical_memory_space();
    let runtime = gpu_runtime(space)?;
    let out_strides = compute_contiguous_strides(source.dims(), MemoryOrder::ColumnMajor);
    let output = Tensor::from_parts(
        alloc_gpu_buffer(source.len(), space)?,
        Arc::from(source.dims()),
        Arc::from(out_strides),
        0,
        space,
        source.preferred_compute_device(),
        None,
        source.is_conjugated(),
        None,
    );
    if source.is_empty() {
        return Ok(output);
    }

    let src_ptr = source
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("source tensor buffer is not on GPU".into()))?;
    let dst_ptr = output
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("output tensor buffer is not on GPU".into()))?
        as *mut T;
    let keep_counts_ptr = keep_counts
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("keep_counts tensor buffer is not on GPU".into()))?;
    let spec = ZeroTrailingByCountsSpec::new(
        source.dims(),
        source.strides(),
        source.offset(),
        output.strides(),
        output.offset(),
        keep_counts.strides(),
        keep_counts.offset(),
        axis,
        structural_rank,
    )?;

    unsafe {
        runtime.zero_trailing_by_counts_raw(src_ptr, dst_ptr, keep_counts_ptr, &spec)?;
    }

    Ok(output)
}

pub(super) fn transfer_tensor<T: Scalar>(
    source: &Tensor<T>,
    target: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    match (source.logical_memory_space(), target) {
        (LogicalMemorySpace::MainMemory, LogicalMemorySpace::GpuMemory { .. }) => {
            let host_data = source
                .buffer()
                .as_slice()
                .ok_or_else(|| Error::DeviceError("CPU tensor is not host-accessible".into()))?;
            let buffer = copy_host_buffer_to_gpu(host_data, target)?;
            rebuild_tensor_in_space(source, buffer, target)
        }
        (LogicalMemorySpace::GpuMemory { .. }, LogicalMemorySpace::MainMemory) => {
            let host_data =
                copy_gpu_buffer_to_host(source.buffer(), source.logical_memory_space())?;
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
