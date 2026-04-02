use std::any::TypeId;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_device::cuda::runtime::{
    self as device_cuda, ContiguousOrder, CudaBuffer, RealBinaryOp, StridedCopySpec,
    StridedCopyTransform, TriangularHalf, TriangularMergeSpec, TriangularPartSpec,
    ZeroTrailingByCountsSpec,
};
use tenferro_device::{Error, LogicalMemorySpace, Result};

use crate::layout::compute_contiguous_strides;
use crate::{DataBuffer, MemoryOrder, Tensor};

fn gpu_runtime(space: LogicalMemorySpace) -> Result<Arc<device_cuda::CudaRuntime>> {
    let LogicalMemorySpace::GpuMemory { device_id } = space else {
        return Err(Error::DeviceError(format!(
            "unsupported CUDA memory space: {space:?}"
        )));
    };
    device_cuda::get_or_init(device_id)
}

fn supports_conj_strided_copy<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<Complex32>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
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

/// Allocate a zero-filled GPU buffer.
///
/// Uses host-to-device copy from a zeroed host vector as a portable fallback.
pub(crate) fn alloc_zeros_gpu<T: Scalar>(
    len: usize,
    target: LogicalMemorySpace,
) -> Result<DataBuffer<T>> {
    if len == 0 {
        return alloc_gpu_buffer(len, target);
    }
    let zeros = vec![T::zero(); len];
    copy_host_buffer_to_gpu(&zeros, target)
}

/// Allocate an uninitialized GPU buffer.
pub(crate) fn alloc_gpu_uninit<T: Scalar>(
    len: usize,
    target: LogicalMemorySpace,
) -> Result<DataBuffer<T>> {
    alloc_gpu_buffer(len, target)
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

pub(super) fn materialize_logical_contiguous_tensor<T>(
    source: &Tensor<T>,
    order: MemoryOrder,
) -> Result<Tensor<T>>
where
    T: Scalar + 'static,
{
    let space = source.logical_memory_space();
    let runtime = gpu_runtime(space)?;
    let out_strides = compute_contiguous_strides(source.dims(), order);
    let output = Tensor::from_parts(
        alloc_gpu_buffer(source.len(), space)?,
        Arc::from(source.dims()),
        Arc::from(out_strides),
        0,
        space,
        None,
        None,
        false,
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
        if source.is_conjugated() && supports_conj_strided_copy::<T>() {
            runtime.copy_strided_raw_with_transform(
                src_ptr,
                dst_ptr,
                &spec,
                StridedCopyTransform::Conj,
            )?;
        } else {
            runtime.copy_strided_raw(src_ptr, dst_ptr, &spec)?;
        }
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

pub(super) fn triangular_part_tensor<T: Scalar>(
    source: &Tensor<T>,
    diagonal: isize,
    lower: bool,
) -> Result<Tensor<T>> {
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
    let spec = TriangularPartSpec::new(
        source.dims(),
        source.strides(),
        source.offset(),
        output.strides(),
        output.offset(),
        diagonal,
        if lower {
            TriangularHalf::Lower
        } else {
            TriangularHalf::Upper
        },
    )?;

    unsafe {
        runtime.triangular_part_raw(src_ptr, dst_ptr, &spec)?;
    }

    Ok(output)
}

pub(super) fn merge_strict_lower_and_upper_tensor<T: Scalar>(
    lower: &Tensor<T>,
    upper: &Tensor<T>,
) -> Result<Tensor<T>> {
    let space = lower.logical_memory_space();
    let runtime = gpu_runtime(space)?;
    let batch_dims = &lower.dims()[2..];
    let output_dims: Vec<usize> = std::iter::once(lower.dims()[0])
        .chain(std::iter::once(upper.dims()[1]))
        .chain(batch_dims.iter().copied())
        .collect();
    let out_strides = compute_contiguous_strides(&output_dims, MemoryOrder::ColumnMajor);
    let output = Tensor::from_parts(
        alloc_gpu_buffer(output_dims.iter().product(), space)?,
        Arc::from(output_dims),
        Arc::from(out_strides),
        0,
        space,
        lower.preferred_compute_device(),
        None,
        lower.is_conjugated(),
        None,
    );
    if output.is_empty() {
        return Ok(output);
    }

    let lower_ptr = lower
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("lower tensor buffer is not on GPU".into()))?;
    let upper_ptr = upper
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("upper tensor buffer is not on GPU".into()))?;
    let dst_ptr = output
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("output tensor buffer is not on GPU".into()))?
        as *mut T;
    let spec = TriangularMergeSpec::new(
        output.dims(),
        lower.strides(),
        lower.offset(),
        upper.strides(),
        upper.offset(),
        output.strides(),
        output.offset(),
    )?;

    unsafe {
        runtime.triangular_merge_raw(lower_ptr, upper_ptr, dst_ptr, &spec)?;
    }

    Ok(output)
}

/// Element-wise addition of two GPU tensors: `result = a + b`.
///
/// Supports strided inputs; produces a contiguous column-major output.
/// Uses `TypeId` dispatch to pick the correct f32/f64 binary kernel.
/// Complex types are not yet supported and will return an error.
pub(super) fn add_strided_tensor<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
    let space = a.logical_memory_space();
    let runtime = gpu_runtime(space)?;
    let dims = a.dims();
    let out_strides = compute_contiguous_strides(dims, crate::MemoryOrder::ColumnMajor);
    let output_buf = alloc_gpu_buffer::<T>(a.len(), space)?;
    if a.is_empty() {
        return Ok(Tensor::from_parts(
            output_buf,
            Arc::from(dims),
            Arc::from(out_strides),
            0,
            space,
            a.preferred_compute_device(),
            None,
            false,
            None,
        ));
    }

    let a_ptr = a
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("add_strided_tensor: lhs not on GPU".into()))?;
    let b_ptr = b
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("add_strided_tensor: rhs not on GPU".into()))?;
    let dst_ptr = output_buf
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("add_strided_tensor: output not on GPU".into()))?
        as *mut T;

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            runtime.pointwise_binary_real_f32_raw(
                RealBinaryOp::Add,
                1.0_f32,
                a_ptr as *const f32,
                dims,
                a.strides(),
                a.offset(),
                b_ptr as *const f32,
                b.strides(),
                b.offset(),
                0.0_f32,
                dst_ptr as *mut f32,
                &out_strides,
                0,
            )?;
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            runtime.pointwise_binary_real_f64_raw(
                RealBinaryOp::Add,
                1.0_f64,
                a_ptr as *const f64,
                dims,
                a.strides(),
                a.offset(),
                b_ptr as *const f64,
                b.strides(),
                b.offset(),
                0.0_f64,
                dst_ptr as *mut f64,
                &out_strides,
                0,
            )?;
        }
    } else {
        return Err(Error::DeviceError(format!(
            "add_strided_tensor: unsupported scalar type {} for GPU addition",
            std::any::type_name::<T>()
        )));
    }

    Ok(Tensor::from_parts(
        output_buf,
        Arc::from(dims),
        Arc::from(out_strides),
        0,
        space,
        a.preferred_compute_device(),
        None,
        false,
        None,
    ))
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
