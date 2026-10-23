//! Explicit host transfers on a CUDA runtime.
//!
//! Every entry point runs on the runtime's own stream under its submission lock: one
//! admission at a time, ordered with the engine that owns the runtime.

use std::sync::Arc;

use tenferro_tensor::{
    Complex32, Complex64, DType, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    StorageBuffer, Tensor, TensorScalar, TypedTensor,
};

use super::runtime::CudaRuntime;
use super::storage::CudaStorage;

/// The typed tensor behind `tensor`, or a typed refusal.
fn typed<T: TensorScalar>(tensor: &Tensor) -> crate::Result<&TypedTensor<T>> {
    tensor
        .as_typed::<T>()
        .ok_or_else(|| crate::Error::runtime_state("cuda transfer", "expected a preset scalar"))
}

pub(crate) fn placement(runtime: &CudaRuntime) -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: runtime.device_ordinal(),
        }),
        cpu_affinity: None,
    }
}

/// One uploaded device allocation per dtype, wrapped as a tensor.
fn upload_typed<T: TensorScalar + Send + Sync + 'static>(
    runtime: &CudaRuntime,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host = typed.host_data()?;
    let bytes: &[u8] = unsafe {
        // SAFETY: a `TensorScalar` payload is plain data with no padding-sensitive
        // invariant, and the slice borrows `host` for the call.
        std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), std::mem::size_of_val(host))
    };
    let buffer = {
        let mut guard = runtime.submit();
        guard
            .upload(bytes)
            .map_err(|error| crate::Error::runtime_state_source("cuda upload", error))?
    };
    let (domain, id) = super::storage::fresh_identity(runtime);
    let storage = CudaStorage {
        bytes: bytes.len(),
        buffer,
        device: runtime.device_ordinal(),
        domain,
        id,
    };
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(storage)),
        placement(runtime),
    )
}

/// Upload a host tensor into a fresh device allocation of `runtime`.
///
/// # Errors
///
/// [`CudaDeviceError::Unavailable`] when the source is not host-resident, the dtype
/// is not a preset scalar, or the runtime cannot allocate or enqueue the copy.
#[doc(hidden)]
pub fn upload_tensor(runtime: &CudaRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    match tensor.dtype() {
        DType::F64 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<f64>(tensor)?,
        )?)),
        DType::F32 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<f32>(tensor)?,
        )?)),
        DType::I32 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<i32>(tensor)?,
        )?)),
        DType::I64 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<i64>(tensor)?,
        )?)),
        DType::C32 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<Complex32>(tensor)?,
        )?)),
        DType::C64 => Ok(Tensor::from_typed(upload_typed(
            runtime,
            typed::<Complex64>(tensor)?,
        )?)),
        DType::Bool | DType::External(_) => Err(crate::Error::unsupported(
            "cuda transfer",
            format!("the transfer route does not serve {:?}", tensor.dtype()),
        )),
    }
}

/// Download a device tensor into host storage.
///
/// # Errors
///
/// [`CudaDeviceError::Unavailable`] when the source is not a CUDA allocation of this
/// runtime, the dtype is not a preset scalar, or the copy cannot be completed.
#[doc(hidden)]
pub fn download_tensor(runtime: &CudaRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    fn take<T: TensorScalar + Send + Sync + 'static>(
        runtime: &CudaRuntime,
        tensor: &Tensor,
    ) -> crate::Result<TypedTensor<T>> {
        let typed = typed::<T>(tensor)?;
        let storage = match typed.buffer() {
            StorageBuffer::Host(_) => {
                return Err(crate::Error::runtime_state(
                    "cuda download",
                    "expected a CUDA allocation",
                ))
            }
            StorageBuffer::Backend(buffer) => buffer
                .as_ref()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| {
                    crate::Error::runtime_state(
                        "cuda download",
                        "expected a CUDA allocation of this backend",
                    )
                })?,
        };
        if storage.domain() != runtime.allocation_domain_id() {
            return Err(crate::Error::runtime_state(
                "cuda download",
                "expected a CUDA allocation of this runtime",
            ));
        }
        let bytes = {
            let mut guard = runtime.submit();
            let mut pending = guard
                .download(&storage.buffer)
                .map_err(|error| crate::Error::runtime_state_source("cuda download", error))?;
            pending
                .wait()
                .map_err(|error| crate::Error::runtime_state_source("cuda download", error))?
        };
        let values: Vec<T> = bytes
            .chunks_exact(std::mem::size_of::<T>().max(1))
            .map(|word| {
                // SAFETY: the chunk has exactly the element size and `T` is plain data.
                unsafe { std::ptr::read_unaligned(word.as_ptr().cast::<T>()) }
            })
            .collect();
        TypedTensor::from_buffer_col_major(
            typed.shape().to_vec(),
            StorageBuffer::Host(values),
            Placement::default(),
        )
    }
    match tensor.dtype() {
        DType::F64 => Ok(Tensor::from_typed(take::<f64>(runtime, tensor)?)),
        DType::F32 => Ok(Tensor::from_typed(take::<f32>(runtime, tensor)?)),
        DType::I32 => Ok(Tensor::from_typed(take::<i32>(runtime, tensor)?)),
        DType::I64 => Ok(Tensor::from_typed(take::<i64>(runtime, tensor)?)),
        DType::C32 => Ok(Tensor::from_typed(take::<Complex32>(runtime, tensor)?)),
        DType::C64 => Ok(Tensor::from_typed(take::<Complex64>(runtime, tensor)?)),
        DType::Bool | DType::External(_) => Err(crate::Error::unsupported(
            "cuda transfer",
            format!("the transfer route does not serve {:?}", tensor.dtype()),
        )),
    }
}

/// Submit an explicit host-to-device copy into an existing CUDA allocation.
///
/// The copy runs on the runtime's own stream, writes straight from the pinned source
/// into the destination allocation, and returns a handle whose completion the caller
/// can poll. A submission refused before admission returns the package unchanged.
///
/// # Errors
///
/// [`TransferSubmitError::Rejected`] when the destination is not a CUDA allocation of
/// this runtime or the source length differs, and
/// [`TransferSubmitError::Runtime`] when the runtime cannot admit the copy.
#[doc(hidden)]
pub fn upload_pending(
    runtime: &CudaRuntime,
    source: super::pinned::PinnedHostBuffer,
    destination: Tensor,
) -> Result<super::pending::PendingUpload, super::pending::TransferSubmitError> {
    use super::pending::{PendingUpload, RejectedTransfer, TransferSubmitError};

    let Some(storage) = super::pending::cuda_storage(&destination) else {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer: source,
            value: destination,
            reason: "the destination is not a CUDA allocation",
        })));
    };
    if storage.domain() != runtime.allocation_domain_id() {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer: source,
            value: destination,
            reason: "the destination belongs to another runtime",
        })));
    }
    if source.len() != storage.bytes() {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer: source,
            value: destination,
            reason: "the pinned source length differs from the destination allocation",
        })));
    }

    let completion = {
        let mut guard = runtime.submit();
        let host = source.as_slice().as_ptr();
        let bytes = source.len();
        let stream_result = unsafe {
            // SAFETY: the device pointer addresses `bytes` bytes of this runtime's
            // allocation and the host pointer addresses `bytes` pinned bytes that the
            // package keeps alive until the copy's completion is observed.
            guard.with_raw(
                vec![t4a_cuda_runtime::RawResource::Buffer(&storage.buffer)],
                |raw| {
                    let device = raw.device_ptr(&storage.buffer).map_err(|error| {
                        crate::Error::runtime_state_source("cuda transfer", error)
                    })?;
                    let stream = raw.stream() as usize as *mut cudarc::driver::sys::CUstream_st;
                    let status = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        device,
                        host.cast(),
                        bytes,
                        stream,
                    );
                    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(crate::Error::runtime_state(
                            "cuda upload",
                            format!("cuMemcpyHtoDAsync_v2 failed with {status:?}"),
                        ));
                    }
                    Ok::<(), crate::Error>(())
                },
            )
        };
        stream_result
            .map_err(|error| TransferSubmitError::Runtime(Box::new(error)))?
            .map_err(|error| TransferSubmitError::Runtime(Box::new(error)))?;
        guard
            .checkpoint()
            .map_err(|error| TransferSubmitError::Runtime(Box::new(error)))?
    };
    Ok(PendingUpload::new(
        source,
        destination,
        completion,
        Arc::clone(&runtime.inner),
    ))
}

/// Submit an explicit device-to-host copy of `value` into a pinned buffer.
///
/// # Errors
///
/// [`TransferSubmitError::Rejected`] when the value is not a CUDA allocation of this
/// runtime or the buffer is shorter than the payload, and
/// [`TransferSubmitError::Runtime`] when the runtime cannot admit the copy.
#[doc(hidden)]
pub fn download_pending(
    runtime: &CudaRuntime,
    value: Tensor,
    mut buffer: super::pinned::PinnedHostBuffer,
) -> Result<super::pending::PendingDownload, super::pending::TransferSubmitError> {
    use super::pending::{PendingDownload, RejectedTransfer, TransferSubmitError};

    let Some(storage) = super::pending::cuda_storage(&value) else {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer,
            value,
            reason: "the value is not a CUDA allocation",
        })));
    };
    if storage.domain() != runtime.allocation_domain_id() {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer,
            value,
            reason: "the value belongs to another runtime",
        })));
    }
    if buffer.len() != storage.bytes() {
        return Err(TransferSubmitError::Rejected(Box::new(RejectedTransfer {
            buffer,
            value,
            reason: "the pinned buffer is shorter than the payload",
        })));
    }

    let completion = submit_download(runtime, storage, &mut buffer)
        .map_err(|error| TransferSubmitError::Runtime(Box::new(error)))?;
    let _ = ();
    Ok(PendingDownload::new(
        value,
        buffer,
        completion,
        Arc::clone(&runtime.inner),
    ))
}

/// Submit the device-to-host copy for `storage` into `buffer` and record its
/// completion, without taking ownership of a value tensor.
pub(crate) fn submit_download(
    runtime: &CudaRuntime,
    storage: &super::storage::CudaStorage,
    buffer: &mut super::pinned::PinnedHostBuffer,
) -> crate::Result<t4a_cuda_runtime::Completion> {
    {
        let mut guard = runtime.submit();
        // The copy writes through the mutable borrow: a pointer derived from a shared view
        // would not authorize the device write.
        let host = buffer.as_mut_slice().as_mut_ptr();
        let bytes = storage.bytes();
        let stream_result = unsafe {
            // SAFETY: the device pointer addresses `bytes` bytes of this runtime's
            // allocation and the host pointer addresses at least `bytes` pinned bytes
            // that the package keeps alive until the copy's completion is observed.
            guard.with_raw(
                vec![t4a_cuda_runtime::RawResource::Buffer(&storage.buffer)],
                |raw| {
                    let device = raw.device_ptr(&storage.buffer).map_err(|error| {
                        crate::Error::runtime_state_source("cuda transfer", error)
                    })?;
                    let stream = raw.stream() as usize as *mut cudarc::driver::sys::CUstream_st;
                    let status = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                        host.cast(),
                        device,
                        bytes,
                        stream,
                    );
                    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        return Err(crate::Error::runtime_state(
                            "cuda download",
                            format!("cuMemcpyDtoHAsync_v2 failed with {status:?}"),
                        ));
                    }
                    Ok::<(), crate::Error>(())
                },
            )
        };
        stream_result
            .map_err(|error| crate::Error::runtime_state_source("cuda download", error))??;
        {
            guard
                .checkpoint()
                .map_err(|error| crate::Error::runtime_state_source("cuda download", error))
        }
    }
}
