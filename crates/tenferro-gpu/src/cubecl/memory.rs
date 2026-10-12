//! Host-to-device and device-to-host transfers via CubeCL-managed allocations.

use cubecl::client::Client;
use cubecl::prelude::CubeElement;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::DType;

use super::dispatch;
use crate::cubecl::runtime::{CudaRuntime, PINNED_SCALAR_BYTES};
use crate::types::{
    CubeclBuffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, StorageBuffer,
    Tensor, TensorScalar, TypedTensor,
};
use cubecl::frontend::Scalar;

/// Upload a host tensor into a CubeCL-managed GPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{cuda::upload_tensor, cuda::CudaRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _upload: fn(&CudaRuntime, &Tensor) -> Result<Tensor> = upload_tensor;
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the source is backend-resident
/// or belongs to another placement, [`crate::Error::Unsupported`] for a dtype
/// unavailable in CubeCL, or [`crate::Error::BackendSource`] on allocation.
pub fn upload_tensor(rt: &CudaRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor.dtype() {
        DType::F64 => upload_typed::<f64>(rt, client, gpu_typed::<f64>("upload_tensor", tensor)?)
            .map(Tensor::from_typed::<f64>),
        DType::F32 => upload_typed::<f32>(rt, client, gpu_typed::<f32>("upload_tensor", tensor)?)
            .map(Tensor::from_typed::<f32>),
        DType::I32 => upload_typed::<i32>(rt, client, gpu_typed::<i32>("upload_tensor", tensor)?)
            .map(Tensor::from_typed::<i32>),
        DType::I64 => upload_typed::<i64>(rt, client, gpu_typed::<i64>("upload_tensor", tensor)?)
            .map(Tensor::from_typed::<i64>),
        DType::Bool => upload_bool(rt, client, gpu_typed::<bool>("upload_tensor", tensor)?)
            .map(Tensor::from_typed::<bool>),
        DType::C64 => {
            upload_typed::<Complex64>(rt, client, gpu_typed::<Complex64>("upload_tensor", tensor)?)
                .map(Tensor::from_typed::<tenferro_tensor::Complex64>)
        }
        DType::C32 => {
            upload_typed::<Complex32>(rt, client, gpu_typed::<Complex32>("upload_tensor", tensor)?)
                .map(Tensor::from_typed::<tenferro_tensor::Complex32>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "upload_tensor",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

/// The typed tensor behind `tensor`, or a typed refusal.
///
/// Callers reach this from a match on the tensor's dtype, so `None` means the tag
/// table and the runtime dtype disagree rather than a caller mistake.
fn gpu_typed<'a, T: TensorScalar>(
    op: &'static str,
    tensor: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the GPU memory path requires a preset scalar")
    })
}

/// Download a CubeCL-managed GPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{cuda::download_tensor, cuda::CudaRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _download: fn(&CudaRuntime, &Tensor) -> Result<Tensor> = download_tensor;
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] for a host-backed or foreign tensor,
/// [`crate::Error::BackendSource`] when synchronization/readback fails, or a
/// typed validation error when device data cannot be decoded.
pub fn download_tensor(rt: &CudaRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    ensure_tensor_resident_on_runtime(rt, tensor, "download")?;
    match tensor.dtype() {
        DType::F64 => download_typed::<f64>(rt, gpu_typed::<f64>("download_tensor", tensor)?)
            .map(Tensor::from_typed::<f64>),
        DType::F32 => download_typed::<f32>(rt, gpu_typed::<f32>("download_tensor", tensor)?)
            .map(Tensor::from_typed::<f32>),
        DType::I32 => download_typed::<i32>(rt, gpu_typed::<i32>("download_tensor", tensor)?)
            .map(Tensor::from_typed::<i32>),
        DType::I64 => download_typed::<i64>(rt, gpu_typed::<i64>("download_tensor", tensor)?)
            .map(Tensor::from_typed::<i64>),
        DType::Bool => download_bool(rt, gpu_typed::<bool>("download_tensor", tensor)?)
            .map(Tensor::from_typed::<bool>),
        DType::C64 => {
            download_typed::<Complex64>(rt, gpu_typed::<Complex64>("download_tensor", tensor)?)
                .map(Tensor::from_typed::<tenferro_tensor::Complex64>)
        }
        DType::C32 => {
            download_typed::<Complex32>(rt, gpu_typed::<Complex32>("download_tensor", tensor)?)
                .map(Tensor::from_typed::<tenferro_tensor::Complex32>)
        }
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "download_tensor",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

fn upload_typed<T: CubeElement + TensorScalar + Scalar + Clone + Send + Sync + 'static>(
    rt: &CudaRuntime,
    client: &Client,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = match typed.buffer() {
        StorageBuffer::Host(data) => data,
        StorageBuffer::Backend(buffer) => {
            return Err(crate::Error::runtime_state(
                "upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let byte_len = T::as_bytes(host_data).len();
    // A payload larger than the staging chunk goes through the runtime's pinned staging lease: the
    // device then reads pinned memory instead of the pageable transport, which is where the measured
    // 17x lives for these sizes (#1945 U4 item 4). Chunked uploads also wait for their own chunks,
    // because a reused staging buffer and a borrowed host source cannot stay safe behind an early
    // return.
    // The routing cutoff is the staging chunk: above it the pageable blocking route loses its
    // bandwidth (17x at 256 MiB) and this route pays, measured at 6.9x for a 256 MiB upload. Only
    // uploads are routed through it - a chunked download does not pay on this hardware.
    let handle = if byte_len > super::runtime::STAGING_CHUNK_BYTES {
        let handle = client.empty(byte_len);
        super::transfer::upload_tensor_chunked(
            rt,
            &handle,
            byte_len,
            T::as_bytes(host_data),
            "upload",
        )?;
        handle
    } else {
        // The host buffer is borrowed, so CubeCL stages exactly one copy of it: the device write runs
        // after this call returns (#2009).
        client.create_from_slice(T::as_bytes(host_data))
    };
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(CubeclBuffer::new(
            handle,
            byte_len,
            rt.device_ordinal(),
            rt.allocation_domain_id(),
        ))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: rt.device_ordinal(),
            }),
            cpu_affinity: None,
        },
    )
}

/// Read a compact tensor of at most `PINNED_SCALAR_BYTES` through the runtime's
/// pinned staging slot.
fn download_scalar<T: CubeElement + TensorScalar + Scalar + Clone + 'static>(
    rt: &CudaRuntime,
    typed: &TypedTensor<T>,
    byte_len: usize,
) -> crate::Result<Vec<T>> {
    let ptr = super::gemm::typed_device_ptr(rt, typed, "download")?;
    let retained = dispatch::cubecl_buffer(typed, "download")?.handle().clone();
    let mut bytes = vec![0_u8; byte_len];
    rt.download_scalar_bytes(ptr as u64, &mut bytes, "download", retained)?;
    Ok(T::from_bytes(&bytes).to_vec())
}

fn download_typed<T: CubeElement + TensorScalar + Scalar + Clone + 'static>(
    rt: &CudaRuntime,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let handle = match typed.buffer() {
        StorageBuffer::Host(_) => {
            return Err(crate::Error::runtime_state(
                "download",
                "expected CubeCL buffer",
            ));
        }
        StorageBuffer::Backend(buffer) => cubecl_handle_from_backend(buffer.as_ref(), "download")?,
    };

    if typed.n_elements() == 0 {
        return TypedTensor::from_buffer_col_major(
            typed.shape().to_vec(),
            StorageBuffer::Host(Vec::new()),
            Placement::default(),
        );
    }

    // Small-payload fast path. A Krylov loop reads back one reduction result
    // per iteration. CubeCL's `read_one` already stages through pinned memory,
    // so this is not a pageable-to-pinned conversion; what it avoids is
    // CubeCL's staging-reservation machinery and one of two barriers. The
    // general path below synchronizes the stream and then lets `read_one` wait
    // on its own fence after the copy, while this issues the copy and waits
    // once. Neither path synchronizes the device. The gate is a byte length, so
    // a short vector takes it too, not only a scalar.
    let byte_len = typed
        .n_elements()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| {
            crate::Error::invalid_argument("download", "shape", "byte length overflows")
        })?;
    if byte_len <= PINNED_SCALAR_BYTES {
        let data = download_scalar(rt, typed, byte_len)?;
        return TypedTensor::from_buffer_col_major(
            typed.shape().to_vec(),
            StorageBuffer::Host(data),
            Placement::default(),
        );
    }

    let data = if handle.size_in_used() == byte_len as u64 {
        // Not chunked: staging a download through pinned memory measured no better than this route,
        // because reading this machine's pinned allocations from the host runs at about 2 GB/s
        // (`docs/design/large-transfer-staging-1945-u4.md` §3). The upload direction is where the
        // chunked route pays, since host writes into pinned memory are an order of magnitude faster.
        download_owned_vec::<T>(rt, handle, typed.n_elements(), byte_len, "download")?
    } else {
        // A handle that does not span exactly the tensor's elements keeps the
        // whole-allocation read and its element-count validation below.
        rt.synchronize()?;
        let bytes = rt
            .client()
            .read_one(handle)
            .map_err(|err| crate::Error::backend_source("download", err))?;
        T::from_bytes(&bytes).to_vec()
    };
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Host(data),
        Placement::default(),
    )
}

/// Download a whole allocation of `len` elements into a freshly owned `Vec<T>`.
///
/// The device copies straight into the vector the host tensor will own, so the
/// payload crosses host memory once instead of being staged by CubeCL and then
/// copied again (#2009). The vector is allocated with its final element type,
/// so its alignment is `T`'s, and it is only exposed after the copy completed.
pub(super) fn download_owned_vec<T: CubeElement>(
    rt: &CudaRuntime,
    handle: cubecl_runtime::server::Handle,
    len: usize,
    byte_len: usize,
    op: &'static str,
) -> crate::Result<Vec<T>> {
    let mut data = Vec::<T>::with_capacity(len);
    // SAFETY: the spare capacity of `data` holds `len` elements, i.e.
    // `byte_len` bytes, and `data` outlives the call. On error the device may
    // still write into it, so it is leaked below rather than freed.
    let copied = unsafe { rt.download_into_host(handle, data.as_mut_ptr().cast(), byte_len, op) };
    match copied {
        Ok(()) => {
            // SAFETY: the completed copy initialized all `len` elements, and
            // `T: CubeElement` is `Pod`, so any byte pattern is a valid `T`.
            unsafe { data.set_len(len) };
            Ok(data)
        }
        Err(err) => {
            std::mem::forget(data);
            Err(err)
        }
    }
}

fn upload_bool(
    rt: &CudaRuntime,
    client: &Client,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    let host_data = match typed.buffer() {
        StorageBuffer::Host(data) => data,
        StorageBuffer::Backend(buffer) => {
            return Err(crate::Error::runtime_state(
                "upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let bytes: Vec<u8> = host_data.iter().map(|&value| u8::from(value)).collect();
    let byte_len = bytes.len();
    // The converted bytes are owned: hand them to CubeCL without a staging copy.
    let handle = client.create(cubecl::bytes::Bytes::from_elems(bytes));
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(CubeclBuffer::new(
            handle,
            byte_len,
            rt.device_ordinal(),
            rt.allocation_domain_id(),
        ))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: rt.device_ordinal(),
            }),
            cpu_affinity: None,
        },
    )
}

fn download_bool(rt: &CudaRuntime, typed: &TypedTensor<bool>) -> crate::Result<TypedTensor<bool>> {
    let handle = match typed.buffer() {
        StorageBuffer::Host(_) => {
            return Err(crate::Error::runtime_state(
                "download",
                "expected CubeCL buffer",
            ));
        }
        StorageBuffer::Backend(buffer) => cubecl_handle_from_backend(buffer.as_ref(), "download")?,
    };

    if typed.n_elements() == 0 {
        return TypedTensor::from_vec_col_major(typed.shape().to_vec(), Vec::new());
    }

    rt.synchronize()?;
    let bytes = rt
        .client()
        .read_one(handle)
        .map_err(|err| crate::Error::backend_source("download", err))?;
    let data = bytes.iter().map(|&byte| byte != 0).collect();
    TypedTensor::from_vec_col_major(typed.shape().to_vec(), data)
}

fn ensure_tensor_resident_on_runtime(
    rt: &CudaRuntime,
    tensor: &Tensor,
    op: &'static str,
) -> crate::Result<()> {
    match tensor.dtype() {
        DType::F64 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<f64>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::F32 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<f32>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::I32 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<i32>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::I64 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<i64>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::Bool => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<bool>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::C64 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<Complex64>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        DType::C32 => dispatch::ensure_resident_on_runtime(
            rt,
            gpu_typed::<Complex32>("ensure_tensor_resident_on_runtime", tensor)?,
            op,
        ),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "ensure_tensor_resident_on_runtime",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

fn cubecl_handle_from_backend<T: 'static>(
    buffer: &dyn crate::BackendStorage<T>,
    op: &'static str,
) -> crate::Result<cubecl_runtime::server::Handle> {
    buffer
        .as_any()
        .downcast_ref::<CubeclBuffer>()
        .map(|buffer| buffer.handle().clone())
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "expected CubeCL buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            )
        })
}
