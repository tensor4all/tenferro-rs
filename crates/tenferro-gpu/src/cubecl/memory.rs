//! Host-to-device and device-to-host transfers via CubeCL-managed allocations.

use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};

use super::dispatch;
use crate::cubecl::runtime::CudaRuntime;
use crate::types::{
    CubeclBuffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, StorageBuffer,
    Tensor, TypedTensor,
};

/// Upload a host tensor into a CubeCL-managed GPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{upload_tensor, CudaRuntime};
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
    match tensor {
        Tensor::F64(t) => upload_typed::<f64>(rt, client, t).map(Tensor::F64),
        Tensor::F32(t) => upload_typed::<f32>(rt, client, t).map(Tensor::F32),
        Tensor::I32(t) => upload_typed::<i32>(rt, client, t).map(Tensor::I32),
        Tensor::I64(t) => upload_typed::<i64>(rt, client, t).map(Tensor::I64),
        Tensor::Bool(t) => upload_bool(rt, client, t).map(Tensor::Bool),
        Tensor::C64(t) => upload_typed::<Complex64>(rt, client, t).map(Tensor::C64),
        Tensor::C32(t) => upload_typed::<Complex32>(rt, client, t).map(Tensor::C32),
    }
}

/// Download a CubeCL-managed GPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{download_tensor, CudaRuntime};
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
    match tensor {
        Tensor::F64(t) => download_typed::<f64>(rt, t).map(Tensor::F64),
        Tensor::F32(t) => download_typed::<f32>(rt, t).map(Tensor::F32),
        Tensor::I32(t) => download_typed::<i32>(rt, t).map(Tensor::I32),
        Tensor::I64(t) => download_typed::<i64>(rt, t).map(Tensor::I64),
        Tensor::Bool(t) => download_bool(rt, t).map(Tensor::Bool),
        Tensor::C64(t) => download_typed::<Complex64>(rt, t).map(Tensor::C64),
        Tensor::C32(t) => download_typed::<Complex32>(rt, t).map(Tensor::C32),
    }
}

fn upload_typed<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &CudaRuntime,
    client: &ComputeClient<CubeclCudaRuntime>,
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

    let handle = client.create_from_slice(T::as_bytes(host_data));
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(CubeclBuffer::new(
            handle,
            host_data.len(),
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

fn download_typed<T: CubeElement + Clone + 'static>(
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

    rt.synchronize()?;
    let bytes = rt
        .client()
        .read_one(handle)
        .map_err(|err| crate::Error::backend_source("download", err))?;
    let data = T::from_bytes(&bytes).to_vec();
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Host(data),
        Placement::default(),
    )
}

fn upload_bool(
    rt: &CudaRuntime,
    client: &ComputeClient<CubeclCudaRuntime>,
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
    let handle = client.create_from_slice(&bytes);
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(CubeclBuffer::new(
            handle,
            host_data.len(),
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
    match tensor {
        Tensor::F64(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::F32(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::I32(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::I64(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::Bool(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::C64(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
        Tensor::C32(tensor) => dispatch::ensure_resident_on_runtime(rt, tensor, op),
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
