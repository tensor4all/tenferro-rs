//! Host-to-device and device-to-host transfers via CubeCL-managed allocations.

use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_cuda::CudaRuntime;
use num_complex::{Complex32, Complex64};
use std::sync::Arc;

use crate::cubecl::runtime::CubeclRuntime;
use crate::types::{
    Buffer, ComputeDevice, CubeclBuffer, DeviceKind, GpuBackendKind, MemoryKind, Placement, Tensor,
    TypedTensor,
};

/// Upload a host tensor into a CubeCL-managed GPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cubecl::{upload_tensor, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _upload: fn(&CubeclRuntime, &Tensor) -> Result<Tensor> = upload_tensor;
/// ```
pub fn upload_tensor(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor {
        Tensor::F64(t) => upload_typed::<f64>(client, rt.device_ordinal(), t).map(Tensor::F64),
        Tensor::F32(t) => upload_typed::<f32>(client, rt.device_ordinal(), t).map(Tensor::F32),
        Tensor::I32(t) => upload_typed::<i32>(client, rt.device_ordinal(), t).map(Tensor::I32),
        Tensor::I64(t) => upload_typed::<i64>(client, rt.device_ordinal(), t).map(Tensor::I64),
        Tensor::Bool(t) => upload_bool(client, rt.device_ordinal(), t).map(Tensor::Bool),
        Tensor::C64(t) => {
            upload_typed::<Complex64>(client, rt.device_ordinal(), t).map(Tensor::C64)
        }
        Tensor::C32(t) => {
            upload_typed::<Complex32>(client, rt.device_ordinal(), t).map(Tensor::C32)
        }
    }
}

/// Download a CubeCL-managed GPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cubecl::{download_tensor, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _download: fn(&CubeclRuntime, &Tensor) -> Result<Tensor> = download_tensor;
/// ```
pub fn download_tensor(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor {
        Tensor::F64(t) => download_typed::<f64>(client, t).map(Tensor::F64),
        Tensor::F32(t) => download_typed::<f32>(client, t).map(Tensor::F32),
        Tensor::I32(t) => download_typed::<i32>(client, t).map(Tensor::I32),
        Tensor::I64(t) => download_typed::<i64>(client, t).map(Tensor::I64),
        Tensor::Bool(t) => download_bool(client, t).map(Tensor::Bool),
        Tensor::C64(t) => download_typed::<Complex64>(client, t).map(Tensor::C64),
        Tensor::C32(t) => download_typed::<Complex32>(client, t).map(Tensor::C32),
    }
}

/// Extract the raw CUDA device pointer from a CubeCL-managed tensor allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cubecl::{device_ptr, CubeclRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _device_ptr: fn(&CubeclRuntime, &Tensor) -> Result<u64> = device_ptr;
/// ```
pub fn device_ptr(rt: &CubeclRuntime, tensor: &Tensor) -> crate::Result<u64> {
    let handle = cubecl_handle(tensor)?;
    let resource = rt
        .client()
        .get_resource(handle)
        .map_err(|err| crate::Error::backend_failure("device_ptr", format!("{err:?}")))?;
    Ok(resource.resource().ptr)
}

fn upload_typed<T: CubeElement + Clone + Send + Sync + 'static>(
    client: &ComputeClient<CudaRuntime>,
    device_ordinal: usize,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = match &typed.buffer {
        Buffer::Host(data) => data,
        Buffer::Backend(buffer) => {
            return Err(crate::Error::backend_failure(
                "upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let handle = client.create_from_slice(T::as_bytes(host_data));
    Ok(TypedTensor {
        buffer: Buffer::Backend(Arc::new(CubeclBuffer::new(handle, host_data.len()))),
        shape: typed.shape.clone(),
        placement: Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: device_ordinal,
            }),
        },
    })
}

fn download_typed<T: CubeElement + Clone + 'static>(
    client: &ComputeClient<CudaRuntime>,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let handle = match &typed.buffer {
        Buffer::Host(_) => {
            return Err(crate::Error::backend_failure(
                "download",
                "expected CubeCL buffer",
            ));
        }
        Buffer::Backend(buffer) => cubecl_handle_from_backend(buffer.as_ref(), "download")?,
    };

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_failure("download", format!("{err:?}")))?;
    let data = T::from_bytes(&bytes).to_vec();
    Ok(TypedTensor::from_vec_col_major(typed.shape.clone(), data))
}

fn upload_bool(
    client: &ComputeClient<CudaRuntime>,
    device_ordinal: usize,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    let host_data = match &typed.buffer {
        Buffer::Host(data) => data,
        Buffer::Backend(buffer) => {
            return Err(crate::Error::backend_failure(
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
    Ok(TypedTensor {
        buffer: Buffer::Backend(Arc::new(CubeclBuffer::new(handle, host_data.len()))),
        shape: typed.shape.clone(),
        placement: Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: device_ordinal,
            }),
        },
    })
}

fn download_bool(
    client: &ComputeClient<CudaRuntime>,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    let handle = match &typed.buffer {
        Buffer::Host(_) => {
            return Err(crate::Error::backend_failure(
                "download",
                "expected CubeCL buffer",
            ));
        }
        Buffer::Backend(buffer) => cubecl_handle_from_backend(buffer.as_ref(), "download")?,
    };

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_failure("download", format!("{err:?}")))?;
    let data = bytes.iter().map(|&byte| byte != 0).collect();
    Ok(TypedTensor::from_vec_col_major(typed.shape.clone(), data))
}

fn cubecl_handle(tensor: &Tensor) -> crate::Result<cubecl_runtime::server::Handle> {
    match tensor {
        Tensor::F64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::F32(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::I32(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::I64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::Bool(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::C64(t) => cubecl_handle_from_buffer(&t.buffer),
        Tensor::C32(t) => cubecl_handle_from_buffer(&t.buffer),
    }
}

fn cubecl_handle_from_buffer<T: 'static>(
    buffer: &Buffer<T>,
) -> crate::Result<cubecl_runtime::server::Handle> {
    match buffer {
        Buffer::Host(_) => Err(crate::Error::backend_failure(
            "cubecl_handle",
            "expected CubeCL buffer",
        )),
        Buffer::Backend(buffer) => cubecl_handle_from_backend(buffer.as_ref(), "cubecl_handle"),
    }
}

fn cubecl_handle_from_backend<T: 'static>(
    buffer: &dyn crate::BackendBuffer<T>,
    op: &'static str,
) -> crate::Result<cubecl_runtime::server::Handle> {
    buffer
        .as_any()
        .downcast_ref::<CubeclBuffer<T>>()
        .map(|buffer| buffer.handle.clone())
        .ok_or_else(|| {
            crate::Error::backend_failure(
                op,
                format!(
                    "expected CubeCL buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            )
        })
}
