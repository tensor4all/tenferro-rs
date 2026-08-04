use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_wgpu::WgpuRuntime;
use num_complex::{Complex32, Complex64};

use super::{
    ensure_resident_on_runtime, typed_from_webgpu, webgpu_handle_from_backend, WebGpuBuffer,
    WebGpuRuntime,
};
use crate::{StorageBuffer, Tensor, TypedTensor};

/// Upload a host tensor into a CubeCL-managed WebGPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{upload_webgpu_tensor, WebGpuRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _upload: fn(&WebGpuRuntime, &Tensor) -> Result<Tensor> = upload_webgpu_tensor;
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the source buffer is backend
/// resident or belongs to another placement, [`crate::Error::Unsupported`] for
/// a dtype unavailable in WebGPU, or [`crate::Error::BackendSource`] on
/// allocation.
pub fn upload_webgpu_tensor(rt: &WebGpuRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    match tensor {
        Tensor::F64(t) => upload_typed::<f64>(rt, t).map(Tensor::F64),
        Tensor::F32(t) => upload_typed::<f32>(rt, t).map(Tensor::F32),
        Tensor::I32(t) => upload_typed::<i32>(rt, t).map(Tensor::I32),
        Tensor::I64(t) => upload_typed::<i64>(rt, t).map(Tensor::I64),
        Tensor::Bool(t) => upload_bool(rt, t).map(Tensor::Bool),
        Tensor::C64(t) => upload_typed::<Complex64>(rt, t).map(Tensor::C64),
        Tensor::C32(t) => upload_typed::<Complex32>(rt, t).map(Tensor::C32),
    }
}

/// Download a CubeCL-managed WebGPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{download_webgpu_tensor, WebGpuRuntime};
/// use tenferro_tensor::{Result, Tensor};
///
/// let _download: fn(&WebGpuRuntime, &Tensor) -> Result<Tensor> = download_webgpu_tensor;
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] for missing or foreign device state,
/// [`crate::Error::BackendSource`] when queue synchronization/readback fails,
/// or a typed validation error when bytes do not match the tensor shape.
pub fn download_webgpu_tensor(rt: &WebGpuRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
    let client = rt.client();
    match tensor {
        Tensor::F64(t) => download_typed::<f64>(rt, client, t).map(Tensor::F64),
        Tensor::F32(t) => download_typed::<f32>(rt, client, t).map(Tensor::F32),
        Tensor::I32(t) => download_typed::<i32>(rt, client, t).map(Tensor::I32),
        Tensor::I64(t) => download_typed::<i64>(rt, client, t).map(Tensor::I64),
        Tensor::Bool(t) => download_bool(rt, client, t).map(Tensor::Bool),
        Tensor::C64(t) => download_typed::<Complex64>(rt, client, t).map(Tensor::C64),
        Tensor::C32(t) => download_typed::<Complex32>(rt, client, t).map(Tensor::C32),
    }
}

pub(super) fn upload_typed<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &WebGpuRuntime,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = match typed.buffer() {
        StorageBuffer::Host(data) => data,
        StorageBuffer::Backend(buffer) => {
            return Err(crate::Error::runtime_state(
                "webgpu_upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let byte_len = T::as_bytes(host_data).len();
    let handle = rt.client().create_from_slice(T::as_bytes(host_data));
    let buffer = WebGpuBuffer::new_for_runtime(rt, handle, byte_len, "webgpu_upload")?;
    let tensor = typed_from_webgpu(typed.shape().to_vec(), buffer, rt)?;
    rt.record_upload(byte_len);
    Ok(tensor)
}

pub(super) fn download_typed<T: CubeElement + Clone + 'static>(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = match typed.buffer() {
        StorageBuffer::Host(_) => {
            return Err(crate::Error::runtime_state(
                "webgpu_download",
                "expected WebGPU backend buffer",
            ));
        }
        StorageBuffer::Backend(buffer) => {
            webgpu_handle_from_backend(buffer.as_ref(), "webgpu_download")?
        }
    };

    if typed.n_elements() == 0 {
        return TypedTensor::from_buffer_col_major(
            typed.shape().to_vec(),
            StorageBuffer::Host(Vec::new()),
            crate::Placement::default(),
        );
    }

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_source("webgpu_download", err))?;
    let data = T::from_bytes(&bytes).to_vec();
    rt.record_download(bytes.len());
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Host(data),
        crate::Placement::default(),
    )
}

fn upload_bool(rt: &WebGpuRuntime, typed: &TypedTensor<bool>) -> crate::Result<TypedTensor<bool>> {
    let host_data = match typed.buffer() {
        StorageBuffer::Host(data) => data,
        StorageBuffer::Backend(buffer) => {
            return Err(crate::Error::runtime_state(
                "webgpu_upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let bytes: Vec<u8> = host_data.iter().map(|&value| u8::from(value)).collect();
    let handle = rt.client().create_from_slice(&bytes);
    let buffer = WebGpuBuffer::new_for_runtime(rt, handle, bytes.len(), "webgpu_upload")?;
    rt.record_upload(bytes.len());
    TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        StorageBuffer::Backend(Box::new(buffer)),
        super::webgpu_placement(rt),
    )
}

fn download_bool(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = match typed.buffer() {
        StorageBuffer::Host(_) => {
            return Err(crate::Error::runtime_state(
                "webgpu_download",
                "expected WebGPU backend buffer",
            ));
        }
        StorageBuffer::Backend(buffer) => {
            webgpu_handle_from_backend(buffer.as_ref(), "webgpu_download")?
        }
    };

    if typed.n_elements() == 0 {
        return TypedTensor::from_vec_col_major(typed.shape().to_vec(), Vec::new());
    }

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_source("webgpu_download", err))?;
    let data = bytes.iter().map(|&byte| byte != 0).collect();
    rt.record_download(bytes.len());
    TypedTensor::from_vec_col_major(typed.shape().to_vec(), data)
}
