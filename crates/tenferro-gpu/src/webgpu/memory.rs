use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_wgpu::WgpuRuntime;
use num_complex::{Complex32, Complex64};
use std::sync::Arc;

use super::{
    ensure_resident_on_runtime, webgpu_handle_from_backend, webgpu_placement, WebGpuBuffer,
    WebGpuRuntime,
};
use crate::{Buffer, Tensor, TypedTensor};

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
pub fn upload_webgpu_tensor(rt: &WebGpuRuntime, tensor: &Tensor) -> crate::Result<Tensor> {
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
    client: &ComputeClient<WgpuRuntime>,
    device_ordinal: usize,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = match typed.buffer() {
        Buffer::Host(data) => data,
        Buffer::Backend(buffer) => {
            return Err(crate::Error::backend_failure(
                "webgpu_upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let handle = client.create_from_slice(T::as_bytes(host_data));
    Ok(TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        Buffer::Backend(Arc::new(WebGpuBuffer::new(handle, host_data.len()))),
        webgpu_placement(device_ordinal),
    )?)
}

pub(super) fn download_typed<T: CubeElement + Clone + 'static>(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = match typed.buffer() {
        Buffer::Host(_) => {
            return Err(crate::Error::backend_failure(
                "webgpu_download",
                "expected WebGPU backend buffer",
            ));
        }
        Buffer::Backend(buffer) => webgpu_handle_from_backend(buffer.as_ref(), "webgpu_download")?,
    };

    if typed.n_elements() == 0 {
        return Ok(TypedTensor::from_vec_col_major(
            typed.shape().to_vec(),
            Vec::new(),
        )?);
    }

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_failure("webgpu_download", format!("{err:?}")))?;
    let data = T::from_bytes(&bytes).to_vec();
    Ok(TypedTensor::from_vec_col_major(
        typed.shape().to_vec(),
        data,
    )?)
}

fn upload_bool(
    client: &ComputeClient<WgpuRuntime>,
    device_ordinal: usize,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    let host_data = match typed.buffer() {
        Buffer::Host(data) => data,
        Buffer::Backend(buffer) => {
            return Err(crate::Error::backend_failure(
                "webgpu_upload",
                format!(
                    "expected host buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            ));
        }
    };

    let bytes: Vec<u8> = host_data.iter().map(|&value| u8::from(value)).collect();
    let handle = client.create_from_slice(&bytes);
    Ok(TypedTensor::from_buffer_col_major(
        typed.shape().to_vec(),
        Buffer::Backend(Arc::new(WebGpuBuffer::new(handle, host_data.len()))),
        webgpu_placement(device_ordinal),
    )?)
}

fn download_bool(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = match typed.buffer() {
        Buffer::Host(_) => {
            return Err(crate::Error::backend_failure(
                "webgpu_download",
                "expected WebGPU backend buffer",
            ));
        }
        Buffer::Backend(buffer) => webgpu_handle_from_backend(buffer.as_ref(), "webgpu_download")?,
    };

    if typed.n_elements() == 0 {
        return Ok(TypedTensor::from_vec_col_major(
            typed.shape().to_vec(),
            Vec::new(),
        )?);
    }

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_failure("webgpu_download", format!("{err:?}")))?;
    let data = bytes.iter().map(|&byte| byte != 0).collect();
    Ok(TypedTensor::from_vec_col_major(
        typed.shape().to_vec(),
        data,
    )?)
}
