use cubecl::client::ComputeClient;
use cubecl::prelude::CubeElement;
use cubecl_wgpu::WgpuRuntime;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::DType;

use super::{
    ensure_resident_on_runtime, prepared_webgpu_tensor, typed_from_webgpu, WebGpuBuffer,
    WebGpuRuntime,
};
use crate::{Tensor, TypedTensor};

/// The typed tensor behind `tensor`, or a typed refusal.
///
/// Callers reach this from a match on the tensor's dtype, so `None` means the tag
/// table and the runtime dtype disagree rather than a caller mistake.
fn webgpu_typed<'a, T: tenferro_tensor::TensorScalar>(
    op: &'static str,
    tensor: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(op, "the WebGPU memory path requires a preset scalar")
    })
}

/// Upload a host tensor into a CubeCL-managed WebGPU allocation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{webgpu::upload_webgpu_tensor, webgpu::WebGpuRuntime};
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
    match tensor.dtype() {
        DType::F64 => upload_typed::<f64>(rt, webgpu_typed::<f64>("upload_webgpu_tensor", tensor)?)
            .map(Tensor::from_typed::<f64>),
        DType::F32 => upload_typed::<f32>(rt, webgpu_typed::<f32>("upload_webgpu_tensor", tensor)?)
            .map(Tensor::from_typed::<f32>),
        DType::I32 => upload_typed::<i32>(rt, webgpu_typed::<i32>("upload_webgpu_tensor", tensor)?)
            .map(Tensor::from_typed::<i32>),
        DType::I64 => upload_typed::<i64>(rt, webgpu_typed::<i64>("upload_webgpu_tensor", tensor)?)
            .map(Tensor::from_typed::<i64>),
        DType::Bool => upload_bool(rt, webgpu_typed::<bool>("upload_webgpu_tensor", tensor)?)
            .map(Tensor::from_typed::<bool>),
        DType::C64 => upload_typed::<Complex64>(
            rt,
            webgpu_typed::<Complex64>("upload_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<num_complex::Complex64>),
        DType::C32 => upload_typed::<Complex32>(
            rt,
            webgpu_typed::<Complex32>("upload_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<num_complex::Complex32>),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "upload_webgpu_tensor",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

/// Download a CubeCL-managed WebGPU tensor back to host memory.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{webgpu::download_webgpu_tensor, webgpu::WebGpuRuntime};
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
    match tensor.dtype() {
        DType::F64 => download_typed::<f64>(
            rt,
            client,
            webgpu_typed::<f64>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<f64>),
        DType::F32 => download_typed::<f32>(
            rt,
            client,
            webgpu_typed::<f32>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<f32>),
        DType::I32 => download_typed::<i32>(
            rt,
            client,
            webgpu_typed::<i32>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<i32>),
        DType::I64 => download_typed::<i64>(
            rt,
            client,
            webgpu_typed::<i64>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<i64>),
        DType::Bool => download_bool(
            rt,
            client,
            webgpu_typed::<bool>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<bool>),
        DType::C64 => download_typed::<Complex64>(
            rt,
            client,
            webgpu_typed::<Complex64>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<num_complex::Complex64>),
        DType::C32 => download_typed::<Complex32>(
            rt,
            client,
            webgpu_typed::<Complex32>("download_webgpu_tensor", tensor)?,
        )
        .map(Tensor::from_typed::<num_complex::Complex32>),
        // A caller-owned payload has no GPU implementation for this operation.
        DType::External(_) => Err(crate::Error::unsupported(
            "download_webgpu_tensor",
            "an externally defined payload is not supported by this GPU operation",
        )),
    }
}

pub(super) fn upload_typed<T: CubeElement + crate::TensorScalar + Clone + Send + Sync + 'static>(
    rt: &WebGpuRuntime,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    let host_data = typed.host_data().map_err(|error| {
        crate::Error::runtime_state("webgpu_upload", format!("expected host buffer: {error}"))
    })?;

    let byte_len = T::as_bytes(host_data).len();
    let handle = rt.client().create_from_slice(T::as_bytes(host_data));
    let buffer = WebGpuBuffer::new_for_runtime(rt, handle, byte_len, "webgpu_upload")?;
    let tensor = typed_from_webgpu(typed.shape().to_vec(), buffer, rt)?;
    rt.record_upload(byte_len);
    Ok(tensor)
}

pub(super) fn download_typed<T: CubeElement + crate::TensorScalar + Clone + 'static>(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = prepared_webgpu_tensor(typed, "webgpu_download")?.handle;

    if typed.n_elements() == 0 {
        return TypedTensor::from_vec_col_major(typed.shape().to_vec(), Vec::new());
    }

    let bytes = client
        .read_one(handle)
        .map_err(|err| crate::Error::backend_source("webgpu_download", err))?;
    let data = T::from_bytes(&bytes).to_vec();
    rt.record_download(bytes.len());
    TypedTensor::from_vec_col_major(typed.shape().to_vec(), data)
}

fn upload_bool(rt: &WebGpuRuntime, typed: &TypedTensor<bool>) -> crate::Result<TypedTensor<bool>> {
    let host_data = typed.host_data().map_err(|error| {
        crate::Error::runtime_state("webgpu_upload", format!("expected host buffer: {error}"))
    })?;

    let bytes: Vec<u8> = host_data.iter().map(|&value| u8::from(value)).collect();
    let handle = rt.client().create_from_slice(&bytes);
    let buffer = WebGpuBuffer::new_for_runtime(rt, handle, bytes.len(), "webgpu_upload")?;
    rt.record_upload(bytes.len());
    TypedTensor::from_backend_allocation(
        typed.shape().to_vec(),
        Box::new(buffer),
        super::webgpu_placement(rt),
    )
}

fn download_bool(
    rt: &WebGpuRuntime,
    client: &ComputeClient<WgpuRuntime>,
    typed: &TypedTensor<bool>,
) -> crate::Result<TypedTensor<bool>> {
    ensure_resident_on_runtime(rt, typed, "webgpu_download")?;
    let handle = prepared_webgpu_tensor(typed, "webgpu_download")?.handle;

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
