//! Owner-scoped CubeCL integration helpers for standard operation crates.
//!
//! This module is intentionally narrow: it exposes the launch, allocation, and
//! pointer bridges needed by operation-family crates that provide CUDA kernels
//! against tenferro's CubeCL runtime, without exposing the backend's raw buffer
//! representation on `CudaRuntime` or `CubeclBuffer` themselves.

use std::ffi::c_void;
use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, TensorBinding};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use crate::{TensorRank, TypedTensor};

use super::{dispatch, CudaRuntime};

/// CubeCL-owned byte allocation kept alive for CUDA-library workspace calls.
pub struct DeviceByteBuffer {
    handle: Option<cubecl_runtime::server::Handle>,
    ptr: *mut c_void,
}

impl fmt::Debug for DeviceByteBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceByteBuffer")
            .field("is_empty", &self.is_empty())
            .field("ptr", &self.ptr)
            .finish_non_exhaustive()
    }
}

impl DeviceByteBuffer {
    /// Return an empty workspace.
    pub fn none() -> Self {
        Self {
            handle: None,
            ptr: std::ptr::null_mut(),
        }
    }

    /// Return the CUDA device pointer for this workspace.
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Return whether this workspace owns a live CubeCL allocation.
    pub fn is_empty(&self) -> bool {
        self.handle.is_none()
    }
}

pub(crate) fn cuda_device_ptr_from_addr(addr: u64, op: &'static str) -> crate::Result<*mut c_void> {
    let addr = usize::try_from(addr).map_err(|_| {
        crate::Error::backend_failure(op, format!("CUDA device address {addr} exceeds usize"))
    })?;
    Ok(std::ptr::with_exposed_provenance_mut::<c_void>(addr))
}

/// Run a closure with the CubeCL compute client.
///
/// This is for operation-family kernel launches that cannot be implemented
/// inside `tenferro-gpu` without creating a dependency cycle.
pub fn with_cubecl_client<R>(
    rt: &CudaRuntime,
    launch: impl FnOnce(&ComputeClient<CubeclCudaRuntime>) -> R,
) -> R {
    launch(rt.client())
}

/// Flush the CubeCL client after an unchecked kernel launch.
pub fn flush_cubecl_client(rt: &CudaRuntime, op: &'static str) -> crate::Result<()> {
    rt.client()
        .flush()
        .map_err(|err| crate::Error::backend_failure(op, format!("CubeCL launch failed: {err:?}")))
}

/// Return the CUDA stream pointer for libraries that must enqueue onto CubeCL's stream.
pub fn raw_cuda_stream(rt: &CudaRuntime, op: &'static str) -> crate::Result<u64> {
    rt.raw_cuda_stream()
        .map_err(|err| crate::Error::backend_failure(op, err.to_string()))
}

/// Return the launch cube count for a one-dimensional kernel domain.
pub fn cube_count_for_len(len: usize) -> crate::Result<CubeCount> {
    dispatch::cube_count_for_len(len)
}

/// Return the standard one-dimensional CubeCL launch dimension.
pub fn cube_dim_1d() -> CubeDim {
    dispatch::cube_dim_1d()
}

/// Allocate a dense GPU tensor on the runtime's device.
pub fn alloc_output<T: CubeElement + Clone + Send + Sync + 'static>(
    rt: &CudaRuntime,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    dispatch::alloc_output(rt, shape)
}

/// Validate that a tensor is backed by a CubeCL buffer.
pub fn ensure_typed_tensor_resident<T: 'static>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<()> {
    dispatch::cubecl_buffer(tensor, op)?;
    Ok(())
}

/// Build a CubeCL tensor binding for operation-family kernels.
pub fn typed_tensor_binding<T: CubeElement + Clone>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<TensorBinding<CubeclCudaRuntime>> {
    dispatch::typed_tensor_binding(tensor, op)
}

/// Build a CubeCL array argument for operation-family kernels.
pub fn typed_tensor_array_arg<T: CubeElement + Clone>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    dispatch::typed_tensor_array_arg(tensor, op)
}

/// Return a raw CUDA device pointer for a CubeCL-backed tensor.
pub fn typed_device_ptr<T: 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<*mut c_void> {
    dispatch::ensure_resident_on_runtime(rt, tensor, op)?;
    let buffer = dispatch::cubecl_buffer(tensor, op)?;
    let resource = rt
        .client()
        .get_resource(buffer.handle().clone())
        .map_err(|err| {
            crate::Error::backend_failure(op, format!("failed to obtain CubeCL resource: {err:?}"))
        })?;
    // The residency check above ties this raw FFI pointer to the caller's runtime/device.
    cuda_device_ptr_from_addr(resource.resource().ptr, op)
}

/// Upload host data into a dense GPU tensor on the runtime's device.
pub fn upload_typed_tensor<T>(
    rt: &CudaRuntime,
    shape: Vec<usize>,
    data: Vec<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone + Send + Sync + 'static,
{
    let len = data.len();
    let handle = rt.client().create_from_slice(T::as_bytes(&data));
    dispatch::typed_from_cubecl(
        shape,
        crate::CubeclBuffer::new(handle, len),
        rt.device_ordinal(),
    )
}

/// Download a dense CubeCL-backed typed tensor to host memory.
pub fn download_typed_tensor<T>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone + 'static,
{
    dispatch::ensure_resident_on_runtime(rt, tensor, op)?;
    let buffer = dispatch::cubecl_buffer(tensor, op)?;
    if tensor.n_elements() == 0 {
        return TypedTensor::from_vec_col_major(tensor.shape().to_vec(), Vec::new());
    }
    rt.synchronize()?;
    let bytes = rt
        .client()
        .read_one(buffer.handle().clone())
        .map_err(|err| {
            crate::Error::backend_failure(op, format!("failed to download tensor: {err:?}"))
        })?;
    TypedTensor::from_vec_col_major(tensor.shape().to_vec(), T::from_bytes(&bytes).to_vec())
}

/// Allocate a CubeCL-owned byte workspace and return its CUDA pointer.
pub fn alloc_device_bytes(
    rt: &CudaRuntime,
    nbytes: usize,
    op: &'static str,
) -> crate::Result<DeviceByteBuffer> {
    if nbytes == 0 {
        return Ok(DeviceByteBuffer::none());
    }
    let handle = rt.client().empty(nbytes);
    device_bytes_from_handle(rt, handle, op)
}

/// Upload bytes into a CubeCL-owned workspace and return its CUDA pointer.
pub fn upload_device_bytes(
    rt: &CudaRuntime,
    bytes: &[u8],
    op: &'static str,
) -> crate::Result<DeviceByteBuffer> {
    if bytes.is_empty() {
        return Ok(DeviceByteBuffer::none());
    }
    let handle = rt.client().create_from_slice(bytes);
    device_bytes_from_handle(rt, handle, op)
}

fn device_bytes_from_handle(
    rt: &CudaRuntime,
    handle: cubecl_runtime::server::Handle,
    op: &'static str,
) -> crate::Result<DeviceByteBuffer> {
    let resource = rt.client().get_resource(handle.clone()).map_err(|err| {
        crate::Error::backend_failure(op, format!("failed to obtain CubeCL resource: {err:?}"))
    })?;
    Ok(DeviceByteBuffer {
        handle: Some(handle),
        ptr: cuda_device_ptr_from_addr(resource.resource().ptr, op)?,
    })
}
