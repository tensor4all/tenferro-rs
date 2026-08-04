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

use crate::{TensorRank, TensorScalar, TypedTensor};

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

    /// Borrow the CUDA device pointer for the duration of `f`.
    ///
    /// The pointer is only exposed while this owner is borrowed, so callers
    /// cannot obtain an unscoped pointer from the workspace handle.
    pub fn with_ptr(&self, f: impl FnOnce(*mut c_void)) {
        f(self.ptr)
    }

    /// Return whether this workspace owns a live CubeCL allocation.
    pub fn is_empty(&self) -> bool {
        self.handle.is_none()
    }
}

pub(crate) fn cuda_device_ptr_from_addr(addr: u64, op: &'static str) -> crate::Result<*mut c_void> {
    let addr = usize::try_from(addr).map_err(|_| {
        crate::Error::invalid_argument(
            op,
            "device_address",
            format!("CUDA device address {addr} exceeds usize"),
        )
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
/// # Errors
///
/// Returns [`crate::Error::BackendSource`] when CubeCL cannot flush the client.
pub fn flush_cubecl_client(rt: &CudaRuntime, op: &'static str) -> crate::Result<()> {
    rt.client()
        .flush()
        .map_err(|err| crate::Error::backend_source(op, err))
}

/// Borrow the CUDA stream pointer for libraries that must enqueue onto CubeCL's stream.
///
/// The stream is passed only to `f`; callers must not retain the raw handle
/// after the callback returns.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaRuntime;
/// use tenferro_gpu::cuda::interop::with_raw_cuda_stream;
///
/// # fn example(rt: &CudaRuntime) -> tenferro_tensor::Result<()> {
/// with_raw_cuda_stream(rt, "example", |_stream| {})?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::BackendSource`] when CubeCL cannot expose the
/// stream, or [`crate::Error::RuntimeState`] when its server is unavailable.
pub fn with_raw_cuda_stream(
    rt: &CudaRuntime,
    op: &'static str,
    f: impl FnOnce(u64),
) -> crate::Result<()> {
    let stream = rt
        .raw_cuda_stream()
        .map_err(|err| crate::Error::backend_source(op, err))?;
    f(stream);
    Ok(())
}

/// Return the launch cube count for a one-dimensional kernel domain.
/// # Errors
///
/// Returns [`crate::Error::Validation`] containing
/// [`tenferro_tensor::ValidationError::InvalidArgument`] when the
/// one-dimensional launch for `len` elements would require more than
/// `u32::MAX` CubeCL workgroups.
pub fn cube_count_for_len(len: usize) -> crate::Result<CubeCount> {
    dispatch::cube_count_for_len(len)
}

/// Return the standard one-dimensional CubeCL launch dimension.
pub fn cube_dim_1d() -> CubeDim {
    dispatch::cube_dim_1d()
}

/// Allocate a dense GPU tensor on the runtime's device.
/// # Errors
///
/// Returns [`crate::Error::Validation`] with `InvalidArgument` when the shape
/// product overflows, or [`crate::Error::BackendSource`] when allocation fails.
pub fn alloc_output<T: CubeElement + TensorScalar + Clone + Send + Sync + 'static>(
    rt: &CudaRuntime,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    dispatch::alloc_output(rt, shape)
}

/// Validate that a tensor is backed by a CubeCL buffer.
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the tensor is host-backed,
/// belongs to another GPU runtime, or has the wrong backend buffer family.
pub fn ensure_typed_tensor_resident<T: 'static>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<()> {
    dispatch::cubecl_buffer(tensor, op)?;
    Ok(())
}

/// Build a CubeCL tensor binding for operation-family kernels.
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the tensor is not CubeCL
/// resident, or [`crate::Error::Validation`] when its layout cannot be bound.
pub fn typed_tensor_binding<T: CubeElement + TensorScalar + Clone>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<TensorBinding<CubeclCudaRuntime>> {
    dispatch::typed_tensor_binding(tensor, op)
}

/// Build a CubeCL array argument for operation-family kernels.
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the tensor is not CubeCL
/// resident, or [`crate::Error::Validation`] when its layout cannot be bound.
pub fn typed_tensor_array_arg<T: CubeElement + TensorScalar + Clone>(
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
    dispatch::typed_tensor_array_arg(tensor, op)
}

/// Borrow a raw CUDA device pointer for a CubeCL-backed tensor.
///
/// The pointer is passed only to `f`, while the residency-checked tensor and
/// runtime remain borrowed by this call. Callers must not retain the pointer
/// after `f` returns.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::interop::with_typed_device_ptr;
/// use tenferro_gpu::cuda::CudaRuntime;
/// use tenferro_tensor::TypedTensor;
///
/// # fn example(rt: &CudaRuntime, tensor: &TypedTensor<f32>) -> tenferro_tensor::Result<()> {
/// with_typed_device_ptr(rt, tensor, "example", |_ptr| {})?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] for a non-resident or foreign tensor,
/// [`crate::Error::BackendSource`] when its resource cannot be inspected, or
/// [`crate::Error::Validation`] when the pointer address overflows `usize`.
pub fn with_typed_device_ptr<T: TensorScalar + 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
    f: impl FnOnce(*mut c_void),
) -> crate::Result<()> {
    dispatch::ensure_resident_on_runtime(rt, tensor, op)?;
    let prepared = dispatch::prepared_tensor_access(tensor, op)?;
    let resource = rt
        .client()
        .get_resource(prepared.into_handle())
        .map_err(|err| crate::Error::backend_source(op, err))?;
    // The residency check above ties this raw FFI pointer to the caller's
    // runtime/device for the duration of the callback.
    let ptr = cuda_device_ptr_from_addr(resource.resource().ptr, op)?;
    f(ptr);
    Ok(())
}

/// Upload host data into a dense GPU tensor on the runtime's device.
/// # Errors
///
/// Returns [`crate::Error::Validation`] when `shape` and `data` have different
/// element counts, or [`crate::Error::BackendSource`] when device allocation
/// fails.
pub fn upload_typed_tensor<T>(
    rt: &CudaRuntime,
    shape: Vec<usize>,
    data: Vec<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
{
    let byte_len = T::as_bytes(&data).len();
    let handle = rt.client().create_from_slice(T::as_bytes(&data));
    dispatch::typed_from_cubecl(
        shape,
        crate::CubeclBuffer::new(
            handle,
            byte_len,
            rt.device_ordinal(),
            rt.allocation_domain_id(),
        ),
        rt.device_ordinal(),
    )
}

/// Download a dense CubeCL-backed typed tensor to host memory.
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] for a host-backed or foreign tensor,
/// [`crate::Error::BackendSource`] when synchronization/readback fails, or a
/// typed validation error when downloaded bytes do not form the declared shape.
pub fn download_typed_tensor<T>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + TensorScalar + Clone + 'static,
{
    dispatch::ensure_resident_on_runtime(rt, tensor, op)?;
    let prepared = dispatch::prepared_tensor_access(tensor, op)?;
    if tensor.n_elements() == 0 {
        return TypedTensor::from_vec_col_major(tensor.shape().to_vec(), Vec::new());
    }
    rt.synchronize()?;
    let bytes = rt
        .client()
        .read_one(prepared.into_handle())
        .map_err(|err| crate::Error::backend_source(op, err))?;
    TypedTensor::from_vec_col_major(tensor.shape().to_vec(), T::from_bytes(&bytes).to_vec())
}

/// Allocate a CubeCL-owned byte workspace and return its CUDA pointer.
/// # Errors
///
/// Returns [`crate::Error::BackendSource`] when CubeCL cannot allocate or
/// inspect the workspace resource, or [`crate::Error::Validation`] when its
/// pointer address cannot be represented as `usize`.
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
/// # Errors
///
/// Returns [`crate::Error::BackendSource`] when CubeCL cannot upload or inspect
/// the workspace resource, or [`crate::Error::Validation`] on pointer overflow.
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
    let resource = rt
        .client()
        .get_resource(handle.clone())
        .map_err(|err| crate::Error::backend_source(op, err))?;
    Ok(DeviceByteBuffer {
        handle: Some(handle),
        ptr: cuda_device_ptr_from_addr(resource.resource().ptr, op)?,
    })
}
