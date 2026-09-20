//! Owner-scoped CubeCL integration helpers internal to this crate.
//!
//! This module is intentionally narrow: it provides the launch, allocation,
//! and device-address helpers used by this crate's raw/session and kernel
//! surfaces, without exposing the backend's raw buffer representation on
//! `CudaRuntime` or `CubeclBuffer` themselves. It is `pub(crate)` and is not
//! re-exported publicly; operation-family crates consume the credentialed
//! `cuda::raw`/`cuda::cubecl` sessions instead (issue #1597).

use std::ffi::c_void;
use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, CubePrimitive, TensorBinding};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};

use crate::{TensorRank, TensorScalar, TypedTensor};
use tenferro_tensor::{DType, TensorRead, TensorViewMut, TensorWrite, TypedTensorViewMut};

use super::error::unsupported_dtype;
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

/// Allocate and fill a dense CUDA tensor with semantic zeros on `rt`.
///
/// This is an owner-scoped bridge for operation-family padding and empty-input
/// preparation. It reuses the backend's existing fill-zero kernel and never
/// uploads a host tensor or exposes a device pointer to the caller.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] with
/// [`crate::ValidationError::InvalidArgument`] when the shape product, output
/// byte length, or launch count overflows, [`crate::Error::RuntimeState`] when the
/// output is not resident on `rt`, or [`crate::Error::BackendSource`] when
/// allocation or backend resource inspection fails.
#[doc(hidden)]
pub fn alloc_zero_output<T>(rt: &CudaRuntime, shape: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + TensorScalar + Clone + Send + Sync + 'static,
{
    let mut output = alloc_output::<T>(rt, shape)?;
    // One stream-ordered memset over the fresh allocation: dtype-agnostic,
    // exact `+0.0` bits, and no kernel compilation on the first call for a
    // dtype. This is the same primitive `fill_zero_write` uses.
    let len = output.n_elements();
    if len > 0 {
        let handle = dispatch::cubecl_buffer(&output, "alloc_zero_output")?
            .handle()
            .clone();
        let prepared = dispatch::prepared_tensor_mut_access(&mut output, "alloc_zero_output")?;
        fill_zero_span::<T>(rt, prepared, 0, len, handle)?;
    }
    Ok(output)
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
/// after `f` returns. This internal module is not part of the public API; the
/// pointer-escape contract is enforced by source-contract tests.
#[cfg_attr(not(test), allow(dead_code))] // used by unit tests; kept for the scoped-accessor contract.
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] for a non-resident or foreign tensor,
/// [`crate::Error::BackendSource`] when its resource cannot be inspected, or
/// [`crate::Error::Validation`] when the pointer address overflows `usize`.
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

/// Retain a clone of a resident tensor's CubeCL allocation handle.
///
/// The returned [`DeviceByteBuffer`] holds a reference-counted clone of the
/// tensor's allocation handle, so the device memory stays alive until the
/// guard is dropped (or intentionally forgotten). Vendor libraries that
/// enqueue asynchronous work against the tensor's address can use this to
/// prevent allocation reclamation racing an in-flight kernel after a failed
/// synchronization barrier.
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when `tensor` is host-backed,
/// belongs to a non-CubeCL backend family, belongs to a different CUDA
/// runtime domain, or is not resident on `rt`'s device, or
/// [`crate::Error::BackendSource`] when CubeCL cannot inspect the retained
/// resource.
pub(crate) fn retain_tensor_bytes<T: 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, impl TensorRank>,
    op: &'static str,
) -> crate::Result<DeviceByteBuffer> {
    // The public raw seam must uphold the resident-tensor contract itself:
    // validate exact-runtime residency (allocation domain + device placement)
    // before cloning the handle, so a foreign-runtime or host tensor can never
    // be retained by a session that does not own it.
    dispatch::ensure_resident_on_runtime(rt, tensor, op)?;
    let buffer = dispatch::cubecl_buffer(tensor, op)?;
    device_bytes_from_handle(rt, buffer.handle().clone(), op)
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

const SCALE_OP: &str = "scale_tensor_write";

/// Scale a writable CUDA tensor in place by a real device-resident factor.
///
/// The output retains its existing placement and allocation owner. Only
/// compact, zero-offset writable targets are accepted because the shared
/// structural kernels operate on a one-dimensional contiguous array.
///
/// # Errors
///
/// Returns a typed unsupported-dtype error for integer and boolean outputs,
/// or a runtime/validation error when the output is host-backed, belongs to a
/// different CUDA runtime, has an invalid buffer/layout, or cannot be bound.
#[doc(hidden)]
pub fn scale_tensor_write(
    rt: &CudaRuntime,
    output: TensorWrite<'_>,
    factor: f64,
) -> crate::Result<()> {
    ensure_tensor_write_resident(rt, &output, SCALE_OP)?;
    let dtype = output.dtype();
    if !matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64) {
        return Err(unsupported_dtype(SCALE_OP, dtype));
    }

    match output {
        TensorWrite::Tensor(output) => match output.dtype() {
            DType::F32 => scale_typed_tensor(
                rt,
                output
                    .as_typed_mut::<f32>()
                    .expect("the dtype guard selects this arm"),
                factor as f32,
                launch_scale_f32,
            ),
            DType::F64 => scale_typed_tensor(
                rt,
                output
                    .as_typed_mut::<f64>()
                    .expect("the dtype guard selects this arm"),
                factor,
                launch_scale_f64,
            ),
            DType::C32 => scale_typed_tensor(
                rt,
                output
                    .as_typed_mut::<Complex32>()
                    .expect("the dtype guard selects this arm"),
                Complex32::new(factor as f32, 0.0),
                launch_scale_c32,
            ),
            DType::C64 => scale_typed_tensor(
                rt,
                output
                    .as_typed_mut::<Complex64>()
                    .expect("the dtype guard selects this arm"),
                Complex64::new(factor, 0.0),
                launch_scale_c64,
            ),
            _ => Err(unsupported_dtype(SCALE_OP, dtype)),
        },
        TensorWrite::View(mut output) => match &mut output {
            TensorViewMut::F32(output) => {
                scale_typed_view(rt, output, factor as f32, launch_scale_f32)
            }
            TensorViewMut::F64(output) => scale_typed_view(rt, output, factor, launch_scale_f64),
            TensorViewMut::C32(output) => scale_typed_view(
                rt,
                output,
                Complex32::new(factor as f32, 0.0),
                launch_scale_c32,
            ),
            TensorViewMut::C64(output) => {
                scale_typed_view(rt, output, Complex64::new(factor, 0.0), launch_scale_c64)
            }
            _ => Err(unsupported_dtype(SCALE_OP, dtype)),
        },
    }
}

fn ensure_tensor_write_resident(
    rt: &CudaRuntime,
    output: &TensorWrite<'_>,
    op: &'static str,
) -> crate::Result<()> {
    let read = output.as_read();
    match &read {
        TensorRead::Tensor(output) => match output.dtype() {
            DType::F32 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<f32>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::F64 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<f64>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::I32 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<i32>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::I64 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<i64>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::Bool => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<bool>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::C32 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<Complex32>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            DType::C64 => dispatch::ensure_resident_on_runtime(
                rt,
                output
                    .as_typed::<Complex64>()
                    .expect("the dtype guard selects this arm"),
                op,
            ),
            // A caller-owned payload has no GPU implementation for this operation.
            _ => Err(crate::Error::unsupported(
                "ensure_tensor_write_resident",
                "an externally defined payload is not supported by this GPU operation",
            )),
        },
        TensorRead::View(output) => match output {
            crate::TensorView::F32(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::F64(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::I32(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::I64(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::Bool(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::C32(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
            crate::TensorView::C64(output) => {
                dispatch::ensure_view_resident_on_runtime(rt, output, op)
            }
        },
    }
}

/// Shared typed scaling bridge used by CUDA operation-family code that already
/// owns a typed mutable tensor and factor.
pub(crate) fn scale_typed_tensor<T, F>(
    rt: &CudaRuntime,
    output: &mut TypedTensor<T>,
    factor: T,
    launch: F,
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
    F: FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
{
    scale_typed_tensor_for_op(rt, output, factor, SCALE_OP, launch)
}

pub(crate) fn scale_typed_tensor_for_op<T, F>(
    rt: &CudaRuntime,
    output: &mut TypedTensor<T>,
    factor: T,
    op: &'static str,
    launch: F,
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
    F: FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
{
    dispatch::ensure_resident_on_runtime(rt, output, op)?;
    let len = output.n_elements();
    validate_scale_buffer(op, len, output.buffer().len())?;
    if len == 0 {
        return Ok(());
    }
    let count = dispatch::cube_count_for_len(len)?;
    let dim = dispatch::cube_dim_1d();
    let mut output_view = output.as_view_mut();
    let output_arg = dispatch::typed_view_mut_array_arg(&mut output_view, op)?;
    launch_scaled(rt, output_arg, factor, count, dim, op, launch)
}

fn scale_typed_view<T, F>(
    rt: &CudaRuntime,
    output: &mut TypedTensorViewMut<'_, T>,
    factor: T,
    launch: F,
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
    F: FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
{
    dispatch::ensure_view_mut_resident_on_runtime(rt, output, SCALE_OP)?;
    if output.offset() != 0 || !output.is_col_major_contiguous()? {
        return Err(crate::Error::invalid_argument(
            SCALE_OP,
            "layout",
            "CUDA tensor scaling requires a zero-offset column-major view",
        ));
    }
    let len = output.n_elements();
    let buffer_len = output
        .backend_buffer()
        .ok_or_else(|| crate::Error::runtime_state(SCALE_OP, "expected a CUDA backend buffer"))?
        .len();
    validate_scale_buffer(SCALE_OP, len, buffer_len)?;
    if len == 0 {
        return Ok(());
    }
    let count = dispatch::cube_count_for_len(len)?;
    let dim = dispatch::cube_dim_1d();
    let output_arg = dispatch::typed_view_mut_array_arg(output, SCALE_OP)?;
    launch_scaled(rt, output_arg, factor, count, dim, SCALE_OP, launch)
}

fn validate_scale_buffer(op: &'static str, len: usize, buffer_len: usize) -> crate::Result<()> {
    if len > buffer_len {
        return Err(crate::Error::runtime_state(
            op,
            format!(
                "CUDA tensor scaling output has {len} logical elements but its buffer has {buffer_len}"
            ),
        ));
    }
    Ok(())
}

fn launch_scaled<T, F>(
    rt: &CudaRuntime,
    output: ArrayArg<CubeclCudaRuntime>,
    factor: T,
    count: CubeCount,
    dim: CubeDim,
    op: &'static str,
    launch: F,
) -> crate::Result<()>
where
    T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
    F: FnOnce(
        &ComputeClient<CubeclCudaRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<CubeclCudaRuntime>,
        ArrayArg<CubeclCudaRuntime>,
    ),
{
    let factor = upload_typed_tensor(rt, vec![1], vec![factor])?;
    let factor = dispatch::typed_tensor_array_arg(&factor, op)?;
    launch(rt.client(), count, dim, output, factor);
    Ok(())
}

fn launch_scale_f32(
    client: &ComputeClient<CubeclCudaRuntime>,
    count: CubeCount,
    dim: CubeDim,
    output: ArrayArg<CubeclCudaRuntime>,
    factor: ArrayArg<CubeclCudaRuntime>,
) {
    // SAFETY: the typed scaling bridge validates residency, buffer length, and
    // the one-dimensional launch domain before this unchecked kernel launch.
    // INVARIANT: the bridge validates exact runtime residency, a zero-offset
    // compact span with len <= buffer_len, and cube_count_for_len(len) before
    // this binding is consumed.
    unsafe {
        crate::kernels::structural::scale_in_place_float_kernel::launch_unchecked::<
            f32,
            CubeclCudaRuntime,
        >(client, count, dim, output, factor);
    }
}

fn launch_scale_f64(
    client: &ComputeClient<CubeclCudaRuntime>,
    count: CubeCount,
    dim: CubeDim,
    output: ArrayArg<CubeclCudaRuntime>,
    factor: ArrayArg<CubeclCudaRuntime>,
) {
    // SAFETY: the typed scaling bridge validates residency, buffer length, and
    // the one-dimensional launch domain before this unchecked kernel launch.
    // INVARIANT: the bridge validates exact runtime residency, a zero-offset
    // compact span with len <= buffer_len, and cube_count_for_len(len) before
    // this binding is consumed.
    unsafe {
        crate::kernels::structural::scale_in_place_float_kernel::launch_unchecked::<
            f64,
            CubeclCudaRuntime,
        >(client, count, dim, output, factor);
    }
}

fn launch_scale_c32(
    client: &ComputeClient<CubeclCudaRuntime>,
    count: CubeCount,
    dim: CubeDim,
    output: ArrayArg<CubeclCudaRuntime>,
    factor: ArrayArg<CubeclCudaRuntime>,
) {
    // SAFETY: the typed scaling bridge validates residency, buffer length, and
    // the one-dimensional launch domain before this unchecked kernel launch.
    // INVARIANT: the bridge validates exact runtime residency, a zero-offset
    // compact span with len <= buffer_len, and cube_count_for_len(len) before
    // this binding is consumed.
    unsafe {
        crate::kernels::structural::scale_in_place_complex_kernel::launch_unchecked::<
            Complex32,
            CubeclCudaRuntime,
        >(client, count, dim, output, factor);
    }
}

fn launch_scale_c64(
    client: &ComputeClient<CubeclCudaRuntime>,
    count: CubeCount,
    dim: CubeDim,
    output: ArrayArg<CubeclCudaRuntime>,
    factor: ArrayArg<CubeclCudaRuntime>,
) {
    // SAFETY: the typed scaling bridge validates residency, buffer length, and
    // the one-dimensional launch domain before this unchecked kernel launch.
    // INVARIANT: the bridge validates exact runtime residency, a zero-offset
    // compact span with len <= buffer_len, and cube_count_for_len(len) before
    // this binding is consumed.
    unsafe {
        crate::kernels::structural::scale_in_place_complex_kernel::launch_unchecked::<
            Complex64,
            CubeclCudaRuntime,
        >(client, count, dim, output, factor);
    }
}

const FILL_ZERO_OP: &str = "fill_zero_write";

/// Overwrite every element a destination addresses with an exact `+0.0`.
///
/// This is the `beta = 0` reset a consumer needs when it re-executes an
/// accumulate-form operation into storage it owns: the previous contents are
/// never read, so a stale `NaN`/`Inf` cannot survive and `-0.0` is normalized,
/// which `0 * y` cannot promise. Compact spans (including a nonzero offset) are
/// filled with one stream-ordered `cuMemsetD8Async`, so no kernel is compiled
/// and no device or host memory is allocated. A strided destination region runs
/// one native fill kernel over its logical coordinates, leaving every element
/// outside the region untouched.
///
/// # Errors
///
/// Returns [`crate::Error::RuntimeState`] when the destination is not resident
/// on this runtime, [`crate::Error::Validation`] when the destination layout
/// or its byte span cannot be represented, and [`crate::Error::BackendSource`]
/// when the fill cannot be enqueued.
pub fn fill_zero_write(rt: &CudaRuntime, output: TensorWrite<'_>) -> crate::Result<()> {
    ensure_tensor_write_resident(rt, &output, FILL_ZERO_OP)?;
    match output {
        TensorWrite::Tensor(output) => match output.dtype() {
            DType::F32 => fill_zero_typed_tensor::<f32>(rt, output),
            DType::F64 => fill_zero_typed_tensor::<f64>(rt, output),
            DType::I32 => fill_zero_typed_tensor::<i32>(rt, output),
            DType::I64 => fill_zero_typed_tensor::<i64>(rt, output),
            DType::Bool => fill_zero_typed_tensor::<bool>(rt, output),
            DType::C32 => fill_zero_typed_tensor::<Complex32>(rt, output),
            DType::C64 => fill_zero_typed_tensor::<Complex64>(rt, output),
            dtype => Err(unsupported_dtype(FILL_ZERO_OP, dtype)),
        },
        TensorWrite::View(mut output) => match &mut output {
            TensorViewMut::F32(output) => fill_zero_typed_view::<f32>(rt, output),
            TensorViewMut::F64(output) => fill_zero_typed_view::<f64>(rt, output),
            TensorViewMut::I32(output) => fill_zero_typed_view::<i32>(rt, output),
            TensorViewMut::I64(output) => fill_zero_typed_view::<i64>(rt, output),
            TensorViewMut::Bool(output) => fill_zero_bool_view(rt, output),
            TensorViewMut::C32(output) => fill_zero_typed_view::<Complex32>(rt, output),
            TensorViewMut::C64(output) => fill_zero_typed_view::<Complex64>(rt, output),
        },
    }
}

fn fill_zero_typed_tensor<T>(rt: &CudaRuntime, output: &mut crate::Tensor) -> crate::Result<()>
where
    T: TensorScalar + Clone + Send + Sync + 'static,
{
    let output = output
        .as_typed_mut::<T>()
        .ok_or_else(|| unsupported_dtype(FILL_ZERO_OP, T::dtype()))?;
    let len = output.n_elements();
    if len == 0 {
        return Ok(());
    }
    let handle = dispatch::cubecl_buffer(output, FILL_ZERO_OP)?
        .handle()
        .clone();
    let prepared = dispatch::prepared_tensor_mut_access(output, FILL_ZERO_OP)?;
    // An owned runtime tensor is a compact span starting at its allocation.
    fill_zero_span::<T>(rt, prepared, 0, len, handle)
}

fn fill_zero_typed_view<T>(
    rt: &CudaRuntime,
    output: &mut TypedTensorViewMut<'_, T>,
) -> crate::Result<()>
where
    T: CubeElement + CubePrimitive + TensorScalar + Clone + Send + Sync + 'static,
{
    dispatch::ensure_view_mut_resident_on_runtime(rt, output, FILL_ZERO_OP)?;
    let len = output.n_elements();
    if len == 0 {
        return Ok(());
    }
    if output.is_col_major_contiguous()? {
        return fill_zero_view_span::<T>(rt, output, len);
    }
    fill_zero_strided_view::<T>(rt, output, len)
}

/// `Bool` destinations reuse the dtype-agnostic memset for a compact span.
///
/// A strided `Bool` region would need a `u8`-bound fill kernel, which is the
/// same gap the CUDA backend already reports for `Bool` materialization and
/// `copy_into`, so it stays an explicit typed error rather than a silent
/// host round trip.
fn fill_zero_bool_view(
    rt: &CudaRuntime,
    output: &mut TypedTensorViewMut<'_, bool>,
) -> crate::Result<()> {
    dispatch::ensure_view_mut_resident_on_runtime(rt, output, FILL_ZERO_OP)?;
    let len = output.n_elements();
    if len == 0 {
        return Ok(());
    }
    if output.is_col_major_contiguous()? {
        return fill_zero_view_span::<bool>(rt, output, len);
    }
    Err(crate::Error::unsupported(
        FILL_ZERO_OP,
        "CUDA fill_zero_write does not support a strided Bool destination; \
         fill a compact Bool destination or download the tensor to host",
    ))
}

fn fill_zero_view_span<T>(
    rt: &CudaRuntime,
    output: &mut TypedTensorViewMut<'_, T>,
    len: usize,
) -> crate::Result<()>
where
    T: TensorScalar + Clone + 'static,
{
    let handle = dispatch::cubecl_view_mut_buffer(output, FILL_ZERO_OP)?
        .handle()
        .clone();
    let offset = output.offset();
    let prepared = dispatch::prepared_view_mut_access(output, FILL_ZERO_OP)?;
    fill_zero_span::<T>(rt, prepared, offset, len, handle)
}

/// Fill a compact device span of `len` elements with zero bytes.
///
/// The memset is enqueued on the runtime's CUDA stream, so it is ordered with
/// the CubeCL kernels around it. `flush_cubecl` first submits CubeCL's pending
/// host-side launches, which a raw stream call would otherwise overtake.
fn fill_zero_span<T: 'static>(
    rt: &CudaRuntime,
    prepared: dispatch::CubeclPreparedAccess,
    offset: isize,
    len: usize,
    handle: cubecl_runtime::server::Handle,
) -> crate::Result<()> {
    let byte_len = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        crate::Error::invalid_argument(FILL_ZERO_OP, "shape", "destination byte length overflows")
    })?;
    rt.set_current_cuda_context(FILL_ZERO_OP)?;
    let ptr = dispatch::offset_device_ptr::<T>(rt, prepared, offset, FILL_ZERO_OP)?;
    rt.flush_cubecl(FILL_ZERO_OP)?;
    let stream = rt.raw_cuda_stream()?;
    let cross_stream = if rt.is_current_stream_slot(&handle) {
        Vec::new()
    } else {
        vec![handle]
    };
    // SAFETY: `ptr` resolves the destination's prepared device access on this
    // runtime, `byte_len` is the checked byte product of the validated compact
    // element span reachable from that pointer, and `stream` is this runtime's
    // CUDA stream for the current CubeCL stream slot. The memset writes only
    // that span and reads nothing.
    let result = unsafe {
        cudarc::driver::result::memset_d8_async(
            ptr as u64,
            0,
            byte_len,
            stream as usize as cudarc::driver::sys::CUstream,
        )
    }
    .map_err(|err| crate::Error::backend_source(FILL_ZERO_OP, err));
    rt.finish_vendor_enqueue(FILL_ZERO_OP, cross_stream, result)
}

/// Fill the logical coordinates of a strided destination region with zero.
fn fill_zero_strided_view<T>(
    rt: &CudaRuntime,
    output: &mut TypedTensorViewMut<'_, T>,
    len: usize,
) -> crate::Result<()>
where
    T: CubeElement + CubePrimitive + TensorScalar + Clone + Send + Sync + 'static,
{
    let dims = output.shape().to_vec();
    let strides = output
        .strides()
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| {
                crate::Error::invalid_argument(
                    FILL_ZERO_OP,
                    "layout",
                    "destination stride exceeds the CubeCL i64 metadata limit",
                )
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let offset = i64::try_from(output.offset()).map_err(|_| {
        crate::Error::invalid_argument(
            FILL_ZERO_OP,
            "layout",
            "destination offset exceeds the CubeCL i64 metadata limit",
        )
    })?;
    let rank = dims.len();
    let count = dispatch::cube_count_for_len(len)?;
    let dim = dispatch::cube_dim_1d();
    let output_arg = dispatch::typed_view_mut_array_arg(output, FILL_ZERO_OP)?;
    unsafe {
        // SAFETY: the array binding covers the destination's whole root
        // allocation, the view layout was validated as reachable and injective
        // when the view was created, and the launch domain visits each logical
        // coordinate of the region exactly once.
        crate::kernels::structural::fill_zero_view_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
            rt.client(),
            count,
            dim,
            output_arg,
            dispatch::comptime_sequence(&dims),
            dispatch::comptime_sequence(&strides),
            offset,
            len,
            rank,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
