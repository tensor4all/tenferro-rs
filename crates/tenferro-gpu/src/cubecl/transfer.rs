//! Pending device→host handoff with owned pinned storage (#1945 U4, slice 1).
//!
//! This is the first slice of the asynchronous transfer package. It owns everything one copy needs
//! — the destination pinned buffer, the source allocation retention, the provider pin and a CUDA
//! event — and hands the filled buffer back only from [`PendingDownload::wait`], so a CPU read of a
//! buffer the device may still be writing is unrepresentable.
//!
//! What it deliberately does **not** claim (see `docs/design/asynchronous-transfer-1945-u4.md`):
//! the submission itself uses the existing audited interop boundary and is not advertised as
//! wait-free, there is no dependent-device-consumer token, and there is no source-publication or
//! cross-domain route.
//!
//! # Examples
//!
//! ```no_run
//! use tenferro_gpu::cuda::{download_pending, PinnedHostBuffer};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
//! # let value: tenferro_tensor::Tensor = unimplemented!();
//! let bytes = value.shape().iter().product::<usize>() * 8;
//! let buffer = PinnedHostBuffer::new(&runtime, bytes)?;
//! let pending = download_pending(&runtime, &value, buffer)?;
//! let filled = pending.wait()?;
//! assert_eq!(filled.len(), bytes);
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use cubecl::prelude::CubeElement;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent, CUevent_flags, CUresult, CUstream};
use cudarc::runtime::sys as cuda_sys;

use tenferro_tensor::{DType, Tensor, TensorScalar};

use super::dispatch;
use super::runtime::CudaRuntime;
use num_complex::{Complex32, Complex64};

const OP: &str = "download_pending";

/// Owned pinned host storage for a pending transfer.
///
/// The buffer belongs to the transfer that fills it and comes back from
/// [`PendingDownload::wait`], so no caller borrow has to outlive a DMA it cannot control. It is
/// `!Send + !Sync`: the allocation is bound to the CUDA context that created it.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::PinnedHostBuffer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// let buffer = PinnedHostBuffer::new(&runtime, 1024)?;
/// assert_eq!(buffer.len(), 1024);
/// # Ok(())
/// # }
/// ```
pub struct PinnedHostBuffer {
    runtime: CudaRuntime,
    ptr: *mut u8,
    len: usize,
    abandoned: bool,
}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedHostBuffer")
            .field("len", &self.len)
            .field("abandoned", &self.abandoned)
            .finish_non_exhaustive()
    }
}

impl PinnedHostBuffer {
    /// Allocate `bytes` of pinned host memory on `runtime`'s device.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidArgument`] for a zero length, and
    /// [`crate::Error::BackendSource`] when the CUDA context cannot be selected or the host
    /// allocation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::PinnedHostBuffer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// let buffer = PinnedHostBuffer::new(&runtime, 64)?;
    /// assert_eq!(buffer.len(), 64);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(runtime: &CudaRuntime, bytes: usize) -> crate::Result<Self> {
        if bytes == 0 {
            return Err(crate::Error::invalid_argument(
                OP,
                "bytes",
                "a pinned host buffer must be non-empty",
            ));
        }
        runtime.set_current_cuda_context(OP)?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the primary context is current on this thread, and the allocation is released by
        // `Drop` (or deliberately abandoned) before the runtime it belongs to is dropped.
        unsafe { cuda_sys::cudaHostAlloc(&mut ptr, bytes, cuda_sys::cudaHostAllocDefault) }
            .result()
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        Ok(Self {
            runtime: runtime.clone(),
            ptr: ptr.cast::<u8>(),
            len: bytes,
            abandoned: false,
        })
    }

    /// Length of the allocation in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the allocation is empty. A buffer is never empty in practice; this exists for the
    /// usual `len`/`is_empty` pairing.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Fill or inspect the buffer before it is handed to a transfer, and read it after the
    /// transfer returned it.
    ///
    /// A transfer owns the buffer while it is in flight, so this is only reachable when no copy can
    /// still be using it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::PinnedHostBuffer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// let mut buffer = PinnedHostBuffer::new(&runtime, 8)?;
    /// buffer.as_mut_slice().copy_from_slice(&1.0_f64.to_ne_bytes());
    /// assert_eq!(buffer.as_slice(), 1.0_f64.to_ne_bytes());
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is live, owned by this value for `len` bytes, and no transfer
        // holds it (a transfer takes ownership), so the exclusive borrow is not aliased.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Borrow the bytes of a completed transfer.
    ///
    /// The contents are only meaningful after the transfer that filled this buffer completed; a
    /// buffer can only be read from a [`PendingDownload::wait`] result for that reason.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is live and owned by this value for `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Give up ownership without freeing: the device may still be writing here.
    fn abandon(&mut self) {
        self.abandoned = true;
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if self.abandoned || self.ptr.is_null() {
            return;
        }
        if self.runtime.set_current_cuda_context(OP).is_err() {
            // Without the owning context current the pointer cannot be freed safely, so it is
            // leaked rather than handed to a driver call that would fail or free the wrong device.
            return;
        }
        // SAFETY: this value owns exactly one live `cudaHostAlloc` allocation, and the context that
        // created it is current.
        if let Err(err) = unsafe { cuda_sys::cudaFreeHost(self.ptr.cast()) }.result() {
            report_pinned_release_error(&err);
        }
    }
}

/// A device→host copy in flight.
///
/// Owns the destination buffer, the source allocation retention, the provider pin and the copy's
/// event. [`Self::wait`] returns the filled buffer; dropping the handle resolves the copy before
/// releasing anything, and abandons (leaks) the buffer and the retained source when completion
/// cannot be proven — freeing memory the device may still write is never an option.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{download_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let value: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = value.shape().iter().product::<usize>() * 8;
/// let buffer = PinnedHostBuffer::new(&runtime, bytes)?;
/// let mut pending = download_pending(&runtime, &value, buffer)?;
/// if pending.is_ready()? {
///     let filled = pending.wait()?;
///     assert_eq!(filled.len(), bytes);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PendingDownload<'source> {
    runtime: CudaRuntime,
    buffer: Option<PinnedHostBuffer>,
    event: CUevent,
    resource: Option<Box<dyn std::any::Any + Send>>,
    source: PhantomData<&'source Tensor>,
}

impl PendingDownload<'_> {
    /// Whether the copy's event has completed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the event query itself fails for a reason other
    /// than "not ready".
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::{download_pending, PinnedHostBuffer};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// # let value: tenferro_tensor::Tensor = unimplemented!();
    /// let buffer = PinnedHostBuffer::new(&runtime, 8)?;
    /// let mut pending = download_pending(&runtime, &value, buffer)?;
    /// let _ready: bool = pending.is_ready()?;
    /// let filled = pending.wait()?;
    /// assert_eq!(filled.len(), 8);
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_ready(&mut self) -> crate::Result<bool> {
        if self.buffer.is_none() {
            return Ok(true);
        }
        self.runtime.set_current_cuda_context(OP)?;
        // SAFETY: the event is live and owned by this handle.
        match unsafe { cuda_result::event::query(self.event) } {
            Ok(()) => Ok(true),
            Err(err) if err.0 == CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(err) => Err(crate::Error::backend_source(OP, err)),
        }
    }

    /// Wait for the copy and return the filled pinned buffer.
    ///
    /// The buffer's bytes are only readable through this result, which is what keeps a CPU read of
    /// an in-flight buffer unrepresentable.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the context cannot be selected or the event
    /// synchronization fails. Nothing is published on failure: the buffer and the retained source
    /// are abandoned, because a failed copy cannot prove what the device is still touching.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::{download_pending, PinnedHostBuffer};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// # let value: tenferro_tensor::Tensor = unimplemented!();
    /// let buffer = PinnedHostBuffer::new(&runtime, 8)?;
    /// let filled = download_pending(&runtime, &value, buffer)?.wait()?;
    /// assert_eq!(filled.as_slice().len(), 8);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait(mut self) -> crate::Result<PinnedHostBuffer> {
        if let Err(err) = self.resolve() {
            self.abandon();
            return Err(err);
        }
        self.buffer.take().ok_or_else(|| {
            crate::Error::runtime_state(OP, "the pending download already completed")
        })
    }

    /// Block until the event completes and release the retained source.
    fn resolve(&mut self) -> crate::Result<()> {
        if self.buffer.is_none() {
            return Ok(());
        }
        self.runtime.set_current_cuda_context(OP)?;
        // SAFETY: the event is live and owned by this handle.
        unsafe { cuda_result::event::synchronize(self.event) }
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        self.resource.take();
        // SAFETY: the event is live, owned, and no longer needed.
        if let Err(err) = unsafe { cuda_result::event::destroy(self.event) } {
            return Err(crate::Error::backend_source(OP, err));
        }
        Ok(())
    }

    /// Drop the buffer and the retained source without freeing them.
    fn abandon(&mut self) {
        if let Some(mut buffer) = self.buffer.take() {
            buffer.abandon();
        }
        if let Some(resource) = self.resource.take() {
            // The copy's outcome is unknown, so the source allocation is leaked exactly as the
            // existing blocking path leaks it on an unproven barrier.
            std::mem::forget(resource);
        }
    }
}

impl Drop for PendingDownload<'_> {
    fn drop(&mut self) {
        if self.buffer.is_none() {
            return;
        }
        // Resolve before releasing. A blocking resolution here is deliberate: the alternative is
        // freeing host memory the device may still write or leaving the caller with no safe owner.
        if self.resolve().is_err() {
            self.abandon();
        }
    }
}

/// Start a device→host copy into `buffer` and return its pending handle.
///
/// The copy is enqueued after the source's pending CubeCL work through the audited interop
/// submission, which also flushes producer errors, so a failed producer surfaces here instead of
/// being masked by a successful copy. The returned handle borrows `source` for its lifetime, so the
/// source cannot be dropped or mutated while the copy may read it.
///
/// # Errors
///
/// Returns [`crate::Error::InvalidArgument`] when the buffer length does not match the source's
/// byte length, [`crate::Error::Unsupported`] for a dtype this route does not serve,
/// [`crate::Error::RuntimeState`] when the source is not a CubeCL device buffer, and
/// [`crate::Error::BackendSource`] when the submission, the resource lookup or the copy fails.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{download_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let value: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = value.shape().iter().product::<usize>() * 8;
/// let buffer = PinnedHostBuffer::new(&runtime, bytes)?;
/// let filled = download_pending(&runtime, &value, buffer)?.wait()?;
/// assert_eq!(filled.as_slice().len(), bytes);
/// # Ok(())
/// # }
/// ```
pub fn download_pending<'source>(
    runtime: &CudaRuntime,
    source: &'source Tensor,
    buffer: PinnedHostBuffer,
) -> crate::Result<PendingDownload<'source>> {
    let (handle, byte_len) = device_allocation(source)?;
    if byte_len != buffer.len() {
        return Err(crate::Error::invalid_argument(
            OP,
            "buffer",
            format!(
                "the pinned buffer holds {} bytes but the source holds {byte_len}",
                buffer.len()
            ),
        ));
    }

    // Publication boundary: report the producer's failures and dispatch every task queued before
    // this point to the server thread, without retiring staged bytes — so the producer's kernels are
    // on the stream when the raw copy below is enqueued and this call does not wait for device
    // progress. The caller's own event is the completion witness.
    runtime.check_errors(OP)?;
    let resource = runtime
        .client()
        .get_resource(handle)
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
    if available < byte_len {
        return Err(crate::Error::Internal(format!(
            "{OP}: download of {byte_len} bytes exceeds the {available}-byte allocation"
        )));
    }

    runtime.set_current_cuda_context(OP)?;
    let stream = runtime.raw_cuda_stream()? as usize as CUstream;
    // SAFETY: the context is current, and the event is created, recorded and destroyed only while
    // this handle owns it.
    let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    let src = resource.resource().ptr;
    // SAFETY: `src` is the device address of a live CubeCL allocation of at least `byte_len` bytes
    // (checked above) retained by `resource`; `buffer` owns `byte_len` bytes of pinned host memory;
    // `stream` is the CubeCL stream `get_resource` ordered the allocation on.
    let enqueued = unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(buffer.as_mut_ptr().cast(), src, byte_len, stream)
    }
    .result()
    .and_then(|()| unsafe { cuda_result::event::record(event, stream) });

    if let Err(err) = enqueued {
        // Nothing can prove what the device is still touching, so the buffer and the source
        // allocation are abandoned instead of being freed early.
        let mut buffer = buffer;
        buffer.abandon();
        std::mem::forget(resource);
        let _ = unsafe { cuda_result::event::destroy(event) };
        return Err(crate::Error::backend_source(OP, err));
    }

    Ok(PendingDownload {
        runtime: runtime.clone(),
        buffer: Some(buffer),
        event,
        resource: Some(Box::new(resource)),
        source: PhantomData,
    })
}

/// The source's CubeCL allocation handle and its byte length.
fn device_allocation(source: &Tensor) -> crate::Result<(cubecl_runtime::server::Handle, usize)> {
    match source.dtype() {
        DType::F64 => device_allocation_typed::<f64>(source),
        DType::F32 => device_allocation_typed::<f32>(source),
        DType::I64 => device_allocation_typed::<i64>(source),
        DType::I32 => device_allocation_typed::<i32>(source),
        DType::C64 => device_allocation_typed::<Complex64>(source),
        DType::C32 => device_allocation_typed::<Complex32>(source),
        other => Err(crate::Error::unsupported(
            OP,
            format!("the pending download route does not serve {other:?}"),
        )),
    }
}

fn device_allocation_typed<T: CubeElement + TensorScalar + 'static>(
    source: &Tensor,
) -> crate::Result<(cubecl_runtime::server::Handle, usize)> {
    let typed = source
        .as_typed::<T>()
        .ok_or_else(|| crate::Error::unsupported(OP, "expected a preset numeric scalar"))?;
    let buffer = dispatch::cubecl_buffer(typed, OP)?;
    let bytes = typed.n_elements().saturating_mul(size_of::<T>());
    Ok((buffer.handle().clone(), bytes))
}

/// Report a pinned-host release failure without unwinding from `Drop`.
fn report_pinned_release_error(error: &impl std::fmt::Debug) {
    eprintln!("tenferro-gpu: failed to release pinned host buffer during Drop: {error:?}");
}

/// A pinned host→device copy in flight.
///
/// Owns the source buffer, the provider pin, the source allocation retention and the copy's event,
/// and holds the destination tensor **mutably** for its lifetime, so the device tensor cannot be
/// read, mutated or dropped while the copy may still write it. [`Self::wait`] returns the source
/// buffer for reuse; the destination borrow ends when the handle is consumed.
///
/// The mutable borrow is enforced by the compiler (see
/// `tests/ui/cuda_pending_upload_borrows_the_destination.rs`), so a caller cannot read, mutate or
/// drop the device tensor while the copy is in flight.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let mut device: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = device.shape().iter().product::<usize>() * 8;
/// let source = PinnedHostBuffer::new(&runtime, bytes)?;
/// let mut pending = upload_pending(&runtime, source, &mut device)?;
/// if pending.is_ready()? {
///     let reusable = pending.wait()?;
///     assert_eq!(reusable.len(), bytes);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PendingUpload<'dst> {
    runtime: CudaRuntime,
    source: Option<PinnedHostBuffer>,
    event: CUevent,
    resource: Option<Box<dyn std::any::Any + Send>>,
    destination: Option<&'dst mut Tensor>,
}

impl PendingUpload<'_> {
    /// Whether the copy's event has completed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the event query itself fails for a reason other
    /// than "not ready".
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// # let mut device: tenferro_tensor::Tensor = unimplemented!();
    /// let bytes = device.shape().iter().product::<usize>() * 8;
    /// let source = PinnedHostBuffer::new(&runtime, bytes)?;
    /// let mut pending = upload_pending(&runtime, source, &mut device)?;
    /// let _ready: bool = pending.is_ready()?;
    /// let _ = pending.wait()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_ready(&mut self) -> crate::Result<bool> {
        if self.source.is_none() {
            return Ok(true);
        }
        self.runtime.set_current_cuda_context(OP)?;
        // SAFETY: the event is live and owned by this handle.
        match unsafe { cuda_result::event::query(self.event) } {
            Ok(()) => Ok(true),
            Err(err) if err.0 == CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(err) => Err(crate::Error::backend_source(OP, err)),
        }
    }

    /// Wait for the copy and return the source buffer for reuse.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the context cannot be selected or the event
    /// synchronization fails. On failure the destination must be discarded: an unproven copy
    /// cannot establish that its contents are valid.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// # let mut device: tenferro_tensor::Tensor = unimplemented!();
    /// let bytes = device.shape().iter().product::<usize>() * 8;
    /// let source = PinnedHostBuffer::new(&runtime, bytes)?;
    /// let reusable = upload_pending(&runtime, source, &mut device)?.wait()?;
    /// assert_eq!(reusable.len(), bytes);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait(mut self) -> crate::Result<PinnedHostBuffer> {
        if let Err(err) = self.resolve() {
            self.abandon();
            return Err(err);
        }
        self.source
            .take()
            .ok_or_else(|| crate::Error::runtime_state(OP, "the pending upload already completed"))
    }

    /// Block until the event completes and release the retention.
    fn resolve(&mut self) -> crate::Result<()> {
        if self.source.is_none() {
            return Ok(());
        }
        self.runtime.set_current_cuda_context(OP)?;
        // SAFETY: the event is live and owned by this handle.
        unsafe { cuda_result::event::synchronize(self.event) }
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        self.resource.take();
        // SAFETY: the event is live, owned, and no longer needed.
        if let Err(err) = unsafe { cuda_result::event::destroy(self.event) } {
            return Err(crate::Error::backend_source(OP, err));
        }
        Ok(())
    }

    /// Drop the source buffer without freeing it.
    fn abandon(&mut self) {
        if let Some(mut buffer) = self.source.take() {
            buffer.abandon();
        }
        if let Some(resource) = self.resource.take() {
            std::mem::forget(resource);
        }
    }
}

impl Drop for PendingUpload<'_> {
    fn drop(&mut self) {
        if self.source.is_none() {
            return;
        }
        // Resolve before releasing the destination borrow, so the caller never receives a tensor
        // whose producer is still running.
        if self.resolve().is_err() {
            self.abandon();
        }
        self.destination.take();
    }
}

/// Start a pinned host→device copy into `destination` and return its pending handle.
///
/// The copy is enqueued through the same audited interop submission as
/// [`download_pending`], and `destination` is borrowed mutably for the handle's lifetime: a device
/// consumer reads it through the caller after [`PendingUpload::wait`], never while the copy is in
/// flight. Strided views are not accepted; a caller holding one materializes it first.
///
/// # Errors
///
/// Returns [`crate::Error::InvalidArgument`] when the source buffer length does not match the
/// destination's byte length, [`crate::Error::Unsupported`] for a dtype this route does not serve,
/// [`crate::Error::RuntimeState`] when the destination is not a CubeCL device buffer, and
/// [`crate::Error::BackendSource`] when the submission, the resource lookup or the copy fails.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let mut device: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = device.shape().iter().product::<usize>() * 8;
/// let source = PinnedHostBuffer::new(&runtime, bytes)?;
/// let reusable = upload_pending(&runtime, source, &mut device)?.wait()?;
/// assert_eq!(reusable.len(), bytes);
/// # Ok(())
/// # }
/// ```
pub fn upload_pending<'dst>(
    runtime: &CudaRuntime,
    source: PinnedHostBuffer,
    destination: &'dst mut Tensor,
) -> crate::Result<PendingUpload<'dst>> {
    let (handle, byte_len) = device_allocation(destination)?;
    if byte_len != source.len() {
        return Err(crate::Error::invalid_argument(
            OP,
            "source",
            format!(
                "the pinned buffer holds {} bytes but the destination holds {byte_len}",
                source.len()
            ),
        ));
    }

    // Publication boundary: the same non-blocking check as `download_pending`, so pending work on
    // the destination is on the stream before the raw copy without waiting for device progress.
    runtime.check_errors(OP)?;
    let resource = runtime
        .client()
        .get_resource(handle)
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
    if available < byte_len {
        return Err(crate::Error::Internal(format!(
            "{OP}: upload of {byte_len} bytes exceeds the {available}-byte allocation"
        )));
    }

    runtime.set_current_cuda_context(OP)?;
    let stream = runtime.raw_cuda_stream()? as usize as CUstream;
    let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    let dst = resource.resource().ptr;
    // SAFETY: `dst` is the device address of a live CubeCL allocation of at least `byte_len` bytes
    // (checked above); `source` owns `byte_len` bytes of pinned host memory; `stream` is the CubeCL
    // stream `get_resource` ordered the allocation on.
    let enqueued = unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, source.as_mut_ptr().cast(), byte_len, stream)
    }
    .result()
    .and_then(|()| unsafe { cuda_result::event::record(event, stream) });

    if let Err(err) = enqueued {
        let mut source = source;
        source.abandon();
        std::mem::forget(resource);
        let _ = unsafe { cuda_result::event::destroy(event) };
        return Err(crate::Error::backend_source(OP, err));
    }

    Ok(PendingUpload {
        runtime: runtime.clone(),
        source: Some(source),
        event,
        resource: Some(Box::new(resource)),
        destination: Some(destination),
    })
}
