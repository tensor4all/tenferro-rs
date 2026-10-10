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
//! // The value and the buffer are moved into the transfer; `wait` hands them back.
//! let (_value, filled) = download_pending(&runtime, value, buffer)
//!     .map_err(|err| err.to_string())?
//!     .wait()
//!     .map_err(|err| err.to_string())?;
//! assert_eq!(filled.len(), bytes);
//! # Ok(())
//! # }
//! ```

use cubecl::prelude::CubeElement;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent, CUevent_flags, CUresult, CUstream};
use cudarc::runtime::sys as cuda_sys;

use tenferro_tensor::{DType, Tensor, TensorScalar};

use super::dispatch;
use super::runtime::{CudaRuntime, CudaRuntimeState};
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
    /// Weak so that a buffer held by the runtime's own retirement queue cannot keep the runtime
    /// alive: without this, dropping every external handle would leave the queue's records owning a
    /// strong reference back into the state that owns them.
    runtime: std::sync::Weak<CudaRuntimeState>,
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
            runtime: runtime.weak_state(),
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
        let Some(state) = self.runtime.upgrade() else {
            // The runtime is already gone, so its context cannot be selected; the pointer is leaked
            // rather than handed to a driver call that would free the wrong device's memory.
            return;
        };
        if state.set_current_cuda_context(OP).is_err() {
            return;
        }
        // SAFETY: this value owns exactly one live `cudaHostAlloc` allocation, and the context that
        // created it is current.
        if let Err(err) = unsafe { cuda_sys::cudaFreeHost(self.ptr.cast()) }.result() {
            report_pinned_release_error(&err);
        }
    }
}

/// One detached transfer awaiting a completion witness.
///
/// Dropping a pending handle *detaches observation*: the record keeps the moved value, the pinned
/// buffer, the allocation retention and the provider pin until the event proves completion. Nothing
/// here is observable to the caller again, which is what makes `mem::forget` and a dropped handle
/// safe for the admitted package.
pub(crate) struct TransferRetirement {
    event: CUevent,
    source: Option<Tensor>,
    buffer: Option<PinnedHostBuffer>,
    resource: Option<Box<dyn std::any::Any + Send>>,
}

impl TransferRetirement {
    /// Release the record once, after its event completed.
    fn release(mut self, runtime: &CudaRuntimeState) {
        let _ = runtime.set_current_cuda_context(OP);
        // SAFETY: the event is live and owned by this record, and its work has completed.
        let _ = unsafe { cuda_result::event::destroy(self.event) };
        self.source.take();
        self.buffer.take();
        self.resource.take();
    }
}

/// Transfers whose caller detached observation, bounded like the workspace queue.
pub(crate) const DEFAULT_TRANSFER_RETIREMENT_CAPACITY: usize = 16;

#[derive(Default)]
pub(crate) struct TransferRetirementQueue {
    entries: Vec<TransferRetirement>,
    leaked: Vec<TransferRetirement>,
}

impl TransferRetirementQueue {
    /// Release every record whose transfer has completed. Never waits for the device.
    pub(crate) fn drain(&mut self, runtime: &CudaRuntimeState) {
        let mut index = 0;
        while index < self.entries.len() {
            let completed = runtime.set_current_cuda_context(OP).is_ok()
                // SAFETY: the event is live and owned by this entry.
                && unsafe { cuda_result::event::query(self.entries[index].event) }.is_ok();
            if completed {
                self.entries.remove(index).release(runtime);
            } else {
                index += 1;
            }
        }
    }

    /// Take ownership of a detached record, keeping the queue bounded.
    ///
    /// A completed record is released, so the queue normally drains as work finishes. At capacity the
    /// queue drains once; a record that is still in flight when the queue stays full is retained
    /// permanently rather than waited for — dropping a handle never blocks on the device, and the
    /// storage-ownership contract prefers permanent retention over releasing bindings whose completion
    /// is unknown.
    pub(crate) fn detach(&mut self, runtime: &CudaRuntimeState, record: TransferRetirement) {
        if self.entries.len() >= DEFAULT_TRANSFER_RETIREMENT_CAPACITY {
            self.drain(runtime);
        }
        if self.entries.len() >= DEFAULT_TRANSFER_RETIREMENT_CAPACITY {
            self.leaked.push(record);
            return;
        }
        self.entries.push(record);
    }

    /// Retain a record permanently: its completion was never proven, or the event was never recorded,
    /// so no later query may be trusted to release it.
    pub(crate) fn retain_for_soundness(&mut self, record: TransferRetirement) {
        self.leaked.push(record);
    }

    /// Number of records still waiting for completion, including the permanently retained ones.
    pub(crate) fn pending(&self) -> usize {
        self.entries.len() + self.leaked.len()
    }
}

/// Pre-admission rejection: nothing was enqueued, so the caller keeps what it submitted.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::TransferRejected;
///
/// fn describe(rejected: TransferRejected) -> String {
///     let bytes = rejected.buffer.len();
///     format!("{} after {} usable bytes were refused", rejected.error, bytes)
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub struct TransferRejected {
    /// Why the transfer was refused.
    pub error: crate::Error,
    /// The submitted value, unchanged.
    pub source: Tensor,
    /// The submitted pinned buffer, unchanged.
    pub buffer: PinnedHostBuffer,
}

/// A submission that could not be admitted.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::TransferSubmitError;
///
/// fn describe(error: TransferSubmitError) -> String {
///     match error {
///         TransferSubmitError::Rejected(rejected) => rejected.error.to_string(),
///         TransferSubmitError::Unproven { error } => error.to_string(),
///     }
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub enum TransferSubmitError {
    /// No enqueue happened: the caller gets the submitted package back.
    Rejected(Box<TransferRejected>),
    /// The enqueue may have happened, so the package stays provider-private: it is neither returned
    /// nor released, and the runtime retains it until its event proves completion.
    Unproven {
        /// The failure that prevented completion from being proven.
        error: crate::Error,
    },
}

impl std::fmt::Display for TransferSubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(rejected) => write!(formatter, "transfer rejected: {}", rejected.error),
            Self::Unproven { error } => {
                write!(formatter, "completion could not be proven: {error}")
            }
        }
    }
}

impl std::error::Error for TransferSubmitError {}

/// Completion could not be proven: diagnostics only, and no owner is returned.
///
/// The runtime retains the moved value, the pinned buffer, the allocation retention and the event
/// permanently, because neither releasing nor reusing them is known to be safe.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::CompletionUnproven;
///
/// fn describe(unproven: CompletionUnproven) -> String {
///     format!("no owner returned: {}", unproven.error)
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub struct CompletionUnproven {
    /// The failure that prevented completion from being proven.
    pub error: crate::Error,
}

impl std::fmt::Display for CompletionUnproven {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "completion could not be proven: {}", self.error)
    }
}

impl std::error::Error for CompletionUnproven {}

/// A device→host copy in flight.
///
/// Owns the submitted value, the pinned destination, the allocation retention, the provider pin and
/// the copy's event. [`Self::wait`] is the only way to get the value and the buffer back; dropping
/// the handle detaches observation and leaves them with the runtime's retirement queue.
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
/// let mut pending = download_pending(&runtime, value, buffer).map_err(|err| err.to_string())?;
/// if pending.is_ready()? {
///     let (_value, filled) = pending.wait().map_err(|err| err.to_string())?;
///     assert_eq!(filled.len(), bytes);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PendingDownload {
    runtime: CudaRuntime,
    source: Option<Tensor>,
    buffer: Option<PinnedHostBuffer>,
    event: CUevent,
    resource: Option<Box<dyn std::any::Any + Send>>,
    /// Whether this handle still owns the live event. A resolved transfer has destroyed it, and a
    /// detached one has handed it to the retirement queue, so neither may touch it again.
    live: bool,
}

impl PendingDownload {
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
    /// let bytes = value.shape().iter().product::<usize>() * 8;
    /// let buffer = PinnedHostBuffer::new(&runtime, bytes)?;
    /// let mut pending = download_pending(&runtime, value, buffer).map_err(|err| err.to_string())?;
    /// let _ready: bool = pending.is_ready()?;
    /// let (_value, _filled) = pending.wait().map_err(|err| err.to_string())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_ready(&mut self) -> crate::Result<bool> {
        self.runtime.drain_transfer_retirements();
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

    /// Wait for the copy and return the submitted value and the filled pinned buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionUnproven`] when the event synchronization reports a device failure: the
    /// submitted value and buffer are then retained by the runtime, neither returned nor released.
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
    /// let (value, filled) = download_pending(&runtime, value, buffer)
    ///     .map_err(|err| err.to_string())?
    ///     .wait()
    ///     .map_err(|err| err.to_string())?;
    /// assert_eq!(filled.as_slice().len(), bytes);
    /// assert_eq!(value.shape().iter().product::<usize>() * 8, bytes);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait(mut self) -> Result<(Tensor, PinnedHostBuffer), CompletionUnproven> {
        if let Err(error) = self.resolve() {
            // Unproven completion is a terminal outcome: the record is retained for soundness, never
            // handed to the drainable queue where a later query could release it.
            self.retain_terminal();
            return Err(CompletionUnproven { error });
        }
        match (self.source.take(), self.buffer.take()) {
            (Some(source), Some(buffer)) => Ok((source, buffer)),
            _ => Err(CompletionUnproven {
                error: crate::Error::runtime_state(OP, "the pending download already completed"),
            }),
        }
    }

    /// Block until the event completes, then release the retention exactly once.
    fn resolve(&mut self) -> crate::Result<()> {
        if self.source.is_none() {
            return Ok(());
        }
        self.runtime.set_current_cuda_context(OP)?;
        // SAFETY: the event is live and owned by this handle.
        unsafe { cuda_result::event::synchronize(self.event) }
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        // Completion is proven from here on: the retention and the event are released once. A
        // teardown failure below is a diagnostic, not an unproven completion.
        self.resource.take();
        // SAFETY: the event is live, owned, and no longer needed.
        let _ = unsafe { cuda_result::event::destroy(self.event) };
        self.live = false;
        Ok(())
    }

    /// Hand the record to the runtime's retirement queue.
    /// Retain the record permanently, without touching the event again.
    fn retain_terminal(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        self.runtime
            .retain_transfer_for_soundness(TransferRetirement {
                event: self.event,
                source: self.source.take(),
                buffer: self.buffer.take(),
                resource: self.resource.take(),
            });
    }

    fn detach(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        let record = TransferRetirement {
            event: self.event,
            source: self.source.take(),
            buffer: self.buffer.take(),
            resource: self.resource.take(),
        };
        self.runtime.detach_transfer(record);
    }
}

impl Drop for PendingDownload {
    fn drop(&mut self) {
        // Detached drop: no device wait here. The retirement queue releases the record once its event
        // proves completion, and retains it permanently when it cannot.
        self.detach();
    }
}

/// Report a pinned-host release failure without unwinding from `Drop`.
fn report_pinned_release_error(error: &impl std::fmt::Debug) {
    eprintln!("tenferro-gpu: failed to release pinned host buffer during Drop: {error:?}");
}

/// Refuse a transfer before anything is enqueued, returning the submitted package.
fn rejected(error: crate::Error, source: Tensor, buffer: PinnedHostBuffer) -> TransferSubmitError {
    // Boxed so the error channel stays small on the common success path; the rejected package itself
    // is carried unchanged.
    TransferSubmitError::Rejected(Box::new(TransferRejected {
        error,
        source,
        buffer,
    }))
}

/// Start a device→host copy, taking ownership of `source` and `buffer`.
///
/// The copy is published through CubeCL's `check_errors` (producer errors reported, host queue
/// dispatched, no drop-queue rotation) and the source allocation is retained for the whole flight.
/// `wait` is the only way to get the value and the buffer back.
///
/// # Errors
///
/// Returns [`TransferSubmitError::Rejected`] when the transfer is refused before anything is
/// enqueued — an unserved dtype, a buffer whose length does not match, a producer failure reported at
/// the publication boundary or an unresolvable allocation — and carries the
/// unchanged value and buffer back with it. Returns [`TransferSubmitError::Unproven`] when the copy
/// may have been enqueued but completion could not be established; the value, buffer, retention and
/// event then stay provider-private and are never returned or released.
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
/// let filled = download_pending(&runtime, value, buffer)
///     .map_err(|err| err.to_string())?
///     .wait()
///     .map_err(|err| err.to_string())?
///     .1;
/// assert_eq!(filled.as_slice().len(), bytes);
/// # Ok(())
/// # }
/// ```
pub fn download_pending(
    runtime: &CudaRuntime,
    source: Tensor,
    buffer: PinnedHostBuffer,
) -> Result<PendingDownload, TransferSubmitError> {
    // Every submission drains what earlier detachments left behind, so a caller that never polls or
    // synchronizes still bounds the queue.
    runtime.drain_transfer_retirements();
    let (handle, byte_len) = match device_allocation(&source) {
        Ok(allocation) => allocation,
        Err(error) => return Err(rejected(error, source, buffer)),
    };
    if byte_len != buffer.len() {
        return Err(rejected(
            crate::Error::invalid_argument(
                OP,
                "buffer",
                format!(
                    "the pinned buffer holds {} bytes but the source holds {byte_len}",
                    buffer.len()
                ),
            ),
            source,
            buffer,
        ));
    }

    if let Err(error) = runtime.check_errors(OP) {
        return Err(rejected(error, source, buffer));
    }
    let resource = match runtime.client().get_resource(handle) {
        Ok(resource) => resource,
        Err(err) => {
            return Err(rejected(
                crate::Error::backend_source(OP, err),
                source,
                buffer,
            ));
        }
    };
    let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
    if available < byte_len {
        return Err(rejected(
            crate::Error::Internal(format!(
                "{OP}: download of {byte_len} bytes exceeds the {available}-byte allocation"
            )),
            source,
            buffer,
        ));
    }

    // Past this point an enqueue may happen, so failures retain the package instead of returning it.
    let stream = match runtime
        .set_current_cuda_context(OP)
        .and_then(|()| runtime.raw_cuda_stream())
    {
        Ok(stream) => stream as usize as CUstream,
        Err(error) => return Err(rejected(error, source, buffer)),
    };
    let event = match cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING) {
        Ok(event) => event,
        Err(err) => {
            return Err(rejected(
                crate::Error::backend_source(OP, err),
                source,
                buffer,
            ))
        }
    };
    let src = resource.resource().ptr;
    // SAFETY: `src` is the device address of a live CubeCL allocation of at least `byte_len` bytes
    // (checked above) retained by `resource`; `buffer` owns `byte_len` bytes of pinned host memory;
    // `stream` is the CubeCL stream the publication boundary ordered the allocation on.
    let enqueued = unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(buffer.as_mut_ptr().cast(), src, byte_len, stream)
    }
    .result()
    .and_then(|()| unsafe { cuda_result::event::record(event, stream) });

    if let Err(err) = enqueued {
        // The event may not have been recorded, and an unrecorded event queries as complete, so this
        // record must never be released by a later drain: it is retained for soundness.
        runtime.retain_transfer_for_soundness(TransferRetirement {
            event,
            source: Some(source),
            buffer: Some(buffer),
            resource: Some(Box::new(resource)),
        });
        return Err(TransferSubmitError::Unproven {
            error: crate::Error::backend_source(OP, err),
        });
    }

    Ok(PendingDownload {
        runtime: runtime.clone(),
        source: Some(source),
        buffer: Some(buffer),
        event,
        resource: Some(Box::new(resource)),
        live: true,
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
    runtime.drain_transfer_retirements();
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
