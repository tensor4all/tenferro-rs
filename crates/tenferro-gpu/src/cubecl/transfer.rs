//! Pending transfers, and the publication of a produced value (#1945 U4).
//!
//! The pending routes move a value between host and device with owned admission; `exec_session`'s
//! `CudaHeldSession::publish` builds a [`SourcePublication`], which keeps an owned value, its retained
//! allocation address and a frontier recorded on the producing session's stream, so a stage can publish
//! now and consume after its session and backend handle are gone.
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
use cubecl::stream_id::StreamId;
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
}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedHostBuffer")
            .field("len", &self.len)
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
        // `Drop` before the runtime it belongs to is dropped; a transfer that owns this buffer keeps
        // it alive for as long as the copy may need it.
        unsafe { cuda_sys::cudaHostAlloc(&mut ptr, bytes, cuda_sys::cudaHostAllocDefault) }
            .result()
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        Ok(Self {
            runtime: runtime.weak_state(),
            ptr: ptr.cast::<u8>(),
            len: bytes,
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
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
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
    value: Option<Tensor>,
    buffer: Option<PinnedHostBuffer>,
    resource: Option<Box<dyn std::any::Any + Send>>,
}

impl TransferRetirement {
    /// Release the record once, after its event completed.
    fn release(mut self, runtime: &CudaRuntimeState) {
        let _ = runtime.set_current_cuda_context(OP);
        // SAFETY: the event is live and owned by this record, and its work has completed.
        let _ = unsafe { cuda_result::event::destroy(self.event) };
        self.value.take();
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

    /// Forget every retained and in-flight record, so that runtime field destruction cannot free one.
    pub(crate) fn forget_all(&mut self) {
        for record in self.leaked.drain(..) {
            std::mem::forget(record);
        }
        for record in self.entries.drain(..) {
            std::mem::forget(record);
        }
    }

    /// Final handling at runtime teardown.
    ///
    /// Records whose completion was never proven (or whose event was never recorded) are retained
    /// permanently: `mem::forget` keeps their tensors, buffers, retention and events alive instead of
    /// letting the runtime's own destruction free memory the device may still touch. Records that are
    /// still merely in flight are released best effort, and their events destroyed, because the
    /// runtime has already retired its streams by this point.
    pub(crate) fn release_at_teardown(&mut self, runtime: &CudaRuntimeState) {
        self.drain(runtime);
        for record in self.entries.drain(..) {
            record.release(runtime);
        }
        for record in self.leaked.drain(..) {
            std::mem::forget(record);
        }
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
    pub value: Tensor,
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
                value: self.source.take(),
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
            value: self.source.take(),
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
fn rejected(error: crate::Error, value: Tensor, buffer: PinnedHostBuffer) -> TransferSubmitError {
    // Boxed so the error channel stays small on the common success path; the rejected package itself
    // is carried unchanged.
    TransferSubmitError::Rejected(Box::new(TransferRejected {
        error,
        value,
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
            value: Some(source),
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
/// Owns the source buffer, the destination value, the provider pin, the allocation retention and the
/// copy's event. [`Self::wait`] is the only way to get them back; dropping the handle detaches the
/// record into the runtime's retirement queue, which releases it once its event proves completion and
/// retains it otherwise.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let device: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = device.shape().iter().product::<usize>() * 8;
/// let source = PinnedHostBuffer::new(&runtime, bytes)?;
/// let mut pending = upload_pending(&runtime, source, device).map_err(|err| err.to_string())?;
/// if pending.is_ready()? {
///     let (device, source) = pending.wait().map_err(|err| err.to_string())?;
///     assert_eq!(source.len(), bytes);
///     assert_eq!(device.shape().iter().product::<usize>() * 8, bytes);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PendingUpload {
    runtime: CudaRuntime,
    source: Option<PinnedHostBuffer>,
    destination: Option<Tensor>,
    event: CUevent,
    resource: Option<Box<dyn std::any::Any + Send>>,
    /// Whether this handle still owns the live event (see `PendingDownload`).
    live: bool,
}

impl PendingUpload {
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
    /// # let device: tenferro_tensor::Tensor = unimplemented!();
    /// let bytes = device.shape().iter().product::<usize>() * 8;
    /// let source = PinnedHostBuffer::new(&runtime, bytes)?;
    /// let mut pending = upload_pending(&runtime, source, device).map_err(|err| err.to_string())?;
    /// let _ready: bool = pending.is_ready()?;
    /// let _ = pending.wait().map_err(|err| err.to_string())?;
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

    /// Wait for the copy and return the destination value and the source buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionUnproven`] when the event synchronization reports a device failure: the
    /// destination value and the source buffer are then retained by the runtime, neither returned nor
    /// released.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// # let device: tenferro_tensor::Tensor = unimplemented!();
    /// let bytes = device.shape().iter().product::<usize>() * 8;
    /// let source = PinnedHostBuffer::new(&runtime, bytes)?;
    /// let (device, source) = upload_pending(&runtime, source, device)
    ///     .map_err(|err| err.to_string())?
    ///     .wait()
    ///     .map_err(|err| err.to_string())?;
    /// assert_eq!(source.len(), bytes);
    /// assert_eq!(device.shape().iter().product::<usize>() * 8, bytes);
    /// # Ok(())
    /// # }
    /// ```
    pub fn wait(mut self) -> Result<(Tensor, PinnedHostBuffer), CompletionUnproven> {
        if let Err(error) = self.resolve() {
            self.retain_terminal();
            return Err(CompletionUnproven { error });
        }
        match (self.destination.take(), self.source.take()) {
            (Some(destination), Some(source)) => Ok((destination, source)),
            _ => Err(CompletionUnproven {
                error: crate::Error::runtime_state(OP, "the pending upload already completed"),
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
        self.resource.take();
        // SAFETY: the event is live, owned, and no longer needed.
        let _ = unsafe { cuda_result::event::destroy(self.event) };
        self.live = false;
        Ok(())
    }

    /// Retain the record permanently, without touching the event again.
    fn retain_terminal(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        self.runtime
            .retain_transfer_for_soundness(TransferRetirement {
                event: self.event,
                value: self.destination.take(),
                buffer: self.source.take(),
                resource: self.resource.take(),
            });
    }

    /// Hand the record to the runtime's retirement queue.
    fn detach(&mut self) {
        if !self.live {
            return;
        }
        self.live = false;
        self.runtime.detach_transfer(TransferRetirement {
            event: self.event,
            value: self.destination.take(),
            buffer: self.source.take(),
            resource: self.resource.take(),
        });
    }
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        // Detached drop: no device wait here (see `PendingDownload`).
        self.detach();
    }
}

/// Start a pinned host→device copy, taking ownership of `source` and `destination`.
///
/// The copy is published through the same non-blocking boundary as [`download_pending`] and the
/// destination allocation is retained for the whole flight.
///
/// # Errors
///
/// Returns [`TransferSubmitError::Rejected`] when the transfer is refused before anything is enqueued
/// — an unserved dtype, a host tensor destination, a buffer whose length does not match, a producer
/// failure reported at the publication boundary or an unresolvable allocation — and carries the
/// unchanged buffer and destination back with it. Returns [`TransferSubmitError::Unproven`] when the
/// copy may have been enqueued but completion could not be established.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::{upload_pending, PinnedHostBuffer};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
/// # let device: tenferro_tensor::Tensor = unimplemented!();
/// let bytes = device.shape().iter().product::<usize>() * 8;
/// let source = PinnedHostBuffer::new(&runtime, bytes)?;
/// let (_device, source) = upload_pending(&runtime, source, device)
///     .map_err(|err| err.to_string())?
///     .wait()
///     .map_err(|err| err.to_string())?;
/// assert_eq!(source.len(), bytes);
/// # Ok(())
/// # }
/// ```
pub fn upload_pending(
    runtime: &CudaRuntime,
    source: PinnedHostBuffer,
    destination: Tensor,
) -> Result<PendingUpload, TransferSubmitError> {
    runtime.drain_transfer_retirements();
    let (handle, byte_len) = match device_allocation(&destination) {
        Ok(allocation) => allocation,
        Err(error) => return Err(rejected(error, destination, source)),
    };
    if byte_len != source.len() {
        return Err(rejected(
            crate::Error::invalid_argument(
                OP,
                "source",
                format!(
                    "the pinned buffer holds {} bytes but the destination holds {byte_len}",
                    source.len()
                ),
            ),
            destination,
            source,
        ));
    }

    if let Err(error) = runtime.check_errors(OP) {
        return Err(rejected(error, destination, source));
    }
    let resource = match runtime.client().get_resource(handle) {
        Ok(resource) => resource,
        Err(err) => {
            return Err(rejected(
                crate::Error::backend_source(OP, err),
                destination,
                source,
            ));
        }
    };
    let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
    if available < byte_len {
        return Err(rejected(
            crate::Error::Internal(format!(
                "{OP}: upload of {byte_len} bytes exceeds the {available}-byte allocation"
            )),
            destination,
            source,
        ));
    }

    let stream = match runtime
        .set_current_cuda_context(OP)
        .and_then(|()| runtime.raw_cuda_stream())
    {
        Ok(stream) => stream as usize as CUstream,
        Err(error) => return Err(rejected(error, destination, source)),
    };
    let event = match cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING) {
        Ok(event) => event,
        Err(err) => {
            return Err(rejected(
                crate::Error::backend_source(OP, err),
                destination,
                source,
            ));
        }
    };
    let dst = resource.resource().ptr;
    // SAFETY: `dst` is the device address of a live CubeCL allocation of at least `byte_len` bytes
    // (checked above) retained by `resource`; `source` owns `byte_len` bytes of pinned host memory;
    // `stream` is the CubeCL stream the publication boundary ordered the allocation on.
    let enqueued = unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, source.as_mut_ptr().cast(), byte_len, stream)
    }
    .result()
    .and_then(|()| unsafe { cuda_result::event::record(event, stream) });

    if let Err(err) = enqueued {
        // The event may not have been recorded, and an unrecorded event queries as complete, so this
        // record must never be released by a later drain: it is retained for soundness.
        runtime.retain_transfer_for_soundness(TransferRetirement {
            event,
            value: Some(destination),
            buffer: Some(source),
            resource: Some(Box::new(resource)),
        });
        return Err(TransferSubmitError::Unproven {
            error: crate::Error::backend_source(OP, err),
        });
    }

    Ok(PendingUpload {
        runtime: runtime.clone(),
        source: Some(source),
        destination: Some(destination),
        event,
        resource: Some(Box::new(resource)),
        live: true,
    })
}

/// Refuse a publication before anything is enqueued, returning the value.
fn publication_rejected(error: crate::Error, value: Tensor) -> PublicationAdmissionError {
    PublicationAdmissionError::Rejected(Box::new(PublicationRejected { error, value }))
}

/// A consuming copy that could not be enqueued, or whose publication is terminal.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::PublicationError;
///
/// fn describe(error: PublicationError) -> String {
///     match error {
///         PublicationError::Rejected { error, .. } => error.to_string(),
///         PublicationError::Unproven { error } => error.to_string(),
///     }
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub enum PublicationError {
    /// Nothing was enqueued: the publication and the buffer come back unchanged, so the caller can fix
    /// the buffer and retry, or await the frontier it still owns.
    Rejected {
        /// Why the copy was refused.
        error: crate::Error,
        /// The publication, unchanged.
        publication: Box<SourcePublication>,
        /// The submitted buffer, unchanged.
        buffer: PinnedHostBuffer,
    },
    /// The copy may have been enqueued: nothing is returned, and the value, buffer, allocation and
    /// provider pin stay provider-private.
    Unproven {
        /// The failure that prevented completion from being proven.
        error: crate::Error,
    },
}

impl std::fmt::Display for PublicationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected { error, .. } => write!(formatter, "published copy rejected: {error}"),
            Self::Unproven { error } => write!(formatter, "published copy unproven: {error}"),
        }
    }
}

impl std::error::Error for PublicationError {}

/// A publication that could not be admitted, or whose completion could not be proven.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::PublicationAdmissionError;
///
/// fn describe(error: PublicationAdmissionError) -> String {
///     match error {
///         PublicationAdmissionError::Rejected(rejected) => rejected.error.to_string(),
///         PublicationAdmissionError::Unproven { error } => error.to_string(),
///     }
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub enum PublicationAdmissionError {
    /// Nothing was enqueued, so the caller keeps the value.
    ///
    /// Boxed so the error channel stays small on the common success path; the rejected value itself is
    /// carried unchanged.
    Rejected(Box<PublicationRejected>),
    /// The enqueue may have happened, or the publication already failed terminally: no owner is returned.
    Unproven {
        /// The failure that prevented completion from being proven.
        error: crate::Error,
    },
}

/// A publication that was refused before anything was enqueued.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::PublicationRejected;
///
/// fn describe(rejected: PublicationRejected) -> String {
///     format!("{} after the value of shape {:?} came back", rejected.error, rejected.value.shape())
/// }
/// let _ = describe;
/// ```
#[derive(Debug)]
pub struct PublicationRejected {
    /// Why the publication was refused.
    pub error: crate::Error,
    /// The submitted value, unchanged.
    pub value: Tensor,
}

impl std::fmt::Display for PublicationAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(rejected) => {
                write!(formatter, "publication rejected: {}", rejected.error)
            }
            Self::Unproven { error } => {
                write!(formatter, "publication completion unproven: {error}")
            }
        }
    }
}

impl std::error::Error for PublicationAdmissionError {}

const PUBLICATION_OP: &str = "publish_source";

/// A value whose production has been published: owned, retained, and carrying a producer frontier
/// recorded on the stream that produced it.
///
/// The value is reachable only through [`Self::download_into`]. There is deliberately no public
/// read accessor: an owned `&Tensor` would carry storage lifetime and Rust read authority, but not
/// this publication's readiness, because ordinary CubeCL resource alignment follows the
/// allocation's creating stream rather than this frontier - so a consumer handed that borrow on
/// another stream could read before the producer's work completed. Serving a device consumer needs
/// the dependency contract, which this revision does not specify.
///
/// Dropping a publication that was never consumed detaches its record into the runtime's bounded
/// transfer-retirement queue without waiting, and a terminal publication is retained permanently.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::SourcePublication;
///
/// fn byte_len(publication: &SourcePublication) -> usize {
///     publication.bytes()
/// }
///
///
/// let _ = byte_len;
/// ```
pub struct SourcePublication {
    runtime: CudaRuntime,
    value: Option<Tensor>,
    /// Device address of the published value's storage. The owned value keeps the allocation alive, so
    /// resolution's retention guard is not needed after the address is recorded.
    address: u64,
    bytes: usize,
    stream: StreamId,
    frontier: Option<CUevent>,
    /// Set once a readiness wait failed: the publication is terminal and keeps everything it owns.
    failure: Option<crate::Error>,
    /// Whether the terminal record was already handed to the queue.
    terminal_recorded: bool,
}

impl std::fmt::Debug for SourcePublication {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourcePublication")
            .field("bytes", &self.bytes)
            .field("terminal", &self.failure.is_some())
            .finish_non_exhaustive()
    }
}

impl SourcePublication {
    /// Byte length of the published value.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let publication: tenferro_gpu::cuda::SourcePublication = unimplemented!();
    /// assert_eq!(publication.bytes(), 64);
    /// ```
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether the producer's frontier has completed. Never waits.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] once the publication is terminal, so a failed
    /// publication can never report readiness, and [`crate::Error::BackendSource`] when the event
    /// query itself fails for a reason other than "not ready".
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let publication: tenferro_gpu::cuda::SourcePublication = unimplemented!();
    /// let mut publication = publication;
    /// let _ready: bool = publication.frontier_ready()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn frontier_ready(&mut self) -> crate::Result<bool> {
        self.runtime.drain_transfer_retirements();
        if let Some(failure) = &self.failure {
            return Err(crate::Error::runtime_state(
                PUBLICATION_OP,
                format!("the publication is terminal and can never become ready: {failure}"),
            ));
        }
        let Some(frontier) = self.frontier else {
            return Ok(true);
        };
        self.runtime.set_current_cuda_context(PUBLICATION_OP)?;
        // SAFETY: the event is live and owned by this publication.
        match unsafe { cuda_result::event::query(frontier) } {
            Ok(()) => Ok(true),
            Err(err) if err.0 == CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(err) => {
                // A device failure observed here makes the publication terminal, exactly as a failed
                // synchronization does, so it can never be consumed as if it were ready.
                self.failure = Some(crate::Error::runtime_state(
                    PUBLICATION_OP,
                    "the frontier query reported a device failure",
                ));
                Err(crate::Error::backend_source(PUBLICATION_OP, err))
            }
        }
    }

    /// Explicit readiness boundary: wait for the producer's frontier.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationAdmissionError::Unproven`] when the event synchronization reports a
    /// device failure, or when the publication is already terminal. Nothing is released: the
    /// publication stays terminal and retains its value, allocation, frontier and provider pin until
    /// it is dropped, at which point they are retained permanently.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let mut publication: tenferro_gpu::cuda::SourcePublication = unimplemented!();
    /// publication.await_frontier().map_err(|err| err.to_string())?;
    /// # Ok::<(), String>(())
    /// ```
    pub fn await_frontier(&mut self) -> Result<(), PublicationAdmissionError> {
        if let Some(failure) = &self.failure {
            return Err(PublicationAdmissionError::Unproven {
                error: crate::Error::runtime_state(
                    PUBLICATION_OP,
                    format!("the publication is already terminal: {failure}"),
                ),
            });
        }
        let Some(frontier) = self.frontier else {
            return Ok(());
        };
        let outcome = self
            .runtime
            .set_current_cuda_context(PUBLICATION_OP)
            .and_then(|()| {
                // SAFETY: the event is live and owned by this publication.
                unsafe { cuda_result::event::synchronize(frontier) }
                    .map_err(|err| crate::Error::backend_source(PUBLICATION_OP, err))
            });
        match outcome {
            Ok(()) => Ok(()),
            Err(error) => {
                self.failure = Some(error);
                Err(PublicationAdmissionError::Unproven {
                    error: crate::Error::runtime_state(
                        PUBLICATION_OP,
                        "the publication's completion could not be proven",
                    ),
                })
            }
        }
    }

    /// Consume the publication by copying it into `buffer`.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationError::Rejected`] when nothing was enqueued - the buffer's length does
    /// not match, or the publication is already terminal - carrying the publication and the buffer
    /// back; and [`PublicationError::Unproven`] when the copy may have been enqueued, in which case
    /// the value, buffer, allocation and provider pin are retained and nothing is returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let publication: tenferro_gpu::cuda::SourcePublication = unimplemented!();
    /// # let buffer: tenferro_gpu::cuda::PinnedHostBuffer = unimplemented!();
    /// let pending = publication
    ///     .download_into(buffer)
    ///     .map_err(|err| err.to_string())?;
    /// let (_value, filled) = pending.wait().map_err(|err| err.to_string())?;
    /// assert_eq!(filled.len(), 64);
    /// # Ok::<(), String>(())
    /// ```
    pub fn download_into(
        mut self,
        buffer: PinnedHostBuffer,
    ) -> Result<PendingDownload, PublicationError> {
        if self.failure.is_some() {
            // Nothing was enqueued in this call, so the publication and the buffer both come back.
            return Err(PublicationError::Rejected {
                error: crate::Error::runtime_state(
                    PUBLICATION_OP,
                    "the publication is terminal; its value and frontier are retained, not consumed",
                ),
                publication: Box::new(self),
                buffer,
            });
        }
        if buffer.len() != self.bytes {
            // Nothing is destroyed here: the frontier is what makes the returned publication usable,
            // and dropping it would hand back a value whose readiness is no longer carried.
            return Err(PublicationError::Rejected {
                error: crate::Error::invalid_argument(
                    PUBLICATION_OP,
                    "buffer",
                    format!(
                        "the pinned buffer holds {} bytes but the published value holds {}",
                        buffer.len(),
                        self.bytes
                    ),
                ),
                publication: Box::new(self),
                buffer,
            });
        }

        let stream = self.stream;
        let event = match stream.executes(|| -> crate::Result<CUevent> {
            self.runtime.set_current_cuda_context(PUBLICATION_OP)?;
            let raw_stream = self.runtime.raw_cuda_stream()? as usize as CUstream;
            let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)
                .map_err(|err| crate::Error::backend_source(PUBLICATION_OP, err))?;
            let source = self.address;
            // SAFETY: `source` is the device address of the allocation this publication retained,
            // `buffer` owns `self.bytes` bytes of pinned host memory, and `raw_stream` is the captured
            // stream, applied for the duration of this closure.
            let enqueued = unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    buffer.as_mut_ptr().cast(),
                    source,
                    self.bytes,
                    raw_stream,
                )
            }
            .result()
            .and_then(|()| unsafe { cuda_result::event::record(event, raw_stream) });
            match enqueued {
                Ok(()) => Ok(event),
                Err(err) => {
                    // An event that was never recorded queries as complete, so it must never be
                    // destroyed or handed to a later release: it is left leaked on purpose.
                    Err(crate::Error::backend_source(PUBLICATION_OP, err))
                }
            }
        }) {
            Ok(event) => event,
            Err(error) => {
                self.failure = Some(error);
                // The copy may have been enqueued, so the buffer is retained with the value: releasing
                // host memory the device may still be writing is what the contract forbids.
                self.retain_terminal(Some(buffer));
                return Err(PublicationError::Unproven {
                    error: crate::Error::runtime_state(
                        PUBLICATION_OP,
                        "the consuming copy could not be enqueued",
                    ),
                });
            }
        };

        // The frontier is ordered before the copy on the same stream, so it is no longer needed.
        self.release_frontier();
        Ok(PendingDownload {
            runtime: self.runtime.clone(),
            source: self.value.take(),
            buffer: Some(buffer),
            event,
            resource: None,
            live: true,
        })
    }

    /// Destroy the frontier event, which no one needs once the copy is enqueued.
    fn release_frontier(&mut self) {
        if let Some(frontier) = self.frontier.take() {
            let _ = self.runtime.set_current_cuda_context(PUBLICATION_OP);
            // SAFETY: the event is live, owned by this publication, and no longer needed.
            let _ = unsafe { cuda_result::event::destroy(frontier) };
        }
    }

    /// Hand the record to the permanent-retention side of the queue, exactly once.
    fn retain_terminal(&mut self, buffer: Option<PinnedHostBuffer>) {
        if self.terminal_recorded {
            return;
        }
        self.terminal_recorded = true;
        let record = TransferRetirement {
            event: self.frontier.take().unwrap_or(std::ptr::null_mut()),
            value: self.value.take(),
            buffer,
            resource: None,
        };
        self.runtime.retain_transfer_for_soundness(record);
    }

    /// Hand the record to the runtime's drainable retirement queue.
    fn detach(&mut self) {
        let Some(frontier) = self.frontier.take() else {
            return;
        };
        self.runtime.detach_transfer(TransferRetirement {
            event: frontier,
            value: self.value.take(),
            buffer: None,
            resource: None,
        });
    }
}

impl Drop for SourcePublication {
    fn drop(&mut self) {
        if self.failure.is_some() {
            self.retain_terminal(None);
        } else {
            self.detach();
        }
    }
}

/// Publish `value`, recording its frontier on `stream`.
///
/// The caller must have entered `stream` already, so that the dispatch, the resource resolution and the
/// event record all happen on the stream that produced the value.
pub(crate) fn publish_in_stream(
    runtime: &CudaRuntime,
    stream: StreamId,
    value: Tensor,
) -> Result<SourcePublication, PublicationAdmissionError> {
    runtime.drain_transfer_retirements();
    let (handle, bytes) = match device_allocation(&value) {
        Ok(allocation) => allocation,
        Err(error) => return Err(publication_rejected(error, value)),
    };
    // The dispatch and the resource resolution happen on the stream the caller entered, so the
    // frontier recorded below is ordered after everything that produced the value on that stream.
    if let Err(error) = runtime.check_errors(PUBLICATION_OP) {
        return Err(publication_rejected(error, value));
    }
    let resource = match runtime.client().get_resource(handle) {
        Ok(resource) => resource,
        Err(err) => {
            return Err(publication_rejected(
                crate::Error::backend_source(PUBLICATION_OP, err),
                value,
            ));
        }
    };
    let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
    if available < bytes {
        return Err(publication_rejected(
            crate::Error::Internal(format!(
                "{PUBLICATION_OP}: published value of {bytes} bytes exceeds the {available}-byte allocation"
            )),
            value,
        ));
    }
    let address = resource.resource().ptr;
    // The value keeps the allocation alive, so resolution's guard is released here and only the
    // address is retained.
    drop(resource);

    // Past this point an enqueue may happen: a failure retains the package for soundness.
    let prepared = (|| -> crate::Result<CUevent> {
        runtime.set_current_cuda_context(PUBLICATION_OP)?;
        let raw_stream = runtime.raw_cuda_stream()? as usize as CUstream;
        let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map_err(|err| crate::Error::backend_source(PUBLICATION_OP, err))?;
        // SAFETY: `event` is live and owned by this call, and `raw_stream` is the stream just resolved.
        match unsafe { cuda_result::event::record(event, raw_stream) } {
            Ok(()) => Ok(event),
            Err(err) => {
                // An event that was never recorded queries as complete, so it must never be released:
                // `CUevent` is a raw pointer with no destructor, so leaving it untouched leaks it.
                Err(crate::Error::backend_source(PUBLICATION_OP, err))
            }
        }
    })();
    let event = match prepared {
        Ok(event) => event,
        Err(error) => {
            runtime.retain_transfer_for_soundness(TransferRetirement {
                event: std::ptr::null_mut(),
                value: Some(value),
                buffer: None,
                resource: None,
            });
            return Err(PublicationAdmissionError::Unproven { error });
        }
    };

    Ok(SourcePublication {
        runtime: runtime.clone(),
        value: Some(value),
        address,
        bytes,
        stream,
        frontier: Some(event),
        failure: None,
        terminal_recorded: false,
    })
}
