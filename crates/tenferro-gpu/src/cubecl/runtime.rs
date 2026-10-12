//! CubeCL CUDA runtime initialization and synchronization.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};

use cubecl::client::Client;
use cubecl_environment::stream::StreamId;
use cubecl::Device;
use cubecl_cuda::CudaDevice;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{CUcontext, CUdevice, CUevent, CUresult};
use cudarc::runtime::{result as cuda_result, sys as cuda_sys, sys::cudaStream_t};
use tenferro_tensor::AllocationDomainId;

use super::device::{
    cuda_devices, unavailable_device_error, CudaDeviceError, CudaDeviceId, CudaDeviceInfo,
};
use super::identity::GpuExtensionCapability;

/// Returns `true` if a CUDA device can initialize a CubeCL runtime.
///
/// Use this in test helpers to skip GPU tests on machines without hardware.
pub fn gpu_available() -> bool {
    let library_present = std::panic::catch_unwind(|| {
        // SAFETY: `is_culib_present` only probes candidate library names and
        // does not call CUDA function pointers or retain a library handle.
        unsafe { cudarc::driver::sys::is_culib_present() }
    })
    .unwrap_or(false);
    if !library_present {
        return false;
    }
    let Ok(devices) = cuda_devices() else {
        return false;
    };
    let Some(device_id) = devices.first().map(|device| device.id()) else {
        return false;
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(runtime) = CudaRuntime::new(device_id) else {
            return false;
        };
        runtime.synchronize().is_ok()
    }))
    .unwrap_or(false)
}

/// RAII guard that attempts to restore the thread's previous CUDA device and
/// current context when dropped.
///
/// Used by the `with_raw` enter/exit protocol: the guard is created after the
/// calling thread's device/context are saved and the tenferro primary context
/// is activated. Drop attempts best-effort restoration of the saved state on
/// normal return, `Err`, and unwind; a restoration failure is logged to
/// stderr (non-panicking) and never returned.
pub(crate) struct RawContextRestore {
    saved_device: Result<i32, cudarc::runtime::result::RuntimeError>,
    saved_context: Result<Option<CUcontext>, cudarc::driver::result::DriverError>,
    op: &'static str,
}

impl RawContextRestore {
    /// Save the current device/context, then activate `device`/`context`.
    pub(crate) fn enter(op: &'static str, device: i32, context: CUcontext) -> crate::Result<Self> {
        let saved_device = cudarc::runtime::result::device::get();
        let saved_context = cudarc::driver::result::ctx::get_current();
        cudarc::runtime::result::device::set(device)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        if let Err(err) = unsafe { cudarc::driver::result::ctx::set_current(context) } {
            // Roll the device and context back so a partial activation failure
            // cannot leave the caller's thread on a different device or with a
            // different current context (setting the device can implicitly
            // change the thread's current context to the new primary).
            if let Ok(previous_device) = saved_device {
                let _ = cudarc::runtime::result::device::set(previous_device);
            }
            match saved_context {
                Ok(Some(previous)) => {
                    let _ = unsafe { cudarc::driver::result::ctx::set_current(previous) };
                }
                Ok(None) => {
                    let _ =
                        unsafe { cudarc::driver::result::ctx::set_current(std::ptr::null_mut()) };
                }
                Err(_) => {}
            }
            return Err(crate::Error::backend_source(op, err));
        }
        Ok(Self {
            saved_device,
            saved_context,
            op,
        })
    }

    fn restore(&self) {
        let mut stderr = std::io::stderr();
        if let Ok(device) = self.saved_device {
            if let Err(err) = cudarc::runtime::result::device::set(device) {
                let _ = writeln!(
                    stderr,
                    "tenferro-gpu: failed to restore CUDA device during {}: {err:?}",
                    self.op
                );
            }
        }
        match self.saved_context {
            Ok(Some(context)) => {
                if let Err(err) = unsafe { cudarc::driver::result::ctx::set_current(context) } {
                    let _ = writeln!(
                        stderr,
                        "tenferro-gpu: failed to restore CUDA context during {}: {err:?}",
                        self.op
                    );
                }
            }
            // The thread had no current context before the guard; restore that
            // state instead of leaving the tenferro primary context current.
            Ok(None) => {
                if let Err(err) =
                    unsafe { cudarc::driver::result::ctx::set_current(std::ptr::null_mut()) }
                {
                    let _ = writeln!(
                        stderr,
                        "tenferro-gpu: failed to clear CUDA context during {}: {err:?}",
                        self.op
                    );
                }
            }
            // The saved-context query itself failed; nothing can be restored.
            Err(_) => {}
        }
    }
}

impl Drop for RawContextRestore {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Opaque identity of one exact CUDA runtime instance.
///
/// Cloning the identity preserves the underlying executable runtime witness;
/// constructing another runtime, even for the same device ordinal, produces a
/// distinct identity. The cache key intentionally carries no provider or
/// device identifier and grants no execution authority.
#[derive(Clone, Debug)]
pub struct CudaRuntimeIdentity {
    marker: Arc<u8>,
}

impl CudaRuntimeIdentity {
    fn fresh() -> Self {
        Self {
            marker: Arc::new(0),
        }
    }
}

impl PartialEq for CudaRuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.marker, &other.marker)
    }
}

impl Eq for CudaRuntimeIdentity {}

impl Hash for CudaRuntimeIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // INVARIANT: `marker` is retained by every clone of this identity, so
        // its Arc allocation address is move/clone-invariant while witnessed.
        state.write_usize(Arc::as_ptr(&self.marker) as usize);
    }
}

/// CubeCL CUDA runtime wrapper.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaRuntime;
///
/// let _ctor: fn(tenferro_gpu::cuda::CudaDeviceId) ->
///     Result<CudaRuntime, tenferro_gpu::cuda::CudaDeviceError> = CudaRuntime::new;
/// let _sync: fn(&CudaRuntime) -> tenferro_tensor::Result<()> =
///     CudaRuntime::synchronize;
/// ```
#[derive(Clone)]
pub struct CudaRuntime {
    inner: Arc<CudaRuntimeState>,
}

pub(crate) struct CudaRuntimeState {
    client: Client,
    device_id: CudaDeviceId,
    device_ordinal: usize,
    device_info: CudaDeviceInfo,
    primary_context: CudaPrimaryContext,
    identity: CudaRuntimeIdentity,
    allocation_domain: AllocationDomainId,
    // Memoized raw CUDA stream handles keyed by the bounded CubeCL stream-pool
    // slot, not the process-global and monotonically increasing `StreamId`.
    //
    // INVARIANT: in pinned CubeCL rev a2adda17, the CUDA server maps each
    // `StreamId` to a fixed `StreamPool` slot whose `CUstream` is created once
    // and never destroyed or replaced while the server is alive, and the
    // server outlives the `Client` clone owned by this state. The table
    // is owned by this runtime object (not thread-local/global), holds one
    // entry per CubeCL stream slot, and is dropped with the runtime.
    raw_streams: Box<[OnceLock<u64>]>,
    // One lazily created cuBLAS handle per bounded CubeCL stream-pool slot.
    // Each slot lock covers pointer-mode selection and the enqueue itself:
    // distinct `StreamId`s can map to the same physical stream, and cuBLAS
    // handle configuration is mutable.
    cublas_handles: Box<[Mutex<Option<CublasStreamHandle>>]>,
    // Lazily allocated pinned-host staging slot for single-scalar downloads
    // (`cudaHostAlloc`, `PINNED_SCALAR_BYTES` bytes). Freed in `Drop` with
    // `cudaFreeHost` while the primary context is still retained.
    pinned_scalar: Mutex<PinnedScalarSlot>,
    // Vendor workspaces whose CubeCL handle may only return to the pool after
    // their stream reaches the event recorded at retirement time.
    workspace_retirements: Mutex<super::workspace_retirement::WorkspaceRetirementQueue>,
    // Transfers whose caller detached observation; released once their event completes, and
    // retained permanently when completion cannot be proven.
    transfer_retirements: Mutex<super::transfer::TransferRetirementQueue>,
    // Pinned-host staging for chunked blocking transfers (#1945 U4 item 4), leased to one transfer at
    // a time and fixed at `STAGING_CHUNK_BYTES`. A lease that returns here is proven complete; one
    // whose completion was never proven is retired through `transfer_retirements` and never returns.
    pinned_staging: Mutex<PinnedStagingSlot>,
}

/// One cached cuBLAS handle bound to a fixed CUDA stream.
struct CublasStreamHandle(cublas_sys::cublasHandle_t);

/// Pinned-host staging slot; `ptr` is null until the first scalar download.
struct PinnedScalarSlot {
    ptr: *mut std::ffi::c_void,
}

/// Size of the runtime-owned pinned staging slot: large enough for the widest
/// supported scalar (`Complex64`, 16 bytes).
pub(crate) const PINNED_SCALAR_BYTES: usize = 16;

/// Bytes of pinned staging a chunked blocking transfer moves at a time.
///
/// Fixed: the slot is allocated once, before the first enqueue, and never grows, so a transfer
/// performs no allocation inside its chunk loop.
pub(crate) const STAGING_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// Pinned-host staging slot for chunked blocking transfers; empty until the first such transfer.
struct PinnedStagingSlot {
    buffer: Option<super::transfer::PinnedHostBuffer>,
}

/// One chunked transfer's lease on the runtime's pinned staging buffer.
///
/// The slot hands the buffer out instead of sharing it, so two concurrent chunked transfers never
/// stage through the same memory. A lease is disposed of explicitly: [`PinnedStagingLease::release`]
/// once the last chunk's completion is proven, [`PinnedStagingLease::retire`] when it is not. Dropping
/// a lease without either is the unwinding path, and it leaks the buffer rather than freeing memory the
/// device may still be reading or writing.
///
/// `!Send + !Sync` by construction: it owns pinned host memory bound to the CUDA context that created
/// it, exactly like the buffer it holds.
pub(crate) struct PinnedStagingLease {
    buffer: Option<super::transfer::PinnedHostBuffer>,
    /// The chunk loop's retained `get_resource` guard, so an unwind after a possible enqueue cannot
    /// drop retention for storage the device may still be writing.
    resource: Option<Box<dyn std::any::Any + Send>>,
    /// Device address of that resource, captured so the loop needs no separate copy.
    resource_ptr: u64,
    /// Set before every possible enqueue and cleared only by a wait that proved completion.
    unproven: bool,
    runtime: std::sync::Weak<CudaRuntimeState>,
}

impl PinnedStagingLease {
    /// Bytes this lease stages at a time.
    pub(crate) fn len(&self) -> usize {
        self.buffer
            .as_ref()
            .map_or(0, super::transfer::PinnedHostBuffer::len)
    }

    /// The leased pinned buffer.
    pub(crate) fn buffer(&self) -> &super::transfer::PinnedHostBuffer {
        self.buffer
            .as_ref()
            .expect("a live lease holds its buffer until it is released or retired")
    }

    /// The leased pinned buffer, mutably: the chunk loop fills it before an upload.
    pub(crate) fn buffer_mut(&mut self) -> &mut super::transfer::PinnedHostBuffer {
        self.buffer
            .as_mut()
            .expect("a live lease holds its buffer until it is released or retired")
    }

    /// Mark the lease unproven before an enqueue that may follow.
    ///
    /// A reused event proves only the work it captured, so a previous chunk's success must never be
    /// read as this chunk's completion: the loop marks the lease unproven first and clears it only
    /// after the wait that follows this enqueue succeeds.
    pub(crate) fn begin_chunk(&mut self) {
        self.unproven = true;
    }

    /// Record that the chunk enqueued after [`Self::begin_chunk`] completed.
    pub(crate) fn end_chunk(&mut self) {
        self.unproven = false;
    }

    /// Return the buffer to the runtime's slot after a proven completion.
    ///
    /// Releasing an unproven lease would hand back a buffer a chunk may still be writing, so it is
    /// leaked instead - in release builds as well, not only under the debug assertion - and the slot
    /// stays empty for the next transfer to allocate a replacement.
    pub(crate) fn release(mut self) {
        if self.unproven {
            // A caller that releases an unproven lease would return a buffer a chunk may still be
            // writing: leak it and the resource instead of trusting the caller's bookkeeping.
            self.leak();
            return;
        }
        self.restore();
    }

    /// Attach the chunk loop's retained device resource to this lease.
    ///
    /// `runtime_handle` is only used for the address; the resource itself stays here so that the
    /// lease owns everything a chunk may still touch.
    pub(crate) fn into_chunk_lease(
        mut self,
        resource: Option<Box<dyn std::any::Any + Send>>,
        address: u64,
    ) -> Self {
        self.resource = resource;
        self.resource_ptr = address;
        self
    }

    /// Device address of the retained resource.
    pub(crate) fn resource_ptr(&self) -> u64 {
        self.resource_ptr
    }

    /// Retire a lease whose completion was never proven.
    ///
    /// The event was never recorded or its record failed, so the buffer, the event and the device
    /// resource are retained permanently by the runtime's retirement queue.
    pub(crate) fn retire(
        mut self,
        event: CUevent,
        resource: Option<Box<dyn std::any::Any + Send>>,
    ) {
        let resource = resource.or_else(|| self.resource.take());
        let Some(buffer) = self.buffer.take() else {
            // An event whose record may not have happened must never be destroyed: `CUevent` is a raw
            // pointer, so dropping it here *is* leaving it to the device, and no destructor runs.
            std::mem::forget(resource);
            let _ = event;
            return;
        };
        let Some(state) = self.runtime.upgrade() else {
            // The runtime is gone, so its context cannot be selected: leak rather than free memory that
            // belongs to a device this lease can no longer name. The event is a raw pointer, so it is
            // simply not destroyed.
            std::mem::forget(buffer);
            let _ = event;
            return;
        };
        state.retain_staging_for_soundness(event, buffer, resource);
    }

    fn leak(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            std::mem::forget(buffer);
        }
        if let Some(resource) = self.resource.take() {
            std::mem::forget(resource);
        }
    }

    fn restore(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let Some(state) = self.runtime.upgrade() else {
            // The runtime is gone, so the buffer leaks rather than being freed through a context this
            // lease can no longer select - the same fallback the buffer's own `Drop` takes.
            std::mem::forget(buffer);
            return;
        };
        let mut slot = state
            .pinned_staging
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.unproven = false;
        if slot.buffer.is_none() {
            slot.buffer = Some(buffer);
        }
        self.resource = None;
    }
}

impl Drop for PinnedStagingLease {
    fn drop(&mut self) {
        if self.unproven {
            // Unwinding between an enqueue and its proven wait: the buffer and the retained resource
            // may still be part of a DMA, so they are leaked rather than freed, and the buffer never
            // returns to the slot.
            self.leak();
            return;
        }
        self.restore();
    }
}

// SAFETY: `CudaRuntimeState` owns a retained CUDA primary context and a CubeCL
// client for one device ordinal. Methods set the context current before raw CUDA
// calls, and backend/executor layers serialize mutating tensor execution. The
// raw cuBLAS handles and the pinned staging pointers are plain CUDA resource
// addresses owned by this state and released in `Drop`.
unsafe impl Send for CudaRuntimeState {}
// SAFETY: Shared state access exposes immutable runtime handles; synchronization
// and stream queries use explicit CUDA/CubeCL handles and do not mutate Rust
// aliasing-visible fields. cuBLAS handles are locked per bounded physical
// stream slot, and a pinned staging buffer is touched either while its slot mutex is held or by the
// one lease that took it out of the slot, which serializes access to that buffer.
unsafe impl Sync for CudaRuntimeState {}

impl fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device_id", &self.inner.device_id)
            .finish_non_exhaustive()
    }
}

struct CudaPrimaryContext {
    cuda_device: CUdevice,
    cuda_context: CUcontext,
}

impl CudaPrimaryContext {
    fn retain(cuda_device: CUdevice) -> crate::Result<Self> {
        let cuda_context = unsafe { cudarc::driver::result::primary_ctx::retain(cuda_device) }
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        Ok(Self {
            cuda_device,
            cuda_context,
        })
    }

    fn context(&self) -> CUcontext {
        self.cuda_context
    }
}

impl Drop for CudaPrimaryContext {
    fn drop(&mut self) {
        if let Err(err) = unsafe { cudarc::driver::result::primary_ctx::release(self.cuda_device) }
        {
            report_cuda_primary_context_release_error(&err);
        }
    }
}

#[cold]
fn report_cuda_primary_context_release_error(err: &impl fmt::Debug) {
    eprintln!("tenferro-gpu: failed to release CUDA primary context during Drop: {err:?}");
}

#[cold]
fn report_cuda_runtime_drop_error(err: &crate::Error) {
    eprintln!("tenferro-gpu: failed to synchronize CUDA runtime during Drop: {err}");
}

impl CudaRuntime {
    /// Initialize the CubeCL CUDA runtime on the caller-selected device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId, cuda::CudaRuntime};
    ///
    /// let _ctor: fn(CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> = CudaRuntime::new;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CudaDeviceError::Discovery`] when fallback discovery for an
    /// invalid selected ordinal fails, [`CudaDeviceError::Unavailable`] when
    /// that ordinal is not available, or [`CudaDeviceError::Initialization`]
    /// when CUDA driver, runtime, context, or CubeCL client initialization
    /// fails.
    pub fn new(device_id: CudaDeviceId) -> Result<Self, CudaDeviceError> {
        let device_ordinal = usize::try_from(device_id.ordinal()).map_err(|source| {
            cuda_initialization_error(device_id, "convert_device_ordinal", source)
        })?;
        let cuda_ordinal = i32::try_from(device_id.ordinal()).map_err(|source| {
            cuda_initialization_error(device_id, "convert_cuda_ordinal", source)
        })?;
        cudarc::driver::result::init()
            .map_err(|source| cuda_initialization_error(device_id, "initialize_driver", source))?;
        let cuda_device = match cudarc::driver::result::device::get(cuda_ordinal) {
            Ok(cuda_device) => cuda_device,
            Err(source) if is_invalid_device_lookup(source) => {
                return Err(unavailable_device_error(device_id, cuda_devices()?));
            }
            Err(source) => {
                return Err(cuda_initialization_error(device_id, "get_device", source));
            }
        };
        let primary_context = CudaPrimaryContext::retain(cuda_device).map_err(|source| {
            cuda_initialization_error(device_id, "retain_primary_context", source)
        })?;
        unsafe { cudarc::driver::result::ctx::set_current(primary_context.context()) }.map_err(
            |source| cuda_initialization_error(device_id, "set_current_context", source),
        )?;
        cudarc::runtime::result::device::set(cuda_ordinal)
            .map_err(|source| cuda_initialization_error(device_id, "set_device", source))?;
        let device = CudaDevice::new(device_ordinal);
        let client = Device::Cuda(device).client();
        let discovered = cuda_devices()?;
        let device_info = discovered
            .iter()
            .find(|info| info.id() == device_id)
            .cloned()
            .ok_or_else(|| unavailable_device_error(device_id, discovered))?;
        Ok(Self {
            inner: Arc::new(CudaRuntimeState {
                client,
                device_id,
                device_ordinal,
                device_info,
                primary_context,
                identity: CudaRuntimeIdentity::fresh(),
                allocation_domain: AllocationDomainId::fresh(),
                raw_streams: (0..cubecl_stream_slots())
                    .map(|_| OnceLock::new())
                    .collect(),
                cublas_handles: (0..cubecl_stream_slots())
                    .map(|_| Mutex::new(None))
                    .collect(),
                pinned_scalar: Mutex::new(PinnedScalarSlot {
                    ptr: std::ptr::null_mut(),
                }),
                workspace_retirements: Mutex::new(Default::default()),
                transfer_retirements: Mutex::new(Default::default()),
                pinned_staging: Mutex::new(PinnedStagingSlot { buffer: None }),
            }),
        })
    }

    pub(crate) fn client(&self) -> &Client {
        &self.inner.client
    }

    /// Lease the runtime's pinned staging buffer for a chunked blocking transfer.
    ///
    /// The buffer is allocated on first use and never grown, and the lease takes it out of the slot, so
    /// a concurrent chunked transfer allocates its own rather than sharing this one.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the CUDA context cannot be selected or the pinned
    /// allocation fails.
    pub(crate) fn acquire_pinned_staging(&self) -> crate::Result<PinnedStagingLease> {
        let mut slot = self
            .inner
            .pinned_staging
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let buffer = match slot.buffer.take() {
            Some(buffer) => buffer,
            None => super::transfer::PinnedHostBuffer::new(self, STAGING_CHUNK_BYTES)?,
        };
        Ok(PinnedStagingLease {
            buffer: Some(buffer),
            resource: None,
            resource_ptr: 0,
            unproven: false,
            runtime: self.weak_state(),
        })
    }

    /// Return the caller-selected CUDA device identity that this runtime targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceId, cuda::CudaRuntime};
    ///
    /// let _device_id: fn(&CudaRuntime) -> CudaDeviceId = CudaRuntime::device_id;
    /// ```
    pub fn device_id(&self) -> CudaDeviceId {
        self.inner.device_id
    }

    /// Return immutable metadata for the runtime's device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _info: fn(&CudaRuntime) -> &tenferro_gpu::cuda::CudaDeviceInfo =
    ///     CudaRuntime::device_info;
    /// ```
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.inner.device_info
    }

    /// Return the allocation ownership domain of this runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _domain: fn(&CudaRuntime) -> tenferro_tensor::AllocationDomainId =
    ///     CudaRuntime::allocation_domain;
    /// ```
    pub fn allocation_domain(&self) -> AllocationDomainId {
        self.inner.allocation_domain
    }

    /// Report whether this CUDA session supports a GPU extension capability.
    ///
    /// The CUDA provider supports the full extension vocabulary: external
    /// CubeCL kernels, native module loading, runtime compilation (NVRTC), raw
    /// stream borrowing, and same-device copy. `PeerCopy` is reported as a
    /// directional query; availability is hardware-dependent and is checked
    /// per source/destination pair rather than here.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::GpuExtensionCapability;
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _supports: fn(&CudaRuntime, GpuExtensionCapability) -> bool =
    ///     CudaRuntime::supports_extension;
    /// ```
    pub fn supports_extension(&self, capability: GpuExtensionCapability) -> bool {
        capabilities_for_device(capability)
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
    }

    pub(crate) fn primary_context(&self) -> CUcontext {
        self.inner.primary_context.context()
    }

    /// Run `f` with the tenferro primary context current on this thread.
    ///
    /// Saves the calling thread's current CUDA device/context, activates the
    /// tenferro primary context for the duration of `f`, and attempts to
    /// restore the saved state on every exit path (normal return, `Err`, and
    /// unwind). Restoration is best-effort: a failure to restore the
    /// caller's previous device/context is logged to stderr rather than
    /// returned. This is the scoped context authority used by vendor-library
    /// lifecycle paths (plan creation/retirement) that run outside a
    /// raw-session callback.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the tenferro primary
    /// context cannot be activated (device or context driver failure); a
    /// partial activation is best-effort rolled back before the error is
    /// returned (rollback failures are discarded).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _check: fn(&CudaRuntime) -> tenferro_tensor::Result<u64> = |rt| {
    ///     rt.with_current_context("test.context", || 7)
    /// };
    /// ```
    pub fn with_current_context<R>(
        &self,
        op: &'static str,
        f: impl FnOnce() -> R,
    ) -> crate::Result<R> {
        let device_ordinal = i32::try_from(self.device_ordinal())
            .map_err(|source| crate::Error::backend_source(op, source))?;
        let _guard = RawContextRestore::enter(op, device_ordinal, self.primary_context())?;
        Ok(f())
    }

    /// Flush pending CubeCL work on the current stream.
    ///
    /// Used by the raw-session enter protocol so raw library calls observe
    /// previously enqueued CubeCL work.
    pub(crate) fn flush_cubecl(&self, op: &'static str) -> crate::Result<()> {
        self.inner.flush_cubecl(op)
    }

    /// Report the errors of work already submitted, without waiting for device progress.
    ///
    /// CubeCL's `flush` reports producer errors but also retires staged host bytes and arms a fence
    /// the next submission synchronizes; a raw copy that must be ordered after pending kernels but
    /// must not wait for the device uses this instead. The host task queue is dispatched either way,
    /// so work queued before this call has reached the stream when it returns.
    pub(crate) fn check_errors(&self, op: &'static str) -> crate::Result<()> {
        self.inner.check_errors(op)
    }

    /// A weak handle to this runtime's state, for values that must not keep it alive.
    pub(crate) fn weak_state(&self) -> std::sync::Weak<CudaRuntimeState> {
        Arc::downgrade(&self.inner)
    }

    /// Return the opaque identity of this exact executable runtime instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _identity: fn(&CudaRuntime) -> tenferro_gpu::cuda::CudaRuntimeIdentity =
    ///     CudaRuntime::runtime_identity;
    /// ```
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity {
        self.inner.identity.clone()
    }

    pub(crate) fn allocation_domain_id(&self) -> AllocationDomainId {
        self.inner.allocation_domain
    }

    pub(crate) fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
        self.inner.set_current_cuda_context(op)
    }

    /// Detach a transfer record: the runtime keeps it until its event proves completion.
    pub(crate) fn detach_transfer(&self, record: super::transfer::TransferRetirement) {
        let mut queue = self
            .inner
            .transfer_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        queue.detach(&self.inner, record);
    }

    /// Retain a transfer record permanently: its completion was never proven, or its event was never
    /// recorded, so no later query may be trusted to release it.
    pub(crate) fn retain_transfer_for_soundness(
        &self,
        record: super::transfer::TransferRetirement,
    ) {
        let mut queue = self
            .inner
            .transfer_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        queue.retain_for_soundness(record);
    }

    /// Release detached transfers whose event already completed. Never waits for the device.
    pub(crate) fn drain_transfer_retirements(&self) {
        let mut queue = self
            .inner
            .transfer_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        queue.drain(&self.inner);
    }

    /// Number of detached transfers still awaiting a completion witness.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let runtime: tenferro_gpu::cuda::CudaRuntime = unimplemented!();
    /// let _pending = runtime.pending_transfer_retirements();
    /// ```
    pub fn pending_transfer_retirements(&self) -> usize {
        self.inner
            .transfer_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pending()
    }

    pub(crate) fn workspace_retirements(
        &self,
    ) -> &Mutex<super::workspace_retirement::WorkspaceRetirementQueue> {
        &self.inner.workspace_retirements
    }

    pub(crate) fn state(&self) -> &CudaRuntimeState {
        &self.inner
    }

    pub(crate) fn raw_cuda_stream(&self) -> crate::Result<u64> {
        self.inner.raw_cuda_stream()
    }

    /// Run one cuBLAS enqueue with the handle for the current CubeCL stream.
    ///
    /// The caller must have the tenferro primary context current on this
    /// thread (see [`CudaRuntimeState::set_current_cuda_context`]).
    pub(crate) fn with_cublas_handle<R>(
        &self,
        op: &'static str,
        pointer_mode: cublas_sys::cublasPointerMode_t,
        cross_stream_handles: Vec<cubecl_runtime::server::Handle>,
        execute: impl FnOnce(cublas_sys::cublasHandle_t) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.inner
            .with_cublas_handle(op, pointer_mode, cross_stream_handles, execute)
    }

    pub(crate) fn finish_vendor_enqueue<R>(
        &self,
        op: &'static str,
        cross_stream_handles: Vec<cubecl_runtime::server::Handle>,
        result: crate::Result<R>,
    ) -> crate::Result<R> {
        self.inner
            .finish_vendor_enqueue(op, cross_stream_handles, result)
    }

    pub(crate) fn stream_slot(&self) -> usize {
        self.inner.stream_slot()
    }

    pub(crate) fn stream_slot_count(&self) -> usize {
        self.inner.raw_streams.len()
    }

    pub(crate) fn is_current_stream_slot(&self, handle: &cubecl_runtime::server::Handle) -> bool {
        self.inner.stream_slot_for(handle.stream) == self.inner.stream_slot()
    }

    /// Download up to [`PINNED_SCALAR_BYTES`] bytes from a device address
    /// through the runtime-owned pinned staging slot.
    ///
    /// Enqueues an async device-to-host copy on the current thread's CubeCL
    /// stream and synchronizes only that stream, so previously enqueued work
    /// on the stream is observed without a device-wide barrier.
    pub(crate) fn download_scalar_bytes(
        &self,
        device_addr: u64,
        out: &mut [u8],
        op: &'static str,
        retained: cubecl_runtime::server::Handle,
    ) -> crate::Result<()> {
        self.inner
            .download_scalar_bytes(device_addr, out, op, retained)
    }

    /// Resolve `handle`'s device address on the current stream, then flush
    /// pending CubeCL work, in one device-thread hand-off.
    ///
    /// This is exactly `client().get_resource(handle)` followed by
    /// `flush_cubecl`, in that order and on the same stream, so the ordering a
    /// raw vendor enqueue relies on is unchanged; it only saves the second
    /// blocking round trip (#1887). It does not publish a write (that remains
    /// tensor4all/cubecl#16).
    pub(crate) fn resolve_and_flush(
        &self,
        handle: cubecl_runtime::server::Handle,
        op: &'static str,
    ) -> crate::Result<u64> {
        use cubecl::server::Server;

        let stream_id = StreamId::current();
        let binding = handle.binding();
        let resolved = self
            .inner
            .client
            .with_server(move |server| {
                let resource = server.get_resource(binding, stream_id);
                server.flush(stream_id).map(|()| resource)
            })
            .ok_or_else(|| crate::Error::runtime_state(op, "CubeCL server is unavailable"))?
            .map_err(|err| crate::Error::backend_source(op, err))?
            .map_err(|err| crate::Error::backend_source(op, err))?;
        Ok(resolved.resource().ptr)
    }

    /// Copy `len` bytes of a CubeCL allocation into caller-owned host memory.
    ///
    /// See `CudaRuntimeState::download_into_host` for the ordering and
    /// failure contract.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes and stay allocated until
    /// this call returns. On error the caller must not free or reuse `dst`
    /// (the device may still be writing it): leak it instead.
    pub(crate) unsafe fn download_into_host(
        &self,
        handle: cubecl_runtime::server::Handle,
        dst: *mut u8,
        len: usize,
        op: &'static str,
    ) -> crate::Result<()> {
        // SAFETY: forwarded caller contract.
        unsafe { self.inner.download_into_host(handle, dst, len, op) }
    }

    /// Block the current thread until work submitted to the current CUDA stream completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntime;
    ///
    /// let _sync: fn(&CudaRuntime) -> tenferro_tensor::Result<()> =
    ///     CudaRuntime::synchronize;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when CubeCL cannot expose the
    /// current stream, or [`crate::Error::BackendSource`] when CUDA context or
    /// stream synchronization fails.
    pub fn synchronize(&self) -> crate::Result<()> {
        self.inner.synchronize()
    }
}

impl CudaRuntimeState {
    /// Retain a chunked transfer's staging package permanently: its completion was never proven, or
    /// its event was never recorded, so no later query may be trusted to release it.
    fn retain_staging_for_soundness(
        &self,
        event: CUevent,
        buffer: super::transfer::PinnedHostBuffer,
        resource: Option<Box<dyn std::any::Any + Send>>,
    ) {
        let mut queue = self
            .transfer_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        queue.retain_for_soundness(super::transfer::staging_retirement(event, buffer, resource));
    }

    fn stream_slot(&self) -> usize {
        self.stream_slot_for(StreamId::current())
    }

    fn stream_slot_for(&self, stream_id: StreamId) -> usize {
        stream_id.value as usize % self.raw_streams.len()
    }

    pub(crate) fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
        // Fast path: the tenferro primary context is already current on this
        // thread. `cuCtxGetCurrent` only reads driver thread state, so this
        // skips the per-op `cudaSetDevice` + `cuCtxSetCurrent` round trips.
        // Runtime-API calls made afterwards operate on the current driver
        // context, so no separate runtime-API device activation is needed.
        if let Ok(Some(current)) = cudarc::driver::result::ctx::get_current() {
            if current == self.primary_context.context() {
                return Ok(());
            }
        }
        // INVARIANT: CUDA ordinals are device identifiers; bad ordinals are
        // reported by CUDA instead of indexing memory in tenferro.
        let device_ordinal = i32::try_from(self.device_id.ordinal())
            .map_err(|source| crate::Error::backend_source(op, source))?;
        cudarc::runtime::result::device::set(device_ordinal)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        unsafe { cudarc::driver::result::ctx::set_current(self.primary_context.context()) }
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    fn raw_cuda_stream(&self) -> crate::Result<u64> {
        let stream_id = StreamId::current();
        let slot = self.stream_slot();
        if let Some(&stream) = self.raw_streams[slot].get() {
            return Ok(stream);
        }
        let stream = self
            .client
            .with_server(move |server| {
                server
                    .raw_stream(stream_id)
                    .map(|stream| stream as u64)
                    .map_err(|err| crate::Error::backend_source("raw_cuda_stream", err))
            })
            .ok_or_else(|| {
                crate::Error::runtime_state("raw_cuda_stream", "CubeCL server is unavailable")
            })??;
        Ok(*self.raw_streams[slot].get_or_init(|| stream))
    }

    fn flush_cubecl(&self, op: &'static str) -> crate::Result<()> {
        self.client
            .flush()
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    fn check_errors(&self, op: &'static str) -> crate::Result<()> {
        self.client
            .check_errors()
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "cubecl_runtime_synchronize";
        // A cached raw stream does not drain CubeCL's host-side launch queue.
        self.flush_cubecl(OP)?;
        let stream = self.raw_cuda_stream()?;
        self.synchronize_raw_stream(stream, OP)?;
        // An explicit barrier also resolves deferred workspace retirements, so
        // callers that synchronize observe released workspaces.
        self.workspace_retirements
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .drain_blocking(self);
        // Same for transfers whose caller detached observation: after a barrier their events have
        // completed, so the records release their value, buffer, retention and event.
        {
            let mut queue = self
                .transfer_retirements
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            queue.drain(self);
        }
        Ok(())
    }

    pub(crate) fn synchronize_raw_stream(
        &self,
        stream: u64,
        op: &'static str,
    ) -> crate::Result<()> {
        self.set_current_cuda_context(op)?;
        unsafe { cuda_result::stream::synchronize(stream as usize as cudaStream_t) }
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    fn retire_initialized_streams(&self) -> bool {
        const OP: &str = "cuda_runtime_drop";
        if let Err(error) = self.set_current_cuda_context(OP) {
            report_cuda_runtime_drop_error(&error);
            return false;
        }
        let mut retired = true;
        for stream in &self.raw_streams {
            let Some(&stream) = stream.get() else {
                continue;
            };
            // SAFETY: every initialized entry is a CubeCL-owned stream that
            // remains live until this runtime state and its client are dropped.
            if let Err(source) =
                unsafe { cuda_result::stream::synchronize(stream as usize as cudaStream_t) }
            {
                retired = false;
                report_cuda_runtime_drop_error(&crate::Error::backend_source(OP, source));
            }
        }
        retired
    }

    fn with_cublas_handle<R>(
        &self,
        op: &'static str,
        pointer_mode: cublas_sys::cublasPointerMode_t,
        cross_stream_handles: Vec<cubecl_runtime::server::Handle>,
        execute: impl FnOnce(cublas_sys::cublasHandle_t) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let poisoned = || crate::Error::runtime_state(op, "cuBLAS handle cache lock poisoned");
        let slot = self.stream_slot();
        let mut cached = self.cublas_handles[slot].lock().map_err(|_| poisoned())?;
        let handle = match *cached {
            Some(ref handle) => handle.0,
            None => {
                if !cublas_library_present() {
                    return Err(crate::Error::io_source(op, CublasLibraryMissing));
                }
                let stream = self.raw_cuda_stream()? as usize as cublas_sys::cudaStream_t;
                let mut raw = std::ptr::null_mut();
                // SAFETY: the caller holds the tenferro primary context current;
                // the handle is created on this device and bound to this stream.
                check_cublas(op, "cublasCreate", unsafe {
                    cublas_sys::cublasCreate_v2(&mut raw)
                })?;
                // SAFETY: `raw` was just created and `stream` is owned by this
                // runtime for the lifetime of the cached handle.
                if let Err(err) = check_cublas(op, "cublasSetStream", unsafe {
                    cublas_sys::cublasSetStream_v2(raw, stream)
                }) {
                    // SAFETY: `raw` is live and not stored anywhere else.
                    let _ = unsafe { cublas_sys::cublasDestroy_v2(raw) };
                    return Err(err);
                }
                *cached = Some(CublasStreamHandle(raw));
                raw
            }
        };
        // SAFETY: the per-stream slot lock is held across configuration and
        // enqueue, so no caller can race this mutable handle state.
        check_cublas(op, "cublasSetPointerMode", unsafe {
            cublas_sys::cublasSetPointerMode_v2(handle, pointer_mode)
        })?;
        let result = execute(handle);
        self.finish_vendor_enqueue(op, cross_stream_handles, result)
    }

    fn finish_vendor_enqueue<R>(
        &self,
        op: &'static str,
        cross_stream_handles: Vec<cubecl_runtime::server::Handle>,
        result: crate::Result<R>,
    ) -> crate::Result<R> {
        if cross_stream_handles.is_empty() {
            return result;
        }
        let retirement = self.synchronize();
        match (result, retirement) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(retirement)) => {
                // No completion barrier was proven. Retain the foreign-stream
                // allocations so their owners cannot reclaim or mutate them.
                std::mem::forget(cross_stream_handles);
                Err(crate::Error::backend_source(op, retirement))
            }
            (Err(error), Err(_retirement)) => {
                std::mem::forget(cross_stream_handles);
                Err(error)
            }
        }
    }

    fn download_scalar_bytes(
        &self,
        device_addr: u64,
        out: &mut [u8],
        op: &'static str,
        retained: cubecl_runtime::server::Handle,
    ) -> crate::Result<()> {
        if out.len() > PINNED_SCALAR_BYTES {
            return Err(crate::Error::Internal(format!(
                "pinned scalar staging supports at most {PINNED_SCALAR_BYTES} bytes, got {}",
                out.len()
            )));
        }
        // Reused output addresses can already be cached: pointer lookup is not
        // a queue barrier. Submit pending kernels before the raw D2H copy.
        self.flush_cubecl(op)?;
        self.set_current_cuda_context(op)?;
        let stream = self.raw_cuda_stream()? as usize as cudaStream_t;
        // ponytail: one shared staging slot serializes concurrent scalar
        // downloads per runtime; add per-thread slots if that lock contends.
        let mut slot = self
            .pinned_scalar
            .lock()
            .map_err(|_| crate::Error::runtime_state(op, "pinned scalar staging lock poisoned"))?;
        if slot.ptr.is_null() {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: the primary context is current; the allocation is freed
            // in this state's `Drop` with `cudaFreeHost`.
            unsafe {
                cuda_sys::cudaHostAlloc(
                    &mut ptr,
                    PINNED_SCALAR_BYTES,
                    cuda_sys::cudaHostAllocDefault,
                )
            }
            .result()
            .map_err(|err| crate::Error::backend_source(op, err))?;
            slot.ptr = ptr;
        }
        let src = super::interop::cuda_device_ptr_from_addr(device_addr, op)?;
        // SAFETY: `slot.ptr` is a live pinned allocation of PINNED_SCALAR_BYTES
        // bytes, `out.len()` is validated above, and the mutex guard keeps the
        // slot exclusive until the copy below is known to have completed. Every
        // exit that cannot prove completion abandons the slot instead of
        // returning it, so exclusivity never rests on an unproven barrier.
        let staging = unsafe { std::slice::from_raw_parts_mut(slot.ptr.cast::<u8>(), out.len()) };
        // Neither submitting the copy nor waiting on it proves the device is
        // done with `staging` and `retained` once it reports an error: an async
        // CUDA call can surface a failure from an earlier launch on the stream,
        // so a non-success return says nothing about what is still running.
        // Both paths therefore leak the source allocation and abandon the
        // staging slot rather than let a later download reuse a destination the
        // device may still write, or let `Drop` `cudaFreeHost` it. Each failure
        // leaks one slot and one handle; the next call allocates fresh ones.
        // SAFETY: `src` is a residency-checked device address owned by this
        // runtime, the copy length equals the destination slice length, and
        // `stream` is the memoized CubeCL stream the copy is enqueued on.
        let completed = unsafe { cuda_result::memcpy_dtoh_async(staging, src, stream) }
            .and_then(|()| unsafe { cuda_result::stream::synchronize(stream) });
        if let Err(err) = completed {
            std::mem::forget(retained);
            slot.ptr = std::ptr::null_mut();
            return Err(crate::Error::backend_source(op, err));
        }
        out.copy_from_slice(staging);
        Ok(())
    }

    /// Copy `len` bytes from the start of `handle` straight into host memory.
    ///
    /// This replaces `read_one` plus a host copy for whole-tensor downloads
    /// (#2009): CubeCL's read stages into its own pinned (or, above 100 MB,
    /// freshly zeroed pageable) buffer, and the result then had to be copied
    /// again into the tensor's `Vec<T>`. Here the driver copies directly into
    /// the destination the caller will own.
    ///
    /// Ordering: pending CubeCL launches are flushed, and `get_resource`
    /// resolves the allocation on the current stream exactly as `read_one`
    /// does, so writes queued on other CubeCL streams are waited for on this
    /// stream. The copy is enqueued on that stream and the stream is
    /// synchronized before returning, so `dst` is complete on `Ok`.
    ///
    /// Failure: an error from the copy or the barrier does not prove the
    /// device is done with either side, so the source allocation is leaked and
    /// the caller must leak `dst` as well (see the safety contract).
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes until this returns, and
    /// must not be freed or reused after an error.
    unsafe fn download_into_host(
        &self,
        handle: cubecl_runtime::server::Handle,
        dst: *mut u8,
        len: usize,
        op: &'static str,
    ) -> crate::Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.flush_cubecl(op)?;
        let resource = self
            .client
            .get_resource::<cubecl_cuda::CudaServer>(handle)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
        if available < len {
            return Err(crate::Error::Internal(format!(
                "{op}: download of {len} bytes exceeds the {available}-byte allocation"
            )));
        }
        let src = resource.resource().ptr;
        self.set_current_cuda_context(op)?;
        let stream = self.raw_cuda_stream()? as usize as cudarc::driver::sys::CUstream;
        // SAFETY: `src` is the device address of a live CubeCL allocation of at
        // least `len` bytes (checked above) kept alive by `resource`; `dst` is
        // valid for `len` byte writes per the caller contract; `stream` is the
        // current CubeCL stream that `get_resource` ordered the allocation on.
        let completed = unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(dst.cast(), src, len, stream).result()
        }
        .and_then(|()| unsafe { cudarc::driver::result::stream::synchronize(stream) });
        if let Err(err) = completed {
            std::mem::forget(resource);
            return Err(crate::Error::backend_source(op, err));
        }
        Ok(())
    }

    /// Destroy cached cuBLAS handles and free the pinned staging slot.
    ///
    /// Called from `Drop` after all initialized streams have retired, which
    /// leaves the primary context current on the dropping thread.
    /// Takes `&self`: every resource it releases is behind one of this state's mutexes, and a shared
    /// receiver keeps it callable from the tests that have to observe the release.
    fn release_cuda_library_resources(&self) {
        for cached in &self.cublas_handles {
            if let Ok(mut handle) = cached.lock() {
                let Some(handle) = handle.take() else {
                    continue;
                };
                // SAFETY: each stored handle is live and no longer reachable.
                if let Err(err) = unsafe { cublas_sys::cublasDestroy_v2(handle.0) }.result() {
                    report_cuda_resource_release_error("cuBLAS handle", &err);
                }
            }
        }
        if let Ok(mut slot) = self.pinned_scalar.lock() {
            if !slot.ptr.is_null() {
                // SAFETY: the slot owns exactly one live cudaHostAlloc allocation.
                if let Err(err) = unsafe { cuda_sys::cudaFreeHost(slot.ptr) }.result() {
                    report_cuda_resource_release_error("pinned scalar staging", &err);
                }
                slot.ptr = std::ptr::null_mut();
            }
        }
        // A buffer in the staging slot is only ever populated by a proven completion, so it is as safe
        // to free here as the scalar slot; a retired lease never returns to the slot. The release is
        // explicit because the buffer's own `Drop` cannot upgrade its weak runtime reference while this
        // state is being destroyed, and would leak the allocation.
        if let Ok(mut slot) = self.pinned_staging.lock() {
            let cached = slot.buffer.take();
            drop(slot);
            if let Some(buffer) = cached {
                if self.set_current_cuda_context("runtime teardown").is_ok() {
                    // SAFETY: the primary context is retained and was just made current.
                    unsafe { buffer.release_in_current_context() };
                } else {
                    std::mem::forget(buffer);
                }
            }
        }
    }
}

fn cubecl_stream_slots() -> usize {
    usize::from(CubeClRuntimeConfig::get().streaming.max_streams.max(1))
}

/// Typed load failure for the dynamically loaded cuBLAS library.
#[derive(Debug, thiserror::Error)]
#[error(
    "cuBLAS shared library not found; ensure `LD_LIBRARY_PATH` includes the CUDA toolkit library directory"
)]
struct CublasLibraryMissing;

/// Report whether the cuBLAS shared library can be dynamically loaded.
fn cublas_library_present() -> bool {
    use std::sync::OnceLock;
    static PRESENT: OnceLock<bool> = OnceLock::new();
    // SAFETY: `is_culib_present` only probes candidate library names and does
    // not call cuBLAS function pointers or retain a library handle.
    *PRESENT.get_or_init(|| unsafe { cublas_sys::is_culib_present() })
}

/// Map a non-success cuBLAS status to a typed provider error.
pub(super) fn check_cublas(
    op: &'static str,
    call: &'static str,
    status: cublas_sys::cublasStatus_t,
) -> crate::Result<()> {
    if matches!(status, cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
        Ok(())
    } else {
        Err(super::error::provider_status(
            op,
            "cuBLAS",
            call,
            status as i32,
        ))
    }
}

#[cold]
fn report_cuda_resource_release_error(what: &'static str, err: &impl fmt::Debug) {
    eprintln!("tenferro-gpu: failed to release {what} during Drop: {err:?}");
}

fn is_invalid_device_lookup(source: DriverError) -> bool {
    source.0 == CUresult::CUDA_ERROR_INVALID_DEVICE
}

/// CUDA provider support for the shared GPU extension vocabulary.
///
/// See [`GpuExtensionCapability`](super::identity::GpuExtensionCapability) for
/// the vocabulary. `PeerCopy` is hardware/topology dependent and is therefore
/// reported false at the provider level; the directional query in the explicit
/// multi-GPU copy API decides availability per source/destination pair.
pub(crate) fn capabilities_for_device(capability: GpuExtensionCapability) -> bool {
    !matches!(capability, GpuExtensionCapability::PeerCopy)
}

fn cuda_initialization_error<E>(
    device: CudaDeviceId,
    operation: &'static str,
    source: E,
) -> CudaDeviceError
where
    E: std::error::Error + Send + Sync + 'static,
{
    CudaDeviceError::Initialization {
        device,
        operation,
        source: Box::new(source),
    }
}

impl Drop for CudaRuntimeState {
    fn drop(&mut self) {
        // Drop cannot surface errors, but the runtime must not release the
        // primary context while queued kernels on any initialized slot may
        // still reference it. Retirement is asked once: a second call could turn
        // a failed retirement into a successful-looking one and skip the
        // permanent-retention branch below.
        let retired = self.retire_initialized_streams();
        if retired {
            // Every stream is retired, so queued workspace retirements are
            // complete and their handles can return to the pool.
            self.workspace_retirements
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .drain_blocking(self);
            // Transfers whose caller detached observation are released best effort; the ones whose
            // completion was never proven are retained permanently instead of being freed here.
            self.transfer_retirements
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .release_at_teardown(self);
            // Retirement left the primary context current; release CUDA
            // library resources before the retained primary context drops.
            self.release_cuda_library_resources();
        }
        // Streams are not confirmed done, so nothing they may still touch may be freed: a cached
        // staging buffer is leaked, and the retirement queue's records - in flight and already retained
        // alike - are forgotten rather than handed to field destruction. A record's `value` and
        // `resource` guards are device storage, which is exactly what must survive here.
        if !retired {
            if let Ok(mut slot) = self.pinned_staging.lock() {
                if let Some(buffer) = slot.buffer.take() {
                    std::mem::forget(buffer);
                }
            }
            // Records kept for soundness must not be freed by field destruction either, so they are
            // forgotten rather than merely moved between queues.
            self.transfer_retirements
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .forget_all();
        }
        // On retirement failure, raw library resources intentionally leak:
        // their pointer-only owners have no Drop implementation, so Rust does
        // not reclaim resources that may still be in use asynchronously.
    }
}

#[cfg(test)]
mod tests;
