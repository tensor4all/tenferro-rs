//! CubeCL CUDA runtime initialization and synchronization.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Arc;

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime as CubeclCudaRuntime};
use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{CUcontext, CUdevice, CUresult};
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};
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
                    let _ = unsafe {
                        cudarc::driver::result::ctx::set_current(std::ptr::null_mut())
                    };
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
                if let Err(err) = unsafe {
                    cudarc::driver::result::ctx::set_current(std::ptr::null_mut())
                } {
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

struct CudaRuntimeState {
    client: ComputeClient<CubeclCudaRuntime>,
    device_id: CudaDeviceId,
    device_ordinal: usize,
    device_info: CudaDeviceInfo,
    primary_context: CudaPrimaryContext,
    identity: CudaRuntimeIdentity,
    allocation_domain: AllocationDomainId,
}

// SAFETY: `CudaRuntimeState` owns a retained CUDA primary context and a CubeCL
// client for one device ordinal. Methods set the context current before raw CUDA
// calls, and backend/executor layers serialize mutating tensor execution.
unsafe impl Send for CudaRuntimeState {}
// SAFETY: Shared state access exposes immutable runtime handles; synchronization
// and stream queries use explicit CUDA/CubeCL handles and do not mutate Rust
// aliasing-visible fields.
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
        let client = CubeclCudaRuntime::client(&device);
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
            }),
        })
    }

    pub(crate) fn client(&self) -> &ComputeClient<CubeclCudaRuntime> {
        &self.inner.client
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
    /// partial activation is rolled back before the error is returned.
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
        self.client()
            .flush()
            .map_err(|err| crate::Error::backend_source(op, err))
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

    pub(crate) fn raw_cuda_stream(&self) -> crate::Result<u64> {
        self.inner.raw_cuda_stream()
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
    fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
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
        self.client
            .with_server(|server| {
                server
                    .raw_stream(StreamId::current())
                    .map(|stream| stream as u64)
                    .map_err(|err| crate::Error::backend_source("raw_cuda_stream", err))
            })
            .ok_or_else(|| {
                crate::Error::runtime_state("raw_cuda_stream", "CubeCL server is unavailable")
            })?
    }

    fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "cubecl_runtime_synchronize";
        self.set_current_cuda_context(OP)?;
        let stream = self.raw_cuda_stream()? as usize as cudaStream_t;
        unsafe { cuda_result::stream::synchronize(stream) }
            .map_err(|err| crate::Error::backend_source(OP, err))
    }
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
        // primary context while queued kernels may still reference it.
        if let Err(err) = self.synchronize() {
            report_cuda_runtime_drop_error(&err);
        }
    }
}

#[cfg(test)]
mod tests;
