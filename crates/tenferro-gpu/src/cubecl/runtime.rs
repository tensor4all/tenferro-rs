//! CubeCL CUDA runtime initialization and synchronization.

use std::fmt;
use std::sync::Arc;

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime as CubeclCudaRuntime};
use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{CUcontext, CUdevice, CUresult};
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};
use tenferro_tensor::AllocationDomainId;

use super::device::{cuda_devices, unavailable_device_error, CudaDeviceError, CudaDeviceId};

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

/// Opaque identity of one exact CUDA runtime instance.
///
/// Cloning the identity preserves the underlying executable runtime witness;
/// constructing another runtime, even for the same device ordinal, produces a
/// distinct identity. The token intentionally carries no provider or device
/// identifier and grants no execution authority.
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

    /// Return a stable, non-authoritative token for owner-scoped caches.
    ///
    /// The token is derived from the retained identity allocation, so cloning
    /// or moving this witness preserves it while independently constructed
    /// identities remain distinct while their witnesses are retained. It does
    /// not expose a CUDA handle, context, stream, or execution authority.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaRuntimeIdentity;
    ///
    /// let _token: fn(&CudaRuntimeIdentity) -> usize =
    ///     CudaRuntimeIdentity::cache_discriminator;
    /// ```
    #[doc(hidden)]
    pub fn cache_discriminator(&self) -> usize {
        // INVARIANT: `marker` is retained by every clone of this identity, so
        // its Arc allocation address is move/clone-invariant while witnessed.
        Arc::as_ptr(&self.marker) as usize
    }
}

impl PartialEq for CudaRuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.marker, &other.marker)
    }
}

impl Eq for CudaRuntimeIdentity {}

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
        Ok(Self {
            inner: Arc::new(CudaRuntimeState {
                client,
                device_id,
                device_ordinal,
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

    pub(crate) fn device_ordinal(&self) -> usize {
        self.inner.device_ordinal
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

    #[doc(hidden)]
    pub fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
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
mod identity_tests {
    use super::{gpu_available, is_invalid_device_lookup, CudaRuntime, CudaRuntimeIdentity};
    use crate::cuda::CudaBackend;
    use cudarc::driver::{result::DriverError, sys::CUresult};

    #[test]
    fn selected_device_lookup_classifies_only_cuda_invalid_device() {
        assert!(is_invalid_device_lookup(DriverError(
            CUresult::CUDA_ERROR_INVALID_DEVICE
        )));
        assert!(!is_invalid_device_lookup(DriverError(
            CUresult::CUDA_ERROR_INVALID_VALUE
        )));
    }

    #[test]
    fn cuda_runtime_identity_is_clone_stable_and_instance_scoped() {
        let first = CudaRuntimeIdentity::fresh();
        let first_token = first.cache_discriminator();
        let clone = first.clone();
        let moved = clone;
        let independent = CudaRuntimeIdentity::fresh();

        assert_eq!(first, moved);
        assert_eq!(first_token, moved.cache_discriminator());
        assert_ne!(first, independent);
        assert_ne!(first_token, independent.cache_discriminator());
    }

    #[test]
    fn cuda_backend_identity_tracks_the_exact_runtime_when_hardware_is_available() {
        if !gpu_available() {
            return;
        }

        let device = super::cuda_devices()
            .expect("CUDA device discovery should succeed")
            .into_iter()
            .next()
            .expect("CUDA device should be available")
            .id();
        let first = CudaBackend::new(device).expect("CUDA backend should initialize");
        let clone = first.clone();
        let independent = CudaBackend::new(device).expect("second CUDA backend should initialize");

        let first_identity = first.runtime_identity();
        let clone_identity = clone.runtime_identity();
        let independent_identity = independent.runtime_identity();
        assert_eq!(first_identity, clone_identity);
        assert_eq!(
            first_identity.cache_discriminator(),
            clone_identity.cache_discriminator()
        );
        assert_ne!(first_identity, independent_identity);
        assert_ne!(
            first_identity.cache_discriminator(),
            independent_identity.cache_discriminator()
        );

        let runtime_clone = first.runtime().clone();
        let _: CudaRuntime = runtime_clone;
    }
}
