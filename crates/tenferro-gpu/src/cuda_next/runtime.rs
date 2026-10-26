//! The runtime handle: one `t4a-cuda-runtime` runtime, one device, one allocation
//! domain, one submission lock for explicit host transfers.

use std::sync::{Arc, Mutex};

use tenferro_tensor::AllocationDomainId;

use crate::cuda::{CudaDeviceError, CudaDeviceId, CudaDeviceInfo};

/// Identity of one opened runtime; equal only for clones of the same handle.
#[derive(Debug, Clone)]
#[doc(hidden)]
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

impl std::hash::Hash for CudaRuntimeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // The marker allocation is retained by every clone, so its address is
        // move- and clone-invariant while an identity is witnessed.
        state.write_usize(Arc::as_ptr(&self.marker) as usize);
    }
}

/// State shared by every handle to one runtime.
pub(crate) struct CudaRuntimeState {
    /// The execution owner. Explicit transfers take this lock for one admission;
    /// numerical submission borrows it mutably through the engine.
    pub(crate) runtime: Mutex<t4a_cuda_runtime::Runtime>,
    pub(crate) device_id: CudaDeviceId,
    pub(crate) device_info: CudaDeviceInfo,
    pub(crate) allocation_domain: AllocationDomainId,
    pub(crate) identity: CudaRuntimeIdentity,
    /// Handles dropped while their work was still queued. The runtime's retirement
    /// already owns the staging; this queue owns only the caller's packages (a pinned
    /// source, a tensor handle) until their completion is observed.
    pub(crate) detached: Mutex<Vec<Detached>>,
}

/// One dropped handle retained until its completion is observed.
pub(crate) struct Detached {
    /// The caller's package: the queued copy still reads or writes it, so it stays owned
    /// here until the completion below proves the work done.
    #[allow(dead_code)]
    pub(crate) keepalive: Box<dyn std::any::Any + Send>,
    pub(crate) completion: Option<t4a_cuda_runtime::Completion>,
}

/// A cheap, cloneable handle to one opened CUDA runtime.
///
/// The handle carries no submission authority of its own: explicit host transfers
/// take the runtime's submission lock for the duration of one admission, and
/// numerical submission belongs to the [`CudaEngine`](super::CudaEngine) that owns
/// the same state.
#[derive(Clone)]
#[doc(hidden)]
pub struct CudaRuntime {
    pub(crate) inner: Arc<CudaRuntimeState>,
}

impl CudaRuntime {
    /// Open the CUDA runtime on `device_id` with the default configuration.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError`] when the device is unknown or the driver cannot open it.
    #[doc(hidden)]
    pub fn new(device_id: CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> {
        Self::with_config(device_id, t4a_cuda_runtime::Config::default())
    }

    /// Open the CUDA runtime on `device_id` with an explicit configuration.
    ///
    /// The configuration bounds the runtime's staging chunk and pending budget, which is
    /// what a caller uses to force chunked transfers.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError`] when the device is unknown, the configuration is invalid, or
    /// the driver cannot open the device.
    #[doc(hidden)]
    pub fn with_config(
        device_id: CudaDeviceId,
        config: t4a_cuda_runtime::Config,
    ) -> Result<CudaRuntime, CudaDeviceError> {
        let devices = crate::cuda::cuda_devices()?;
        let info = devices
            .iter()
            .find(|info| info.id() == device_id)
            .cloned()
            .ok_or_else(|| CudaDeviceError::Unavailable {
                requested: device_id,
                discovered: devices.into_boxed_slice(),
            })?;
        let runtime = t4a_cuda_runtime::Runtime::open(device_id.ordinal() as usize, config)
            .map_err(|error| CudaDeviceError::Initialization {
                device: device_id,
                operation: "t4a_cuda_runtime::Runtime::open",
                source: Box::new(error),
            })?;
        Ok(CudaRuntime {
            inner: Arc::new(CudaRuntimeState {
                runtime: Mutex::new(runtime),
                device_id,
                device_info: info,
                allocation_domain: AllocationDomainId::fresh(),
                identity: CudaRuntimeIdentity::fresh(),
                detached: Mutex::new(Vec::new()),
            }),
        })
    }

    /// The device this runtime owns.
    #[doc(hidden)]
    pub fn device_id(&self) -> CudaDeviceId {
        self.inner.device_id
    }

    /// The device metadata this runtime was opened with.
    #[doc(hidden)]
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.inner.device_info
    }

    /// The ordinal of the owning device.
    #[doc(hidden)]
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_id.ordinal() as usize
    }

    /// The identity of this runtime; distinct for every opened runtime.
    #[doc(hidden)]
    pub fn identity(&self) -> &CudaRuntimeIdentity {
        &self.inner.identity
    }

    /// The shared-allocation domain every buffer of this runtime belongs to.
    #[doc(hidden)]
    pub fn allocation_domain_id(&self) -> AllocationDomainId {
        self.inner.allocation_domain
    }

    /// NVRTC compilation attempts this runtime's artifact cache performed.
    pub fn compile_attempts(&self) -> u64 {
        self.submit().compile_attempts()
    }

    /// Module loads this runtime performed.
    pub fn module_loads(&self) -> u64 {
        self.submit().module_loads()
    }

    /// Artifact-cache hits this runtime served.
    pub fn artifact_cache_hits(&self) -> u64 {
        self.submit().artifact_cache_hits()
    }

    /// Run `closure` inside a raw scope on this runtime's stream, under the submission
    /// lock.
    ///
    /// # Safety
    ///
    /// As [`crate::Runtime::with_raw`](t4a_cuda_runtime::Runtime::with_raw): the closure
    /// must use this runtime's stream, keep its enqueues within the registered resources
    /// and obey each kernel's or vendor library's contract.
    pub(crate) unsafe fn with_raw_locked<R>(
        &self,
        resources: Vec<t4a_cuda_runtime::RawResource<'_>>,
        closure: impl FnOnce(t4a_cuda_runtime::RawScope<'_>) -> R,
    ) -> crate::Result<R> {
        let mut guard = self.submit();
        // SAFETY: forwarded to the caller's contract.
        unsafe { guard.with_raw(resources, closure) }
            .map_err(|error| crate::Error::runtime_state_source("cuda raw scope", error))
    }

    /// Resource counters of this runtime.
    ///
    /// The snapshot is the runtime's own: live and pooled device bytes, pending
    /// host/module bytes, retirement records and the transfer counters. The
    /// compatibility package adds its detached-handle count beside them
    /// ([`CudaRuntime::pending_transfer_retirements`]).
    #[doc(hidden)]
    pub fn stats(&self) -> t4a_cuda_runtime::Stats {
        let stats = self.submit().stats();
        self.retire_detached();
        stats
    }

    /// Copy `value` from `self` into a fresh allocation of `destination`.
    ///
    /// The copy is ordered on both runtimes' streams: the producer may be dropped and
    /// the value reused immediately, and a later read on the destination stream observes
    /// the copied bytes without a host wait.
    ///
    /// # Errors
    ///
    /// A typed refusal when the two handles are the same runtime, when `value` is not a
    /// CUDA allocation of this runtime, or when the driver cannot admit the copy.
    #[doc(hidden)]
    pub fn copy_to(
        &self,
        destination: &CudaRuntime,
        value: &tenferro_tensor::Tensor,
    ) -> crate::Result<tenferro_tensor::Tensor> {
        use tenferro_tensor::{Complex32, Complex64, DType, TensorScalar, TypedTensor};

        if self.identity() == destination.identity() {
            return Err(crate::Error::runtime_state(
                "cuda copy",
                "the source and destination runtimes are the same",
            ));
        }
        let storage = super::pending::cuda_storage(value).ok_or_else(|| {
            crate::Error::runtime_state("cuda copy", "expected a CUDA allocation")
        })?;
        if storage.device_ordinal() != self.device_ordinal() {
            return Err(crate::Error::runtime_state(
                "cuda copy",
                "the value belongs to another runtime",
            ));
        }

        // Both runtimes participate in admission. Take their submission locks in a
        // stable order (by allocation domain) so two opposing copies cannot invert them,
        // while the copy direction stays source -> destination in both branches.
        let source_is_lower = self.inner.allocation_domain < destination.inner.allocation_domain;
        let (buffer, _completion) = if source_is_lower {
            let mut source_guard = self.submit();
            let mut destination_guard = destination.submit();
            let (buffer, completion, _route) = destination_guard
                .copy_between(&mut source_guard, storage.allocation())
                .map_err(|error| crate::Error::runtime_state_source("cuda copy", error))?;
            (buffer, completion)
        } else {
            let mut destination_guard = destination.submit();
            let mut source_guard = self.submit();
            let (buffer, completion, _route) = destination_guard
                .copy_between(&mut source_guard, storage.allocation())
                .map_err(|error| crate::Error::runtime_state_source("cuda copy", error))?;
            (buffer, completion)
        };
        fn rebuild<T: TensorScalar + Send + Sync + 'static>(
            value: &tenferro_tensor::Tensor,
            buffer: t4a_cuda_runtime::Buffer,
            destination: &CudaRuntime,
        ) -> crate::Result<tenferro_tensor::Tensor> {
            let typed: &TypedTensor<T> = value.as_typed::<T>().ok_or_else(|| {
                crate::Error::runtime_state("cuda copy", "expected a preset scalar")
            })?;
            let storage = super::storage::fresh_storage(destination, buffer);
            TypedTensor::<T>::from_buffer_col_major(
                typed.shape().to_vec(),
                tenferro_tensor::StorageBuffer::Backend(Box::new(storage)),
                super::transfer::placement(destination),
            )
            .map(tenferro_tensor::Tensor::from_typed)
        }
        match value.dtype() {
            DType::F64 => rebuild::<f64>(value, buffer, destination),
            DType::F32 => rebuild::<f32>(value, buffer, destination),
            DType::I32 => rebuild::<i32>(value, buffer, destination),
            DType::I64 => rebuild::<i64>(value, buffer, destination),
            DType::C32 => rebuild::<Complex32>(value, buffer, destination),
            DType::C64 => rebuild::<Complex64>(value, buffer, destination),
            DType::Bool | DType::External(_) => Err(crate::Error::unsupported(
                "cuda copy",
                "the transfer route does not serve this dtype",
            )),
        }
    }

    /// Read a runtime allocation back into a byte vector.
    ///
    /// # Errors
    ///
    /// [`cuda::CudaDeviceError::Initialization`] when the copy cannot be admitted or
    /// completed.
    ///
    /// [`cuda::CudaDeviceError::Initialization`]: crate::cuda::CudaDeviceError::Initialization
    pub fn download_bytes(&self, buffer: &t4a_cuda_runtime::Buffer) -> crate::Result<Vec<u8>> {
        let mut guard = self.submit();
        let mut pending = guard
            .download(buffer)
            .map_err(|error| crate::Error::runtime_state_source("cuda download", error))?;
        pending
            .wait()
            .map_err(|error| crate::Error::runtime_state_source("cuda download", error))
    }

    /// Admit one raw host-to-device upload and return its allocation.
    ///
    /// This is the seam's own upload path, used by callers that already hold bytes (the
    /// publication test queues work with it); a typed [`upload_tensor`] wraps it with
    /// the tensor and placement metadata.
    ///
    /// [`upload_tensor`]: super::upload_tensor
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the runtime cannot allocate or enqueue
    /// the copy.
    #[doc(hidden)]
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<t4a_cuda_runtime::Buffer, CudaDeviceError> {
        let mut guard = self.submit();
        guard
            .upload(bytes)
            .map_err(|error| CudaDeviceError::Initialization {
                device: self.inner.device_id,
                operation: "t4a_cuda_runtime::Runtime::upload",
                source: Box::new(error),
            })
    }

    /// Allocate a pinned host buffer owned by this runtime.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the driver cannot allocate pinned
    /// memory.
    #[doc(hidden)]
    pub fn alloc_pinned(
        &self,
        bytes: usize,
    ) -> Result<t4a_cuda_runtime::PinnedHostBuffer, CudaDeviceError> {
        let mut guard = self.submit();
        guard
            .alloc_pinned(bytes)
            .map_err(|error| CudaDeviceError::Initialization {
                device: self.inner.device_id,
                operation: "t4a_cuda_runtime::Runtime::alloc_pinned",
                source: Box::new(error),
            })
    }

    /// Wait for every queued operation and retire the detached handles it completes.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the stream reports a failure.
    #[doc(hidden)]
    pub fn synchronize(&self) -> Result<(), CudaDeviceError> {
        {
            // Hold the submission lock through the reclamation: a concurrent clone cannot
            // enqueue a transfer it then detaches in the gap between the stream wait and
            // the queue drain, which would free its pinned source while still queued.
            let mut guard = self.submit();
            guard
                .synchronize()
                .map_err(|error| CudaDeviceError::Initialization {
                    device: self.inner.device_id,
                    operation: "t4a_cuda_runtime::Runtime::synchronize",
                    source: Box::new(error),
                })?;
            self.retire_detached_locked();
        }
        Ok(())
    }

    /// Detached pending handles that have not been retired yet, the compatibility
    /// package's own count (upstream `pending_records` counts retirement epochs, a
    /// different quantity).
    #[doc(hidden)]
    pub fn pending_transfer_retirements(&self) -> u64 {
        self.inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len() as u64
    }

    /// Retain a dropped handle until its completion is observed.
    pub(crate) fn detach(
        &self,
        keepalive: Box<dyn std::any::Any + Send>,
        completion: Option<t4a_cuda_runtime::Completion>,
    ) {
        self.inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Detached {
                keepalive,
                completion,
            });
    }

    /// Release the completed detached packages while the caller already holds the
    /// submission lock.
    fn retire_detached_locked(&self) {
        let mut detached = self
            .inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        detached.retain_mut(|entry| match entry.completion.as_mut() {
            Some(completion) => !matches!(completion.poll(), Ok(Some(()))),
            None => false,
        });
    }

    /// Release the detached packages whose completion is observed, keeping the rest.
    ///
    /// Reclamation is explicit: a package is only released once its own completion says
    /// the work is done, so a queued transfer never loses its source or destination.
    pub(crate) fn retire_detached(&self) {
        let mut detached = self
            .inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        detached.retain_mut(|entry| match entry.completion.as_mut() {
            Some(completion) => match completion.poll() {
                Ok(Some(())) => false,
                Ok(None) => true,
                // A failed completion is kept: the failure surfaces through the next
                // explicit observation rather than being dropped silently here.
                Err(_) => true,
            },
            None => false,
        });
    }

    /// Synchronise and then reclaim what completed, for teardown paths that cannot
    /// report an error.
    pub(crate) fn synchronize_best_effort(&self) {
        {
            let mut guard = self.submit();
            let _ = guard.synchronize();
        }
        self.retire_detached();
    }

    /// Take the submission lock for one explicit transfer.
    pub(crate) fn submit(&self) -> std::sync::MutexGuard<'_, t4a_cuda_runtime::Runtime> {
        self.inner
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl Drop for CudaRuntimeState {
    fn drop(&mut self) {
        // Last handle: drain the stream, then release the detached packages. A failure
        // leaves their completion unproved, so the queue is leaked instead of freeing
        // pinned memory a queued copy may still read or write.
        let proved = self
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .synchronize()
            .is_ok();
        if proved {
            let mut detached = self
                .detached
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            detached.clear();
        } else {
            let leaked = std::mem::take(
                &mut *self
                    .detached
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
            );
            std::mem::forget(leaked);
        }
    }
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device", &self.inner.device_id)
            .field("domain", &self.inner.allocation_domain)
            .finish()
    }
}
