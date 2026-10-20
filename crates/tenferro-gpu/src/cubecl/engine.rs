//! The runtime-facing side of [`CudaEngine`]: prepared-program execution,
//! pending-run retirement, publication and teardown.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro_runtime::{EngineId, EngineTarget, EventDomainRun, ExecutionContextIdentity};

use super::event_domain::CudaEventDomainDriver;
use super::transfer::{publish_in_stream, PublicationAdmissionError, SourcePublication};
use super::{CudaEngine, MAX_PENDING_RUNS};

/// Why a [`CudaEngine`] refuses an operation.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaEngineError;
///
/// let error = CudaEngineError::Broken;
/// assert!(error.to_string().contains("panicked"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CudaEngineError {
    /// An earlier operation on this engine panicked. Waiting for and cleaning up
    /// its earlier work still runs; replace the engine to continue.
    #[error("the CUDA engine is broken: an earlier operation panicked; replace the engine")]
    Broken,
    /// The device reported a fault CUDA makes sticky (an illegal address, a
    /// device-side assertion, an ECC error, ...). The device's driver context can
    /// no longer be used in this process: every engine on the device fails the
    /// same way, and opening the device again is not a recovery.
    #[error(
        "the CUDA device is poisoned ({reason}); its context cannot be used again in this \
         process, and reopening the device is not a recovery"
    )]
    DevicePoisoned {
        /// What the device reported.
        reason: String,
    },
}

/// Devices whose driver context a sticky fault has already destroyed in this process.
///
/// A sticky fault belongs to the device's primary context, not to one engine, so every engine
/// opened on that device must fail the same way; reopening it is not a recovery.
fn poisoned_devices() -> &'static Mutex<HashMap<[u8; 16], String>> {
    static POISONED: OnceLock<Mutex<HashMap<[u8; 16], String>>> = OnceLock::new();
    POISONED.get_or_init(|| Mutex::new(HashMap::new()))
}

fn poisoned_device_reason(uuid: [u8; 16]) -> Option<String> {
    poisoned_devices()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&uuid)
        .cloned()
}

fn mark_device_poisoned(uuid: [u8; 16], reason: String) {
    poisoned_devices()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .entry(uuid)
        .or_insert(reason);
}

/// CUDA statuses the driver makes sticky for the whole context; the driver API documents each
/// with "the context cannot be used, so it must be destroyed".
fn sticky_cuda_status(status: cudarc::driver::sys::CUresult) -> bool {
    use cudarc::driver::sys::CUresult;
    matches!(
        status,
        CUresult::CUDA_ERROR_ECC_UNCORRECTABLE
            | CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
            | CUresult::CUDA_ERROR_LAUNCH_TIMEOUT
            | CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
            | CUresult::CUDA_ERROR_ASSERT
            | CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR
            | CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
            | CUresult::CUDA_ERROR_MISALIGNED_ADDRESS
            | CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE
            | CUresult::CUDA_ERROR_INVALID_PC
            | CUresult::CUDA_ERROR_LAUNCH_FAILED
    )
}

/// The sticky device fault an error carries anywhere in its source chain, if any.
#[cfg_attr(test, allow(unused))]
pub(crate) fn device_poison_reason(error: &crate::Error) -> Option<String> {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = source {
        if current
            .downcast_ref::<cubecl_runtime::poison::DevicePoison>()
            .is_some()
        {
            return Some(current.to_string());
        }
        if let Some(launch) = current.downcast_ref::<cubecl_runtime::server::LaunchError>() {
            if launch.is_device_poisoned() {
                return Some(current.to_string());
            }
        }
        if let Some(server) = current.downcast_ref::<cubecl_runtime::server::ServerError>() {
            if server.is_device_poisoned() {
                return Some(current.to_string());
            }
        }
        if let Some(io) = current.downcast_ref::<cubecl_runtime::server::IoError>() {
            if io.is_device_poisoned() {
                return Some(current.to_string());
            }
        }
        if let Some(driver) = current.downcast_ref::<cudarc::driver::DriverError>() {
            if sticky_cuda_status(driver.0) {
                return Some(current.to_string());
            }
        }
        source = current.source();
    }
    None
}

impl CudaEngine {
    /// Run one operation of this engine.
    ///
    /// Refuses the operation when the engine is broken or its device poisoned, admits it against
    /// the unfinished-memory limit, and records a panic (engine broken) or a sticky device fault
    /// (device poisoned) it raises.
    pub(crate) fn guarded<R>(
        &mut self,
        op: &'static str,
        f: impl FnOnce(&mut Self) -> crate::Result<R>,
    ) -> crate::Result<R> {
        #[cfg(test)]
        let device_uuid = self
            .test_device_uuid
            .unwrap_or_else(|| self.runtime().device_info().uuid().to_bytes());
        #[cfg(not(test))]
        let device_uuid = self.runtime().device_info().uuid().to_bytes();
        if let Some(reason) = self
            .device_poisoned
            .clone()
            .or_else(|| poisoned_device_reason(device_uuid))
        {
            // Record the device-wide reason on this engine so its public accessor reports it too.
            self.device_poisoned = Some(reason.clone());
            return Err(crate::Error::runtime_state_source(
                op,
                CudaEngineError::DevicePoisoned { reason },
            ));
        }
        if self.broken {
            return Err(crate::Error::runtime_state_source(
                op,
                CudaEngineError::Broken,
            ));
        }
        let _reservation = self.runtime().admit_unfinished(op, 0)?;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self))) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => {
                if let Some(reason) = device_poison_reason(&error) {
                    mark_device_poisoned(device_uuid, reason.clone());
                    self.device_poisoned = Some(reason);
                }
                Err(error)
            }
            Err(payload) => {
                self.broken = true;
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Test seam: record the device-wide poison a sticky fault would have caused.
    ///
    /// A real sticky fault cannot be induced without destroying the test process's CUDA context, so
    /// the device fault itself is unit-tested on the status classification and this seam proves the
    /// cross-engine propagation.
    #[cfg(test)]
    pub(crate) fn use_test_device_uuid(&mut self, uuid: [u8; 16]) {
        self.test_device_uuid = Some(uuid);
    }

    #[cfg(test)]
    pub(crate) fn record_device_poison_for_test(&mut self, reason: &str) {
        let uuid = self
            .test_device_uuid
            .unwrap_or_else(|| self.runtime().device_info().uuid().to_bytes());
        mark_device_poisoned(uuid, reason.to_owned());
        self.device_poisoned = Some(reason.to_owned());
    }

    /// Whether an earlier operation panicked; a broken engine refuses operations.
    ///
    /// # Examples
    ///
    /// ```
    /// let _check: fn(&tenferro_gpu::cuda::CudaEngine) -> bool = tenferro_gpu::cuda::CudaEngine::is_broken;
    /// ```
    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// The sticky device fault this engine observed, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// let _check: fn(&tenferro_gpu::cuda::CudaEngine) -> Option<&str> =
    ///     tenferro_gpu::cuda::CudaEngine::device_poisoned;
    /// ```
    pub fn device_poisoned(&self) -> Option<&str> {
        self.device_poisoned.as_deref()
    }

    /// Limit the device memory that unfinished work of this engine may retain.
    ///
    /// Unfinished work retains memory until the device is done with it: vendor workspaces and
    /// transfer packages waiting for their completion event (tensors the caller still holds and
    /// idle recycled memory are not counted). Before each operation the engine releases completed
    /// work and, if the retained bytes plus what the operation will retain would exceed the limit,
    /// waits for its stream. A transfer that would retain more than the limit by itself is refused
    /// before anything is submitted. `None` (the default) sets no limit.
    ///
    /// # Examples
    ///
    /// ```
    /// let _set: fn(&mut tenferro_gpu::cuda::CudaEngine, Option<u64>) =
    ///     tenferro_gpu::cuda::CudaEngine::set_unfinished_memory_limit;
    /// ```
    pub fn set_unfinished_memory_limit(&mut self, bytes: Option<u64>) {
        self.runtime().set_unfinished_memory_limit(bytes);
    }

    /// The unfinished-memory limit, if one is set.
    ///
    /// # Examples
    ///
    /// ```
    /// let _get: fn(&tenferro_gpu::cuda::CudaEngine) -> Option<u64> =
    ///     tenferro_gpu::cuda::CudaEngine::unfinished_memory_limit;
    /// ```
    pub fn unfinished_memory_limit(&self) -> Option<u64> {
        self.runtime().unfinished_memory_limit()
    }

    /// Device bytes currently retained by this engine's unfinished work.
    ///
    /// # Examples
    ///
    /// ```
    /// let _bytes: fn(&tenferro_gpu::cuda::CudaEngine) -> u64 =
    ///     tenferro_gpu::cuda::CudaEngine::unfinished_memory_bytes;
    /// ```
    pub fn unfinished_memory_bytes(&self) -> u64 {
        self.runtime().unfinished_memory_bytes()
    }

    /// The runtime engine id this engine runs prepared programs as.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// let _id: fn(&CudaEngine) -> &tenferro_runtime::EngineId = CudaEngine::runtime_engine_id;
    /// ```
    pub fn runtime_engine_id(&self) -> &EngineId {
        &self.runtime_engine_id
    }

    /// Run prepared programs under a runtime engine id other than the default;
    /// register the engine with [`super::cuda_runtime_engine_registration`]
    /// after choosing it. Two engines registered in one runtime need distinct ids.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// let _with: fn(CudaEngine, tenferro_runtime::EngineId) -> CudaEngine =
    ///     CudaEngine::with_runtime_engine_id;
    /// ```
    #[must_use]
    pub fn with_runtime_engine_id(mut self, id: EngineId) -> Self {
        self.runtime_engine_id = id;
        self
    }

    /// Publish a value produced on this engine so it can be read later, after
    /// the engine or the tensor handle is gone.
    ///
    /// The publication keeps the value, its allocation and a frontier event
    /// recorded on this engine's stream. Every write to an engine's memory is
    /// queued on its stream, so the frontier covers all of them.
    ///
    /// # Errors
    ///
    /// Returns [`PublicationAdmissionError::Rejected`] when the value cannot be
    /// published (an unserved dtype, a host tensor, an unresolvable allocation
    /// or a producer failure reported at dispatch) and hands the value back.
    /// Returns [`PublicationAdmissionError::Unproven`] when the frontier could
    /// not be recorded after a possible enqueue; the value is then retained for
    /// soundness and never returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{CudaEngine, PublicationAdmissionError, SourcePublication};
    ///
    /// let _publish: fn(&mut CudaEngine, tenferro_tensor::Tensor)
    ///     -> Result<SourcePublication, PublicationAdmissionError> = CudaEngine::publish;
    /// ```
    pub fn publish(
        &mut self,
        value: tenferro_tensor::Tensor,
    ) -> Result<SourcePublication, PublicationAdmissionError> {
        let stream = self.stream;
        let runtime = self.runtime().clone();
        stream.executes(|| publish_in_stream(&runtime, stream, value))
    }

    /// Retire every prepared-program run this engine still holds, waiting for
    /// their device work. Returns the first retirement error; every run is
    /// retired (or retained for soundness by its own drop) either way.
    pub(crate) fn retire_pending_runs(&mut self) -> crate::Result<()> {
        let mut first_error = None;
        for mut run in self.pending_runs.drain(..) {
            if let Err(error) = run.drain() {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            None => Ok(()),
            Some(error) => Err(crate::Error::runtime_state_source(
                "CudaEngine::retire_pending_runs",
                error,
            )),
        }
    }
}

impl tenferro_runtime::ExecutionEngine for CudaEngine {
    fn target(&self) -> EngineTarget {
        EngineTarget::new(
            self.runtime_engine_id.clone(),
            ExecutionContextIdentity::of::<CudaEngine>(),
            Some(self.runtime().allocation_domain_id()),
        )
    }

    fn take_extension_caches(&mut self) -> tenferro_runtime::ExtensionCacheStore {
        std::mem::take(&mut self.extension_caches)
    }

    fn restore_extension_caches(&mut self, caches: tenferro_runtime::ExtensionCacheStore) {
        self.extension_caches = caches;
    }

    fn mark_broken(&mut self) {
        self.broken = true;
    }

    fn event_domain_driver(&self) -> Arc<dyn tenferro_runtime::EventDomainDriver> {
        Arc::new(CudaEventDomainDriver::new(
            self.runtime().clone(),
            self.stream,
        ))
    }

    fn retire_run(&mut self, run: Box<dyn EventDomainRun>) -> tenferro_runtime::Result<()> {
        // The run's device work stays ordered on this engine's stream, so its
        // outputs are usable by later operations right away; it is retired at
        // the next flush/synchronize, at drop, or when too many accumulate.
        self.pending_runs.push(run);
        if self.pending_runs.len() > MAX_PENDING_RUNS {
            let mut oldest = self.pending_runs.remove(0);
            oldest.drain()?;
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Drop for CudaEngine {
    /// Best-effort retirement: wait for the engine's stream and retire pending
    /// runs. Drop cannot report errors; memory whose completion cannot be
    /// proven stays retained by the runtime's own soundness policy.
    fn drop(&mut self) {
        let stream = self.stream;
        let _ = stream.executes(|| self.runtime().synchronize());
        let _ = self.retire_pending_runs();
    }
}
