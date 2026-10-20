//! The runtime-facing side of [`CudaEngine`]: prepared-program execution,
//! pending-run retirement, publication and teardown.

use std::sync::Arc;

use tenferro_runtime::{EngineId, EngineTarget, EventDomainRun, ExecutionContextIdentity};

use super::event_domain::CudaEventDomainDriver;
use super::transfer::{publish_in_stream, PublicationAdmissionError, SourcePublication};
use super::{CudaEngine, MAX_PENDING_RUNS};

impl CudaEngine {
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
