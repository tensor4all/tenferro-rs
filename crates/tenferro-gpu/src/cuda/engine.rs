//! The CUDA engine: one runtime, one submission owner, one module cache.

use std::marker::PhantomData;

use crate::cuda::{CudaDeviceError, CudaDeviceId};
use crate::gpu::GpuError;

use super::runtime::CudaRuntime;

/// A failure of the engine itself.
#[derive(Debug)]
#[non_exhaustive]
#[doc(hidden)]
pub enum CudaEngineError {
    /// The runtime could not be opened.
    Open(CudaDeviceError),
    /// The engine was asked to plan or launch an operation it does not serve.
    Seam(GpuError),
}

impl std::fmt::Display for CudaEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaEngineError::Open(error) => write!(f, "cannot open the CUDA engine: {error}"),
            CudaEngineError::Seam(error) => write!(f, "CUDA engine failure: {error}"),
        }
    }
}

impl std::error::Error for CudaEngineError {}

impl From<CudaDeviceError> for CudaEngineError {
    fn from(error: CudaDeviceError) -> Self {
        CudaEngineError::Open(error)
    }
}

impl From<GpuError> for CudaEngineError {
    fn from(error: GpuError) -> Self {
        CudaEngineError::Seam(error)
    }
}

/// How many prepared-program runs the engine retires lazily before draining the oldest.
const MAX_PENDING_RUNS: usize = 16;

/// The runtime engine id this backend registers under.
///
/// The same identifier the CubeCL-backed engine uses, because both are implementations of the
/// same backend: the cutover replaces the implementation, not the registration name.
pub(crate) const DEFAULT_RUNTIME_ENGINE_ID: &str = "tenferro.cuda.default.v1";

/// The built-in CUDA engine id, which is a valid identifier by construction.
pub(crate) fn default_runtime_engine_id() -> tenferro_runtime::EngineId {
    match tenferro_runtime::EngineId::new(DEFAULT_RUNTIME_ENGINE_ID) {
        Ok(id) => id,
        Err(_) => unreachable!("the built-in CUDA engine id is a valid identifier"),
    }
}

/// A CUDA engine: one runtime, one numerical submission path, one module cache.
///
/// `Send`, not `Sync`: submission borrows the runtime mutably on the calling thread.
/// A downstream crate that hands an engine to one worker at a time is unaffected;
/// two threads submitting concurrently must serialise outside the engine. Explicit
/// host transfers go through [`CudaRuntime`](super::CudaRuntime), which serialises on
/// the same runtime's submission lock.
#[doc(hidden)]
pub struct CudaEngine {
    runtime: CudaRuntime,
    /// The cuTENSOR handle, created on the first contraction: a plan cache lives beside
    /// it, so a contraction reuses its plan instead of planning on every call.
    cutensor: Option<super::Cutensor>,
    /// The runtime engine id this engine runs prepared programs as.
    runtime_engine_id: tenferro_runtime::EngineId,
    /// Extension caches the runtime lends to extension executors during a run.
    pub(crate) extension_caches: std::cell::RefCell<tenferro_runtime::ExtensionCacheStore>,
    /// Dispatchers whose translation units are already prepared. Preparing one is a runtime
    /// round trip, so an operation reuses the set its predecessors built.
    pub(crate) prepared: std::cell::RefCell<super::dispatch::PreparedDispatchers>,
    /// Prepared-program runs whose device work may still be in flight; they are retired at the
    /// next synchronise, when the queue grows past [`MAX_PENDING_RUNS`], or at drop.
    pending_runs: Vec<Box<dyn tenferro_runtime::EventDomainRun>>,
    /// Set when the runtime reported an unrecoverable failure.
    broken: bool,
    /// How many borrowed views this engine materialized into a compact operand copy.
    ///
    /// A native read entry consumes an operand's own layout, so this stays put; it moves only
    /// where an operation still has to stage a view (a rank the by-value layout cannot carry,
    /// or an entry that has no stride-consuming kernel yet). Measurement for the read-entry
    /// tests, not a public surface: [`CudaEngine::staged_reads`] is crate-private.
    staged_reads: std::cell::Cell<u64>,
    // `!Sync` explicitly rather than by inheritance from the runtime's auto traits.
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// The engine moves between threads with its runtime and is deliberately not shared:
// `_not_sync` above makes `CudaEngine: !Sync`. `Send` is asserted here.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<CudaEngine>();
};

impl CudaEngine {
    /// Open the CUDA engine on `device_id`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() {
    /// use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
    ///
    /// # #[cfg(feature = "cuda")]
    /// let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0));
    /// # let _ = engine;
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`CudaEngineError::Open`] when the device is unknown or the driver cannot open
    /// it.
    #[doc(hidden)]
    pub fn open(device_id: CudaDeviceId) -> Result<CudaEngine, CudaEngineError> {
        Ok(CudaEngine {
            runtime: CudaRuntime::new(device_id)?,
            cutensor: None,
            runtime_engine_id: default_runtime_engine_id(),
            extension_caches: std::cell::RefCell::new(
                tenferro_runtime::ExtensionCacheStore::default(),
            ),
            prepared: std::cell::RefCell::new(super::dispatch::PreparedDispatchers::default()),
            pending_runs: Vec::new(),
            broken: false,
            staged_reads: std::cell::Cell::new(0),
            _not_sync: PhantomData,
        })
    }

    /// Publish a produced value with a frontier a reader can wait on.
    ///
    /// The frontier is one completion checkpoint recorded on this engine's stream, so
    /// admission never synchronises; a reader polls
    /// [`SourcePublication::frontier_ready`](super::publication::SourcePublication::frontier_ready)
    /// or waits for it.
    ///
    /// # Errors
    ///
    /// [`PublicationAdmissionError::Rejected`] when the value is not a CUDA allocation
    /// of this engine, and [`PublicationAdmissionError::Runtime`] when the checkpoint
    /// cannot be recorded.
    #[doc(hidden)]
    pub fn publish(
        &mut self,
        value: tenferro_tensor::Tensor,
    ) -> Result<super::publication::SourcePublication, super::publication::PublicationAdmissionError>
    {
        use super::publication::{
            PublicationAdmissionError, RejectedPublication, SourcePublication,
        };

        let Some(storage) = super::pending::cuda_storage(&value) else {
            return Err(PublicationAdmissionError::Rejected(Box::new(
                RejectedPublication {
                    value,
                    reason: "the value is not a CUDA allocation",
                },
            )));
        };
        // A reader waits on *this* engine's stream frontier, so a value from another
        // allocation domain (a second engine on the same ordinal included) is not ordered by it.
        if storage.device_ordinal() != self.runtime.device_ordinal()
            || storage.domain() != self.runtime().allocation_domain_id()
        {
            return Err(PublicationAdmissionError::Rejected(Box::new(
                RejectedPublication {
                    value,
                    reason: "the value belongs to another engine",
                },
            )));
        }
        let frontier = {
            let mut guard = self.runtime.submit();
            guard
                .checkpoint()
                .map_err(|error| PublicationAdmissionError::Runtime(Box::new(error)))?
        };
        Ok(SourcePublication::new(
            value,
            frontier,
            std::sync::Arc::clone(&self.runtime.inner),
        ))
    }

    /// The runtime engine id this engine registers under.
    #[doc(hidden)]
    pub fn runtime_engine_id(&self) -> &tenferro_runtime::EngineId {
        &self.runtime_engine_id
    }

    /// Identity of the runtime this engine owns.
    #[doc(hidden)]
    pub fn runtime_identity(&self) -> super::runtime::CudaRuntimeIdentity {
        self.runtime.identity().clone()
    }

    /// The device this engine opened.
    pub(crate) fn device_id(&self) -> super::CudaDeviceId {
        self.runtime.device_id()
    }

    /// Record one operand materialization for [`CudaEngine::staged_reads`].
    pub(crate) fn note_staged_read(&self) {
        self.staged_reads.set(self.staged_reads.get() + 1);
    }

    /// Refuse numerical submission on an engine the runtime marked broken.
    ///
    /// # Errors
    ///
    /// [`CudaEngineError::Seam`] with the seam's `Broken` cause after an unrecoverable failure; the
    /// engine can still synchronise and retire, which is what teardown needs.
    pub(crate) fn ensure_healthy(&self) -> Result<(), CudaEngineError> {
        if self.broken {
            return Err(CudaEngineError::Seam(crate::gpu::GpuError::Broken {
                detail: "the runtime marked this engine broken".to_string(),
            }));
        }
        Ok(())
    }

    /// The engine's cuTENSOR handle, created on first use.
    ///
    /// # Errors
    ///
    /// [`CudaEngineError::Seam`] when the library cannot be loaded or the handle cannot
    /// be created on this runtime's context.
    pub(crate) fn cutensor(&mut self) -> Result<&mut super::Cutensor, CudaEngineError> {
        if self.cutensor.is_none() {
            self.cutensor =
                Some(super::Cutensor::new(&self.runtime).map_err(CudaEngineError::from)?);
        }
        Ok(self.cutensor.as_mut().expect("set above"))
    }

    /// The runtime this engine owns.
    #[doc(hidden)]
    pub fn runtime(&self) -> &CudaRuntime {
        &self.runtime
    }

    /// The runtime this engine owns, mutably, for numerical submission.
    #[doc(hidden)]
    pub fn runtime_mut(&mut self) -> &mut CudaRuntime {
        &mut self.runtime
    }
}

impl std::fmt::Debug for CudaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEngine")
            .field("device", &self.runtime.device_id())
            .finish()
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        // Best effort: drain the stream so a publication's frontier is complete when
        // the engine goes away. A buffer keeps the allocation domain alive; this
        // never closes a runtime that another handle still uses.
        self.runtime.synchronize_best_effort();
    }
}

impl tenferro_runtime::ExecutionEngine for CudaEngine {
    fn target(&self) -> tenferro_runtime::EngineTarget {
        tenferro_runtime::EngineTarget::new(
            self.runtime_engine_id.clone(),
            tenferro_runtime::ExecutionContextIdentity::of::<CudaEngine>(),
            Some(self.runtime.allocation_domain_id()),
        )
    }

    fn take_extension_caches(&mut self) -> tenferro_runtime::ExtensionCacheStore {
        std::mem::take(&mut *self.extension_caches.borrow_mut())
    }

    fn restore_extension_caches(&mut self, caches: tenferro_runtime::ExtensionCacheStore) {
        *self.extension_caches.borrow_mut() = caches;
    }

    fn mark_broken(&mut self) {
        self.broken = true;
    }

    fn event_domain_driver(&self) -> std::sync::Arc<dyn tenferro_runtime::EventDomainDriver> {
        std::sync::Arc::new(super::event_domain::StreamEventDomainDriver::new(
            self.runtime.clone(),
        ))
    }

    fn retire_run(
        &mut self,
        run: Box<dyn tenferro_runtime::EventDomainRun>,
    ) -> tenferro_runtime::Result<()> {
        // The run's device work stays ordered on this engine's stream, so its outputs are usable
        // by later operations right away; it is retired at the next synchronise, at drop, or when
        // too many accumulate.
        self.pending_runs.push(run);
        if self.pending_runs.len() > MAX_PENDING_RUNS {
            let mut oldest = self.pending_runs.remove(0);
            oldest.drain()?;
        }
        Ok(())
    }

    fn synchronize(&mut self) -> tenferro_runtime::Result<()> {
        let synchronized = self
            .runtime
            .synchronize()
            .map_err(super::event_domain::event_domain_error);
        // Every queued run is retired even when the runtime itself failed: otherwise a broken
        // runtime would keep its runs queued forever.
        let mut first_error = None;
        for mut run in self.pending_runs.drain(..) {
            if let Err(error) = run.drain() {
                first_error.get_or_insert(error);
            }
        }
        match (synchronized, first_error) {
            (Ok(()), None) => Ok(()),
            (Err(error), _) | (_, Some(error)) => Err(error),
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
