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
    /// use tenferro_gpu::cuda_next::{CudaDeviceId, CudaEngine};
    ///
    /// # #[cfg(feature = "cuda-next")]
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
        if storage.device_ordinal() != self.runtime.device_ordinal() {
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
