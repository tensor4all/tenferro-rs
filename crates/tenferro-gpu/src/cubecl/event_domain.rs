use std::any::Any;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cubecl_environment::stream::StreamId;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags, CUstream};
use tenferro_runtime::runtime::{EventDomainDriver, EventDomainId, EventDomainRun, EventToken};
use tenferro_runtime::Error as RuntimeError;

use super::CudaRuntime;
use crate::event_domain_admission::admit_event_tokens;
use crate::event_retirement::{
    best_effort_retirement, retire_pending, take_pending_retirement, EventDomainRunState,
};

const EVENT_OP: &str = "cuda_event_domain";

#[derive(Clone, Debug)]
pub(super) struct CudaEventDomainDriver {
    runtime: CudaRuntime,
    /// The stream of the engine this driver was taken from; every run of the
    /// driver records and waits on it.
    stream_id: StreamId,
}

impl CudaEventDomainDriver {
    pub(super) fn new(runtime: CudaRuntime, stream_id: StreamId) -> Self {
        Self { runtime, stream_id }
    }
}

pub(super) fn admit_cuda_tokens<'a, R>(
    dependencies: &'a [Arc<dyn EventToken>],
    expected: EventDomainId,
    launch: impl FnOnce(&[&'a CudaEventToken]) -> tenferro_runtime::Result<R>,
) -> tenferro_runtime::Result<R> {
    admit_event_tokens::<CudaEventToken, R>(dependencies, expected, "non-CUDA event token", launch)
}

impl EventDomainDriver for CudaEventDomainDriver {
    fn begin_run(
        &self,
        domain: EventDomainId,
    ) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(CudaEventDomainRun {
            runtime: self.runtime.clone(),
            domain,
            stream_id: self.stream_id,
            state: EventDomainRunState::Pending,
            coalesced: false,
        }))
    }
}

#[derive(Debug)]
struct CudaEventDomainRun {
    runtime: CudaRuntime,
    domain: EventDomainId,
    stream_id: StreamId,
    state: EventDomainRunState,
    /// Set once by [`EventDomainRun::enable_single_domain_coalescing`] when the
    /// scheduler proved that this run is the only domain and has no transfers.
    coalesced: bool,
}

impl EventDomainRun for CudaEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enable_single_domain_coalescing(&mut self) {
        self.coalesced = true;
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        if self.coalesced {
            return self.enqueue_coalesced(launch);
        }
        admit_cuda_tokens(dependencies, self.domain, |dependencies| {
            self.stream_id
                .executes(|| self.enqueue_on_current_stream(dependencies, launch))
        })
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        if !take_pending_retirement(&mut self.state) {
            return Ok(());
        }
        match catch_unwind(AssertUnwindSafe(|| {
            self.stream_id
                .executes(|| retire_cuda_run(&self.runtime).map_err(RuntimeError::from))
        })) {
            Ok(Ok(())) => {
                self.state = EventDomainRunState::Retired;
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                CudaEventDomainPanic,
            ))),
        }
    }
}

impl CudaEventDomainRun {
    /// Enqueue one coalesced node.
    ///
    /// Same-stream submission order already orders the run's nodes, so this
    /// skips the per-node CUDA event and dependency waits. The run's final
    /// [`EventDomainRun::drain`] dispatch fence and retirement barrier still
    /// cover every launch, including output collection after the last node.
    fn enqueue_coalesced(
        &mut self,
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.stream_id.executes(|| {
            self.runtime
                .set_current_cuda_context(EVENT_OP)
                .map_err(RuntimeError::from)?;
            let stream = raw_stream(&self.runtime).map_err(RuntimeError::from)?;
            let mut cleanup = SubmissionCleanupGuard::new(&self.runtime, stream);
            if let Err(launch_error) = launch() {
                return Err(cleanup.finish_with_error(launch_error));
            }
            cleanup.disarm();
            Ok(Arc::new(CoalescedEventToken {
                domain: self.domain,
                runtime: self.runtime.clone(),
                stream_id: self.stream_id,
            }) as Arc<dyn EventToken>)
        })
    }

    fn enqueue_on_current_stream(
        &mut self,
        dependencies: &[&CudaEventToken],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.runtime
            .set_current_cuda_context(EVENT_OP)
            .map_err(RuntimeError::from)?;
        let stream = raw_stream(&self.runtime).map_err(RuntimeError::from)?;

        for &dependency in dependencies {
            unsafe {
                cuda_result::stream::wait_event(
                    stream,
                    dependency.event.raw(),
                    CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                )
            }
            .map_err(cuda_backend_error)?;
        }

        let event = CudaEventHandle::new(self.runtime.clone()).map_err(RuntimeError::from)?;
        let mut cleanup = SubmissionCleanupGuard::new(&self.runtime, stream);
        if let Err(launch_error) = launch() {
            return Err(cleanup.finish_with_error(launch_error));
        }
        // CubeCL hands `#[cube]` launches to its own device thread instead of
        // this raw stream, so without this dispatch fence the event below would
        // be recorded before the launched kernels reach the stream and
        // `retire_cuda_run` would synchronize an empty stream (#2182 B3). This
        // is a dispatch fence, not a device barrier: it only makes the event
        // (and therefore the run's retirement) cover the queued launches.
        #[cfg(test)]
        self.runtime.record_scheduler_fence();
        if let Err(flush_error) = self.runtime.flush_cubecl(EVENT_OP) {
            return Err(cleanup.finish_with_error(RuntimeError::from(flush_error)));
        }
        if let Err(record_error) = event.record(stream) {
            return Err(cleanup.finish_with_error(RuntimeError::from(record_error)));
        }
        #[cfg(test)]
        self.runtime.record_scheduler_event();
        cleanup.disarm();
        Ok(Arc::new(CudaEventToken {
            domain: self.domain,
            event,
        }))
    }
}

impl Drop for CudaEventDomainRun {
    fn drop(&mut self) {
        retire_pending(&mut self.state, || {
            best_effort_retirement(|| {
                let result = self.stream_id.executes(|| retire_cuda_run(&self.runtime));
                if let Err(error) = result {
                    eprintln!(
                        "tenferro-gpu: failed to drain CUDA event-domain run during Drop: {error}"
                    );
                }
            });
        });
    }
}

struct SubmissionCleanupGuard<'a> {
    runtime: &'a CudaRuntime,
    stream: CUstream,
    armed: bool,
}

impl<'a> SubmissionCleanupGuard<'a> {
    fn new(runtime: &'a CudaRuntime, stream: CUstream) -> Self {
        Self {
            runtime,
            stream,
            armed: true,
        }
    }

    fn finish_with_error(&mut self, primary: RuntimeError) -> RuntimeError {
        let cleanup = synchronize_stream(self.runtime, self.stream);
        self.armed = false;
        match cleanup {
            Ok(()) => primary,
            Err(cleanup) => RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                CudaSubmissionCleanupError { primary, cleanup },
            )),
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SubmissionCleanupGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            best_effort_retirement(|| {
                if let Err(error) = synchronize_stream(self.runtime, self.stream) {
                    eprintln!(
                        "tenferro-gpu: failed to retire CUDA work during submission unwind: {error}"
                    );
                }
            });
        }
    }
}

#[derive(Debug)]
pub(super) struct CudaEventToken {
    domain: EventDomainId,
    event: Arc<CudaEventHandle>,
}

impl EventToken for CudaEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.event.wait()
    }
}

/// Completion token returned for each node of a coalesced single-domain run.
///
/// No per-node CUDA event is recorded on this path, so the token carries no
/// stream position of its own. It retains the run's captured [`StreamId`] so a
/// wait from any worker retires that stream, not the waiting thread's current
/// one. Eligibility proves it is never consumed by a transfer; waiting on it
/// synchronizes the run's whole stream, which over-waits rather than
/// under-waits if that precondition were ever violated.
#[derive(Debug)]
struct CoalescedEventToken {
    domain: EventDomainId,
    runtime: CudaRuntime,
    stream_id: StreamId,
}

impl EventToken for CoalescedEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.stream_id
            .executes(|| retire_cuda_run(&self.runtime).map_err(RuntimeError::from))
    }
}

struct CudaEventHandle {
    runtime: CudaRuntime,
    raw: RawCudaEvent,
}

#[derive(Debug)]
struct RawCudaEvent(CUevent);

// SAFETY: the retained CUDA primary context owns the event's device lifetime.
// Every operation selects that context first, and CUDA event record/wait/destroy
// operations are thread-safe for an externally synchronized handle.
unsafe impl Send for RawCudaEvent {}
// SAFETY: shared access only records or waits through CUDA's thread-safe driver
// API; destruction requires unique ownership of the enclosing handle.
unsafe impl Sync for RawCudaEvent {}

impl CudaEventHandle {
    fn new(runtime: CudaRuntime) -> crate::Result<Arc<Self>> {
        runtime.set_current_cuda_context(EVENT_OP)?;
        let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING)
            .map_err(|source| crate::Error::backend_source(EVENT_OP, source))?;
        Ok(Arc::new(Self {
            runtime,
            raw: RawCudaEvent(event),
        }))
    }

    fn raw(&self) -> CUevent {
        self.raw.0
    }

    fn record(&self, stream: CUstream) -> crate::Result<()> {
        self.runtime.set_current_cuda_context(EVENT_OP)?;
        unsafe { cuda_result::event::record(self.raw(), stream) }
            .map_err(|source| crate::Error::backend_source(EVENT_OP, source))
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.runtime
            .set_current_cuda_context(EVENT_OP)
            .map_err(RuntimeError::from)?;
        unsafe { cuda_result::event::synchronize(self.raw()) }.map_err(cuda_backend_error)
    }
}

impl fmt::Debug for CudaEventHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaEventHandle")
            .field("device_ordinal", &self.runtime.device_ordinal())
            .finish_non_exhaustive()
    }
}

impl Drop for CudaEventHandle {
    fn drop(&mut self) {
        best_effort_retirement(|| {
            if let Err(error) = self.runtime.set_current_cuda_context(EVENT_OP) {
                eprintln!(
                    "tenferro-gpu: failed to select CUDA context before event destruction: {error}"
                );
                return;
            }
            if let Err(error) = unsafe { cuda_result::event::destroy(self.raw()) } {
                eprintln!("tenferro-gpu: failed to destroy CUDA event during Drop: {error:?}");
            }
        });
    }
}

fn raw_stream(runtime: &CudaRuntime) -> crate::Result<CUstream> {
    runtime
        .raw_cuda_stream()
        .map(|stream| stream as usize as CUstream)
}

fn synchronize_stream(runtime: &CudaRuntime, stream: CUstream) -> crate::Result<()> {
    // Dispatch CubeCL's host queue before the raw barrier, so the barrier covers
    // the launches it still holds (#2182 B3). The barrier is attempted even when
    // the dispatch fails, and the dispatch result is folded into the barrier
    // result instead of replacing it.
    #[cfg(test)]
    runtime.record_scheduler_fence();
    let dispatch = runtime.flush_cubecl(EVENT_OP);
    let activation = runtime.set_current_cuda_context(EVENT_OP);
    // `cuStreamSynchronize` is the retirement barrier even when it reports an
    // asynchronous launch failure discovered while waiting. Invalid
    // context/handle errors mean the retained runtime domain is unusable. The
    // barrier is still attempted when context selection reports an error so
    // cleanup never returns before trying either retirement path.
    let barrier = unsafe { cuda_result::stream::synchronize(stream) };
    #[cfg(test)]
    runtime.record_scheduler_retirement();
    let barrier = match (activation, barrier) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(activation), Ok(())) => Err(crate::Error::backend_source(
            EVENT_OP,
            CudaStreamActivationError { activation },
        )),
        (Ok(()), Err(barrier)) => Err(crate::Error::backend_source(EVENT_OP, barrier)),
        (Err(activation), Err(barrier)) => Err(crate::Error::backend_source(
            EVENT_OP,
            CudaStreamRetirementError {
                activation,
                barrier,
            },
        )),
    };
    match (dispatch, barrier) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(dispatch), Ok(())) => Err(crate::Error::backend_source(
            EVENT_OP,
            CudaStreamDispatchError { dispatch },
        )),
        (Ok(()), Err(barrier)) => Err(barrier),
        (Err(dispatch), Err(barrier)) => Err(crate::Error::backend_source(
            EVENT_OP,
            CudaStreamDispatchAndRetirementError { dispatch, barrier },
        )),
    }
}

fn retire_cuda_run(runtime: &CudaRuntime) -> crate::Result<()> {
    let stream = match raw_stream(runtime) {
        Ok(stream) => return synchronize_stream(runtime, stream),
        Err(primary) => primary,
    };
    // Stream lookup can fail after work was admitted. Dispatch the queued CubeCL
    // launches first so the context-wide fallback barrier can cover them; a
    // dispatch failure is then kept as the primary error with the barrier
    // outcome preserved.
    let dispatch = runtime.flush_cubecl(EVENT_OP);
    // A context-wide barrier is the retirement fallback; returning the lookup
    // error after it succeeds preserves the operational failure without
    // releasing live storage.
    let activation = runtime.set_current_cuda_context(EVENT_OP);
    let fallback = cuda_result::ctx::synchronize();
    #[cfg(test)]
    runtime.record_scheduler_retirement();
    let barrier = match (activation, fallback) {
        (Ok(()), Ok(())) => stream,
        (Err(activation), Ok(())) => crate::Error::backend_source(
            EVENT_OP,
            CudaContextActivationFallbackError {
                primary: stream,
                activation,
            },
        ),
        (Ok(()), Err(fallback)) => crate::Error::backend_source(
            EVENT_OP,
            CudaRetirementFallbackError {
                primary: stream,
                fallback,
            },
        ),
        (Err(activation), Err(fallback)) => crate::Error::backend_source(
            EVENT_OP,
            CudaRetirementFallbackCombinedError {
                primary: stream,
                activation,
                fallback,
            },
        ),
    };
    match dispatch {
        Ok(()) => Err(barrier),
        Err(dispatch) => Err(crate::Error::backend_source(
            EVENT_OP,
            CudaRetirementDispatchError { dispatch, barrier },
        )),
    }
}

fn cuda_backend_error(source: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    RuntimeError::from(crate::Error::backend_source(EVENT_OP, source))
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA event-domain operation panicked")]
struct CudaEventDomainPanic;

#[derive(Debug, thiserror::Error)]
#[error("CUDA stream lookup failed ({primary}); context retirement also reported {fallback}")]
struct CudaRetirementFallbackError {
    primary: crate::Error,
    #[source]
    fallback: cudarc::driver::DriverError,
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA context selection reported {activation} before stream retirement completed")]
struct CudaStreamActivationError {
    #[source]
    activation: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA context selection failed ({activation}); stream retirement also reported {barrier}")]
struct CudaStreamRetirementError {
    activation: crate::Error,
    #[source]
    barrier: cudarc::driver::DriverError,
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA stream lookup failed ({primary}); context selection also reported {activation}")]
struct CudaContextActivationFallbackError {
    primary: crate::Error,
    #[source]
    activation: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "CUDA stream lookup failed ({primary}); context selection failed ({activation}); \
     context retirement also reported {fallback}"
)]
struct CudaRetirementFallbackCombinedError {
    primary: crate::Error,
    activation: crate::Error,
    #[source]
    fallback: cudarc::driver::DriverError,
}

#[derive(Debug, thiserror::Error)]
#[error("submission failed ({primary}); CUDA stream cleanup also failed ({cleanup})")]
struct CudaSubmissionCleanupError {
    primary: RuntimeError,
    #[source]
    cleanup: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA CubeCL dispatch before stream retirement failed")]
struct CudaStreamDispatchError {
    #[source]
    dispatch: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error("CUDA CubeCL dispatch failed ({dispatch}); the stream retirement barrier also failed ({barrier})")]
struct CudaStreamDispatchAndRetirementError {
    dispatch: crate::Error,
    #[source]
    barrier: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "CUDA CubeCL dispatch before the context retirement fallback failed ({dispatch}); \
     the fallback barrier also reported {barrier}"
)]
struct CudaRetirementDispatchError {
    dispatch: crate::Error,
    #[source]
    barrier: crate::Error,
}
