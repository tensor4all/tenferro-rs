use std::any::Any;
use std::fmt;
use std::sync::Arc;

use cubecl::stream_id::StreamId;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent, CUevent_flags, CUevent_wait_flags, CUstream};
use tenferro_runtime::runtime::{EventDomainDriver, EventDomainRun, EventToken};
use tenferro_runtime::Error as RuntimeError;

use super::CudaRuntime;

const EVENT_OP: &str = "cuda_event_domain";

#[derive(Clone, Debug)]
pub(super) struct CudaEventDomainDriver {
    runtime: CudaRuntime,
}

impl CudaEventDomainDriver {
    pub(super) fn new(runtime: CudaRuntime) -> Self {
        Self { runtime }
    }
}

impl EventDomainDriver for CudaEventDomainDriver {
    fn begin_run(&self) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(CudaEventDomainRun {
            runtime: self.runtime.clone(),
            stream_id: StreamId::current(),
        }))
    }
}

#[derive(Debug)]
struct CudaEventDomainRun {
    runtime: CudaRuntime,
    stream_id: StreamId,
}

impl EventDomainRun for CudaEventDomainRun {
    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.stream_id
            .executes(|| self.enqueue_on_current_stream(dependencies, launch))
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        self.stream_id.executes(|| {
            let stream = raw_stream(&self.runtime).map_err(RuntimeError::from)?;
            synchronize_stream(&self.runtime, stream).map_err(RuntimeError::from)
        })
    }
}

impl CudaEventDomainRun {
    fn enqueue_on_current_stream(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.runtime
            .set_current_cuda_context(EVENT_OP)
            .map_err(RuntimeError::from)?;
        let stream = raw_stream(&self.runtime).map_err(RuntimeError::from)?;

        for dependency in dependencies {
            match dependency.as_any().downcast_ref::<CudaEventToken>() {
                Some(token)
                    if token.event.runtime.device_ordinal() == self.runtime.device_ordinal() =>
                {
                    unsafe {
                        cuda_result::stream::wait_event(
                            stream,
                            token.event.raw(),
                            CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                        )
                    }
                    .map_err(cuda_backend_error)?;
                }
                _ => dependency.wait()?,
            }
        }

        let event = CudaEventHandle::new(self.runtime.clone()).map_err(RuntimeError::from)?;
        let mut cleanup = SubmissionCleanupGuard::new(&self.runtime, stream);
        if let Err(launch_error) = launch() {
            return Err(cleanup.finish_with_error(launch_error));
        }
        if let Err(record_error) = event.record(stream) {
            return Err(cleanup.finish_with_error(RuntimeError::from(record_error)));
        }
        cleanup.disarm();
        Ok(Arc::new(CudaEventToken { event }))
    }
}

impl Drop for CudaEventDomainRun {
    fn drop(&mut self) {
        let result = self.stream_id.executes(|| {
            let stream = raw_stream(&self.runtime)?;
            synchronize_stream(&self.runtime, stream)
        });
        if let Err(error) = result {
            eprintln!("tenferro-gpu: failed to drain CUDA event-domain run during Drop: {error}");
        }
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
            if let Err(error) = synchronize_stream(self.runtime, self.stream) {
                eprintln!(
                    "tenferro-gpu: failed to retire CUDA work during submission unwind: {error}"
                );
            }
        }
    }
}

#[derive(Debug)]
struct CudaEventToken {
    event: Arc<CudaEventHandle>,
}

impl EventToken for CudaEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.event.wait()
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
        if let Err(error) = self.runtime.set_current_cuda_context(EVENT_OP) {
            eprintln!(
                "tenferro-gpu: failed to select CUDA context before event destruction: {error}"
            );
            return;
        }
        if let Err(error) = unsafe { cuda_result::event::destroy(self.raw()) } {
            eprintln!("tenferro-gpu: failed to destroy CUDA event during Drop: {error:?}");
        }
    }
}

fn raw_stream(runtime: &CudaRuntime) -> crate::Result<CUstream> {
    runtime
        .raw_cuda_stream()
        .map(|stream| stream as usize as CUstream)
}

fn synchronize_stream(runtime: &CudaRuntime, stream: CUstream) -> crate::Result<()> {
    runtime.set_current_cuda_context(EVENT_OP)?;
    // `cuStreamSynchronize` is the retirement barrier even when it reports an
    // asynchronous launch failure discovered while waiting. Invalid
    // context/handle errors instead mean the retained runtime domain is
    // unusable; callers preserve that fatal backend error and must not reuse
    // the domain.
    unsafe { cuda_result::stream::synchronize(stream) }
        .map_err(|source| crate::Error::backend_source(EVENT_OP, source))
}

fn cuda_backend_error(source: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    RuntimeError::from(crate::Error::backend_source(EVENT_OP, source))
}

#[derive(Debug, thiserror::Error)]
#[error("submission failed ({primary}); CUDA stream cleanup also failed ({cleanup})")]
struct CudaSubmissionCleanupError {
    primary: RuntimeError,
    #[source]
    cleanup: crate::Error,
}
