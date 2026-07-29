use std::any::Any;
use std::sync::Arc;

use cubecl::stream_id::StreamId;
use cubecl_wgpu::WgpuSubmission;
use tenferro_runtime::runtime::{EventDomainDriver, EventDomainRun, EventToken};
use tenferro_runtime::Error as RuntimeError;

use super::WebGpuRuntime;

const EVENT_OP: &str = "webgpu_event_domain";

#[derive(Clone, Debug)]
pub(super) struct WebGpuEventDomainDriver {
    runtime: WebGpuRuntime,
    identity: Arc<()>,
}

impl WebGpuEventDomainDriver {
    pub(super) fn new(runtime: WebGpuRuntime) -> Self {
        Self {
            runtime,
            identity: Arc::new(()),
        }
    }
}

impl EventDomainDriver for WebGpuEventDomainDriver {
    fn begin_run(&self) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(WebGpuEventDomainRun {
            runtime: self.runtime.clone(),
            identity: Arc::clone(&self.identity),
            stream_id: StreamId::current(),
        }))
    }
}

#[derive(Debug)]
struct WebGpuEventDomainRun {
    runtime: WebGpuRuntime,
    identity: Arc<()>,
    stream_id: StreamId,
}

impl EventDomainRun for WebGpuEventDomainRun {
    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.stream_id
            .executes(|| self.enqueue_on_current_stream(dependencies, launch))
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        self.stream_id
            .executes(|| submit_and_wait(&self.runtime, self.stream_id))
    }
}

impl WebGpuEventDomainRun {
    fn enqueue_on_current_stream(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        for dependency in dependencies {
            match dependency.as_any().downcast_ref::<WebGpuEventToken>() {
                Some(token) if Arc::ptr_eq(&token.identity, &self.identity) => {
                    // WGPU preserves submission order on one queue. The completion
                    // remains a scheduler dependency without a host-side wait.
                }
                _ => dependency.wait()?,
            }
        }

        let mut cleanup = SubmissionCleanupGuard::new(&self.runtime, self.stream_id);
        if let Err(launch_error) = launch() {
            return Err(cleanup.finish_with_error(launch_error));
        }
        let submission = match submit_completion(&self.runtime, self.stream_id) {
            Ok(submission) => submission,
            Err(submit_error) => return Err(cleanup.finish_with_error(submit_error)),
        };
        cleanup.disarm();
        Ok(Arc::new(WebGpuEventToken {
            identity: Arc::clone(&self.identity),
            submission,
        }))
    }
}

impl Drop for WebGpuEventDomainRun {
    fn drop(&mut self) {
        if let Err(error) = self
            .stream_id
            .executes(|| submit_and_wait(&self.runtime, self.stream_id))
        {
            eprintln!("tenferro-gpu: failed to drain WebGPU event-domain run during Drop: {error}");
        }
    }
}

struct SubmissionCleanupGuard<'a> {
    runtime: &'a WebGpuRuntime,
    stream_id: StreamId,
    armed: bool,
}

impl<'a> SubmissionCleanupGuard<'a> {
    fn new(runtime: &'a WebGpuRuntime, stream_id: StreamId) -> Self {
        Self {
            runtime,
            stream_id,
            armed: true,
        }
    }

    fn finish_with_error(&mut self, primary: RuntimeError) -> RuntimeError {
        let cleanup = submit_and_wait(self.runtime, self.stream_id);
        self.armed = false;
        match cleanup {
            Ok(()) => primary,
            Err(cleanup) => RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                WebGpuSubmissionCleanupError { primary, cleanup },
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
            if let Err(error) = submit_and_wait(self.runtime, self.stream_id) {
                eprintln!(
                    "tenferro-gpu: failed to retire WebGPU work during submission unwind: {error}"
                );
            }
        }
    }
}

#[derive(Debug)]
struct WebGpuEventToken {
    identity: Arc<()>,
    submission: WgpuSubmission,
}

impl EventToken for WebGpuEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.submission.wait().map_err(webgpu_backend_error)
    }
}

fn submit_completion(
    runtime: &WebGpuRuntime,
    stream_id: StreamId,
) -> tenferro_runtime::Result<WgpuSubmission> {
    runtime
        .client()
        .with_server(move |server| server.submit_stream_completion(stream_id))
        .ok_or_else(|| {
            RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                WebGpuServerUnavailable,
            ))
        })?
        .map_err(webgpu_backend_error)
}

fn submit_and_wait(runtime: &WebGpuRuntime, stream_id: StreamId) -> tenferro_runtime::Result<()> {
    submit_completion(runtime, stream_id)?
        .wait()
        .map_err(webgpu_backend_error)
}

fn webgpu_backend_error(source: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    RuntimeError::from(crate::Error::backend_source(EVENT_OP, source))
}

#[derive(Debug, thiserror::Error)]
#[error("the CubeCL WebGPU server is unavailable")]
struct WebGpuServerUnavailable;

#[derive(Debug, thiserror::Error)]
#[error("submission failed ({primary}); WebGPU queue cleanup also failed ({cleanup})")]
struct WebGpuSubmissionCleanupError {
    primary: RuntimeError,
    #[source]
    cleanup: RuntimeError,
}
