use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cubecl::stream_id::StreamId;
use cubecl_wgpu::WgpuSubmission;
use tenferro_runtime::runtime::{
    EventDomainDriver, EventDomainError, EventDomainId, EventDomainOperation, EventDomainRun,
    EventToken,
};
use tenferro_runtime::Error as RuntimeError;

use super::WebGpuRuntime;
use crate::event_retirement::{
    best_effort_retirement, retire_pending, take_pending_retirement, EventDomainRunState,
};

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
    fn begin_run(
        &self,
        domain: EventDomainId,
    ) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(WebGpuEventDomainRun {
            runtime: self.runtime.clone(),
            domain,
            identity: Arc::clone(&self.identity),
            stream_id: StreamId::current(),
            state: EventDomainRunState::Pending,
        }))
    }
}

#[derive(Debug)]
struct WebGpuEventDomainRun {
    runtime: WebGpuRuntime,
    domain: EventDomainId,
    identity: Arc<()>,
    stream_id: StreamId,
    state: EventDomainRunState,
}

impl EventDomainRun for WebGpuEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.stream_id
            .executes(|| self.enqueue_on_current_stream(dependencies, launch))
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        if !take_pending_retirement(&mut self.state) {
            return Ok(());
        }
        match catch_unwind(AssertUnwindSafe(|| {
            self.stream_id
                .executes(|| retire_webgpu_run(&self.runtime, self.stream_id))
        })) {
            Ok(Ok(())) => {
                self.state = EventDomainRunState::Retired;
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                WebGpuEventDomainPanic,
            ))),
        }
    }
}

impl WebGpuEventDomainRun {
    fn enqueue_on_current_stream(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        for dependency in dependencies {
            let actual = dependency.origin();
            if actual != self.domain {
                return Err(RuntimeError::from(
                    EventDomainError::DependencyDomainMismatch {
                        operation: EventDomainOperation::Enqueue,
                        node_index: None,
                        expected: self.domain,
                        actual,
                    },
                ));
            }
            let token = dependency
                .as_any()
                .downcast_ref::<WebGpuEventToken>()
                .ok_or_else(|| {
                    RuntimeError::from(EventDomainError::IncompatibleTokenType {
                        operation: EventDomainOperation::Enqueue,
                        node_index: None,
                        expected: self.domain,
                        actual,
                        token_type: "non-WebGPU event token",
                    })
                })?;
            if !Arc::ptr_eq(&token.identity, &self.identity) {
                return Err(RuntimeError::from(
                    EventDomainError::IncompatibleTokenType {
                        operation: EventDomainOperation::Enqueue,
                        node_index: None,
                        expected: self.domain,
                        actual,
                        token_type: "WebGPU event token from another queue",
                    },
                ));
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
            domain: self.domain,
            identity: Arc::clone(&self.identity),
            submission,
        }))
    }
}

impl Drop for WebGpuEventDomainRun {
    fn drop(&mut self) {
        retire_pending(&mut self.state, || {
            best_effort_retirement(|| {
                if let Err(error) = self
                    .stream_id
                    .executes(|| retire_webgpu_run(&self.runtime, self.stream_id))
                {
                    eprintln!(
                        "tenferro-gpu: failed to drain WebGPU event-domain run during Drop: {error}"
                    );
                }
            });
        });
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
        let cleanup = retire_webgpu_run(self.runtime, self.stream_id);
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
            best_effort_retirement(|| {
                if let Err(error) = retire_webgpu_run(self.runtime, self.stream_id) {
                    eprintln!(
                        "tenferro-gpu: failed to retire WebGPU work during submission unwind: {error}"
                    );
                }
            });
        }
    }
}

#[derive(Debug)]
struct WebGpuEventToken {
    domain: EventDomainId,
    identity: Arc<()>,
    submission: WgpuSubmission,
}

impl EventToken for WebGpuEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
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

fn retire_webgpu_run(runtime: &WebGpuRuntime, stream_id: StreamId) -> tenferro_runtime::Result<()> {
    let submission = match submit_completion(runtime, stream_id) {
        Ok(submission) => submission,
        Err(primary) => {
            // If the exact stream completion cannot be submitted, synchronize
            // the whole CubeCL client before allowing scheduler storage to drop.
            return match runtime.synchronize() {
                Ok(()) => Err(primary),
                Err(fallback) => Err(RuntimeError::from(crate::Error::backend_source(
                    EVENT_OP,
                    WebGpuRetirementFallbackError { primary, fallback },
                ))),
            };
        }
    };
    // `wait` is itself the retirement barrier even when it reports a queued
    // execution failure.
    submission.wait().map_err(webgpu_backend_error)
}

fn webgpu_backend_error(source: impl std::error::Error + Send + Sync + 'static) -> RuntimeError {
    RuntimeError::from(crate::Error::backend_source(EVENT_OP, source))
}

#[derive(Debug, thiserror::Error)]
#[error("WebGPU event-domain operation panicked")]
struct WebGpuEventDomainPanic;

#[derive(Debug, thiserror::Error)]
#[error("the CubeCL WebGPU server is unavailable")]
struct WebGpuServerUnavailable;

#[derive(Debug, thiserror::Error)]
#[error(
    "WebGPU completion submission failed ({primary}); client retirement also reported {fallback}"
)]
struct WebGpuRetirementFallbackError {
    primary: RuntimeError,
    #[source]
    fallback: crate::Error,
}

#[derive(Debug, thiserror::Error)]
#[error("submission failed ({primary}); WebGPU queue cleanup also failed ({cleanup})")]
struct WebGpuSubmissionCleanupError {
    primary: RuntimeError,
    #[source]
    cleanup: RuntimeError,
}
