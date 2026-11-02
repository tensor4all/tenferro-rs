//! The runtime's event domains over the new path's stream.
//!
//! Every run is one stream position: [`CudaRuntime::checkpoint`] records a completion after the
//! launch and its token waits on that completion, so a dependency is a stream-ordered wait rather
//! than a device event. Retirement is the runtime's own synchronise.

use std::any::Any;
use std::sync::{Arc, Mutex};

use tenferro_runtime::runtime::{EventDomainDriver, EventDomainId, EventDomainRun, EventToken};
use tenferro_runtime::Error as RuntimeError;

use super::runtime::CudaRuntime;
use crate::event_domain_admission::admit_event_tokens;
use crate::event_retirement::{
    best_effort_retirement, retire_pending, take_pending_retirement, EventDomainRunState,
};

const EVENT_OP: &str = "cuda_event_domain";

/// Wrap a runtime failure as the runtime's own backend error, as the seam report does.
pub(crate) fn event_domain_error(
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    RuntimeError::from(crate::Error::backend_source(EVENT_OP, source))
}

/// One runtime's event-domain factory: every run orders on the runtime's single stream.
#[derive(Debug, Clone)]
pub(crate) struct StreamEventDomainDriver {
    runtime: CudaRuntime,
}

impl StreamEventDomainDriver {
    pub(crate) fn new(runtime: CudaRuntime) -> Self {
        Self { runtime }
    }
}

impl EventDomainDriver for StreamEventDomainDriver {
    fn begin_run(
        &self,
        domain: EventDomainId,
    ) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(StreamEventDomainRun {
            runtime: self.runtime.clone(),
            domain,
            state: EventDomainRunState::Pending,
            coalesced: false,
            shared_token: None,
        }))
    }
}

/// One execution run over this runtime's stream.
#[derive(Debug)]
struct StreamEventDomainRun {
    runtime: CudaRuntime,
    domain: EventDomainId,
    state: EventDomainRunState,
    /// Set when the scheduler proved that this run is the only domain and holds no transfer
    /// node. Every launch of the run goes to this runtime's one stream, so submission order
    /// alone orders its nodes and a per-node completion would carry no information.
    coalesced: bool,
    /// The token every coalesced completion reports: a coalesced run's token only carries the
    /// domain, so one shared token answers every node instead of one per node.
    shared_token: Option<Arc<dyn EventToken>>,
}

impl EventDomainRun for StreamEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    /// Take the coalesced path: one stream, no transfer node, submission order is stream order.
    ///
    /// This is what removes the per-node checkpoints a prepared run otherwise records. On
    /// `linalg_ad_large_grad_sum_qr_jvp_f64_256` a call records 47 completions and the scheduler
    /// polls them 558 times, which is host time nothing in the arithmetic needs (#2396).
    fn enable_single_domain_coalescing(&mut self) -> bool {
        self.coalesced = true;
        true
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        if self.coalesced && dependencies.is_empty() {
            // Submission order is the only ordering this run needs, so the node's completion is
            // the run's own retirement: launch and hand back the shared domain token.
            let _deferred = self.runtime.defer_completion();
            launch()?;
            let token = self.shared_token.get_or_insert_with(|| {
                Arc::new(CoalescedEventToken {
                    domain: self.domain,
                }) as Arc<dyn EventToken>
            });
            return Ok(Arc::clone(token));
        }
        admit_event_tokens::<StreamEventToken, _>(
            dependencies,
            self.domain,
            "non-CUDA event token",
            |tokens| {
                // The dependency tokens name stream positions of this same runtime; waiting on
                // them orders this launch behind that work.
                for token in tokens {
                    token.wait()?;
                }
                // The run owns the completion of what it enqueues: the checkpoint recorded below
                // is this node's stream position, and the run's retirement is the barrier. An
                // operation inside `launch` therefore does not wait for its own kernel (#2382).
                {
                    let _deferred = self.runtime.defer_completion();
                    launch()?;
                }
                let completion = {
                    let mut guard = self.runtime.submit();
                    let completion = guard.checkpoint().map_err(event_domain_error)?;
                    drop(guard);
                    self.runtime.note_completion_checkpoint();
                    completion
                };
                let token = StreamEventToken {
                    domain: self.domain,
                    completion: Mutex::new(completion),
                };
                Ok(Arc::new(token) as Arc<dyn EventToken>)
            },
        )
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        if !take_pending_retirement(&mut self.state) {
            return Ok(());
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.runtime.synchronize().map_err(event_domain_error)
        }));
        // Both outcomes are retirement boundaries. The runtime's own synchronise can report an
        // observation error before it reaches the stream barrier, so an error path waits for the
        // stream itself before the run reports failure.
        let waited = match &result {
            Ok(Ok(())) => true,
            _ => self.wait_stream(),
        };
        if waited {
            self.state = EventDomainRunState::Retired;
        }
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(RuntimeError::from(crate::Error::backend_source(
                EVENT_OP,
                std::io::Error::other("the event-domain retirement panicked"),
            ))),
        }
    }
}

impl StreamEventDomainRun {
    /// Wait for this runtime's stream directly, for the paths where the runtime's own
    /// synchronise reported an error before its barrier.
    fn wait_stream(&self) -> bool {
        let waited = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the scope hands out this runtime's own stream and nothing else; a barrier on
            // it neither reads nor writes an operand.
            unsafe {
                self.runtime
                    .with_raw_locked(Vec::new(), |scope| {
                        let stream =
                            scope.stream() as usize as *mut cudarc::driver::sys::CUstream_st;
                        cudarc::driver::sys::cuStreamSynchronize(stream)
                    })
                    .is_ok()
            }
        }));
        matches!(waited, Ok(true))
    }
}

impl Drop for StreamEventDomainRun {
    fn drop(&mut self) {
        retire_pending(&mut self.state, || {
            best_effort_retirement(|| {
                if let Err(error) = self.runtime.synchronize() {
                    eprintln!(
                        "tenferro-gpu: failed to drain the CUDA event-domain run during Drop: \
                         {error}"
                    );
                }
            });
        });
    }
}

/// The domain-only token a coalesced run reports.
///
/// A coalesced run keeps no per-node stream position: its nodes are ordered by submission order
/// and its completion is the run's retirement. The token therefore carries the domain, which is
/// all a coalesced run's dependencies read.
pub(crate) struct CoalescedEventToken {
    domain: EventDomainId,
}

impl std::fmt::Debug for CoalescedEventToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoalescedEventToken")
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

impl EventToken for CoalescedEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        // Nothing to wait on: the run's own retirement is the barrier for work this token names.
        Ok(())
    }
}

/// One stream position a later launch can wait on.
pub(crate) struct StreamEventToken {
    domain: EventDomainId,
    completion: Mutex<t4a_cuda_runtime::Completion>,
}

impl std::fmt::Debug for StreamEventToken {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamEventToken")
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

impl EventToken for StreamEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        let mut completion = self
            .completion
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        completion.wait().map_err(event_domain_error)
    }
}
