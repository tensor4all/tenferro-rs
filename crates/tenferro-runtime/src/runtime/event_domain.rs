use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::EventDomainId;

/// Classifies the runtime boundary at which event-domain provenance is
/// validated.
///
/// The value is semantic control data for EventDomainError; provider
/// function names are deliberately not part of this contract.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::EventDomainOperation;
///
/// assert_eq!(EventDomainOperation::Enqueue, EventDomainOperation::Enqueue);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventDomainOperation {
    /// A driver is beginning a run for a frozen domain.
    BeginRun,
    /// A run is admitting dependencies and a launch.
    Enqueue,
    /// A run is being retired after all submitted work.
    Drain,
    /// The scheduler is host-bridging a source completion for a transfer.
    TransferBridge,
    /// The scheduler is validating a returned completion token.
    ValidateCompletion,
}

impl fmt::Display for EventDomainOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::BeginRun => "begin run",
            Self::Enqueue => "enqueue",
            Self::Drain => "drain",
            Self::TransferBridge => "transfer bridge",
            Self::ValidateCompletion => "validate completion",
        };
        formatter.write_str(name)
    }
}

/// Structured event-domain admission and provenance failure.
///
/// Every mismatch retains the expected and actual domains together with the
/// operation and optional scheduled-node context. Providers use the same
/// vocabulary for direct admission failures.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::EventDomainError;
///
/// fn inspect(error: &EventDomainError) {
///     let _ = error;
/// }
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EventDomainError {
    /// A driver returned a run for a domain other than the requested domain.
    #[error(
        "{operation} node {node_index:?} returned run domain {actual:?}, expected {expected:?}"
    )]
    RunDomainMismatch {
        /// Operation validating the run.
        operation: EventDomainOperation,
        /// Scheduled node, when the validation belongs to a node.
        node_index: Option<usize>,
        /// Domain requested by the runtime.
        expected: EventDomainId,
        /// Domain reported by the driver run.
        actual: EventDomainId,
    },
    /// A dependency token was not from the domain admitted at this boundary.
    #[error(
        "{operation} node {node_index:?} received dependency origin {actual:?}, expected {expected:?}"
    )]
    DependencyDomainMismatch {
        /// Operation validating the dependency.
        operation: EventDomainOperation,
        /// Scheduled node, when the validation belongs to a node.
        node_index: Option<usize>,
        /// Domain that may admit the dependency.
        expected: EventDomainId,
        /// Origin reported by the dependency token.
        actual: EventDomainId,
    },
    /// A driver returned a completion token with the wrong origin.
    #[error(
        "{operation} node {node_index:?} returned completion origin {actual:?}, expected {expected:?}"
    )]
    CompletionTokenDomainMismatch {
        /// Operation validating the completion.
        operation: EventDomainOperation,
        /// Scheduled node, when the validation belongs to a node.
        node_index: Option<usize>,
        /// Completion domain assigned by the schedule.
        expected: EventDomainId,
        /// Origin reported by the returned token.
        actual: EventDomainId,
    },
    /// A public event-domain run panicked while being explicitly retired.
    #[error("{operation} for event domain {domain:?} panicked: {message}")]
    DrainPanicked {
        /// Operation that contained the provider panic.
        operation: EventDomainOperation,
        /// Domain whose run was being retired.
        domain: EventDomainId,
        /// Safe diagnostic text extracted from the panic payload.
        message: String,
    },
    /// A token has the right origin but cannot be admitted by this provider.
    #[error(
        "{operation} node {node_index:?} cannot admit token type {token_type} for event domain {actual:?} (expected {expected:?})"
    )]
    IncompatibleTokenType {
        /// Operation validating the token.
        operation: EventDomainOperation,
        /// Scheduled node, when the validation belongs to a node.
        node_index: Option<usize>,
        /// Domain the run admits.
        expected: EventDomainId,
        /// Origin reported by the token.
        actual: EventDomainId,
        /// Provider-neutral diagnostic name of the token type.
        token_type: &'static str,
    },
    /// The scheduler's host bridge could not wait for a source completion.
    #[error(
        "{operation} node {node_index:?} failed waiting for dependency origin {actual:?} before destination domain {expected:?}: {source}"
    )]
    DependencyWaitFailed {
        /// Operation validating the transfer bridge.
        operation: EventDomainOperation,
        /// Scheduled node containing the failed bridge.
        node_index: Option<usize>,
        /// Destination domain that was not launched.
        expected: EventDomainId,
        /// Origin domain whose token failed to wait.
        actual: EventDomainId,
        /// Original host-wait failure.
        #[source]
        source: crate::error::BoxError,
    },
}

/// Opaque completion token produced by one event-domain run.
///
/// `wait` is repeatable and safe to call concurrently. A schedule may fan one
/// completion out to multiple foreign event domains, so implementations must
/// not consume the token on the first wait.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::EventToken;
///
/// fn inspect(token: &dyn EventToken) -> tenferro_runtime::Result<()> {
///     let _origin = token.origin();
///     let _opaque = token.as_any();
///     token.wait()?;
///     token.wait()
/// }
/// ```
pub trait EventToken: fmt::Debug + Send + Sync + 'static {
    /// Return the token as [`Any`] for driver-specific inspection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EventDomainDriver, EventDomainId};
    ///
    /// fn inspect(
    ///     driver: &dyn EventDomainDriver,
    ///     domain: EventDomainId,
    /// ) -> tenferro_runtime::Result<()> {
    ///     let mut run = driver.begin_run(domain)?;
    ///     let mut launch = || Ok(());
    ///     let token = run.enqueue(&[], &mut launch)?;
    ///     let _opaque = token.as_any();
    ///     Ok(())
    /// }
    /// ```
    fn as_any(&self) -> &dyn Any;

    /// Return the exact frozen event domain that produced this token.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EventDomainId, EventToken};
    ///
    /// fn inspect(token: &dyn EventToken, expected: EventDomainId) {
    ///     assert_eq!(token.origin(), expected);
    /// }
    /// ```
    fn origin(&self) -> EventDomainId;

    /// Wait on the host until this completion becomes ready.
    ///
    /// The runtime scheduler uses this repeatable host bridge only after it
    /// classifies a scheduled cross-domain transfer dependency. Destination
    /// drivers must not use it as a generic foreign-token fallback.
    ///
    /// # Errors
    ///
    /// Returns a typed backend or runtime error when the native completion
    /// reports failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EventToken;
    ///
    /// fn wait_repeatably(token: &dyn EventToken) -> tenferro_runtime::Result<()> {
    ///     token.wait()?;
    ///     token.wait()
    /// }
    /// ```
    fn wait(&self) -> crate::Result<()>;
}

/// Per-execution state owned by one event domain.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EventDomainId, EventDomainRun};
///
/// fn submit(
///     run: &mut dyn EventDomainRun,
///     domain: EventDomainId,
/// ) -> tenferro_runtime::Result<()> {
///     assert_eq!(run.domain(), domain);
///     let mut launch = || Ok(());
///     run.enqueue(&[], &mut launch)?;
///     run.drain()
/// }
/// ```
pub trait EventDomainRun: fmt::Debug + Send {
    /// Return the exact domain admitted by this run.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EventDomainId, EventDomainRun};
    ///
    /// fn inspect(run: &dyn EventDomainRun, expected: EventDomainId) {
    ///     assert_eq!(run.domain(), expected);
    /// }
    /// ```
    fn domain(&self) -> EventDomainId;

    /// Enqueue one launch after its dependency tokens.
    ///
    /// # Errors
    ///
    /// Returns a typed runtime error when a dependency token is incompatible
    /// with this domain, native queue submission fails, or `launch` returns an
    /// error.
    ///
    /// The implementation must not invoke `launch` when dependency admission
    /// or waiting fails. After dependencies are admitted, it invokes `launch`
    /// exactly once without holding a driver lock. The closure submits work to
    /// the native queue; it does not wait for that work to complete.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EventDomainRun;
    ///
    /// fn submit(run: &mut dyn EventDomainRun) -> tenferro_runtime::Result<()> {
    ///     let mut launched = false;
    ///     let mut launch = || {
    ///         launched = true;
    ///         Ok(())
    ///     };
    ///     run.enqueue(&[], &mut launch)?;
    ///     assert!(launched);
    ///     Ok(())
    /// }
    /// ```
    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>>;

    /// Wait for all work already enqueued in this run.
    ///
    /// # Errors
    ///
    /// Returns the first driver-defined typed error, such as
    /// [`crate::Error::Internal`], when a queued completion reports failure or
    /// the native queue cannot be synchronized.
    ///
    /// Success and error are both retirement boundaries: before returning, the
    /// run must ensure that no previously enqueued work can access tensors,
    /// tokens, or native resources retained by the caller. An error reports the
    /// completion failure; it must not mean that work is still using those
    /// resources. Implementations must not unwind from this method.
    ///
    /// Draining observes already-progressing work. It must not require another
    /// event domain's drain call to start that work.
    ///
    /// Dropping a run must perform equivalent best-effort retirement when
    /// explicit draining is skipped by panic unwinding. Neither explicit
    /// draining nor implicit retirement may unwind into the caller.
    fn drain(&mut self) -> crate::Result<()>;
}

/// Runtime-owned factory for per-execution event-domain state.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EventDomainDriver, EventDomainId};
///
/// fn begin(
///     driver: &dyn EventDomainDriver,
///     domain: EventDomainId,
/// ) -> tenferro_runtime::Result<()> {
///     let mut run = driver.begin_run(domain)?;
///     run.drain()
/// }
/// ```
pub trait EventDomainDriver: fmt::Debug + Send + Sync + 'static {
    /// Begin one isolated execution run for exactly `domain`.
    ///
    /// # Errors
    ///
    /// Returns a driver-defined typed error, such as [`crate::Error::Internal`],
    /// when native queue, stream, or per-run event-state allocation fails, or
    /// when the domain cannot admit a new run.
    fn begin_run(&self, domain: EventDomainId) -> crate::Result<Box<dyn EventDomainRun>>;
}

/// Completion token returned by the blocking immediate domain.
#[derive(Debug)]
struct ReadyEventToken {
    origin: EventDomainId,
}

impl EventToken for ReadyEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.origin
    }

    fn wait(&self) -> crate::Result<()> {
        Ok(())
    }
}

/// Blocking event-domain driver used by synchronous CPU engines.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::ImmediateEventDomainDriver;
///
/// let _driver = ImmediateEventDomainDriver::new();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ImmediateEventDomainDriver;

impl ImmediateEventDomainDriver {
    /// Construct a blocking immediate driver.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::runtime::ImmediateEventDomainDriver;
    ///
    /// let _driver = ImmediateEventDomainDriver::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl EventDomainDriver for ImmediateEventDomainDriver {
    fn begin_run(&self, domain: EventDomainId) -> crate::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(ImmediateEventDomainRun { domain }))
    }
}

#[derive(Debug)]
struct ImmediateEventDomainRun {
    domain: EventDomainId,
}

impl EventDomainRun for ImmediateEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        for dependency in dependencies {
            let actual = dependency.origin();
            if actual != self.domain {
                return Err(crate::Error::from(
                    EventDomainError::DependencyDomainMismatch {
                        operation: EventDomainOperation::Enqueue,
                        node_index: None,
                        expected: self.domain,
                        actual,
                    },
                ));
            }
            dependency.wait()?;
        }
        launch()?;
        Ok(Arc::new(ReadyEventToken {
            origin: self.domain,
        }))
    }

    fn drain(&mut self) -> crate::Result<()> {
        Ok(())
    }
}
