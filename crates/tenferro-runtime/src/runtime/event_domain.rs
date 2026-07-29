use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// Opaque completion token produced by one event-domain run.
pub trait EventToken: fmt::Debug + Send + Sync + 'static {
    /// Return the token as [`Any`] for driver-specific inspection.
    fn as_any(&self) -> &dyn Any;
}

/// Per-execution state owned by one event domain.
pub trait EventDomainRun: fmt::Debug + Send {
    /// Enqueue one launch after its dependency tokens.
    ///
    /// # Errors
    ///
    /// Returns a typed runtime error when a dependency is incompatible with
    /// this domain or when the launch fails.
    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>>;

    /// Wait for all work already enqueued in this run.
    ///
    /// # Errors
    ///
    /// Returns the first domain completion failure.
    fn drain(&mut self) -> crate::Result<()>;
}

/// Runtime-owned factory for per-execution event-domain state.
pub trait EventDomainDriver: fmt::Debug + Send + Sync + 'static {
    /// Begin one isolated execution run.
    ///
    /// # Errors
    ///
    /// Returns a typed runtime error when the domain cannot admit a new run.
    fn begin_run(&self) -> crate::Result<Box<dyn EventDomainRun>>;
}

/// Completion token returned by the blocking immediate domain.
#[derive(Debug)]
pub struct ReadyEventToken;

impl EventToken for ReadyEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Blocking event-domain driver used by synchronous CPU engines.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImmediateEventDomainDriver;

impl ImmediateEventDomainDriver {
    /// Construct a blocking immediate driver.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl EventDomainDriver for ImmediateEventDomainDriver {
    fn begin_run(&self) -> crate::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(ImmediateEventDomainRun))
    }
}

#[derive(Debug)]
struct ImmediateEventDomainRun;

impl EventDomainRun for ImmediateEventDomainRun {
    fn enqueue(
        &mut self,
        _dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        launch()?;
        Ok(Arc::new(ReadyEventToken))
    }

    fn drain(&mut self) -> crate::Result<()> {
        Ok(())
    }
}
