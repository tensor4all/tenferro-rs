use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// Opaque completion token produced by one event-domain run.
///
/// `wait` is repeatable and safe to call concurrently. A schedule may fan one
/// completion out to multiple foreign event domains, so implementations must
/// not consume the token on the first wait.
///
/// # Examples
///
/// ```
/// use std::any::Any;
/// use tenferro_runtime::runtime::EventToken;
///
/// #[derive(Debug)]
/// struct Ready;
///
/// impl EventToken for Ready {
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///
///     fn wait(&self) -> tenferro_runtime::Result<()> {
///         Ok(())
///     }
/// }
///
/// let token = Ready;
/// token.wait()?;
/// token.wait()?;
/// assert!(token.as_any().is::<Ready>());
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
pub trait EventToken: fmt::Debug + Send + Sync + 'static {
    /// Return the token as [`Any`] for driver-specific inspection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::runtime::{
    ///     EventDomainDriver, ImmediateEventDomainDriver,
    /// };
    ///
    /// let driver = ImmediateEventDomainDriver::new();
    /// let mut run = driver.begin_run()?;
    /// let mut launch = || Ok(());
    /// let token = run.enqueue(&[], &mut launch)?;
    /// let _opaque = token.as_any();
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    fn as_any(&self) -> &dyn Any;

    /// Wait on the host until this completion becomes ready.
    ///
    /// Native drivers should encode same-backend dependencies in their queue or
    /// stream. They use this fallback for foreign token types that cannot be
    /// represented as a native dependency.
    ///
    /// # Errors
    ///
    /// Returns a typed backend or runtime error when the native completion
    /// reports failure, or when a foreign completion token cannot be waited on.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::runtime::{
    ///     EventDomainDriver, ImmediateEventDomainDriver,
    /// };
    ///
    /// let driver = ImmediateEventDomainDriver::new();
    /// let mut run = driver.begin_run()?;
    /// let mut launch = || Ok(());
    /// run.enqueue(&[], &mut launch)?.wait()?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    fn wait(&self) -> crate::Result<()>;
}

/// Per-execution state owned by one event domain.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::{
///     EventDomainDriver, EventDomainRun, ImmediateEventDomainDriver,
/// };
///
/// fn submit(run: &mut dyn EventDomainRun) -> tenferro_runtime::Result<()> {
///     let mut launch = || Ok(());
///     run.enqueue(&[], &mut launch)?;
///     run.drain()
/// }
///
/// let driver = ImmediateEventDomainDriver::new();
/// let mut run = driver.begin_run()?;
/// submit(run.as_mut())?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
pub trait EventDomainRun: fmt::Debug + Send {
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
    /// use tenferro_runtime::runtime::{
    ///     EventDomainDriver, ImmediateEventDomainDriver,
    /// };
    ///
    /// let driver = ImmediateEventDomainDriver::new();
    /// let mut run = driver.begin_run()?;
    /// let mut launched = false;
    /// let mut launch = || {
    ///     launched = true;
    ///     Ok(())
    /// };
    /// run.enqueue(&[], &mut launch)?;
    /// assert!(launched);
    /// # Ok::<(), tenferro_runtime::Error>(())
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
    /// Draining observes already-progressing work. It must not require another
    /// event domain's `drain` call to start that work.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::runtime::{
    ///     EventDomainDriver, ImmediateEventDomainDriver,
    /// };
    ///
    /// let driver = ImmediateEventDomainDriver::new();
    /// let mut run = driver.begin_run()?;
    /// run.drain()?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    fn drain(&mut self) -> crate::Result<()>;
}

/// Runtime-owned factory for per-execution event-domain state.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::{
///     EventDomainDriver, ImmediateEventDomainDriver,
/// };
///
/// let driver = ImmediateEventDomainDriver::new();
/// let mut run = driver.begin_run()?;
/// let mut launch = || Ok(());
/// let completion = run.enqueue(&[], &mut launch)?;
/// completion.wait()?;
/// run.drain()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
pub trait EventDomainDriver: fmt::Debug + Send + Sync + 'static {
    /// Begin one isolated execution run.
    ///
    /// # Errors
    ///
    /// Returns a driver-defined typed error, such as [`crate::Error::Internal`],
    /// when native queue, stream, or per-run event-state allocation fails, or
    /// when the domain cannot admit a new run.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::runtime::{
    ///     EventDomainDriver, ImmediateEventDomainDriver,
    /// };
    ///
    /// let driver = ImmediateEventDomainDriver::new();
    /// let mut run = driver.begin_run()?;
    /// run.drain()?;
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    fn begin_run(&self) -> crate::Result<Box<dyn EventDomainRun>>;
}

/// Completion token returned by the blocking immediate domain.
#[derive(Debug)]
struct ReadyEventToken;

impl EventToken for ReadyEventToken {
    fn as_any(&self) -> &dyn Any {
        self
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
/// use tenferro_runtime::runtime::{
///     EventDomainDriver, ImmediateEventDomainDriver,
/// };
///
/// let driver = ImmediateEventDomainDriver::new();
/// driver.begin_run()?.drain()?;
/// # Ok::<(), tenferro_runtime::Error>(())
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
    fn begin_run(&self) -> crate::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(ImmediateEventDomainRun))
    }
}

#[derive(Debug)]
struct ImmediateEventDomainRun;

impl EventDomainRun for ImmediateEventDomainRun {
    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        for dependency in dependencies {
            dependency.wait()?;
        }
        launch()?;
        Ok(Arc::new(ReadyEventToken))
    }

    fn drain(&mut self) -> crate::Result<()> {
        Ok(())
    }
}
