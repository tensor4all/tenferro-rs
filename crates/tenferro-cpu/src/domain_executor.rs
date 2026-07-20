use std::fmt::Debug;
use std::num::NonZeroUsize;

/// Inner parallel-region support offered by a CPU domain executor.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuInnerParallelism;
///
/// assert_ne!(CpuInnerParallelism::None, CpuInnerParallelism::Rayon);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuInnerParallelism {
    /// The executor cannot host provider-owned inner parallel regions.
    None,
    /// The executor can host a Rayon-compatible inner parallel region.
    Rayon,
}

/// Re-entry capability of one CPU domain executor.
///
/// This describes executor-level same-executor entry only. It never grants
/// permission for recursive public [`crate::CpuBackend`] entry, which remains a
/// separate backend contract.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuExecutorReentrancy;
///
/// let policy = CpuExecutorReentrancy::Rejected;
/// assert_eq!(policy, CpuExecutorReentrancy::Rejected);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuExecutorReentrancy {
    /// Nested entry into the same executor is rejected.
    Rejected,
    /// The executor supports nested entry into that same executor.
    SameExecutor,
}

/// Affinity claim made by a CPU domain executor.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuExecutorAffinity;
///
/// let affinity = CpuExecutorAffinity::CallerDeclaredUnverified;
/// assert_ne!(affinity, CpuExecutorAffinity::TenferroPinnedVerified);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuExecutorAffinity {
    /// Tenferro pinned the workers and verified their placement.
    TenferroPinnedVerified,
    /// The caller declared worker placement, but tenferro did not verify it.
    CallerDeclaredUnverified,
    /// The executor makes no worker-placement claim.
    None,
}

/// Ownership of CPU executor shutdown.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuExecutorShutdown;
///
/// assert_ne!(
///     CpuExecutorShutdown::TenferroOwned,
///     CpuExecutorShutdown::CallerOwned,
/// );
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuExecutorShutdown {
    /// Tenferro owns executor shutdown.
    TenferroOwned,
    /// The caller owns executor shutdown and executor lifetime policy.
    CallerOwned,
}

/// Immutable construction-time capabilities of a CPU domain executor.
///
/// # Examples
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use tenferro_cpu::{
///     CpuDomainExecutorCapabilities, CpuExecutorAffinity, CpuExecutorReentrancy,
///     CpuExecutorShutdown, CpuInnerParallelism,
/// };
///
/// let capabilities = CpuDomainExecutorCapabilities {
///     worker_count: NonZeroUsize::new(4).unwrap(),
///     outer_parallelism: true,
///     inner_parallelism: CpuInnerParallelism::Rayon,
///     reentrancy: CpuExecutorReentrancy::Rejected,
///     affinity: CpuExecutorAffinity::TenferroPinnedVerified,
///     shutdown: CpuExecutorShutdown::TenferroOwned,
/// };
/// assert_eq!(capabilities.worker_count.get(), 4);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuDomainExecutorCapabilities {
    /// Number of workers made available to this domain.
    pub worker_count: NonZeroUsize,
    /// Whether indexed outer fork/join submission is supported.
    pub outer_parallelism: bool,
    /// Provider-owned inner parallel-region support.
    pub inner_parallelism: CpuInnerParallelism,
    /// Same-executor re-entry capability.
    pub reentrancy: CpuExecutorReentrancy,
    /// Worker-affinity claim and verification level.
    pub affinity: CpuExecutorAffinity,
    /// Executor shutdown owner.
    pub shutdown: CpuExecutorShutdown,
}

/// Failure at the CPU executor admission or scheduling boundary.
///
/// Operation and provider errors do not belong in this type. Executors use
/// these variants only for their own admission, scheduling, cancellation, and
/// panic-bridge failures.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuDomainExecutorError;
///
/// let error = CpuDomainExecutorError::Admission {
///     message: "domain is busy".to_string(),
/// };
/// assert!(matches!(error, CpuDomainExecutorError::Admission { .. }));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CpuDomainExecutorError {
    /// The executor rejected entry before scheduling work.
    #[error("CPU domain executor admission failed: {message}")]
    Admission {
        /// Executor-owned diagnostic.
        message: String,
    },
    /// The executor could not schedule or complete submitted work.
    #[error("CPU domain executor scheduling failed: {message}")]
    Scheduling {
        /// Executor-owned diagnostic.
        message: String,
    },
    /// The executor cancelled submitted work.
    #[error("CPU domain executor cancelled work: {message}")]
    Cancellation {
        /// Executor-owned diagnostic.
        message: String,
    },
    /// The executor converted a worker panic into a typed failure.
    #[error("CPU domain executor worker panicked: {message}")]
    PanicBridge {
        /// Executor-owned diagnostic.
        message: String,
    },
}

/// One borrowed job installed synchronously into a CPU domain executor.
///
/// The executor must finish using the job before [`CpuDomainExecutor::install`]
/// returns; a borrowed job never escapes that call.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJob};
///
/// struct Job(bool);
/// impl ScopedCpuJob for Job {
///     fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
///         self.0 = true;
///         Ok(())
///     }
/// }
/// let mut job = Job(false);
/// job.run().unwrap();
/// assert!(job.0);
/// ```
pub trait ScopedCpuJob: Send {
    /// Run this job once on the executor-selected calling context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJob};
    ///
    /// struct Job;
    /// impl ScopedCpuJob for Job {
    ///     fn run(&mut self) -> Result<(), CpuDomainExecutorError> { Ok(()) }
    /// }
    /// assert!(Job.run().is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the executor admission, scheduling, cancellation, or panic-bridge
    /// failure observed while running the job.
    fn run(&mut self) -> Result<(), CpuDomainExecutorError>;
}

/// Synchronously submitted indexed jobs for engine-owned outer scheduling.
///
/// Implementations expose a borrowed logical range `0..len()` without
/// allocating a job collection. Every indexed call must finish before
/// [`CpuDomainExecutor::submit`] returns.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJobs};
///
/// struct Jobs;
/// impl ScopedCpuJobs for Jobs {
///     fn len(&self) -> usize { 2 }
///     fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError> {
///         assert!(index < self.len());
///         Ok(())
///     }
/// }
/// let jobs: &dyn ScopedCpuJobs = &Jobs;
/// assert_eq!(jobs.len(), 2);
/// jobs.run(1).unwrap();
/// ```
pub trait ScopedCpuJobs: Sync {
    /// Return the number of indexed jobs in this synchronous submission.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJobs};
    ///
    /// struct Jobs;
    /// impl ScopedCpuJobs for Jobs {
    ///     fn len(&self) -> usize { 3 }
    ///     fn run(&self, _index: usize) -> Result<(), CpuDomainExecutorError> { Ok(()) }
    /// }
    /// assert_eq!(Jobs.len(), 3);
    /// ```
    fn len(&self) -> usize;

    /// Return whether this submission contains no indexed jobs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJobs};
    ///
    /// struct Jobs;
    /// impl ScopedCpuJobs for Jobs {
    ///     fn len(&self) -> usize { 0 }
    ///     fn run(&self, _index: usize) -> Result<(), CpuDomainExecutorError> { Ok(()) }
    /// }
    /// assert!(Jobs.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run one indexed job synchronously.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainExecutorError, ScopedCpuJobs};
    ///
    /// struct Jobs;
    /// impl ScopedCpuJobs for Jobs {
    ///     fn len(&self) -> usize { 1 }
    ///     fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError> {
    ///         assert_eq!(index, 0);
    ///         Ok(())
    ///     }
    /// }
    /// Jobs.run(0).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the executor admission, scheduling, cancellation, or panic-bridge
    /// failure observed while running the indexed job.
    fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError>;
}

/// Object-safe synchronous executor for one CPU resource domain.
///
/// `submit` is an indexed fork/join boundary and `install` is one borrowed
/// provider-owned inner-region entry. Neither method may retain its borrowed
/// job after returning.
///
/// # Examples
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use tenferro_cpu::{
///     CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError,
///     CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
///     CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
/// };
///
/// #[derive(Debug)]
/// struct Inline;
/// impl CpuDomainExecutor for Inline {
///     fn capabilities(&self) -> CpuDomainExecutorCapabilities {
///         CpuDomainExecutorCapabilities {
///             worker_count: NonZeroUsize::new(1).unwrap(),
///             outer_parallelism: false,
///             inner_parallelism: CpuInnerParallelism::None,
///             reentrancy: CpuExecutorReentrancy::Rejected,
///             affinity: CpuExecutorAffinity::None,
///             shutdown: CpuExecutorShutdown::CallerOwned,
///         }
///     }
///     fn submit(&self, _jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
///         Ok(())
///     }
///     fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
///         job.run()
///     }
/// }
/// let executor: &dyn CpuDomainExecutor = &Inline;
/// assert_eq!(executor.capabilities().worker_count.get(), 1);
/// ```
pub trait CpuDomainExecutor: Debug + Send + Sync + 'static {
    /// Return immutable construction-time executor capabilities.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_cpu::{
    ///     CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError,
    ///     CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    ///     CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
    /// };
    /// # #[derive(Debug)] struct Inline;
    /// # impl CpuDomainExecutor for Inline {
    /// # fn capabilities(&self) -> CpuDomainExecutorCapabilities {
    /// # CpuDomainExecutorCapabilities { worker_count: NonZeroUsize::new(2).unwrap(),
    /// # outer_parallelism: true, inner_parallelism: CpuInnerParallelism::None,
    /// # reentrancy: CpuExecutorReentrancy::Rejected, affinity: CpuExecutorAffinity::None,
    /// # shutdown: CpuExecutorShutdown::CallerOwned }
    /// # }
    /// # fn submit(&self, _jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> { Ok(()) }
    /// # fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> { job.run() }
    /// # }
    /// let executor: &dyn CpuDomainExecutor = &Inline;
    /// assert_eq!(executor.capabilities().worker_count.get(), 2);
    /// ```
    fn capabilities(&self) -> CpuDomainExecutorCapabilities;

    /// Submit all indexed jobs as one synchronous fork/join operation.
    ///
    /// All `0..jobs.len()` jobs must be complete when this method returns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use tenferro_cpu::{
    ///     CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError,
    ///     CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    ///     CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
    /// };
    /// # #[derive(Debug)] struct Inline;
    /// # impl CpuDomainExecutor for Inline {
    /// # fn capabilities(&self) -> CpuDomainExecutorCapabilities {
    /// # CpuDomainExecutorCapabilities { worker_count: NonZeroUsize::new(1).unwrap(),
    /// # outer_parallelism: true, inner_parallelism: CpuInnerParallelism::None,
    /// # reentrancy: CpuExecutorReentrancy::Rejected, affinity: CpuExecutorAffinity::None,
    /// # shutdown: CpuExecutorShutdown::CallerOwned }
    /// # }
    /// # fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
    /// # for index in 0..jobs.len() { jobs.run(index)?; } Ok(())
    /// # }
    /// # fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> { job.run() }
    /// # }
    /// struct Jobs<'a>(&'a AtomicUsize);
    /// impl ScopedCpuJobs for Jobs<'_> {
    ///     fn len(&self) -> usize { 2 }
    ///     fn run(&self, _index: usize) -> Result<(), CpuDomainExecutorError> {
    ///         self.0.fetch_add(1, Ordering::Relaxed);
    ///         Ok(())
    ///     }
    /// }
    /// let count = AtomicUsize::new(0);
    /// Inline.submit(&Jobs(&count)).unwrap();
    /// assert_eq!(count.load(Ordering::Relaxed), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuDomainExecutorError::Admission`],
    /// [`CpuDomainExecutorError::Scheduling`],
    /// [`CpuDomainExecutorError::Cancellation`], or
    /// [`CpuDomainExecutorError::PanicBridge`] for executor-owned failures.
    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError>;

    /// Enter one synchronous provider-owned inner parallel region.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_cpu::{
    ///     CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError,
    ///     CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    ///     CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
    /// };
    /// # #[derive(Debug)] struct Inline;
    /// # impl CpuDomainExecutor for Inline {
    /// # fn capabilities(&self) -> CpuDomainExecutorCapabilities {
    /// # CpuDomainExecutorCapabilities { worker_count: NonZeroUsize::new(1).unwrap(),
    /// # outer_parallelism: false, inner_parallelism: CpuInnerParallelism::None,
    /// # reentrancy: CpuExecutorReentrancy::Rejected, affinity: CpuExecutorAffinity::None,
    /// # shutdown: CpuExecutorShutdown::CallerOwned }
    /// # }
    /// # fn submit(&self, _jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> { Ok(()) }
    /// # fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> { job.run() }
    /// # }
    /// struct Job(bool);
    /// impl ScopedCpuJob for Job {
    ///     fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
    ///         self.0 = true;
    ///         Ok(())
    ///     }
    /// }
    /// let mut job = Job(false);
    /// Inline.install(&mut job).unwrap();
    /// assert!(job.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuDomainExecutorError::Admission`],
    /// [`CpuDomainExecutorError::Scheduling`],
    /// [`CpuDomainExecutorError::Cancellation`], or
    /// [`CpuDomainExecutorError::PanicBridge`] for executor-owned failures.
    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError>;
}

pub(crate) struct ScopedJob<F, R> {
    operation: Option<F>,
    result: Option<R>,
}

pub(crate) fn scoped_job<F, R>(operation: F) -> ScopedJob<F, R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    ScopedJob {
        operation: Some(operation),
        result: None,
    }
}

impl<F, R> ScopedJob<F, R> {
    fn into_result(self) -> Result<R, CpuDomainExecutorError> {
        self.result
            .ok_or_else(|| CpuDomainExecutorError::Scheduling {
                message: "executor returned success without running the scoped CPU job".to_string(),
            })
    }
}

impl<F, R> ScopedCpuJob for ScopedJob<F, R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
        let operation =
            self.operation
                .take()
                .ok_or_else(|| CpuDomainExecutorError::Scheduling {
                    message: "executor attempted to run a scoped CPU job more than once"
                        .to_string(),
                })?;
        self.result = Some(operation());
        Ok(())
    }
}

pub(crate) fn install_scoped<F, R>(
    executor: &dyn CpuDomainExecutor,
    operation: F,
) -> Result<R, CpuDomainExecutorError>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let mut job = scoped_job(operation);
    executor.install(&mut job)?;
    job.into_result()
}

pub(crate) struct IndexedJobs<F> {
    len: usize,
    run: F,
}

pub(crate) fn indexed_jobs<F>(len: usize, run: F) -> IndexedJobs<F>
where
    F: Fn(usize) -> Result<(), CpuDomainExecutorError> + Sync,
{
    IndexedJobs { len, run }
}

impl<F> ScopedCpuJobs for IndexedJobs<F>
where
    F: Fn(usize) -> Result<(), CpuDomainExecutorError> + Sync,
{
    fn len(&self) -> usize {
        self.len
    }

    fn run(&self, index: usize) -> Result<(), CpuDomainExecutorError> {
        if index >= self.len {
            return Err(CpuDomainExecutorError::Scheduling {
                message: format!(
                    "executor requested scoped CPU job index {index}, but the submission has {} jobs",
                    self.len
                ),
            });
        }
        (self.run)(index)
    }
}

#[cfg(test)]
mod tests;
