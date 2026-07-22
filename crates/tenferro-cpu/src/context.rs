use std::env;
use std::num::NonZeroUsize;
use std::sync::Arc;

use rayon::prelude::*;
use thiserror::Error as ThisError;

use crate::affinity::{CpuAffinityError, SystemThreadAffinity, ThreadAffinity};
use crate::arbiter::{
    current_execution_owner, register_worker_execution_scope, worker_execution_scope_matches,
    ExecutionScopeState,
};
use crate::domain_executor::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, ScopedCpuJob, ScopedCpuJobs,
};
use crate::{CpuId, CpuSet, Error, ErrorKind, Result, ValidationKind};

/// Failure to construct a CPU context with pinned Rayon workers.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuContextError;
///
/// let error = CpuContextError::InvalidThreadCount;
/// assert!(error.to_string().contains("thread count"));
/// ```
#[derive(Debug, ThisError)]
pub enum CpuContextError {
    /// A context must contain at least one worker.
    #[error("thread count must be at least 1")]
    InvalidThreadCount,
    /// A pinned engine rejected more workers than assigned CPUs.
    ///
    /// Retained for source compatibility. Current constructors oversubscribe by
    /// assigning workers cyclically when the worker count exceeds the CPU set.
    #[error("requested {workers} workers for only {cpus} assigned CPUs")]
    TooManyWorkers {
        /// Requested Rayon worker count.
        workers: usize,
        /// Number of logical CPUs in the execution domain.
        cpus: usize,
    },
    /// Rayon could not construct the custom thread pool.
    #[error("failed to build pinned CPU thread pool: {source}")]
    PoolBuild {
        /// Rayon or OS thread-spawn error.
        #[source]
        source: rayon::ThreadPoolBuildError,
    },
    /// A worker could not set or verify its assigned CPU affinity.
    #[error("failed to pin worker {worker} to CPU {cpu}: {source}")]
    WorkerPinning {
        /// Stable Rayon worker index.
        worker: usize,
        /// Assigned operating-system logical CPU.
        cpu: CpuId,
        /// OS or verification failure.
        #[source]
        source: CpuAffinityError,
    },
    /// A worker terminated before reporting startup affinity.
    #[error("worker startup channel closed before all workers reported: {source}")]
    WorkerStartupClosed {
        /// Channel receive failure from the worker startup handshake.
        #[source]
        source: std::sync::mpsc::RecvError,
    },
}

/// Reusable CPU execution context carrying CPU parallelism policy.
///
/// `CpuContext` stores the requested thread count as a kernel-level
/// parallelism hint and owns the Rayon pool used by multi-threaded CPU work.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuContext;
///
/// let ctx = CpuContext::with_threads(1).unwrap();
/// let value = ctx.install(|| 1 + 1);
/// assert_eq!(value, 2);
/// assert_eq!(ctx.num_threads(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct CpuContext {
    num_threads: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
    pinned_cpus: Option<CpuSet>,
    execution_scope: Arc<ExecutionScopeState>,
}

impl CpuContext {
    /// Create a CPU context from `RAYON_NUM_THREADS`, or fall back to a
    /// single-threaded context with a stderr warning when validation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::from_env();
    /// assert!(ctx.num_threads() >= 1);
    /// ```
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|err| {
            eprintln!(
                "tenferro_cpu: falling back to single-threaded CPU context after configuration error: {err}"
            );
            Self::single_threaded()
        })
    }

    /// Try to create a CPU context from `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_from_env()
    ///     .unwrap_or_else(|_| CpuContext::with_threads(1).unwrap());
    /// assert!(ctx.num_threads() >= 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuContextError`] when `RAYON_NUM_THREADS` is malformed or
    /// requests an invalid worker count.
    pub fn try_from_env() -> Result<Self> {
        match env::var("RAYON_NUM_THREADS") {
            Ok(value) => {
                let num_threads = value.parse::<usize>().map_err(|err| {
                    Error::extension(
                        "CpuContext::try_from_env",
                        "cpu",
                        ErrorKind::Validation(ValidationKind::InvalidArgument),
                        err,
                    )
                })?;
                Self::with_threads(num_threads).map_err(|err| match err {
                    Error::Validation { source, .. } => {
                        Error::validation("CpuContext::try_from_env", source)
                    }
                    err => err,
                })
            }
            Err(env::VarError::NotPresent) => {
                Self::with_threads(super::affinity::available_parallelism())
            }
            Err(err) => Err(Error::extension(
                "CpuContext::try_from_env",
                "cpu",
                ErrorKind::Validation(ValidationKind::InvalidArgument),
                err,
            )),
        }
    }

    /// Create a CPU context with a fixed parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2).unwrap();
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuContextError::InvalidThreadCount`] through
    /// [`Error::Validation`] when `num_threads` is zero, or
    /// [`Error::BackendSource`] when Rayon rejects the thread pool.
    pub fn with_threads(num_threads: usize) -> Result<Self> {
        if num_threads == 0 {
            return Err(Error::invalid_argument(
                "CpuContext::with_threads",
                "configuration",
                "thread count must be at least 1",
            ));
        }
        let execution_scope = Arc::new(ExecutionScopeState::default());
        let pool = if num_threads == 1 {
            None
        } else {
            let (startup_tx, startup_rx) = std::sync::mpsc::channel();
            let worker_scope = Arc::clone(&execution_scope);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .start_handler(move |_| {
                    register_worker_execution_scope(Arc::clone(&worker_scope));
                    let _ = startup_tx.send(());
                })
                .build()
                .map_err(|source| Error::backend_source("CpuContext::with_threads", source))?;
            for _ in 0..num_threads {
                startup_rx
                    .recv()
                    .map_err(|source| Error::backend_source("CpuContext::with_threads", source))?;
            }
            Some(Arc::new(pool))
        };
        Ok(Self {
            num_threads,
            pool,
            pinned_cpus: None,
            execution_scope,
        })
    }

    /// Create a Rayon context whose workers are pinned to assigned logical CPUs.
    ///
    /// A real Rayon pool is constructed even when `num_threads` is one. The
    /// When the worker count exceeds the assigned CPU count, workers are
    /// assigned cyclically to the exact CPU set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{process_cpu_affinity, CpuContext};
    ///
    /// if let Some(allowed) = process_cpu_affinity() {
    ///     let one_cpu = tenferro_cpu::CpuSet::new([allowed.as_slice()[0]])?;
    ///     let context = CpuContext::with_pinned_cpus(one_cpu.clone(), 1)?;
    ///     assert_eq!(context.pinned_cpus(), Some(&one_cpu));
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuContextError::InvalidThreadCount`] for zero workers or an
    /// affinity error when workers cannot be pinned.
    pub fn with_pinned_cpus(
        cpus: CpuSet,
        num_threads: usize,
    ) -> std::result::Result<Self, CpuContextError> {
        Self::with_pinned_cpus_using(cpus, num_threads, SystemThreadAffinity)
    }

    pub(crate) fn with_pinned_cpus_using<A: ThreadAffinity>(
        cpus: CpuSet,
        num_threads: usize,
        affinity: A,
    ) -> std::result::Result<Self, CpuContextError> {
        if num_threads == 0 {
            return Err(CpuContextError::InvalidThreadCount);
        }
        let execution_scope = Arc::new(ExecutionScopeState::default());
        let assigned_cpus = Arc::new(select_worker_cpus(&cpus, num_threads));
        let (startup_tx, startup_rx) = std::sync::mpsc::channel();
        let pool_assigned_cpus = Arc::clone(&assigned_cpus);
        let worker_scope = Arc::clone(&execution_scope);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .spawn_handler(move |thread| {
                let worker = thread.index();
                let cpu = pool_assigned_cpus[worker];
                let startup_tx = startup_tx.clone();
                let affinity = affinity.clone();
                let worker_scope = Arc::clone(&worker_scope);
                std::thread::Builder::new()
                    .name(format!("tenferro-cpu-{cpu}"))
                    .spawn(move || {
                        register_worker_execution_scope(Arc::clone(&worker_scope));
                        let result = affinity.pin_current(cpu).and_then(|observed| {
                            (observed.len() == 1 && observed.contains(cpu))
                                .then_some(())
                                .ok_or_else(|| CpuAffinityError::Verification {
                                    observed: observed.as_slice().to_vec(),
                                })
                        });
                        let _ = startup_tx.send((worker, cpu, result));
                        thread.run();
                    })
                    .map(|_| ())
            })
            .build()
            .map_err(|source| CpuContextError::PoolBuild { source })?;
        let pool = Arc::new(pool);
        for _ in 0..num_threads {
            let (worker, cpu, result) = startup_rx
                .recv()
                .map_err(|source| CpuContextError::WorkerStartupClosed { source })?;
            if let Err(source) = result {
                return Err(CpuContextError::WorkerPinning {
                    worker,
                    cpu,
                    source,
                });
            }
        }
        Ok(Self {
            num_threads,
            pool: Some(pool),
            pinned_cpus: Some(cpus),
            execution_scope,
        })
    }

    fn single_threaded() -> Self {
        Self {
            num_threads: 1,
            pool: None,
            pinned_cpus: None,
            execution_scope: Arc::new(ExecutionScopeState::default()),
        }
    }

    /// Return this context's CPU parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2).unwrap();
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Return the worker CPU domain for a pinned context.
    ///
    /// Legacy thread-count-only contexts return `None` because they do not own
    /// worker affinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// assert_eq!(CpuContext::with_threads(1)?.pinned_cpus(), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn pinned_cpus(&self) -> Option<&CpuSet> {
        self.pinned_cpus.as_ref()
    }

    /// Run a closure inside this context's CPU execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(1).unwrap();
    /// let value = ctx.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(op),
            None => op(),
        }
    }

    pub(crate) fn install_if_needed<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        if self.pool.is_some() && worker_execution_scope_matches(&self.execution_scope) {
            op()
        } else {
            self.install(op)
        }
    }

    #[cfg(test)]
    pub(crate) fn owns_current_worker_for_test(&self) -> bool {
        worker_execution_scope_matches(&self.execution_scope)
    }
}

impl CpuDomainExecutor for CpuContext {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        // INVARIANT: every CpuContext constructor rejects zero workers, and
        // `num_threads` is private so it cannot be invalidated after creation.
        let worker_count = match NonZeroUsize::new(self.num_threads) {
            Some(worker_count) => worker_count,
            None => unreachable!("CpuContext must contain at least one worker"),
        };
        CpuDomainExecutorCapabilities {
            worker_count,
            outer_parallelism: self.num_threads > 1,
            inner_parallelism: if self.pool.is_some() {
                CpuInnerParallelism::Rayon
            } else {
                CpuInnerParallelism::None
            },
            // This permits internal entry through the same executor. Public
            // CpuBackend re-entry remains guarded by BACKEND_REENTRY_PANIC.
            reentrancy: CpuExecutorReentrancy::SameExecutor,
            affinity: if self.pinned_cpus.is_some() {
                CpuExecutorAffinity::TenferroPinnedVerified
            } else {
                CpuExecutorAffinity::None
            },
            shutdown: CpuExecutorShutdown::TenferroOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> std::result::Result<(), CpuDomainExecutorError> {
        let _scope = current_execution_owner().map(|owner| self.execution_scope.enter(owner));
        if self.pool.is_none() {
            return (0..jobs.len()).try_for_each(|index| jobs.run(index));
        }
        self.install_if_needed(|| {
            (0..jobs.len())
                .into_par_iter()
                .try_for_each(|index| jobs.run(index))
        })
    }

    fn install(
        &self,
        job: &mut dyn ScopedCpuJob,
    ) -> std::result::Result<(), CpuDomainExecutorError> {
        let _scope = current_execution_owner().map(|owner| self.execution_scope.enter(owner));
        self.install_if_needed(|| job.run())
    }
}

fn select_worker_cpus(cpus: &CpuSet, num_threads: usize) -> Vec<CpuId> {
    if num_threads == 1 {
        return vec![cpus.as_slice()[cpus.len() / 2]];
    }
    if num_threads > cpus.len() {
        return (0..num_threads)
            .map(|worker| cpus.as_slice()[worker % cpus.len()])
            .collect();
    }
    (0..num_threads)
        .map(|worker| {
            let index = ((worker as u128) * ((cpus.len() - 1) as u128)
                / ((num_threads - 1) as u128)) as usize;
            cpus.as_slice()[index]
        })
        .collect()
}

#[cfg(test)]
mod tests;
