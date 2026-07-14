use std::env;
use std::sync::{Arc, Mutex};

use thiserror::Error as ThisError;

use crate::affinity::{SystemThreadAffinity, ThreadAffinity};
use crate::arbiter::{set_pool_execution_owner, with_execution_owner, ResourceOwner};
use crate::{CpuId, CpuSet, Error, Result};

#[derive(Debug, Default)]
struct ExecutionOwnerState {
    owner: Option<ResourceOwner>,
    depth: usize,
}

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
#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum CpuContextError {
    /// A context must contain at least one worker.
    #[error("thread count must be at least 1")]
    InvalidThreadCount,
    /// A pinned engine cannot create more workers than assigned CPUs.
    #[error("requested {workers} workers for only {cpus} assigned CPUs")]
    TooManyWorkers {
        /// Requested Rayon worker count.
        workers: usize,
        /// Number of logical CPUs in the execution domain.
        cpus: usize,
    },
    /// Rayon could not construct the custom thread pool.
    #[error("failed to build pinned CPU thread pool: {message}")]
    PoolBuild {
        /// Rayon or OS thread-spawn error detail.
        message: String,
    },
    /// A worker could not set or verify its assigned CPU affinity.
    #[error("failed to pin worker {worker} to CPU {cpu}: {message}")]
    WorkerPinning {
        /// Stable Rayon worker index.
        worker: usize,
        /// Assigned operating-system logical CPU.
        cpu: CpuId,
        /// OS or verification failure detail.
        message: String,
    },
    /// A worker terminated before reporting startup affinity.
    #[error("worker startup channel closed before all workers reported")]
    WorkerStartupClosed,
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
    execution_owner: Arc<Mutex<ExecutionOwnerState>>,
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
    pub fn try_from_env() -> Result<Self> {
        match env::var("RAYON_NUM_THREADS") {
            Ok(value) => {
                let num_threads = value.parse::<usize>().map_err(|err| Error::InvalidConfig {
                    op: "CpuContext::try_from_env",
                    message: format!("invalid RAYON_NUM_THREADS value {value:?}: {err}"),
                })?;
                Self::with_threads(num_threads).map_err(|err| match err {
                    Error::InvalidConfig { message, .. } => Error::InvalidConfig {
                        op: "CpuContext::try_from_env",
                        message: format!("invalid RAYON_NUM_THREADS value {value:?}: {message}"),
                    },
                    err => err,
                })
            }
            Err(env::VarError::NotPresent) => {
                Self::with_threads(super::affinity::available_parallelism())
            }
            Err(err) => Err(Error::InvalidConfig {
                op: "CpuContext::try_from_env",
                message: format!("failed to read RAYON_NUM_THREADS: {err}"),
            }),
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
    /// Returns an error when `num_threads` is zero or Rayon rejects the pool.
    pub fn with_threads(num_threads: usize) -> Result<Self> {
        if num_threads == 0 {
            return Err(Error::InvalidConfig {
                op: "CpuContext::with_threads",
                message: "thread count must be at least 1".into(),
            });
        }
        let pool = if num_threads == 1 {
            None
        } else {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .map_err(|err| Error::InvalidConfig {
                        op: "CpuContext::with_threads",
                        message: format!("failed to build CPU thread pool: {err}"),
                    })?,
            ))
        };
        Ok(Self {
            num_threads,
            pool,
            pinned_cpus: None,
            execution_owner: Arc::new(Mutex::new(ExecutionOwnerState::default())),
        })
    }

    /// Create a Rayon context whose workers are pinned to assigned logical CPUs.
    ///
    /// A real Rayon pool is constructed even when `num_threads` is one. The
    /// worker count cannot exceed the assigned CPU count.
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
        if num_threads > cpus.len() {
            return Err(CpuContextError::TooManyWorkers {
                workers: num_threads,
                cpus: cpus.len(),
            });
        }

        let assigned_cpus = Arc::new(select_worker_cpus(&cpus, num_threads));
        let (startup_tx, startup_rx) = std::sync::mpsc::channel();
        let pool_assigned_cpus = Arc::clone(&assigned_cpus);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .spawn_handler(move |thread| {
                let worker = thread.index();
                let cpu = pool_assigned_cpus[worker];
                let startup_tx = startup_tx.clone();
                let affinity = affinity.clone();
                std::thread::Builder::new()
                    .name(format!("tenferro-cpu-{cpu}"))
                    .spawn(move || {
                        let result = affinity.pin_current(cpu).and_then(|observed| {
                            (observed.len() == 1 && observed.contains(cpu))
                                .then_some(())
                                .ok_or_else(|| {
                                    format!(
                                        "verification returned affinity {:?}",
                                        observed.as_usize_vec()
                                    )
                                })
                        });
                        let _ = startup_tx.send((worker, cpu, result));
                        thread.run();
                    })
                    .map(|_| ())
            })
            .build()
            .map_err(|err| CpuContextError::PoolBuild {
                message: err.to_string(),
            })?;
        let pool = Arc::new(pool);
        for _ in 0..num_threads {
            let (worker, cpu, result) = startup_rx
                .recv()
                .map_err(|_| CpuContextError::WorkerStartupClosed)?;
            if let Err(message) = result {
                return Err(CpuContextError::WorkerPinning {
                    worker,
                    cpu,
                    message,
                });
            }
        }
        Ok(Self {
            num_threads,
            pool: Some(pool),
            pinned_cpus: Some(cpus),
            execution_owner: Arc::new(Mutex::new(ExecutionOwnerState::default())),
        })
    }

    fn single_threaded() -> Self {
        Self {
            num_threads: 1,
            pool: None,
            pinned_cpus: None,
            execution_owner: Arc::new(Mutex::new(ExecutionOwnerState::default())),
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

    pub(crate) fn install_with_execution_owner<R: Send>(
        &self,
        owner: ResourceOwner,
        op: impl FnOnce() -> R + Send,
    ) -> R {
        let should_broadcast = {
            let mut state = self
                .execution_owner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match state.owner {
                None => {
                    state.owner = Some(owner);
                    state.depth = 1;
                    true
                }
                Some(active) if active == owner => {
                    state.depth += 1;
                    false
                }
                Some(active) => panic!(
                    "CPU execution owner invariant violated: active {active:?}, requested {owner:?}"
                ),
            }
        };
        if should_broadcast {
            self.broadcast_execution_owner(Some(owner));
        }

        struct OwnerGuard<'a> {
            context: &'a CpuContext,
            owner: ResourceOwner,
        }

        impl Drop for OwnerGuard<'_> {
            fn drop(&mut self) {
                let should_clear = {
                    let mut state = self
                        .context
                        .execution_owner
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    debug_assert_eq!(state.owner, Some(self.owner));
                    state.depth -= 1;
                    if state.depth == 0 {
                        state.owner = None;
                        true
                    } else {
                        false
                    }
                };
                if should_clear {
                    self.context.broadcast_execution_owner(None);
                }
            }
        }

        let _guard = OwnerGuard {
            context: self,
            owner,
        };
        self.install(|| with_execution_owner(owner, op))
    }

    fn broadcast_execution_owner(&self, owner: Option<ResourceOwner>) {
        if let Some(pool) = &self.pool {
            pool.broadcast(|_| set_pool_execution_owner(owner));
        }
    }

    /// Return the faer parallelism policy for work run inside this context.
    ///
    /// `Par::rayon(0)` is intentional for multi-threaded contexts: faer reads
    /// `rayon::current_num_threads()`, so calls made under [`Self::install`]
    /// inherit this context's Rayon pool size.
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn faer_par(&self) -> faer::Par {
        if self.num_threads == 1 {
            faer::Par::Seq
        } else {
            faer::Par::rayon(0)
        }
    }

    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn faer_seq(&self) -> faer::Par {
        faer::Par::Seq
    }
}

fn select_worker_cpus(cpus: &CpuSet, num_threads: usize) -> Vec<CpuId> {
    if num_threads == 1 {
        return vec![cpus.as_slice()[cpus.len() / 2]];
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
