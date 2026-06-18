use std::env;
use std::sync::Arc;

use crate::{Error, Result};

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
/// let ctx = CpuContext::with_threads(1);
/// let value = ctx.install(|| 1 + 1);
/// assert_eq!(value, 2);
/// assert_eq!(ctx.num_threads(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct CpuContext {
    num_threads: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl CpuContext {
    /// Create a CPU context from `RAYON_NUM_THREADS`, or fall back to the
    /// process-visible CPU count.
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
        Self::try_from_env()
            .unwrap_or_else(|_| Self::with_threads(super::affinity::available_parallelism()))
    }

    /// Try to create a CPU context from `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_from_env()
    ///     .unwrap_or_else(|_| CpuContext::with_threads(1));
    /// assert!(ctx.num_threads() >= 1);
    /// ```
    pub fn try_from_env() -> Result<Self> {
        match env::var("RAYON_NUM_THREADS") {
            Ok(value) => {
                let num_threads = value.parse::<usize>().map_err(|err| Error::InvalidConfig {
                    op: "CpuContext::try_from_env",
                    message: format!("invalid RAYON_NUM_THREADS value {value:?}: {err}"),
                })?;
                Self::try_with_threads(num_threads).map_err(|err| match err {
                    Error::InvalidConfig { message, .. } => Error::InvalidConfig {
                        op: "CpuContext::try_from_env",
                        message: format!("invalid RAYON_NUM_THREADS value {value:?}: {message}"),
                    },
                    err => err,
                })
            }
            Err(env::VarError::NotPresent) => {
                Self::try_with_threads(super::affinity::available_parallelism())
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
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `num_threads` is invalid or Rayon rejects the pool. This is
    /// the compatibility wrapper; new runtime-facing code should use
    /// [`CpuContext::try_with_threads`] to receive a typed error.
    pub fn with_threads(num_threads: usize) -> Self {
        // Intentional compatibility panic wrapper. Keep `try_with_threads` as
        // the non-panicking API so audits do not mistake this for an unhandled
        // validation path.
        match Self::try_with_threads(num_threads) {
            Ok(ctx) => ctx,
            Err(err) => panic!("{err}"),
        }
    }

    /// Try to create a CPU context with a fixed parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_with_threads(1).unwrap();
    /// assert_eq!(ctx.num_threads(), 1);
    /// ```
    pub fn try_with_threads(num_threads: usize) -> Result<Self> {
        if num_threads == 0 {
            return Err(Error::InvalidConfig {
                op: "CpuContext::try_with_threads",
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
                        op: "CpuContext::try_with_threads",
                        message: format!("failed to build CPU thread pool: {err}"),
                    })?,
            ))
        };
        Ok(Self { num_threads, pool })
    }

    /// Return this context's CPU parallelism hint.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Run a closure inside this context's CPU execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(1);
    /// let value = ctx.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(pool) => pool.install(op),
            None => op(),
        }
    }

    /// Return the faer parallelism policy for this context.
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn faer_par(&self) -> faer::Par {
        if self.num_threads == 1 {
            faer::Par::Seq
        } else {
            faer::Par::rayon(0)
        }
    }
}

#[cfg(test)]
mod tests;
