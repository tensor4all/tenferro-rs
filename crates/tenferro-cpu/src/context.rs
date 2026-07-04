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
/// let ctx = CpuContext::with_threads(1).unwrap();
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
        Ok(Self { num_threads, pool })
    }

    fn single_threaded() -> Self {
        Self {
            num_threads: 1,
            pool: None,
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

#[cfg(test)]
mod tests;
