use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};

use crate::{Error, Result};

/// Reusable CPU execution context backed by an owned rayon thread pool.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cpu::CpuContext;
///
/// let ctx = CpuContext::with_threads(1);
/// let seen = ctx.install(|| rayon::current_num_threads());
/// assert_eq!(seen, 1);
/// ```
#[derive(Clone)]
pub struct CpuContext {
    pool: Arc<rayon::ThreadPool>,
}

fn shared_pools() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn try_get_or_create_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    if num_threads == 0 {
        return Err(Error::InvalidConfig {
            op: "CpuContext::try_with_threads",
            message: "thread count must be at least 1".into(),
        });
    }

    let mut pools = match shared_pools().lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("tenferro-tensor: CPU thread-pool cache mutex was poisoned; recovering");
            poisoned.into_inner()
        }
    };
    if let Some(pool) = pools.get(&num_threads) {
        return Ok(Arc::clone(pool));
    }

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|err| Error::BackendFailure {
                op: "CpuContext::try_with_threads",
                message: format!("failed to create rayon thread pool: {err}"),
            })?,
    );
    pools.insert(num_threads, Arc::clone(&pool));
    Ok(pool)
}

impl CpuContext {
    /// Create a CPU context from `RAYON_NUM_THREADS`, or fall back to the
    /// process-visible CPU count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::from_env();
    /// let _ = ctx.num_threads();
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
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_from_env()
    ///     .unwrap_or_else(|_| CpuContext::with_threads(1));
    /// let _ = ctx.num_threads();
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
                    Error::BackendFailure { message, .. } => Error::BackendFailure {
                        op: "CpuContext::try_from_env",
                        message,
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

    /// Create a CPU context with a fixed rayon thread-pool size.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        match Self::try_with_threads(num_threads) {
            Ok(ctx) => ctx,
            Err(err) => panic!("{err}"),
        }
    }

    /// Try to create a CPU context with a fixed rayon thread-pool size.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_with_threads(1).unwrap();
    /// assert_eq!(ctx.num_threads(), 1);
    /// ```
    pub fn try_with_threads(num_threads: usize) -> Result<Self> {
        Ok(Self {
            pool: try_get_or_create_pool(num_threads)?,
        })
    }

    /// Return the number of threads in this context's owned rayon pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Run a closure inside this context's owned rayon thread pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(1);
    /// let value = ctx.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R>(&self, op: impl FnOnce() -> R + Send) -> R
    where
        R: Send,
    {
        self.pool.install(op)
    }

    /// Return the faer parallelism policy for this context.
    #[cfg(feature = "cpu-faer")]
    pub(crate) fn faer_par(&self) -> faer::Par {
        if self.num_threads() == 1 {
            faer::Par::Seq
        } else {
            faer::Par::rayon(0)
        }
    }
}
