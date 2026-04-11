use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};

use crate::{Error, Result};

/// Reusable CPU execution context backed by an owned rayon thread pool.
///
/// # Examples
///
/// ```ignore
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

pub(crate) fn get_or_create_pool(num_threads: usize) -> Arc<rayon::ThreadPool> {
    let mut pools = shared_pools()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&num_threads) {
        return Arc::clone(pool);
    }

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to create rayon thread pool: {e}")),
    );
    pools.insert(num_threads, Arc::clone(&pool));
    pool
}

impl CpuContext {
    /// Create a CPU context from `RAYON_NUM_THREADS`, or fall back to the
    /// process-visible CPU count.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::try_from_env().unwrap();
    /// let _ = ctx.num_threads();
    /// ```
    pub fn try_from_env() -> Result<Self> {
        match env::var("RAYON_NUM_THREADS") {
            Ok(value) => {
                let num_threads = value.parse::<usize>().map_err(|err| Error::InvalidConfig {
                    op: "CpuContext::try_from_env",
                    message: format!("invalid RAYON_NUM_THREADS value {value:?}: {err}"),
                })?;
                if num_threads == 0 {
                    return Err(Error::InvalidConfig {
                        op: "CpuContext::try_from_env",
                        message: "RAYON_NUM_THREADS must be at least 1".to_string(),
                    });
                }
                Ok(Self::with_threads(num_threads))
            }
            Err(env::VarError::NotPresent) => {
                Ok(Self::with_threads(super::affinity::available_parallelism()))
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
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// let ctx = CpuContext::with_threads(2);
    /// assert_eq!(ctx.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        assert!(num_threads >= 1, "thread count must be >= 1");
        Self {
            pool: get_or_create_pool(num_threads),
        }
    }

    /// Return the number of threads in this context's owned rayon pool.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
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
