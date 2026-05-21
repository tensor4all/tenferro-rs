use std::env;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};

use lru::LruCache;

use crate::{CacheStats, Error, Result};

/// Default number of distinct CPU thread-pool sizes retained process-wide.
pub const DEFAULT_CPU_THREAD_POOL_CACHE_CAPACITY: usize = 16;

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

struct SharedThreadPoolCache {
    pools: LruCache<usize, Arc<rayon::ThreadPool>>,
}

impl SharedThreadPoolCache {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            pools: LruCache::new(capacity),
        }
    }

    fn clear(&mut self) {
        self.pools.clear();
    }

    fn set_capacity(&mut self, capacity: NonZeroUsize) {
        self.pools.resize(capacity);
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.pools.len(),
            retained_bytes: self.pools.len()
                * (size_of::<usize>() + size_of::<Arc<rayon::ThreadPool>>()),
        }
    }
}

fn default_thread_pool_cache_capacity() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_CPU_THREAD_POOL_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN)
}

fn shared_pools() -> &'static Mutex<SharedThreadPoolCache> {
    static POOLS: OnceLock<Mutex<SharedThreadPoolCache>> = OnceLock::new();
    POOLS.get_or_init(|| {
        Mutex::new(SharedThreadPoolCache::new(
            default_thread_pool_cache_capacity(),
        ))
    })
}

fn lock_shared_pools() -> std::sync::MutexGuard<'static, SharedThreadPoolCache> {
    match shared_pools().lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("tenferro-tensor: CPU thread-pool cache mutex was poisoned; recovering");
            poisoned.into_inner()
        }
    }
}

pub(crate) fn try_get_or_create_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    if num_threads == 0 {
        return Err(Error::InvalidConfig {
            op: "CpuContext::try_with_threads",
            message: "thread count must be at least 1".into(),
        });
    }

    let mut pools = lock_shared_pools();
    if let Some(pool) = pools.pools.get(&num_threads) {
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
    pools.pools.put(num_threads, Arc::clone(&pool));
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

    /// Current process-wide CPU thread-pool cache capacity.
    ///
    /// The cache is keyed by requested thread count. Retained-byte stats only
    /// count tenferro's cache handles, not OS thread stacks owned by live pools.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// assert!(CpuContext::shared_pool_cache_capacity().get() > 0);
    /// ```
    pub fn shared_pool_cache_capacity() -> NonZeroUsize {
        lock_shared_pools().pools.cap()
    }

    /// Resize the process-wide CPU thread-pool cache.
    ///
    /// Shrinking evicts least-recently-used cached handles. Existing
    /// [`CpuContext`] values keep their own `Arc` handles alive.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// CpuContext::set_shared_pool_cache_capacity(NonZeroUsize::new(1).unwrap());
    /// assert_eq!(CpuContext::shared_pool_cache_capacity().get(), 1);
    /// ```
    pub fn set_shared_pool_cache_capacity(capacity: NonZeroUsize) {
        lock_shared_pools().set_capacity(capacity);
    }

    /// Clear all cached process-wide CPU thread-pool handles.
    ///
    /// Existing [`CpuContext`] values remain usable because they own `Arc`
    /// handles to their pools.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// CpuContext::clear_shared_pool_cache();
    /// assert_eq!(CpuContext::shared_pool_cache_stats().entries, 0);
    /// ```
    pub fn clear_shared_pool_cache() {
        lock_shared_pools().clear();
    }

    /// Return process-wide CPU thread-pool cache stats.
    ///
    /// `retained_bytes` reports tenferro's retained cache handles, not OS thread
    /// stacks or rayon's internal allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuContext;
    ///
    /// CpuContext::clear_shared_pool_cache();
    /// let stats = CpuContext::shared_pool_cache_stats();
    /// assert_eq!(stats.entries, 0);
    /// assert_eq!(stats.retained_bytes, 0);
    /// ```
    pub fn shared_pool_cache_stats() -> CacheStats {
        lock_shared_pools().stats()
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{CpuContext, DEFAULT_CPU_THREAD_POOL_CACHE_CAPACITY};

    #[test]
    fn shared_pool_cache_is_bounded_and_clearable() {
        CpuContext::clear_shared_pool_cache();
        CpuContext::set_shared_pool_cache_capacity(NonZeroUsize::new(1).unwrap());

        let _one = CpuContext::with_threads(1);
        let _two = CpuContext::with_threads(2);

        let stats = CpuContext::shared_pool_cache_stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.retained_bytes > 0);

        CpuContext::clear_shared_pool_cache();
        assert_eq!(CpuContext::shared_pool_cache_stats().entries, 0);
        CpuContext::set_shared_pool_cache_capacity(
            NonZeroUsize::new(DEFAULT_CPU_THREAD_POOL_CACHE_CAPACITY).unwrap(),
        );
    }
}
