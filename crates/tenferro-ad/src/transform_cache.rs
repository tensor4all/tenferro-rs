use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard};

use tenferro_runtime::{CacheStats, Error, Result};

const DEFAULT_AD_TRANSFORM_CACHE_ENTRIES: usize = 128;
const DEFAULT_AD_TRANSFORM_CACHE_RETAINED_BYTES: usize = 64 * 1024 * 1024;

/// Retention limits for AD transform graph caches.
///
/// The retained-byte limit is a logical payload estimate, not process RSS.
///
/// # Examples
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use tenferro_ad::AdTransformCacheLimits;
///
/// let limits = AdTransformCacheLimits::new(NonZeroUsize::new(4).unwrap());
/// assert_eq!(limits.max_entries().get(), 4);
/// assert!(limits.max_retained_bytes().is_some());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdTransformCacheLimits {
    max_entries: NonZeroUsize,
    max_retained_bytes: Option<NonZeroUsize>,
}

impl AdTransformCacheLimits {
    /// Create AD transform cache limits with the default retained-byte bound.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_ad::AdTransformCacheLimits;
    ///
    /// let limits = AdTransformCacheLimits::new(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(limits.max_entries().get(), 2);
    /// ```
    pub fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            max_retained_bytes: Some(
                NonZeroUsize::new(DEFAULT_AD_TRANSFORM_CACHE_RETAINED_BYTES)
                    .unwrap_or(NonZeroUsize::MIN),
            ),
        }
    }

    /// Return the maximum number of retained AD transform entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdTransformCacheLimits;
    ///
    /// assert!(AdTransformCacheLimits::default().max_entries().get() > 0);
    /// ```
    pub fn max_entries(self) -> NonZeroUsize {
        self.max_entries
    }

    /// Return the logical retained-byte bound, when one is configured.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::AdTransformCacheLimits;
    ///
    /// assert!(AdTransformCacheLimits::default().max_retained_bytes().is_some());
    /// ```
    pub fn max_retained_bytes(self) -> Option<NonZeroUsize> {
        self.max_retained_bytes
    }

    /// Return limits with a new logical retained-byte bound.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_ad::AdTransformCacheLimits;
    ///
    /// let limits = AdTransformCacheLimits::default()
    ///     .with_max_retained_bytes(NonZeroUsize::new(1024).unwrap());
    /// assert_eq!(limits.max_retained_bytes().unwrap().get(), 1024);
    /// ```
    pub fn with_max_retained_bytes(mut self, max_retained_bytes: NonZeroUsize) -> Self {
        self.max_retained_bytes = Some(max_retained_bytes);
        self
    }
}

impl Default for AdTransformCacheLimits {
    fn default() -> Self {
        Self::new(
            NonZeroUsize::new(DEFAULT_AD_TRANSFORM_CACHE_ENTRIES).unwrap_or(NonZeroUsize::MIN),
        )
    }
}

#[derive(Debug)]
pub(crate) struct AdTransformCache {
    store: Mutex<AdTransformCacheStore>,
}

impl AdTransformCache {
    pub(crate) fn new() -> Self {
        Self {
            store: Mutex::new(AdTransformCacheStore::default()),
        }
    }

    pub(crate) fn limits(&self) -> Result<AdTransformCacheLimits> {
        Ok(self.lock_store()?.limits)
    }

    pub(crate) fn set_limits(&self, limits: AdTransformCacheLimits) -> Result<()> {
        self.lock_store()?.set_limits(limits);
        Ok(())
    }

    pub(crate) fn clear(&self) -> Result<()> {
        self.lock_store()?.clear();
        Ok(())
    }

    pub(crate) fn stats(&self) -> Result<CacheStats> {
        Ok(self.lock_store()?.stats())
    }

    fn lock_store(&self) -> Result<MutexGuard<'_, AdTransformCacheStore>> {
        self.store
            .lock()
            .map_err(|_| Error::Internal("AD transform cache lock poisoned".to_string()))
    }
}

#[derive(Debug)]
struct AdTransformCacheStore {
    limits: AdTransformCacheLimits,
    stats: CacheStats,
}

impl AdTransformCacheStore {
    fn set_limits(&mut self, limits: AdTransformCacheLimits) {
        self.limits = limits;
    }

    fn clear(&mut self) {
        self.stats = CacheStats::empty();
    }

    fn stats(&self) -> CacheStats {
        self.stats
    }
}

impl Default for AdTransformCacheStore {
    fn default() -> Self {
        Self {
            limits: AdTransformCacheLimits::default(),
            stats: CacheStats::empty(),
        }
    }
}
