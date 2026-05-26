//! Cache accounting primitives shared by tensor backends and facade runtimes.

/// Entry and retained-byte accounting for one cache.
///
/// `retained_bytes` reports the cache-owned logical payload estimate. It does
/// not include allocator arena slack, operating-system RSS, or memory retained
/// by unrelated process allocators.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::CacheStats;
///
/// let stats = CacheStats {
///     entries: 2,
///     retained_bytes: 128,
/// };
/// assert_eq!(stats.entries, 2);
/// assert_eq!(stats.retained_bytes, 128);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache entries currently retained.
    pub entries: usize,
    /// Cache-owned retained payload estimate in bytes.
    pub retained_bytes: usize,
}

impl CacheStats {
    /// Return an empty stats snapshot.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::CacheStats;
    ///
    /// let stats = CacheStats::empty();
    /// assert_eq!(stats.entries, 0);
    /// assert_eq!(stats.retained_bytes, 0);
    /// ```
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Control surface required for backend runtime caches owned by higher-level runtimes.
///
/// Backend caches use this trait so an `Engine` can clear and inspect the cache
/// without knowing backend-specific entry types.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{CacheStats, RuntimeCacheControl};
///
/// let mut cache = ();
/// assert_eq!(cache.stats(), CacheStats::empty());
/// cache.clear();
/// assert_eq!(cache.stats().entries, 0);
/// ```
pub trait RuntimeCacheControl: Default {
    /// Remove every retained cache entry.
    fn clear(&mut self);

    /// Snapshot retained entries and retained bytes.
    fn stats(&self) -> CacheStats;
}

impl RuntimeCacheControl for () {
    fn clear(&mut self) {}

    fn stats(&self) -> CacheStats {
        CacheStats::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheStats, RuntimeCacheControl};

    #[test]
    fn cache_stats_empty_reports_zeroes() {
        assert_eq!(
            CacheStats::empty(),
            CacheStats {
                entries: 0,
                retained_bytes: 0
            }
        );
    }

    #[test]
    fn unit_runtime_cache_control_is_empty_and_clearable() {
        let mut cache = ();
        assert_eq!(cache.stats(), CacheStats::empty());
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
    }
}
