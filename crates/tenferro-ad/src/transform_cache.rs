use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard};

use lru::LruCache;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{CacheStats, Error, Result};
use tidu::eager::RecordedGraph;
use tidu::LinearizedGraph;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EagerAdTransformCacheKey {
    recorded_graph_fingerprint: u64,
    output_slots: Vec<usize>,
}

impl EagerAdTransformCacheKey {
    pub(crate) fn new(graph: &RecordedGraph<StdTensorOp>, output_slots: &[usize]) -> Self {
        Self {
            recorded_graph_fingerprint: eager_recorded_graph_fingerprint(graph),
            output_slots: output_slots.to_vec(),
        }
    }
}

fn eager_recorded_graph_fingerprint(graph: &RecordedGraph<StdTensorOp>) -> u64 {
    let mut hasher = DefaultHasher::new();
    graph.input_keys().hash(&mut hasher);
    graph.output_keys().hash(&mut hasher);
    graph.as_graph().inputs().hash(&mut hasher);
    graph.as_graph().outputs().hash(&mut hasher);
    for value in graph.as_graph().values() {
        value.key.hash(&mut hasher);
        value.producer.hash(&mut hasher);
    }
    for op in graph.as_graph().operations() {
        op.operation.hash(&mut hasher);
        op.inputs.hash(&mut hasher);
        op.outputs.hash(&mut hasher);
        op.role.hash(&mut hasher);
    }
    hasher.finish()
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

    pub(crate) fn get_eager_linearized(
        &self,
        key: &EagerAdTransformCacheKey,
    ) -> Result<Option<Arc<LinearizedGraph<StdTensorOp>>>> {
        Ok(self.lock_store()?.get_eager_linearized(key))
    }

    pub(crate) fn put_eager_linearized(
        &self,
        key: EagerAdTransformCacheKey,
        value: Arc<LinearizedGraph<StdTensorOp>>,
    ) -> Result<()> {
        self.lock_store()?.put_eager_linearized(key, value);
        Ok(())
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
    entries: LruCache<AdTransformCacheKey, AdTransformCacheEntryWithStats>,
    stats: CacheStats,
}

impl AdTransformCacheStore {
    fn set_limits(&mut self, limits: AdTransformCacheLimits) {
        self.limits = limits;
        self.evict_to_limits();
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.stats = CacheStats::empty();
    }

    fn stats(&self) -> CacheStats {
        self.stats
    }

    fn get_eager_linearized(
        &mut self,
        key: &EagerAdTransformCacheKey,
    ) -> Option<Arc<LinearizedGraph<StdTensorOp>>> {
        self.entries
            .get(&AdTransformCacheKey::EagerLinearize(key.clone()))
            .and_then(|entry| match &entry.entry {
                AdTransformCacheEntry::EagerLinearized(linear) => Some(Arc::clone(linear)),
            })
    }

    fn put_eager_linearized(
        &mut self,
        key: EagerAdTransformCacheKey,
        value: Arc<LinearizedGraph<StdTensorOp>>,
    ) {
        let key = AdTransformCacheKey::EagerLinearize(key);
        let entry = AdTransformCacheEntry::EagerLinearized(value);
        self.put_entry(key, entry);
    }

    fn put_entry(&mut self, key: AdTransformCacheKey, entry: AdTransformCacheEntry) {
        let retained_bytes = ad_transform_cache_entry_retained_bytes(&key, &entry);
        let entry = AdTransformCacheEntryWithStats {
            entry,
            retained_bytes,
        };
        self.stats.entries = self.entries.len();
        self.stats.retained_bytes = self.stats.retained_bytes.saturating_add(retained_bytes);
        if let Some((_old_key, old_entry)) = self.entries.push(key, entry) {
            self.stats.retained_bytes = self
                .stats
                .retained_bytes
                .saturating_sub(old_entry.retained_bytes);
        }
        self.stats.entries = self.entries.len();
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.limits.max_entries.get()
            || self
                .limits
                .max_retained_bytes
                .is_some_and(|limit| self.stats.retained_bytes > limit.get())
        {
            let Some((_key, entry)) = self.entries.pop_lru() else {
                break;
            };
            self.stats.retained_bytes = self
                .stats
                .retained_bytes
                .saturating_sub(entry.retained_bytes);
        }
        self.stats.entries = self.entries.len();
    }
}

impl Default for AdTransformCacheStore {
    fn default() -> Self {
        Self {
            limits: AdTransformCacheLimits::default(),
            entries: LruCache::unbounded(),
            stats: CacheStats::empty(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AdTransformCacheKey {
    EagerLinearize(EagerAdTransformCacheKey),
}

enum AdTransformCacheEntry {
    EagerLinearized(Arc<LinearizedGraph<StdTensorOp>>),
}

impl fmt::Debug for AdTransformCacheEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EagerLinearized(_) => f.write_str("EagerLinearized(..)"),
        }
    }
}

#[derive(Debug)]
struct AdTransformCacheEntryWithStats {
    entry: AdTransformCacheEntry,
    retained_bytes: usize,
}

fn ad_transform_cache_entry_retained_bytes(
    key: &AdTransformCacheKey,
    entry: &AdTransformCacheEntry,
) -> usize {
    size_of::<AdTransformCacheKey>()
        + ad_transform_cache_key_retained_bytes(key)
        + size_of::<AdTransformCacheEntry>()
        + ad_transform_cache_value_retained_bytes(entry)
}

fn ad_transform_cache_key_retained_bytes(key: &AdTransformCacheKey) -> usize {
    match key {
        AdTransformCacheKey::EagerLinearize(key) => {
            key.output_slots.capacity() * size_of::<usize>()
        }
    }
}

fn ad_transform_cache_value_retained_bytes(entry: &AdTransformCacheEntry) -> usize {
    match entry {
        AdTransformCacheEntry::EagerLinearized(linear) => {
            size_of::<Arc<LinearizedGraph<StdTensorOp>>>()
                + size_of::<LinearizedGraph<StdTensorOp>>()
                + size_of_val(linear.as_graph().values())
                + size_of_val(linear.as_graph().operations())
                + size_of_val(linear.tangent_inputs())
                + size_of_val(linear.tangent_outputs())
        }
    }
}
