//! Generic runtime caches for extension executors.
//!
//! Extension payloads describe operation semantics. Runtime plans, vendor
//! handles, and other mutable execution state belong here instead, behind
//! explicit bounded cache ownership.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;

use lru::LruCache;
use tenferro_tensor::CacheStats;

/// Default number of type-erased extension cache entries retained per owner.
pub const DEFAULT_EXTENSION_CACHE_CAPACITY: usize = 256;

/// A stable key for one extension-owned runtime cache entry.
///
/// `family_id` names the extension family, `cache_name` names the specific
/// cache within that family, and `discriminator` is chosen by the extension
/// executor from shape, dtype, device, or other runtime planning inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtensionCacheKey {
    /// Extension family that owns this cache entry.
    pub family_id: &'static str,
    /// Cache namespace within the extension family.
    pub cache_name: &'static str,
    /// Extension-defined stable discriminator for this runtime entry.
    pub discriminator: u64,
}

impl ExtensionCacheKey {
    /// Build an extension cache key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionCacheKey;
    ///
    /// let key = ExtensionCacheKey::new("example.identity.v1", "plans", 7);
    /// assert_eq!(key.cache_name, "plans");
    /// ```
    pub const fn new(
        family_id: &'static str,
        cache_name: &'static str,
        discriminator: u64,
    ) -> Self {
        Self {
            family_id,
            cache_name,
            discriminator,
        }
    }
}

/// Selector used to inspect or clear extension cache entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtensionCacheSelector {
    /// Select every extension cache entry.
    All,
    /// Select every entry owned by an extension family.
    Family { family_id: &'static str },
    /// Select entries in one named cache for one extension family.
    Cache {
        family_id: &'static str,
        cache_name: &'static str,
    },
}

impl ExtensionCacheSelector {
    /// Return `true` when this selector includes `key`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ExtensionCacheKey, ExtensionCacheSelector};
    ///
    /// let key = ExtensionCacheKey::new("example.identity.v1", "plans", 0);
    /// assert!(ExtensionCacheSelector::Family {
    ///     family_id: "example.identity.v1",
    /// }.matches(&key));
    /// ```
    pub fn matches(&self, key: &ExtensionCacheKey) -> bool {
        match *self {
            Self::All => true,
            Self::Family { family_id } => key.family_id == family_id,
            Self::Cache {
                family_id,
                cache_name,
            } => key.family_id == family_id && key.cache_name == cache_name,
        }
    }
}

/// Bounded retention limits for extension runtime caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionCacheLimits {
    max_entries: NonZeroUsize,
}

impl ExtensionCacheLimits {
    /// Build limits from a maximum entry count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_runtime::ExtensionCacheLimits;
    ///
    /// let limits = ExtensionCacheLimits::new(NonZeroUsize::new(4).unwrap());
    /// assert_eq!(limits.max_entries().get(), 4);
    /// ```
    pub const fn new(max_entries: NonZeroUsize) -> Self {
        Self { max_entries }
    }

    /// Maximum entries retained by the store.
    pub const fn max_entries(self) -> NonZeroUsize {
        self.max_entries
    }
}

impl Default for ExtensionCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: NonZeroUsize::new(DEFAULT_EXTENSION_CACHE_CAPACITY)
                .unwrap_or(NonZeroUsize::MIN),
        }
    }
}

trait ExtensionCacheValue: Send + Sync {
    fn as_any(&self) -> &(dyn Any + Send + Sync);
    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync);
    fn retained_bytes(&self) -> usize;
}

struct FixedRetainedBytes<T> {
    value: T,
    retained_bytes: usize,
}

impl<T> ExtensionCacheValue for FixedRetainedBytes<T>
where
    T: Any + Send + Sync + 'static,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        &self.value
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        &mut self.value
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

struct DynamicRetainedBytes<T, F> {
    value: T,
    retained_bytes: F,
}

impl<T, F> ExtensionCacheValue for DynamicRetainedBytes<T, F>
where
    T: Any + Send + Sync + 'static,
    F: Fn(&T) -> usize + Send + Sync + 'static,
{
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        &self.value
    }

    fn as_any_mut(&mut self) -> &mut (dyn Any + Send + Sync) {
        &mut self.value
    }

    fn retained_bytes(&self) -> usize {
        (self.retained_bytes)(&self.value)
    }
}

struct ExtensionCacheEntry {
    value: Box<dyn ExtensionCacheValue>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ExtensionCacheEventStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}

impl ExtensionCacheEventStats {
    fn to_cache_stats(self, entries: usize, retained_bytes: usize) -> CacheStats {
        CacheStats {
            entries,
            retained_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            clears: self.clears,
        }
    }
}

/// Bounded type-erased cache storage owned by an extension executor.
pub struct ExtensionCacheStore {
    limits: ExtensionCacheLimits,
    entries: LruCache<ExtensionCacheKey, ExtensionCacheEntry>,
    events: ExtensionCacheEventStats,
    family_events: HashMap<&'static str, ExtensionCacheEventStats>,
    cache_events: HashMap<(&'static str, &'static str), ExtensionCacheEventStats>,
}

impl fmt::Debug for ExtensionCacheStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionCacheStore")
            .field("limits", &self.limits)
            .field("stats", &self.stats(ExtensionCacheSelector::All))
            .finish_non_exhaustive()
    }
}

impl ExtensionCacheStore {
    /// Create an empty cache store with default limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionCacheStore;
    ///
    /// let store = ExtensionCacheStore::new();
    /// assert_eq!(store.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self::with_limits(ExtensionCacheLimits::default())
    }

    /// Create an empty cache store with explicit limits.
    pub fn with_limits(limits: ExtensionCacheLimits) -> Self {
        Self {
            entries: LruCache::new(limits.max_entries()),
            limits,
            events: ExtensionCacheEventStats::default(),
            family_events: HashMap::new(),
            cache_events: HashMap::new(),
        }
    }

    /// Return the active cache limits.
    pub const fn limits(&self) -> ExtensionCacheLimits {
        self.limits
    }

    /// Resize the store and evict least-recently-used entries if needed.
    pub fn set_limits(&mut self, limits: ExtensionCacheLimits) {
        while self.entries.len() > limits.max_entries().get() {
            if let Some((key, _)) = self.entries.pop_lru() {
                self.record_eviction(&key);
            }
        }
        self.entries.resize(limits.max_entries());
        self.limits = limits;
    }

    /// Current entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or replace a typed cache entry.
    pub fn put<T>(&mut self, key: ExtensionCacheKey, value: T, retained_bytes: usize)
    where
        T: Any + Send + Sync + 'static,
    {
        self.put_entry(
            key,
            ExtensionCacheEntry {
                value: Box::new(FixedRetainedBytes {
                    value,
                    retained_bytes,
                }),
            },
        );
    }

    /// Insert or replace a typed cache entry whose retained bytes are computed
    /// from the current value whenever stats are requested.
    ///
    /// Use this for entries that mutate after insertion, such as compiled
    /// execution plans with backend-owned nested caches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ExtensionCacheKey, ExtensionCacheStore, ExtensionCacheSelector};
    ///
    /// let mut store = ExtensionCacheStore::new();
    /// let key = ExtensionCacheKey::new("example.cache.v1", "plans", 0);
    /// store.put_with_retained_bytes(key, Vec::<usize>::with_capacity(2), |values| {
    ///     values.capacity().saturating_mul(std::mem::size_of::<usize>())
    /// });
    /// let values = store.get_mut::<Vec<usize>>(&key).unwrap();
    /// values.reserve_exact(4);
    /// let retained_capacity = values.capacity();
    ///
    /// assert_eq!(
    ///     store.stats(ExtensionCacheSelector::All).retained_bytes,
    ///     retained_capacity * std::mem::size_of::<usize>()
    /// );
    /// ```
    pub fn put_with_retained_bytes<T, F>(
        &mut self,
        key: ExtensionCacheKey,
        value: T,
        retained_bytes: F,
    ) where
        T: Any + Send + Sync + 'static,
        F: Fn(&T) -> usize + Send + Sync + 'static,
    {
        self.put_entry(
            key,
            ExtensionCacheEntry {
                value: Box::new(DynamicRetainedBytes {
                    value,
                    retained_bytes,
                }),
            },
        );
    }

    /// Get a typed cache entry, updating its LRU position.
    pub fn get<T>(&mut self, key: &ExtensionCacheKey) -> Option<&T>
    where
        T: Any + Send + Sync + 'static,
    {
        let is_hit = self
            .entries
            .peek(key)
            .is_some_and(|entry| entry.value.as_any().is::<T>());
        if is_hit {
            self.record_hit(key);
            self.entries
                .get(key)
                .and_then(|entry| entry.value.as_any().downcast_ref::<T>())
        } else {
            let _ = self.entries.get(key);
            self.record_miss(key);
            None
        }
    }

    /// Get a mutable typed cache entry, updating its LRU position.
    pub fn get_mut<T>(&mut self, key: &ExtensionCacheKey) -> Option<&mut T>
    where
        T: Any + Send + Sync + 'static,
    {
        let is_hit = self
            .entries
            .peek(key)
            .is_some_and(|entry| entry.value.as_any().is::<T>());
        if is_hit {
            self.record_hit(key);
            self.entries
                .get_mut(key)
                .and_then(|entry| entry.value.as_any_mut().downcast_mut::<T>())
        } else {
            let _ = self.entries.get_mut(key);
            self.record_miss(key);
            None
        }
    }

    /// Clear entries selected by `selector`.
    pub fn clear_selected(&mut self, selector: ExtensionCacheSelector) {
        self.record_clear(selector);
        if selector == ExtensionCacheSelector::All {
            self.entries.clear();
            return;
        }

        let keys: Vec<_> = self
            .entries
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| selector.matches(key))
            .collect();
        for key in keys {
            self.entries.pop(&key);
        }
    }

    /// Clear every extension cache entry.
    pub fn clear(&mut self) {
        self.record_clear(ExtensionCacheSelector::All);
        self.entries.clear();
    }

    /// Return cache-style stats for entries selected by `selector`.
    pub fn stats(&self, selector: ExtensionCacheSelector) -> CacheStats {
        let entries = self
            .entries
            .iter()
            .filter(|(key, _)| selector.matches(key))
            .count();
        let retained_bytes = self
            .entries
            .iter()
            .filter(|(key, _)| selector.matches(key))
            .map(|(_, entry)| entry.value.retained_bytes())
            .fold(0usize, usize::saturating_add);
        self.event_stats(selector)
            .to_cache_stats(entries, retained_bytes)
    }

    fn put_entry(&mut self, key: ExtensionCacheKey, entry: ExtensionCacheEntry) {
        if let Some((removed_key, _)) = self.entries.push(key, entry) {
            if removed_key != key {
                self.record_eviction(&removed_key);
            }
        }
    }

    fn event_stats(&self, selector: ExtensionCacheSelector) -> ExtensionCacheEventStats {
        match selector {
            ExtensionCacheSelector::All => self.events,
            ExtensionCacheSelector::Family { family_id } => self
                .family_events
                .get(family_id)
                .copied()
                .unwrap_or_default(),
            ExtensionCacheSelector::Cache {
                family_id,
                cache_name,
            } => self
                .cache_events
                .get(&(family_id, cache_name))
                .copied()
                .unwrap_or_default(),
        }
    }

    fn event_stats_for_key_mut(
        &mut self,
        key: &ExtensionCacheKey,
    ) -> (
        &mut ExtensionCacheEventStats,
        &mut ExtensionCacheEventStats,
        &mut ExtensionCacheEventStats,
    ) {
        let family = self.family_events.entry(key.family_id).or_default();
        let cache = self
            .cache_events
            .entry((key.family_id, key.cache_name))
            .or_default();
        (&mut self.events, family, cache)
    }

    fn record_hit(&mut self, key: &ExtensionCacheKey) {
        let (all, family, cache) = self.event_stats_for_key_mut(key);
        all.hits = all.hits.saturating_add(1);
        family.hits = family.hits.saturating_add(1);
        cache.hits = cache.hits.saturating_add(1);
    }

    fn record_miss(&mut self, key: &ExtensionCacheKey) {
        let (all, family, cache) = self.event_stats_for_key_mut(key);
        all.misses = all.misses.saturating_add(1);
        family.misses = family.misses.saturating_add(1);
        cache.misses = cache.misses.saturating_add(1);
    }

    fn record_eviction(&mut self, key: &ExtensionCacheKey) {
        let (all, family, cache) = self.event_stats_for_key_mut(key);
        all.evictions = all.evictions.saturating_add(1);
        family.evictions = family.evictions.saturating_add(1);
        cache.evictions = cache.evictions.saturating_add(1);
    }

    fn record_clear(&mut self, selector: ExtensionCacheSelector) {
        self.ensure_event_scopes_for_current_entries(selector);
        self.events.clears = self.events.clears.saturating_add(1);
        match selector {
            ExtensionCacheSelector::All => {
                for stats in self.family_events.values_mut() {
                    stats.clears = stats.clears.saturating_add(1);
                }
                for stats in self.cache_events.values_mut() {
                    stats.clears = stats.clears.saturating_add(1);
                }
            }
            ExtensionCacheSelector::Family { family_id } => {
                self.record_family_clear(family_id);
                for ((cache_family_id, _), stats) in &mut self.cache_events {
                    if *cache_family_id == family_id {
                        stats.clears = stats.clears.saturating_add(1);
                    }
                }
            }
            ExtensionCacheSelector::Cache {
                family_id,
                cache_name,
            } => {
                self.record_family_clear(family_id);
                self.record_cache_clear(family_id, cache_name);
            }
        }
    }

    fn ensure_event_scopes_for_current_entries(&mut self, selector: ExtensionCacheSelector) {
        let keys: Vec<_> = self
            .entries
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| selector.matches(key))
            .collect();
        for key in keys {
            self.family_events.entry(key.family_id).or_default();
            self.cache_events
                .entry((key.family_id, key.cache_name))
                .or_default();
        }
    }

    fn record_family_clear(&mut self, family_id: &'static str) {
        let stats = self.family_events.entry(family_id).or_default();
        stats.clears = stats.clears.saturating_add(1);
    }

    fn record_cache_clear(&mut self, family_id: &'static str, cache_name: &'static str) {
        let stats = self
            .cache_events
            .entry((family_id, cache_name))
            .or_default();
        stats.clears = stats.clears.saturating_add(1);
    }
}

impl Default for ExtensionCacheStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
