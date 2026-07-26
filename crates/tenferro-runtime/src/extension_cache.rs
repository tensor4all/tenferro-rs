//! Generic runtime caches for extension executors.
//!
//! Extension payloads describe operation semantics. Runtime plans, vendor
//! handles, and other mutable execution state belong here instead, behind
//! explicit bounded cache ownership.

use std::any::Any;
use std::fmt;
use std::num::NonZeroUsize;

use lru::LruCache;
use tenferro_tensor::CacheStats;

/// Default number of type-erased extension cache entries retained per owner.
pub const DEFAULT_EXTENSION_CACHE_CAPACITY: usize = 256;
/// Default logical retained-byte bound for extension cache entries.
pub const DEFAULT_EXTENSION_CACHE_RETAINED_BYTES: usize = 64 * 1024 * 1024;

const DEFAULT_EXTENSION_CACHE_RETAINED_BYTES_NONZERO: NonZeroUsize =
    match NonZeroUsize::new(DEFAULT_EXTENSION_CACHE_RETAINED_BYTES) {
        Some(value) => value,
        None => NonZeroUsize::MIN,
    };

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
    max_retained_bytes: Option<NonZeroUsize>,
}

impl ExtensionCacheLimits {
    /// Build limits from a maximum entry count and the default byte bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_runtime::ExtensionCacheLimits;
    ///
    /// let limits = ExtensionCacheLimits::new(NonZeroUsize::new(4).unwrap());
    /// assert_eq!(limits.max_entries().get(), 4);
    /// assert!(limits.max_retained_bytes().is_some());
    /// ```
    pub const fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            max_retained_bytes: Some(DEFAULT_EXTENSION_CACHE_RETAINED_BYTES_NONZERO),
        }
    }

    /// Maximum entries retained by the store.
    pub const fn max_entries(self) -> NonZeroUsize {
        self.max_entries
    }

    /// Maximum logical retained bytes, when configured.
    pub const fn max_retained_bytes(self) -> Option<NonZeroUsize> {
        self.max_retained_bytes
    }

    /// Return limits with a new logical retained-byte bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_runtime::ExtensionCacheLimits;
    ///
    /// let limits = ExtensionCacheLimits::default()
    ///     .with_max_retained_bytes(NonZeroUsize::new(1024).unwrap());
    /// assert_eq!(limits.max_retained_bytes().unwrap().get(), 1024);
    /// ```
    pub const fn with_max_retained_bytes(mut self, max_retained_bytes: NonZeroUsize) -> Self {
        self.max_retained_bytes = Some(max_retained_bytes);
        self
    }
}

impl Default for ExtensionCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: NonZeroUsize::new(DEFAULT_EXTENSION_CACHE_CAPACITY)
                .unwrap_or(NonZeroUsize::MIN),
            max_retained_bytes: Some(DEFAULT_EXTENSION_CACHE_RETAINED_BYTES_NONZERO),
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

#[derive(Clone, Copy, Debug)]
struct FamilyEventStats {
    family_id: &'static str,
    stats: ExtensionCacheEventStats,
}

#[derive(Clone, Copy, Debug)]
struct CacheEventStats {
    family_id: &'static str,
    cache_name: &'static str,
    stats: ExtensionCacheEventStats,
}

/// Bounded type-erased cache storage owned by an extension executor.
pub struct ExtensionCacheStore {
    limits: ExtensionCacheLimits,
    entries: LruCache<ExtensionCacheKey, ExtensionCacheEntry>,
    events: ExtensionCacheEventStats,
    family_events: Vec<FamilyEventStats>,
    cache_events: Vec<CacheEventStats>,
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
            family_events: Vec::new(),
            cache_events: Vec::new(),
        }
    }

    /// Return the active cache limits.
    pub const fn limits(&self) -> ExtensionCacheLimits {
        self.limits
    }

    /// Resize the store and evict least-recently-used entries if needed.
    pub fn set_limits(&mut self, limits: ExtensionCacheLimits) {
        self.limits = limits;
        self.evict_to_limits();
        self.entries.resize(limits.max_entries());
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
        let Self {
            entries,
            events,
            family_events,
            cache_events,
            ..
        } = self;
        let Some(entry) = entries.get(key) else {
            record_miss(events, family_events, cache_events, key);
            return None;
        };
        if entry.value.as_any().is::<T>() {
            record_hit(events, family_events, cache_events, key);
            entry.value.as_any().downcast_ref::<T>()
        } else {
            record_miss(events, family_events, cache_events, key);
            None
        }
    }

    /// Get a mutable typed cache entry, updating its LRU position.
    pub fn get_mut<T>(&mut self, key: &ExtensionCacheKey) -> Option<&mut T>
    where
        T: Any + Send + Sync + 'static,
    {
        let Self {
            entries,
            events,
            family_events,
            cache_events,
            ..
        } = self;
        let Some(entry) = entries.get_mut(key) else {
            record_miss(events, family_events, cache_events, key);
            return None;
        };
        if entry.value.as_any().is::<T>() {
            record_hit(events, family_events, cache_events, key);
            entry.value.as_any_mut().downcast_mut::<T>()
        } else {
            record_miss(events, family_events, cache_events, key);
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
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.limits.max_entries().get()
            || self.retained_bytes_exceeds_limit()
        {
            let Some((key, _)) = self.entries.pop_lru() else {
                break;
            };
            self.record_eviction(&key);
        }
    }

    fn retained_bytes_exceeds_limit(&self) -> bool {
        self.limits
            .max_retained_bytes
            .is_some_and(|limit| self.retained_bytes() > limit.get())
    }

    fn retained_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, entry)| entry.value.retained_bytes())
            .fold(0usize, usize::saturating_add)
    }

    fn event_stats(&self, selector: ExtensionCacheSelector) -> ExtensionCacheEventStats {
        match selector {
            ExtensionCacheSelector::All => self.events,
            ExtensionCacheSelector::Family { family_id } => {
                family_event_stats(&self.family_events, family_id)
                    .copied()
                    .unwrap_or_default()
            }
            ExtensionCacheSelector::Cache {
                family_id,
                cache_name,
            } => cache_event_stats(&self.cache_events, family_id, cache_name)
                .copied()
                .unwrap_or_default(),
        }
    }

    fn record_eviction(&mut self, key: &ExtensionCacheKey) {
        record_eviction(
            &mut self.events,
            &mut self.family_events,
            &mut self.cache_events,
            key,
        );
    }

    fn record_clear(&mut self, selector: ExtensionCacheSelector) {
        self.ensure_event_scopes_for_current_entries(selector);
        self.events.clears = self.events.clears.saturating_add(1);
        match selector {
            ExtensionCacheSelector::All => {
                for scoped in &mut self.family_events {
                    scoped.stats.clears = scoped.stats.clears.saturating_add(1);
                }
                for scoped in &mut self.cache_events {
                    scoped.stats.clears = scoped.stats.clears.saturating_add(1);
                }
            }
            ExtensionCacheSelector::Family { family_id } => {
                self.record_family_clear(family_id);
                for scoped in &mut self.cache_events {
                    if scoped.family_id == family_id {
                        scoped.stats.clears = scoped.stats.clears.saturating_add(1);
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
            ensure_family_event_stats(&mut self.family_events, key.family_id);
            ensure_cache_event_stats(&mut self.cache_events, key.family_id, key.cache_name);
        }
    }

    fn record_family_clear(&mut self, family_id: &'static str) {
        let stats = ensure_family_event_stats(&mut self.family_events, family_id);
        stats.clears = stats.clears.saturating_add(1);
    }

    fn record_cache_clear(&mut self, family_id: &'static str, cache_name: &'static str) {
        let stats = ensure_cache_event_stats(&mut self.cache_events, family_id, cache_name);
        stats.clears = stats.clears.saturating_add(1);
    }
}

fn family_event_stats<'a>(
    events: &'a [FamilyEventStats],
    family_id: &'static str,
) -> Option<&'a ExtensionCacheEventStats> {
    events
        .iter()
        .find(|event| event.family_id == family_id)
        .map(|event| &event.stats)
}

fn cache_event_stats<'a>(
    events: &'a [CacheEventStats],
    family_id: &'static str,
    cache_name: &'static str,
) -> Option<&'a ExtensionCacheEventStats> {
    events
        .iter()
        .find(|event| event.family_id == family_id && event.cache_name == cache_name)
        .map(|event| &event.stats)
}

fn ensure_family_event_stats<'a>(
    events: &'a mut Vec<FamilyEventStats>,
    family_id: &'static str,
) -> &'a mut ExtensionCacheEventStats {
    if let Some(index) = events.iter().position(|event| event.family_id == family_id) {
        return &mut events[index].stats;
    }
    events.push(FamilyEventStats {
        family_id,
        stats: ExtensionCacheEventStats::default(),
    });
    &mut events
        .last_mut()
        .expect("just-pushed family event stats")
        .stats
}

fn ensure_cache_event_stats<'a>(
    events: &'a mut Vec<CacheEventStats>,
    family_id: &'static str,
    cache_name: &'static str,
) -> &'a mut ExtensionCacheEventStats {
    if let Some(index) = events
        .iter()
        .position(|event| event.family_id == family_id && event.cache_name == cache_name)
    {
        return &mut events[index].stats;
    }
    events.push(CacheEventStats {
        family_id,
        cache_name,
        stats: ExtensionCacheEventStats::default(),
    });
    &mut events
        .last_mut()
        .expect("just-pushed cache event stats")
        .stats
}

fn record_hit(
    events: &mut ExtensionCacheEventStats,
    family_events: &mut Vec<FamilyEventStats>,
    cache_events: &mut Vec<CacheEventStats>,
    key: &ExtensionCacheKey,
) {
    events.hits = events.hits.saturating_add(1);
    let family = ensure_family_event_stats(family_events, key.family_id);
    family.hits = family.hits.saturating_add(1);
    let cache = ensure_cache_event_stats(cache_events, key.family_id, key.cache_name);
    cache.hits = cache.hits.saturating_add(1);
}

fn record_miss(
    events: &mut ExtensionCacheEventStats,
    family_events: &mut Vec<FamilyEventStats>,
    cache_events: &mut Vec<CacheEventStats>,
    key: &ExtensionCacheKey,
) {
    events.misses = events.misses.saturating_add(1);
    let family = ensure_family_event_stats(family_events, key.family_id);
    family.misses = family.misses.saturating_add(1);
    let cache = ensure_cache_event_stats(cache_events, key.family_id, key.cache_name);
    cache.misses = cache.misses.saturating_add(1);
}

fn record_eviction(
    events: &mut ExtensionCacheEventStats,
    family_events: &mut Vec<FamilyEventStats>,
    cache_events: &mut Vec<CacheEventStats>,
    key: &ExtensionCacheKey,
) {
    events.evictions = events.evictions.saturating_add(1);
    let family = ensure_family_event_stats(family_events, key.family_id);
    family.evictions = family.evictions.saturating_add(1);
    let cache = ensure_cache_event_stats(cache_events, key.family_id, key.cache_name);
    cache.evictions = cache.evictions.saturating_add(1);
}

impl Default for ExtensionCacheStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
