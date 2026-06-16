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
                .expect("DEFAULT_EXTENSION_CACHE_CAPACITY must be non-zero"),
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

/// Bounded type-erased cache storage owned by an extension executor.
pub struct ExtensionCacheStore {
    limits: ExtensionCacheLimits,
    entries: LruCache<ExtensionCacheKey, ExtensionCacheEntry>,
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
        }
    }

    /// Return the active cache limits.
    pub const fn limits(&self) -> ExtensionCacheLimits {
        self.limits
    }

    /// Resize the store and evict least-recently-used entries if needed.
    pub fn set_limits(&mut self, limits: ExtensionCacheLimits) {
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
        self.entries.put(
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
    ///     values.capacity() * std::mem::size_of::<usize>()
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
        self.entries.put(
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
        self.entries
            .get(key)
            .and_then(|entry| entry.value.as_any().downcast_ref::<T>())
    }

    /// Get a mutable typed cache entry, updating its LRU position.
    pub fn get_mut<T>(&mut self, key: &ExtensionCacheKey) -> Option<&mut T>
    where
        T: Any + Send + Sync + 'static,
    {
        self.entries
            .get_mut(key)
            .and_then(|entry| entry.value.as_any_mut().downcast_mut::<T>())
    }

    /// Clear entries selected by `selector`.
    pub fn clear_selected(&mut self, selector: ExtensionCacheSelector) {
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
        self.entries.clear();
    }

    /// Return cache-style stats for entries selected by `selector`.
    pub fn stats(&self, selector: ExtensionCacheSelector) -> CacheStats {
        CacheStats {
            entries: self
                .entries
                .iter()
                .filter(|(key, _)| selector.matches(key))
                .count(),
            retained_bytes: self
                .entries
                .iter()
                .filter(|(key, _)| selector.matches(key))
                .map(|(_, entry)| entry.value.retained_bytes())
                .sum(),
        }
    }
}

impl Default for ExtensionCacheStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
