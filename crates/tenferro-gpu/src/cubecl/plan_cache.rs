//! Hash-keyed LRU core shared by cuTENSOR plan caches.
//!
//! The cache is keyed by a caller-computed 64-bit hash of the borrowed plan
//! spec, so cache hits never materialize an owned key. The fully materialized
//! key is retained inside each entry and verified on lookup, so a 64-bit hash
//! collision degrades to a plan rebuild instead of a wrong plan.

use std::num::NonZeroUsize;

use lru::LruCache;
use tenferro_tensor::CacheStats;

/// One retained entry: the materialized key for collision verification, the
/// cached value, and the entry's logical retained-byte estimate.
struct PlanCacheEntry<K, V> {
    key: K,
    value: V,
    retained_bytes: usize,
}

/// Bounded LRU plan cache with O(1) hits and incrementally maintained
/// retained-byte accounting.
///
/// Retained bytes are the cache's owned/logical payload estimate: the sum of
/// the per-entry estimates supplied at insertion plus the cache struct
/// itself. The estimate changes only on insertion and eviction, so lookups
/// stay allocation-free.
pub(super) struct LruPlanCache<K, V> {
    entries: LruCache<u64, PlanCacheEntry<K, V>>,
    stats: CacheStats,
    entry_retained_bytes: usize,
}

impl<K, V> LruPlanCache<K, V> {
    pub(super) fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            entries: LruCache::new(max_entries),
            stats: CacheStats::empty(),
            entry_retained_bytes: 0,
        }
    }

    /// Ensure an entry for `hash` exists, building it on a miss.
    ///
    /// A hit promotes the entry to most-recently-used in O(1). `matches`
    /// verifies the stored key against the caller's borrowed spec; a same-hash
    /// entry for a different spec (a 64-bit hash collision) is rebuilt and
    /// replaces the colliding entry. `build` returns the materialized key, the
    /// value, and the entry's logical retained-byte estimate.
    ///
    /// Returns `true` when the retained entries changed (insert, eviction, or
    /// collision replacement) so callers can refresh external accounting only
    /// when needed.
    ///
    /// # Errors
    ///
    /// Propagates the `build` error on a miss; the cache is unchanged in that
    /// case.
    pub(super) fn ensure(
        &mut self,
        hash: u64,
        matches: impl FnOnce(&K) -> bool,
        build: impl FnOnce() -> crate::Result<(K, V, usize)>,
    ) -> crate::Result<bool> {
        if let Some(entry) = self.entries.get(&hash) {
            if matches(&entry.key) {
                self.stats.hits = self.stats.hits.saturating_add(1);
                return Ok(false);
            }
        }
        self.stats.misses = self.stats.misses.saturating_add(1);
        let (key, value, retained_bytes) = build()?;
        self.entry_retained_bytes = self.entry_retained_bytes.saturating_add(retained_bytes);
        if let Some((_, removed)) = self.entries.push(
            hash,
            PlanCacheEntry {
                key,
                value,
                retained_bytes,
            },
        ) {
            // Either the least-recently-used entry was evicted at capacity or
            // a same-hash collision entry was replaced; both drop one entry.
            self.entry_retained_bytes = self
                .entry_retained_bytes
                .saturating_sub(removed.retained_bytes);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        Ok(true)
    }

    /// Return the value for `hash` when the stored key matches, promoting the
    /// entry to most-recently-used.
    pub(super) fn get(&mut self, hash: u64, matches: impl FnOnce(&K) -> bool) -> Option<&V> {
        let entry = self.entries.get(&hash)?;
        matches(&entry.key).then_some(&entry.value)
    }

    /// Add retained bytes lazily allocated by an existing cache value.
    pub(super) fn add_retained_bytes(
        &mut self,
        hash: u64,
        matches: impl FnOnce(&K) -> bool,
        added: usize,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&hash) else {
            return false;
        };
        if !matches(&entry.key) {
            return false;
        }
        entry.retained_bytes = entry.retained_bytes.saturating_add(added);
        self.entry_retained_bytes = self.entry_retained_bytes.saturating_add(added);
        true
    }

    pub(super) fn max_entries(&self) -> NonZeroUsize {
        self.entries.cap()
    }

    /// Replace the entry bound, evicting least-recently-used entries first.
    pub(super) fn set_max_entries(&mut self, max_entries: NonZeroUsize) {
        while self.entries.len() > max_entries.get() {
            let Some((_, removed)) = self.entries.pop_lru() else {
                break;
            };
            self.entry_retained_bytes = self
                .entry_retained_bytes
                .saturating_sub(removed.retained_bytes);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.entries.resize(max_entries);
    }

    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.retained_bytes(),
            ..self.stats
        }
    }

    /// The cache's owned/logical retained-byte estimate, maintained
    /// incrementally on insert and evict.
    pub(super) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.entry_retained_bytes)
    }

    /// Iterate the retained values in most- to least-recently-used order.
    #[cfg(test)]
    pub(super) fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, entry)| &entry.value)
    }
}
