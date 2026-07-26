use std::fmt;
use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, MutexGuard};

use lru::LruCache;
use tenferro_runtime::program::{
    FrozenProgram, ProgramValueMetadata, SemanticFingerprint, SemanticProgram,
};
use tenferro_runtime::{CacheStats, Error, ErrorPhase, Result};

use crate::semantic_transform::SemanticAdProgram;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SemanticAdTransformKind {
    Jvp,
    Vjp,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SemanticAdTransformCacheKey {
    kind: SemanticAdTransformKind,
    input_fingerprint: SemanticFingerprint,
    input_metadata: Box<[ProgramValueMetadata]>,
    active_inputs: Box<[bool]>,
    active_outputs: Box<[bool]>,
}

impl SemanticAdTransformCacheKey {
    pub(crate) fn jvp(input: &FrozenProgram, active_inputs: &[bool]) -> Self {
        Self {
            kind: SemanticAdTransformKind::Jvp,
            input_fingerprint: input.program.semantic_fingerprint(),
            input_metadata: semantic_input_metadata(input),
            active_inputs: active_inputs.into(),
            active_outputs: Box::new([]),
        }
    }

    pub(crate) fn vjp(
        input: &FrozenProgram,
        active_inputs: &[bool],
        active_outputs: &[bool],
    ) -> Self {
        Self {
            kind: SemanticAdTransformKind::Vjp,
            input_fingerprint: input.program.semantic_fingerprint(),
            input_metadata: semantic_input_metadata(input),
            active_inputs: active_inputs.into(),
            active_outputs: active_outputs.into(),
        }
    }
}

fn semantic_input_metadata(input: &FrozenProgram) -> Box<[ProgramValueMetadata]> {
    input.input_metadata_with_bound_shapes()
}

#[derive(Clone)]
struct CachedSemanticAdTransform {
    input: Arc<SemanticProgram>,
    output: Arc<SemanticAdProgram>,
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

    pub(crate) fn get_semantic(
        &self,
        key: &SemanticAdTransformCacheKey,
        input: &FrozenProgram,
    ) -> Result<Option<Arc<SemanticAdProgram>>> {
        Ok(self.lock_store()?.get_semantic(key, input))
    }

    pub(crate) fn put_semantic(
        &self,
        key: SemanticAdTransformCacheKey,
        input: &FrozenProgram,
        output: Arc<SemanticAdProgram>,
    ) -> Result<()> {
        self.lock_store()?.put_semantic(key, input, output);
        Ok(())
    }

    fn lock_store(&self) -> Result<MutexGuard<'_, AdTransformCacheStore>> {
        self.store.lock().map_err(|_| {
            Error::runtime_state("ad_transform_cache", ErrorPhase::Compile, "lock poisoned")
        })
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
        let clears = self.stats.clears.saturating_add(1);
        self.entries.clear();
        self.stats = CacheStats {
            clears,
            ..CacheStats::empty()
        };
    }

    fn stats(&self) -> CacheStats {
        self.stats
    }

    fn get_semantic(
        &mut self,
        key: &SemanticAdTransformCacheKey,
        input: &FrozenProgram,
    ) -> Option<Arc<SemanticAdProgram>> {
        let cache_key = AdTransformCacheKey::Semantic(key.clone());
        let result = self
            .entries
            .get(&cache_key)
            .and_then(|entry| match &entry.entry {
                AdTransformCacheEntry::Semantic(bucket) => bucket
                    .iter()
                    .find(|cached| cached.input.semantic_eq(input.program.as_ref()))
                    .map(|cached| Arc::clone(&cached.output)),
            });
        if result.is_some() {
            self.stats.hits = self.stats.hits.saturating_add(1);
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
        }
        result
    }

    fn put_semantic(
        &mut self,
        key: SemanticAdTransformCacheKey,
        input: &FrozenProgram,
        output: Arc<SemanticAdProgram>,
    ) {
        let cache_key = AdTransformCacheKey::Semantic(key);
        let mut bucket = match self.entries.pop(&cache_key) {
            Some(entry) => {
                self.stats.retained_bytes = self
                    .stats
                    .retained_bytes
                    .saturating_sub(entry.retained_bytes);
                match entry.entry {
                    AdTransformCacheEntry::Semantic(bucket) => bucket,
                }
            }
            None => Vec::new(),
        };
        if let Some(cached) = bucket
            .iter_mut()
            .find(|cached| cached.input.semantic_eq(input.program.as_ref()))
        {
            cached.output = output;
        } else {
            bucket.push(CachedSemanticAdTransform {
                input: Arc::clone(&input.program),
                output,
            });
        }
        self.put_entry(cache_key, AdTransformCacheEntry::Semantic(bucket));
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
            self.stats.evictions = self.stats.evictions.saturating_add(1);
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
    Semantic(SemanticAdTransformCacheKey),
}

enum AdTransformCacheEntry {
    Semantic(Vec<CachedSemanticAdTransform>),
}

impl fmt::Debug for AdTransformCacheEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Semantic(bucket) => f
                .debug_tuple("Semantic")
                .field(&format_args!("{} entries", bucket.len()))
                .finish(),
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
        AdTransformCacheKey::Semantic(key) => {
            key.input_metadata.len() * size_of::<ProgramValueMetadata>()
                + key.active_inputs.len() * size_of::<bool>()
                + key.active_outputs.len() * size_of::<bool>()
        }
    }
}

fn ad_transform_cache_value_retained_bytes(entry: &AdTransformCacheEntry) -> usize {
    match entry {
        AdTransformCacheEntry::Semantic(bucket) => {
            size_of_val(bucket.as_slice())
                + bucket
                    .iter()
                    .map(|cached| {
                        size_of::<CachedSemanticAdTransform>()
                            + semantic_program_retained_bytes(cached.input.as_ref())
                            + semantic_program_retained_bytes(
                                cached.output.frozen().program.as_ref(),
                            )
                            + size_of_val(cached.output.derivative_input_indices())
                            + size_of_val(cached.output.derivative_output_indices())
                    })
                    .fold(0usize, usize::saturating_add)
        }
    }
}

fn semantic_program_retained_bytes(program: &SemanticProgram) -> usize {
    size_of::<SemanticProgram>()
        + size_of_val(program.inputs())
        + size_of_val(program.outputs())
        + program.operations().len() * size_of::<usize>()
        + program.shape_guards().len() * size_of::<usize>()
}
