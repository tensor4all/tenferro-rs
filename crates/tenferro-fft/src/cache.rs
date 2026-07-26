use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use num_traits::{Float, FromPrimitive};
use rustfft::{Fft, FftNum, FftPlanner};
use tenferro_runtime::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
use tenferro_tensor::{CacheStats, RuntimeCacheControl};

use crate::FFT_EXTENSION_FAMILY_ID;

/// Runtime cache namespace used for private RustFFT plans.
pub const FFT_PLAN_CACHE_NAME: &str = "rustfft-plans";

/// Default number of typed entries retained by a caller-owned [`FftPlanCache`].
pub const DEFAULT_FFT_PLAN_CACHE_CAPACITY: usize = 64;

/// Select the private CPU RustFFT plan entries in an extension runtime cache.
///
/// Other [`crate::FftBackend`] implementations use distinct cache names in the
/// same FFT extension family.
///
/// # Examples
///
/// ```
/// let selector = tenferro_fft::fft_plan_cache_selector();
/// assert!(matches!(
///     selector,
///     tenferro_runtime::ExtensionCacheSelector::Cache { .. }
/// ));
/// ```
pub const fn fft_plan_cache_selector() -> ExtensionCacheSelector {
    ExtensionCacheSelector::Cache {
        family_id: FFT_EXTENSION_FAMILY_ID,
        cache_name: FFT_PLAN_CACHE_NAME,
    }
}

/// Bounded, caller-owned typed cache for backend FFT plans and workspaces.
///
/// [`crate::FftExecutor`] owns one cache and passes its store to every
/// [`crate::FftBackend`] through [`crate::FftExecutionCache`]. Backends retain
/// private `Send + Sync + 'static` values under their own
/// [`ExtensionCacheKey`] namespace. Entry limits, LRU eviction, clearing,
/// aggregate entry counts, and retained-byte accounting apply uniformly to CPU
/// and non-CPU entries.
///
/// The CPU backend stores RustFFT plans in the private `rustfft-plans`
/// namespace. Its retained-byte estimate includes the exact plan key and the
/// cache-owned `Arc` handle; allocations opaque to RustFFT are excluded.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use tenferro_fft::FftPlanCache;
///
/// let cache = FftPlanCache::with_capacity(NonZeroUsize::new(2).unwrap());
/// assert_eq!(cache.capacity().get(), 2);
/// assert_eq!(cache.stats().entries, 0);
/// ```
pub struct FftPlanCache {
    store: ExtensionCacheStore,
}

impl fmt::Debug for FftPlanCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FftPlanCache")
            .field("capacity", &self.capacity())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl FftPlanCache {
    /// Create an empty typed cache with an explicit maximum entry count.
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            store: ExtensionCacheStore::with_limits(ExtensionCacheLimits::new(capacity)),
        }
    }

    /// Maximum number of retained backend entries across all namespaces.
    pub fn capacity(&self) -> NonZeroUsize {
        self.store.limits().max_entries()
    }

    /// Return complete cache retention limits.
    pub fn limits(&self) -> ExtensionCacheLimits {
        self.store.limits()
    }

    /// Replace complete cache retention limits.
    pub fn set_limits(&mut self, limits: ExtensionCacheLimits) {
        self.store.set_limits(limits);
    }

    /// Resize the cache, evicting least-recently-used entries when necessary.
    pub fn set_capacity(&mut self, capacity: NonZeroUsize) {
        let mut limits = ExtensionCacheLimits::new(capacity);
        if let Some(max_retained_bytes) = self.store.limits().max_retained_bytes() {
            limits = limits.with_max_retained_bytes(max_retained_bytes);
        }
        self.store.set_limits(limits);
    }

    /// Remove every retained backend plan or workspace.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Snapshot aggregate entries and backend-reported retained bytes.
    pub fn stats(&self) -> CacheStats {
        self.store.stats(ExtensionCacheSelector::All)
    }

    pub(crate) fn store_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.store
    }

    pub(crate) fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        ExtensionFftPlanCache::new(&mut self.store).plan_f32(len, forward)
    }

    pub(crate) fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        ExtensionFftPlanCache::new(&mut self.store).plan_f64(len, forward)
    }

    #[cfg(test)]
    pub(crate) fn contains_f64(&mut self, len: usize, forward: bool) -> bool {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        };
        self.store
            .get::<ExtensionFftPlanEntry>(&extension_plan_key(key))
            .is_some_and(|entry| entry.matches_f64(key))
    }
}

impl Default for FftPlanCache {
    fn default() -> Self {
        Self::with_capacity(
            NonZeroUsize::new(DEFAULT_FFT_PLAN_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
        )
    }
}

impl RuntimeCacheControl for FftPlanCache {
    fn clear(&mut self) {
        Self::clear(self);
    }

    fn stats(&self) -> CacheStats {
        Self::stats(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FftPlanDType {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FftPlanKey {
    len: usize,
    forward: bool,
    dtype: FftPlanDType,
}

enum CachedFftPlan {
    F32(Arc<dyn Fft<f32>>),
    F64(Arc<dyn Fft<f64>>),
}

struct ExtensionFftPlanEntry {
    key: FftPlanKey,
    plan: CachedFftPlan,
}

impl ExtensionFftPlanEntry {
    #[cfg(test)]
    fn matches_f64(&self, key: FftPlanKey) -> bool {
        self.key == key && matches!(self.plan, CachedFftPlan::F64(_))
    }
}

pub(crate) trait FftPlanProvider: Send {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>>;
    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>>;
}

impl FftPlanProvider for FftPlanCache {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        Self::plan_f32(self, len, forward)
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        Self::plan_f64(self, len, forward)
    }
}

pub(crate) trait CachedFftPlanScalar: FftNum + Float + FromPrimitive + 'static {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>>;
}

impl CachedFftPlanScalar for f32 {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>> {
        plans.plan_f32(len, forward)
    }
}

impl CachedFftPlanScalar for f64 {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>> {
        plans.plan_f64(len, forward)
    }
}

pub(crate) fn cached_fft_plan<T: CachedFftPlanScalar, P: FftPlanProvider + ?Sized>(
    plans: &mut P,
    len: usize,
    forward: bool,
) -> Arc<dyn Fft<T>> {
    T::plan(plans, len, forward)
}

pub(crate) struct ExtensionFftPlanCache<'a> {
    entries: &'a mut ExtensionCacheStore,
}

impl<'a> ExtensionFftPlanCache<'a> {
    pub(crate) fn new(entries: &'a mut ExtensionCacheStore) -> Self {
        Self { entries }
    }
}

fn extension_plan_key(key: FftPlanKey) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    ExtensionCacheKey::new(
        FFT_EXTENSION_FAMILY_ID,
        FFT_PLAN_CACHE_NAME,
        hasher.finish(),
    )
}

impl FftPlanProvider for ExtensionFftPlanCache<'_> {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F32,
        };
        let cache_key = extension_plan_key(key);
        if let Some(cached) = self.entries.get::<ExtensionFftPlanEntry>(&cache_key) {
            if cached.key == key {
                if let CachedFftPlan::F32(plan) = &cached.plan {
                    return Arc::clone(plan);
                }
            }
        }
        let plan = build_fft_plan::<f32>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionFftPlanEntry {
                key,
                plan: CachedFftPlan::F32(Arc::clone(&plan)),
            },
            fft_plan_retained_bytes(),
        );
        plan
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        };
        let cache_key = extension_plan_key(key);
        if let Some(cached) = self.entries.get::<ExtensionFftPlanEntry>(&cache_key) {
            if cached.key == key {
                if let CachedFftPlan::F64(plan) = &cached.plan {
                    return Arc::clone(plan);
                }
            }
        }
        let plan = build_fft_plan::<f64>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionFftPlanEntry {
                key,
                plan: CachedFftPlan::F64(Arc::clone(&plan)),
            },
            fft_plan_retained_bytes(),
        );
        plan
    }
}

fn build_fft_plan<T: FftNum + 'static>(len: usize, forward: bool) -> Arc<dyn Fft<T>> {
    let mut planner = FftPlanner::<T>::new();
    if forward {
        planner.plan_fft_forward(len)
    } else {
        planner.plan_fft_inverse(len)
    }
}

const fn fft_plan_retained_bytes() -> usize {
    std::mem::size_of::<FftPlanKey>() + std::mem::size_of::<CachedFftPlan>()
}
