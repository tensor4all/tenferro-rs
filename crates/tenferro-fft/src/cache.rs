use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use num_traits::{Float, FromPrimitive};
use rustfft::{Fft, FftNum, FftPlanner};
use tenferro_runtime::{ExtensionCacheKey, ExtensionCacheSelector, ExtensionCacheStore};
use tenferro_tensor::{CacheStats, RuntimeCacheControl};

use crate::FFT_EXTENSION_FAMILY_ID;

/// Runtime cache namespace used for RustFFT plans.
pub const FFT_PLAN_CACHE_NAME: &str = "rustfft-plans";

/// Default number of plans retained by a caller-owned [`FftPlanCache`].
pub const DEFAULT_FFT_PLAN_CACHE_CAPACITY: usize = 64;

/// Select the FFT plan entries in an extension runtime cache.
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

/// Bounded, caller-owned LRU cache of RustFFT plans.
///
/// Retained-byte statistics include the cache-owned key and `Arc` handle for
/// each entry. RustFFT does not expose the allocations owned by an opaque plan,
/// so those allocations are intentionally excluded from the estimate.
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
    entries: LruCache<FftPlanKey, CachedFftPlan>,
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
    /// Create an empty plan cache with an explicit maximum entry count.
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            entries: LruCache::new(capacity),
        }
    }

    /// Maximum number of retained plans.
    pub fn capacity(&self) -> NonZeroUsize {
        self.entries.cap()
    }

    /// Resize the cache, evicting least-recently-used plans when necessary.
    pub fn set_capacity(&mut self, capacity: NonZeroUsize) {
        self.entries.resize(capacity);
    }

    /// Remove every retained plan.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Snapshot the number of plans and known cache-owned bytes retained.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.entries.len().saturating_mul(fft_plan_retained_bytes()),
        }
    }

    pub(crate) fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F32,
        };
        if let Some(CachedFftPlan::F32(plan)) = self.entries.get(&key) {
            return Arc::clone(plan);
        }
        let plan = build_fft_plan::<f32>(len, forward);
        self.entries.put(key, CachedFftPlan::F32(Arc::clone(&plan)));
        plan
    }

    pub(crate) fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        };
        if let Some(CachedFftPlan::F64(plan)) = self.entries.get(&key) {
            return Arc::clone(plan);
        }
        let plan = build_fft_plan::<f64>(len, forward);
        self.entries.put(key, CachedFftPlan::F64(Arc::clone(&plan)));
        plan
    }

    #[cfg(test)]
    pub(crate) fn contains_f64(&self, len: usize, forward: bool) -> bool {
        self.entries.contains(&FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        })
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

#[derive(Clone)]
struct ExtensionF32Plan {
    key: FftPlanKey,
    plan: Arc<dyn Fft<f32>>,
}

#[derive(Clone)]
struct ExtensionF64Plan {
    key: FftPlanKey,
    plan: Arc<dyn Fft<f64>>,
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
        if let Some(cached) = self.entries.get::<ExtensionF32Plan>(&cache_key) {
            if cached.key == key {
                return Arc::clone(&cached.plan);
            }
        }
        let plan = build_fft_plan::<f32>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionF32Plan {
                key,
                plan: Arc::clone(&plan),
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
        if let Some(cached) = self.entries.get::<ExtensionF64Plan>(&cache_key) {
            if cached.key == key {
                return Arc::clone(&cached.plan);
            }
        }
        let plan = build_fft_plan::<f64>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionF64Plan {
                key,
                plan: Arc::clone(&plan),
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
