use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use lru::LruCache;
use smallvec::{Array, SmallVec};
use strided_kernel::{
    ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedGatherPlan, ErasedScatterPlan,
    KernelDType,
};
use tenferro_tensor::CacheStats;

/// Limits for the CPU indexed-plan cache.
///
/// A zero entry or byte limit disables retention. Limits apply independently
/// to each initialized CPU execution engine.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::IndexedPlanCacheLimits;
///
/// let limits = IndexedPlanCacheLimits::new(32, 1024 * 1024);
/// assert_eq!(limits.max_entries(), 32);
/// assert_eq!(limits.max_retained_bytes(), 1024 * 1024);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedPlanCacheLimits {
    max_entries: usize,
    max_retained_bytes: usize,
}

impl IndexedPlanCacheLimits {
    /// Construct indexed-plan cache limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::IndexedPlanCacheLimits;
    ///
    /// let limits = IndexedPlanCacheLimits::new(8, 4096);
    /// assert_eq!(limits.max_entries(), 8);
    /// ```
    pub const fn new(max_entries: usize, max_retained_bytes: usize) -> Self {
        Self {
            max_entries,
            max_retained_bytes,
        }
    }

    /// Maximum retained plan count per CPU execution engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::IndexedPlanCacheLimits;
    ///
    /// assert_eq!(IndexedPlanCacheLimits::new(8, 4096).max_entries(), 8);
    /// ```
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Maximum logical retained bytes per CPU execution engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::IndexedPlanCacheLimits;
    ///
    /// assert_eq!(
    ///     IndexedPlanCacheLimits::new(8, 4096).max_retained_bytes(),
    ///     4096
    /// );
    /// ```
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }
}

pub(crate) const DEFAULT_INDEXED_PLAN_CACHE_LIMITS: IndexedPlanCacheLimits =
    IndexedPlanCacheLimits::new(256, 8 * 1024 * 1024);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum IndexedPlanFamily {
    Gather,
    Scatter,
    DynamicSlice,
    DynamicUpdateSlice,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct IndexedPlanKey {
    family: IndexedPlanFamily,
    dtype: KernelDType,
    index_dtype: KernelDType,
    dims: SmallVec<[SmallVec<[usize; 8]>; 4]>,
    strides: SmallVec<[SmallVec<[isize; 8]>; 4]>,
    config: SmallVec<[SmallVec<[usize; 8]>; 5]>,
}

impl IndexedPlanKey {
    pub(crate) fn from_slices(
        family: IndexedPlanFamily,
        dtype: KernelDType,
        index_dtype: KernelDType,
        dims: &[&[usize]],
        strides: &[&[isize]],
        config: &[&[usize]],
    ) -> Self {
        Self {
            family,
            dtype,
            index_dtype,
            dims: dims
                .iter()
                .map(|values| values.iter().copied().collect())
                .collect(),
            strides: strides
                .iter()
                .map(|values| values.iter().copied().collect())
                .collect(),
            config: config
                .iter()
                .map(|values| values.iter().copied().collect())
                .collect(),
        }
    }

    fn retained_bytes(&self) -> usize {
        nested_smallvec_retained_bytes(&self.dims)
            .saturating_add(nested_smallvec_retained_bytes(&self.strides))
            .saturating_add(nested_smallvec_retained_bytes(&self.config))
    }

    fn logical_payload_bytes(&self) -> usize {
        nested_smallvec_logical_bytes(&self.dims)
            .saturating_add(nested_smallvec_logical_bytes(&self.strides))
            .saturating_add(nested_smallvec_logical_bytes(&self.config))
    }
}

fn smallvec_retained_bytes<A: Array>(values: &SmallVec<A>) -> usize {
    if values.spilled() {
        values.capacity().saturating_mul(size_of::<A::Item>())
    } else {
        0
    }
}

fn nested_smallvec_retained_bytes<A, B>(values: &SmallVec<A>) -> usize
where
    A: Array<Item = SmallVec<B>>,
    B: Array,
{
    smallvec_retained_bytes(values).saturating_add(
        values
            .iter()
            .map(smallvec_retained_bytes)
            .fold(0usize, usize::saturating_add),
    )
}

fn nested_smallvec_logical_bytes<A, B>(values: &SmallVec<A>) -> usize
where
    A: Array<Item = SmallVec<B>>,
    B: Array,
{
    values.iter().fold(0usize, |total, inner| {
        total.saturating_add(inner.len().saturating_mul(size_of::<B::Item>()))
    })
}

#[derive(Clone, Debug)]
enum IndexedPlan {
    Gather(Arc<ErasedGatherPlan>),
    Scatter(Arc<ErasedScatterPlan>),
    DynamicSlice(Arc<ErasedDynamicSlicePlan>),
    DynamicUpdateSlice(Arc<ErasedDynamicUpdateSlicePlan>),
}

#[derive(Debug)]
struct IndexedPlanCacheEntry {
    plan: IndexedPlan,
    retained_bytes: usize,
}

/// Engine-owned cache for compile-once indexed traversal plans.
///
/// Each [`crate::engine::EngineResources`] value owns one cache for the
/// lifetime of its CPU execution engine. The bounded default retains at most
/// 256 plans and 8 MiB of logical plan/key payload. Entries are keyed by the
/// complete family, dtype, index dtype, layout, and operation configuration,
/// and least-recently-used entries are evicted when either bound is exceeded.
///
/// [`crate::CpuBackend`] exposes cache-specific limits, clear, and aggregate
/// stats methods. Its [`tenferro_runtime::runtime::RuntimeCacheOwner`]
/// implementation also includes this cache in runtime aggregate stats/clear.
#[derive(Debug)]
pub(crate) struct IndexedPlanCache {
    entries: LruCache<IndexedPlanKey, IndexedPlanCacheEntry>,
    limits: IndexedPlanCacheLimits,
    retained_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}

impl Default for IndexedPlanCache {
    fn default() -> Self {
        Self::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS)
    }
}

impl IndexedPlanCache {
    pub(crate) fn new(limits: IndexedPlanCacheLimits) -> Self {
        Self {
            entries: LruCache::unbounded(),
            limits,
            retained_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            clears: 0,
        }
    }

    pub(crate) fn set_limits(&mut self, limits: IndexedPlanCacheLimits) {
        self.limits = limits;
        self.evict_to_limits();
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.retained_bytes = 0;
        self.clears = self.clears.saturating_add(1);
    }

    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.retained_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            clears: self.clears,
        }
    }

    pub(crate) fn gather<E>(
        &mut self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedGatherPlan, E>,
    ) -> Result<Arc<ErasedGatherPlan>, E> {
        if let Some(IndexedPlan::Gather(plan)) = self.lookup(&key) {
            return Ok(plan);
        }
        let plan = Arc::new(compile()?);
        self.insert(key, IndexedPlan::Gather(Arc::clone(&plan)));
        Ok(plan)
    }

    pub(crate) fn scatter<E>(
        &mut self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedScatterPlan, E>,
    ) -> Result<Arc<ErasedScatterPlan>, E> {
        if let Some(IndexedPlan::Scatter(plan)) = self.lookup(&key) {
            return Ok(plan);
        }
        let plan = Arc::new(compile()?);
        self.insert(key, IndexedPlan::Scatter(Arc::clone(&plan)));
        Ok(plan)
    }

    pub(crate) fn dynamic_slice<E>(
        &mut self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedDynamicSlicePlan, E>,
    ) -> Result<Arc<ErasedDynamicSlicePlan>, E> {
        if let Some(IndexedPlan::DynamicSlice(plan)) = self.lookup(&key) {
            return Ok(plan);
        }
        let plan = Arc::new(compile()?);
        self.insert(key, IndexedPlan::DynamicSlice(Arc::clone(&plan)));
        Ok(plan)
    }

    pub(crate) fn dynamic_update_slice<E>(
        &mut self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedDynamicUpdateSlicePlan, E>,
    ) -> Result<Arc<ErasedDynamicUpdateSlicePlan>, E> {
        if let Some(IndexedPlan::DynamicUpdateSlice(plan)) = self.lookup(&key) {
            return Ok(plan);
        }
        let plan = Arc::new(compile()?);
        self.insert(key, IndexedPlan::DynamicUpdateSlice(Arc::clone(&plan)));
        Ok(plan)
    }

    fn lookup(&mut self, key: &IndexedPlanKey) -> Option<IndexedPlan> {
        let Some(entry) = self.entries.get(key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.hits = self.hits.saturating_add(1);
        Some(entry.plan.clone())
    }

    fn insert(&mut self, key: IndexedPlanKey, plan: IndexedPlan) {
        if self.limits.max_entries == 0 || self.limits.max_retained_bytes == 0 {
            return;
        }
        let retained_bytes = indexed_plan_retained_bytes(&key, &plan);
        if retained_bytes > self.limits.max_retained_bytes {
            return;
        }
        let entry = IndexedPlanCacheEntry {
            plan,
            retained_bytes,
        };
        if let Some(replaced) = self.entries.put(key, entry) {
            self.retained_bytes = self.retained_bytes.saturating_sub(replaced.retained_bytes);
        }
        self.retained_bytes = self.retained_bytes.saturating_add(retained_bytes);
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.limits.max_entries
            || self.retained_bytes > self.limits.max_retained_bytes
        {
            let Some((_key, entry)) = self.entries.pop_lru() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(entry.retained_bytes);
            self.evictions = self.evictions.saturating_add(1);
        }
    }
}

fn indexed_plan_retained_bytes(key: &IndexedPlanKey, plan: &IndexedPlan) -> usize {
    let plan_header = match plan {
        IndexedPlan::Gather(plan) => size_of_val(plan.as_ref()),
        IndexedPlan::Scatter(plan) => size_of_val(plan.as_ref()),
        IndexedPlan::DynamicSlice(plan) => size_of_val(plan.as_ref()),
        IndexedPlan::DynamicUpdateSlice(plan) => size_of_val(plan.as_ref()),
    };
    // INVARIANT: erased indexed plans clone the key's layout/config vectors
    // and may derive additional axis vectors. Charge twice the logical key
    // payload in addition to any spilled key capacity so inline SmallVec keys
    // still account for the upstream plan's heap-owned Vec payload.
    size_of::<IndexedPlanCacheEntry>()
        .saturating_add(size_of::<IndexedPlanKey>())
        .saturating_add(plan_header)
        .saturating_add(key.retained_bytes())
        .saturating_add(key.logical_payload_bytes().saturating_mul(2))
}

#[cfg(test)]
mod tests;
