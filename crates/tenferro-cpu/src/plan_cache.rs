//! Shareable cache of immutable CPU numerical plans.
//!
//! A plan (a contraction plan, an indexed-traversal plan) is worked out once
//! and never changed afterwards, so any number of computations may read it at
//! the same time. [`CpuPlanCache`] holds those plans behind one lock that is
//! taken only to look a plan up or insert it; the computation runs on the
//! shared plan after the lock is released. Engines own a private cache by
//! default and may be given a shared one, so several engines plan once.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::gemm::GemmAnalysisCache;
use crate::indexed_plan_cache::{IndexedPlanCache, IndexedPlanCacheLimits, IndexedPlanKey};
use crate::CacheStats;
use strided_kernel::{
    ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedGatherPlan, ErasedScatterPlan,
};

/// A shareable, bounded cache of immutable CPU numerical plans.
///
/// Cloning the handle shares the cache. The lock inside is held only while a
/// plan is looked up or inserted, never while a computation runs on it.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuEngine, CpuPlanCache};
///
/// let plans = CpuPlanCache::new();
/// let first = CpuEngine::serial().with_plan_cache(plans.clone());
/// let second = CpuEngine::serial().with_plan_cache(plans.clone());
/// assert!(first.plan_cache().shares_with(second.plan_cache()));
/// assert_eq!(plans.stats().entries, 0);
/// ```
#[derive(Clone)]
pub struct CpuPlanCache {
    inner: Arc<Mutex<PlanCacheState>>,
}

pub(crate) struct PlanCacheState {
    pub(crate) gemm: GemmAnalysisCache,
    pub(crate) indexed: IndexedPlanCache,
}

impl fmt::Debug for CpuPlanCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        f.debug_struct("CpuPlanCache")
            .field("entries", &stats.entries)
            .field("retained_bytes", &stats.retained_bytes)
            .finish_non_exhaustive()
    }
}

impl Default for CpuPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuPlanCache {
    /// An empty cache with the default bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// assert_eq!(plans.stats().entries, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PlanCacheState {
                gemm: GemmAnalysisCache::default(),
                indexed: IndexedPlanCache::default(),
            })),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, PlanCacheState> {
        // INVARIANT: the lock guards only lookup and insertion of immutable
        // plans; a panic while it is held leaves complete entries only.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Whether `other` is a handle to the same cache.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// assert!(plans.shares_with(&plans.clone()));
    /// assert!(!plans.shares_with(&tenferro_cpu::CpuPlanCache::new()));
    /// ```
    pub fn shares_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Combined statistics of every plan family.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuPlanCache::new().stats().hits, 0);
    /// ```
    pub fn stats(&self) -> CacheStats {
        let state = self.lock();
        let gemm = tenferro_tensor::RuntimeCacheControl::stats(&state.gemm);
        let indexed = state.indexed.stats();
        CacheStats {
            entries: gemm.entries.saturating_add(indexed.entries),
            retained_bytes: gemm.retained_bytes.saturating_add(indexed.retained_bytes),
            hits: gemm.hits.saturating_add(indexed.hits),
            misses: gemm.misses.saturating_add(indexed.misses),
            evictions: gemm.evictions.saturating_add(indexed.evictions),
            clears: gemm.clears.saturating_add(indexed.clears),
        }
    }

    /// Statistics of the contraction-plan family alone.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuPlanCache::new().contraction_stats().entries, 0);
    /// ```
    pub fn contraction_stats(&self) -> CacheStats {
        tenferro_tensor::RuntimeCacheControl::stats(&self.lock().gemm)
    }

    /// Statistics of the indexed-traversal family alone.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuPlanCache::new().indexed_stats().entries, 0);
    /// ```
    pub fn indexed_stats(&self) -> CacheStats {
        self.lock().indexed.stats()
    }

    /// Drop every retained plan.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.clear();
    /// assert_eq!(plans.stats().entries, 0);
    /// ```
    pub fn clear(&self) {
        let mut state = self.lock();
        tenferro_tensor::RuntimeCacheControl::clear(&mut state.gemm);
        state.indexed.clear();
    }

    /// Drop every retained indexed-traversal plan.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.clear_indexed();
    /// assert_eq!(plans.indexed_stats().entries, 0);
    /// ```
    pub fn clear_indexed(&self) {
        self.lock().indexed.clear();
    }

    /// Maximum number of retained contraction plans.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.set_contraction_capacity(8);
    /// assert_eq!(plans.contraction_capacity(), 8);
    /// ```
    pub fn contraction_capacity(&self) -> usize {
        self.lock().gemm.capacity()
    }

    /// Bound the number of retained contraction plans.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.set_contraction_capacity(0);
    /// assert_eq!(plans.contraction_capacity(), 0);
    /// ```
    pub fn set_contraction_capacity(&self, max_plans: usize) {
        self.lock().gemm.set_capacity(max_plans);
    }

    /// The byte bound on the contraction plans' retained memory.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// assert_eq!(plans.contraction_limit_bytes(), 1 << 30);
    /// ```
    pub fn contraction_limit_bytes(&self) -> usize {
        self.lock().gemm.limit_bytes()
    }

    /// Bound the bytes the cached contraction plans retain; zero retains none.
    ///
    /// A cached plan owns tcontract's scatter tables, one `i64` per index of
    /// each role group of the packed strategy. They scale with the operands'
    /// free extents, not with the rank: an ordinary GEMM plan is a few KiB, a
    /// two-qubit gate on a 26-qubit complex64 state holds 384 MiB. The bytes
    /// counted are [`CpuPlanCache::stats`]'s `retained_bytes` for this family
    /// ([`CpuPlanCache::contraction_stats`]), and the default is 1 GiB.
    ///
    /// When storing a plan takes the total over the bound, the least recently
    /// used plans (a hit or a store counts as use) are evicted until it fits,
    /// and each eviction is counted in `evictions`. A plan that alone exceeds
    /// the bound is used for its call and not kept, so the next call rebuilds
    /// its tables. Recency is the order because a repeated workload re-reads
    /// what it used last; a loop over more plans than fit rebuilds every one of
    /// them, so raise the bound to the loop's working set. An evicted plan's
    /// memory is freed when its last holder drops it, which is immediately
    /// unless a contraction is still running on it. Plans that stay under the
    /// bound behave as before.
    ///
    /// Neither [`CpuEngine::reset_buffer_pool`](crate::CpuEngine::reset_buffer_pool)
    /// nor dropping an engine that shares this cache releases cached plans;
    /// [`CpuPlanCache::clear`] does.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.set_contraction_limit_bytes(1 << 20);
    /// assert_eq!(plans.contraction_limit_bytes(), 1 << 20);
    /// ```
    pub fn set_contraction_limit_bytes(&self, max_retained_bytes: usize) {
        self.lock().gemm.set_limit_bytes(max_retained_bytes);
    }

    /// Bounds of the indexed-traversal family.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// let limits = plans.indexed_limits();
    /// plans.set_indexed_limits(limits);
    /// assert_eq!(plans.indexed_limits(), limits);
    /// ```
    pub fn indexed_limits(&self) -> IndexedPlanCacheLimits {
        self.lock().indexed.limits()
    }

    /// Replace the bounds of the indexed-traversal family, evicting as needed.
    ///
    /// # Examples
    ///
    /// ```
    /// let plans = tenferro_cpu::CpuPlanCache::new();
    /// plans.set_indexed_limits(plans.indexed_limits());
    /// ```
    pub fn set_indexed_limits(&self, limits: IndexedPlanCacheLimits) {
        self.lock().indexed.set_limits(limits);
    }

    pub(crate) fn gather<E>(
        &self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedGatherPlan, E>,
    ) -> Result<Arc<ErasedGatherPlan>, E> {
        self.lock().indexed.gather(key, compile)
    }

    pub(crate) fn scatter<E>(
        &self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedScatterPlan, E>,
    ) -> Result<Arc<ErasedScatterPlan>, E> {
        self.lock().indexed.scatter(key, compile)
    }

    pub(crate) fn dynamic_slice<E>(
        &self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedDynamicSlicePlan, E>,
    ) -> Result<Arc<ErasedDynamicSlicePlan>, E> {
        self.lock().indexed.dynamic_slice(key, compile)
    }

    pub(crate) fn dynamic_update_slice<E>(
        &self,
        key: IndexedPlanKey,
        compile: impl FnOnce() -> Result<ErasedDynamicUpdateSlicePlan, E>,
    ) -> Result<Arc<ErasedDynamicUpdateSlicePlan>, E> {
        self.lock().indexed.dynamic_update_slice(key, compile)
    }
}
