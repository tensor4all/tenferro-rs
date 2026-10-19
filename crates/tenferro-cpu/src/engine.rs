//! The CPU execution engine: a caller-held value that owns one Rayon pool (or
//! none) and the workspace every operation on it writes.
//!
//! tenferro never builds threads. The caller builds the pool, usually from
//! [`thread_pool_builder`], and hands it to [`CpuEngine::new`]; a
//! [`CpuEngine::serial`] engine has no pool at all. Every operation takes the
//! engine through `&mut`, so one computation uses it at a time without run-time
//! locks. An operation starts on the calling thread and enters the pool only
//! after a provider has chosen more than one worker for it.

use std::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_tensor::{AllocationDomainId, SharedTensorAllocationDomain};

use crate::buffer_pool::{BufferPool, BufferPoolStats};
use crate::contraction::ExecutionResources;
use crate::provider::CpuExecutionContext;
use crate::{CacheStats, CpuPlanCache};

/// Stack size the [`thread_pool_builder`] preset reserves for every worker.
///
/// Rust's `std::thread` default is 2 MiB, but provider code can recurse with
/// large private frames: a `NUM_THREADS=64` OpenBLAS build keeps a
/// `job_t job[64]` (about 541 KiB per level, measured) on the stack of every
/// recursive `dgetrf_parallel` frame, so 2 MiB allowed only three levels and
/// aborted with a stack overflow at n>=256. Sixteen MiB leaves room for about
/// thirty such frames, which is the order of the 8 MiB main-thread default that
/// provider calls receive outside a pool.
///
/// # Examples
///
/// ```
/// assert_eq!(tenferro_cpu::DEFAULT_WORKER_STACK_BYTES, 16 << 20);
/// ```
pub const DEFAULT_WORKER_STACK_BYTES: usize = 16 << 20;

/// Smallest worker stack a caller should configure for a pool that runs
/// provider code.
///
/// # Examples
///
/// ```
/// assert!(tenferro_cpu::MIN_WORKER_STACK_BYTES < tenferro_cpu::DEFAULT_WORKER_STACK_BYTES);
/// ```
pub const MIN_WORKER_STACK_BYTES: usize = 64 << 10;

/// A `rayon::ThreadPoolBuilder` preset with the worker stack size the CPU
/// providers need ([`DEFAULT_WORKER_STACK_BYTES`]).
///
/// This is a convenience, not a construction mode: the caller sets the thread
/// count, pins workers if it wants to, builds the pool and hands it to
/// [`CpuEngine::new`].
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{thread_pool_builder, CpuEngine};
///
/// let pool = thread_pool_builder().num_threads(2).build()?;
/// let engine = CpuEngine::new(pool);
/// assert_eq!(engine.num_threads(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn thread_pool_builder() -> rayon::ThreadPoolBuilder {
    rayon::ThreadPoolBuilder::new().stack_size(DEFAULT_WORKER_STACK_BYTES)
}

/// Typed refusals reported by a [`CpuEngine`] before an operation runs.
///
/// They reach callers as the source of a [`crate::Error::RuntimeState`].
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngineError;
///
/// assert!(CpuEngineError::CallerInOwnPool.to_string().contains("worker"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CpuEngineError {
    /// The operation was called from a worker of the engine's own pool.
    ///
    /// Some provider strategies cannot start on the pool's own worker (a
    /// team that needs every worker at once is refused), and the others would
    /// run with less parallelism than the engine promises. tenferro rejects
    /// the call instead of letting the outcome depend on the provider.
    #[error(
        "a CPU engine operation was called from a worker of the engine's own pool; \
         call it from a thread outside that pool"
    )]
    CallerInOwnPool,
    /// An earlier operation panicked while it was using the engine.
    ///
    /// The engine's workspace may be half updated, so later operations are
    /// refused; replace the engine with a new one.
    #[error("the CPU engine is broken by an earlier panic; build a new engine")]
    Broken,
}

/// A caller-held CPU execution engine.
///
/// The engine owns one Rayon pool (or none for [`CpuEngine::serial`]) and the
/// workspace operations write: the buffer pool that recycles outputs, the
/// contraction scratch, and a handle to a [`CpuPlanCache`]. It implements
/// [`tenferro_tensor::Engine`], so every operation takes it as `&mut`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngine;
/// use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
///
/// let mut engine = CpuEngine::serial();
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
/// let c = engine.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))?;
/// assert_eq!(c.as_slice::<f64>()?, &[4.0, 6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct CpuEngine {
    pub(crate) resources: ExecutionResources,
    pub(crate) buffers: BufferPool,
    pub(crate) plans: CpuPlanCache,
    pub(crate) batch_policy: crate::CpuBatchPolicy,
    pub(crate) allocation_domain: Option<Arc<dyn SharedTensorAllocationDomain>>,
    runtime_clears: u64,
    pub(crate) broken: bool,
}

impl std::fmt::Debug for CpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuEngine")
            .field("num_threads", &self.num_threads())
            .field("batch_policy", &self.batch_policy)
            .field("allocation_domain", &self.allocation_domain())
            .field("broken", &self.broken)
            .finish_non_exhaustive()
    }
}

impl CpuEngine {
    fn with_resources(resources: ExecutionResources) -> Self {
        Self {
            resources,
            buffers: BufferPool::new(),
            plans: CpuPlanCache::new(),
            batch_policy: crate::CpuBatchPolicy::default(),
            allocation_domain: None,
            runtime_clears: 0,
            broken: false,
        }
    }

    /// An engine that computes on `pool`, which it takes over.
    ///
    /// Every operation may use all workers of the pool; a caller that wants
    /// fewer threads builds a smaller pool. Build the pool with
    /// [`thread_pool_builder`] to get the worker stack the providers need.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = tenferro_cpu::thread_pool_builder().num_threads(4).build()?;
    /// let engine = tenferro_cpu::CpuEngine::new(pool);
    /// assert_eq!(engine.num_threads(), 4);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(pool: rayon::ThreadPool) -> Self {
        Self::with_resources(ExecutionResources::new(Some(pool)))
    }

    /// An engine with no pool: every operation runs on the calling thread.
    ///
    /// A serial engine never touches Rayon's global pool, so it is safe to use
    /// from inside any other pool.
    ///
    /// # Examples
    ///
    /// ```
    /// let engine = tenferro_cpu::CpuEngine::serial();
    /// assert!(engine.pool().is_none());
    /// assert_eq!(engine.num_threads(), 1);
    /// ```
    pub fn serial() -> Self {
        Self::with_resources(ExecutionResources::new(None))
    }

    /// Use `plans` as this engine's plan cache, shared with every other engine
    /// that holds a clone of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuEngine, CpuPlanCache};
    ///
    /// let plans = CpuPlanCache::new();
    /// let engine = CpuEngine::serial().with_plan_cache(plans.clone());
    /// assert!(engine.plan_cache().shares_with(&plans));
    /// ```
    #[must_use]
    pub fn with_plan_cache(mut self, plans: CpuPlanCache) -> Self {
        self.plans = plans;
        self
    }

    /// The plan cache this engine reads plans from.
    ///
    /// # Examples
    ///
    /// ```
    /// let engine = tenferro_cpu::CpuEngine::serial();
    /// assert_eq!(engine.plan_cache().stats().entries, 0);
    /// ```
    pub fn plan_cache(&self) -> &CpuPlanCache {
        &self.plans
    }

    /// Bind outputs and managed inputs to a shared allocation domain.
    ///
    /// # Examples
    ///
    /// ```
    /// let engine = tenferro_cpu::CpuEngine::serial();
    /// assert!(engine.allocation_domain().is_none());
    /// ```
    #[must_use]
    pub fn with_allocation_domain(mut self, domain: Arc<dyn SharedTensorAllocationDomain>) -> Self {
        self.allocation_domain = Some(domain);
        self
    }

    /// The shared allocation domain this engine is bound to, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().allocation_domain(), None);
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain.as_ref().map(|domain| domain.id())
    }

    /// The shared allocator this engine is bound to, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(tenferro_cpu::CpuEngine::serial().shared_allocation_domain().is_none());
    /// ```
    pub fn shared_allocation_domain(&self) -> Option<&Arc<dyn SharedTensorAllocationDomain>> {
        self.allocation_domain.as_ref()
    }

    /// This engine with a different default batch policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy, CpuEngine};
    ///
    /// let engine = CpuEngine::serial()
    ///     .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));
    /// assert_eq!(engine.batch_policy().strategy, CpuBatchStrategy::Sequential);
    /// ```
    #[must_use]
    pub fn with_batch_policy(mut self, policy: crate::CpuBatchPolicy) -> Self {
        self.batch_policy = policy;
        self
    }

    /// Replace the default batch policy, returning the previous one.
    ///
    /// For a one-operation override use [`crate::with_batch_policy`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy, CpuEngine};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let previous = engine.set_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    /// assert_eq!(previous.strategy, CpuBatchStrategy::Auto);
    /// ```
    pub fn set_batch_policy(&mut self, policy: crate::CpuBatchPolicy) -> crate::CpuBatchPolicy {
        std::mem::replace(&mut self.batch_policy, policy)
    }

    /// The default batch policy of this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBatchStrategy, CpuEngine};
    /// assert_eq!(CpuEngine::serial().batch_policy().strategy, CpuBatchStrategy::Auto);
    /// ```
    #[must_use]
    pub fn batch_policy(&self) -> crate::CpuBatchPolicy {
        self.batch_policy
    }

    /// The pool this engine computes on, or `None` for a serial engine.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = tenferro_cpu::thread_pool_builder().num_threads(2).build()?;
    /// let engine = tenferro_cpu::CpuEngine::new(pool);
    /// assert_eq!(engine.pool().map(|pool| pool.current_num_threads()), Some(2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pool(&self) -> Option<&rayon::ThreadPool> {
        self.resources.rayon_pool()
    }

    /// The number of workers operations may use: the pool size, or one.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().num_threads(), 1);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.thread_budget().get()
    }

    pub(crate) fn thread_budget(&self) -> NonZeroUsize {
        self.resources.thread_budget()
    }

    /// Whether an earlier operation broke this engine (see
    /// [`CpuEngineError::Broken`]).
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(!tenferro_cpu::CpuEngine::serial().is_broken());
    /// ```
    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// The provider execution token for this engine: its pool, its budget and
    /// its batch policy.
    ///
    /// A downstream kernel that calls a provider (cpueinsum, tcontract,
    /// tlinalg) directly passes this token, so it uses the same threads, budget
    /// and batch policy as tenferro. Getting the token applies the same checks
    /// as an operation; what the kernel then does with the (copyable) token is
    /// that kernel's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] with a [`CpuEngineError`] source
    /// when called from a worker of the engine's own pool or when the engine is
    /// broken.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let exec = engine.exec()?;
    /// assert_eq!(exec.budget(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn exec(&mut self) -> crate::Result<cpueinsum::Exec<'_>> {
        self.check_entry("CpuEngine::exec")?;
        CpuExecutionContext::new(&self.resources, self.batch_policy).contraction_exec()
    }

    /// Borrow the provider token, the buffer pool and the N-ary contraction
    /// scratch for one downstream call.
    ///
    /// # Errors
    ///
    /// Returns the entry refusals of [`CpuEngine::exec`], or `op`'s own error.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let len = engine.with_contraction_exec(|_, buffers, workspaces| {
    ///     workspaces.with_scratch::<f64, _>(4, buffers.max_retained_capacity_bytes(), |scratch| {
    ///         Ok(scratch.len())
    ///     })
    /// })?;
    /// assert_eq!(len, 4);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[doc(hidden)]
    pub fn with_contraction_exec<R>(
        &mut self,
        op: impl FnOnce(
            &cpueinsum::Exec<'_>,
            &mut BufferPool,
            &crate::ContractionWorkspaces,
        ) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.check_entry("CpuEngine::with_contraction_exec")?;
        let context = CpuExecutionContext::new(&self.resources, self.batch_policy);
        let _retention = self
            .resources
            .retention_guard(self.buffers.max_retained_capacity_bytes());
        let exec = context.contraction_exec()?;
        let nary = &self.resources.nary;
        guard_unwind(&mut self.buffers, &mut self.broken, |buffers| {
            op(&exec, buffers, nary)
        })
    }

    /// Borrow the execution context and buffer pool for a linalg provider call.
    ///
    /// The provider chooses its own execution through the context; this
    /// callback itself runs on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns the entry refusals of [`CpuEngine::exec`], or `op`'s own error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBatchStrategy, CpuEngine};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let strategy = engine.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy))?;
    /// assert_eq!(strategy, CpuBatchStrategy::Auto);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[doc(hidden)]
    pub fn with_linalg_pool<R>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.check_entry("CpuEngine::with_linalg_pool")?;
        let context = CpuExecutionContext::new(&self.resources, self.batch_policy);
        guard_unwind(&mut self.buffers, &mut self.broken, |buffers| {
            op(&context, buffers)
        })
    }

    /// Refuse an operation that cannot run on this engine now.
    pub(crate) fn check_entry(&self, op: &'static str) -> crate::Result<()> {
        if self.broken {
            return Err(crate::Error::runtime_state_source(
                op,
                CpuEngineError::Broken,
            ));
        }
        if self
            .resources
            .rayon_pool()
            .is_some_and(|pool| pool.current_thread_index().is_some())
        {
            return Err(crate::Error::runtime_state_source(
                op,
                CpuEngineError::CallerInOwnPool,
            ));
        }
        Ok(())
    }

    /// Number of retained buffers in the engine's buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.buffers.len()
    }

    /// Retained-buffer statistics of the engine's buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().buffer_pool_stats().buffers, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffers.stats()
    }

    /// The buffer pool as cache statistics (entries and retained bytes).
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().buffer_pool_cache_stats().entries, 0);
    /// ```
    pub fn buffer_pool_cache_stats(&self) -> CacheStats {
        let stats = self.buffers.stats();
        CacheStats {
            entries: stats.buffers,
            retained_bytes: stats.capacity_bytes,
            ..CacheStats::default()
        }
    }

    /// The retention limit of recycled buffers, in retained capacity bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// engine.set_buffer_pool_limit_bytes(0);
    /// assert_eq!(engine.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn buffer_pool_limit_bytes(&self) -> usize {
        self.buffers.max_retained_capacity_bytes()
    }

    /// Bound the bytes of recycled buffers and contraction scratch retained
    /// between operations; zero disables retention.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// engine.set_buffer_pool_limit_bytes(1 << 20);
    /// assert_eq!(engine.buffer_pool_limit_bytes(), 1 << 20);
    /// ```
    pub fn set_buffer_pool_limit_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.buffers
            .set_max_retained_capacity_bytes(max_retained_capacity_bytes);
        self.resources.nary.trim_to(max_retained_capacity_bytes);
        if self.resources.retained_bytes() > max_retained_capacity_bytes {
            self.resources.trim();
        }
    }

    /// Drop every retained buffer and contraction scratch.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// engine.reset_buffer_pool();
    /// assert_eq!(engine.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        self.buffers.clear();
        self.resources.nary.clear_all();
        self.resources.trim();
    }

    /// Aggregate statistics of everything this engine retains between
    /// operations: buffers, contraction scratch and its plan cache.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tenferro_cpu::CpuEngine::serial().runtime_cache_stats().entries, 0);
    /// ```
    #[doc(hidden)]
    pub fn runtime_cache_stats(&self) -> CacheStats {
        let (workspace_entries, nary_bytes) = self.resources.nary.total_stats();
        let workspace_bytes = nary_bytes.saturating_add(self.resources.retained_bytes());
        let buffers = self.buffers.cache_stats();
        let plans = self.plans.stats();
        CacheStats {
            entries: buffers
                .entries
                .saturating_add(plans.entries)
                .saturating_add(workspace_entries),
            retained_bytes: buffers
                .retained_bytes
                .saturating_add(plans.retained_bytes)
                .saturating_add(workspace_bytes),
            hits: plans.hits,
            misses: plans.misses,
            evictions: plans.evictions,
            clears: self.runtime_clears,
        }
    }

    /// Drop everything this engine retains between operations, including its
    /// plan cache (and therefore the plans of every engine sharing it).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// engine.clear_runtime_caches();
    /// assert_eq!(engine.runtime_cache_stats().clears, 1);
    /// ```
    #[doc(hidden)]
    pub fn clear_runtime_caches(&mut self) {
        self.reset_buffer_pool();
        self.plans.clear();
        self.runtime_clears = self.runtime_clears.saturating_add(1);
    }
}

/// Run one operation's use of the buffer pool, recovering the pool's
/// accounting and marking the engine broken if it unwinds.
///
/// Buffers checked out during the operation are counted as in flight. A normal
/// exit forgets that accounting (the buffers left as outputs); an unwind
/// replenishes the pool with replacements for the buffers it lost and marks the
/// engine broken, since other workspace may be half updated.
pub(crate) fn guard_unwind<R>(
    buffers: &mut BufferPool,
    broken: &mut bool,
    op: impl FnOnce(&mut BufferPool) -> R,
) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(&mut *buffers))) {
        Ok(value) => {
            buffers.clear_in_flight_retained();
            value
        }
        Err(payload) => {
            buffers.replenish_in_flight_retained();
            *broken = true;
            std::panic::resume_unwind(payload)
        }
    }
}

impl CpuEngine {
    /// Split borrows for the operation helpers: the execution context (which
    /// borrows only the pool resources), the buffer pool, the plan cache and
    /// the broken flag.
    pub(crate) fn parts(
        &mut self,
    ) -> (
        CpuExecutionContext<'_>,
        &mut BufferPool,
        &CpuPlanCache,
        &mut bool,
    ) {
        (
            CpuExecutionContext::new(&self.resources, self.batch_policy),
            &mut self.buffers,
            &self.plans,
            &mut self.broken,
        )
    }
}

#[cfg(test)]
mod tests;
