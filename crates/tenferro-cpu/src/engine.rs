use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::buffer_pool::BufferPool;
use crate::gemm::GemmAnalysisCache;
use crate::indexed_plan_cache::IndexedPlanCache;
use crate::placement::ResolvedCpuPlacement;
use crate::resource_domain::CpuResourceDomain;
use crate::{CpuContext, CpuContextError, CpuDomainId, CpuSet};

#[derive(Debug)]
pub(crate) struct EngineResources {
    pub(crate) buffers: BufferPool,
    pub(crate) gemm_analysis_cache: GemmAnalysisCache,
    pub(crate) indexed_plan_cache: IndexedPlanCache,
    /// N-ary contraction scratch. A session that runs on the engine's shared
    /// resources keeps `None` and uses the context-wide store, exactly as
    /// before. A reentrant child execution gets its own store here, because the
    /// shared one is a single `try_lock`ed lease that assumes one active
    /// execution per owner: two concurrent children would otherwise race on it.
    pub(crate) nary: Option<crate::ContractionWorkspaces>,
    pub(crate) runtime_clears: u64,
}

impl EngineResources {
    pub(crate) fn new(buffer_limit: usize) -> Self {
        Self {
            buffers: BufferPool::with_max_retained_capacity_bytes(buffer_limit),
            gemm_analysis_cache: GemmAnalysisCache::default(),
            indexed_plan_cache: IndexedPlanCache::default(),
            nary: None,
            runtime_clears: 0,
        }
    }

    /// Resources for a reentrant child execution: same shape as [`Self::new`],
    /// but with a private N-ary scratch store.
    pub(crate) fn for_child_execution(buffer_limit: usize) -> Self {
        Self {
            nary: Some(crate::ContractionWorkspaces::default()),
            ..Self::new(buffer_limit)
        }
    }
}

/// One tenferro-owned CPU execution engine.
///
/// tenferro uses only pools it builds, so the engine always owns its context;
/// there is no borrowed or externally managed executor.
#[derive(Debug)]
pub(crate) struct CpuEngine {
    domain: CpuResourceDomain,
    pub(crate) context: Arc<CpuContext>,
    /// Reusable numerical resources, or the fact that a held session has them
    /// checked out; see [`EngineResourceCheckout`].
    pub(crate) resources: Mutex<ResourceSlot>,
}

/// One engine's reusable numerical resources, or the fact that a held session
/// has checked them out.
#[derive(Debug)]
pub(crate) enum ResourceSlot {
    /// The engine owns the resources and a session may check them out.
    ///
    /// Boxed so the slot stays small: the resources are moved as a whole, never
    /// per field, and the engine allocates this once.
    Ready(Box<EngineResources>),
    /// A held session owns the resources until it closes.
    CheckedOut,
}

impl ResourceSlot {
    /// Borrow the checked-out resources, or `None` while a held session owns them.
    pub(crate) fn ready_mut(&mut self) -> Option<&mut EngineResources> {
        match self {
            Self::Ready(resources) => Some(resources),
            Self::CheckedOut => None,
        }
    }
}

/// Owned checkout of one engine's reusable numerical resources.
///
/// A held CPU session owns this value instead of holding the engine's mutex for
/// its whole lifetime, so no shared plan-cache or buffer-pool lock spans an
/// operation, a kernel or a join (#1945 A2). Dropping the checkout returns the
/// resources; the return is unconditional and recovers a poisoned mutex exactly
/// as the scoped execution path does.
pub(crate) struct EngineResourceCheckout {
    engine: Arc<CpuEngine>,
    /// The engine's boxed resources, moved between the slot and this checkout so
    /// no session entry allocates.
    resources: Option<Box<EngineResources>>,
}

impl EngineResourceCheckout {
    /// Take the engine's resources, or `None` when another holder has them.
    ///
    /// The engine builds its resources eagerly, so a vacant slot always means
    /// "checked out", never "not created yet".
    pub(crate) fn take(engine: &Arc<CpuEngine>) -> Option<Self> {
        let mut slot = engine
            .resources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let resources = match std::mem::replace(&mut *slot, ResourceSlot::CheckedOut) {
            ResourceSlot::Ready(resources) => resources,
            vacant @ ResourceSlot::CheckedOut => {
                *slot = vacant;
                return None;
            }
        };
        drop(slot);
        Some(Self {
            engine: Arc::clone(engine),
            resources: Some(resources),
        })
    }

    /// Borrow the checked-out resources for one operation view.
    ///
    /// The checkout is released only by [`Self::return_resources`], which runs
    /// from the held session's `close` or `Drop`; no other `&mut` method of the
    /// session can observe it absent.
    pub(crate) fn resources_mut(&mut self) -> &mut EngineResources {
        match self.resources.as_deref_mut() {
            Some(resources) => resources,
            None => unreachable!("engine resources are released only by the held session"),
        }
    }

    fn return_resources(&mut self) {
        let Some(resources) = self.resources.take() else {
            return;
        };
        let mut slot = self
            .engine
            .resources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // INVARIANT: the arbiter grants one root execution per engine, and a
        // reentrant child owns private resources, so the slot cannot be refilled
        // while this checkout is live. Replacing instead of asserting keeps a
        // violated invariant from leaking resources without introducing a panic.
        debug_assert!(
            matches!(&*slot, ResourceSlot::CheckedOut),
            "engine resources returned twice"
        );
        *slot = ResourceSlot::Ready(resources);
    }
}

impl Drop for EngineResourceCheckout {
    fn drop(&mut self) {
        self.return_resources();
    }
}

impl CpuEngine {
    pub(crate) fn new(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        thread_budget: usize,
        buffer_limit: usize,
    ) -> Result<Self, CpuContextError> {
        let worker_count = thread_budget.min(placement.cpus().len());
        let thread_budget =
            NonZeroUsize::new(worker_count).ok_or(CpuContextError::InvalidThreadCount)?;
        let context = CpuContext::with_pinned_cpus(placement.cpus().clone(), worker_count)?;
        let caller_cpus = caller_affinity_for(&placement);
        Ok(Self::from_context(
            id,
            placement,
            Arc::new(context),
            thread_budget,
            buffer_limit,
            caller_cpus,
        ))
    }

    pub(crate) fn from_context(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        context: Arc<CpuContext>,
        thread_budget: NonZeroUsize,
        buffer_limit: usize,
        caller_cpus: Option<CpuSet>,
    ) -> Self {
        Self {
            domain: CpuResourceDomain::new(
                id,
                placement,
                Arc::clone(&context),
                thread_budget,
                caller_cpus,
            ),
            context,
            resources: Mutex::new(ResourceSlot::Ready(Box::new(EngineResources::new(
                buffer_limit,
            )))),
        }
    }

    pub(crate) fn domain(&self) -> &CpuResourceDomain {
        &self.domain
    }

    pub(crate) fn placement(&self) -> &ResolvedCpuPlacement {
        self.domain.placement()
    }
}

#[cfg(test)]
mod tests;

/// Caller-affinity target of a resolved placement: `None` for the wildcard
/// all-allowed set, the node's CPUs for an explicit node placement.
pub(crate) fn caller_affinity_for(placement: &ResolvedCpuPlacement) -> Option<CpuSet> {
    match placement {
        ResolvedCpuPlacement::NumaNode { cpus, .. } => Some(cpus.clone()),
        ResolvedCpuPlacement::AllAllowed { .. } => None,
    }
}
