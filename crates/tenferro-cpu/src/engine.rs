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
    pub(crate) runtime_clears: u64,
}

impl EngineResources {
    pub(crate) fn new(buffer_limit: usize) -> Self {
        Self {
            buffers: BufferPool::with_max_retained_capacity_bytes(buffer_limit),
            gemm_analysis_cache: GemmAnalysisCache::default(),
            indexed_plan_cache: IndexedPlanCache::default(),
            runtime_clears: 0,
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
    pub(crate) resources: Mutex<EngineResources>,
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
            resources: Mutex::new(EngineResources::new(buffer_limit)),
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
