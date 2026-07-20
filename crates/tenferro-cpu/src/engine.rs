use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use crate::buffer_pool::BufferPool;
use crate::gemm::GemmAnalysisCache;
use crate::resource_domain::CpuResourceDomain;
use crate::{
    CpuContext, CpuContextError, CpuDomainExecutor, CpuDomainId, CpuDomainOwnership,
    CpuExecutorAffinity, CpuPlacementGuarantee, ExternalCpuDomain, ResolvedCpuPlacement,
};

#[derive(Debug)]
pub(crate) struct EngineResources {
    pub(crate) buffers: BufferPool,
    pub(crate) gemm_analysis_cache: GemmAnalysisCache,
}

impl EngineResources {
    pub(crate) fn new(buffer_limit: usize) -> Self {
        Self {
            buffers: BufferPool::with_max_retained_capacity_bytes(buffer_limit),
            gemm_analysis_cache: GemmAnalysisCache::default(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CpuEngine {
    domain: CpuResourceDomain,
    // INVARIANT: this is only a concrete alias of `domain.executor` for the
    // temporary phase-1 provider context. External engines never populate it.
    compatibility_context: Option<Arc<CpuContext>>,
    pub(crate) resources: Mutex<EngineResources>,
}

impl CpuEngine {
    pub(crate) fn new_managed(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        thread_budget: usize,
        buffer_limit: usize,
    ) -> Result<Self, CpuContextError> {
        let worker_count = thread_budget.min(placement.cpus().len());
        let thread_budget =
            NonZeroUsize::new(worker_count).ok_or(CpuContextError::InvalidThreadCount)?;
        let context = CpuContext::with_pinned_cpus(placement.cpus().clone(), worker_count)?;
        Ok(Self::from_managed_context(
            id,
            placement,
            Arc::new(context),
            thread_budget,
            CpuPlacementGuarantee::ExactDeclared,
            buffer_limit,
        ))
    }

    pub(crate) fn from_context(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        context: Arc<CpuContext>,
        buffer_limit: usize,
    ) -> Self {
        let capabilities = context.capabilities();
        let placement_guarantee =
            if capabilities.affinity == CpuExecutorAffinity::TenferroPinnedVerified {
                CpuPlacementGuarantee::ExactDeclared
            } else {
                CpuPlacementGuarantee::AdvisoryDeclared
            };
        Self::from_managed_context(
            id,
            placement,
            context,
            capabilities.worker_count,
            placement_guarantee,
            buffer_limit,
        )
    }

    fn from_managed_context(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        context: Arc<CpuContext>,
        thread_budget: NonZeroUsize,
        placement_guarantee: CpuPlacementGuarantee,
        buffer_limit: usize,
    ) -> Self {
        let executor: Arc<dyn CpuDomainExecutor> = context.clone();
        Self {
            domain: CpuResourceDomain::new(
                id,
                placement,
                executor,
                thread_budget,
                placement_guarantee,
                CpuDomainOwnership::Managed,
            ),
            compatibility_context: Some(context),
            resources: Mutex::new(EngineResources::new(buffer_limit)),
        }
    }

    pub(crate) fn from_external(domain: ExternalCpuDomain, buffer_limit: usize) -> Self {
        Self {
            domain: domain.into(),
            compatibility_context: None,
            resources: Mutex::new(EngineResources::new(buffer_limit)),
        }
    }

    pub(crate) fn domain(&self) -> &CpuResourceDomain {
        &self.domain
    }

    pub(crate) fn placement(&self) -> &ResolvedCpuPlacement {
        self.domain.placement()
    }

    #[cfg(test)]
    pub(crate) fn compatibility_context(&self) -> Option<&CpuContext> {
        self.compatibility_context.as_deref()
    }

    pub(crate) fn compatibility_context_arc(&self) -> Option<Arc<CpuContext>> {
        self.compatibility_context.clone()
    }
}

#[cfg(test)]
mod tests;
