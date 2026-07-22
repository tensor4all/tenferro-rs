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
    pub(crate) resources: Mutex<EngineResources>,
}

impl CpuEngine {
    pub(crate) fn new_managed(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        thread_budget: usize,
        buffer_limit: usize,
    ) -> Result<Self, CpuContextError> {
        let thread_budget =
            NonZeroUsize::new(thread_budget).ok_or(CpuContextError::InvalidThreadCount)?;
        let context = CpuContext::with_pinned_cpus(placement.cpus().clone(), thread_budget.get())?;
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
            resources: Mutex::new(EngineResources::new(buffer_limit)),
        }
    }

    pub(crate) fn from_external(domain: ExternalCpuDomain, buffer_limit: usize) -> Self {
        Self {
            domain: domain.into(),
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
