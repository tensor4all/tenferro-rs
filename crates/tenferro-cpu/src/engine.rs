use std::sync::{Arc, Mutex};

use crate::buffer_pool::BufferPool;
use crate::gemm::GemmAnalysisCache;
use crate::{CpuContext, CpuContextError, ResolvedCpuPlacement};

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
    placement: ResolvedCpuPlacement,
    context: Arc<CpuContext>,
    pub(crate) resources: Mutex<EngineResources>,
}

impl CpuEngine {
    pub(crate) fn new(
        placement: ResolvedCpuPlacement,
        thread_budget: usize,
        buffer_limit: usize,
    ) -> Result<Self, CpuContextError> {
        let worker_count = thread_budget.min(placement.cpus().len());
        let context = CpuContext::with_pinned_cpus(placement.cpus().clone(), worker_count)?;
        Ok(Self {
            placement,
            context: Arc::new(context),
            resources: Mutex::new(EngineResources::new(buffer_limit)),
        })
    }

    pub(crate) fn from_context(
        placement: ResolvedCpuPlacement,
        context: Arc<CpuContext>,
        buffer_limit: usize,
    ) -> Self {
        Self {
            placement,
            context,
            resources: Mutex::new(EngineResources::new(buffer_limit)),
        }
    }

    pub(crate) fn placement(&self) -> &ResolvedCpuPlacement {
        &self.placement
    }

    pub(crate) fn context(&self) -> &CpuContext {
        self.context.as_ref()
    }

    pub(crate) fn context_arc(&self) -> Arc<CpuContext> {
        Arc::clone(&self.context)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
