//! The internal description of one tenferro-owned CPU resource domain.
//!
//! tenferro uses only pools it builds, so a domain is exactly an identity, the
//! resolved CPU set it was admitted for, and the pool its numerical work may
//! use. `CpuDomainId` remains placement metadata on tensors; the domain object
//! itself is crate-internal.

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::placement::ResolvedCpuPlacement;
use crate::{CpuContext, CpuDomainId, CpuSet};

/// One admitted tenferro-owned CPU resource domain.
#[derive(Debug)]
pub(crate) struct CpuResourceDomain {
    id: CpuDomainId,
    placement: ResolvedCpuPlacement,
    context: Arc<CpuContext>,
    thread_budget: NonZeroUsize,
    /// CPU set the caller-affinity guard intersects with, or `None` for the
    /// wildcard placement. An explicitly requested CPU set or NUMA node is
    /// constrained even when the resolved placement is the all-allowed set.
    caller_cpus: Option<CpuSet>,
}

impl CpuResourceDomain {
    pub(crate) fn new(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        context: Arc<CpuContext>,
        thread_budget: NonZeroUsize,
        caller_cpus: Option<CpuSet>,
    ) -> Self {
        Self {
            id,
            placement,
            context,
            thread_budget,
            caller_cpus,
        }
    }

    pub(crate) fn id(&self) -> CpuDomainId {
        self.id
    }

    pub(crate) fn placement(&self) -> &ResolvedCpuPlacement {
        &self.placement
    }

    pub(crate) fn cpus(&self) -> &CpuSet {
        self.placement.cpus()
    }

    /// CPU set the caller-affinity guard intersects with, or `None` for the
    /// wildcard placement, which performs no affinity syscall.
    pub(crate) fn caller_cpus(&self) -> Option<&CpuSet> {
        self.caller_cpus.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn context(&self) -> &Arc<CpuContext> {
        &self.context
    }

    pub(crate) fn rayon_pool(&self) -> Option<&rayon::ThreadPool> {
        self.context.rayon_pool()
    }

    pub(crate) fn thread_budget(&self) -> NonZeroUsize {
        self.thread_budget
    }
}
