use std::num::NonZeroUsize;
use std::sync::Arc;

use thiserror::Error;

use crate::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainId, CpuPlacementGuarantee, CpuSet,
    ResolvedCpuPlacement,
};

/// Ownership class of a CPU resource domain.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuDomainOwnership;
///
/// assert_ne!(
///     CpuDomainOwnership::Managed,
///     CpuDomainOwnership::ExternalManaged,
/// );
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuDomainOwnership {
    /// Tenferro constructed and owns the resource domain.
    Managed,
    /// The application supplied and owns the executor resource policy.
    ExternalManaged,
}

/// Typed failure to construct an externally managed CPU resource domain.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::ExternalCpuDomainError;
///
/// let error = ExternalCpuDomainError::ThreadBudgetExceedsWorkerCount {
///     thread_budget: 4,
///     worker_count: 2,
/// };
/// assert!(error.to_string().contains("4"));
/// ```
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ExternalCpuDomainError {
    /// The resolved placement contains no logical CPUs.
    #[error("external CPU domain placement must contain at least one CPU")]
    EmptyPlacementCpuSet,
    /// The executor reported no workers.
    #[error("external CPU domain executor must report at least one worker")]
    ZeroExecutorWorkers,
    /// The requested thread budget is larger than the executor worker count.
    #[error(
        "external CPU domain thread budget {thread_budget} exceeds executor worker count {worker_count}"
    )]
    ThreadBudgetExceedsWorkerCount {
        /// Requested maximum number of participating threads.
        thread_budget: usize,
        /// Workers reported by the supplied executor.
        worker_count: usize,
    },
}

#[derive(Debug)]
pub(crate) struct CpuResourceDomain {
    id: CpuDomainId,
    placement: ResolvedCpuPlacement,
    executor: Arc<dyn CpuDomainExecutor>,
    thread_budget: NonZeroUsize,
    placement_guarantee: CpuPlacementGuarantee,
    ownership: CpuDomainOwnership,
}

impl CpuResourceDomain {
    pub(crate) fn new(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        executor: Arc<dyn CpuDomainExecutor>,
        thread_budget: NonZeroUsize,
        placement_guarantee: CpuPlacementGuarantee,
        ownership: CpuDomainOwnership,
    ) -> Self {
        Self {
            id,
            placement,
            executor,
            thread_budget,
            placement_guarantee,
            ownership,
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

    pub(crate) fn executor(&self) -> &Arc<dyn CpuDomainExecutor> {
        &self.executor
    }

    pub(crate) fn thread_budget(&self) -> NonZeroUsize {
        self.thread_budget
    }

    pub(crate) fn placement_guarantee(&self) -> CpuPlacementGuarantee {
        self.placement_guarantee
    }

    pub(crate) fn ownership(&self) -> CpuDomainOwnership {
        self.ownership
    }

    pub(crate) fn executor_capabilities(&self) -> CpuDomainExecutorCapabilities {
        self.executor().capabilities()
    }
}

/// Caller-supplied descriptor for one externally managed CPU resource domain.
///
/// The descriptor retains the supplied executor without replacing its pool or
/// changing its affinity claim. Registration and process-CPU-set validation
/// are performed later by [`crate::CpuBackend`].
///
/// # Examples
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use std::sync::Arc;
/// use tenferro_cpu::{
///     CpuContext, CpuDomainOwnership, CpuId, CpuPlacementGuarantee, CpuSet,
///     ExternalCpuDomain, ResolvedCpuPlacement,
/// };
/// use tenferro_tensor::CpuDomainId;
///
/// let domain = ExternalCpuDomain::new(
///     CpuDomainId::new(7),
///     ResolvedCpuPlacement::AllAllowed {
///         cpus: CpuSet::new([CpuId::new(0)])?,
///     },
///     Arc::new(CpuContext::with_threads(1)?),
///     NonZeroUsize::new(1).unwrap(),
///     CpuPlacementGuarantee::AdvisoryDeclared,
/// )?;
/// assert_eq!(domain.ownership(), CpuDomainOwnership::ExternalManaged);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct ExternalCpuDomain {
    domain: CpuResourceDomain,
}

impl ExternalCpuDomain {
    /// Construct one externally managed CPU resource-domain descriptor.
    ///
    /// The executor is retained for the complete descriptor lifetime. Exact
    /// and advisory placement values remain caller declarations and do not
    /// alter the executor's affinity capability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use std::sync::Arc;
    /// use tenferro_cpu::{
    ///     CpuContext, CpuId, CpuPlacementGuarantee, CpuSet, ExternalCpuDomain,
    ///     ResolvedCpuPlacement,
    /// };
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// let domain = ExternalCpuDomain::new(
    ///     CpuDomainId::new(3),
    ///     ResolvedCpuPlacement::AllAllowed {
    ///         cpus: CpuSet::new([CpuId::new(0)])?,
    ///     },
    ///     Arc::new(CpuContext::with_threads(1)?),
    ///     NonZeroUsize::new(1).unwrap(),
    ///     CpuPlacementGuarantee::ExactDeclared,
    /// )?;
    /// assert_eq!(domain.id(), CpuDomainId::new(3));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ExternalCpuDomainError::EmptyPlacementCpuSet`] for an empty
    /// resolved CPU set, [`ExternalCpuDomainError::ZeroExecutorWorkers`] when
    /// the executor reports no workers, or
    /// [`ExternalCpuDomainError::ThreadBudgetExceedsWorkerCount`] when
    /// `thread_budget` is greater than the executor's worker count.
    pub fn new(
        id: CpuDomainId,
        placement: ResolvedCpuPlacement,
        executor: Arc<dyn CpuDomainExecutor>,
        thread_budget: NonZeroUsize,
        placement_guarantee: CpuPlacementGuarantee,
    ) -> Result<Self, ExternalCpuDomainError> {
        let worker_count = executor.capabilities().worker_count.get();
        validate_external_domain_config(placement.cpus().len(), worker_count, thread_budget)?;
        Ok(Self {
            domain: CpuResourceDomain::new(
                id,
                placement,
                executor,
                thread_budget,
                placement_guarantee,
                CpuDomainOwnership::ExternalManaged,
            ),
        })
    }

    /// Return the caller-stable identity of this CPU domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::ExternalCpuDomain;
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// let _id: fn(&ExternalCpuDomain) -> CpuDomainId = ExternalCpuDomain::id;
    /// ```
    pub fn id(&self) -> CpuDomainId {
        self.domain.id()
    }

    /// Return the declared resolved placement.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{ExternalCpuDomain, ResolvedCpuPlacement};
    ///
    /// let _placement: fn(&ExternalCpuDomain) -> &ResolvedCpuPlacement =
    ///     ExternalCpuDomain::placement;
    /// ```
    pub fn placement(&self) -> &ResolvedCpuPlacement {
        self.domain.placement()
    }

    /// Return the logical CPUs declared for this domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuSet, ExternalCpuDomain};
    ///
    /// let _cpus: fn(&ExternalCpuDomain) -> &CpuSet = ExternalCpuDomain::cpus;
    /// ```
    pub fn cpus(&self) -> &CpuSet {
        self.domain.cpus()
    }

    /// Return the nonzero thread budget requested for tenferro work.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_cpu::ExternalCpuDomain;
    ///
    /// let _budget: fn(&ExternalCpuDomain) -> NonZeroUsize =
    ///     ExternalCpuDomain::thread_budget;
    /// ```
    pub fn thread_budget(&self) -> NonZeroUsize {
        self.domain.thread_budget()
    }

    /// Return whether placement is an exact or advisory declaration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuPlacementGuarantee, ExternalCpuDomain};
    ///
    /// let _guarantee: fn(&ExternalCpuDomain) -> CpuPlacementGuarantee =
    ///     ExternalCpuDomain::placement_guarantee;
    /// ```
    pub fn placement_guarantee(&self) -> CpuPlacementGuarantee {
        self.domain.placement_guarantee()
    }

    /// Return the external ownership diagnostic.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainOwnership, ExternalCpuDomain};
    ///
    /// let _ownership: fn(&ExternalCpuDomain) -> CpuDomainOwnership =
    ///     ExternalCpuDomain::ownership;
    /// ```
    pub fn ownership(&self) -> CpuDomainOwnership {
        self.domain.ownership()
    }

    /// Return the supplied executor's immutable capability descriptor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuDomainExecutorCapabilities, ExternalCpuDomain};
    ///
    /// let _capabilities: fn(&ExternalCpuDomain) -> CpuDomainExecutorCapabilities =
    ///     ExternalCpuDomain::executor_capabilities;
    /// ```
    pub fn executor_capabilities(&self) -> CpuDomainExecutorCapabilities {
        self.domain.executor_capabilities()
    }
}

impl From<ExternalCpuDomain> for CpuResourceDomain {
    fn from(domain: ExternalCpuDomain) -> Self {
        domain.domain
    }
}

fn validate_external_domain_config(
    cpu_count: usize,
    worker_count: usize,
    thread_budget: NonZeroUsize,
) -> Result<(), ExternalCpuDomainError> {
    if cpu_count == 0 {
        return Err(ExternalCpuDomainError::EmptyPlacementCpuSet);
    }
    if worker_count == 0 {
        return Err(ExternalCpuDomainError::ZeroExecutorWorkers);
    }
    if thread_budget.get() > worker_count {
        return Err(ExternalCpuDomainError::ThreadBudgetExceedsWorkerCount {
            thread_budget: thread_budget.get(),
            worker_count,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
