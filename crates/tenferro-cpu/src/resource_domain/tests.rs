use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::*;
use crate::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuDomainId,
    CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown, CpuId, CpuInnerParallelism,
    CpuPlacementGuarantee, CpuSet, CpuSetError, NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob,
    ScopedCpuJobs,
};

#[test]
fn external_domain_guarantees_round_trip_without_upgrading_affinity() {
    for guarantee in [
        CpuPlacementGuarantee::ExactDeclared,
        CpuPlacementGuarantee::AdvisoryDeclared,
    ] {
        let domain = ExternalCpuDomain::new(
            CpuDomainId::new(7),
            node_placement(0, &[0, 1]),
            Arc::new(TestExecutor::new(2)),
            nonzero(2),
            guarantee,
        )
        .unwrap();

        assert_eq!(domain.placement_guarantee(), guarantee);
        assert_eq!(
            domain.executor_capabilities().affinity,
            CpuExecutorAffinity::CallerDeclaredUnverified
        );
    }
}

#[test]
fn worker_budget_mismatch_returns_typed_error() {
    let error = ExternalCpuDomain::new(
        CpuDomainId::new(7),
        node_placement(0, &[0, 1]),
        Arc::new(TestExecutor::new(2)),
        nonzero(3),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap_err();

    assert_eq!(
        error,
        ExternalCpuDomainError::ThreadBudgetExceedsWorkerCount {
            thread_budget: 3,
            worker_count: 2,
        }
    );
    assert!(error.to_string().contains("3"));
    assert!(error.to_string().contains("2"));
}

#[test]
fn external_node_domain_reports_public_diagnostics() {
    let placement = node_placement(4, &[3, 5]);
    let executor = Arc::new(TestExecutor::new(3));
    let expected_capabilities = executor.capabilities();
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(9),
        placement.clone(),
        executor,
        nonzero(2),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();

    assert_eq!(domain.id(), CpuDomainId::new(9));
    assert_eq!(domain.placement(), &placement);
    assert_eq!(domain.cpus().as_usize_vec(), vec![3, 5]);
    assert_eq!(domain.thread_budget(), nonzero(2));
    assert_eq!(
        domain.placement_guarantee(),
        CpuPlacementGuarantee::ExactDeclared
    );
    assert_eq!(domain.ownership(), CpuDomainOwnership::ExternalManaged);
    assert_eq!(domain.executor_capabilities(), expected_capabilities);
}

#[test]
fn external_all_allowed_domain_reports_public_diagnostics() {
    let placement = all_allowed_placement(&[1, 8]);
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(11),
        placement.clone(),
        Arc::new(TestExecutor::new(2)),
        nonzero(1),
        CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .unwrap();

    assert_eq!(domain.placement(), &placement);
    assert_eq!(domain.placement().node_id(), None);
    assert_eq!(domain.cpus().as_usize_vec(), vec![1, 8]);
    assert_eq!(
        domain.placement_guarantee(),
        CpuPlacementGuarantee::AdvisoryDeclared
    );
    assert_eq!(domain.ownership(), CpuDomainOwnership::ExternalManaged);
}

#[test]
fn external_domain_retains_executor_owner() {
    let drops = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(TestExecutor::with_drop_counter(2, Arc::clone(&drops)));
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(7),
        node_placement(0, &[0, 1]),
        executor,
        nonzero(2),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(domain);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn empty_cpu_set_is_rejected_at_the_safe_public_boundary() {
    assert_eq!(CpuSet::new(Vec::<CpuId>::new()), Err(CpuSetError::Empty));
}

#[test]
fn zero_thread_budget_is_unrepresentable_at_the_safe_public_boundary() {
    assert_eq!(NonZeroUsize::new(0), None);
}

#[test]
fn defensive_validation_rejects_empty_placement() {
    assert_eq!(
        validate_external_domain_config(0, 1, nonzero(1)),
        Err(ExternalCpuDomainError::EmptyPlacementCpuSet)
    );
}

#[test]
fn defensive_validation_rejects_zero_executor_workers() {
    assert_eq!(
        validate_external_domain_config(1, 0, nonzero(1)),
        Err(ExternalCpuDomainError::ZeroExecutorWorkers)
    );
}

#[test]
fn managed_resource_domain_preserves_ownership_and_executor_arc() {
    let executor: Arc<dyn CpuDomainExecutor> = Arc::new(TestExecutor::new(2));
    let domain = CpuResourceDomain::new(
        CpuDomainId::new(3),
        node_placement(1, &[2, 3]),
        Arc::clone(&executor),
        nonzero(2),
        CpuPlacementGuarantee::ExactDeclared,
        CpuDomainOwnership::Managed,
    );

    assert_eq!(domain.ownership(), CpuDomainOwnership::Managed);
    assert!(Arc::ptr_eq(domain.executor(), &executor));
}

#[test]
fn external_domain_moves_into_resource_domain_without_replacing_executor() {
    let executor: Arc<dyn CpuDomainExecutor> = Arc::new(TestExecutor::new(2));
    let external = ExternalCpuDomain::new(
        CpuDomainId::new(5),
        all_allowed_placement(&[0, 1]),
        Arc::clone(&executor),
        nonzero(2),
        CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .unwrap();

    let domain: CpuResourceDomain = external.into();

    assert_eq!(domain.ownership(), CpuDomainOwnership::ExternalManaged);
    assert!(Arc::ptr_eq(domain.executor(), &executor));
}

fn node_placement(node: usize, cpus: &[usize]) -> ResolvedCpuPlacement {
    ResolvedCpuPlacement::NumaNode {
        id: NumaNodeId::new(node),
        cpus: cpu_set(cpus),
    }
}

fn all_allowed_placement(cpus: &[usize]) -> ResolvedCpuPlacement {
    ResolvedCpuPlacement::AllAllowed {
        cpus: cpu_set(cpus),
    }
}

fn cpu_set(cpus: &[usize]) -> CpuSet {
    CpuSet::new(cpus.iter().copied().map(CpuId::new)).unwrap()
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

#[derive(Debug)]
struct TestExecutor {
    workers: NonZeroUsize,
    drops: Option<Arc<AtomicUsize>>,
}

impl TestExecutor {
    fn new(workers: usize) -> Self {
        Self {
            workers: nonzero(workers),
            drops: None,
        }
    }

    fn with_drop_counter(workers: usize, drops: Arc<AtomicUsize>) -> Self {
        Self {
            workers: nonzero(workers),
            drops: Some(drops),
        }
    }
}

impl Drop for TestExecutor {
    fn drop(&mut self) {
        if let Some(drops) = &self.drops {
            drops.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CpuDomainExecutor for TestExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: self.workers,
            outer_parallelism: self.workers.get() > 1,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}
