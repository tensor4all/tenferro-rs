use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use super::super::*;
use crate::{
    CpuDomainExecutor, CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuDomainId,
    CpuDomainOwnership, CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown, CpuId,
    CpuInnerParallelism, CpuPlacementGuarantee, CpuSet, ExternalCpuDomain, ScopedCpuJob,
    ScopedCpuJobs,
};

#[test]
fn external_registry_routes_without_reconstructing_executors() {
    let allowed = discover_cpu_topology().unwrap().allowed_cpus().clone();
    let node_runs = Arc::new(AtomicUsize::new(0));
    let all_runs = Arc::new(AtomicUsize::new(0));
    let node = external_domain(
        10,
        node_placement(0, allowed.clone()),
        2,
        1,
        CpuPlacementGuarantee::ExactDeclared,
        Arc::clone(&node_runs),
    );
    let all = external_domain(
        11,
        all_allowed_placement(allowed),
        3,
        2,
        CpuPlacementGuarantee::ExactDeclared,
        Arc::clone(&all_runs),
    );

    let backend =
        CpuBackend::from_external_managed_domains(CpuDomainId::new(10), [node, all]).unwrap();
    assert_eq!(
        backend.execution_info().execution_mode(),
        CpuExecutionMode::ExternalManaged
    );
    backend.install(|| {});
    assert_eq!(node_runs.load(Ordering::Relaxed), 1);
    assert_eq!(all_runs.load(Ordering::Relaxed), 0);

    let node = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0)))
        .unwrap();
    node.install(|| {});
    let all = backend.for_placement(CpuPlacement::AllAllowed).unwrap();
    all.install(|| {});
    assert_eq!(node_runs.load(Ordering::Relaxed), 2);
    assert_eq!(all_runs.load(Ordering::Relaxed), 1);
    assert!(backend.supports_placement(CpuPlacement::Auto));
    assert!(backend.supports_placement(CpuPlacement::AllAllowed));
    assert!(!backend.supports_placement(CpuPlacement::NumaNode(NumaNodeId::new(99))));
}

#[test]
fn external_diagnostics_distinguish_worker_count_and_thread_budget() {
    let topology = topology([0, 1, 2]);
    let placement = node_placement(7, cpu_set([0, 1]));
    let backend = external_backend(
        CpuDomainId::new(17),
        [external_domain(
            17,
            placement.clone(),
            3,
            2,
            CpuPlacementGuarantee::AdvisoryDeclared,
            Arc::new(AtomicUsize::new(0)),
        )],
        topology,
    )
    .unwrap();

    let info = backend.execution_info();
    assert_eq!(info.execution_mode(), CpuExecutionMode::ExternalManaged);
    assert_eq!(info.domain_id(), CpuDomainId::new(17));
    assert_eq!(info.resolved_placement(), Some(&placement));
    assert_eq!(info.domain_cpus(), placement.cpus());
    assert_eq!(info.worker_count(), 3);
    assert_eq!(info.thread_budget(), 2);
    assert_eq!(backend.num_threads(), 2);
    assert_eq!(
        info.placement_guarantee(),
        CpuPlacementGuarantee::AdvisoryDeclared
    );
    assert_eq!(info.domain_ownership(), CpuDomainOwnership::ExternalManaged);
    assert_eq!(
        info.executor_affinity(),
        CpuExecutorAffinity::CallerDeclaredUnverified
    );
    assert_eq!(info.executor_shutdown(), CpuExecutorShutdown::CallerOwned);
}

#[test]
fn external_diagnostics_never_upgrade_caller_affinity_or_shutdown_ownership() {
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(5),
        node_placement(0, cpu_set([0])),
        Arc::new(CpuContext::with_threads(1).unwrap()),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    let backend = external_backend(CpuDomainId::new(5), [domain], topology([0])).unwrap();
    let info = backend.execution_info();

    assert_eq!(
        info.executor_affinity(),
        CpuExecutorAffinity::CallerDeclaredUnverified
    );
    assert_eq!(info.executor_shutdown(), CpuExecutorShutdown::CallerOwned);
}

#[test]
fn explicit_unregistered_external_placement_is_typed_and_never_falls_back() {
    let installs = Arc::new(AtomicUsize::new(0));
    let backend = external_backend(
        CpuDomainId::new(1),
        [external_domain(
            1,
            node_placement(0, cpu_set([0])),
            1,
            1,
            CpuPlacementGuarantee::ExactDeclared,
            Arc::clone(&installs),
        )],
        topology([0, 1]),
    )
    .unwrap();

    for requested in [
        CpuPlacement::NumaNode(NumaNodeId::new(9)),
        CpuPlacement::AllAllowed,
    ] {
        assert!(matches!(
            backend.for_placement(requested),
            Err(CpuPlacementError::UnregisteredExternalPlacement {
                requested: actual,
            }) if actual == requested
        ));
    }
    assert_eq!(installs.load(Ordering::Relaxed), 0);
}

#[test]
fn external_registry_rejects_an_empty_registry() {
    let error = registry_error(external_backend(
        CpuDomainId::new(1),
        Vec::<ExternalCpuDomain>::new(),
        topology([0]),
    ));
    assert_eq!(error, ExternalCpuDomainRegistryError::EmptyRegistry);
}

#[test]
fn external_registry_rejects_duplicate_domain_ids() {
    let domains = [
        external_domain_for_validation(3, node_placement(0, cpu_set([0]))),
        external_domain_for_validation(3, node_placement(1, cpu_set([1]))),
    ];
    let error = registry_error(external_backend(
        CpuDomainId::new(3),
        domains,
        topology([0, 1]),
    ));
    assert_eq!(
        error,
        ExternalCpuDomainRegistryError::DuplicateDomainId {
            id: CpuDomainId::new(3)
        }
    );
}

#[test]
fn external_registry_rejects_duplicate_placement_identities() {
    let duplicate_node = registry_error(external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, node_placement(4, cpu_set([0]))),
            external_domain_for_validation(2, node_placement(4, cpu_set([1]))),
        ],
        topology([0, 1]),
    ));
    assert_eq!(
        duplicate_node,
        ExternalCpuDomainRegistryError::DuplicatePlacementIdentity {
            placement: CpuPlacement::NumaNode(NumaNodeId::new(4))
        }
    );

    let duplicate_all_allowed = registry_error(external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, all_allowed_placement(cpu_set([0, 1]))),
            external_domain_for_validation(2, all_allowed_placement(cpu_set([0, 1]))),
        ],
        topology([0, 1]),
    ));
    assert_eq!(
        duplicate_all_allowed,
        ExternalCpuDomainRegistryError::DuplicatePlacementIdentity {
            placement: CpuPlacement::AllAllowed
        }
    );
}

#[test]
fn external_registry_rejects_cpus_outside_the_process_allowed_set() {
    let error = registry_error(external_backend(
        CpuDomainId::new(8),
        [external_domain_for_validation(
            8,
            node_placement(0, cpu_set([0, 9])),
        )],
        topology([0, 1]),
    ));
    assert_eq!(
        error,
        ExternalCpuDomainRegistryError::CpuOutsideAllowedSet {
            domain: CpuDomainId::new(8),
            cpu: CpuId::new(9),
        }
    );
}

#[test]
fn external_registry_requires_the_declared_default_domain() {
    let error = registry_error(external_backend(
        CpuDomainId::new(99),
        [external_domain_for_validation(
            1,
            node_placement(0, cpu_set([0])),
        )],
        topology([0]),
    ));
    assert_eq!(
        error,
        ExternalCpuDomainRegistryError::MissingDefaultDomain {
            default_domain: CpuDomainId::new(99)
        }
    );
}

#[test]
fn exact_all_allowed_external_domain_must_equal_the_allowed_set() {
    let declared = cpu_set([0]);
    let allowed = cpu_set([0, 1]);
    let error = registry_error(external_backend(
        CpuDomainId::new(4),
        [external_domain_for_validation(
            4,
            all_allowed_placement(declared.clone()),
        )],
        CpuTopology::all_allowed(allowed.clone()),
    ));
    assert_eq!(
        error,
        ExternalCpuDomainRegistryError::ExactAllAllowedMismatch {
            domain: CpuDomainId::new(4),
            declared,
            allowed,
        }
    );
}

#[test]
fn distinct_external_placement_identities_may_overlap() {
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, node_placement(0, cpu_set([0, 1]))),
            external_domain_for_validation(2, node_placement(1, cpu_set([1, 2]))),
        ],
        topology([0, 1, 2]),
    )
    .unwrap();

    assert!(backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0)))
        .is_ok());
    assert!(backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .is_ok());
}

#[test]
fn disjoint_external_domains_execute_concurrently() {
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, node_placement(0, cpu_set([0]))),
            external_domain_for_validation(2, node_placement(1, cpu_set([1]))),
        ],
        topology([0, 1]),
    )
    .unwrap();
    let first = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0)))
        .unwrap();
    let second = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let run = |backend: CpuBackend| {
        let barrier = Arc::clone(&barrier);
        let active = Arc::clone(&active);
        let max_active = Arc::clone(&max_active);
        std::thread::spawn(move || {
            backend.install(|| {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now, Ordering::SeqCst);
                barrier.wait();
                active.fetch_sub(1, Ordering::SeqCst);
            });
        })
    };
    let first = run(first);
    let second = run(second);
    barrier.wait();
    first.join().unwrap();
    second.join().unwrap();

    assert_eq!(max_active.load(Ordering::SeqCst), 2);
}

#[test]
fn overlapping_exact_and_advisory_external_domains_serialize() {
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain(
                1,
                node_placement(0, cpu_set([0, 1])),
                1,
                1,
                CpuPlacementGuarantee::ExactDeclared,
                Arc::new(AtomicUsize::new(0)),
            ),
            external_domain(
                2,
                node_placement(1, cpu_set([1, 2])),
                1,
                1,
                CpuPlacementGuarantee::AdvisoryDeclared,
                Arc::new(AtomicUsize::new(0)),
            ),
        ],
        topology([0, 1, 2]),
    )
    .unwrap();
    let first = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0)))
        .unwrap();
    let second = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();

    let permit = first.try_acquire_execution_permit_for_test().unwrap();
    assert!(permit.is_some());
    let blocked = std::thread::spawn(move || second.try_acquire_execution_permit_for_test())
        .join()
        .unwrap()
        .unwrap();
    assert!(blocked.is_none());
    drop(permit);

    let second = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();
    assert!(
        std::thread::spawn(move || second.try_acquire_execution_permit_for_test())
            .join()
            .unwrap()
            .unwrap()
            .is_some()
    );
}

#[test]
fn external_domain_permit_releases_when_the_operation_panics() {
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, node_placement(0, cpu_set([0, 1]))),
            external_domain_for_validation(2, node_placement(1, cpu_set([1, 2]))),
        ],
        topology([0, 1, 2]),
    )
    .unwrap();
    let first = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0)))
        .unwrap();
    let second = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();

    assert!(catch_unwind(AssertUnwindSafe(|| first.install(|| panic!("forced")))).is_err());
    assert_eq!(second.install(|| 7), 7);
}

#[test]
fn external_executor_error_keeps_its_diagnostic_and_releases_the_permit() {
    let rejected = ExternalCpuDomain::new(
        CpuDomainId::new(1),
        node_placement(0, cpu_set([0, 1])),
        Arc::new(RejectingExecutor),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            rejected,
            external_domain_for_validation(2, node_placement(1, cpu_set([1, 2]))),
        ],
        topology([0, 1, 2]),
    )
    .unwrap();

    let message = catch_unwind(AssertUnwindSafe(|| backend.install(|| ())))
        .err()
        .map(super::panic_message)
        .expect("executor admission failure should panic at the infallible API");
    assert!(message.contains("external executor failed"), "{message}");
    assert!(message.contains("fixture rejection"), "{message}");

    let overlapping = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();
    assert_eq!(overlapping.install(|| 7), 7);
}

#[test]
fn external_domain_rejects_nested_backend_entry() {
    let backend = external_backend(
        CpuDomainId::new(1),
        [external_domain_for_validation(
            1,
            node_placement(0, cpu_set([0])),
        )],
        topology([0]),
    )
    .unwrap();
    let nested = backend.clone();

    let message = catch_unwind(AssertUnwindSafe(|| {
        backend.install(|| nested.install(|| ()))
    }))
    .err()
    .map(super::panic_message)
    .expect("nested external execution should panic");

    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
fn external_provider_staging_is_explicitly_deferred_to_task_six() {
    let installs = Arc::new(AtomicUsize::new(0));
    let mut backend = external_backend(
        CpuDomainId::new(1),
        [external_domain(
            1,
            node_placement(0, cpu_set([0])),
            1,
            1,
            CpuPlacementGuarantee::ExactDeclared,
            Arc::clone(&installs),
        )],
        topology([0]),
    )
    .unwrap();

    let message = catch_unwind(AssertUnwindSafe(|| backend.with_linalg_pool(|_| ())))
        .err()
        .map(super::panic_message)
        .expect("phase-1 provider staging must reject external domains");

    assert!(message.contains("phase-2 execution-context migration"));
    assert_eq!(installs.load(Ordering::Relaxed), 0);
}

#[test]
fn external_registry_and_handle_clones_retain_executor_owners() {
    let first_drops = Arc::new(AtomicUsize::new(0));
    let second_drops = Arc::new(AtomicUsize::new(0));
    let backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain_with_drop_counter(
                1,
                node_placement(0, cpu_set([0])),
                Arc::clone(&first_drops),
            ),
            external_domain_with_drop_counter(
                2,
                node_placement(1, cpu_set([1])),
                Arc::clone(&second_drops),
            ),
        ],
        topology([0, 1]),
    )
    .unwrap();
    let first = backend.clone();
    let second = backend
        .for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1)))
        .unwrap();
    second.install(|| {
        assert_eq!(first_drops.load(Ordering::Relaxed), 0);
        assert_eq!(second_drops.load(Ordering::Relaxed), 0);
    });

    drop(backend);
    drop(first);
    assert_eq!(first_drops.load(Ordering::Relaxed), 0);
    assert_eq!(second_drops.load(Ordering::Relaxed), 0);
    drop(second);
    assert_eq!(first_drops.load(Ordering::Relaxed), 1);
    assert_eq!(second_drops.load(Ordering::Relaxed), 1);
}

#[test]
fn external_prebuilt_engines_are_counted_once_by_buffer_controls() {
    let mut backend = external_backend(
        CpuDomainId::new(1),
        [
            external_domain_for_validation(1, node_placement(0, cpu_set([0]))),
            external_domain_for_validation(2, node_placement(1, cpu_set([1]))),
        ],
        topology([0, 1]),
    )
    .unwrap();

    let engines = backend
        .shared
        .initialized_engines("external_prebuilt_engines_are_counted_once_by_buffer_controls")
        .unwrap();
    assert_eq!(engines.len(), 2);
    backend.set_buffer_pool_limit_bytes(17).unwrap();
    for engine in engines {
        assert_eq!(
            engine
                .resources
                .lock()
                .unwrap()
                .buffers
                .max_retained_capacity_bytes(),
            17
        );
    }
}

fn external_backend(
    default_domain: CpuDomainId,
    domains: impl IntoIterator<Item = ExternalCpuDomain>,
    topology: CpuTopology,
) -> Result<CpuBackend, CpuBackendError> {
    CpuBackend::from_external_managed_domains_with_topology_and_arbiter(
        default_domain,
        domains,
        topology,
        ResourceArbiter::new(),
    )
}

fn registry_error(result: Result<CpuBackend, CpuBackendError>) -> ExternalCpuDomainRegistryError {
    match result.unwrap_err() {
        CpuBackendError::ExternalRegistry(error) => error,
        other => panic!("unexpected external registry error: {other:?}"),
    }
}

fn external_domain_for_validation(id: u64, placement: ResolvedCpuPlacement) -> ExternalCpuDomain {
    external_domain(
        id,
        placement,
        1,
        1,
        CpuPlacementGuarantee::ExactDeclared,
        Arc::new(AtomicUsize::new(0)),
    )
}

fn external_domain(
    id: u64,
    placement: ResolvedCpuPlacement,
    workers: usize,
    thread_budget: usize,
    guarantee: CpuPlacementGuarantee,
    installs: Arc<AtomicUsize>,
) -> ExternalCpuDomain {
    ExternalCpuDomain::new(
        CpuDomainId::new(id),
        placement,
        Arc::new(CountingExecutor {
            workers: NonZeroUsize::new(workers).unwrap(),
            installs,
            drops: None,
        }),
        NonZeroUsize::new(thread_budget).unwrap(),
        guarantee,
    )
    .unwrap()
}

fn external_domain_with_drop_counter(
    id: u64,
    placement: ResolvedCpuPlacement,
    drops: Arc<AtomicUsize>,
) -> ExternalCpuDomain {
    ExternalCpuDomain::new(
        CpuDomainId::new(id),
        placement,
        Arc::new(CountingExecutor {
            workers: NonZeroUsize::new(1).unwrap(),
            installs: Arc::new(AtomicUsize::new(0)),
            drops: Some(drops),
        }),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap()
}

fn node_placement(id: usize, cpus: CpuSet) -> ResolvedCpuPlacement {
    ResolvedCpuPlacement::NumaNode {
        id: NumaNodeId::new(id),
        cpus,
    }
}

fn all_allowed_placement(cpus: CpuSet) -> ResolvedCpuPlacement {
    ResolvedCpuPlacement::AllAllowed { cpus }
}

fn topology<const N: usize>(cpus: [usize; N]) -> CpuTopology {
    CpuTopology::all_allowed(cpu_set(cpus))
}

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}

#[derive(Debug)]
struct CountingExecutor {
    workers: NonZeroUsize,
    installs: Arc<AtomicUsize>,
    drops: Option<Arc<AtomicUsize>>,
}

impl Drop for CountingExecutor {
    fn drop(&mut self) {
        if let Some(drops) = &self.drops {
            drops.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CpuDomainExecutor for CountingExecutor {
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
        self.installs.fetch_add(1, Ordering::Relaxed);
        job.run()
    }
}

#[derive(Debug)]
struct RejectingExecutor;

impl CpuDomainExecutor for RejectingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, _jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        Err(CpuDomainExecutorError::Admission {
            message: "fixture rejection".to_owned(),
        })
    }

    fn install(&self, _job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        Err(CpuDomainExecutorError::Admission {
            message: "fixture rejection".to_owned(),
        })
    }
}
