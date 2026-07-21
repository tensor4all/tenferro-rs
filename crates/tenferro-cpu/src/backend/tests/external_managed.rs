use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

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

    let mut entered = observe_external_overlap(first, second, Duration::from_secs(2)).unwrap();
    entered.sort_unstable();
    assert_eq!(entered, [0, 1]);
}

#[test]
fn overlap_observer_times_out_and_releases_serialized_workers() {
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

    let error = observe_external_overlap(first, second, Duration::from_millis(50)).unwrap_err();

    assert!(error.contains("did not overlap"), "{error}");
}

#[test]
fn overlap_observer_tolerates_slow_initial_worker_scheduling() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        entered_tx.send(0).unwrap();
        entered_tx.send(1).unwrap();
    });

    let observed = receive_overlap_entries(&entered_rx, Duration::from_millis(50));
    worker.join().unwrap();

    assert_eq!(observed.unwrap(), [0, 1]);
}

fn observe_external_overlap(
    first: CpuBackend,
    second: CpuBackend,
    overlap_timeout: Duration,
) -> Result<[usize; 2], String> {
    const RELEASE_MARGIN: Duration = Duration::from_secs(2);

    let release_timeout = overlap_timeout.saturating_add(RELEASE_MARGIN);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_first_tx, release_first_rx) = mpsc::channel();
    let (release_second_tx, release_second_rx) = mpsc::channel();
    let spawn_worker = |index, backend: CpuBackend, release_rx: mpsc::Receiver<()>| {
        let entered_tx = entered_tx.clone();
        std::thread::spawn(move || {
            backend.install(move || {
                entered_tx
                    .send(index)
                    .expect("overlap observer must remain alive");
                release_rx
                    .recv_timeout(release_timeout)
                    .expect("overlap observer must release every entered worker");
            });
        })
    };
    let first_worker = spawn_worker(0, first, release_first_rx);
    let second_worker = spawn_worker(1, second, release_second_rx);
    drop(entered_tx);

    let observed = receive_overlap_entries(&entered_rx, overlap_timeout);

    let first_release = release_first_tx.send(());
    let second_release = release_second_tx.send(());
    let first_join = first_worker.join();
    let second_join = second_worker.join();

    if first_release.is_err() || second_release.is_err() {
        return Err("an overlap worker exited before its release signal".to_owned());
    }
    if first_join.is_err() || second_join.is_err() {
        return Err("an overlap worker panicked".to_owned());
    }
    observed
}

fn receive_overlap_entries(
    entered_rx: &mpsc::Receiver<usize>,
    overlap_timeout: Duration,
) -> Result<[usize; 2], String> {
    const FIRST_ENTRY_TIMEOUT: Duration = Duration::from_secs(2);

    let first = entered_rx
        .recv_timeout(FIRST_ENTRY_TIMEOUT)
        .map_err(|error| format!("first external domain did not enter: {error}"))?;
    let second = entered_rx
        .recv_timeout(overlap_timeout)
        .map_err(|error| format!("second external domain did not overlap: {error}"))?;
    Ok([first, second])
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
    assert!(message.contains("executor failed"), "{message}");
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
fn external_linalg_execution_uses_the_supplied_no_inner_executor() {
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

    backend
        .with_linalg_pool(|context, _| {
            assert_eq!(context.parallel_mode(), crate::ParallelMode::Sequential);
            Ok(())
        })
        .unwrap();

    assert_eq!(installs.load(Ordering::Relaxed), 1);
}

#[test]
fn external_elementwise_and_session_operations_use_the_supplied_no_inner_executor() {
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
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let output = backend.add(&lhs, &rhs).unwrap();
    assert_eq!(output.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert_eq!(installs.load(Ordering::Relaxed), 1);

    backend.with_backend_session(|session| {
        let output = session.add(&lhs, &rhs).unwrap();
        assert_eq!(output.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    });
    assert_eq!(installs.load(Ordering::Relaxed), 2);
}

#[test]
fn external_provider_dot_uses_the_supplied_no_inner_executor() {
    let installs = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
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
        topology([0]),
    )
    .unwrap();
    let bundle = CpuProviderBundle::builder(backend.kind())
        .prefer_general_contraction_provider(Arc::new(super::CountingGeneralProvider {
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let mut backend = backend.with_provider_bundle(bundle).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    backend.dot_general(&lhs, &rhs, &config).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
}

#[test]
fn sequential_direct_session_native_dot_and_linalg_each_enter_exactly_once() {
    let installs = Arc::new(AtomicUsize::new(0));
    let submits = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(RejectReentryCountingExecutor {
        active: AtomicBool::new(false),
        installs: Arc::clone(&installs),
        submits: Arc::clone(&submits),
    });
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(1),
        node_placement(0, cpu_set([0])),
        executor,
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    let backend = external_backend(CpuDomainId::new(1), [domain], topology([0])).unwrap();
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let bundle = CpuProviderBundle::builder(backend.kind())
        .prefer_general_contraction_provider(Arc::new(super::CountingGeneralProvider {
            calls: Arc::clone(&provider_calls),
        }))
        .build()
        .unwrap();
    let mut backend = backend.with_provider_bundle(bundle).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();

    let direct_calls = AtomicUsize::new(0);
    backend.install(|| {
        direct_calls.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(direct_calls.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 1);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    backend.add(&lhs, &rhs).unwrap();
    assert_eq!(installs.load(Ordering::Relaxed), 2);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    backend.with_backend_session(|session| {
        session.add(&lhs, &rhs).unwrap();
    });
    assert_eq!(installs.load(Ordering::Relaxed), 3);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    let linalg_calls = AtomicUsize::new(0);
    backend
        .with_linalg_pool(|context, _| {
            assert_eq!(context.parallel_mode(), crate::ParallelMode::Sequential);
            linalg_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();
    assert_eq!(linalg_calls.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 4);
    assert_eq!(submits.load(Ordering::Relaxed), 0);

    backend
        .dot_general(
            &lhs,
            &rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    assert_eq!(provider_calls.load(Ordering::Relaxed), 1);
    assert_eq!(installs.load(Ordering::Relaxed), 5);
    assert_eq!(submits.load(Ordering::Relaxed), 0);
}

#[test]
fn external_executor_error_is_preserved_as_a_typed_tensor_source() {
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(1),
        node_placement(0, cpu_set([0])),
        Arc::new(RejectingExecutor),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    let mut backend = external_backend(CpuDomainId::new(1), [domain], topology([0])).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

    let error = backend.add(&lhs, &rhs).unwrap_err();
    let crate::Error::BackendSource { source, .. } = error else {
        panic!("executor failure must retain a typed source");
    };
    assert!(matches!(
        source.downcast_ref::<CpuDomainExecutorError>(),
        Some(CpuDomainExecutorError::Admission { message }) if message == "fixture rejection"
    ));
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

#[derive(Debug)]
struct RejectReentryCountingExecutor {
    active: AtomicBool,
    installs: Arc<AtomicUsize>,
    submits: Arc<AtomicUsize>,
}

struct ReentryGuard<'a>(&'a AtomicBool);

impl Drop for ReentryGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl RejectReentryCountingExecutor {
    fn enter(&self) -> Result<ReentryGuard<'_>, CpuDomainExecutorError> {
        self.active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ReentryGuard(&self.active))
            .map_err(|_| CpuDomainExecutorError::Admission {
                message: "test executor rejected nested entry".to_owned(),
            })
    }
}

impl CpuDomainExecutor for RejectReentryCountingExecutor {
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

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        let _guard = self.enter()?;
        self.submits.fetch_add(1, Ordering::Relaxed);
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        let _guard = self.enter()?;
        self.installs.fetch_add(1, Ordering::Relaxed);
        job.run()
    }
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
