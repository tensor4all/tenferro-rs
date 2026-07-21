use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

#[cfg(feature = "cpu-tblis-linked")]
use num_complex::{Complex32, Complex64};

use super::*;

mod external_managed;

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_owned();
    }
    "<non-string panic payload>".to_owned()
}

#[test]
fn cpu_tensor_kernel_parallel_features_are_wired() {
    let workspace_manifest = include_str!("../../../../Cargo.toml");
    let cpu_manifest = include_str!("../../Cargo.toml");

    let strided_kernel_line = workspace_manifest
        .lines()
        .find(|line| line.trim_start().starts_with("strided-kernel ="))
        .expect("workspace manifest should declare strided-kernel");
    assert!(
        strided_kernel_line.contains("features")
            && strided_kernel_line.contains("\"parallel\""),
        "workspace strided-kernel dependency must enable the parallel feature: {strided_kernel_line}"
    );

    let cpu_faer_line = cpu_manifest
        .lines()
        .find(|line| line.trim_start().starts_with("cpu-faer ="))
        .expect("tenferro-cpu manifest should declare cpu-faer");
    assert!(
        cpu_faer_line.contains("strided-einsum2/parallel"),
        "cpu-faer must propagate strided-einsum2/parallel: {cpu_faer_line}"
    );
}

#[test]
fn provider_context_source_cannot_reenter_or_bypass_the_executor_boundary() {
    let provider = include_str!("../provider.rs");
    let dot_runtime = include_str!("../dot_runtime.rs");
    let exec_session = include_str!("../exec_session.rs");

    assert!(provider.contains("pub(crate) struct CpuOperationEntry"));
    assert!(!provider.contains("pub fn install<R: Send>"));
    assert!(!provider.contains("pub fn submit(&self"));
    assert!(!provider.contains("fn with_parallel_mode"));
    assert!(!provider.contains("fn sequential_child"));
    assert!(!dot_runtime.contains("direct_blas"));
    assert!(!dot_runtime.contains("context.ownership()"));
    assert!(exec_session.contains("pub(crate) entry: CpuOperationEntry"));
    assert!(!exec_session.contains("pub(crate) context: CpuExecutionContext"));
}

#[test]
fn native_production_entry_points_use_the_centralized_context_policy() {
    let backend = include_str!("../backend.rs");
    let exec_session = include_str!("../exec_session.rs");

    let try_install = backend
        .split_once("fn try_install")
        .unwrap()
        .1
        .split_once("fn install_with_pool")
        .unwrap()
        .0;
    let install_with_pool = backend
        .split_once("fn install_with_pool")
        .unwrap()
        .1
        .split_once("pub fn with_linalg_pool")
        .unwrap()
        .0;
    let run_native = exec_session
        .split_once("fn run_native")
        .unwrap()
        .1
        .split_once("impl TensorDeviceTransfer")
        .unwrap()
        .0;

    for source in [try_install, install_with_pool, run_native] {
        assert!(source.contains("preferred_engine_mode"));
        assert!(source.contains("with_native_parallelism"));
        assert!(!source.contains("enter(ParallelMode::Sequential"));
    }
}

#[test]
fn native_kernel_modules_cannot_select_ambient_or_ad_hoc_execution_policies() {
    let provider = include_str!("../provider.rs");
    let native_modules = [
        ("analytic", include_str!("../analytic.rs")),
        ("elementwise", include_str!("../elementwise.rs")),
        ("indexing", include_str!("../indexing.rs")),
        ("reduction", include_str!("../reduction.rs")),
        ("structural", include_str!("../structural.rs")),
    ];

    assert!(!provider.contains("ExecutionPolicy::AmbientRayon"));
    assert_eq!(
        provider
            .matches("strided_kernel::with_execution_policy(")
            .count(),
        1
    );
    for (name, source) in native_modules {
        assert!(
            !source.contains("ExecutionPolicy::") && !source.contains("with_execution_policy("),
            "{name} must inherit native policy from CpuExecutionContext"
        );
        assert!(
            !source.contains("rayon::") && !source.contains("into_par_iter("),
            "{name} must not fan out through ambient Rayon"
        );
    }
}

#[test]
fn direct_native_scope_uses_the_selected_rayon_budget() {
    let backend = CpuBackend::with_threads(2).unwrap();
    let participants = backend
        .try_install(|| Ok(crate::provider::tests::run_unscoped_native_map(true)))
        .unwrap();

    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
}

#[test]
fn default_backend_kind_prefers_blas_when_compiled() {
    let backend = CpuBackend::new();

    #[cfg(feature = "cpu-blas")]
    assert_eq!(backend.kind(), CpuBackendKind::Blas);
    #[cfg(all(not(feature = "cpu-blas"), feature = "cpu-faer"))]
    assert_eq!(backend.kind(), CpuBackendKind::Faer);
}

#[test]
fn provider_bundle_is_installed_at_construction_and_shared_by_clones() {
    let bundle = crate::dot_runtime::CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .build()
        .unwrap();
    let backend = CpuBackend::new()
        .with_provider_bundle(bundle.clone())
        .unwrap();
    let cloned = backend.clone();

    assert!(Arc::ptr_eq(backend.provider_bundle.inner(), bundle.inner()));
    assert!(Arc::ptr_eq(cloned.provider_bundle.inner(), bundle.inner()));
}

#[test]
fn provider_bundle_custom_builder_rejects_missing_mandatory_slots() {
    let error = crate::dot_runtime::CpuProviderBundle::custom_builder()
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("GEMM"));
    assert!(error.to_string().contains("layout"));
}

#[derive(Debug)]
struct CountingGeneralProvider {
    calls: Arc<AtomicUsize>,
}

impl crate::provider::CpuGeneralContractionProvider for CountingGeneralProvider {
    fn dot_general(
        &self,
        _context: &crate::provider::CpuExecutionContext<'_>,
        _request: crate::provider::CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<crate::provider::CpuProviderOutcome> {
        self.calls.fetch_add(1, AtomicOrdering::Relaxed);
        Ok(crate::provider::CpuProviderOutcome::Executed)
    }
}

#[test]
fn direct_and_cached_sessions_share_the_installed_provider_slot() {
    let calls = Arc::new(AtomicUsize::new(0));
    let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .prefer_general_contraction_provider(Arc::new(CountingGeneralProvider {
            calls: Arc::clone(&calls),
        }))
        .build()
        .unwrap();
    let mut backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    let lhs = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    backend.dot_general(&lhs, &rhs, &config).unwrap();
    assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);

    backend.with_backend_session(|session| {
        session
            .dot_general_cached(None, &lhs, &rhs, &config)
            .unwrap();
    });
    assert_eq!(calls.load(AtomicOrdering::Relaxed), 2);
}

#[test]
fn public_buffer_pool_controls_report_poisoned_engine_resources() {
    let mut backend = CpuBackend::new();
    let original_limit = backend.buffer_pool_limit_bytes();
    let engine = Arc::clone(&backend.engine);
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _resources = engine.resources.lock().unwrap();
        panic!("poison CPU engine resources for regression test");
    }));
    assert!(poison.is_err());

    for error in [
        backend.buffer_pool_len().unwrap_err(),
        backend.buffer_pool_stats().unwrap_err(),
        backend.buffer_pool_cache_stats().unwrap_err(),
        backend.set_buffer_pool_limit_bytes(0).unwrap_err(),
        backend.reset_buffer_pool().unwrap_err(),
    ] {
        assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
        assert!(error.to_string().contains("poison"));
    }
    assert_eq!(backend.buffer_pool_limit_bytes(), original_limit);
}

#[test]
fn public_buffer_pool_controls_report_poisoned_engine_registry() {
    let backend = CpuBackend::new();
    let shared = Arc::clone(&backend.shared);
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let CpuEngineRegistry::ManagedLazy(registry) = &shared.engines else {
            panic!("default backend should use the managed engine registry");
        };
        let _engines = registry.node_engines.lock().unwrap();
        panic!("poison CPU engine registry for regression test");
    }));
    assert!(poison.is_err());

    let error = backend.buffer_pool_len().unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("engine registry lock poisoned"));
}

#[test]
fn explicit_backend_kind_constructor_records_selection() {
    let backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
}

#[test]
fn cpu_backend_can_be_bound_to_a_shared_allocation_domain() {
    #[derive(Debug)]
    struct TestDomain(tenferro_tensor::AllocationDomainId);
    impl tenferro_tensor::SharedTensorAllocationDomain for TestDomain {
        fn id(&self) -> tenferro_tensor::AllocationDomainId {
            self.0
        }

        fn allocate(
            &self,
            _dtype: tenferro_tensor::DType,
            _shape: &[usize],
        ) -> tenferro_tensor::Result<tenferro_tensor::Tensor> {
            Err(tenferro_tensor::Error::unsupported(
                "test_allocate",
                "not implemented by test domain",
            ))
        }
    }

    let domain = tenferro_tensor::AllocationDomainId::fresh();
    let backend = CpuBackend::new().with_allocation_domain(Arc::new(TestDomain(domain)));

    assert_eq!(backend.allocation_domain(), Some(domain));
    assert_eq!(backend.clone().allocation_domain(), Some(domain));
    assert_eq!(backend.shared_allocation_domain().unwrap().id(), domain);
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn placement_handle_clones_share_coordinator_engine_and_resources() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    let mut placed = backend.for_placement(CpuPlacement::AllAllowed).unwrap();
    let clone = placed.clone();

    assert_eq!(
        placed.coordinator_id_for_test(),
        clone.coordinator_id_for_test()
    );
    assert_eq!(placed.placement(), CpuPlacement::AllAllowed);
    assert!(matches!(
        placed.resolved_placement(),
        Some(ResolvedCpuPlacement::AllAllowed { .. })
    ));
    placed
        .with_linalg_pool(|_, pool| {
            <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0]);
            Ok(())
        })
        .unwrap();
    assert_eq!(clone.buffer_pool_len().unwrap(), 1);
}

#[test]
#[cfg(feature = "cpu-faer")]
fn placement_capabilities_follow_public_backend_kind() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    assert!(backend.supports_placement(CpuPlacement::Auto));
    assert_eq!(
        backend.supports_placement(CpuPlacement::AllAllowed),
        cfg!(any(target_os = "linux", target_os = "android"))
    );
    assert!(!backend.topology().allowed_cpus().is_empty());
}

#[test]
fn execution_info_exposes_stable_kind_and_placement_contract() {
    let backend = CpuBackend::new();
    let info = backend.execution_info();

    assert_eq!(info.backend_kind(), backend.kind());
    assert_eq!(info.requested_placement(), CpuPlacement::Auto);
    assert_eq!(info.resolved_placement(), backend.resolved_placement());
    assert_eq!(info.topology(), backend.topology());
    assert_eq!(info.worker_count(), backend.num_threads());
    #[cfg(feature = "cpu-blas")]
    assert_eq!(
        info.execution_mode(),
        CpuExecutionMode::ProviderDefaultExclusive
    );
    #[cfg(all(
        not(feature = "cpu-blas"),
        feature = "cpu-faer",
        any(target_os = "linux", target_os = "android")
    ))]
    assert_eq!(info.execution_mode(), CpuExecutionMode::Managed);
    #[cfg(all(
        not(feature = "cpu-blas"),
        feature = "cpu-faer",
        not(any(target_os = "linux", target_os = "android"))
    ))]
    assert_eq!(info.execution_mode(), CpuExecutionMode::Compatibility);
    assert!(!info.provider_diagnostic().is_empty());
}

#[test]
#[cfg(feature = "cpu-blas")]
fn blas_provider_session_body_stays_outside_the_rayon_engine() {
    use tenferro_tensor::BackendSessionHost;

    let mut backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    backend.with_backend_session(|_| {
        assert!(rayon::current_thread_index().is_none());
    });
}

#[test]
#[cfg(feature = "cpu-blas")]
fn blas_auto_is_provider_exclusive_and_explicit_placement_is_rejected() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();

    assert!(backend.for_placement(CpuPlacement::AllAllowed).is_err());
    assert!(!backend.supports_placement(CpuPlacement::AllAllowed));
    assert!(backend.resolved_placement().is_none());
}

#[test]
#[cfg(feature = "cpu-blas")]
fn independently_constructed_backends_share_global_provider_exclusion() {
    let first = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    let second = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();

    let _permit = first.shared.arbiter.acquire_provider_exclusive().unwrap();
    let second_arbiter = second.shared.arbiter.clone();
    let blocked = std::thread::spawn(move || {
        second_arbiter
            .try_acquire_provider_exclusive()
            .unwrap()
            .is_none()
    })
    .join()
    .unwrap();
    assert!(blocked);
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn direct_nested_clone_install_is_rejected_in_a_managed_scope() {
    let backend = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let nested = backend.clone();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.install(|| nested.install(|| 7_u32))
    }));

    let message = outcome
        .err()
        .map(panic_message)
        .expect("nesting should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn direct_nested_independent_engine_is_rejected_in_a_managed_scope() {
    let outer = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let middle = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        outer.install(|| middle.install(|| 11_u32))
    }));

    let message = outcome
        .err()
        .map(panic_message)
        .expect("nesting should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(feature = "cpu-faer")]
fn cross_pool_wait_cannot_misclassify_a_scheduled_sibling_as_direct_nesting() {
    let outer = CpuBackend::from_context(Arc::new(CpuContext::with_threads(2).unwrap()));
    let middle = CpuBackend::from_context(Arc::new(CpuContext::with_threads(2).unwrap()));
    let sibling = outer.clone();

    let (_, sibling_outcome) = outer.install(move || {
        rayon::join(
            || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    middle.install(|| std::thread::sleep(Duration::from_millis(50)))
                }))
            },
            || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sibling.install(|| ()))),
        )
    });

    let message = sibling_outcome
        .err()
        .map(panic_message)
        .expect("scheduled sibling reentry should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn stolen_rayon_child_task_backend_reentry_is_rejected() {
    let outer = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let nested = outer.clone();
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        outer.install(|| {
            rayon::scope(|scope| {
                let completed_tx = completed_tx.clone();
                scope.spawn(move |_| {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        nested.install(|| 13_u32)
                    }));
                    completed_tx.send(outcome.err().map(panic_message)).unwrap();
                });
                std::thread::sleep(Duration::from_millis(100));
            });
        });
    });

    let message = completed_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("parallel child reentry should fail without deadlocking")
        .expect("parallel child reentry should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn parallel_rayon_sibling_backend_reentry_is_rejected() {
    let outer = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let first = outer.clone();
    let second = outer.clone();
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    outer.install(|| {
        rayon::scope(|scope| {
            for nested in [first, second] {
                let completed_tx = completed_tx.clone();
                scope.spawn(move |_| {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        nested.install(|| ())
                    }));
                    completed_tx.send(outcome.err().map(panic_message)).unwrap();
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        });
    });

    for _ in 0..2 {
        let message = completed_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("parallel sibling reentry should fail without deadlocking")
            .expect("parallel sibling reentry should panic");
        assert!(
            message.contains("another CPU backend execution"),
            "{message}"
        );
    }
}

#[test]
#[cfg(feature = "cpu-faer")]
fn shared_context_work_is_not_mistaken_for_backend_reentry() {
    let context = Arc::new(CpuContext::with_threads(2).unwrap());
    let backend = CpuBackend::from_context(Arc::clone(&context));
    let nested = backend.clone();

    let message = backend.install(|| {
        let (completed_tx, completed_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                context.install(|| nested.install(|| ()))
            }));
            completed_tx.send(outcome.err().map(panic_message)).unwrap();
        });
        completed_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("shared-context work should fail without deadlocking")
            .expect("shared-context work should not bypass backend exclusion")
    });

    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn shared_execution_scope_is_cleared_after_panic() {
    let backend = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.install(|| panic!("forced nested execution panic"));
    }));

    assert!(panic.is_err());
    assert_eq!(backend.install(|| 17_u32), 17);
}

#[test]
#[cfg(all(feature = "cpu-faer", any(target_os = "linux", target_os = "android")))]
fn nested_clone_tensor_operation_is_rejected_in_a_managed_scope() {
    let mut backend = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let mut nested = backend.clone();
    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_| nested.add(&lhs, &rhs))
    }));

    let message = outcome
        .err()
        .map(panic_message)
        .expect("nesting should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(feature = "cpu-blas")]
fn nested_provider_session_is_rejected() {
    let mut backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    let mut nested = backend.clone();
    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_| nested.add(&lhs, &rhs))
    }));

    let message = outcome
        .err()
        .map(panic_message)
        .expect("nesting should panic");
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
}

#[test]
#[cfg(all(feature = "cpu-faer", feature = "cpu-blas"))]
fn parallel_rayon_siblings_cannot_bypass_provider_exclusion() {
    let outer = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer).unwrap();
    let provider = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).unwrap();
    let first = provider.clone();
    let second = provider;
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    outer.install(|| {
        rayon::scope(|scope| {
            for nested in [first, second] {
                let completed_tx = completed_tx.clone();
                scope.spawn(move |_| {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        nested.install(|| ())
                    }));
                    completed_tx.send(outcome.err().map(panic_message)).unwrap();
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        });
    });

    for _ in 0..2 {
        let message = completed_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("provider sibling reentry should fail without deadlocking")
            .expect("provider sibling reentry should panic");
        assert!(
            message.contains("another CPU backend execution"),
            "{message}"
        );
    }
}

#[test]
#[cfg(feature = "cpu-blas")]
fn explicit_blas_backend_kind_constructor_records_selection() {
    let backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::Blas);
}

#[test]
fn explicit_dot_general_provider_maps_to_a_new_bundle() {
    let backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    let base_bundle = backend.provider_bundle().clone();
    let backend = backend.with_dot_general_provider(DotGeneralProvider::TblisIfAvailable);

    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    assert!(!backend.provider_bundle().shares_identity_with(&base_bundle));
}

#[test]
#[cfg(feature = "cpu-tblis-linked")]
fn tblis_dot_general_matches_column_major_matmul() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisRequired);
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = backend.dot_general(&lhs, &rhs, &config).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[76.0, 100.0, 103.0, 136.0]);
}

#[test]
#[cfg(feature = "cpu-tblis-linked")]
fn tblis_dot_general_read_into_accum_applies_alpha_beta() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisRequired);
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: tenferro_tensor::ContractionScalar::F64(2.0),
        beta: tenferro_tensor::ContractionScalar::F64(3.0),
    };

    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            accumulation,
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();

    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[155.0, 203.0, 209.0, 275.0]
    );
}

#[test]
#[cfg(feature = "cpu-tblis-linked")]
fn tblis_dot_general_supports_c64() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisRequired);
    let lhs = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(-1.0, 2.0),
        ],
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(0.5, -1.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(1.5, -3.0),
        ],
    )
    .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = backend.dot_general(&lhs, &rhs, &config).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_complex64_close(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(13.0, 4.5),
            Complex64::new(-6.0, 4.5),
            Complex64::new(3.75, -10.0),
            Complex64::new(0.75, 8.5),
        ],
        1.0e-12,
    );
}

#[test]
#[cfg(feature = "cpu-tblis-linked")]
fn tblis_dot_general_c32_conj_accum() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisRequired);
    let lhs = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(2.0, 0.5),
            Complex32::new(3.0, -1.0),
            Complex32::new(-1.0, 1.0),
        ],
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(0.5, 1.0),
            Complex32::new(4.0, -1.0),
            Complex32::new(-2.0, 0.25),
            Complex32::new(1.5, 3.0),
        ],
    )
    .unwrap();
    let mut out =
        Tensor::from_vec_col_major(vec![2, 2], vec![Complex32::new(1.0, -2.0); 4]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: true,
        rhs_conj: false,
        alpha: tenferro_tensor::ContractionScalar::C32(Complex32::new(2.0, -1.0)),
        beta: tenferro_tensor::ContractionScalar::C32(Complex32::new(-1.0, 0.5)),
    };

    backend
        .dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
            accumulation,
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();

    assert_complex32_close(
        out.as_slice::<Complex32>().unwrap(),
        &[
            Complex32::new(32.0, -11.0),
            Complex32::new(-8.25, 3.5),
            Complex32::new(14.75, 32.0),
            Complex32::new(-7.75, -1.125),
        ],
        1.0e-5,
    );
}

#[test]
#[cfg(feature = "cpu-tblis-provider")]
fn tblis_dot_general_falls_back_for_scalar_output_inner_product() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisIfAvailable);
    let lhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = backend.dot_general(&lhs, &rhs, &config).unwrap();

    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[32.0]);
}

#[test]
#[cfg(feature = "cpu-tblis-provider")]
fn tblis_dot_general_falls_back_for_zero_size_matmul() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisIfAvailable);
    let lhs = Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new()).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![0, 3], Vec::<f64>::new()).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = backend.dot_general(&lhs, &rhs, &config).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0; 6]);
}

#[cfg(feature = "cpu-tblis-linked")]
fn assert_complex64_close(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tol,
            "complex64 mismatch at {idx}: actual={actual:?} expected={expected:?}"
        );
    }
}

#[cfg(feature = "cpu-tblis-linked")]
fn assert_complex32_close(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tol,
            "complex32 mismatch at {idx}: actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn with_threads_and_kind_records_selection_and_validates_threads() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::default_compiled()).unwrap();
    assert_eq!(backend.num_threads(), 1);
    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());

    let err = match CpuBackend::with_threads_and_kind(0, CpuBackendKind::default_compiled()) {
        Ok(_) => panic!("expected invalid thread count to fail"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        CpuBackendError::Tensor(crate::Error::Validation {
            op: "CpuBackend::with_threads_and_kind",
            ..
        })
    ));
}

#[test]
#[cfg(not(feature = "cpu-blas"))]
fn unavailable_blas_backend_kind_reports_config_errors() {
    let err = match CpuBackend::with_kind(CpuBackendKind::Blas) {
        Ok(_) => panic!("expected unavailable BLAS backend to fail"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        CpuBackendError::Tensor(crate::Error::Validation {
            op: "CpuBackend::with_kind",
            ..
        })
    ));

    let mut backend = CpuBackend::compatibility(
        Arc::new(CpuContext::with_threads(1).unwrap()),
        crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        CpuBackendKind::Blas,
    );
    let retained = backend
        .with_linalg_pool(|_, pool| {
            <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0]);
            Ok(pool.len())
        })
        .unwrap();
    assert_eq!(retained, 1);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);

    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut cache = gemm::GemmAnalysisCache::default();

    for result in [
        backend.dot_general_cached(&mut cache, Some(0), &lhs, &rhs, &config),
        backend.dot_general_with_conj_cached(&mut cache, Some(1), &lhs, &rhs, &config, false, true),
        backend.dot_general_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &config,
        ),
    ] {
        let err = result.unwrap_err();
        assert_eq!(err.kind(), tenferro_tensor::ErrorKind::Unsupported);
        assert!(err.to_string().contains("GEMM"));
    }
}

#[test]
fn cpu_session_profile_helpers_cover_current_profile_mode() {
    let state = cpu_session_profile_state();
    state
        .lock()
        .expect("CPU session profile mutex poisoned")
        .clear();

    let profiling_enabled = cpu_session_profile_enabled();
    let _ = cpu_session_profile_print_every();

    let value = profile_cpu_session_section("test.profile_section", || 7);
    assert_eq!(value, 7);
    record_cpu_session_profile("test.manual_record", Duration::from_nanos(1));

    let entries = state.lock().expect("CPU session profile mutex poisoned");
    if profiling_enabled {
        assert!(entries.contains_key("test.profile_section"));
        assert!(entries.contains_key("test.manual_record"));
    } else {
        assert!(entries.is_empty());
    }
    drop(entries);

    maybe_print_cpu_session_profile();
}

#[test]
fn with_linalg_pool_restores_backend_pool_and_context() {
    let mut backend = CpuBackend::with_threads(1).unwrap();

    let len_inside_pool = backend
        .with_linalg_pool(|context, pool| {
            assert_eq!(context.thread_budget().get(), 1);
            <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0, 3.0, 4.0]);
            Ok(pool.len())
        })
        .unwrap();

    assert_eq!(len_inside_pool, 1);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn linalg_pool_acquire_then_panic_replenishes_buffer_but_reports_poison() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend
        .with_linalg_pool(|_, pool| {
            <f64 as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
            Ok(())
        })
        .unwrap();
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
    assert_eq!(
        backend.buffer_pool_stats().unwrap().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = backend.with_linalg_pool::<()>(|_, pool| {
            let _in_flight = unsafe { <f64 as PoolScalar>::pool_acquire(pool, 1024) };
            assert_eq!(pool.retained_capacity_bytes(), 0);
            panic!("forced panic after pool acquisition");
        });
    }));

    assert!(result.is_err());
    let resources = backend.engine.resources.lock().unwrap_err().into_inner();
    assert_eq!(resources.buffers.len(), 1);
    assert_eq!(
        resources.buffers.stats().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );
    drop(resources);
    assert_eq!(
        backend.buffer_pool_len().unwrap_err().kind(),
        tenferro_tensor::ErrorKind::RuntimeState
    );
}

#[test]
fn cached_dot_dispatch_reports_dtype_mismatches() {
    let mut backend = CpuBackend::new();
    let mut cache = gemm::GemmAnalysisCache::default();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let rhs = Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let dot_error = backend.dot_general_cached(&mut cache, Some(0), &lhs, &rhs, &config);
    assert!(matches!(
        dot_error,
        Err(crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let dot_conj_error =
        backend.dot_general_with_conj_cached(&mut cache, Some(1), &lhs, &rhs, &config, true, false);
    assert!(matches!(
        dot_conj_error,
        Err(crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}

#[test]
fn with_threads_rejects_invalid_thread_count() {
    let result: Result<CpuBackend, CpuBackendError> = CpuBackend::with_threads(0);
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        CpuBackendError::Tensor(crate::Error::Validation {
            op: "CpuBackend::with_threads",
            ..
        })
    ));
}

#[test]
fn backend_error_keeps_placement_failure_typed() {
    let error = CpuBackendError::placement(
        "CpuBackend::try_new",
        CpuPlacementError::TopologyDiscovery {
            requested: CpuPlacement::Auto,
            backend: CpuBackendKind::Faer,
            source: CpuTopologyError::InvalidCpuList {
                list: "bad".to_owned(),
                reason: "test failure",
            },
        },
    );
    assert!(matches!(
        error.placement_error(),
        Some(CpuPlacementError::TopologyDiscovery {
            requested: CpuPlacement::Auto,
            backend: CpuBackendKind::Faer,
            source: CpuTopologyError::InvalidCpuList { .. },
        })
    ));
}

#[test]
fn placement_error_conversion_uses_runtime_state_for_environment_failures() {
    let error = CpuBackendError::placement(
        "CpuBackend::try_new",
        CpuPlacementError::ManagedAffinityUnavailable {
            requested: CpuPlacement::AllAllowed,
            backend: CpuBackendKind::Faer,
        },
    );

    let error: crate::Error = error.into();
    assert_eq!(error.kind(), crate::ErrorKind::RuntimeState);
    assert!(matches!(
        std::error::Error::source(&error),
        Some(source) if source.downcast_ref::<CpuPlacementError>().is_some()
    ));
}

#[test]
fn placement_error_conversion_keeps_unsupported_affinity_distinct() {
    let error = CpuBackendError::placement(
        "CpuBackend::with_placement",
        CpuPlacementError::ExternalProviderAffinityUnmanaged {
            requested: CpuPlacement::AllAllowed,
            backend: CpuBackendKind::Blas,
        },
    );

    let error: crate::Error = error.into();
    assert_eq!(error.kind(), crate::ErrorKind::Unsupported);
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn fallible_backend_construction_preserves_topology_error_category() {
    let source = CpuTopologyError::InvalidCpuList {
        list: "not-a-cpu".to_owned(),
        reason: "component is not a CPU number",
    };

    let error = resolve_discovered_topology(CpuBackendKind::Faer, Err(source)).unwrap_err();
    match error {
        CpuPlacementError::TopologyDiscovery {
            requested: CpuPlacement::Auto,
            backend: CpuBackendKind::Faer,
            source: CpuTopologyError::InvalidCpuList { list, .. },
        } => assert_eq!(list, "not-a-cpu"),
        other => panic!("unexpected placement error: {other:?}"),
    }
}

#[test]
#[cfg(feature = "cpu-faer")]
fn unavailable_affinity_auto_placement_reuses_compatibility_engine() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();

    let placed = backend
        .for_placement_with_affinity(CpuPlacement::Auto, false)
        .unwrap();

    assert_eq!(
        placed.execution_info().execution_mode(),
        CpuExecutionMode::Compatibility
    );
    assert_eq!(placed.context_id_for_test(), backend.context_id_for_test());
}

#[cfg(all(
    feature = "cpu-faer",
    not(any(target_os = "linux", target_os = "android"))
))]
#[test]
fn explicit_managed_affinity_reports_engine_construction_error_when_unsupported() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();

    let error = backend
        .for_placement_with_affinity(CpuPlacement::AllAllowed, true)
        .unwrap_err();

    assert!(matches!(
        error,
        CpuPlacementError::EngineConstruction {
            requested: CpuPlacement::AllAllowed,
            backend: CpuBackendKind::Faer,
            ..
        }
    ));
    assert!(error.to_string().contains("unsupported on this platform"));
}
