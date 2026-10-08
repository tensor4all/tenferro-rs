use std::time::Duration;
use tenferro_tensor::DotGeneralConfig;

use super::*;
use tenferro_tensor::TensorRead;
use tenferro_tensor::{BackendSessionHost, SessionEntryError};

mod execution_scope;
mod output_affinity;
#[cfg(feature = "blas")]
mod provider_session;

/// Run `f` on the CPU execution session of one fresh backend session entry.
///
/// The backend owner is not an execution surface (#1946 F6). This mirrors
/// `crate::tests::with_cpu_session`, which is not compiled under
/// `provider-inject` while these backend unit tests are.
pub(super) fn with_cpu_session<R: Send>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R {
    backend
        .with_backend_session(|session| {
            crate::with_cpu_exec_session(session, f)
                .expect("CpuBackend must expose its CpuExecSession")
        })
        .expect("CPU session entry must be admitted")
}

/// Assert that `error` is the typed CPU reentry rejection.
fn assert_reentered(error: &crate::Error) {
    assert!(
        matches!(
            error,
            crate::Error::SessionEntry {
                source: SessionEntryError::Reentered {
                    backend: CPU_BACKEND
                }
            }
        ),
        "{error}"
    );
}

fn assert_worker_rejected(error: &crate::Error) {
    assert!(
        matches!(
            error,
            crate::Error::SessionEntry {
                source: SessionEntryError::Reentered {
                    backend: CPU_BACKEND
                } | SessionEntryError::Contended {
                    backend: CPU_BACKEND,
                    ..
                }
            }
        ),
        "{error}"
    );
}

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

    assert!(
        !workspace_manifest.contains("strided-einsum2")
            && !cpu_manifest.contains("strided-einsum2"),
        "tenferro-rs must not retain the removed strided-einsum2 dependency"
    );
}

#[test]
fn indexed_plan_cache_configuration_poison_is_typed() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let shared = Arc::clone(&backend.shared);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = shared.indexed_plan_cache_limits.lock().unwrap();
        panic!("poison indexed-plan cache configuration");
    }));
    assert!(panic.is_err());

    let error = backend.indexed_plan_cache_limits().unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::RuntimeState);
    let error = backend
        .set_indexed_plan_cache_limits(IndexedPlanCacheLimits::new(1, 1))
        .unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::RuntimeState);
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
fn fresh_tagging_is_field_only_and_has_no_dynamic_lookup_or_metadata_clone() {
    let backend = include_str!("../backend.rs");
    let tagging = backend
        .split_once("fn tag_fresh_output")
        .expect("CPU backend should define one fresh-output tagger")
        .1
        .split_once("pub(crate) fn elementwise_read_into_fallback_with_pool(")
        .expect("the pooled elementwise fallback should follow fresh-output tagging")
        .0;

    assert!(tagging.contains("set_cpu_affinity(Some(domain))"));
    for forbidden in ["placement().clone", "format!", "HashMap", ".hash("] {
        assert!(
            !tagging.contains(forbidden),
            "fresh CPU tagging must not contain `{forbidden}`"
        );
    }
}

#[test]
fn native_production_entry_points_use_the_centralized_context_policy() {
    let backend = include_str!("../backend.rs");
    let exec_session = include_str!("../exec_session.rs");

    // The backend owner is not an execution surface (#1946 F6): native
    // execution with the buffer pool is entered only by the session.
    for owner_entry in [
        "fn try_install",
        "fn install_with_pool",
        "fn with_linalg_pool",
    ] {
        assert!(
            !backend.contains(owner_entry),
            "CpuBackend must not regain the owner-side native entry `{owner_entry}`"
        );
    }

    let run_native = exec_session
        .split_once("fn run_native")
        .unwrap()
        .1
        .split_once("impl TensorDeviceTransfer")
        .unwrap()
        .0;

    assert!(run_native.contains("preferred_engine_mode"));
    assert!(run_native.contains("with_native_parallelism"));
    assert!(!run_native.contains("enter(ParallelMode::Sequential"));
}

#[test]
fn native_kernel_modules_cannot_select_ambient_or_ad_hoc_execution_policies() {
    let provider = include_str!("../provider.rs");
    let native_modules = [
        ("analytic", include_str!("../analytic.rs")),
        (
            "elementwise",
            include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs"),
        ),
        (
            "fused",
            include_str!("../../../tenferro-cpu-fused/src/lib.rs"),
        ),
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
    assert!(!provider.contains("ExecContext::ambient("));
    for (name, source) in native_modules {
        assert!(
            !source.contains("ExecutionPolicy::")
                && !source.contains("with_execution_policy(")
                && !source.contains("ExecContext::ambient("),
            "{name} must inherit native policy from CpuExecutionContext"
        );
        assert!(
            !source.contains("rayon::") && !source.contains("into_par_iter("),
            "{name} must not fan out through ambient Rayon"
        );
    }
}

#[test]
fn cpu_hot_kernels_delegate_to_erased_strided_replay() {
    let elementwise = include_str!("../../../tenferro-internal-cpu-kernels/src/elementwise.rs");
    let fused = include_str!("../../../tenferro-cpu-fused/src/lib.rs");
    let indexing = include_str!("../indexing.rs");
    let reduction = include_str!("../reduction.rs");
    let structural = include_str!("../structural.rs");

    assert!(
        fused.contains("ErasedFusedPlan::compile"),
        "CPU fused adapter should delegate replay to strided-fused's erased plan"
    );
    assert!(
        !elementwise.contains("ErasedFusedPlan::compile"),
        "ordinary kernels must not instantiate fused replay"
    );
    assert!(
        indexing.contains("ErasedDynamicSlicePlan::compile")
            && indexing.contains("ErasedDynamicUpdateSlicePlan::compile")
            && indexing.contains("ErasedScatterPlan::compile"),
        "CPU indexed slice/update/scatter execution should delegate to erased strided plans"
    );
    assert!(
        indexing.contains("ErasedSlicePlan::compile")
            && indexing.contains("ErasedPadPlan::compile")
            && indexing.contains("ErasedConcatenatePlan::compile")
            && indexing.contains("ErasedReversePlan::compile"),
        "CPU static indexing execution should delegate to erased strided plans"
    );
    assert!(
        reduction.contains("ErasedReducePlan::compile_axes"),
        "CPU sum/product axis reductions should delegate to erased strided reduce plans"
    );
    for (name, source) in [
        ("elementwise", elementwise),
        ("indexing", indexing),
        ("reduction", reduction),
    ] {
        let compact_source = source.split_whitespace().collect::<String>();
        assert!(
            !compact_source.contains(".execute(&ExecContext::serial()"),
            "{name} erased strided replay should inherit CpuExecutionContext, not force serial"
        );
    }
    let pooled_triangular = structural
        .split_once("fn typed_triangular_mask_with_fill_pool")
        .and_then(|(_, rest)| rest.split_once("fn checked_triangular_extent"))
        .map(|(body, _)| body)
        .expect("pooled triangular kernel should remain discoverable");
    assert!(
        !pooled_triangular.contains("for row in 0..rows")
            && pooled_triangular.contains("strided_kernel::triangular_mask_into_uninit"),
        "CPU pooled triangular masks should delegate traversal to strided"
    );
    assert!(structural.contains("strided_kernel::embed_diagonal_into_uninit"));
    assert!(include_str!("../blas1.rs").contains("strided_kernel::axpby_accum"));
}

// The shared native-participant fixture lives in the broad unit suite, which
// `provider-inject` excludes (its symbols are registered by the integration
// fixture instead).
#[cfg(not(feature = "provider-inject"))]
#[test]
fn explicit_native_operation_uses_the_selected_rayon_budget() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let caller = std::thread::current().id();
    let participants = with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|context, _| {
            assert_eq!(std::thread::current().id(), caller);
            context.with_native_parallelism(|| {
                Ok(crate::tests::native_participants::run_unscoped_native_map(
                    true,
                ))
            })
        })
    })
    .unwrap();

    assert_eq!(participants.max_active(), 2);
    assert_eq!(participants.thread_count(), 2);
}

#[test]
fn workspace_contention_does_not_partially_clear_buffers_or_change_limits() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let engine = Arc::clone(&backend.engine);
    {
        let mut resources = engine.resources.lock().unwrap();
        <f64 as PoolScalar>::pool_release(&mut resources.buffers, Vec::with_capacity(16));
    }
    let before = backend.buffer_pool_len().unwrap();
    let limit = backend.buffer_pool_limit_bytes();
    let _workspace = engine
        .context
        .as_ref()
        .contraction_workspaces()
        .lock()
        .unwrap();
    assert!(backend.reset_buffer_pool().is_err());
    assert!(backend.set_buffer_pool_limit_bytes(0).is_err());
    assert_eq!(backend.buffer_pool_len().unwrap(), before);
    assert_eq!(backend.buffer_pool_limit_bytes(), limit);
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
    let original = CpuBackend::new();
    let original_identity = original.runtime_identity();
    let backend = original.with_allocation_domain(Arc::new(TestDomain(domain)));

    assert_eq!(backend.allocation_domain(), Some(domain));
    assert_eq!(backend.clone().allocation_domain(), Some(domain));
    assert_eq!(backend.shared_allocation_domain().unwrap().id(), domain);
    assert_ne!(backend.runtime_identity(), original_identity);
    assert_eq!(
        backend.runtime_identity(),
        backend.clone().runtime_identity()
    );
}

#[test]
#[cfg(feature = "blas")]
fn single_thread_blas_session_reuses_context_with_sequential_policy() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend
        .with_backend_session(|session| {
            crate::with_cpu_exec_session(session, |cpu| {
                assert!(cpu.entered.is_some());
                cpu.with_linalg_pool(|context, _| {
                    // Linux placement may use a pinned worker even at one thread.
                    // The contract is sequential native policy, not caller-thread identity.
                    assert_eq!(context.parallel_mode(), crate::ParallelMode::Sequential);
                    Ok(())
                })
                .unwrap();
            })
            .unwrap();
        })
        .unwrap();
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn direct_nested_clone_install_is_rejected_in_a_managed_scope() {
    let backend = CpuBackend::with_threads(2).unwrap();
    let nested = backend.clone();
    let mut inner_ran = false;

    let nested_result = backend
        .install(|| nested.install(|| inner_ran = true))
        .unwrap();

    assert_reentered(&nested_result.unwrap_err());
    assert!(!inner_ran);
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn direct_nested_independent_engine_is_rejected_in_a_managed_scope() {
    let outer = CpuBackend::with_threads(2).unwrap();
    let middle = CpuBackend::with_threads(2).unwrap();

    let nested_result = outer.install(|| middle.install(|| 11_u32)).unwrap();

    assert_reentered(&nested_result.unwrap_err());
}

#[test]
#[cfg(feature = "native")]
fn cross_pool_wait_cannot_misclassify_a_scheduled_sibling_as_direct_nesting() {
    let outer = CpuBackend::from_context(Arc::new(CpuContext::with_threads(2).unwrap()));
    let middle = CpuBackend::from_context(Arc::new(CpuContext::with_threads(2).unwrap()));
    let sibling = outer.clone();

    let (_, sibling_outcome) = outer
        .install(move || {
            rayon::join(
                || middle.install(|| std::thread::sleep(Duration::from_millis(50))),
                || sibling.install(|| ()),
            )
        })
        .unwrap();

    assert_worker_rejected(&sibling_outcome.unwrap_err());
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn stolen_rayon_child_task_backend_reentry_is_rejected() {
    let outer = CpuBackend::with_threads(2).unwrap();
    let nested = outer.clone();
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        outer
            .install(|| {
                rayon::scope(|scope| {
                    let completed_tx = completed_tx.clone();
                    scope.spawn(move |_| {
                        completed_tx.send(nested.install(|| 13_u32)).unwrap();
                    });
                    std::thread::sleep(Duration::from_millis(100));
                });
            })
            .unwrap();
    });

    let outcome = completed_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("parallel child reentry should fail without deadlocking");
    assert_worker_rejected(&outcome.unwrap_err());
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn parallel_rayon_sibling_backend_reentry_is_rejected() {
    let outer = CpuBackend::with_threads(2).unwrap();
    let first = outer.clone();
    let second = outer.clone();
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    outer
        .install(|| {
            rayon::scope(|scope| {
                for nested in [first, second] {
                    let completed_tx = completed_tx.clone();
                    scope.spawn(move |_| {
                        completed_tx.send(nested.install(|| ())).unwrap();
                    });
                }
                std::thread::sleep(Duration::from_millis(100));
            });
        })
        .unwrap();

    for _ in 0..2 {
        let outcome = completed_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("parallel sibling reentry should fail without deadlocking");
        assert_worker_rejected(&outcome.unwrap_err());
    }
}

#[test]
#[cfg(feature = "native")]
fn shared_context_work_is_not_mistaken_for_backend_reentry() {
    let context = Arc::new(CpuContext::with_threads(2).unwrap());
    let backend = CpuBackend::from_context(Arc::clone(&context));
    let nested = backend.clone();

    let outcome = backend
        .install(|| {
            let (completed_tx, completed_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                completed_tx
                    .send(context.install(|| nested.install(|| ())))
                    .unwrap();
            });
            completed_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("shared-context work should fail without deadlocking")
        })
        .unwrap();

    // Work on the shared context's pool is still inside the backend's
    // execution, so it cannot bypass backend exclusion.
    assert_worker_rejected(&outcome.unwrap_err());
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn shared_execution_scope_is_cleared_after_panic() {
    let backend = CpuBackend::with_threads(2).unwrap();

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = backend.install(|| panic!("forced nested execution panic"));
    }));

    assert!(panic.is_err());
    assert_eq!(backend.install(|| 17_u32).unwrap(), 17);
}

#[test]
#[cfg(all(feature = "native", any(target_os = "linux", target_os = "android")))]
fn nested_clone_tensor_operation_is_rejected_in_a_managed_scope() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let mut nested = backend.clone();
    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let mut inner_ran = false;

    let nested_entry = backend
        .with_backend_session(|_| {
            nested.with_backend_session(|__s| {
                inner_ran = true;
                __s.add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            })
        })
        .unwrap();

    assert!(
        matches!(
            nested_entry,
            Err(SessionEntryError::Reentered {
                backend: CPU_BACKEND
            })
        ),
        "{nested_entry:?}"
    );
    assert!(!inner_ran);
    // The outer session released its permit; the backend is reusable.
    let sum = backend
        .with_backend_session(|__s| {
            __s.add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0]);
}

#[test]
#[cfg(feature = "blas")]
fn nested_provider_session_is_rejected() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut nested = backend.clone();
    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();

    let nested_result = backend
        .with_backend_session(|_| {
            nested.with_backend_session(|__s| {
                __s.add_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
            })
        })
        .unwrap();

    assert!(
        matches!(
            nested_result,
            Err(SessionEntryError::Reentered {
                backend: CPU_BACKEND
            })
        ),
        "{nested_result:?}"
    );
}

#[test]
#[cfg(all(feature = "native", feature = "blas"))]
fn parallel_rayon_siblings_cannot_bypass_provider_exclusion() {
    let outer = CpuBackend::with_threads(2).unwrap();
    let provider = CpuBackend::with_threads(1).unwrap();
    let first = provider.clone();
    let second = provider;
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();

    outer
        .install(|| {
            rayon::scope(|scope| {
                for nested in [first, second] {
                    let completed_tx = completed_tx.clone();
                    scope.spawn(move |_| {
                        completed_tx.send(nested.install(|| ())).unwrap();
                    });
                }
                std::thread::sleep(Duration::from_millis(100));
            });
        })
        .unwrap();

    for _ in 0..2 {
        let outcome = completed_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("provider sibling reentry should fail without deadlocking");
        assert_reentered(&outcome.unwrap_err());
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

    let len_inside_pool = with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|context, pool| {
            assert_eq!(context.thread_budget().get(), 1);
            <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0, 3.0, 4.0]);
            Ok(pool.len())
        })
    })
    .unwrap();

    assert_eq!(len_inside_pool, 1);
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
}

#[test]
fn linalg_pool_acquire_then_panic_replenishes_buffer_but_reports_poison() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|_, pool| {
            <f64 as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(backend.buffer_pool_len().unwrap(), 1);
    assert_eq!(
        backend.buffer_pool_stats().unwrap().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = with_cpu_session(&mut backend, |cpu| {
            cpu.with_linalg_pool::<()>(|_, pool| {
                let _in_flight = pool.acquire_with_capacity::<f64>(1024);
                assert_eq!(pool.retained_capacity_bytes(), 0);
                panic!("forced panic after pool acquisition");
            })
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
fn uninit_output_partial_write_then_panic_discards_without_replenishment() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|_, pool| {
            <bool as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
            Ok(())
        })
    })
    .unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = with_cpu_session(&mut backend, |cpu| {
            cpu.with_linalg_pool::<()>(|_, pool| {
                let mut output = crate::PooledUninitOutput::<bool>::new(pool, vec![1024]).unwrap();
                output.as_uninit_bytes_mut()[0].write(1);
                panic!("forced panic after a partial uninitialized output write");
            })
        });
    }));

    assert!(result.is_err());
    let resources = backend.engine.resources.lock().unwrap_err().into_inner();
    assert_eq!(resources.buffers.len(), 0);
    assert_eq!(resources.buffers.stats().capacity_bytes, 0);
}

#[test]
fn cached_dot_dispatch_reports_dtype_mismatches() {
    let mut backend = CpuBackend::new();
    let mut cache = gemm::GemmAnalysisCache::default();
    let lhs =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let rhs =
        Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let config = DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    let dot_error = backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.dot_general_cached(Some(0), &lhs, &rhs, &config)
        })
        .unwrap();
    assert!(matches!(
        dot_error,
        Err(crate::Error::Validation {
            op: "dot_general",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let dot_conj_error = backend
        .with_backend_session_cached(&mut cache, |__s| {
            __s.dot_general_with_conj_cached(Some(1), &lhs, &rhs, &config, true, false)
        })
        .unwrap();
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
fn fallible_backend_construction_preserves_topology_error_category() {
    let source = CpuTopologyError::InvalidCpuList {
        list: "not-a-cpu".to_owned(),
        reason: "component is not a CPU number",
    };

    let error = resolve_discovered_topology(Err(source)).unwrap_err();
    match error {
        CpuPlacementError::TopologyDiscovery {
            requested: CpuPlacement::Auto,
            source: CpuTopologyError::InvalidCpuList { list, .. },
        } => assert_eq!(list, "not-a-cpu"),
        other => panic!("unexpected placement error: {other:?}"),
    }
}

#[cfg(all(
    feature = "native",
    not(any(target_os = "linux", target_os = "android"))
))]
#[test]
fn explicit_placement_reports_engine_construction_error_when_unsupported() {
    let error = CpuBackend::builder()
        .unwrap()
        .threads(1)
        .unwrap()
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        CpuBackendError::Placement {
            source: CpuPlacementError::EngineConstruction { .. },
            ..
        }
    ));
    assert!(error.to_string().contains("unsupported on this platform"));
}

#[test]
fn elementwise_into_fallback_returns_its_staged_result_to_the_pool() {
    let mut buffers = BufferPool::new();
    let ctx = ExecContext::default();
    // The fallback (taken when the one-shot kernel does not apply) stages the
    // sum before copying it into `out`.
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    assert_eq!(buffers.len(), 0);

    elementwise_read_into_fallback_with_pool(
        &mut buffers,
        &ctx,
        ElementwiseReadOp::Add,
        &[TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(buffers.len(), 1, "the staged result must be reclaimed");
}
