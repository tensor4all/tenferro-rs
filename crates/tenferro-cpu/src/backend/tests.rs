use std::time::Duration;

use super::*;

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
fn default_backend_kind_prefers_blas_when_compiled() {
    let backend = CpuBackend::new();

    #[cfg(feature = "cpu-blas")]
    assert_eq!(backend.kind(), CpuBackendKind::Blas);
    #[cfg(all(not(feature = "cpu-blas"), feature = "cpu-faer"))]
    assert_eq!(backend.kind(), CpuBackendKind::Faer);
}

#[test]
fn explicit_backend_kind_constructor_records_selection() {
    let backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
}

#[test]
#[cfg(feature = "cpu-faer")]
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
    placed.with_linalg_pool(|pool| {
        <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0]);
    });
    assert_eq!(clone.buffer_pool_len(), 1);
}

#[test]
#[cfg(feature = "cpu-faer")]
fn placement_capabilities_follow_public_backend_kind() {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    assert!(backend.supports_placement(CpuPlacement::Auto));
    assert!(backend.supports_placement(CpuPlacement::AllAllowed));
    assert_eq!(
        backend.topology().allowed_cpus(),
        crate::process_cpu_affinity().as_ref().unwrap()
    );
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
#[cfg(feature = "cpu-blas")]
fn explicit_blas_backend_kind_constructor_records_selection() {
    let backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::Blas);
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
        CpuBackendError::Tensor(crate::Error::InvalidConfig {
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
        CpuBackendError::Tensor(crate::Error::InvalidConfig {
            op: "CpuBackend::with_kind",
            ..
        })
    ));

    let mut backend = CpuBackend::compatibility(
        Arc::new(CpuContext::with_threads(1).unwrap()),
        crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        CpuBackendKind::Blas,
    );
    let retained = backend.with_linalg_pool(|pool| {
        <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0]);
        pool.len()
    });
    assert_eq!(retained, 1);
    assert_eq!(backend.buffer_pool_len(), 1);

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
        assert!(matches!(
            err,
            crate::Error::InvalidConfig {
                op: "dot_general",
                ..
            }
        ));
        assert!(err.to_string().contains("blas"));
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

    let len_inside_pool = backend.with_linalg_pool(|pool| {
        <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0, 3.0, 4.0]);
        pool.len()
    });

    assert_eq!(len_inside_pool, 1);
    assert_eq!(backend.buffer_pool_len(), 1);

    #[cfg(feature = "cpu-faer")]
    assert_eq!(backend.linalg_context().num_threads(), 1);
}

#[test]
fn linalg_pool_acquire_then_panic_replenishes_retained_buffer() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend.with_linalg_pool(|pool| {
        <f64 as PoolScalar>::pool_release(pool, Vec::with_capacity(1024));
    });
    assert_eq!(backend.buffer_pool_len(), 1);
    assert_eq!(
        backend.buffer_pool_stats().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_linalg_pool::<()>(|pool| {
            let _in_flight = unsafe { <f64 as PoolScalar>::pool_acquire(pool, 1024) };
            assert_eq!(pool.retained_capacity_bytes(), 0);
            panic!("forced panic after pool acquisition");
        });
    }));

    assert!(result.is_err());
    assert_eq!(backend.buffer_pool_len(), 1);
    assert_eq!(
        backend.buffer_pool_stats().capacity_bytes,
        1024 * std::mem::size_of::<f64>()
    );
}

#[test]
#[cfg(feature = "cpu-faer")]
fn cached_faer_gemm_pool_helper_enters_owned_rayon_pool() {
    let ambient_threads = rayon::current_num_threads();
    let configured_threads = if ambient_threads == 2 { 3 } else { 2 };
    let mut backend =
        CpuBackend::with_threads_and_kind(configured_threads, CpuBackendKind::Faer).unwrap();
    let mut cache = gemm::GemmAnalysisCache::default();

    let seen_threads =
        backend.install_with_pool_and_gemm_cache(&mut cache, |_, _| rayon::current_num_threads());

    assert_eq!(seen_threads, configured_threads);
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
        Err(crate::Error::DTypeMismatch {
            op: "dot_general",
            ..
        })
    ));

    let dot_conj_error =
        backend.dot_general_with_conj_cached(&mut cache, Some(1), &lhs, &rhs, &config, true, false);
    assert!(matches!(
        dot_conj_error,
        Err(crate::Error::DTypeMismatch {
            op: "dot_general",
            ..
        })
    ));
}

#[test]
fn with_threads_rejects_invalid_thread_count() {
    let result: Result<CpuBackend, CpuBackendError> = CpuBackend::with_threads(0);
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        CpuBackendError::Tensor(crate::Error::InvalidConfig {
            op: "CpuBackend::with_threads",
            ..
        })
    ));
}

#[test]
fn backend_error_keeps_placement_failure_typed() {
    let placement = CpuPlacementError::TopologyDiscovery {
        requested: CpuPlacement::Auto,
        backend: CpuBackendKind::Faer,
        source: CpuTopologyError::InvalidCpuList {
            list: "bad".to_owned(),
            reason: "test failure",
        },
    };

    let error = CpuBackendError::placement("CpuBackend::try_new", placement.clone());
    assert_eq!(error.placement_error(), Some(&placement));
}

#[test]
fn fallible_backend_construction_preserves_topology_error_category() {
    let source = CpuTopologyError::InvalidCpuList {
        list: "not-a-cpu".to_owned(),
        reason: "component is not a CPU number",
    };

    let error = resolve_discovered_topology(CpuBackendKind::Faer, Err(source.clone())).unwrap_err();
    assert_eq!(
        error,
        CpuPlacementError::TopologyDiscovery {
            requested: CpuPlacement::Auto,
            backend: CpuBackendKind::Faer,
            source,
        }
    );
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
