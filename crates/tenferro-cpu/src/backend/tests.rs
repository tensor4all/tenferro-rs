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
#[cfg(feature = "cpu-blas")]
fn explicit_blas_backend_kind_constructor_records_selection() {
    let backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::Blas);
}

#[test]
fn try_with_threads_and_kind_records_selection_and_validates_threads() {
    let backend =
        CpuBackend::try_with_threads_and_kind(1, CpuBackendKind::default_compiled()).unwrap();
    assert_eq!(backend.num_threads(), 1);
    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());

    let err = match CpuBackend::try_with_threads_and_kind(0, CpuBackendKind::default_compiled()) {
        Ok(_) => panic!("expected invalid thread count to fail"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        crate::Error::InvalidConfig {
            op: "CpuBackend::try_with_threads_and_kind",
            ..
        }
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
        crate::Error::InvalidConfig {
            op: "CpuBackend::with_kind",
            ..
        }
    ));

    let mut backend = CpuBackend {
        ctx: Arc::new(CpuContext::with_threads(1)),
        buffers: BufferPool::new(),
        kind: CpuBackendKind::Blas,
    };
    let retained = backend.with_linalg_pool(|pool| {
        <f64 as PoolScalar>::pool_release(pool, vec![1.0, 2.0]);
        pool.len()
    });
    assert_eq!(retained, 1);
    assert_eq!(backend.buffer_pool_len(), 1);

    let lhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]);
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
    let mut backend = CpuBackend::with_threads(1);

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
#[cfg(feature = "cpu-faer")]
fn cached_faer_gemm_pool_helper_enters_owned_rayon_pool() {
    let ambient_threads = rayon::current_num_threads();
    let configured_threads = if ambient_threads == 2 { 3 } else { 2 };
    let mut backend =
        CpuBackend::try_with_threads_and_kind(configured_threads, CpuBackendKind::Faer).unwrap();
    let mut cache = gemm::GemmAnalysisCache::default();

    let seen_threads =
        backend.install_with_pool_and_gemm_cache(&mut cache, |_, _| rayon::current_num_threads());

    assert_eq!(seen_threads, configured_threads);
}

#[test]
fn cached_dot_dispatch_reports_dtype_mismatches() {
    let mut backend = CpuBackend::new();
    let mut cache = gemm::GemmAnalysisCache::default();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0]));
    let rhs = Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![1.0]));
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
fn with_threads_panics_on_invalid_thread_count() {
    let panic = std::panic::catch_unwind(|| CpuBackend::with_threads(0));

    assert!(panic.is_err());
}
