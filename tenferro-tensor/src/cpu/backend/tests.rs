use std::time::Duration;

use super::*;

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
