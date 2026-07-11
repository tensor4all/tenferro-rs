use std::time::Duration;

#[cfg(feature = "cpu-tblis-linked")]
use num_complex::{Complex32, Complex64};

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
fn explicit_dot_general_provider_records_selection() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();

    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    assert_eq!(backend.dot_general_provider(), DotGeneralProvider::Base);

    backend.set_dot_general_provider(DotGeneralProvider::TblisIfAvailable);
    assert_eq!(
        backend.dot_general_provider(),
        DotGeneralProvider::TblisIfAvailable
    );
    assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
}

#[test]
#[cfg(feature = "cpu-tblis-linked")]
fn tblis_dot_general_matches_column_major_matmul() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisRequired);
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
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisRequired);
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
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisRequired);
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
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisRequired);
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
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisIfAvailable);
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
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    backend.set_dot_general_provider(DotGeneralProvider::TblisIfAvailable);
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
        crate::Error::InvalidConfig {
            op: "CpuBackend::with_threads_and_kind",
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
        ctx: Arc::new(CpuContext::with_threads(1).unwrap()),
        buffers: BufferPool::new(),
        kind: CpuBackendKind::Blas,
        dot_general_provider: DotGeneralProvider::Base,
    };
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
    <f64 as PoolScalar>::pool_release(&mut backend.buffers, Vec::with_capacity(1024));
    assert_eq!(backend.buffer_pool_len(), 1);
    assert_eq!(
        backend.buffers.retained_capacity_bytes(),
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
        backend.buffers.retained_capacity_bytes(),
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
    assert!(CpuBackend::with_threads(0).is_err());
}
