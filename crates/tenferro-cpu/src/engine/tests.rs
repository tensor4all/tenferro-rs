use super::*;
#[cfg(target_os = "linux")]
use crate::{process_cpu_affinity, CpuSet};
use crate::{CpuId, ResolvedCpuPlacement};

#[cfg(target_os = "linux")]
#[test]
fn engine_caps_workers_to_its_cpu_domain_and_owns_resources() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: selected.clone(),
    };
    let engine = CpuEngine::new(placement.clone(), usize::MAX, 0).unwrap();

    assert_eq!(engine.context().num_threads(), selected.len());
    assert_eq!(engine.placement(), &placement);
    let resources = engine.resources.lock().unwrap();
    assert_eq!(resources.buffers.max_retained_capacity_bytes(), 0);
    assert_eq!(resources.gemm_analysis_cache.capacity(), 1024);
}

#[test]
fn engine_from_context_preserves_placement_context_and_resources() {
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: crate::CpuSet::singleton(CpuId::new(0)),
    };
    let context = Arc::new(CpuContext::with_threads(1).unwrap());
    let engine = CpuEngine::from_context(placement.clone(), Arc::clone(&context), 4096);

    assert_eq!(engine.placement(), &placement);
    assert_eq!(engine.context().num_threads(), 1);
    assert!(Arc::ptr_eq(&engine.context_arc(), &context));
    let resources = engine.resources.lock().unwrap();
    assert_eq!(resources.buffers.max_retained_capacity_bytes(), 4096);
    assert_eq!(resources.gemm_analysis_cache.capacity(), 1024);
}

#[test]
fn engine_routes_dot_general_through_its_runtime_context() {
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: crate::CpuSet::singleton(CpuId::new(0)),
    };
    let engine = CpuEngine::from_context(
        placement,
        Arc::new(CpuContext::with_threads(1).unwrap()),
        4096,
    );
    let providers = crate::CpuProviderBundle::builder(crate::CpuBackendKind::default_compiled())
        .build()
        .unwrap();
    let lhs = crate::Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = crate::Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut output = crate::Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    let mut cache = crate::gemm::GemmAnalysisCache::default();
    engine
        .execute_dot_general_in_scope(
            &providers,
            &mut cache,
            None,
            crate::TensorRead::from_tensor(&lhs),
            crate::TensorRead::from_tensor(&rhs),
            &tenferro_tensor::DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
            tenferro_tensor::DotGeneralAccumulation::overwrite(tenferro_tensor::DType::F64)
                .unwrap(),
            crate::TensorWrite::from_tensor(&mut output),
        )
        .unwrap();
    assert_eq!(output.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[test]
fn engine_new_reports_unsupported_worker_affinity() {
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: crate::CpuSet::singleton(CpuId::new(0)),
    };

    let error = CpuEngine::new(placement, 1, 0).unwrap_err();

    assert!(matches!(error, CpuContextError::WorkerPinning { .. }));
    assert!(error.to_string().contains("unsupported on this platform"));
}
