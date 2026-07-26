use super::*;
#[cfg(target_os = "linux")]
use crate::{process_cpu_affinity, CpuSet};
use crate::{
    CpuDomainId, CpuDomainOwnership, CpuId, CpuPlacementGuarantee, ExternalCpuDomain,
    ResolvedCpuPlacement,
};

#[cfg(target_os = "linux")]
#[test]
fn engine_caps_workers_to_its_cpu_domain_and_owns_resources() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: selected.clone(),
    };
    let engine =
        CpuEngine::new_managed(CpuDomainId::new(0), placement.clone(), usize::MAX, 0).unwrap();

    assert_eq!(engine.domain().thread_budget().get(), selected.len());
    assert_eq!(engine.placement(), &placement);
    assert_eq!(engine.domain().id(), CpuDomainId::new(0));
    assert_eq!(engine.domain().ownership(), CpuDomainOwnership::Managed);
    assert_eq!(
        engine.domain().placement_guarantee(),
        CpuPlacementGuarantee::ExactDeclared
    );
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
    let engine = CpuEngine::from_context(
        CpuDomainId::new(3),
        placement.clone(),
        Arc::clone(&context),
        4096,
    );

    assert_eq!(engine.placement(), &placement);
    assert_eq!(engine.domain().thread_budget().get(), 1);
    assert_eq!(Arc::strong_count(&context), 2);
    assert_eq!(engine.domain().id(), CpuDomainId::new(3));
    assert_eq!(
        engine.domain().placement_guarantee(),
        CpuPlacementGuarantee::AdvisoryDeclared
    );
    let resources = engine.resources.lock().unwrap();
    assert_eq!(resources.buffers.max_retained_capacity_bytes(), 4096);
    assert_eq!(resources.gemm_analysis_cache.capacity(), 1024);
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[test]
fn engine_new_reports_unsupported_worker_affinity() {
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: crate::CpuSet::singleton(CpuId::new(0)),
    };

    let error = CpuEngine::new_managed(CpuDomainId::new(0), placement, 1, 0).unwrap_err();

    assert!(matches!(error, CpuContextError::WorkerPinning { .. }));
    assert!(error.to_string().contains("unsupported on this platform"));
}

#[test]
fn external_engine_moves_the_resource_domain_without_a_staging_context() {
    let placement = ResolvedCpuPlacement::AllAllowed {
        cpus: crate::CpuSet::singleton(CpuId::new(0)),
    };
    let context = Arc::new(CpuContext::with_threads(1).unwrap());
    let external = ExternalCpuDomain::new(
        CpuDomainId::new(9),
        placement.clone(),
        context,
        std::num::NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .unwrap();

    let engine = CpuEngine::from_external(external, 2048);

    assert_eq!(engine.domain().id(), CpuDomainId::new(9));
    assert_eq!(engine.placement(), &placement);
    assert_eq!(
        engine.domain().ownership(),
        CpuDomainOwnership::ExternalManaged
    );
    assert_eq!(engine.domain().thread_budget().get(), 1);
    assert_eq!(
        engine
            .resources
            .lock()
            .unwrap()
            .buffers
            .max_retained_capacity_bytes(),
        2048
    );
}
