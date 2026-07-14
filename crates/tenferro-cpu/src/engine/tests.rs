use super::*;
use crate::{process_cpu_affinity, CpuSet, ResolvedCpuPlacement};

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
