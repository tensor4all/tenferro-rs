#![cfg(feature = "cuda-next")]

//! The new engine's runtime surface: its prepared-program target and the caches the runtime lends
//! it (issue #2208 U5/U6). The event-domain driver itself needs an [`EventDomainId`], which only
//! a registered runtime mints, so its ordering is exercised where the registration lands.

use tenferro_gpu::cuda_next::{gpu_available, CudaDeviceId, CudaEngine};
use tenferro_runtime::ExecutionEngine;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

#[test]
fn the_engine_reports_the_cuda_target_and_its_allocation_domain() {
    let Some(engine) = engine() else { return };
    let target = engine.target();
    assert_eq!(target.engine_id().as_str(), "tenferro.cuda.default.v1");
    assert_eq!(
        target.allocation_domain(),
        Some(engine.runtime().allocation_domain_id()),
        "the target names the runtime's own allocation domain"
    );
    // The same engine reports the same target twice, and a second engine on the same device has
    // its own allocation domain.
    assert_eq!(
        engine.target().allocation_domain(),
        target.allocation_domain()
    );
    let other = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("second engine");
    assert_ne!(
        other.target().allocation_domain(),
        target.allocation_domain()
    );
}

#[test]
fn the_runtime_lends_and_takes_back_its_extension_caches() {
    let Some(mut engine) = engine() else { return };
    let caches = engine.take_extension_caches();
    // Taking the store twice yields an empty one the second time: the runtime holds it while an
    // extension runs.
    let again = engine.take_extension_caches();
    engine.restore_extension_caches(again);
    engine.restore_extension_caches(caches);
    assert!(engine.as_any_mut().is::<CudaEngine>());
}
