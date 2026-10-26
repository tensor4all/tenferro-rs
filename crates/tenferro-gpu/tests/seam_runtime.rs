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
    use tenferro_runtime::ExtensionCacheKey;

    let Some(mut engine) = engine() else { return };
    let key = ExtensionCacheKey::new("tenferro.test.cuda.next", "plans", 7);
    let mut caches = engine.take_extension_caches();
    caches.put(key, 42u64, 0);
    assert_eq!(caches.len(), 1);
    engine.restore_extension_caches(caches);

    // The runtime hands the same populated store back.
    let mut restored = engine.take_extension_caches();
    assert_eq!(restored.get::<u64>(&key), Some(&42));
    engine.restore_extension_caches(restored);
    assert!(engine.as_any_mut().is::<CudaEngine>());
}

/// A preparation-only registration mints an event domain without any CUDA machinery, which is
/// enough to exercise the new engine's driver: ordering, the dependency wait and retirement.
fn event_domain(suffix: &str) -> tenferro_runtime::EventDomainId {
    use std::sync::Arc;
    use tenferro_runtime::{
        assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
        EngineRegistrationMetadata, HardwareClassId, PreparationOnlyEngineRegistrationConfig,
        ProviderDeviceIdentity, ProviderId, Runtime, StorageClass,
    };

    let engine_id = EngineId::new(format!("tenferro.test.cuda.next.{suffix}")).expect("engine id");
    let storage =
        StorageClass::new(format!("tenferro.test.cuda.next.storage.{suffix}")).expect("storage");
    let metadata = EngineRegistrationMetadata::new(
        engine_id.clone(),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.cuda.next").expect("provider"),
            format!("target:{suffix}"),
        )
        .expect("device identity"),
        HardwareClassId::new("tenferro.test.cuda.next").expect("hardware class"),
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    );
    let registration = assemble_preparation_only_engine_registration(
        PreparationOnlyEngineRegistrationConfig::new(
            metadata,
            tenferro_runtime::ExecutionContextIdentity::of::<()>(),
        ),
    )
    .expect("preparation-only registration");
    let mut builder = Runtime::builder();
    builder.register_engine(registration).expect("registration");
    let runtime = builder.build().expect("runtime");
    runtime
        .snapshot()
        .expect("snapshot")
        .engine(&engine_id)
        .expect("engine snapshot")
        .event_domain_id()
}

#[test]
fn the_driver_orders_a_launch_behind_its_dependency_and_retires_it() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let Some(engine) = engine() else { return };
    let domain = event_domain("ordering");
    let driver = engine.event_domain_driver();

    let launches = AtomicUsize::new(0);
    let first = {
        let mut run = driver.begin_run(domain).expect("first run");
        let mut launch = || {
            launches.fetch_add(1, Ordering::SeqCst);
            // A real upload on the engine's stream, so the recorded completion covers work.
            engine
                .runtime()
                .upload_bytes(&[1u8, 2, 3, 4])
                .map(|_| ())
                .map_err(|error| {
                    tenferro_runtime::Error::runtime_state(
                        "seam_runtime",
                        tenferro_runtime::ErrorPhase::Execution,
                        error.to_string(),
                    )
                })
        };
        let token = run.enqueue(&[], &mut launch).expect("first enqueue");
        token.wait().expect("the first token completes");
        run.drain().expect("first drain");
        token
    };

    // A second run whose launch depends on the first run's token: the enqueue waits on it, so the
    // launch cannot run before the dependency has completed.
    let second = {
        let mut run = driver.begin_run(domain).expect("second run");
        let mut launch = || {
            launches.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
        let token = run
            .enqueue(std::slice::from_ref(&first), &mut launch)
            .expect("dependent enqueue");
        token.wait().expect("the dependent token completes");
        run.drain().expect("second drain");
        token
    };
    second.wait().expect("the last token completes");
    assert_eq!(launches.load(Ordering::SeqCst), 2);
}
