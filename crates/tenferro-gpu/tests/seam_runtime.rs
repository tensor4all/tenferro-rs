#![cfg(feature = "cuda")]

//! The new engine's runtime surface: its prepared-program target and the caches the runtime lends
//! it (issue #2208 U5/U6). The event-domain driver itself needs an [`EventDomainId`], which only
//! a registered runtime mints, so its ordering is exercised where the registration lands.

use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine};
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

#[test]
fn a_prepared_elementwise_program_runs_on_the_new_engine() {
    use tenferro_gpu::cuda::{download_tensor, upload_tensor};
    use tenferro_runtime::{ExecutionEngine, GraphCompiler, Output, Runtime, TracedTensor};
    use tenferro_tensor::{DType, Tensor, TensorRead};

    let Some(mut engine) = engine() else { return };
    let registration =
        tenferro_gpu::cuda::runtime_adapter::cuda_runtime_engine_registration(&engine)
            .expect("registration");
    let mut builder = Runtime::builder();
    builder.register_engine(registration).expect("register");
    let runtime = builder.build().expect("runtime");

    // `x + x`, one elementwise node.
    let input = TracedTensor::input_symbolic_shape(DType::F32, 1).expect("graph input");
    let sum = input.add(&input).expect("graph add");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&sum, &[(&input, DType::F32, &[4])])
        .expect("compile");

    let host =
        Tensor::from_vec_col_major(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).expect("host input");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");
    let target = engine.target();
    let prepared = runtime
        .prepare(&program, &[TensorRead::from_tensor(&device)], &[target])
        .expect("prepare");
    let outputs = runtime
        .run(
            &mut [&mut engine],
            &prepared,
            &[TensorRead::from_tensor(&device)],
        )
        .expect("run");
    let output = outputs
        .into_iter()
        .next()
        .and_then(Output::into_owned)
        .expect("output");
    let values = download_tensor(engine.runtime(), &output).expect("download");
    assert_eq!(
        values.as_slice::<f32>().expect("f32 payload"),
        &[2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn a_prepared_comparison_runs_on_the_new_engine() {
    use tenferro_gpu::cuda::{download_tensor, upload_tensor};
    use tenferro_runtime::{ExecutionEngine, GraphCompiler, Output, Runtime, TracedTensor};
    use tenferro_tensor::{DType, Tensor, TensorRead};

    let Some(mut engine) = engine() else { return };
    let registration =
        tenferro_gpu::cuda::runtime_adapter::cuda_runtime_engine_registration(&engine)
            .expect("registration");
    let mut builder = Runtime::builder();
    builder.register_engine(registration).expect("register");
    let runtime = builder.build().expect("runtime");

    // A comparison's result is a Bool tensor, so the prepared program's output is Bool and its
    // residency check has to reach a Bool allocation.
    let input = TracedTensor::input_symbolic_shape(DType::F32, 1).expect("graph input");
    // Comparing the input with itself needs no second operand: every element is `x < x`, false.
    let compared = input
        .compare(&input, tenferro_tensor::CompareDir::Lt)
        .expect("compare");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&compared, &[(&input, DType::F32, &[4])])
        .expect("compile");

    let host =
        Tensor::from_vec_col_major(vec![4], vec![-1.0f32, 0.5, -2.0, 3.0]).expect("host input");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");
    let target = engine.target();
    let prepared = runtime
        .prepare(&program, &[TensorRead::from_tensor(&device)], &[target])
        .expect("prepare");
    let outputs = runtime
        .run(
            &mut [&mut engine],
            &prepared,
            &[TensorRead::from_tensor(&device)],
        )
        .expect("run");
    let output = outputs
        .into_iter()
        .next()
        .and_then(Output::into_owned)
        .expect("output");
    assert_eq!(output.dtype(), DType::Bool);
    let values = download_tensor(engine.runtime(), &output).expect("download");
    assert_eq!(
        values.as_slice::<bool>().expect("bool payload"),
        &[false, false, false, false]
    );
}
