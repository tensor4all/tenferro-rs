#![cfg(feature = "cuda")]

//! Consumer-copy contract (#1945 U4-2a).
//!
//! An engine publishes a value it produced and later copies that value into a fresh device tensor of
//! the same runtime instance, ordered behind the producer's frontier and running on the engine's own
//! stream; an engine of another runtime instance is refused. These tests are ignored by default;
//! explicitly running them requires a CUDA device.

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
    PublicationError,
};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};

/// Size of the chained producer the ordering test uses, and how many dependent steps it runs.
const CHAINED_ELEMENTS: usize = 16 * 1024 * 1024;
const CHAINED_STEPS: usize = 64;

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn runtime_of(backend: &CudaEngine) -> CudaRuntime {
    backend.runtime().clone()
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

/// Produce `seed + seed` on the engine.
fn produced_value(engine: &mut CudaEngine, seed: &Tensor) -> Tensor {
    engine
        .add_read(TensorRead::from_tensor(seed), TensorRead::from_tensor(seed))
        .expect("engine add")
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_published_value_copies_into_a_fresh_tensor_of_its_engine() {
    let mut engine = backend();
    let runtime = runtime_of(&engine);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let value = produced_value(&mut engine, &seed);
    let publication = engine.publish(value).expect("publish");
    let pending = publication.copy_into(&mut engine).expect("copy");
    let bytes = pending.bytes();
    let value = pending.wait().expect("wait");

    assert_eq!(bytes, 32 * 8);
    let host = download_tensor(&runtime, &value).expect("read back");
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_copy_behind_a_pending_frontier_completes_with_the_produced_bytes() {
    let mut engine = backend();
    let runtime = runtime_of(&engine);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    // A chain of dependent steps leaves the device with queued work, so the frontier is genuinely
    // pending when the copy is submitted.
    let mut value = produced_value(&mut engine, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut engine, &value);
    }
    let mut publication = engine.publish(value).expect("publish");
    assert!(
        !publication.frontier_ready().expect("frontier query"),
        "the producer must still be running when the copy is submitted"
    );

    let copied = publication.copy_into(&mut engine).expect("copy");
    let value = copied.wait().expect("wait");

    let host = download_tensor(&runtime, &value).expect("read back");
    let expected: Vec<f64> = data
        .iter()
        .map(|value| value * 2f64.powi(CHAINED_STEPS as i32))
        .collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_copy_from_another_runtime_instance_is_refused_with_the_publication() {
    let mut producer = backend();
    let runtime = runtime_of(&producer);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let value = produced_value(&mut producer, &seed);
    let publication = producer.publish(value).expect("publish");

    // A second engine on the same device is a *different* runtime instance, so it is not authorized
    // for this publication.
    let mut consumer = backend();
    let error = publication.copy_into(&mut consumer).expect_err("refused");

    let PublicationError::CopyRejected {
        error: reason,
        publication,
    } = error
    else {
        panic!("expected a device-copy rejection");
    };
    assert!(
        reason.to_string().contains("runtime instance"),
        "the refusal must name the runtime mismatch: {reason}"
    );

    // The publication came back unchanged and is still usable: the source value is intact.
    let mut publication = *publication;
    assert_eq!(publication.bytes(), 32 * 8);
    assert!(publication.frontier_ready().expect("frontier query"));
}
