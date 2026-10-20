#![cfg(feature = "cuda")]

//! Consumer-copy contract (#1945 U4-2a).
//!
//! An engine publishes a value it produced and later copies that value into a fresh device tensor of
//! itself or of another engine on the same device (the explicit transfer between engines, #2206),
//! ordered behind the producer's frontier and running on the consumer's own stream. These tests are
//! ignored by default; explicitly running them requires a CUDA device.

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
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
fn a_value_transfers_to_another_engine_and_the_producer_moves_on_at_once() {
    let mut producer = backend();
    let producer_runtime = runtime_of(&producer);
    let mut consumer = backend();
    let consumer_runtime = runtime_of(&consumer);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&producer_runtime, &vector(&data)).expect("seed");

    // A chain of dependent steps keeps the producer busy, so the transfer is submitted while its
    // source is still being written.
    let mut value = produced_value(&mut producer, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut producer, &value);
    }
    let publication = producer.publish(value).expect("publish");
    let pending = publication.copy_into(&mut consumer).expect("copy");

    // The producer goes on immediately: it drops its inputs and allocates and overwrites new
    // values of the same size, which would reuse the source memory if the transfer did not own it.
    drop(seed);
    let scratch_seed =
        upload_tensor(&producer_runtime, &vector(&values(CHAINED_ELEMENTS))).expect("scratch seed");
    for _ in 0..8 {
        let scratch = produced_value(&mut producer, &scratch_seed);
        drop(scratch);
    }

    // The consumer uses the destination on its own stream.
    let received = pending.wait().expect("wait");
    let doubled = consumer
        .add_read(
            TensorRead::from_tensor(&received),
            TensorRead::from_tensor(&received),
        )
        .expect("consumer add");
    let host = download_tensor(&consumer_runtime, &doubled).expect("read back");
    let expected: Vec<f64> = data
        .iter()
        .map(|value| value * 2f64.powi(CHAINED_STEPS as i32 + 1))
        .collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_transfer_outlives_its_producer_engine() {
    let mut producer = backend();
    let producer_runtime = runtime_of(&producer);
    let mut consumer = backend();
    let consumer_runtime = runtime_of(&consumer);
    let data = values(1024);
    let seed = upload_tensor(&producer_runtime, &vector(&data)).expect("seed");

    let value = produced_value(&mut producer, &seed);
    let publication = producer.publish(value).expect("publish");
    let pending = publication.copy_into(&mut consumer).expect("copy");
    // The producing engine goes away while the copy may still run; the transfer owns the source.
    drop(seed);
    drop(producer);

    let received = pending.wait().expect("wait");
    let host = download_tensor(&consumer_runtime, &received).expect("read back");
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_forgotten_transfer_leaks_instead_of_releasing_memory_in_use() {
    let mut producer = backend();
    let producer_runtime = runtime_of(&producer);
    let mut consumer = backend();
    let data = values(1024);
    let seed = upload_tensor(&producer_runtime, &vector(&data)).expect("seed");

    let value = produced_value(&mut producer, &seed);
    let publication = producer.publish(value).expect("publish");
    let pending = publication.copy_into(&mut consumer).expect("copy");
    // Forgetting the handle keeps the source and destination allocated forever; nothing is freed
    // while the copy may still use it, and both engines still work afterwards.
    std::mem::forget(pending);
    let again = produced_value(&mut producer, &seed);
    let host = download_tensor(&producer_runtime, &again).expect("read back");
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
    consumer.synchronize().expect("consumer synchronize");
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_cross_engine_copy_orders_the_source_write_and_the_destination_read() {
    let mut producer = backend();
    let producer_runtime = runtime_of(&producer);
    let mut consumer = backend();
    let consumer_runtime = runtime_of(&consumer);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&producer_runtime, &vector(&data)).expect("seed");

    // The source is still being written when the copy is published and submitted.
    let mut value = produced_value(&mut producer, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut producer, &value);
    }
    let mut publication = producer.publish(value).expect("publish");
    assert!(
        !publication.frontier_ready().expect("frontier query"),
        "the source is still being written when the cross-engine copy is submitted"
    );
    let pending = publication
        .copy_into(&mut consumer)
        .expect("cross-engine copy");

    // The producer reuses its source memory at once. The copy owns the source until its completion,
    // so this overwrite cannot race the destination write.
    drop(seed);
    let scratch_seed =
        upload_tensor(&producer_runtime, &vector(&values(CHAINED_ELEMENTS))).expect("scratch seed");
    for _ in 0..8 {
        drop(produced_value(&mut producer, &scratch_seed));
    }

    // Only `wait` hands out the destination, after the copy has written it; the consumer then reads
    // it on its own stream, ordered behind the copy's completion event.
    let received = pending.wait().expect("wait");
    let doubled = consumer
        .add_read(
            TensorRead::from_tensor(&received),
            TensorRead::from_tensor(&received),
        )
        .expect("consumer reads the destination");

    let host = download_tensor(&consumer_runtime, &doubled).expect("read back");
    let expected: Vec<f64> = data
        .iter()
        .map(|value| value * 2f64.powi(CHAINED_STEPS as i32 + 1))
        .collect();
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}
