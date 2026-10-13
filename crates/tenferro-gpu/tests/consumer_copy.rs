#![cfg(feature = "cuda")]

//! Consumer-copy contract (#1945 U4-2a).
//!
//! A held session publishes a value it produced; a **second** held session on the same runtime instance
//! copies that value into its own device tensor, ordered behind the producer's frontier and running on
//! the consumer's captured stream. These tests are ignored by default; explicitly running them requires
//! a CUDA device.

use cubecl::stream_id::StreamId;
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime,
    PublicationError,
};
use tenferro_tensor::{Tensor, TensorRead};

/// Size of the chained producer the ordering test uses, and how many dependent steps it runs.
const CHAINED_ELEMENTS: usize = 16 * 1024 * 1024;
const CHAINED_STEPS: usize = 64;

fn backend() -> CudaBackend {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn runtime_of(backend: &CudaBackend) -> CudaRuntime {
    backend.runtime().clone()
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

/// Produce `seed + seed` through a held session.
fn produced_value(session: &mut tenferro_gpu::cuda::CudaHeldSession<'_>, seed: &Tensor) -> Tensor {
    session
        .with_session(|view| {
            view.add_read(TensorRead::from_tensor(seed), TensorRead::from_tensor(seed))
        })
        .expect("held callback")
        .expect("held add")
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_published_value_copies_into_a_second_session_on_another_stream() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    // The producer runs on the ambient logical stream and publishes its value.
    let publication = {
        let mut producer = backend.open_session().expect("open producer");
        let value = produced_value(&mut producer, &seed);
        // SAFETY: the value is produced through this session, with no other outstanding write.
        let publication = unsafe { producer.publish(value) }.expect("publish");
        producer.close().expect("close producer");
        publication
    };

    // The consumer session is opened on an explicitly selected, different logical stream, so the copy
    // is genuinely cross-stream rather than two names for the same ambient stream.
    let consumer_stream = StreamId { value: 1 };
    let copied = consumer_stream.executes(|| {
        let mut consumer = backend.open_session().expect("open consumer");
        let pending = publication.copy_into(&mut consumer).expect("copy");
        let bytes = pending.bytes();
        let value = pending.wait().expect("wait");
        consumer.close().expect("close consumer");
        (bytes, value)
    });

    let (bytes, value) = copied;
    assert_eq!(bytes, 32 * 8);
    let host = download_tensor(&runtime, &value).expect("read back");
    let mut expected = data.clone();
    for value in expected.iter_mut() {
        *value *= 2.0;
    }
    assert_eq!(host.as_slice::<f64>().expect("values"), expected.as_slice());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_copy_behind_a_pending_frontier_completes_with_the_produced_bytes() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let mut producer = backend.open_session().expect("open producer");
    // A chain of dependent steps leaves the device with queued work, so the frontier is genuinely
    // pending when the copy is submitted.
    let mut value = produced_value(&mut producer, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut producer, &value);
    }
    // SAFETY: produced through this session, with no other outstanding write.
    let mut publication = unsafe { producer.publish(value) }.expect("publish");
    producer.close().expect("close producer");
    assert!(
        !publication.frontier_ready().expect("frontier query"),
        "the producer must still be running when the copy is submitted"
    );

    let mut consumer = backend.open_session().expect("open consumer");
    let copied = publication.copy_into(&mut consumer).expect("copy");
    let value = copied.wait().expect("wait");
    consumer.close().expect("close consumer");

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
    let mut producer_backend = backend();
    let runtime = runtime_of(&producer_backend);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let publication = {
        let mut producer = producer_backend.open_session().expect("open producer");
        let value = produced_value(&mut producer, &seed);
        // SAFETY: produced through this session, with no other outstanding write.
        let publication = unsafe { producer.publish(value) }.expect("publish");
        producer.close().expect("close producer");
        publication
    };

    // A second backend on the same device is a *different* runtime instance, so it is not authorized
    // for this publication.
    let mut other_backend = backend();
    let mut consumer = other_backend.open_session().expect("open consumer");
    let error = publication.copy_into(&mut consumer).expect_err("refused");
    consumer.close().expect("close consumer");

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
