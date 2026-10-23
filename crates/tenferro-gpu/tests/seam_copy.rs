#![cfg(feature = "cuda-next")]

//! U1 seam cross-engine copy evidence: a value transfers between two runtimes on one
//! device, the producer moves on immediately, and the copy outlives its producer
//! (issue #2208 U1).

use tenferro_gpu::cuda_next::{
    download_pending, download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
    PinnedHostBuffer,
};
use tenferro_tensor::Tensor;

fn engines() -> Option<(CudaEngine, CudaEngine)> {
    if !gpu_available() {
        return None;
    }
    let first = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("first engine");
    let second = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("second engine");
    Some((first, second))
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.25 + 1.0).collect()
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

fn as_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[test]
fn a_value_transfers_to_another_engine_and_the_producer_moves_on() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let data = values(4096);
    let value = upload_tensor(producer.runtime(), &vector(&data)).expect("source value");

    let copied = producer
        .runtime()
        .copy_to(consumer.runtime(), &value)
        .expect("cross-engine copy");
    // The producer may move on at once: the copy is ordered on both streams.
    drop(value);
    let back = download_tensor(consumer.runtime(), &copied).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice()
    );
}

#[test]
fn a_copy_outlives_its_producer_engine() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let data = values(2048);
    let value = upload_tensor(producer.runtime(), &vector(&data)).expect("source value");
    let copied = producer
        .runtime()
        .copy_to(consumer.runtime(), &value)
        .expect("cross-engine copy");
    drop(value);
    // Dropping the producer synchronises its stream; the copied value belongs to the
    // consumer and remains readable.
    drop(producer);
    let back = download_tensor(consumer.runtime(), &copied).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice()
    );
}

#[test]
fn a_cross_engine_copy_orders_the_source_write_and_the_destination_read() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let data = values(8192);
    let value = upload_tensor(producer.runtime(), &vector(&data)).expect("source value");
    let copied = producer
        .runtime()
        .copy_to(consumer.runtime(), &value)
        .expect("cross-engine copy");

    // A pending download on the consumer observes the copied bytes, so the copy's
    // ordering on the destination stream is what makes the read valid.
    let buffer = PinnedHostBuffer::new(consumer.runtime(), data.len() * 8).expect("pinned");
    let (_, filled) = download_pending(consumer.runtime(), copied, buffer)
        .expect("pending download")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
}

#[test]
fn a_copy_between_one_runtime_and_itself_is_refused() {
    let Some((engine, _other)) = engines() else {
        return;
    };
    let data = values(64);
    let value = upload_tensor(engine.runtime(), &vector(&data)).expect("value");
    let identical = engine.runtime().clone();
    assert!(engine.runtime().copy_to(&identical, &value).is_err());
}

#[test]
fn a_host_value_cannot_be_copied_between_engines() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let host = vector(&values(16));
    assert!(producer
        .runtime()
        .copy_to(consumer.runtime(), &host)
        .is_err());
}

#[test]
fn the_source_allocation_may_be_reused_immediately() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let data = values(4096);
    let value = upload_tensor(producer.runtime(), &vector(&data)).expect("source value");
    let copied = producer
        .runtime()
        .copy_to(consumer.runtime(), &value)
        .expect("cross-engine copy");

    // The producer moves on: dropping the source returns its allocation to the runtime's
    // pool, and a fresh upload of the same size reuses it. The copied value must be
    // unaffected, because the copy was ordered before the reuse.
    drop(value);
    let replacement = values(4096);
    let _reused = upload_tensor(producer.runtime(), &vector(&replacement)).expect("replacement");

    let back = download_tensor(consumer.runtime(), &copied).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice()
    );
}

#[test]
fn a_consumer_copy_survives_a_dropped_producer() {
    let Some((producer, consumer)) = engines() else {
        return;
    };
    let data = values(1024);
    let value = upload_tensor(producer.runtime(), &vector(&data)).expect("source value");
    let copied = producer
        .runtime()
        .copy_to(consumer.runtime(), &value)
        .expect("cross-engine copy");
    drop(value);
    drop(producer);

    // A pending download on the consumer still observes the copied bytes.
    let buffer = PinnedHostBuffer::new(consumer.runtime(), 1024 * 8).expect("pinned");
    let (_, filled) = download_pending(consumer.runtime(), copied, buffer)
        .expect("pending download")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
}
