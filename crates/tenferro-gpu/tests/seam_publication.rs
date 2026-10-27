#![cfg(feature = "cuda")]

//! U1 seam publication evidence: a published value keeps its allocation, exposes a
//! frontier a reader can wait on, and the drop/forget/rejection semantics (issue #2208
//! U1).

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_pending, upload_tensor, CudaDeviceId, CudaEngine,
    CudaRuntime, PinnedHostBuffer, PublicationAdmissionError,
};
use tenferro_tensor::Tensor;

/// Enough queued work that a checkpoint recorded behind it is genuinely pending.
const QUEUED_BYTES: usize = 96 * 1024 * 1024;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
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

/// Queue `2 x QUEUED_BYTES` of device-to-device work on `runtime`, so a checkpoint
/// recorded right afterwards is pending when first polled. The uploads are submitted
/// as pending transfers and dropped, which detaches them: no host wait happens between
/// the last submission and the checkpoint.
fn queue_work(runtime: &CudaRuntime) {
    let elements = QUEUED_BYTES / 8;
    let zeros = vec![0.0f64; elements];
    for _ in 0..2 {
        let destination = upload_tensor(runtime, &vector(&zeros)).expect("queued destination");
        let mut source = PinnedHostBuffer::new(runtime, QUEUED_BYTES).expect("pinned");
        source.as_mut_slice().fill(7);
        let _ = upload_pending(runtime, source, destination).expect("queued upload");
    }
}

#[test]
fn a_published_value_survives_its_engine() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let data = values(32);
    let value = upload_tensor(&runtime, &vector(&data)).expect("value");

    let publication = engine.publish(value).expect("publish");
    assert_eq!(publication.bytes(), 32 * 8);
    let buffer = PinnedHostBuffer::new(&runtime, 32 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
}

#[test]
fn a_publication_keeps_its_allocation_when_the_engine_is_dropped() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let data = values(4096);
    let value = upload_tensor(&runtime, &vector(&data)).expect("value");

    let mut publication = engine.publish(value).expect("publish");
    // Dropping the engine synchronises its stream, so the frontier is complete
    // afterwards; the publication must still own the allocation.
    drop(engine);
    assert!(publication.frontier_ready().expect("frontier query"));

    let buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
}

#[test]
fn a_publication_frontier_is_monotone_and_awaitable() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let data = values(4096);
    // Queue work first, so the checkpoint may genuinely be pending when it is first
    // polled. A poll also reclaims whatever completed, so the test asserts the property
    // that holds in both cases: readiness is monotone and awaiting always completes.
    let value = upload_tensor(&runtime, &vector(&data)).expect("value");
    queue_work(&runtime);
    let mut publication = engine.publish(value).expect("publish");

    let first = publication.frontier_ready().expect("frontier query");
    publication.await_frontier().expect("await frontier");
    assert!(publication.frontier_ready().expect("frontier query"));
    assert!(
        first || publication.frontier_ready().expect("frontier query"),
        "a completed frontier must report ready"
    );

    let buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
}

#[test]
fn dropping_a_publication_detaches_and_retires_it() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let value = upload_tensor(&runtime, &vector(&values(1024))).expect("value");
    let publication = engine.publish(value).expect("publish");
    drop(publication);
    assert!(
        runtime.pending_transfer_retirements() > 0,
        "a detached publication is retained until its frontier proves completion"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(
        runtime.pending_transfer_retirements(),
        0,
        "a completed publication is released exactly once"
    );
}

#[test]
fn forgetting_a_publication_leaks_it_in_place() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let value = upload_tensor(&runtime, &vector(&values(64))).expect("value");
    let publication = engine.publish(value).expect("publish");
    std::mem::forget(publication);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        0,
        "a forgotten publication is leaked rather than detached"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), 0);
}

#[test]
fn a_rejected_publication_returns_the_value() {
    let Some(mut engine) = engine() else { return };
    let host = vector(&values(16));
    match engine.publish(host) {
        Err(PublicationAdmissionError::Rejected(rejected)) => {
            assert_eq!(rejected.value.shape(), &[16]);
            assert!(!rejected.reason.is_empty());
        }
        other => panic!(
            "expected a pre-admission rejection, got {:?}",
            other.is_ok()
        ),
    }
}

#[test]
fn the_frontier_outlives_a_copy_that_reads_it() {
    let Some(mut engine) = engine() else { return };
    let runtime = engine.runtime().clone();
    let data = values(2048);
    queue_work(&runtime);
    let value = upload_tensor(&runtime, &vector(&data)).expect("value");
    let publication = engine.publish(value).expect("publish");
    // The copy is admitted behind the frontier and completes without waiting on it
    // first.
    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("copy behind the frontier")
        .wait()
        .expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
    let _ = download_tensor; // the download helper stays part of the seam's surface
}
