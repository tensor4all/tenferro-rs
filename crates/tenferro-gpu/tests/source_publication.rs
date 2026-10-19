#![cfg(feature = "cuda")]

//! Source publication contract (#1945 U4 item 1).
//!
//! A held session publishes a value it produced; the publication owns it, retains its allocation and
//! keeps the producer frontier, so the value can be consumed after the session and even the backend
//! handle are gone. There is deliberately no public read accessor: a device consumer would need the
//! dependency contract, which this revision does not specify. These tests are ignored by default;
//! explicitly running them requires a CUDA device.

use tenferro_gpu::cuda::{
    gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime, PinnedHostBuffer,
    PublicationAdmissionError, PublicationError,
};
use tenferro_tensor::{Tensor, TensorRead};

/// Size of the chained producer the frontier test uses, and how many dependent steps it runs.
///
/// A single large kernel is not enough: the dominating cost of a first launch is its compilation, and the
/// kernel itself then finishes inside the host round trips that publishing performs before its first
/// poll. A chain of dependent steps leaves the device with milliseconds of queued work, so the frontier
/// is still pending when it is first observed.
const CHAINED_ELEMENTS: usize = 16 * 1024 * 1024;
const CHAINED_STEPS: usize = 64;

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
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

fn from_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect()
}

/// Produce `seed + seed` through a held session, so the caller can assert the publication contract.
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
fn a_published_value_survives_its_session() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    // SAFETY: the value is produced through this session, and no other write to it is outstanding.
    let publication = {
        let mut session = backend.open_session().expect("open");
        let value = produced_value(&mut session, &seed);
        let publication = unsafe { session.publish(value) }.expect("publish");
        session.close().expect("close");
        publication
    };

    assert_eq!(publication.bytes(), 32 * 8);
    let buffer = PinnedHostBuffer::new(&runtime, 32 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    let doubled: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(from_bytes(filled.as_slice()), doubled);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_publication_keeps_its_allocation_when_the_backend_handle_is_dropped() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let mut session = backend.open_session().expect("open");
    // A chain of dependent steps leaves the device with queued work, so the handles are released while
    // the producer's frontier is still pending.
    let mut value = produced_value(&mut session, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut session, &value);
    }
    // SAFETY: produced through this session, with no other outstanding write.
    let mut publication = unsafe { session.publish(value) }.expect("publish");
    drop(session);
    drop(backend);
    assert!(
        !publication.frontier_ready().expect("frontier query"),
        "the producer must still be running when its handles are gone"
    );
    drop(seed);

    // The allocation survived both handles: the publication still owns it, and only the copy waits.
    let buffer = PinnedHostBuffer::new(&runtime, CHAINED_ELEMENTS * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    let expected: Vec<f64> = data
        .iter()
        .map(|value| value * 2f64.powi(CHAINED_STEPS as i32))
        .collect();
    assert_eq!(from_bytes(filled.as_slice()), expected);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_publication_reports_a_pending_frontier() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(CHAINED_ELEMENTS);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");

    let mut session = backend.open_session().expect("open");
    // A chain of dependent steps through the session: the device is left with a queue of work, so the
    // frontier is genuinely pending when it is first polled.
    let mut value = produced_value(&mut session, &seed);
    for _ in 1..CHAINED_STEPS {
        value = produced_value(&mut session, &value);
    }
    // SAFETY: produced through this session, with no other outstanding write.
    let mut publication = unsafe { session.publish(value) }.expect("publish");

    // The frontier is recorded once, so a single query settles it: a completed frontier cannot
    // become pending again, and a pending one is what the chain guarantees.
    assert!(
        !publication.frontier_ready().expect("frontier query"),
        "a frontier recorded behind a queued chain must be pending"
    );
    publication.await_frontier().expect("await frontier");

    let buffer = PinnedHostBuffer::new(&runtime, CHAINED_ELEMENTS * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    let expected: Vec<f64> = data
        .iter()
        .map(|value| value * 2f64.powi(CHAINED_STEPS as i32))
        .collect();
    assert_eq!(from_bytes(filled.as_slice()), expected);
}

#[test]
#[ignore = "requires a CUDA device"]
fn dropping_a_publication_detaches_and_retires_it() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let seed = upload_tensor(&runtime, &vector(&values(1024))).expect("seed");

    let before = runtime.pending_transfer_retirements();
    {
        let mut session = backend.open_session().expect("open");
        let value = produced_value(&mut session, &seed);
        // SAFETY: produced through this session, with no other outstanding write.
        let publication = unsafe { session.publish(value) }.expect("publish");
        session.close().expect("close");
        drop(publication);
    }
    assert!(
        runtime.pending_transfer_retirements() > before,
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
#[ignore = "requires a CUDA device"]
fn forgetting_a_publication_leaks_it_in_place() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let seed = upload_tensor(&runtime, &vector(&values(1024))).expect("seed");
    let mut session = backend.open_session().expect("open");
    let value = produced_value(&mut session, &seed);

    let before = runtime.pending_transfer_retirements();
    // SAFETY: produced through this session, with no other outstanding write.
    let publication = unsafe { session.publish(value) }.expect("publish");
    std::mem::forget(publication);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        before,
        "a forgotten publication is leaked in place rather than detached"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), before);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_rejected_publication_returns_the_value() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let host = vector(&values(16));

    // A host tensor is not a published device value: the value comes back unchanged.
    let mut session = backend.open_session().expect("open");
    // SAFETY: nothing is produced here; the rejected value is returned below.
    match unsafe { session.publish(host) } {
        Err(PublicationAdmissionError::Rejected(rejected)) => {
            assert_eq!(rejected.value.shape(), &[16]);
        }
        other => panic!("expected a rejected publication, got {other:?}"),
    }
    session.close().expect("close");

    // The session is unaffected.
    let seed = upload_tensor(&runtime, &vector(&values(16))).expect("seed");
    let mut session = backend.open_session().expect("open");
    let value = produced_value(&mut session, &seed);
    // SAFETY: produced through this session, with no other outstanding write.
    let publication = unsafe { session.publish(value) }.expect("publish");
    session.close().expect("close");
    let buffer = PinnedHostBuffer::new(&runtime, 16 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(buffer)
        .expect("published copy")
        .wait()
        .expect("wait");
    assert_eq!(
        from_bytes(filled.as_slice()),
        values(16).iter().map(|v| v * 2.0).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_mismatched_buffer_returns_the_package() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(32);
    let seed = upload_tensor(&runtime, &vector(&data)).expect("seed");
    let mut session = backend.open_session().expect("open");
    let value = produced_value(&mut session, &seed);
    // SAFETY: produced through this session, with no other outstanding write.
    let publication = unsafe { session.publish(value) }.expect("publish");
    session.close().expect("close");

    let short = PinnedHostBuffer::new(&runtime, 32 * 8 + 8).expect("pinned buffer");
    let publication = match publication.download_into(short) {
        Err(PublicationError::Rejected {
            publication,
            buffer,
            ..
        }) => {
            assert_eq!(buffer.len(), 32 * 8 + 8);
            assert_eq!(publication.bytes(), 32 * 8);
            publication
        }
        other => panic!("expected the publication back, got {other:?}"),
    };

    // The rejection destroyed nothing, so the same publication is consumed with a correct buffer -
    // no second publishing step, and no value whose readiness was lost.
    let exact = PinnedHostBuffer::new(&runtime, 32 * 8).expect("pinned buffer");
    let (_, filled) = publication
        .download_into(exact)
        .expect("published copy")
        .wait()
        .expect("wait");
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(from_bytes(filled.as_slice()), expected);
}
