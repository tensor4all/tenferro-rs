#![cfg(feature = "cuda")]

//! Owning device→host transfer contract (#1945 U4 item 3).
//!
//! These tests pin the owning pending download: the value and the pinned buffer are moved into the
//! transfer and come back only from `wait`; a rejected submission returns them unchanged; dropping or
//! forgetting the handle detaches observation into the runtime's retirement queue instead of
//! releasing anything. They are ignored by default; explicitly running them requires a CUDA device.

use tenferro_gpu::cuda::{
    download_pending, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
    PinnedHostBuffer, TransferSubmitError,
};
use tenferro_tensor::Tensor;

/// Big enough that a copy of it cannot complete before an immediate poll observes the queue.
const DELAYED_ELEMENTS: usize = 8 * 1024 * 1024;

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
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

fn from_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect()
}

fn runtime_of(backend: &CudaEngine) -> CudaRuntime {
    backend.runtime().clone()
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_download_reports_not_ready_then_returns_the_bytes_and_the_value() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(DELAYED_ELEMENTS);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");

    let buffer = PinnedHostBuffer::new(&runtime, DELAYED_ELEMENTS * 8).expect("pinned buffer");
    let mut pending = download_pending(&runtime, device, buffer).expect("pending download");

    // The copy of a payload this size cannot have completed before this immediate poll.
    let mut observed_not_ready = false;
    for _ in 0..8 {
        if !pending.is_ready().expect("event query") {
            observed_not_ready = true;
            break;
        }
    }
    assert!(
        observed_not_ready,
        "a delayed copy must be observable as not ready before it completes"
    );

    let (value, filled) = pending.wait().expect("wait");
    assert_eq!(filled.as_slice(), as_bytes(&data));

    // Both the value and the buffer come back and are reusable: a second transfer over them is correct.
    let (_, again) = download_pending(&runtime, value, filled)
        .expect("second pending download")
        .wait()
        .expect("second wait");
    assert_eq!(again.as_slice(), as_bytes(&data));
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_download_completes_after_its_producing_session_closes() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(4096);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");

    let mut session = backend.open_session().expect("open");
    let produced = session
        .with_session(|view| {
            view.add_read(
                tenferro_tensor::TensorRead::from_tensor(&device),
                tenferro_tensor::TensorRead::from_tensor(&device),
            )
        })
        .expect("held callback")
        .expect("held add");

    let buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let pending = download_pending(&runtime, produced, buffer).expect("pending download");
    session.close().expect("close the producing session");

    let (_, filled) = pending.wait().expect("wait");
    let doubled = from_bytes(filled.as_slice());
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(doubled, expected, "the copy outlives its producing session");
}

#[test]
#[ignore = "requires a CUDA device"]
fn dropping_a_pending_download_detaches_into_the_retirement_queue() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(2048);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");
    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned buffer");

    let before = runtime.pending_transfer_retirements();
    let pending = download_pending(&runtime, device, buffer).expect("pending download");
    drop(pending);
    assert!(
        runtime.pending_transfer_retirements() > before,
        "a detached transfer is retained until its event proves completion"
    );

    // An explicit barrier resolves completion, so the record releases value, buffer, retention and
    // event instead of leaking them.
    runtime.synchronize().expect("synchronize");
    assert_eq!(
        runtime.pending_transfer_retirements(),
        0,
        "a completed detached transfer is released exactly once"
    );

    // The runtime stays usable.
    let device = upload_tensor(&runtime, &vector(&data)).expect("second device payload");
    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned buffer");
    let (_, filled) = download_pending(&runtime, device, buffer)
        .expect("second pending download")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(filled.as_slice()), data);
}

#[test]
#[ignore = "requires a CUDA device"]
fn forgetting_a_pending_download_leaks_it_without_releasing_anything() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(1024);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");
    let buffer = PinnedHostBuffer::new(&runtime, 1024 * 8).expect("pinned buffer");

    let before = runtime.pending_transfer_retirements();
    let pending = download_pending(&runtime, device, buffer).expect("pending download");
    // `mem::forget` does not run `Drop`, so nothing is detached: the handle's value, buffer, retention
    // and event are leaked in place. That is the soundness property: forgetting reduces liveness and
    // cannot hand the caller access to the admitted package or reclaim its buffers early.
    std::mem::forget(pending);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        before,
        "a forgotten handle is leaked in place rather than detached"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), before);

    let device = upload_tensor(&runtime, &vector(&data)).expect("second device payload");
    let buffer = PinnedHostBuffer::new(&runtime, 1024 * 8).expect("pinned buffer");
    let (_, filled) = download_pending(&runtime, device, buffer)
        .expect("second pending download")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(filled.as_slice()), data);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_rejected_transfer_returns_the_submitted_package() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(64);
    let host = vector(&data);
    let device = upload_tensor(&runtime, &host).expect("upload");

    // A buffer of the wrong length is refused before anything is enqueued, and the caller keeps both
    // the value and the buffer.
    let short = PinnedHostBuffer::new(&runtime, 64 * 8 + 8).expect("pinned buffer");
    match download_pending(&runtime, device, short) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.value.shape(), &[64]);
            assert_eq!(rejected.buffer.len(), 64 * 8 + 8);
            // Both are usable again: retry with a correctly sized buffer.
            let source = rejected.value;
            let exact = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned buffer");
            let (_, filled) = download_pending(&runtime, source, exact)
                .expect("retry")
                .wait()
                .expect("wait");
            assert_eq!(from_bytes(filled.as_slice()), data);
        }
        other => panic!("expected a pre-admission rejection, got {other:?}"),
    }

    // A host tensor is refused the same way rather than uploaded implicitly.
    let buffer = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned buffer");
    match download_pending(&runtime, host, buffer) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 64 * 8);
        }
        other => panic!("expected a pre-admission rejection, got {other:?}"),
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_unserved_dtype_is_rejected_and_returns_the_package() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let host = Tensor::from_vec_col_major(vec![4], vec![1.0_f64; 4]).expect("host values");
    let device = upload_tensor(&runtime, &host).expect("upload");

    // `bool` is not served by this route; the rejection is typed and carries the value back.
    let flags = Tensor::from_vec_col_major(vec![4], vec![true; 4]).expect("flags");
    let buffer = PinnedHostBuffer::new(&runtime, 4).expect("pinned buffer");
    match download_pending(&runtime, flags, buffer) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 4);
            assert_eq!(rejected.value.shape(), &[4]);
        }
        other => panic!("expected an unserved dtype to be rejected, got {other:?}"),
    }

    // The runtime is unaffected.
    let buffer = PinnedHostBuffer::new(&runtime, 4 * 8).expect("pinned buffer");
    let (_, filled) = download_pending(&runtime, device, buffer)
        .expect("pending download")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(filled.as_slice()), vec![1.0; 4]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_pending_downloads_in_flight_complete_independently() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let first_data = values(4096);
    let second_data = values(8192);
    let first = upload_tensor(&runtime, &vector(&first_data)).expect("upload");
    let second = upload_tensor(&runtime, &vector(&second_data)).expect("upload");

    let first_buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let second_buffer = PinnedHostBuffer::new(&runtime, 8192 * 8).expect("pinned buffer");
    let first_pending = download_pending(&runtime, first, first_buffer).expect("first pending");
    let second_pending = download_pending(&runtime, second, second_buffer).expect("second pending");

    let (_, first_filled) = first_pending.wait().expect("first wait");
    let (_, second_filled) = second_pending.wait().expect("second wait");
    assert_eq!(from_bytes(first_filled.as_slice()), first_data);
    assert_eq!(from_bytes(second_filled.as_slice()), second_data);
}
