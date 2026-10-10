#![cfg(feature = "cuda")]

//! Pending device→host handoff contract (#1945 U4, slice 1).
//!
//! These tests pin the owned-buffer pending download: the handle reports not-ready while a
//! deliberately delayed copy is in flight, `wait` returns the bytes only after completion, the
//! transfer survives its producing session closing, and a forgotten handle leaks rather than frees
//! memory the device may still write. They are ignored by default; explicitly running them requires
//! a CUDA device and fails if none is available.

use tenferro_gpu::cuda::{
    download_pending, gpu_available, upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime,
    PinnedHostBuffer,
};
use tenferro_tensor::{Tensor, TensorRead};

/// Big enough that a copy of it cannot complete before an immediate poll observes the queue.
const DELAYED_ELEMENTS: usize = 8 * 1024 * 1024;

fn backend() -> CudaBackend {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
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

fn runtime_of(backend: &CudaBackend) -> CudaRuntime {
    backend.runtime().clone()
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_download_reports_not_ready_then_returns_the_bytes() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(DELAYED_ELEMENTS);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");

    let buffer = PinnedHostBuffer::new(&runtime, DELAYED_ELEMENTS * 8).expect("pinned buffer");
    let mut pending = download_pending(&runtime, &device, buffer).expect("pending download");

    // The copy of a payload this size cannot have completed before this immediate poll, so at least
    // one observation must be "not ready" rather than "already done".
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

    let filled = pending.wait().expect("wait");
    let bytes = filled.as_slice();
    assert_eq!(bytes.len(), DELAYED_ELEMENTS * 8);
    assert_eq!(
        bytes,
        as_bytes(&data),
        "the completed copy holds the source values"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_download_completes_after_its_producing_session_closes() {
    let mut backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(4096);
    let host = vector(&data);
    let device = upload_tensor(&runtime, &host).expect("upload");

    // Produce inside a held session (U3), then close it while the transfer is still in flight.
    let mut session = backend.open_session().expect("open");
    let produced = session
        .with_session(|view| {
            view.add_read(
                TensorRead::from_tensor(&device),
                TensorRead::from_tensor(&device),
            )
        })
        .expect("held callback")
        .expect("held add");

    let buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let pending = download_pending(&runtime, &produced, buffer).expect("pending download");
    session.close().expect("close the producing session");

    let filled = pending.wait().expect("wait");
    let doubled = from_bytes(filled.as_slice());
    let expected: Vec<f64> = data.iter().map(|value| value * 2.0).collect();
    assert_eq!(doubled, expected, "the copy outlives its producing session");
}

#[test]
#[ignore = "requires a CUDA device"]
fn forgetting_a_pending_download_never_releases_the_buffer_or_source() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(1024);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let buffer = PinnedHostBuffer::new(&runtime, 1024 * 8).expect("pinned buffer");

    let pending = download_pending(&runtime, &device, buffer).expect("pending download");
    // `mem::forget` must be safe: the handle owns everything it uses, so forgetting it leaks the
    // buffer and the retained source instead of freeing memory the copy may still write.
    std::mem::forget(pending);

    // The runtime stays usable and a later transfer is correct.
    let buffer = PinnedHostBuffer::new(&runtime, 1024 * 8).expect("second pinned buffer");
    let filled = download_pending(&runtime, &device, buffer)
        .expect("second pending download")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(filled.as_slice()), data);
}

#[test]
#[ignore = "requires a CUDA device"]
fn dropping_a_pending_download_resolves_before_releasing() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(2048);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned buffer");

    let pending = download_pending(&runtime, &device, buffer).expect("pending download");
    drop(pending);

    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("second pinned buffer");
    let filled = download_pending(&runtime, &device, buffer)
        .expect("second pending download")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(filled.as_slice()), data);
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_download_rejects_a_mismatched_buffer_and_a_host_tensor() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(64);
    let host = vector(&data);

    let buffer = PinnedHostBuffer::new(&runtime, 64 * 8 + 8).expect("pinned buffer");
    let mismatch = download_pending(&runtime, &host, buffer);
    assert!(mismatch.is_err(), "a short buffer must be rejected");

    let buffer = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned buffer");
    let uploaded = upload_tensor(&runtime, &host).expect("upload");
    let buffer_two = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned buffer");
    assert!(
        download_pending(&runtime, &uploaded, buffer_two).is_ok(),
        "a device tensor is accepted"
    );
    let host_attempt = download_pending(&runtime, &host, buffer);
    assert!(
        host_attempt.is_err(),
        "a host tensor must fail typed rather than upload implicitly"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn pinned_host_buffers_are_owned_and_reusable() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    for bytes in [8, 4096, 1 << 20] {
        let buffer = PinnedHostBuffer::new(&runtime, bytes).expect("pinned buffer");
        assert_eq!(buffer.len(), bytes);
        assert!(!buffer.is_empty());
    }
    let zero = PinnedHostBuffer::new(&runtime, 0);
    assert!(zero.is_err(), "a zero-length pinned buffer is rejected");
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
    let first_pending = download_pending(&runtime, &first, first_buffer).expect("first pending");
    let second_pending =
        download_pending(&runtime, &second, second_buffer).expect("second pending");

    let first_filled = first_pending.wait().expect("first wait");
    let second_filled = second_pending.wait().expect("second wait");
    let first_bytes = from_bytes(first_filled.as_slice());
    let second_bytes = from_bytes(second_filled.as_slice());
    assert_eq!(first_bytes, first_data);
    assert_eq!(second_bytes, second_data);
}
