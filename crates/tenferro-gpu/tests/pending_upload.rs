#![cfg(feature = "cuda")]

//! Pending pinned host→device handoff contract (#1945 U4, slice 2).
//!
//! These tests pin the owned-source pending upload: the handle reports not-ready while a
//! deliberately delayed copy is in flight, `wait` returns the source buffer for reuse only after
//! completion, the destination is exclusively borrowed until then, and failures are typed. They are
//! ignored by default; explicitly running them requires a CUDA device and fails if none is
//! available.

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_pending, upload_tensor, CudaBackend, CudaDeviceId,
    CudaRuntime, PinnedHostBuffer,
};
use tenferro_tensor::Tensor;

/// Big enough that a copy of it cannot complete before an immediate poll observes the queue.
const DELAYED_ELEMENTS: usize = 4 * 1024 * 1024;

fn backend() -> CudaBackend {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn runtime_of(backend: &CudaBackend) -> CudaRuntime {
    backend.runtime().clone()
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

fn from_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect()
}

fn pinned_source(runtime: &CudaRuntime, data: &[f64]) -> PinnedHostBuffer {
    let mut source = PinnedHostBuffer::new(runtime, data.len() * 8).expect("pinned buffer");
    source.as_mut_slice().copy_from_slice(&as_bytes(data));
    source
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_upload_reports_not_ready_then_publishes_the_values() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(DELAYED_ELEMENTS);
    let zeros = vec![0.0_f64; DELAYED_ELEMENTS];
    let mut device = upload_tensor(&runtime, &vector(&zeros)).expect("device destination");

    let source = pinned_source(&runtime, &data);
    let mut pending = upload_pending(&runtime, source, &mut device).expect("pending upload");

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

    let reusable = pending.wait().expect("wait");
    assert_eq!(reusable.len(), DELAYED_ELEMENTS * 8);

    // The destination is usable again once the handle is consumed, and it holds the source values.
    let downloaded = download_tensor(&runtime, &device).expect("download");
    let published = downloaded.as_slice::<f64>().expect("f64 payload").to_vec();
    assert_eq!(
        published, data,
        "the completed copy published the source values"
    );
    assert_eq!(
        from_bytes(reusable.as_slice()),
        data,
        "source bytes stay intact"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn forgetting_a_pending_upload_never_releases_the_source_buffer() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(1024);
    let mut device = upload_tensor(&runtime, &vector(&vec![0.0_f64; 1024])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let pending = upload_pending(&runtime, source, &mut device).expect("pending upload");
    // `mem::forget` must be safe: the handle owns the source buffer, so forgetting it leaks rather
    // than freeing memory the copy may still read.
    std::mem::forget(pending);

    let mut second = upload_tensor(&runtime, &vector(&vec![0.0_f64; 1024])).expect("destination");
    let source = pinned_source(&runtime, &data);
    let reusable = upload_pending(&runtime, source, &mut second)
        .expect("second pending upload")
        .wait()
        .expect("wait");
    assert_eq!(from_bytes(reusable.as_slice()), data);
    let downloaded = download_tensor(&runtime, &second).expect("download");
    assert_eq!(
        downloaded.as_slice::<f64>().expect("f64 payload"),
        &data[..]
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn dropping_a_pending_upload_resolves_before_releasing_the_destination() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(2048);
    let mut device = upload_tensor(&runtime, &vector(&vec![0.0_f64; 2048])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let pending = upload_pending(&runtime, source, &mut device).expect("pending upload");
    drop(pending);

    let downloaded = download_tensor(&runtime, &device).expect("download");
    assert_eq!(
        downloaded.as_slice::<f64>().expect("f64 payload"),
        &data[..],
        "dropping the handle must publish a completed copy"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn pending_upload_rejects_a_mismatched_source_and_a_host_destination() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(64);

    let host_destination = vector(&vec![0.0_f64; 64]);
    let mut host_destination = host_destination;
    let source = pinned_source(&runtime, &data);
    assert!(
        upload_pending(&runtime, source, &mut host_destination).is_err(),
        "a host tensor destination fails typed rather than uploading implicitly"
    );

    let mut device = upload_tensor(&runtime, &vector(&vec![0.0_f64; 64])).expect("destination");
    let source = PinnedHostBuffer::new(&runtime, 64 * 8 + 8).expect("pinned buffer");
    assert!(
        upload_pending(&runtime, source, &mut device).is_err(),
        "a source buffer of the wrong length is rejected"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn pinned_buffer_bytes_round_trip() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(32);
    let source = pinned_source(&runtime, &data);
    assert_eq!(source.as_slice(), as_bytes(&data));
    assert_eq!(source.len(), 32 * 8);
}
