#![cfg(feature = "cuda")]

//! Owning pinned host→device transfer contract (#1945 U4 item 3, host→device direction).
//!
//! The source buffer and the destination value are moved into the transfer and come back only from
//! `wait`; a rejected submission returns them unchanged, and dropping or forgetting the handle detaches
//! or leaks rather than releasing anything. These tests are ignored by default; explicitly running them
//! requires a CUDA device.

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_pending, upload_tensor, CudaBackend, CudaDeviceId,
    CudaRuntime, PinnedHostBuffer, TransferSubmitError,
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
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; DELAYED_ELEMENTS]))
        .expect("device destination");

    let source = pinned_source(&runtime, &data);
    let mut pending = upload_pending(&runtime, source, destination).expect("pending upload");

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

    let (device, reusable) = pending.wait().expect("wait");
    assert_eq!(reusable.len(), DELAYED_ELEMENTS * 8);

    // The destination is readable through the returned value and holds the source bytes.
    let downloaded = download_tensor(&runtime, &device).expect("download");
    let published = downloaded.as_slice::<f64>().expect("f64 payload").to_vec();
    assert_eq!(published, data);

    // Both come back reusable: a second upload over them is correct.
    let (device, _) = upload_pending(&runtime, reusable, device)
        .expect("second pending upload")
        .wait()
        .expect("second wait");
    let again = download_tensor(&runtime, &device).expect("download");
    assert_eq!(again.as_slice::<f64>().expect("f64 payload"), &data[..]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn dropping_a_pending_upload_detaches_into_the_retirement_queue() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(2048);
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 2048])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let before = runtime.pending_transfer_retirements();
    let pending = upload_pending(&runtime, source, destination).expect("pending upload");
    drop(pending);
    assert!(
        runtime.pending_transfer_retirements() > before,
        "a detached transfer is retained until its event proves completion"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), 0);

    // The runtime stays usable.
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 2048])).expect("destination");
    let source = pinned_source(&runtime, &data);
    let (device, _) = upload_pending(&runtime, source, destination)
        .expect("second pending upload")
        .wait()
        .expect("wait");
    let downloaded = download_tensor(&runtime, &device).expect("download");
    assert_eq!(
        downloaded.as_slice::<f64>().expect("f64 payload"),
        &data[..]
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn forgetting_a_pending_upload_leaks_it_without_releasing_anything() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(1024);
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 1024])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let before = runtime.pending_transfer_retirements();
    let pending = upload_pending(&runtime, source, destination).expect("pending upload");
    std::mem::forget(pending);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        before,
        "a forgotten handle is leaked in place rather than detached"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), before);

    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 1024])).expect("destination");
    let source = pinned_source(&runtime, &data);
    let (device, _) = upload_pending(&runtime, source, destination)
        .expect("second pending upload")
        .wait()
        .expect("wait");
    let downloaded = download_tensor(&runtime, &device).expect("download");
    assert_eq!(
        downloaded.as_slice::<f64>().expect("f64 payload"),
        &data[..]
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_rejected_upload_returns_the_submitted_package() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let data = values(64);

    // A host tensor is not a device destination: the rejection carries the buffer and the value back.
    let host_destination = vector(&vec![0.0_f64; 64]);
    let source = pinned_source(&runtime, &data);
    match upload_pending(&runtime, source, host_destination) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 64 * 8);
            assert_eq!(rejected.value.shape(), &[64]);
            // The package is usable again once a device destination is supplied.
            let destination =
                upload_tensor(&runtime, &vector(&vec![0.0_f64; 64])).expect("destination");
            let (device, _) = upload_pending(&runtime, rejected.buffer, destination)
                .expect("retry")
                .wait()
                .expect("wait");
            let downloaded = download_tensor(&runtime, &device).expect("download");
            assert_eq!(
                downloaded.as_slice::<f64>().expect("f64 payload"),
                &data[..]
            );
        }
        other => panic!("expected a pre-admission rejection, got {other:?}"),
    }

    // A source buffer of the wrong length is refused the same way.
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 64])).expect("destination");
    let source = PinnedHostBuffer::new(&runtime, 64 * 8 + 8).expect("pinned buffer");
    match upload_pending(&runtime, source, destination) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 64 * 8 + 8);
        }
        other => panic!("expected a pre-admission rejection, got {other:?}"),
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_pending_uploads_in_flight_complete_independently() {
    let backend = backend();
    let runtime = runtime_of(&backend);
    let first_data = values(4096);
    let second_data = values(8192);

    let first_destination =
        upload_tensor(&runtime, &vector(&vec![0.0_f64; 4096])).expect("destination");
    let second_destination =
        upload_tensor(&runtime, &vector(&vec![0.0_f64; 8192])).expect("destination");
    let first_pending = upload_pending(
        &runtime,
        pinned_source(&runtime, &first_data),
        first_destination,
    )
    .expect("first pending");
    let second_pending = upload_pending(
        &runtime,
        pinned_source(&runtime, &second_data),
        second_destination,
    )
    .expect("second pending");

    let (first_device, _) = first_pending.wait().expect("first wait");
    let (second_device, _) = second_pending.wait().expect("second wait");
    let first = download_tensor(&runtime, &first_device).expect("download");
    let second = download_tensor(&runtime, &second_device).expect("download");
    assert_eq!(
        first.as_slice::<f64>().expect("f64 payload"),
        &first_data[..]
    );
    assert_eq!(
        second.as_slice::<f64>().expect("f64 payload"),
        &second_data[..]
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
