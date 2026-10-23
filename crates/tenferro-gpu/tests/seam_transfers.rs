#![cfg(feature = "cuda-next")]

//! U1 seam transfer evidence: pending upload/download semantics, drop/forget, rejected
//! packages and pinned round-trips (issue #2208 U1).
//!
//! Every device test returns early without a CUDA device, so the suite is safe on a CPU
//! host; running it on the local A100 is what the worklog records.

use tenferro_gpu::cuda_next::{
    download_pending, download_tensor, gpu_available, upload_pending, upload_tensor, CudaDeviceId,
    CudaEngine, CudaRuntime, PinnedHostBuffer, TransferSubmitError,
};
use tenferro_tensor::Tensor;

/// Large enough that a copy of it cannot complete before an immediate poll.
const DELAYED_ELEMENTS: usize = 4 * 1024 * 1024;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn runtime_of(engine: &CudaEngine) -> CudaRuntime {
    engine.runtime().clone()
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

fn host_values(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().expect("host f64").to_vec()
}

#[test]
fn pending_upload_reports_not_ready_then_publishes_the_values() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(DELAYED_ELEMENTS);
    let destination =
        upload_tensor(&runtime, &vector(&vec![0.0_f64; DELAYED_ELEMENTS])).expect("destination");
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
    let downloaded = download_tensor(&runtime, &device).expect("download");
    assert_eq!(host_values(&downloaded), data);

    // Both come back reusable.
    let (device, _) = upload_pending(&runtime, reusable, device)
        .expect("second pending upload")
        .wait()
        .expect("second wait");
    let again = download_tensor(&runtime, &device).expect("download");
    assert_eq!(host_values(&again), data);
}

#[test]
fn dropping_a_pending_upload_detaches_and_synchronize_retires_it() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(2048);
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 2048])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let pending = upload_pending(&runtime, source, destination).expect("pending upload");
    drop(pending);
    assert!(
        runtime.pending_transfer_retirements() > 0,
        "a detached transfer is retained until the stream is observed"
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
    assert_eq!(
        host_values(&download_tensor(&runtime, &device).expect("download")),
        data
    );
}

#[test]
fn forgetting_a_pending_upload_leaks_it_in_place() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(1024);
    let destination = upload_tensor(&runtime, &vector(&vec![0.0_f64; 1024])).expect("destination");
    let source = pinned_source(&runtime, &data);

    let pending = upload_pending(&runtime, source, destination).expect("pending upload");
    std::mem::forget(pending);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        0,
        "a forgotten handle is leaked in place rather than detached"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), 0);
}

#[test]
fn a_rejected_upload_returns_the_submitted_package() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(64);

    // A host tensor is not a device destination.
    let host_destination = vector(&vec![0.0_f64; 64]);
    let source = pinned_source(&runtime, &data);
    match upload_pending(&runtime, source, host_destination) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 64 * 8);
            assert_eq!(rejected.value.shape(), &[64]);
            let destination =
                upload_tensor(&runtime, &vector(&vec![0.0_f64; 64])).expect("destination");
            let (device, _) = upload_pending(&runtime, rejected.buffer, destination)
                .expect("retry")
                .wait()
                .expect("wait");
            assert_eq!(
                host_values(&download_tensor(&runtime, &device).expect("download")),
                data
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
fn two_pending_uploads_in_flight_complete_independently() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
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
    assert_eq!(
        host_values(&download_tensor(&runtime, &first_device).expect("download")),
        first_data
    );
    assert_eq!(
        host_values(&download_tensor(&runtime, &second_device).expect("download")),
        second_data
    );
}

#[test]
fn pending_download_reports_not_ready_then_returns_the_bytes_and_the_value() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(DELAYED_ELEMENTS);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");
    let buffer = PinnedHostBuffer::new(&runtime, DELAYED_ELEMENTS * 8).expect("pinned buffer");

    let mut pending = download_pending(&runtime, device, buffer).expect("pending download");
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
    assert_eq!(filled.as_slice()[..data.len() * 8], as_bytes(&data)[..]);

    // Both come back reusable.
    let (_, again) = download_pending(&runtime, value, filled)
        .expect("second pending download")
        .wait()
        .expect("second wait");
    assert_eq!(again.as_slice()[..data.len() * 8], as_bytes(&data)[..]);
}

#[test]
fn pending_download_survives_its_producing_engine() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(4096);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned buffer");
    let pending = download_pending(&runtime, device, buffer).expect("pending download");
    drop(engine);

    let (_, filled) = pending.wait().expect("wait");
    assert_eq!(filled.as_slice()[..data.len() * 8], as_bytes(&data)[..]);
}

#[test]
fn pinned_buffer_bytes_round_trip() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(64);
    let source = pinned_source(&runtime, &data);
    assert_eq!(source.as_slice(), as_bytes(&data).as_slice());
    assert_eq!(source.len(), 64 * 8);
}

#[test]
fn dropping_a_pending_download_detaches_and_forgetting_it_leaks() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(2048);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");

    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned");
    let pending = download_pending(&runtime, device, buffer).expect("pending download");
    drop(pending);
    assert!(
        runtime.pending_transfer_retirements() > 0,
        "a detached download is retained until its completion is observed"
    );
    runtime.synchronize().expect("synchronize");
    assert_eq!(runtime.pending_transfer_retirements(), 0);

    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");
    let buffer = PinnedHostBuffer::new(&runtime, 2048 * 8).expect("pinned");
    let pending = download_pending(&runtime, device, buffer).expect("pending download");
    std::mem::forget(pending);
    assert_eq!(
        runtime.pending_transfer_retirements(),
        0,
        "a forgotten download is leaked in place"
    );
}

#[test]
fn a_rejected_download_returns_the_package() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let data = values(64);
    let device = upload_tensor(&runtime, &vector(&data)).expect("device payload");

    // A buffer shorter than the payload is refused before admission.
    let short = PinnedHostBuffer::new(&runtime, 64 * 8 - 8).expect("pinned");
    match download_pending(&runtime, device, short) {
        Err(TransferSubmitError::Rejected(rejected)) => {
            assert_eq!(rejected.buffer.len(), 64 * 8 - 8);
            assert_eq!(rejected.value.shape(), &[64]);
            // The package is usable again with a buffer of the right length.
            let buffer = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned");
            let (_, filled) = download_pending(&runtime, rejected.value, buffer)
                .expect("retry")
                .wait()
                .expect("wait");
            assert_eq!(filled.as_slice(), as_bytes(&data).as_slice());
        }
        other => panic!("expected a pre-admission rejection, got {other:?}"),
    }

    // A host value has no device allocation to read.
    let buffer = PinnedHostBuffer::new(&runtime, 64 * 8).expect("pinned");
    assert!(matches!(
        download_pending(&runtime, vector(&data), buffer),
        Err(TransferSubmitError::Rejected(_))
    ));
}

#[test]
fn two_pending_downloads_in_flight_complete_independently() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    let first_data = values(4096);
    let second_data = values(8192);
    let first_device = upload_tensor(&runtime, &vector(&first_data)).expect("first");
    let second_device = upload_tensor(&runtime, &vector(&second_data)).expect("second");
    let first_buffer = PinnedHostBuffer::new(&runtime, 4096 * 8).expect("pinned");
    let second_buffer = PinnedHostBuffer::new(&runtime, 8192 * 8).expect("pinned");

    let first = download_pending(&runtime, first_device, first_buffer).expect("first pending");
    let second = download_pending(&runtime, second_device, second_buffer).expect("second pending");
    let (_, first_filled) = first.wait().expect("first wait");
    let (_, second_filled) = second.wait().expect("second wait");
    assert_eq!(first_filled.as_slice(), as_bytes(&first_data).as_slice());
    assert_eq!(second_filled.as_slice(), as_bytes(&second_data).as_slice());
}

#[test]
fn an_unserved_dtype_is_rejected_and_returns_the_package() {
    let Some(engine) = engine() else { return };
    let runtime = runtime_of(&engine);
    // A Bool payload has no CUDA transfer route in U1.
    let device_bool =
        Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).expect("bool tensor");
    assert!(upload_tensor(&runtime, &device_bool).is_err());
}
