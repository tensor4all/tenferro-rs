#![cfg(feature = "cuda-next")]

//! U1 seam staging evidence: payloads around the runtime's staging chunk round-trip and
//! the runtime's own counters move (issue #2208 U1).

use tenferro_gpu::cuda_next::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
};

/// The runtime's default staging chunk is 8 MiB; the sizes below bracket it.
const CHUNK_BYTES: usize = 8 << 20;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 1.0).collect()
}

fn round_trip(engine: &CudaEngine, bytes: usize) -> (u64, u64) {
    let runtime = engine.runtime();
    let elements = bytes / 8;
    let data = values(elements);
    let host = tenferro_tensor::Tensor::from_vec_col_major(vec![elements], data.clone())
        .expect("host tensor");
    let before = runtime.stats().retirement;
    let device = upload_tensor(runtime, &host).expect("upload");
    let back = download_tensor(runtime, &device).expect("download");
    let after = runtime.stats().retirement;
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice(),
        "a {bytes}-byte payload must round-trip unchanged"
    );
    (
        after.events_recorded.saturating_sub(before.events_recorded),
        after.idle_pinned_bytes as u64,
    )
}

#[test]
fn payloads_around_the_staging_chunk_round_trip() {
    let Some(engine) = engine() else { return };
    // Exactly one chunk, one element under, one over, and several chunks.
    for bytes in [
        CHUNK_BYTES,
        CHUNK_BYTES - 8,
        CHUNK_BYTES + 8,
        CHUNK_BYTES * 3 + 16,
    ] {
        let (events, idle) = round_trip(&engine, bytes);
        assert!(
            events > 0,
            "a {bytes}-byte transfer records at least one completion event"
        );
        // The runtime keeps its staging bounded: idle pinned bytes never exceed the
        // configured idle budget (64 MiB by default).
        assert!(
            idle <= (64 << 20),
            "idle pinned staging must stay bounded, got {idle}"
        );
    }
}

#[test]
fn a_small_payload_leaves_no_pending_host_bytes() {
    let Some(engine) = engine() else { return };
    let runtime = engine.runtime();
    let data = values(128);
    let host =
        tenferro_tensor::Tensor::from_vec_col_major(vec![128], data.clone()).expect("host tensor");
    let device = upload_tensor(runtime, &host).expect("upload");
    let _ = download_tensor(runtime, &device).expect("download");
    let stats = runtime.stats();
    assert_eq!(
        stats.live_bytes,
        128 * 8,
        "one live allocation of the payload"
    );
    assert_eq!(stats.live_allocations, 1);
}

#[test]
fn a_small_staging_budget_forces_chunking_and_drains() {
    if !gpu_available() {
        return;
    }
    // 1 MiB chunks and a 4 MiB pending budget: a 3 MiB payload must go through several
    // chunks and the retirement records must drain afterwards.
    let config = t4a_cuda_runtime::Config {
        max_chunk_bytes: 1 << 20,
        max_idle_pinned_bytes: 4 << 20,
        pending_host_budget: 4 << 20,
        ..t4a_cuda_runtime::Config::default()
    };
    let runtime = CudaRuntime::with_config(CudaDeviceId::from_ordinal(0), config)
        .expect("runtime with a small staging budget");
    let elements = 3 * (1 << 20) / 8;
    let data = values(elements);
    let host =
        tenferro_tensor::Tensor::from_vec_col_major(vec![elements], data.clone()).expect("host");
    let device = upload_tensor(&runtime, &host).expect("chunked upload");
    let after_upload = runtime.stats();
    assert!(
        after_upload.retirement.chunk_allocs >= 3,
        "a 3 MiB payload with 1 MiB chunks must use at least three chunks, got {}",
        after_upload.retirement.chunk_allocs
    );
    let back = download_tensor(&runtime, &device).expect("download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        data.as_slice(),
        "a chunked payload must round-trip unchanged"
    );
    runtime.synchronize().expect("synchronize");
    let drained = runtime.stats();
    assert_eq!(
        drained.retirement.pending_records, 0,
        "every retirement record is reclaimed after the payload completes"
    );
}
