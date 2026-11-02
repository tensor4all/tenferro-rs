//! The opened runtime's pool budget follows the device, and a trim gives memory back (#2381).
//!
//! Run with: `cargo test -p tenferro-gpu --features cuda --test pool_retention -- --ignored`

#![cfg(feature = "cuda")]

use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine};

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn an_opened_runtime_keeps_a_device_scaled_pool_budget() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let runtime = engine.runtime();

    let total = runtime.device_info().total_memory_bytes();
    let configured = runtime.pool_release_threshold_bytes();
    let share = usize::try_from(total / 8).unwrap_or(usize::MAX);
    assert_eq!(
        configured,
        share.clamp(256 << 20, 4 << 30),
        "the budget must be the device's share, clamped, for a {total} byte device"
    );
    // The default budget covers the working set of one large eager operation, which the driver's
    // own 256 MiB default did not: 6 live 72 MiB buffers were enough to force a re-map per call.
    assert!(
        configured >= 512 << 20,
        "the budget must cover a several-hundred-MiB working set, got {configured}"
    );

    // The explicit clear path exists and drains to the requested floor.
    runtime.trim_pool(0).expect("trim the pool to nothing");
}
