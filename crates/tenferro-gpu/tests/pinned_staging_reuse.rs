#![cfg(feature = "cuda")]

//! Download staging is allocated once per runtime, reused, bounded and releasable.
//!
//! A raw-session download used to pin and unpin a fresh host buffer on every
//! call (`cuMemHostAlloc` + `cuMemFreeHost`). Pinning cost far more than the
//! copy it staged, so the buffer now lives in the runtime's resource cache and
//! grows on demand. These tests pin that behaviour: the first download creates
//! one resource, later downloads reuse it, and a larger download grows it once
//! while still returning the right bytes. They also pin the bound added in
//! tenferro-rs #2440: a download above the retained-size limit keeps nothing,
//! lowering the limit releases a larger buffer, and the buffer can be cleared.
//!
//! They are ignored by default; running them requires a CUDA device.

use tenferro_gpu::cuda::{gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::Tensor;

fn engine() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn download(engine: &mut CudaEngine, tensor: &Tensor) -> Vec<f64> {
    engine
        .with_raw("pinned staging test", |raw| {
            raw.download_tensor::<f64>(raw_tensor(tensor), "pinned staging test")
        })
        .expect("download")
        .host_data()
        .expect("host values")
        .to_vec()
}

/// The session reads device tensors, so the upload has to happen first.
fn raw_tensor(tensor: &Tensor) -> &tenferro_tensor::TypedTensor<f64> {
    tensor.as_typed::<f64>().expect("f64 tensor")
}

#[test]
#[ignore = "requires a CUDA device"]
fn repeated_downloads_reuse_one_staging_allocation() {
    let mut engine = engine();
    let host = Tensor::from_vec_col_major(vec![1024], values(1024)).expect("host tensor");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");

    let before = engine.runtime().pinned_staging_allocations();
    for _ in 0..4 {
        assert_eq!(download(&mut engine, &device), values(1024));
    }
    assert_eq!(
        engine.runtime().pinned_staging_allocations() - before,
        1,
        "four downloads must stage through one pinned allocation"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_larger_download_grows_the_staging_allocation_once() {
    let mut engine = engine();
    let small = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![16], values(16)).expect("small host tensor"),
    )
    .expect("upload small");
    let large = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![8192], values(8192)).expect("large host tensor"),
    )
    .expect("upload large");

    let before = engine.runtime().pinned_staging_allocations();
    assert_eq!(download(&mut engine, &small), values(16));
    assert_eq!(download(&mut engine, &small), values(16));
    assert_eq!(download(&mut engine, &large), values(8192));
    assert_eq!(download(&mut engine, &large), values(8192));
    assert_eq!(download(&mut engine, &small), values(16));
    assert_eq!(
        engine.runtime().pinned_staging_allocations() - before,
        2,
        "the buffer is created once and grown once, never re-allocated per download"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_download_above_the_limit_retains_nothing() {
    // tenferro-rs #2440: the staging buffer is page-locked memory the allocator cannot reclaim, so
    // a single large download must not keep it for the runtime's lifetime.
    let mut engine = engine();
    let runtime = engine.runtime();
    runtime.set_pinned_staging_limit_bytes(1024);
    assert_eq!(runtime.pinned_staging_stats().limit_bytes, 1024);

    let large = upload_tensor(
        runtime,
        &Tensor::from_vec_col_major(vec![8192], values(8192)).expect("large host tensor"),
    )
    .expect("upload large");
    // The value still round-trips: the download stages through a buffer it does not keep.
    assert_eq!(download(&mut engine, &large), values(8192));
    let stats = engine.runtime().pinned_staging_stats();
    assert_eq!(
        stats.retained_bytes, 0,
        "a download above the bound must not be retained"
    );
    assert_eq!(stats.limit_bytes, 1024);

    // Below the bound it is retained again, and stays within it.
    let small = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![16], values(16)).expect("small host tensor"),
    )
    .expect("upload small");
    assert_eq!(download(&mut engine, &small), values(16));
    let stats = engine.runtime().pinned_staging_stats();
    assert!(
        stats.retained_bytes > 0 && stats.retained_bytes <= 1024,
        "retained {} bytes against a {} byte bound",
        stats.retained_bytes,
        stats.limit_bytes
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn clear_releases_the_retained_staging() {
    let mut engine = engine();
    let host = Tensor::from_vec_col_major(vec![512], values(512)).expect("host tensor");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");

    assert_eq!(download(&mut engine, &device), values(512));
    assert!(engine.runtime().pinned_staging_stats().retained_bytes > 0);

    let released = engine.runtime().clear_pinned_staging();
    assert_eq!(released, 512 * std::mem::size_of::<f64>());
    let stats = engine.runtime().pinned_staging_stats();
    assert_eq!(stats.retained_bytes, 0, "clear must release the buffer");
    // Clearing is not a one-way door: the next download pins a buffer again and still returns the
    // right values.
    assert_eq!(download(&mut engine, &device), values(512));
    assert!(engine.runtime().pinned_staging_stats().retained_bytes > 0);
    assert_eq!(engine.runtime().clear_pinned_staging(), 512 * 8);
}

#[test]
#[ignore = "requires a CUDA device"]
fn lowering_the_limit_releases_a_larger_buffer() {
    let mut engine = engine();
    let host = Tensor::from_vec_col_major(vec![4096], values(4096)).expect("host tensor");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");
    assert_eq!(download(&mut engine, &device), values(4096));
    assert_eq!(
        engine.runtime().pinned_staging_stats().retained_bytes,
        4096 * 8
    );

    engine.runtime().set_pinned_staging_limit_bytes(1024);
    assert_eq!(
        engine.runtime().pinned_staging_stats().retained_bytes,
        0,
        "lowering the bound below what is retained must release it"
    );
    // A second, smaller download is retained under the new bound.
    let small = upload_tensor(
        engine.runtime(),
        &Tensor::from_vec_col_major(vec![64], values(64)).expect("small host tensor"),
    )
    .expect("upload small");
    assert_eq!(download(&mut engine, &small), values(64));
    let stats = engine.runtime().pinned_staging_stats();
    assert!(stats.retained_bytes > 0 && stats.retained_bytes <= 1024);
}
