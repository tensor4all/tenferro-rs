#![cfg(feature = "cuda")]

//! Download staging is allocated once per runtime and reused.
//!
//! A raw-session download used to pin and unpin a fresh host buffer on every
//! call (`cuMemHostAlloc` + `cuMemFreeHost`). Pinning cost far more than the
//! copy it staged, so the buffer now lives in the runtime's resource cache and
//! grows on demand. These tests pin that behaviour: the first download creates
//! one resource, later downloads reuse it, and a larger download grows it once
//! while still returning the right bytes.
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
