//! A refused device allocation is a synchronous typed error (issue #2327).
//!
//! `CudaEngine::alloc_output` routes to `alloc_tensor`, which submits
//! `alloc_zeroed` on the engine's runtime. The runtime asks its memory pool
//! before constructing a buffer, so a 200 GiB request that the device cannot
//! satisfy returns the typed out-of-memory cause instead of a failed buffer or
//! a huge host reservation. The runtime then has to remain usable for ordinary
//! transfers.

use std::error::Error as _;

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{DType, Tensor};

const REFUSED_BYTES: usize = 200 * 1024 * 1024 * 1024;
const REFUSED_ELEMENTS: usize = REFUSED_BYTES / std::mem::size_of::<f64>();

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine")
}

fn out_of_memory_bytes(error: &tenferro_tensor::Error) -> Option<usize> {
    let mut source = error.source()?;
    loop {
        if let Some(t4a_cuda_runtime::Error::OutOfMemory { bytes }) = source.downcast_ref() {
            return Some(*bytes);
        }
        source = source.source()?;
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_refused_device_allocation_returns_out_of_memory_without_a_buffer() {
    let engine = backend();
    let before = engine.runtime().stats();

    let refused = engine
        .alloc_output(DType::F64, &[REFUSED_ELEMENTS])
        .expect_err("the device cannot satisfy a 200 GiB allocation");
    assert_eq!(
        out_of_memory_bytes(&refused),
        Some(REFUSED_BYTES),
        "expected the runtime's typed out-of-memory cause, got {refused:?}"
    );

    let after = engine.runtime().stats();
    assert_eq!(
        after.live_allocations, before.live_allocations,
        "a refused allocation must not leave a device buffer behind"
    );
    assert_eq!(
        after.live_bytes, before.live_bytes,
        "a refused allocation must not leave device bytes behind"
    );

    // The runtime is not poisoned: a small zero-filled allocation reads back correctly.
    let zeroed = engine
        .alloc_output(DType::F64, &[4])
        .expect("small allocation after the refusal");
    let zeroed_host = download_tensor(engine.runtime(), &zeroed).expect("small download");
    assert_eq!(
        zeroed_host.as_slice::<f64>().expect("f64 payload"),
        &[0.0; 4]
    );

    // An ordinary upload/download round trip still carries the values.
    let values = [1.5_f64, -2.0, 3.25];
    let host =
        Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("host tensor");
    let device = upload_tensor(engine.runtime(), &host).expect("small upload");
    let back = download_tensor(engine.runtime(), &device).expect("small download");
    assert_eq!(
        back.as_slice::<f64>().expect("f64 payload"),
        &values[..],
        "the round trip after the refusal keeps its values"
    );

    // An empty allocation and download remain well-formed.
    let empty = engine
        .alloc_output(DType::F64, &[0])
        .expect("empty allocation after the refusal");
    let empty_host = download_tensor(engine.runtime(), &empty).expect("empty download");
    assert_eq!(empty_host.shape(), &[0]);
    assert!(empty_host
        .as_slice::<f64>()
        .expect("f64 payload")
        .is_empty());
}
