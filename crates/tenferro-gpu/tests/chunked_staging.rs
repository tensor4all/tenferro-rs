#![cfg(feature = "cuda")]

//! Chunked blocking staging contract (#1945 U4 item 4).
//!
//! Above the staging chunk, the blocking `upload_tensor` route moves its payload through the runtime's
//! pinned staging lease instead of the pageable transport. These tests fix the routing boundary and
//! prove the values survive it in both directions: an off-by-one in a chunk offset or a lost remainder
//! is what they are here to catch. Ignored by default; explicitly running them requires a CUDA device.

use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
};
use tenferro_tensor::{StorageBuffer, Tensor};

/// The chunk the runtime stages at a time.
const CHUNK_BYTES: usize = 8 * 1024 * 1024;

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn runtime_of(backend: &CudaEngine) -> CudaRuntime {
    backend.runtime().clone()
}

/// Distinct values per index, so a wrong chunk offset cannot pass as a correct copy.
fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

fn host_values(tensor: &Tensor) -> Vec<f64> {
    let typed = tensor.as_typed::<f64>().expect("f64 tensor");
    match typed.buffer() {
        StorageBuffer::Host(data) => data.clone(),
        StorageBuffer::Backend(_) => panic!("expected host storage after a download"),
    }
}

/// Round-trip `elements` through the blocking routes and compare the values exactly.
fn round_trip(backend: &CudaEngine, elements: usize) {
    let runtime = runtime_of(backend);
    let data = values(elements);
    let device = upload_tensor(&runtime, &vector(&data)).expect("upload");
    let back = download_tensor(&runtime, &device).expect("download");
    assert_eq!(host_values(&back), data, "elements: {elements}");
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_payload_exactly_one_chunk_long_round_trips() {
    // At the chunk size the routes keep their previous behaviour, so this is the boundary from below.
    round_trip(&backend(), CHUNK_BYTES / 8);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_payload_one_element_under_the_chunk_round_trips() {
    round_trip(&backend(), CHUNK_BYTES / 8 - 1);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_payload_one_element_over_the_chunk_round_trips() {
    // Two chunks, the second one element long: the routing boundary and a non-multiple remainder.
    round_trip(&backend(), CHUNK_BYTES / 8 + 1);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_payload_several_chunks_long_round_trips() {
    // Three chunks, the last one three elements: the loop's offsets and its tail handling.
    round_trip(&backend(), 2 * (CHUNK_BYTES / 8) + 3);
}
