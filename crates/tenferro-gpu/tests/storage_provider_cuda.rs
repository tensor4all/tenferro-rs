#![cfg(feature = "cuda")]

//! Final P7 CUDA storage-provider evidence.
//!
//! Device tests are ignored by default because they need real CUDA allocation
//! roots and provider-native bindings. Explicit execution fails without a device;
//! the source-level API contract remains runnable on hosts without CUDA.

#[path = "integration/engine_run.rs"]
mod engine_run;
use engine_run::EngineRun;
use std::fs;
use std::path::Path;

use tenferro_gpu::{
    cuda::gpu_available, cuda::upload_tensor, cuda::CudaDeviceId, cuda::CudaEngine,
};
use tenferro_tensor::{AllocationDomainId, AllocationId, Tensor, TensorRead};

fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
    let Some(tensor) = tensor.as_typed::<f32>() else {
        panic!("P7 provider tests use f32 tensors")
    };
    (tensor.allocation_domain(), tensor.allocation_id())
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_tensor_view_keeps_the_single_root_identity() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let (domain, allocation) = identity(&input);

    let Some(tensor) = input.as_typed::<f32>() else {
        panic!("provider test uses f32 tensors")
    };
    let view = tensor.as_view();
    assert_eq!(view.allocation_domain(), domain);
    assert_eq!(view.allocation_id(), allocation);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_duplicate_is_explicit_same_placement_allocation() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let (domain, allocation) = identity(&input);

    let duplicate = backend
        .run(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&input)))
        .unwrap()
        .unwrap();
    let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
    assert_eq!(duplicate_domain, domain);
    assert_ne!(duplicate_allocation, allocation);
}

#[test]
fn cuda_provider_does_not_expose_safe_unscoped_raw_access() {
    let lib = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("CUDA provider source must be readable");
    assert!(
        !lib.contains("pub fn handle(") && !lib.contains("pub fn device_ptr("),
        "CUDA provider handles and pointers must not be safe unscoped accessors"
    );
    // Device pointers are reachable only through the raw session's `unsafe fn raw_ptr` on a
    // session-lifetime-scoped TensorRef/TensorMut.
    let raw = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda/raw.rs"))
        .expect("the raw session source must be readable");
    assert!(
        !raw.contains("pub fn with_typed_device_ptr") && !raw.contains("pub fn typed_device_ptr("),
        "the raw session must not expose a typed device-pointer function"
    );
}
