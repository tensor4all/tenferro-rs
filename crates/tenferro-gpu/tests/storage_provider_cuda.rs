#![cfg(feature = "cuda")]

//! Final P7 CUDA storage-provider evidence.
//!
//! These tests are hardware-gated because the contract concerns a real CUDA
//! allocation root and the provider-native prepared binding. They are no-ops
//! on machines without an available CUDA device; source-level API contracts
//! remain covered by the integration contract tests.

use std::fs;
use std::path::Path;

use tenferro_gpu::{
    cuda::gpu_available, cuda::upload_tensor, cuda::with_cuda_exec_session, cuda::CudaBackend,
    cuda::CudaDeviceId,
};
use tenferro_tensor::backend::BackendSessionHost as _;
use tenferro_tensor::{AllocationDomainId, AllocationId, Tensor, TensorRead, TensorStructural};

fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
    let Tensor::F32(tensor) = tensor else {
        panic!("P7 provider tests use f32 tensors")
    };
    (tensor.allocation_domain(), tensor.allocation_id())
}

#[test]
fn cuda_tensor_view_keeps_the_single_root_identity() {
    if !gpu_available() {
        return;
    }

    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let (domain, allocation) = identity(&input);

    let Tensor::F32(tensor) = &input else {
        panic!("provider test uses f32 tensors")
    };
    let view = tensor.as_view();
    assert_eq!(view.allocation_domain(), domain);
    assert_eq!(view.allocation_id(), allocation);
}

#[test]
fn cuda_prepared_state_is_consumed_by_the_exact_binding_without_host_mapping() {
    if !gpu_available() {
        return;
    }

    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let Tensor::F32(tensor) = &input else {
        panic!("provider test uses f32 tensors")
    };

    let prepared = tensor
        .prepare_device_read("storage_provider_cuda")
        .expect("CUDA device preparation");
    assert!(
        tensor.host_data().is_err(),
        "device preparation must not create a host mapping"
    );
    drop(prepared);

    let binding = backend
        .with_backend_session(|session| {
            with_cuda_exec_session(session, |sess| {
                sess.with_cubecl("storage_provider_cuda", |cubecl| {
                    cubecl.tensor_binding(tensor, "storage_provider_cuda")
                })
            })
            .expect("CUDA session")
        })
        .expect("the CUDA binding must consume the provider-prepared state");
    drop(binding);
}

#[test]
fn cuda_duplicate_is_explicit_same_placement_allocation() {
    if !gpu_available() {
        return;
    }

    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let input = upload_tensor(backend.runtime(), &host).unwrap();
    let (domain, allocation) = identity(&input);

    let duplicate = backend
        .to_contiguous_read(TensorRead::from_tensor(&input))
        .unwrap();
    let (duplicate_domain, duplicate_allocation) = identity(&duplicate);
    assert_eq!(duplicate_domain, domain);
    assert_ne!(duplicate_allocation, allocation);
}

#[test]
fn cuda_provider_does_not_expose_safe_unscoped_raw_access() {
    let lib = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("CUDA provider source must be readable");
    let interop =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cubecl/interop.rs"))
            .expect("CUDA interop source must be readable");

    assert!(
        !lib.contains("pub fn handle(") && !lib.contains("pub fn device_ptr("),
        "CUDA provider handles and pointers must not be safe unscoped accessors"
    );
    assert!(
        interop.contains("pub fn with_typed_device_ptr")
            && !interop.contains("pub fn typed_device_ptr("),
        "raw CUDA access must stay inside an explicit callback boundary"
    );
}
