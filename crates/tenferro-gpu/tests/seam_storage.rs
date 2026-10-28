#![cfg(feature = "cuda")]

//! U1 seam storage evidence: identity, allocation freshness and the absence of a safe
//! unscoped pointer accessor on the new path (issue #2208).
//!
//! The device tests are `#[ignore]`-free but fail with a typed error on a host without
//! CUDA; `gpu_available()` guards them so the source-level checks still run anywhere.

use std::fs;
use std::path::Path;

use tenferro_gpu::cuda::{gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{AllocationDomainId, AllocationId, Tensor};

fn identity(tensor: &Tensor) -> (Option<AllocationDomainId>, Option<AllocationId>) {
    let tensor = tensor
        .as_typed::<f32>()
        .unwrap_or_else(|| panic!("seam storage tests use f32 tensors"));
    (tensor.allocation_domain(), tensor.allocation_id())
}

#[test]
#[ignore = "requires CUDA"]
fn seam_upload_keeps_the_single_root_identity() {
    if !gpu_available() {
        return;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).expect("host tensor");
    let input = upload_tensor(engine.runtime(), &host).expect("upload");
    let (domain, allocation) = identity(&input);
    assert!(domain.is_some(), "the allocation carries its domain");
    assert!(allocation.is_some(), "the allocation carries an id");

    let tensor = input.as_typed::<f32>().expect("f32 tensor");
    let view = tensor.as_view();
    assert_eq!(view.allocation_domain(), domain);
    assert_eq!(view.allocation_id(), allocation);
}

#[test]
#[ignore = "requires CUDA"]
fn seam_upload_allocates_a_fresh_allocation_per_value() {
    if !gpu_available() {
        return;
    }
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).expect("host tensor");
    let first = upload_tensor(engine.runtime(), &host).expect("first upload");
    let second = upload_tensor(engine.runtime(), &host).expect("second upload");
    let (first_domain, first_allocation) = identity(&first);
    let (second_domain, second_allocation) = identity(&second);
    assert_eq!(first_domain, second_domain, "one runtime, one domain");
    assert_ne!(
        first_allocation, second_allocation,
        "every upload is its own allocation"
    );
}

#[test]
#[ignore = "requires CUDA"]
fn seam_engine_is_send_and_reports_its_device() {
    if !gpu_available() {
        return;
    }
    fn assert_send<T: Send>(_: &T) {}
    let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    assert_send(&engine);
    assert_eq!(engine.runtime().device_ordinal(), 0);
    assert_eq!(engine.runtime().device_id(), CudaDeviceId::from_ordinal(0));
    std::thread::spawn(move || {
        // The engine moves to another thread; submission stays on one owner.
        assert_eq!(engine.runtime().device_ordinal(), 0);
    })
    .join()
    .expect("the engine must move");
}

#[test]
fn seam_provider_does_not_expose_safe_unscoped_pointer_access() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = String::new();
    for entry in [
        "mod.rs",
        "engine.rs",
        "runtime.rs",
        "storage.rs",
        "transfer.rs",
    ] {
        let path = root.join("src/cuda").join(entry);
        sources.push_str(
            &fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}")),
        );
    }
    assert!(
        !sources.contains("pub fn device_ptr(") && !sources.contains("pub fn handle("),
        "the seam must not expose a safe unscoped device pointer or backend handle"
    );
    // The seam's own vocabulary has no pointer accessor either.
    let gpu = fs::read_to_string(root.join("src/gpu/mod.rs")).expect("gpu seam source");
    assert!(
        !gpu.contains("pub fn device_ptr(") && !gpu.contains("pub fn as_ptr("),
        "the backend-neutral seam must not expose a device pointer"
    );
}
