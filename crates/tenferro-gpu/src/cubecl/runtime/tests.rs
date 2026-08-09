use super::{gpu_available, is_invalid_device_lookup, CudaRuntime, CudaRuntimeIdentity};
use crate::cuda::CudaBackend;
use cudarc::driver::{result::DriverError, sys::CUresult};

#[test]
fn selected_device_lookup_classifies_only_cuda_invalid_device() {
    assert!(is_invalid_device_lookup(DriverError(
        CUresult::CUDA_ERROR_INVALID_DEVICE
    )));
    assert!(!is_invalid_device_lookup(DriverError(
        CUresult::CUDA_ERROR_INVALID_VALUE
    )));
}

#[test]
fn cuda_runtime_identity_is_clone_stable_and_instance_scoped() {
    let first = CudaRuntimeIdentity::fresh();
    let first_key = first.cache_discriminator();
    let clone = first.clone();
    let moved = clone;
    let independent = CudaRuntimeIdentity::fresh();

    assert_eq!(first, moved);
    assert_eq!(first_key, moved.cache_discriminator());
    assert_ne!(first, independent);
    assert_ne!(first_key, independent.cache_discriminator());
}

#[test]
fn cuda_backend_identity_tracks_the_exact_runtime_when_hardware_is_available() {
    if !gpu_available() {
        return;
    }

    let device = super::cuda_devices()
        .expect("CUDA device discovery should succeed")
        .into_iter()
        .next()
        .expect("CUDA device should be available")
        .id();
    let first = CudaBackend::new(device).expect("CUDA backend should initialize");
    let clone = first.clone();
    let independent = CudaBackend::new(device).expect("second CUDA backend should initialize");

    let first_identity = first.runtime_identity();
    let clone_identity = clone.runtime_identity();
    let independent_identity = independent.runtime_identity();
    assert_eq!(first_identity, clone_identity);
    assert_eq!(
        first_identity.cache_discriminator(),
        clone_identity.cache_discriminator()
    );
    assert_ne!(first_identity, independent_identity);
    assert_ne!(
        first_identity.cache_discriminator(),
        independent_identity.cache_discriminator()
    );

    let runtime_clone = first.runtime().clone();
    let _: CudaRuntime = runtime_clone;
}
