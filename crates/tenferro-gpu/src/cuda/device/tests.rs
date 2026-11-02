//! Discovery reports an unusable CUDA driver instead of unwinding (#2401).

use super::cuda_devices;
use crate::cuda::CudaDeviceError;

#[test]
fn discovery_reports_an_unloadable_driver_as_a_typed_error() {
    if t4a_cuda_runtime::device_count().is_ok() {
        // A loadable driver takes the real discovery path, which this test does not assert.
        return;
    }
    // `cudarc` panics from its own loader, so a missing guard here is a panic, not a failure.
    let error = cuda_devices().expect_err("discovery must fail without a driver library");
    assert!(matches!(
        &error,
        CudaDeviceError::Discovery {
            operation: "initialize_driver",
            ..
        }
    ));
}
