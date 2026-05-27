use super::*;

#[test]
fn test_preferred_compute_devices_main_memory() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::MainMemory, OpKind::BatchedGemm).unwrap();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0], ComputeDevice::Cpu { device_id: 0 });
}

#[test]
fn test_preferred_compute_devices_main_memory_all_ops() {
    for op in [
        OpKind::Contract,
        OpKind::BatchedGemm,
        OpKind::Reduce,
        OpKind::Trace,
        OpKind::Permute,
        OpKind::ElementwiseMul,
    ] {
        let devices = preferred_compute_devices(LogicalMemorySpace::MainMemory, op).unwrap();
        assert!(
            devices.contains(&ComputeDevice::Cpu { device_id: 0 }),
            "CPU should be available for {:?}",
            op
        );
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_preferred_compute_devices_gpu_memory_without_cuda_feature() {
    let result = preferred_compute_devices(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        OpKind::BatchedGemm,
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        Error::NoCompatibleComputeDevice { space, op } => {
            assert_eq!(space, LogicalMemorySpace::GpuMemory { device_id: 0 });
            assert_eq!(op, OpKind::BatchedGemm);
        }
        _ => panic!("Expected NoCompatibleComputeDevice error"),
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_preferred_compute_devices_pinned_memory_without_cuda_feature() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::PinnedMemory, OpKind::BatchedGemm).unwrap();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0], ComputeDevice::Cpu { device_id: 0 });
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_preferred_compute_devices_managed_memory_without_cuda_feature() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::ManagedMemory, OpKind::BatchedGemm).unwrap();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0], ComputeDevice::Cpu { device_id: 0 });
}

#[cfg(feature = "cuda")]
#[test]
fn test_preferred_compute_devices_gpu_memory_with_cuda_available() {
    if cuda_device_count() > 0 {
        let devices = preferred_compute_devices(
            LogicalMemorySpace::GpuMemory { device_id: 0 },
            OpKind::BatchedGemm,
        )
        .unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0], ComputeDevice::Cuda { device_id: 0 });
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_preferred_compute_devices_gpu_memory_invalid_device() {
    let invalid_device_id = cuda_device_count() + 100;
    let result = preferred_compute_devices(
        LogicalMemorySpace::GpuMemory {
            device_id: invalid_device_id,
        },
        OpKind::BatchedGemm,
    );
    assert!(result.is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn test_preferred_compute_devices_pinned_memory_with_cuda() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::PinnedMemory, OpKind::BatchedGemm).unwrap();
    if cuda_device_count() > 0 {
        assert!(
            devices.contains(&ComputeDevice::Cuda { device_id: 0 }),
            "CUDA device should be preferred for pinned memory"
        );
    }
    assert!(
        devices.contains(&ComputeDevice::Cpu { device_id: 0 }),
        "CPU should be fallback for pinned memory"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_preferred_compute_devices_managed_memory_with_cuda() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::ManagedMemory, OpKind::BatchedGemm).unwrap();
    if cuda_device_count() > 0 {
        assert!(
            devices.contains(&ComputeDevice::Cuda { device_id: 0 }),
            "CUDA device should be preferred for managed memory"
        );
    }
    assert!(
        devices.contains(&ComputeDevice::Cpu { device_id: 0 }),
        "CPU should be fallback for managed memory"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_device_count_safe() {
    let count = cuda_device_count();
    assert!(count < 1000, "device count should be reasonable");
}

#[test]
fn test_compute_device_display() {
    assert_eq!(format!("{}", ComputeDevice::Cpu { device_id: 0 }), "cpu:0");
    assert_eq!(
        format!("{}", ComputeDevice::Cuda { device_id: 2 }),
        "cuda:2"
    );
    assert_eq!(
        format!("{}", ComputeDevice::Rocm { device_id: 1 }),
        "rocm:1"
    );
}

#[test]
fn test_logical_memory_space_equality() {
    assert_eq!(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert_ne!(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        LogicalMemorySpace::GpuMemory { device_id: 1 }
    );
    assert_ne!(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        LogicalMemorySpace::MainMemory
    );
}

#[test]
fn test_error_display() {
    let err = Error::NoCompatibleComputeDevice {
        space: LogicalMemorySpace::GpuMemory { device_id: 0 },
        op: OpKind::BatchedGemm,
    };
    let msg = err.to_string();
    assert!(msg.contains("no compatible compute device"));
    assert!(msg.contains("BatchedGemm"));
}
