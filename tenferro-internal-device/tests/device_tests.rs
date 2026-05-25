//! Tests for tenferro-internal-device: ComputeDevice Display, Error construction,
//! preferred_compute_devices.

use tenferro_device::{
    preferred_compute_devices, ComputeDevice, Error, LogicalMemorySpace, OpKind,
};

// ============================================================================
// ComputeDevice Display formatting
// ============================================================================

#[test]
fn compute_device_cpu_display() {
    assert_eq!(format!("{}", ComputeDevice::Cpu { device_id: 0 }), "cpu:0");
}

#[test]
fn compute_device_cpu_display_nonzero_id() {
    assert_eq!(format!("{}", ComputeDevice::Cpu { device_id: 3 }), "cpu:3");
}

#[test]
fn compute_device_cuda_display() {
    assert_eq!(
        format!("{}", ComputeDevice::Cuda { device_id: 0 }),
        "cuda:0"
    );
}

#[test]
fn compute_device_rocm_display() {
    assert_eq!(
        format!("{}", ComputeDevice::Rocm { device_id: 1 }),
        "rocm:1"
    );
}

// ============================================================================
// ComputeDevice equality
// ============================================================================

#[test]
fn compute_device_equality() {
    assert_eq!(
        ComputeDevice::Cpu { device_id: 0 },
        ComputeDevice::Cpu { device_id: 0 },
    );
}

#[test]
fn compute_device_inequality_different_id() {
    assert_ne!(
        ComputeDevice::Cpu { device_id: 0 },
        ComputeDevice::Cpu { device_id: 1 },
    );
}

#[test]
fn compute_device_inequality_different_variant() {
    assert_ne!(
        ComputeDevice::Cpu { device_id: 0 },
        ComputeDevice::Cuda { device_id: 0 },
    );
}

// ============================================================================
// LogicalMemorySpace equality
// ============================================================================

#[test]
fn memory_space_main_memory_eq() {
    assert_eq!(
        LogicalMemorySpace::MainMemory,
        LogicalMemorySpace::MainMemory
    );
}

#[test]
fn memory_space_gpu_eq_same_device() {
    assert_eq!(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        LogicalMemorySpace::GpuMemory { device_id: 0 },
    );
}

#[test]
fn memory_space_gpu_neq_different_device() {
    assert_ne!(
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        LogicalMemorySpace::GpuMemory { device_id: 1 },
    );
}

// ============================================================================
// Error type construction and display
// ============================================================================

#[test]
fn error_invalid_argument_display() {
    let err = Error::InvalidArgument("bad index".into());
    assert!(err.to_string().contains("bad index"));
}

#[test]
fn error_shape_mismatch_display() {
    let err = Error::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![4, 5],
    };
    let msg = err.to_string();
    assert!(msg.contains("shape mismatch"));
}

#[test]
fn error_rank_mismatch_display() {
    let err = Error::RankMismatch {
        expected: 2,
        got: 3,
    };
    let msg = err.to_string();
    assert!(msg.contains("rank mismatch"));
    assert!(msg.contains('2'));
    assert!(msg.contains('3'));
}

#[test]
fn error_device_error_display() {
    let err = Error::DeviceError("GPU out of memory".into());
    assert!(err.to_string().contains("GPU out of memory"));
}

#[test]
fn error_no_compatible_device_display() {
    let err = Error::NoCompatibleComputeDevice {
        space: LogicalMemorySpace::MainMemory,
        op: OpKind::BatchedGemm,
    };
    assert!(err.to_string().contains("no compatible compute device"));
}

#[test]
fn error_cross_memory_space_display() {
    let err = Error::CrossMemorySpaceOperation {
        left: LogicalMemorySpace::MainMemory,
        right: LogicalMemorySpace::GpuMemory { device_id: 0 },
    };
    assert!(err.to_string().contains("cross-memory-space"));
}

#[test]
fn error_stride_error_display() {
    let err = Error::StrideError("negative stride not supported".into());
    assert!(err.to_string().contains("stride error"));
}

// ============================================================================
// OpKind Debug formatting
// ============================================================================

#[test]
fn op_kind_debug() {
    assert_eq!(format!("{:?}", OpKind::BatchedGemm), "BatchedGemm");
    assert_eq!(format!("{:?}", OpKind::Contract), "Contract");
    assert_eq!(format!("{:?}", OpKind::Reduce), "Reduce");
    assert_eq!(format!("{:?}", OpKind::Trace), "Trace");
    assert_eq!(format!("{:?}", OpKind::Permute), "Permute");
    assert_eq!(format!("{:?}", OpKind::ElementwiseMul), "ElementwiseMul");
}

// ============================================================================
// preferred_compute_devices
// ============================================================================

#[test]
fn preferred_devices_main_memory_returns_cpu() {
    let devices =
        preferred_compute_devices(LogicalMemorySpace::MainMemory, OpKind::BatchedGemm).unwrap();
    assert_eq!(devices, vec![ComputeDevice::Cpu { device_id: 0 }]);
}

#[test]
fn preferred_devices_main_memory_all_op_kinds() {
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
            "MainMemory should support {op:?}"
        );
    }
}

#[test]
fn preferred_devices_gpu_memory_no_backend() {
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
        other => panic!("expected NoCompatibleComputeDevice, got {other:?}"),
    }
}

#[test]
fn preferred_devices_pinned_memory_no_backend() {
    let result = preferred_compute_devices(LogicalMemorySpace::PinnedMemory, OpKind::BatchedGemm);
    let devices = result.expect("PinnedMemory should return CPU as fallback without CUDA feature");
    assert!(
        devices.contains(&ComputeDevice::Cpu { device_id: 0 }),
        "PinnedMemory should support CPU as fallback: {devices:?}"
    );
}

#[test]
fn preferred_devices_managed_memory_no_backend() {
    let result = preferred_compute_devices(LogicalMemorySpace::ManagedMemory, OpKind::Contract);
    let devices = result.expect("ManagedMemory should return CPU as fallback without CUDA feature");
    assert!(
        devices.contains(&ComputeDevice::Cpu { device_id: 0 }),
        "ManagedMemory should support CPU as fallback: {devices:?}"
    );
}
