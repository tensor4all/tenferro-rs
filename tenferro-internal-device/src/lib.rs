//! Device abstraction and shared error types for the tenferro workspace.
//!
//! This crate provides:
//! - [`LogicalMemorySpace`] enum representing where tensor data resides
//! - [`ComputeDevice`] enum representing hardware compute devices
//! - [`OpKind`] enum classifying tensor operations for device selection
//! - [`preferred_compute_devices`] for querying compatible devices
//! - [`Error`] and [`Result`] types used across all tenferro crates
//!
//! # Examples
//!
//! ```
//! use tenferro_device::{LogicalMemorySpace, ComputeDevice};
//!
//! let space = LogicalMemorySpace::MainMemory;
//! let dev = ComputeDevice::Cpu { device_id: 0 };
//! assert_eq!(format!("{dev}"), "cpu:0");
//! ```

use std::fmt;

mod batch_index;
#[cfg(feature = "cuda")]
pub mod cuda;
mod generator;

/// Logical memory space where tensor data resides.
///
/// Separates the concept of "where data lives" from "which hardware
/// computes on it". A tensor on [`MainMemory`](LogicalMemorySpace::MainMemory)
/// can be processed by any CPU, while a tensor on
/// [`GpuMemory`](LogicalMemorySpace::GpuMemory) can be processed by any
/// compute device with access to that GPU memory space.
///
/// The variants align with DLPack `DLDeviceType` constants:
///
/// | Variant | DLPack `device_type` |
/// |---------|---------------------|
/// | `MainMemory` | `kDLCPU` (1) |
/// | `PinnedMemory` | `kDLCUDAHost` (3) / `kDLROCMHost` (11) |
/// | `GpuMemory` | `kDLCUDA` (2) / `kDLROCM` (10) — vendor determined by context |
/// | `ManagedMemory` | `kDLCUDAManaged` (13) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    /// System main memory (CPU-accessible RAM).
    ///
    /// Corresponds to DLPack `kDLCPU` (`device_id = 0`).
    MainMemory,
    /// Host memory pinned for fast GPU transfer (`cudaMallocHost` / `hipMallocHost`).
    ///
    /// Accessible from the CPU but optimized for DMA transfer to/from
    /// a specific GPU. Corresponds to DLPack `kDLCUDAHost` or `kDLROCMHost`
    /// (`device_id = 0`).
    PinnedMemory,
    /// GPU-resident memory identified by a device ID.
    ///
    /// Corresponds to DLPack `kDLCUDA` or `kDLROCM` with the given `device_id`.
    GpuMemory {
        /// Zero-based GPU device index (matches DLPack `device_id`).
        device_id: usize,
    },
    /// CUDA Unified/Managed memory (`cudaMallocManaged`).
    ///
    /// Accessible from both CPU and all GPUs. The CUDA runtime handles
    /// page migration automatically. Corresponds to DLPack `kDLCUDAManaged`
    /// (`device_id = 0`).
    ManagedMemory,
}

/// Compute device that can execute tensor operations.
///
/// Unlike [`LogicalMemorySpace`], which describes where data resides,
/// `ComputeDevice` identifies the hardware that performs the computation.
/// Multiple compute devices may share access to the same memory space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    /// CPU compute device.
    Cpu {
        /// Zero-based CPU device index (0 = default global thread pool).
        device_id: usize,
    },
    /// NVIDIA CUDA compute device.
    Cuda {
        /// Zero-based CUDA device index.
        device_id: usize,
    },
    /// AMD ROCm compute device.
    Rocm {
        /// Zero-based ROCm device index.
        device_id: usize,
    },
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeDevice::Cpu { device_id } => write!(f, "cpu:{device_id}"),
            ComputeDevice::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            ComputeDevice::Rocm { device_id } => write!(f, "rocm:{device_id}"),
        }
    }
}

/// Classification of tensor operations, used to query preferred compute
/// devices for a given operation on a given memory space.
///
/// # Examples
///
/// ```
/// use tenferro_device::OpKind;
///
/// let op = OpKind::BatchedGemm;
/// assert_eq!(format!("{op:?}"), "BatchedGemm");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// General tensor contraction.
    Contract,
    /// Batched GEMM (matrix-matrix multiply).
    BatchedGemm,
    /// Reduction (sum, max, min) over one or more modes.
    Reduce,
    /// Trace (diagonal contraction of paired modes).
    Trace,
    /// Mode permutation (transpose).
    Permute,
    /// Element-wise multiplication.
    ElementwiseMul,
}

/// Return the preferred compute devices for a given memory space.
///
/// Returns compute devices that can execute operations on tensors in the
/// specified memory space. The selection follows these rules:
///
/// - [`MainMemory`](LogicalMemorySpace::MainMemory): Returns CPU devices
/// - [`GpuMemory`](LogicalMemorySpace::GpuMemory): Returns the corresponding
///   CUDA device (when compiled with `cuda` feature and device is available)
/// - [`PinnedMemory`](LogicalMemorySpace::PinnedMemory): Prefers GPU compute
///   if available, falls back to CPU
/// - [`ManagedMemory`](LogicalMemorySpace::ManagedMemory): Prefers GPU compute
///   if available, falls back to CPU
///
/// The returned list is ordered by preference (most preferred first).
///
/// # Errors
///
/// Returns [`Error::NoCompatibleComputeDevice`] if no compute device can
/// execute the given operation on the specified memory space.
///
/// # Examples
///
/// ```
/// use tenferro_device::{
///     preferred_compute_devices, ComputeDevice, LogicalMemorySpace, OpKind,
/// };
///
/// let devices = preferred_compute_devices(
///     LogicalMemorySpace::MainMemory,
///     OpKind::BatchedGemm,
/// ).unwrap();
///
/// // Typically includes CPU for main memory workloads.
/// assert!(devices.contains(&ComputeDevice::Cpu { device_id: 0 }));
/// ```
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    _op_kind: OpKind,
) -> Result<Vec<ComputeDevice>> {
    match space {
        LogicalMemorySpace::MainMemory => Ok(vec![ComputeDevice::Cpu { device_id: 0 }]),
        LogicalMemorySpace::GpuMemory { device_id } => {
            #[cfg(feature = "cuda")]
            {
                if is_cuda_device_available(device_id) {
                    Ok(vec![ComputeDevice::Cuda { device_id }])
                } else {
                    Err(Error::NoCompatibleComputeDevice {
                        space,
                        op: _op_kind,
                    })
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_id;
                Err(Error::NoCompatibleComputeDevice {
                    space,
                    op: _op_kind,
                })
            }
        }
        LogicalMemorySpace::PinnedMemory | LogicalMemorySpace::ManagedMemory => {
            #[cfg(feature = "cuda")]
            {
                let mut devices = Vec::new();
                if let Some(cuda_device) = first_available_cuda_device() {
                    devices.push(cuda_device);
                }
                devices.push(ComputeDevice::Cpu { device_id: 0 });
                Ok(devices)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(vec![ComputeDevice::Cpu { device_id: 0 }])
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_device_count() -> usize {
    std::panic::catch_unwind(|| {
        cudarc::runtime::result::device::get_count()
            .map(|count| count as usize)
            .unwrap_or(0)
    })
    .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn is_cuda_device_available(device_id: usize) -> bool {
    device_id < cuda_device_count()
}

#[cfg(feature = "cuda")]
fn first_available_cuda_device() -> Option<ComputeDevice> {
    let count = cuda_device_count();
    if count > 0 {
        Some(ComputeDevice::Cuda { device_id: 0 })
    } else {
        None
    }
}

/// Error type used across the tenferro workspace.
///
/// # Examples
///
/// ```
/// use tenferro_device::Error;
///
/// let err = Error::InvalidArgument("bad index".into());
/// assert!(err.to_string().contains("bad index"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tensor shapes are incompatible for the requested operation.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Tensor ranks (number of dimensions) do not match.
    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch {
        /// Expected rank.
        expected: usize,
        /// Actual rank.
        got: usize,
    },

    /// An error occurred on the compute device.
    #[error("device error: {0}")]
    DeviceError(String),

    /// No compute device is compatible with the requested operation
    /// on the given memory space.
    #[error("no compatible compute device for {op:?} on {space:?}")]
    NoCompatibleComputeDevice {
        /// The memory space where the tensor data resides.
        space: LogicalMemorySpace,
        /// The operation that was requested.
        op: OpKind,
    },

    /// Operations on tensors in different memory spaces are not supported
    /// without explicit transfer.
    ///
    /// Reserved for future cross-device operations (e.g., GPU-CPU transfers).
    #[allow(dead_code)]
    #[error("cross-memory-space operation between {left:?} and {right:?}")]
    CrossMemorySpaceOperation {
        /// Memory space of the first operand.
        left: LogicalMemorySpace,
        /// Memory space of the second operand.
        right: LogicalMemorySpace,
    },

    /// An invalid argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// An error related to strided memory layout (invalid strides, out-of-bounds offset, etc.).
    #[error("stride error: {0}")]
    StrideError(String),
}

/// Result type alias using [`Error`].
///
/// # Examples
///
/// ```
/// use tenferro_device::Result;
///
/// fn compute() -> Result<usize> {
///     Ok(42)
/// }
/// assert_eq!(compute().unwrap(), 42);
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[doc(hidden)]
pub use batch_index::{
    broadcast_batch_dims, checked_batch_count, flatten_col_major_index,
    unflatten_col_major_index_into, BroadcastBatchIndexer,
};
pub use generator::{with_default_generator, Generator};

#[cfg(test)]
mod tests;
