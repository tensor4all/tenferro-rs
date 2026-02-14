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

/// Logical memory space where tensor data resides.
///
/// Separates the concept of "where data lives" from "which hardware
/// computes on it". A tensor on [`MainMemory`](LogicalMemorySpace::MainMemory)
/// can be processed by any CPU, while a tensor on
/// [`GpuMemory`](LogicalMemorySpace::GpuMemory) can be processed by any
/// compute device with access to that GPU memory space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    /// System main memory (CPU-accessible RAM).
    MainMemory,
    /// GPU-resident memory identified by a space ID.
    GpuMemory {
        /// Logical GPU memory space identifier.
        space_id: usize,
    },
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
    /// AMD HIP compute device.
    Hip {
        /// Zero-based HIP device index.
        device_id: usize,
    },
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeDevice::Cpu { device_id } => write!(f, "cpu:{device_id}"),
            ComputeDevice::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            ComputeDevice::Hip { device_id } => write!(f, "hip:{device_id}"),
        }
    }
}

/// Classification of tensor operations, used to query preferred compute
/// devices for a given operation on a given memory space.
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

/// Return the preferred compute devices for a given operation on a memory space.
///
/// The returned list is ordered by preference (most preferred first).
///
/// # Errors
///
/// Returns [`Error::NoCompatibleComputeDevice`] if no compute device can
/// execute the given operation on the specified memory space.
pub fn preferred_compute_devices(
    _space: LogicalMemorySpace,
    _op_kind: OpKind,
) -> Result<Vec<ComputeDevice>> {
    todo!()
}

/// Error type used across the tenferro workspace.
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

    /// An error propagated from strided-view operations.
    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
