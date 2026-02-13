//! Device abstraction and shared error types for the tenferro workspace.
//!
//! This crate provides:
//! - [`Device`] enum representing compute devices (CPU, CUDA, HIP)
//! - [`Error`] and [`Result`] types used across all tenferro crates
//!
//! # Examples
//!
//! ```
//! use tenferro_device::Device;
//!
//! let dev = Device::Cpu;
//! ```

use std::fmt;

/// Compute device on which tensor data resides.
///
/// Currently only [`Device::Cpu`] is functional.
/// [`Device::Cuda`] and [`Device::Hip`] are defined for future GPU support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device.
    Cpu,
    /// NVIDIA CUDA device with a specific device ID.
    Cuda {
        /// Zero-based CUDA device index.
        device_id: usize,
    },
    /// AMD HIP device with a specific device ID.
    Hip {
        /// Zero-based HIP device index.
        device_id: usize,
    },
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            Device::Hip { device_id } => write!(f, "hip:{device_id}"),
        }
    }
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

    /// An invalid argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// An error propagated from strided-view operations.
    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
