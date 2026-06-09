use thiserror::Error;

/// Error returned by tenferro CubeCL kernel launch helpers.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub(crate) enum CubeclKernelError {
    /// The requested axis is outside the input tensor rank.
    #[error("axis {axis} is out of bounds for rank {rank}")]
    InvalidAxis {
        /// Requested reduction axis.
        axis: usize,
        /// Input tensor rank.
        rank: usize,
    },

    /// The caller supplied an output tensor with a shape that is not keepdims.
    #[error("output shape {actual:?} does not match expected keepdims shape {expected:?}")]
    MismatchOutputShape {
        /// Expected output shape after setting the reduced axis length to one.
        expected: Vec<usize>,
        /// Actual output shape supplied by the caller.
        actual: Vec<usize>,
    },
    /// The selected launch strategy is invalid for the input.
    #[error("invalid reduction strategy: {reason}")]
    InvalidStrategy {
        /// Human-readable reason the strategy cannot be used.
        reason: String,
    },
}

/// Result alias for tenferro CubeCL kernel helpers.
pub(crate) type Result<T> = core::result::Result<T, CubeclKernelError>;
