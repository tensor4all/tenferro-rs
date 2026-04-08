//! Runtime error types for tensor execution.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_tensor::Error;
//!
//! let err = Error::AxisOutOfBounds {
//!     op: "dot_general",
//!     axis: 2,
//!     rank: 1,
//! };
//! assert!(err.to_string().contains("dot_general"));
//! ```

/// Runtime failures produced by tensor execution backends and helpers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::Error;
///
/// let err = Error::MissingValue { slot: 3 };
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("{op}: axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds {
        op: &'static str,
        axis: usize,
        rank: usize,
    },
    #[error("{op}: duplicate {role} axis {axis}")]
    DuplicateAxis {
        op: &'static str,
        axis: usize,
        role: &'static str,
    },
    #[error("{op}: axis {axis} appears in both {first_role} and {second_role}")]
    AxisRoleConflict {
        op: &'static str,
        axis: usize,
        first_role: &'static str,
        second_role: &'static str,
    },
    #[error("{op}: shape mismatch lhs={lhs:?} rhs={rhs:?}")]
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    #[error("{op}: rank mismatch expected {expected}, actual {actual}")]
    RankMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{op}: dtype mismatch lhs={lhs:?} rhs={rhs:?}")]
    DTypeMismatch {
        op: &'static str,
        lhs: crate::DType,
        rhs: crate::DType,
    },
    #[error("{op}: invalid config: {message}")]
    InvalidConfig { op: &'static str, message: String },
    #[error("{op}: backend failure: {message}")]
    BackendFailure { op: &'static str, message: String },
    #[error("missing runtime value for slot {slot}")]
    MissingValue { slot: usize },
}

/// Result type alias for runtime tensor operations.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{Error, Result};
///
/// let output: Result<()> = Err(Error::MissingValue { slot: 0 });
/// ```
pub type Result<T> = std::result::Result<T, Error>;
