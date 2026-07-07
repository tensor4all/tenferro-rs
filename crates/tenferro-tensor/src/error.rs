//! Runtime error types for tensor execution.
//!
//! # Examples
//!
//! ```rust
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
/// ```rust
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
    #[error("{op}: unsupported dtype conversion from {from:?} to {to:?}: {message}")]
    UnsupportedDTypeConversion {
        op: &'static str,
        from: crate::DType,
        to: crate::DType,
        message: String,
    },
    #[error("{backend} backend does not support {op} for dtype {dtype:?}")]
    UnsupportedOpDType {
        op: &'static str,
        dtype: crate::DType,
        backend: crate::BackendId,
    },
    #[error("{op}: division by zero for dtype {dtype:?}")]
    DivisionByZero {
        op: &'static str,
        dtype: crate::DType,
    },
    #[error("{op}: negative integer exponent for dtype {dtype:?}")]
    NegativeIntegerExponent {
        op: &'static str,
        dtype: crate::DType,
    },
    #[error("{op}: invalid config: {message}")]
    InvalidConfig { op: &'static str, message: String },
    #[error("extension family {family_id:?} has no host reference implementation")]
    NoHostReference { family_id: &'static str },
    #[error("{op}: backend failure: {message}")]
    BackendFailure { op: &'static str, message: String },
    #[error("missing runtime value for slot {slot}")]
    MissingValue { slot: usize },
}

impl Error {
    /// Construct a backend failure error while preserving the operation name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Error;
    ///
    /// let err = Error::backend_failure("matmul", "backend rejected launch");
    /// assert!(matches!(
    ///     err,
    ///     Error::BackendFailure {
    ///         op: "matmul",
    ///         ref message,
    ///     } if message == "backend rejected launch"
    /// ));
    /// ```
    pub fn backend_failure(op: &'static str, message: impl std::fmt::Display) -> Self {
        Self::BackendFailure {
            op,
            message: message.to_string(),
        }
    }

    /// Construct a structured unsupported operation/dtype error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{BackendId, DType, Error};
    ///
    /// let err = Error::unsupported_op_dtype("add", DType::Bool, BackendId::Cuda);
    /// assert!(matches!(err, Error::UnsupportedOpDType { backend: BackendId::Cuda, .. }));
    /// ```
    pub fn unsupported_op_dtype(
        op: &'static str,
        dtype: crate::DType,
        backend: crate::BackendId,
    ) -> Self {
        Self::UnsupportedOpDType { op, dtype, backend }
    }

    /// Construct a structured division-by-zero domain error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Error};
    ///
    /// let err = Error::division_by_zero("div", DType::I32);
    /// assert!(matches!(err, Error::DivisionByZero { op: "div", .. }));
    /// ```
    pub fn division_by_zero(op: &'static str, dtype: crate::DType) -> Self {
        Self::DivisionByZero { op, dtype }
    }

    /// Construct a structured negative-integer-exponent domain error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Error};
    ///
    /// let err = Error::negative_integer_exponent("pow", DType::I64);
    /// assert!(matches!(err, Error::NegativeIntegerExponent { op: "pow", .. }));
    /// ```
    pub fn negative_integer_exponent(op: &'static str, dtype: crate::DType) -> Self {
        Self::NegativeIntegerExponent { op, dtype }
    }
}

/// Result type alias for runtime tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Error, Result};
///
/// let output: Result<()> = Err(Error::MissingValue { slot: 0 });
/// ```
pub type Result<T> = std::result::Result<T, Error>;
