//! Error types owned by the einsum crate.
//!
//! Parsing and planning remain einsum-domain concerns, while shared tensor
//! validation is represented by the common validation vocabulary. At erased
//! runtime boundaries [`Error::into_tensor_error`] preserves this distinction
//! and keeps the original einsum error as a typed source.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_einsum::Error;
//! use tenferro_tensor::{ErrorKind, ValidationKind};
//!
//! let err = Error::invalid_subscripts("missing output arrow");
//! assert_eq!(err.kind(), ErrorKind::Validation(ValidationKind::InvalidArgument));
//! ```

use tenferro_tensor::{DType, ErrorKind, ShapeMismatch, ShapeVec, ValidationError, ValidationKind};

use crate::EINSUM_EXTENSION_FAMILY_ID;

/// Domain-specific cause of an einsum planning failure.
///
/// Caller-controlled expressions, shapes, and optimizer options use
/// [`PlanningError::InvalidConfiguration`]. Runtime-state classification is
/// reserved for an unavailable or poisoned planner state.
///
/// # Examples
///
/// ```rust
/// use tenferro_einsum::{Error, PlanningError};
///
/// let error = Error::planning("the requested path is invalid");
/// assert!(matches!(
///     error,
///     Error::Planning {
///         source: PlanningError::InvalidConfiguration { .. }
///     }
/// ));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    /// The requested expression, path, or planner option is invalid.
    #[error("invalid einsum planning configuration: {message}")]
    InvalidConfiguration {
        /// Human-readable configuration detail.
        message: String,
    },
    /// Planner state required by a valid request is unavailable.
    #[error("einsum planning runtime state unavailable: {message}")]
    RuntimeState {
        /// Human-readable state detail.
        message: String,
    },
}

/// Errors produced while parsing, planning, lowering, or executing einsum
/// expressions.
///
/// # Examples
///
/// ```rust
/// use tenferro_einsum::Error;
/// use tenferro_tensor::{ErrorKind, ShapeMismatch, ShapeVec, ValidationKind};
///
/// let err = Error::validation(
///     "einsum",
///     ShapeMismatch::ExpectedActual {
///         expected: ShapeVec::from_vec(vec![2, 3]),
///         actual: ShapeVec::from_vec(vec![2, 4]),
///     }
///     .into(),
/// );
/// assert_eq!(err.kind(), ErrorKind::Validation(ValidationKind::ShapeMismatch));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A shared tensor validation fact discovered by an einsum operation.
    #[error("{op}: {source}")]
    Validation {
        /// Public operation name.
        op: &'static str,
        /// Machine-readable validation payload.
        #[source]
        source: ValidationError,
    },

    /// Einsum notation is malformed or cannot be parsed.
    #[error("invalid einsum subscripts: {message}")]
    InvalidSubscripts {
        /// Human-readable parser detail.
        message: String,
    },

    /// No valid contraction plan could be constructed for the supplied
    /// expression or optimizer configuration.
    #[error("einsum planning failed: {source}")]
    Planning {
        /// Typed planning-domain cause.
        #[source]
        source: PlanningError,
    },

    /// A numerical contraction or backend accumulation failed to converge.
    #[error("einsum numerical failure: {message}")]
    Numerical {
        /// Human-readable numerical detail.
        message: String,
    },

    /// A concrete tensor/backend operation failed.
    #[error(transparent)]
    Tensor(#[from] tenferro_tensor::Error),

    /// Graph construction or extension execution failed in the runtime.
    #[error(transparent)]
    Runtime(#[from] tenferro_runtime::Error),
}

impl Error {
    /// Construct a shared validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    /// use tenferro_tensor::ValidationError;
    ///
    /// let error = Error::validation("einsum", ValidationError::RankMismatch {
    ///     expected: 2,
    ///     actual: 1,
    /// });
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn validation(op: &'static str, source: ValidationError) -> Self {
        Self::Validation { op, source }
    }

    /// Construct an invalid-argument validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::invalid_argument("einsum", "inputs", "at least one input is required");
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn invalid_argument(
        op: &'static str,
        argument: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::validation(
            op,
            ValidationError::InvalidArgument {
                argument,
                message: message.into(),
            },
        )
    }

    /// Construct a shape-mismatch validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::shape_mismatch("einsum", [2, 3], [2, 4]);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn shape_mismatch(
        op: &'static str,
        expected: impl Into<Vec<usize>>,
        actual: impl Into<Vec<usize>>,
    ) -> Self {
        Self::validation(
            op,
            ShapeMismatch::ExpectedActual {
                expected: ShapeVec::from_vec(expected.into()),
                actual: ShapeVec::from_vec(actual.into()),
            }
            .into(),
        )
    }

    /// Construct a dtype-mismatch validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    /// use tenferro_tensor::DType;
    ///
    /// let error = Error::dtype_mismatch("einsum", DType::F32, DType::F64);
    /// assert!(matches!(error, Error::Tensor(_)));
    /// ```
    pub fn dtype_mismatch(op: &'static str, expected: DType, actual: DType) -> Self {
        Self::Tensor(tenferro_tensor::Error::dtype_mismatch(op, expected, actual))
    }

    /// Construct a rank-mismatch validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::rank_mismatch("einsum", 2, 1);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn rank_mismatch(op: &'static str, expected: usize, actual: usize) -> Self {
        Self::validation(op, ValidationError::RankMismatch { expected, actual })
    }

    /// Construct an invalid-notation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::invalid_subscripts("missing `->`");
    /// assert!(matches!(error, Error::InvalidSubscripts { .. }));
    /// ```
    pub fn invalid_subscripts(message: impl Into<String>) -> Self {
        Self::InvalidSubscripts {
            message: message.into(),
        }
    }

    /// Construct a planning failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::planning("no contraction path");
    /// assert!(matches!(error, Error::Planning { .. }));
    /// ```
    pub fn planning(message: impl Into<String>) -> Self {
        Self::Planning {
            source: PlanningError::InvalidConfiguration {
                message: message.into(),
            },
        }
    }

    /// Construct a planning failure caused by unavailable planner state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::{Error, PlanningError};
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let error = Error::planning_runtime_state("planner lock is poisoned");
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// assert!(matches!(
    ///     error,
    ///     Error::Planning {
    ///         source: PlanningError::RuntimeState { .. }
    ///     }
    /// ));
    /// ```
    pub fn planning_runtime_state(message: impl Into<String>) -> Self {
        Self::Planning {
            source: PlanningError::RuntimeState {
                message: message.into(),
            },
        }
    }

    /// Construct a numerical failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    ///
    /// let error = Error::numerical("contraction did not converge");
    /// assert!(matches!(error, Error::Numerical { .. }));
    /// ```
    pub fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical {
            message: message.into(),
        }
    }

    /// Return the stable coarse classification of this einsum failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    /// use tenferro_tensor::{ErrorKind, ValidationKind};
    ///
    /// assert_eq!(
    ///     Error::invalid_subscripts("bad").kind(),
    ///     ErrorKind::Validation(ValidationKind::InvalidArgument),
    /// );
    /// ```
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Validation { source, .. } => ErrorKind::Validation(source.kind()),
            Self::InvalidSubscripts { .. } => {
                ErrorKind::Validation(ValidationKind::InvalidArgument)
            }
            Self::Planning { source } => match source {
                PlanningError::InvalidConfiguration { .. } => {
                    ErrorKind::Validation(ValidationKind::InvalidArgument)
                }
                PlanningError::RuntimeState { .. } => ErrorKind::RuntimeState,
            },
            Self::Numerical { .. } => ErrorKind::NumericalFailure,
            Self::Tensor(error) => error.kind(),
            Self::Runtime(error) => error.kind(),
        }
    }

    /// Promote this error to the tensor error used by a type-erased extension
    /// boundary without formatting away its typed source.
    ///
    /// Shared validation is promoted directly. All crate-local and nested
    /// errors remain a boxed source under the einsum extension family.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::error::Error as _;
    /// use tenferro_einsum::Error;
    /// use tenferro_tensor::{Error as TensorError, ErrorKind, ValidationKind};
    ///
    /// let tensor_error = Error::planning("no valid contraction path")
    ///     .into_tensor_error("einsum_extension");
    /// assert_eq!(
    ///     tensor_error.kind(),
    ///     ErrorKind::Validation(ValidationKind::InvalidArgument)
    /// );
    /// assert!(matches!(tensor_error, TensorError::Extension { .. }));
    /// assert!(tensor_error.source().is_some());
    /// ```
    #[must_use]
    pub fn into_tensor_error(self, op: &'static str) -> tenferro_tensor::Error {
        match self {
            Self::Validation { op, source } => tenferro_tensor::Error::validation(op, source),
            Self::Tensor(error) => error,
            error => {
                let kind = error.kind();
                tenferro_tensor::Error::extension(op, EINSUM_EXTENSION_FAMILY_ID, kind, error)
            }
        }
    }
}

/// Result type alias for einsum parsing, planning, and all public extension
/// APIs.
pub type Result<T> = std::result::Result<T, Error>;
