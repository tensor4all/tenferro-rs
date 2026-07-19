//! Runtime error types for tensor execution.
//!
//! # Examples
//!
//! ```rust
//! let error = tenferro_tensor::Error::shape_mismatch("add", [2], [3]);
//! assert!(matches!(
//!     error,
//!     tenferro_tensor::Error::Validation { op: "add", .. }
//! ));
//! ```

use std::error::Error as StdError;

use tenferro_tensor_core::{ErrorKind, ValidationError};

/// Boxed source used for backend and extension failures whose concrete type is
/// owned by another crate or a vendor API.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// Runtime failures produced by tensor execution backends and helpers.
///
/// Validation failures retain the shared tensor-core payload as a typed source.
/// Backend and extension failures retain opaque typed sources when one exists;
/// text-only vendor failures use [`Error::BackendFailure`].
///
/// # Examples
///
/// ```rust
/// let error = tenferro_tensor::Error::rank_mismatch("reshape", 2, 1);
/// assert!(matches!(
///     error,
///     tenferro_tensor::Error::Validation { op: "reshape", .. }
/// ));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{op}: {source}")]
    Validation {
        op: &'static str,
        #[source]
        source: ValidationError,
    },
    #[error("{op}: unsupported dtype conversion from {from:?} to {to:?}: {message}")]
    UnsupportedDTypeConversion {
        op: &'static str,
        from: crate::DType,
        to: crate::DType,
        message: String,
    },
    #[error("{op}: unsupported dtype {dtype:?}: {message}")]
    UnsupportedDType {
        op: &'static str,
        dtype: crate::DType,
        message: String,
    },
    #[error("{op}: unsupported operation: {message}")]
    Unsupported { op: &'static str, message: String },
    #[error("{op}: backend failure: {message}")]
    BackendFailure { op: &'static str, message: String },
    #[error("{op}: backend failure: {source}")]
    BackendSource {
        op: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("{op}: I/O failure: {source}")]
    IoSource {
        op: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("{op}: runtime state failure: {message}")]
    RuntimeState { op: &'static str, message: String },
    #[error("{op}: runtime state failure: {source}")]
    RuntimeStateSource {
        op: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("{op}: host access failed: {source}")]
    HostAccess {
        op: &'static str,
        #[source]
        source: crate::HostAccessError,
    },
    #[error("{op}: extension {family} failed: {source}")]
    Extension {
        op: &'static str,
        family: &'static str,
        kind: ErrorKind,
        #[source]
        source: BoxError,
    },
    #[error("missing runtime value for slot {slot}")]
    MissingValue { slot: usize },
    #[error("internal tensor error: {0}")]
    Internal(String),
}

impl Error {
    /// Construct an incompatible-shapes validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Error;
    ///
    /// let error = Error::shape_mismatch("add", [2, 3], [2, 4]);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn shape_mismatch(
        op: &'static str,
        lhs: impl Into<Vec<usize>>,
        rhs: impl Into<Vec<usize>>,
    ) -> Self {
        Self::validation(
            op,
            tenferro_tensor_core::ShapeMismatch::IncompatibleShapes {
                lhs: tenferro_tensor_core::ShapeVec::from_vec(lhs.into()),
                rhs: tenferro_tensor_core::ShapeVec::from_vec(rhs.into()),
            }
            .into(),
        )
    }

    /// Construct a rank-mismatch validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Error;
    ///
    /// let error = Error::rank_mismatch("transpose", 2, 3);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn rank_mismatch(op: &'static str, expected: usize, actual: usize) -> Self {
        Self::validation(op, ValidationError::RankMismatch { expected, actual })
    }

    /// Construct an axis-out-of-bounds validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Error;
    ///
    /// let error = Error::axis_out_of_bounds("sum", 2, 2);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn axis_out_of_bounds(op: &'static str, axis: usize, rank: usize) -> Self {
        Self::validation(op, ValidationError::AxisOutOfBounds { axis, rank })
    }

    /// Construct a duplicate-axis validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Error;
    ///
    /// let error = Error::duplicate_axis("transpose", 1, "permutation");
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn duplicate_axis(op: &'static str, axis: usize, role: &'static str) -> Self {
        Self::validation(op, ValidationError::DuplicateAxis { axis, role })
    }

    /// Construct a dtype-mismatch validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Error};
    ///
    /// let error = Error::dtype_mismatch("add", DType::F32, DType::F64);
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn dtype_mismatch(op: &'static str, expected: crate::DType, actual: crate::DType) -> Self {
        Self::validation(
            op,
            ValidationError::DTypeMismatch {
                expected: crate::core_dtype(expected),
                actual: crate::core_dtype(actual),
            },
        )
    }

    /// Wrap shared tensor validation with the operation that requested it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ValidationError};
    ///
    /// let error = Error::validation(
    ///     "transpose",
    ///     ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
    /// );
    /// assert!(matches!(error, Error::Validation { op: "transpose", .. }));
    /// ```
    pub fn validation(op: &'static str, source: ValidationError) -> Self {
        Self::Validation { op, source }
    }

    /// Construct a structured invalid-argument validation error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ErrorKind, ValidationKind};
    ///
    /// let error = Error::invalid_argument("slice", "step", "must be non-zero");
    /// assert_eq!(error.kind(), ErrorKind::Validation(ValidationKind::InvalidArgument));
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

    /// Construct an unsupported dtype conversion error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let error = tenferro_tensor::Error::unsupported_dtype_conversion(
    ///     "convert",
    ///     tenferro_tensor::DType::F64,
    ///     tenferro_tensor::DType::I32,
    ///     "lossy conversion is disabled",
    /// );
    /// assert!(matches!(
    ///     error,
    ///     tenferro_tensor::Error::UnsupportedDTypeConversion { .. }
    /// ));
    /// ```
    pub fn unsupported_dtype_conversion(
        op: &'static str,
        from: crate::DType,
        to: crate::DType,
        message: impl Into<String>,
    ) -> Self {
        Self::UnsupportedDTypeConversion {
            op,
            from,
            to,
            message: message.into(),
        }
    }

    /// Construct an operation-level unsupported-dtype error.
    ///
    /// This is for an operation that cannot run for the supplied dtype. It is
    /// deliberately distinct from [`Error::unsupported_dtype_conversion`],
    /// which is reserved for an actual from-dtype to to-dtype conversion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let error = tenferro_tensor::Error::unsupported_dtype(
    ///     "exp",
    ///     tenferro_tensor::DType::I64,
    ///     "integer exponentials are not implemented",
    /// );
    /// assert!(matches!(
    ///     error,
    ///     tenferro_tensor::Error::UnsupportedDType {
    ///         op: "exp",
    ///         dtype: tenferro_tensor::DType::I64,
    ///         ..
    ///     }
    /// ));
    /// ```
    pub fn unsupported_dtype(
        op: &'static str,
        dtype: crate::DType,
        message: impl Into<String>,
    ) -> Self {
        Self::UnsupportedDType {
            op,
            dtype,
            message: message.into(),
        }
    }

    /// Construct a structured unsupported-operation error.
    ///
    /// Use this for an operation or execution surface that is not implemented
    /// by the selected backend. Dtype conversion failures use
    /// [`Error::unsupported_dtype_conversion`] instead, and operation-specific
    /// typed reasons should use [`Error::extension`] with `ErrorKind::Unsupported`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let error = tenferro_tensor::Error::unsupported(
    ///     "full_piv_lu",
    ///     "backend has no implementation",
    /// );
    /// assert!(matches!(
    ///     error,
    ///     tenferro_tensor::Error::Unsupported { op: "full_piv_lu", .. }
    /// ));
    /// ```
    pub fn unsupported(op: &'static str, message: impl Into<String>) -> Self {
        Self::Unsupported {
            op,
            message: message.into(),
        }
    }

    /// Construct a text-only backend failure.
    ///
    /// Use [`Error::backend_source`] when a typed source is available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let error = tenferro_tensor::Error::backend_failure(
    ///     "matmul",
    ///     "backend rejected launch",
    /// );
    /// assert!(matches!(
    ///     error,
    ///     tenferro_tensor::Error::BackendFailure { op: "matmul", .. }
    /// ));
    /// ```
    pub fn backend_failure(op: &'static str, message: impl Into<String>) -> Self {
        Self::BackendFailure {
            op,
            message: message.into(),
        }
    }

    /// Construct a backend failure while preserving its typed source.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let error = tenferro_tensor::Error::backend_source(
    ///     "load",
    ///     std::io::Error::other("read failed"),
    /// );
    /// assert!(std::error::Error::source(&error).is_some());
    /// ```
    pub fn backend_source<E>(op: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::BackendSource {
            op,
            source: Box::new(source),
        }
    }

    /// Construct an I/O failure while preserving its typed source.
    ///
    /// I/O errors are intentionally separate from backend failures: callers
    /// can classify them as [`ErrorKind::Io`] without parsing a message.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ErrorKind};
    ///
    /// let error = Error::io_source("load", std::io::Error::other("read failed"));
    /// assert_eq!(error.kind(), ErrorKind::Io);
    /// assert!(std::error::Error::source(&error).is_some());
    /// ```
    pub fn io_source<E>(op: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::IoSource {
            op,
            source: Box::new(source),
        }
    }

    /// Construct a runtime-state failure when no typed source exists.
    ///
    /// Use this for missing, uninitialized, or invalid execution state. It is
    /// distinct from [`Error::backend_failure`], which is reserved for
    /// vendor/backend status text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ErrorKind};
    ///
    /// let error = Error::runtime_state("execute", "backend session is not initialized");
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// ```
    pub fn runtime_state(op: &'static str, message: impl Into<String>) -> Self {
        Self::RuntimeState {
            op,
            message: message.into(),
        }
    }

    /// Construct a runtime-state failure while preserving a typed source.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ErrorKind};
    ///
    /// let error = Error::runtime_state_source(
    ///     "execute",
    ///     std::io::Error::other("executor lock poisoned"),
    /// );
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// assert!(std::error::Error::source(&error).is_some());
    /// ```
    pub fn runtime_state_source<E>(op: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::RuntimeStateSource {
            op,
            source: Box::new(source),
        }
    }

    /// Construct an extension failure while preserving its typed source and
    /// coarse classification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::error::Error as _;
    /// use tenferro_tensor::{Error, ErrorKind};
    ///
    /// let error = Error::extension(
    ///     "einsum",
    ///     "einsum",
    ///     ErrorKind::Internal,
    ///     std::io::Error::other("planner failed"),
    /// );
    /// assert!(error.source().is_some());
    /// ```
    pub fn extension<E>(op: &'static str, family: &'static str, kind: ErrorKind, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Extension {
            op,
            family,
            kind,
            source: Box::new(source),
        }
    }

    /// Preserve a typed guarded-host-access failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, HostAccessError};
    ///
    /// let error = Error::host_access(
    ///     "map",
    ///     HostAccessError::Unsupported { backend: "opaque" },
    /// );
    /// assert!(matches!(error, Error::HostAccess { .. }));
    /// ```
    pub fn host_access(op: &'static str, source: crate::HostAccessError) -> Self {
        Self::HostAccess { op, source }
    }

    /// Return the stable coarse classification for this tensor failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Error, ErrorKind, ValidationError, ValidationKind};
    /// use tenferro_tensor::core::DType;
    ///
    /// let error = Error::validation(
    ///     "add",
    ///     ValidationError::DTypeMismatch {
    ///         expected: DType::F32,
    ///         actual: DType::F64,
    ///     },
    /// );
    /// assert_eq!(error.kind(), ErrorKind::Validation(ValidationKind::DTypeMismatch));
    /// ```
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Validation { source, .. } => ErrorKind::Validation(source.kind()),
            Self::UnsupportedDTypeConversion { .. }
            | Self::UnsupportedDType { .. }
            | Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::BackendFailure { .. } | Self::BackendSource { .. } => ErrorKind::BackendFailure,
            Self::IoSource { .. } => ErrorKind::Io,
            Self::RuntimeState { .. }
            | Self::RuntimeStateSource { .. }
            | Self::HostAccess { .. } => ErrorKind::RuntimeState,
            Self::Extension { kind, .. } => *kind,
            Self::MissingValue { .. } => ErrorKind::RuntimeState,
            Self::Internal(_) => ErrorKind::Internal,
        }
    }
}

/// Result type alias for runtime tensor operations.
pub type Result<T> = std::result::Result<T, Error>;
