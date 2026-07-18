//! Domain errors owned by the linear-algebra extension.
//!
//! Numerical failures and unsupported dtypes remain linalg-owned values until
//! they cross into the tensor/runtime boundary. That boundary wraps the value
//! as a typed extension source, so callers can classify the failure without
//! losing the operation-specific payload.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_linalg::Error;
//! use tenferro_tensor::{ErrorKind, DType};
//!
//! let error = Error::UnsupportedDType {
//!     op: "svd",
//!     dtype: DType::I32,
//! };
//! assert_eq!(error.kind(), ErrorKind::Unsupported);
//! ```

use tenferro_tensor::{DType, ErrorKind};

/// Typed diagnostics for provider status and workspace contracts.
#[cfg(any(feature = "cpu-blas", feature = "cuda"))]
#[derive(Debug, thiserror::Error)]
pub(crate) enum BackendError {
    #[cfg(feature = "cuda")]
    #[error("{library} call {call} returned status {status}")]
    ProviderStatus {
        library: &'static str,
        call: &'static str,
        status: i32,
    },
    #[cfg(feature = "cpu-blas")]
    #[error("{library} routine {routine} returned an invalid workspace: {detail}")]
    InvalidWorkspace {
        library: &'static str,
        routine: &'static str,
        detail: String,
    },
}

/// Failures owned by a linalg algorithm or provider boundary.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::Error;
/// use tenferro_tensor::ErrorKind;
///
/// let error = Error::Singular { op: "solve" };
/// assert_eq!(error.kind(), ErrorKind::NumericalFailure);
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The algorithm could not converge for the supplied numeric input.
    #[error("{op} did not converge")]
    NonConvergence {
        /// Linalg operation that failed to converge.
        op: &'static str,
    },
    /// The input matrix or factorization became singular.
    #[error("{op} is singular")]
    Singular {
        /// Linalg operation that encountered a singular matrix.
        op: &'static str,
    },
    /// The selected linalg operation has no implementation for this dtype.
    #[error("{op} does not support dtype {dtype:?}")]
    UnsupportedDType {
        /// Linalg operation that rejected the dtype.
        op: &'static str,
        /// Rejected input dtype.
        dtype: DType,
    },
}

impl Error {
    /// Return the stable coarse classification for this linalg failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::Error;
    /// use tenferro_tensor::ErrorKind;
    ///
    /// assert_eq!(
    ///     Error::NonConvergence { op: "svd" }.kind(),
    ///     ErrorKind::NumericalFailure
    /// );
    /// ```
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::NonConvergence { .. } | Self::Singular { .. } => ErrorKind::NumericalFailure,
            Self::UnsupportedDType { .. } => ErrorKind::Unsupported,
        }
    }
}

/// Result type for linalg-owned domain operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{Error, Result};
///
/// let result: Result<()> = Err(Error::Singular { op: "solve" });
/// assert!(result.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;

/// Wrap a linalg-owned source at the tensor extension boundary.
pub(crate) fn into_tensor_error(op: &'static str, source: Error) -> tenferro_tensor::Error {
    tenferro_tensor::Error::extension(
        op,
        crate::extension::LINALG_EXTENSION_FAMILY_ID,
        source.kind(),
        source,
    )
}

/// Construct a typed unsupported-dtype tensor error for linalg backends.
pub(crate) fn unsupported_dtype(op: &'static str, dtype: DType) -> tenferro_tensor::Error {
    into_tensor_error(op, Error::UnsupportedDType { op, dtype })
}

/// Preserve a provider status as a typed backend source.
#[cfg(feature = "cuda")]
pub(crate) fn backend_status(
    op: &'static str,
    library: &'static str,
    call: &'static str,
    status: i32,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(
        op,
        BackendError::ProviderStatus {
            library,
            call,
            status,
        },
    )
}

/// Preserve an invalid provider workspace response as a typed backend source.
#[cfg(feature = "cpu-blas")]
pub(crate) fn invalid_workspace(
    op: &'static str,
    library: &'static str,
    routine: &'static str,
    detail: impl Into<String>,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(
        op,
        BackendError::InvalidWorkspace {
            library,
            routine,
            detail: detail.into(),
        },
    )
}

#[cfg(all(test, any(feature = "cpu-blas", feature = "cuda")))]
mod tests {
    use std::error::Error as _;

    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn provider_status_keeps_typed_backend_source() {
        let error = backend_status("svd", "cuSOLVER", "cusolverDnSgesvd", 7);

        assert_eq!(error.kind(), ErrorKind::BackendFailure);
        assert!(matches!(
            error.source().and_then(|source| source.downcast_ref()),
            Some(BackendError::ProviderStatus {
                library: "cuSOLVER",
                call: "cusolverDnSgesvd",
                status: 7,
            })
        ));
    }

    #[cfg(feature = "cpu-blas")]
    #[test]
    fn invalid_workspace_keeps_typed_backend_source() {
        let error = invalid_workspace("eigh", "LAPACK", "dsyevd", "query was zero");

        assert_eq!(error.kind(), ErrorKind::BackendFailure);
        assert!(matches!(
            error.source().and_then(|source| source.downcast_ref()),
            Some(BackendError::InvalidWorkspace {
                library: "LAPACK",
                routine: "dsyevd",
                detail,
            }) if detail == "query was zero"
        ));
    }
}
