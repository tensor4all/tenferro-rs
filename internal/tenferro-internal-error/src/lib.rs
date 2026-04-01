//! Internal shared error definitions for `tenferro` surface crates.
//!
//! # Examples
//!
//! ```
//! use tenferro_internal_error::{Error, Result};
//!
//! fn require_runtime(ready: bool) -> Result<()> {
//!     if ready {
//!         Ok(())
//!     } else {
//!         Err(Error::RuntimeNotConfigured)
//!     }
//! }
//!
//! assert!(require_runtime(true).is_ok());
//! assert!(require_runtime(false).is_err());
//! ```

use thiserror::Error;

/// Shared error type for dynamic `tenferro` surface crates.
///
/// # Examples
///
/// ```
/// use tenferro_internal_error::{Error, Result};
///
/// fn maybe_fail(flag: bool) -> Result<()> {
///     if flag {
///         Ok(())
///     } else {
///         Err(Error::InvalidTensorOperands {
///             message: "demo".into(),
///         })
///     }
/// }
///
/// assert!(maybe_fail(true).is_ok());
/// assert!(maybe_fail(false).is_err());
/// ```
#[derive(Debug, Error)]
pub enum Error {
    /// The runtime used by builder `.run()` was not configured.
    #[error("default runtime is not configured; call `set_default_runtime(...)` first")]
    RuntimeNotConfigured,

    /// Wrapper for backend/linalg/einsum errors from tenferro crates.
    #[error(transparent)]
    Backend(#[from] tenferro_device::Error),

    /// Wrapper for AD-rule level errors from `chainrules-core`.
    #[error(transparent)]
    Autodiff(#[from] chainrules_core::AutodiffError),

    /// Tensor operands are structurally invalid for the requested operation.
    #[error("invalid tensor operands: {message}")]
    InvalidTensorOperands { message: String },

    /// Reverse-mode operands belong to different value graphs.
    #[error(
        "reverse-mode operands must share one value graph: expected {expected}, found {found}"
    )]
    MixedReverseGraph { expected: u64, found: u64 },

    /// Operation is not available for the currently selected runtime.
    #[error("operation `{op}` is not supported on runtime `{runtime}`")]
    UnsupportedRuntimeOp {
        op: &'static str,
        runtime: &'static str,
    },

    /// AD operation is not available for the requested mode.
    #[error("AD operation `{op}` is not supported for the provided inputs")]
    UnsupportedAdOp { op: &'static str },

    /// Linalg operation is available only for dense tensor inputs.
    #[error("linalg operation `{op}` requires dense tensor inputs")]
    UnsupportedStructuredLinalg { op: &'static str },
}

/// Convenience result alias for `tenferro` surface errors.
///
/// # Examples
///
/// ```
/// use tenferro_internal_error::{Error, Result};
///
/// let ok: Result<i32> = Ok(1);
/// let err: Result<i32> = Err(Error::InvalidTensorOperands {
///     message: "sample".into(),
/// });
///
/// assert_eq!(ok.unwrap(), 1);
/// assert!(err.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests;
