use thiserror::Error;

/// Crate-wide error type for API skeleton operations.
///
/// # Examples
///
/// ```rust
/// use tenferro::{Error, Result};
///
/// fn maybe_fail(flag: bool) -> Result<()> {
///     if flag {
///         Ok(())
///     } else {
///         Err(Error::InvalidAdTensor {
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

    /// AD tensor operands are structurally invalid for the requested operation.
    #[error("invalid AD tensor operands: {message}")]
    InvalidAdTensor { message: String },

    /// Reverse-mode operands belong to different tapes.
    #[error("reverse-mode operands must share one tape: expected {expected}, found {found}")]
    MixedReverseTape { expected: u64, found: u64 },

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

/// Convenience result alias.
///
/// # Examples
///
/// ```rust
/// use tenferro::{Error, Result};
///
/// let ok: Result<i32> = Ok(1);
/// let err: Result<i32> = Err(Error::InvalidAdTensor {
///     message: "sample".into(),
/// });
///
/// assert_eq!(ok.unwrap(), 1);
/// assert!(err.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;
