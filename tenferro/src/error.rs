//! Error types for the tenferro crate.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro::error::Error;
//!
//! let err = Error::InvalidSubscripts("bad label".into());
//! assert!(err.to_string().contains("bad label"));
//! ```

/// Errors produced by einsum, eval, and other tenferro operations.
///
/// # Examples
///
/// ```ignore
/// use tenferro::error::Error;
///
/// let err = Error::InvalidSubscripts("rank mismatch".into());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Einsum subscript string is invalid or cannot be parsed.
    #[error("invalid subscripts: {0}")]
    InvalidSubscripts(String),

    /// Contraction optimization failed (shape mismatch, bad path, etc.).
    #[error("contraction error: {0}")]
    ContractionError(String),

    /// A required input tensor is missing from the inputs map.
    #[error("missing input: {0}")]
    MissingInput(String),

    /// Reverse-mode gradient requires a scalar output.
    #[error("grad requires a scalar output, got shape {shape:?}")]
    NonScalarGrad { shape: Vec<usize> },

    /// Runtime tensor execution failed in the backend layer.
    #[error(transparent)]
    TensorRuntime(#[from] tenferro_tensor::Error),

    /// An unexpected internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type alias for tenferro operations.
pub type Result<T> = std::result::Result<T, Error>;
