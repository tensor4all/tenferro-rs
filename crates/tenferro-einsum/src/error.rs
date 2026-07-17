//! Error types owned by the einsum crate.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_einsum::Error;
//!
//! let err = Error::InvalidArgument("bad subscripts".into());
//! assert_eq!(err.to_string(), "invalid argument: bad subscripts");
//! ```

/// Errors produced while parsing, planning, or lowering einsum expressions.
///
/// # Examples
///
/// ```rust
/// use tenferro_einsum::Error;
///
/// let err = Error::ShapeMismatch {
///     expected: vec![2, 3],
///     got: vec![2, 4],
/// };
/// assert!(err.to_string().contains("shape mismatch"));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    /// Tensor shapes are incompatible for an einsum expression.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape or dimension sizes.
        expected: Vec<usize>,
        /// Actual shape or dimension sizes.
        got: Vec<usize>,
    },

    /// An invalid einsum argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl Error {
    /// Convert this einsum error into a tensor backend failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_einsum::Error;
    /// use tenferro_tensor::Error as TensorError;
    ///
    /// let err = Error::InvalidArgument("bad subscripts".into())
    ///     .to_tensor_error("einsum_extension");
    ///
    /// assert!(matches!(
    ///     err,
    ///     TensorError::BackendSource {
    ///         op: "einsum_extension",
    ///         ..
    ///     }
    /// ));
    /// ```
    #[must_use]
    pub fn to_tensor_error(&self, op: &'static str) -> tenferro_tensor::Error {
        tenferro_tensor::Error::backend_source(op, self.clone())
    }
}

/// Result type alias for einsum parsing, planning, and lowering.
///
/// # Examples
///
/// ```rust
/// use tenferro_einsum::{Error, Result};
///
/// let result: Result<()> = Err(Error::InvalidArgument("bad input".into()));
/// assert!(result.is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;
