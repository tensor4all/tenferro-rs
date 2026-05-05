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

use tenferro_tensor::DType;

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

    /// A `TracedTensor` passed to `eval_with_inputs` bindings is not a
    /// placeholder (has attached data).
    #[error(
        "binding #{binding_index} is not a placeholder; \
         only tensors built via input_concrete_shape / input_symbolic_shape \
         can be bound"
    )]
    UnexpectedBinding { binding_index: usize },

    /// A placeholder appearing in the graph has no binding supplied.
    #[error("placeholder {input_key} has no binding in eval_with_inputs")]
    UnboundPlaceholder { input_key: String },

    /// The same placeholder was bound more than once in the `bindings` slice.
    #[error("placeholder {input_key} was bound more than once")]
    DuplicateBinding { input_key: String },

    /// A binding tensor's dtype does not match the placeholder's dtype.
    #[error("binding dtype mismatch for placeholder: expected {expected:?}, got {actual:?}")]
    PlaceholderDtypeMismatch { expected: DType, actual: DType },

    /// A binding tensor's shape does not match an `input_concrete_shape`
    /// placeholder's fixed shape.
    #[error(
        "binding shape mismatch for concrete-shape placeholder: \
         expected {expected:?}, got {actual:?}"
    )]
    PlaceholderShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// A binding tensor's rank does not match an `input_symbolic_shape`
    /// placeholder's declared rank.
    #[error(
        "binding rank mismatch for symbolic-shape placeholder: \
         expected rank {expected}, got rank {actual}"
    )]
    PlaceholderRankMismatch { expected: usize, actual: usize },

    /// Operation attempted to mix tensors from different eager contexts.
    #[error(
        "tensors belong to different eager contexts ({lhs} vs {rhs}); \
         use EagerTensor::detach_into or EagerContext::constant_from"
    )]
    ContextMismatch { lhs: ContextId, rhs: ContextId },

    /// An unexpected internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Opaque identifier for an [`EagerContext`], used in [`Error::ContextMismatch`].
///
/// [`EagerContext`]: crate::EagerContext
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(usize);

impl ContextId {
    pub(crate) fn from_ptr<T>(ptr: *const T) -> Self {
        Self(ptr as usize)
    }
}

impl std::fmt::Display for ContextId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ctx@{:x}", self.0)
    }
}

/// Result type alias for tenferro operations.
pub type Result<T> = std::result::Result<T, Error>;
