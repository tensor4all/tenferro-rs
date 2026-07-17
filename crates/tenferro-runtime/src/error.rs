//! Error types for the tenferro runtime crate.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::error::Error;
//!
//! let err = Error::InvalidSubscripts("bad label".into());
//! assert!(err.to_string().contains("bad label"));
//! ```

use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_ops::ShapeRelation;
use tenferro_tensor::{DType, ErrorKind, ValidationError, ValidationKind};

static NEXT_CONTEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// Boxed source used when a runtime registry or compiler subsystem crosses
/// the runtime error boundary with a concrete error owned by another crate.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// Phase at which a runtime failure was discovered.
///
/// The phase is independent from [`ErrorKind`]: the same validation fact can
/// be discovered while building a graph, compiling it for concrete inputs,
/// or executing a compiled program.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ErrorPhase;
///
/// assert_ne!(ErrorPhase::GraphBuild, ErrorPhase::Execution);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorPhase {
    /// A caller-controlled graph construction check failed.
    GraphBuild,
    /// Shape inference or lowering discovered the failure.
    Compile,
    /// Input binding or backend execution discovered the failure.
    Execution,
}

/// Typed reason that a symbolic shape constraint could not be evaluated.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ShapeConstraintEvalError;
///
/// let cause = ShapeConstraintEvalError::MissingInput {
///     input_idx: 2,
///     input_count: 1,
/// };
/// assert_eq!(
///     cause.to_string(),
///     "shape expression references input 2, but only 1 inputs were provided"
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ShapeConstraintEvalError {
    /// An expression referenced an input shape that was not supplied.
    #[error(
        "shape expression references input {input_idx}, but only {input_count} inputs were provided"
    )]
    MissingInput {
        /// Referenced input index.
        input_idx: usize,
        /// Number of supplied input shapes.
        input_count: usize,
    },
    /// An expression referenced an axis outside the selected input's rank.
    #[error("shape expression references input {input_idx} axis {axis}, but its rank is {rank}")]
    AxisOutOfBounds {
        /// Referenced input index.
        input_idx: usize,
        /// Referenced axis.
        axis: usize,
        /// Rank of the selected input.
        rank: usize,
    },
    /// Checked dimension arithmetic overflowed `usize`.
    #[error("shape expression arithmetic overflowed")]
    Overflow,
    /// Checked dimension subtraction underflowed `usize`.
    #[error("shape expression subtraction underflowed")]
    Underflow,
    /// A floor-division divisor evaluated to zero.
    #[error("shape expression divided by zero")]
    DivisionByZero,
}

/// Errors produced by einsum, eval, and other tenferro operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::error::Error;
///
/// let err = Error::InvalidSubscripts("rank mismatch".into());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A shared tensor validation fact, annotated with the runtime phase.
    #[error("{op} ({phase:?}): {source}")]
    Validation {
        /// Public operation name.
        op: &'static str,
        /// Phase that discovered the validation fact.
        phase: ErrorPhase,
        /// Machine-readable validation payload.
        #[source]
        source: ValidationError,
    },

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

    /// A runtime metadata or registry subsystem failed with a typed source.
    #[error("runtime metadata failure: {source}")]
    Metadata {
        /// Typed metadata subsystem source.
        #[source]
        source: BoxError,
    },

    /// A `TracedTensor` passed to graph-executor input bindings is not a
    /// placeholder (has attached data).
    #[error(
        "binding #{binding_index} is not a placeholder; \
         only tensors built via input_concrete_shape / input_symbolic_shape \
         can be bound"
    )]
    UnexpectedBinding { binding_index: usize },

    /// A placeholder appearing in the graph has no binding supplied.
    #[error("placeholder {input_key} has no runtime input binding")]
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
        "tensors belong to different eager AD contexts ({lhs} vs {rhs}); \
         detach into the target context before combining them"
    )]
    ContextMismatch { lhs: ContextId, rhs: ContextId },

    /// An AD transform requires a primitive or extension rule that is not
    /// registered for the requested operation.
    #[error("unsupported {transform} AD rule for {op}")]
    UnsupportedAdRule {
        /// AD transform that requested the rule, such as `grad` or `backward`.
        transform: &'static str,
        /// Operation or extension family identifier that has no applicable rule.
        op: String,
    },

    /// A symbolic extension shape equality evaluated to unequal dimensions.
    #[error(
        "extension family {family:?} shape constraint at instruction {instruction_index:?} failed: {lhs_expr} ({lhs_value}) {relation:?} {rhs_expr} ({rhs_value})"
    )]
    ShapeConstraintViolation {
        /// Stable extension family identifier.
        family: &'static str,
        /// Stable compiled instruction provenance, when assigned.
        instruction_index: Option<usize>,
        /// Shape relation that failed.
        relation: ShapeRelation,
        /// Normalized left-hand expression.
        lhs_expr: String,
        /// Normalized right-hand expression.
        rhs_expr: String,
        /// Concrete left-hand value.
        lhs_value: usize,
        /// Concrete right-hand value.
        rhs_value: usize,
    },

    /// A symbolic extension shape expression could not be evaluated safely.
    #[error(
        "extension family {family:?} shape constraint at instruction {instruction_index:?} could not evaluate {expression} for {relation:?}: {cause}"
    )]
    ShapeConstraintEvaluation {
        /// Stable extension family identifier.
        family: &'static str,
        /// Stable compiled instruction provenance, when assigned.
        instruction_index: Option<usize>,
        /// Shape relation whose expression failed.
        relation: ShapeRelation,
        /// Normalized expression that failed.
        expression: String,
        /// Typed evaluation failure.
        #[source]
        cause: ShapeConstraintEvalError,
    },

    /// An unexpected internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Wrap a shared validation payload with its operation and discovery
    /// phase.
    ///
    /// # Errors
    ///
    /// This constructor does not fail; callers receive the returned
    /// [`Error`] value and can inspect its [`Error::kind`] and
    /// [`Error::phase`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::{ErrorKind, ShapeMismatch, ValidationKind};
    ///
    /// let error = Error::validation(
    ///     "reshape",
    ///     ErrorPhase::GraphBuild,
    ///     ShapeMismatch::ReshapeElementCount { from: 2, to: 3 }.into(),
    /// );
    /// assert_eq!(
    ///     error.kind(),
    ///     ErrorKind::Validation(ValidationKind::ShapeMismatch)
    /// );
    /// assert_eq!(error.phase(), Some(ErrorPhase::GraphBuild));
    /// ```
    pub fn validation(op: &'static str, phase: ErrorPhase, source: ValidationError) -> Self {
        Self::Validation { op, phase, source }
    }

    /// Construct a validation error for a caller-controlled argument whose
    /// failure does not have a more specific shared payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    ///
    /// let error = Error::invalid_argument(
    ///     "broadcast_in_dim",
    ///     ErrorPhase::GraphBuild,
    ///     "dims",
    ///     "dimension mapping has the wrong length",
    /// );
    /// assert!(matches!(error, Error::Validation { .. }));
    /// ```
    pub fn invalid_argument(
        op: &'static str,
        phase: ErrorPhase,
        argument: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::validation(
            op,
            phase,
            ValidationError::InvalidArgument {
                argument,
                message: message.into(),
            },
        )
    }

    /// Return the stable coarse classification of this runtime failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::{ErrorKind, ValidationError, ValidationKind};
    ///
    /// let error = Error::validation(
    ///     "transpose",
    ///     ErrorPhase::GraphBuild,
    ///     ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
    /// );
    /// assert_eq!(error.kind(), ErrorKind::Validation(ValidationKind::AxisOutOfBounds));
    /// ```
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Validation { source, .. } => ErrorKind::Validation(source.kind()),
            Self::InvalidSubscripts(_) => ErrorKind::Validation(ValidationKind::InvalidArgument),
            Self::ContractionError(_) => ErrorKind::RuntimeState,
            Self::MissingInput(_)
            | Self::UnexpectedBinding { .. }
            | Self::UnboundPlaceholder { .. }
            | Self::DuplicateBinding { .. }
            | Self::ContextMismatch { .. } => ErrorKind::RuntimeState,
            Self::NonScalarGrad { .. } => ErrorKind::Validation(ValidationKind::InvalidArgument),
            Self::TensorRuntime(error) => error.kind(),
            Self::Metadata { .. } => ErrorKind::Internal,
            Self::PlaceholderDtypeMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::DTypeMismatch)
            }
            Self::PlaceholderShapeMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::ShapeMismatch)
            }
            Self::PlaceholderRankMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::RankMismatch)
            }
            Self::UnsupportedAdRule { .. } => ErrorKind::Unsupported,
            Self::ShapeConstraintViolation { .. } => {
                ErrorKind::Validation(ValidationKind::ShapeMismatch)
            }
            Self::ShapeConstraintEvaluation { .. } => {
                ErrorKind::Validation(ValidationKind::InvalidArgument)
            }
            Self::Internal(_) => ErrorKind::Internal,
        }
    }

    /// Return the discovery phase when this error has one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::ValidationError;
    ///
    /// let error = Error::validation(
    ///     "reshape",
    ///     ErrorPhase::Compile,
    ///     ValidationError::RankMismatch { expected: 2, actual: 1 },
    /// );
    /// assert_eq!(error.phase(), Some(ErrorPhase::Compile));
    /// ```
    pub fn phase(&self) -> Option<ErrorPhase> {
        match self {
            Self::Validation { phase, .. } => Some(*phase),
            Self::TensorRuntime(_) => Some(ErrorPhase::Execution),
            Self::Metadata { .. } => Some(ErrorPhase::Compile),
            Self::PlaceholderDtypeMismatch { .. }
            | Self::PlaceholderShapeMismatch { .. }
            | Self::PlaceholderRankMismatch { .. }
            | Self::UnexpectedBinding { .. }
            | Self::UnboundPlaceholder { .. }
            | Self::DuplicateBinding { .. } => Some(ErrorPhase::Execution),
            _ => None,
        }
    }
}

/// Opaque identifier for an eager AD runtime, used in [`Error::ContextMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContextId(usize);

impl ContextId {
    /// Generate a fresh opaque runtime context identifier.
    ///
    /// Runtime implementations use this when constructing a new execution
    /// context. The value is intentionally opaque and is only useful in error
    /// reporting and equality checks.
    pub fn fresh() -> Self {
        let id = NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }
}

impl std::fmt::Display for ContextId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ctx@{:x}", self.0)
    }
}

/// Result type alias for tenferro operations.
pub type Result<T> = std::result::Result<T, Error>;
