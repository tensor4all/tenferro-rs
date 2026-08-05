//! Error types for the tenferro runtime crate.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::error::{Error, ErrorPhase};
//!
//! let err = Error::invalid_argument(
//!     "einsum",
//!     ErrorPhase::GraphBuild,
//!     "subscripts",
//!     "bad label",
//! );
//! assert!(err.to_string().contains("bad label"));
//! ```

use std::error::Error as StdError;
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_ops::{dim_expr::DimExprEvalError, ShapeRelation, SymDimConversionError};
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

impl From<DimExprEvalError> for ShapeConstraintEvalError {
    fn from(error: DimExprEvalError) -> Self {
        match error {
            DimExprEvalError::InputOutOfBounds {
                input_idx,
                input_count,
            } => Self::MissingInput {
                input_idx,
                input_count,
            },
            DimExprEvalError::AxisOutOfBounds {
                input_idx,
                axis,
                rank,
            } => Self::AxisOutOfBounds {
                input_idx,
                axis,
                rank,
            },
            DimExprEvalError::AddOverflow { .. } | DimExprEvalError::MulOverflow { .. } => {
                Self::Overflow
            }
            DimExprEvalError::SubUnderflow { .. } => Self::Underflow,
            DimExprEvalError::FloorDivByZero { .. } => Self::DivisionByZero,
        }
    }
}

/// Errors produced by einsum, eval, and other tenferro operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::error::{Error, ErrorPhase};
///
/// let err = Error::invalid_argument(
///     "einsum",
///     ErrorPhase::GraphBuild,
///     "subscripts",
///     "rank mismatch",
/// );
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

    /// A required input tensor is missing from the inputs map.
    #[error("missing input: {0}")]
    MissingInput(String),

    /// Reverse-mode gradient requires a scalar output.
    #[error("grad requires a scalar output, got shape {shape:?}")]
    NonScalarGrad { shape: Vec<usize> },

    /// The operation is known not to support the requested input or
    /// configuration at the phase where it was requested.
    #[error("{op} ({phase:?}) is unsupported: {message}")]
    Unsupported {
        /// Operation that does not provide the requested behavior.
        op: &'static str,
        /// Phase that established the unsupported combination.
        phase: ErrorPhase,
        /// Human-readable unsupported-operation detail.
        message: String,
    },

    /// Runtime tensor execution failed in the backend layer.
    #[error(transparent)]
    TensorRuntime(#[from] tenferro_tensor::Error),

    /// A typed extension-domain error crossed a runtime registry boundary.
    #[error("extension {family} ({phase:?}) failed for {op}: {source}")]
    Extension {
        /// Operation that discovered the extension failure.
        op: &'static str,
        /// Phase that discovered the extension failure.
        phase: ErrorPhase,
        /// Stable extension family identifier.
        family: &'static str,
        /// Coarse classification supplied by the extension owner.
        kind: ErrorKind,
        /// Original extension-domain source.
        #[source]
        source: BoxError,
    },

    /// Executor, cache, registry, or device state is unavailable or invalid.
    #[error("{op} ({phase:?}): runtime state failure: {message}")]
    RuntimeState {
        /// Operation whose state was unavailable.
        op: &'static str,
        /// Phase that discovered the invalid state.
        phase: ErrorPhase,
        /// Human-readable state detail.
        message: String,
    },

    /// A runtime-state failure retaining a typed source.
    #[error("{op} ({phase:?}): runtime state failure: {source}")]
    RuntimeStateSource {
        /// Operation whose state was unavailable.
        op: &'static str,
        /// Phase that discovered the invalid state.
        phase: ErrorPhase,
        /// Typed state source.
        #[source]
        source: BoxError,
    },

    /// A primary error with a second typed error retained as suppressed
    /// metadata.
    ///
    /// The standard error source chain follows `primary`. The suppressed
    /// error is intentionally exposed through [`Error::suppressed`] because
    /// [`StdError::source`](std::error::Error::source) can represent only one
    /// source without losing the primary error's semantics.
    #[error("primary error: {primary}; suppressed error: {suppressed}")]
    WithSuppressed {
        /// The operation's primary failure and the standard error-chain source.
        #[source]
        primary: Box<Error>,
        /// A typed secondary failure retained for diagnostics and recovery.
        suppressed: Box<Error>,
    },

    /// A runtime event-domain provenance or admission contract failed.
    #[error("event-domain operation failed: {source}")]
    EventDomain {
        /// Structured event-domain failure with expected/actual provenance.
        #[from]
        #[source]
        source: crate::runtime::EventDomainError,
    },

    /// A `TracedTensor` supplied as a compiled-graph input binding is not a
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

    /// The number of ordered tensors supplied to a compiled graph is invalid.
    #[error("compiled graph expects {expected} ordered inputs, got {actual}")]
    GraphInputCountMismatch { expected: usize, actual: usize },

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

    /// A binding tensor dimension exceeds a semantic input's declared bound.
    #[error("binding dimension {axis} exceeds semantic input upper bound {bound}: got {actual}")]
    PlaceholderShapeBoundExceeded {
        /// Axis whose runtime extent exceeded the bound.
        axis: usize,
        /// Evaluated upper bound.
        bound: usize,
        /// Runtime extent.
        actual: usize,
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

    /// A typed AD rule source that crossed an external message-only callback.
    #[error("{transform} AD rule failed: {source}")]
    AdRuleSource {
        /// AD transform that requested the rule.
        transform: &'static str,
        /// Original typed source from the AD rule context.
        #[source]
        source: BoxError,
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

    /// A symbolic dimension could not be converted into the graph's local
    /// dimension-expression vocabulary.
    #[error("{op} ({phase:?}): symbolic shape conversion failed: {source}")]
    SymbolicShapeConversion {
        /// Operation that requested the symbolic shape conversion.
        op: &'static str,
        /// Phase that discovered the invalid symbolic reference.
        phase: ErrorPhase,
        /// Typed symbolic-dimension conversion failure.
        #[source]
        source: SymDimConversionError,
    },

    /// A runtime dimension expression could not be evaluated for concrete
    /// input shapes.
    #[error("runtime shape expression {expression} could not evaluate: {cause}")]
    ShapeExpressionEvaluation {
        /// Expression that failed during execution.
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

    /// Construct a dtype-mismatch validation error using the runtime dtype
    /// vocabulary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{DType, Error, ErrorPhase};
    /// use tenferro_tensor::{ErrorKind, ValidationKind};
    ///
    /// let error = Error::dtype_mismatch(
    ///     "add",
    ///     ErrorPhase::GraphBuild,
    ///     DType::F32,
    ///     DType::F64,
    /// );
    /// assert_eq!(error.kind(), ErrorKind::Validation(ValidationKind::DTypeMismatch));
    /// ```
    pub fn dtype_mismatch(
        op: &'static str,
        phase: ErrorPhase,
        expected: DType,
        actual: DType,
    ) -> Self {
        Self::validation(
            op,
            phase,
            ValidationError::DTypeMismatch {
                expected: core_dtype(expected),
                actual: core_dtype(actual),
            },
        )
    }

    /// Preserve a typed extension-domain source at the runtime boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::error::Error as _;
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let source = std::io::Error::new(std::io::ErrorKind::Other, "extension failed");
    /// let error = Error::extension(
    ///     "einsum",
    ///     ErrorPhase::GraphBuild,
    ///     "example.extension.v1",
    ///     ErrorKind::RuntimeState,
    ///     source,
    /// );
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// assert!(error.source().is_some());
    /// ```
    pub fn extension<E>(
        op: &'static str,
        phase: ErrorPhase,
        family: &'static str,
        kind: ErrorKind,
        source: E,
    ) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Extension {
            op,
            phase,
            family,
            kind,
            source: Box::new(source),
        }
    }

    /// Construct a runtime-state failure for an unavailable or invalid
    /// executor, cache, registry, or device state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let error = Error::runtime_state(
    ///     "executor",
    ///     ErrorPhase::Execution,
    ///     "the executor is not initialized",
    /// );
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// ```
    pub fn runtime_state(op: &'static str, phase: ErrorPhase, message: impl Into<String>) -> Self {
        Self::RuntimeState {
            op,
            phase,
            message: message.into(),
        }
    }

    /// Preserve a typed source for an unavailable or invalid runtime state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::error::Error as _;
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let error = Error::runtime_state_source(
    ///     "metadata",
    ///     ErrorPhase::Compile,
    ///     std::io::Error::other("registry lock poisoned"),
    /// );
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// assert!(error.source().is_some());
    /// ```
    pub fn runtime_state_source<E>(op: &'static str, phase: ErrorPhase, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::RuntimeStateSource {
            op,
            phase,
            source: Box::new(source),
        }
    }

    /// Retain a typed secondary error while preserving the primary error's
    /// classification and standard source chain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    ///
    /// let error = Error::with_suppressed(
    ///     Error::unsupported("backend", ErrorPhase::Execution, "primary"),
    ///     Error::runtime_state("runtime", ErrorPhase::Execution, "suppressed"),
    /// );
    /// assert!(error.primary().is_some());
    /// assert!(error.suppressed().is_some());
    /// ```
    pub fn with_suppressed(primary: Self, suppressed: Self) -> Self {
        Self::WithSuppressed {
            primary: Box::new(primary),
            suppressed: Box::new(suppressed),
        }
    }

    /// Return the primary error when this value is a suppressed-error
    /// aggregate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    ///
    /// let error = Error::with_suppressed(
    ///     Error::unsupported("backend", ErrorPhase::Execution, "primary"),
    ///     Error::runtime_state("runtime", ErrorPhase::Execution, "suppressed"),
    /// );
    /// assert_eq!(error.primary().unwrap().phase(), Some(ErrorPhase::Execution));
    /// ```
    pub fn primary(&self) -> Option<&Self> {
        match self {
            Self::WithSuppressed { primary, .. } => Some(primary),
            _ => None,
        }
    }

    /// Return the typed suppressed error when this value is an aggregate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    ///
    /// let error = Error::with_suppressed(
    ///     Error::unsupported("backend", ErrorPhase::Execution, "primary"),
    ///     Error::runtime_state("runtime", ErrorPhase::Execution, "suppressed"),
    /// );
    /// assert_eq!(error.suppressed().unwrap().phase(), Some(ErrorPhase::Execution));
    /// ```
    pub fn suppressed(&self) -> Option<&Self> {
        match self {
            Self::WithSuppressed { suppressed, .. } => Some(suppressed),
            _ => None,
        }
    }

    /// Preserve a typed source returned by an AD rule through a callback
    /// protocol that can carry only a rendered message.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::error::Error as _;
    /// use tenferro_runtime::Error;
    ///
    /// let error = Error::ad_rule_source(
    ///     "jvp",
    ///     std::io::Error::other("shape metadata missing"),
    /// );
    /// assert!(error.source().is_some());
    /// ```
    pub fn ad_rule_source<E>(transform: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::AdRuleSource {
            transform,
            source: Box::new(source),
        }
    }

    /// Construct an operation-level unsupported error with an explicit
    /// discovery phase.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::{Error, ErrorPhase};
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let error = Error::unsupported(
    ///     "compare",
    ///     ErrorPhase::Compile,
    ///     "complex values have no total order",
    /// );
    /// assert_eq!(error.kind(), ErrorKind::Unsupported);
    /// assert_eq!(error.phase(), Some(ErrorPhase::Compile));
    /// ```
    pub fn unsupported(op: &'static str, phase: ErrorPhase, message: impl Into<String>) -> Self {
        Self::Unsupported {
            op,
            phase,
            message: message.into(),
        }
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
            Self::MissingInput(_)
            | Self::UnexpectedBinding { .. }
            | Self::UnboundPlaceholder { .. }
            | Self::DuplicateBinding { .. }
            | Self::ContextMismatch { .. } => ErrorKind::RuntimeState,
            Self::NonScalarGrad { .. } => ErrorKind::Validation(ValidationKind::InvalidArgument),
            Self::GraphInputCountMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::InvalidArgument)
            }
            Self::Unsupported { .. } | Self::UnsupportedAdRule { .. } => ErrorKind::Unsupported,
            Self::AdRuleSource { .. } => ErrorKind::Validation(ValidationKind::InvalidArgument),
            Self::TensorRuntime(error) => error.kind(),
            Self::Extension { kind, .. } => *kind,
            Self::RuntimeState { .. }
            | Self::RuntimeStateSource { .. }
            | Self::EventDomain { .. } => ErrorKind::RuntimeState,
            Self::WithSuppressed { primary, .. } => primary.kind(),
            Self::PlaceholderDtypeMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::DTypeMismatch)
            }
            Self::PlaceholderShapeMismatch { .. } | Self::PlaceholderShapeBoundExceeded { .. } => {
                ErrorKind::Validation(ValidationKind::ShapeMismatch)
            }
            Self::PlaceholderRankMismatch { .. } => {
                ErrorKind::Validation(ValidationKind::RankMismatch)
            }
            Self::ShapeConstraintViolation { .. } => {
                ErrorKind::Validation(ValidationKind::ShapeMismatch)
            }
            Self::ShapeConstraintEvaluation { .. } => {
                ErrorKind::Validation(ValidationKind::InvalidArgument)
            }
            Self::SymbolicShapeConversion { .. } => {
                ErrorKind::Validation(ValidationKind::InvalidArgument)
            }
            Self::ShapeExpressionEvaluation { .. } => {
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
            Self::Unsupported { phase, .. } => Some(*phase),
            Self::Extension { phase, .. } => Some(*phase),
            Self::RuntimeState { phase, .. } | Self::RuntimeStateSource { phase, .. } => {
                Some(*phase)
            }
            Self::WithSuppressed { primary, .. } => primary.phase(),
            Self::AdRuleSource { .. } => Some(ErrorPhase::GraphBuild),
            Self::PlaceholderDtypeMismatch { .. }
            | Self::PlaceholderShapeMismatch { .. }
            | Self::PlaceholderRankMismatch { .. }
            | Self::GraphInputCountMismatch { .. }
            | Self::UnexpectedBinding { .. }
            | Self::UnboundPlaceholder { .. }
            | Self::DuplicateBinding { .. } => Some(ErrorPhase::Execution),
            Self::EventDomain { .. } => Some(ErrorPhase::Execution),
            Self::SymbolicShapeConversion { phase, .. } => Some(*phase),
            Self::ShapeExpressionEvaluation { .. } => Some(ErrorPhase::Execution),
            _ => None,
        }
    }
}

fn core_dtype(dtype: DType) -> tenferro_tensor::core::DType {
    match dtype {
        DType::F32 => tenferro_tensor::core::DType::F32,
        DType::F64 => tenferro_tensor::core::DType::F64,
        DType::I32 => tenferro_tensor::core::DType::I32,
        DType::I64 => tenferro_tensor::core::DType::I64,
        DType::Bool => tenferro_tensor::core::DType::Bool,
        DType::C32 => tenferro_tensor::core::DType::C32,
        DType::C64 => tenferro_tensor::core::DType::C64,
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

#[cfg(test)]
mod tests;
