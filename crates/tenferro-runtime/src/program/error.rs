/// Errors reported while adding semantic-program structure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramBuildError {
    /// A value token was issued by another builder.
    #[error("value does not belong to this semantic-program builder")]
    ForeignValue,
    /// An import root belongs to another semantic program.
    #[error("import root does not belong to the source semantic program")]
    ForeignImportRoot,
    /// Import bindings were frozen for another semantic program.
    #[error("import bindings do not belong to the source semantic program")]
    ForeignBindings,
    /// Structured control flow is reserved for the later region/block model.
    #[error("semantic control-flow construct {construct:?} is not supported")]
    UnsupportedControlFlow {
        /// Frontend construct that could not be represented.
        construct: &'static str,
    },
    /// A frozen source program failed import-time structural validation.
    #[error("source semantic program is invalid for import: {source}")]
    InvalidImport {
        #[source]
        source: ProgramStructuralError,
    },
    /// A binding target is a computed value rather than an external input.
    #[error("tensor bindings may target only semantic-program inputs")]
    BindingTargetNotInput,
    /// A shape guard target is not produced by a semantic operation.
    #[error("semantic shape guards may target only operation outputs")]
    GuardTargetNotOperationOutput,
    /// An external input already has a tensor binding.
    #[error("semantic-program input already has a tensor binding")]
    DuplicateBinding,
    /// The builder cannot represent another value slot.
    #[error("semantic-program value count exceeds the supported u32 range")]
    TooManyValues,
    /// An operation received the wrong number of SSA inputs.
    #[error("semantic operation expects {expected} inputs, got {actual}")]
    Arity {
        /// Declared input count.
        expected: usize,
        /// Supplied input count.
        actual: usize,
    },
    /// Semantic dtype or shape inference failed.
    #[error("semantic operation metadata inference failed: {source}")]
    Metadata {
        /// Original typed runtime inference error.
        #[source]
        source: Box<crate::Error>,
    },
    /// Inference did not return the declared number of outputs.
    #[error("semantic operation declares {expected} outputs, inferred {actual}")]
    OutputMetadataCount {
        /// Declared output count.
        expected: usize,
        /// Inferred metadata count.
        actual: usize,
    },
    /// An extension did not explicitly declare its observable effects.
    #[error("extension family {family:?} has no semantic effect declaration")]
    UndeclaredExtensionEffects {
        /// Stable extension family.
        family: &'static str,
    },
    /// An extension did not explicitly declare output aliases.
    #[error("extension family {family:?} has no semantic alias declaration")]
    UndeclaredExtensionAliases {
        /// Stable extension family.
        family: &'static str,
    },
    /// An extension declared an invalid resource family.
    #[error("extension family {family:?} declared an invalid effect resource: {source}")]
    InvalidEffectResource {
        /// Stable extension family.
        family: &'static str,
        /// Typed resource validation failure.
        #[source]
        source: EffectResourceError,
    },
    /// An alias referenced an input or output outside the operation arity.
    #[error(
        "semantic alias index is out of bounds: output {output}/{output_count}, input {input:?}/{input_count}"
    )]
    AliasOutOfBounds {
        /// Referenced output.
        output: usize,
        /// Output arity.
        output_count: usize,
        /// Referenced input, when applicable.
        input: Option<usize>,
        /// Input arity.
        input_count: usize,
    },
    /// Alias declarations did not cover every output exactly once.
    #[error("semantic aliases must cover {expected} outputs exactly once, got {actual}")]
    AliasCoverage {
        /// Output arity.
        expected: usize,
        /// Number of distinct valid output declarations.
        actual: usize,
    },
}

/// Errors reported while querying an immutable semantic program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProgramQueryError {
    /// A value token belongs to another program or names no frozen value.
    #[error("value does not belong to this semantic program")]
    ForeignValue,
}

/// Internal structural validation failures detected during atomic freeze.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramStructuralError {
    /// An operation references a value outside the frozen value table.
    #[error("semantic operation references a value outside the program")]
    InvalidValueReference,
    /// An operation output is not strictly after all prior values.
    #[error("semantic operation outputs violate SSA ordering")]
    InvalidSsaOrder,
}

/// Tensor-binding validation failures detected during atomic freeze.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramBindingError {
    /// A pending binding no longer names an external input.
    #[error("tensor binding target is not a semantic-program input")]
    InvalidTarget,
    /// Tensor dtype differs from the input declaration.
    #[error("tensor binding dtype mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        /// Declared input dtype.
        expected: tenferro_tensor::DType,
        /// Bound tensor dtype.
        actual: tenferro_tensor::DType,
    },
    /// Tensor rank differs from the input declaration.
    #[error("tensor binding rank mismatch: expected {expected}, got {actual}")]
    RankMismatch {
        /// Declared input rank.
        expected: usize,
        /// Bound tensor rank.
        actual: usize,
    },
    /// A statically exact dimension differs from the input declaration.
    #[error("tensor binding extent mismatch at axis {axis}: expected {expected}, got {actual}")]
    ExactExtentMismatch {
        /// Mismatching axis.
        axis: usize,
        /// Declared exact extent.
        expected: usize,
        /// Bound tensor extent.
        actual: usize,
    },
    /// A bounded dimension exceeds the declared upper bound.
    #[error(
        "tensor binding extent exceeds upper bound at axis {axis}: bound {bound}, got {actual}"
    )]
    UpperBoundExceeded {
        /// Axis whose bound was exceeded.
        axis: usize,
        /// Declared upper bound.
        bound: usize,
        /// Bound tensor extent.
        actual: usize,
    },
}

/// Errors reported while atomically freezing semantic structure and bindings.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramFinishError {
    /// One requested output belongs to another builder or names no value.
    #[error("program output does not belong to this semantic-program builder")]
    ForeignOutput,
    /// Frozen structure failed an invariant check.
    #[error("semantic-program structural validation failed: {source}")]
    StructuralValidation {
        /// Typed structural invariant failure.
        #[source]
        source: ProgramStructuralError,
    },
    /// A tensor default or large constant does not match its input declaration.
    #[error("semantic-program binding finalization failed: {source}")]
    BindingFinalization {
        /// Typed binding mismatch.
        #[source]
        source: ProgramBindingError,
    },
}

/// Failures produced by a validated semantic transform transaction.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SemanticTransformError {
    /// Program construction or import failed inside the transform.
    #[error(transparent)]
    Build(#[from] ProgramBuildError),
    /// Atomic freeze failed after the transform returned roots.
    #[error(transparent)]
    Finish(#[from] ProgramFinishError),
    /// The transform returned a value not owned by its destination builder.
    #[error("semantic transform returned a foreign destination value")]
    ForeignReturnedValue,
    /// The transform did not carry every input tensor binding forward.
    #[error("semantic transform discarded one or more tensor bindings")]
    DroppedBindings,
    /// The transform deliberately rejected this input.
    #[error("semantic transform rejected the input")]
    Rejected,
}

/// Invalid typed effect-resource identity.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EffectResourceError {
    /// Resource-family names must be nonempty and versioned.
    #[error("effect resource family must be a nonempty versioned identifier")]
    InvalidFamily,
}
