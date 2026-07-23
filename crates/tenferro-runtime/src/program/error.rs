/// Errors reported while adding semantic-program structure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramBuildError {
    /// A value token was issued by another builder.
    #[error("value does not belong to this semantic-program builder")]
    ForeignValue,
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

/// Invalid typed effect-resource identity.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EffectResourceError {
    /// Resource-family names must be nonempty and versioned.
    #[error("effect resource family must be a nonempty versioned identifier")]
    InvalidFamily,
}
