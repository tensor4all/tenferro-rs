/// Errors reported while adding semantic-program structure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProgramBuildError {
    /// A value token was issued by another builder.
    #[error("value does not belong to this semantic-program builder")]
    ForeignValue,
    /// The builder cannot represent another value slot.
    #[error("semantic-program value count exceeds the supported u32 range")]
    TooManyValues,
}

/// Invalid typed effect-resource identity.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EffectResourceError {
    /// Resource-family names must be nonempty and versioned.
    #[error("effect resource family must be a nonempty versioned identifier")]
    InvalidFamily,
}
