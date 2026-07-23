//! Immutable backend-neutral semantic programs.

mod builder;
mod error;
mod metadata;
mod value;

pub use builder::SemanticProgramBuilder;
pub use error::{EffectResourceError, ProgramBuildError};
pub use metadata::{
    Alias, AliasKind, Effect, EffectAccess, EffectResource, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind, ShapeGuard,
};
pub use value::ProgramValue;

#[cfg(test)]
mod tests;
