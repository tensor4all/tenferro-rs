//! Immutable backend-neutral semantic programs.

mod builder;
mod error;
mod metadata;
mod op;
mod value;

pub use builder::SemanticProgramBuilder;
pub use error::{EffectResourceError, ProgramBuildError};
pub use metadata::{
    Alias, AliasKind, Effect, EffectAccess, EffectResource, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind, ShapeGuard,
};
pub use op::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
pub use value::ProgramValue;

#[cfg(test)]
mod tests;
