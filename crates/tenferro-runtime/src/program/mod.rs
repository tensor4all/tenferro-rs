//! Immutable backend-neutral semantic programs.

mod bindings;
mod builder;
mod error;
mod import;
mod metadata;
mod op;
mod semantic;
mod value;

pub use bindings::ProgramBindings;
pub use builder::SemanticProgramBuilder;
pub use error::{
    EffectResourceError, ProgramBindingError, ProgramBuildError, ProgramFinishError,
    ProgramQueryError, ProgramStructuralError,
};
pub use import::{ImportedProgramValues, ProgramImport};
pub use metadata::{
    Alias, AliasKind, Effect, EffectAccess, EffectResource, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind,
    SemanticProvenanceKind, SemanticProvenanceView, ShapeGuard,
};
pub use op::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
pub use semantic::{FrozenProgram, SemanticProgram};
pub use value::{BindingKey, ProgramValue};

#[cfg(test)]
mod tests;
