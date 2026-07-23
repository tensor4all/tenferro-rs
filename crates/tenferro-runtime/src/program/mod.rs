//! Immutable backend-neutral semantic programs.

mod bindings;
mod builder;
mod error;
mod identity;
mod import;
mod metadata;
mod op;
mod semantic;
mod transform;
mod value;

pub use bindings::ProgramBindings;
pub use builder::SemanticProgramBuilder;
pub use error::{
    EffectResourceError, ProgramBindingError, ProgramBuildError, ProgramFinishError,
    ProgramQueryError, ProgramStructuralError, SemanticTransformError,
};
pub use identity::SemanticFingerprint;
pub use import::{ImportedProgramValues, ProgramImport};
pub use metadata::{
    Alias, AliasKind, Effect, EffectAccess, EffectResource, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticPlacementKind,
    SemanticProvenanceKind, SemanticProvenanceView, ShapeGuard,
};
pub use op::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
pub use semantic::{FrozenProgram, SemanticProgram};
pub use transform::{SemanticTransform, SemanticTransformContext, TransformIdentity};
pub use value::{BindingKey, ProgramValue};

#[cfg(test)]
mod tests;
