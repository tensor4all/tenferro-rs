mod error;
mod identity;
mod policy;
mod signature;
mod specialization;

pub use error::{
    IdentityError, IdentityKind, InputSignatureError, InputSpecializationRequirementsError,
    PlacementConstraintError, PrepareError, RankRequirement, SpecializationError,
};
pub use identity::{
    EngineId, ExecutionContextIdentity, HardwareClassId, RegistrationIdentity, RuntimeEpoch,
    RuntimeId,
};
pub use policy::{
    CacheInFlightBehavior, Determinism, ExecutionPolicy, LayoutClass, PrepareOptions,
    PrepareOptionsKey, ProgramPlacementConstraint, ResolvedPlanningConfig, ResolvedPlanningKey,
    ResolvedProgramPlacement, StorageClass,
};
pub use signature::{InputSignature, InputSignatureEntry};
pub use specialization::{
    InputSpecializationProjection, InputSpecializationRequirements,
    InputSpecializationRequirementsBuilder, LayoutProjection, LayoutSpecialization,
    PlacementProjection, PlacementSpecialization, SpecializationProjection,
    SpecializationRequirements,
};

#[cfg(test)]
mod tests;
