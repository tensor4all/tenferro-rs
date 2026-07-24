mod cache_owner;
mod capability;
mod error;
mod extension_provider;
mod identity;
mod policy;
mod signature;
mod snapshot;
mod specialization;

pub use cache_owner::{
    CacheOwnerError, CacheOwnerFailure, CacheOwnerId, CacheStats, RuntimeCacheError,
    RuntimeCacheOwner, RuntimeStateError,
};
pub use capability::{
    CoreCapabilityBundle, CoreCapabilityBundleBuilder, CoreCapabilityKind, CorePrepareContext,
    DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime,
    ErasedExecutionContext, IndexingPrepareRequest, IndexingRuntime, LayoutPrepareRequest,
    LayoutRuntime, PreparationKeySummary, PrepareCapability, PreparedOperation,
    PreparedOperationBinding, PreparedOperationHandle, ReductionPrepareRequest, ReductionRuntime,
    UnsupportedReason,
};
pub use error::{
    ExecutionContextMismatch, ExecutionPolicyError, IdentityError, IdentityKind,
    InputSignatureError, InputSpecializationRequirementsError, PlacementConstraintError,
    PrepareError, ProviderContractError, RankRequirement, RegistrationKey, RuntimeConfigError,
    RuntimeReconfigureError, SpecializationError,
};
pub use extension_provider::{ExtensionEngine, ExtensionPlanningConfig, ExtensionPrepareRequest};
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
pub use snapshot::{
    EngineRegistration, EngineSnapshotView, Runtime, RuntimeConfigBuilder, RuntimeConfigSnapshot,
    RuntimeReconfiguration,
};
pub use specialization::{
    InputSpecializationProjection, InputSpecializationRequirements,
    InputSpecializationRequirementsBuilder, LayoutProjection, LayoutSpecialization,
    PlacementProjection, PlacementSpecialization, SpecializationProjection,
    SpecializationRequirements,
};

#[cfg(test)]
mod tests;
