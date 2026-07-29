mod cache;
mod cache_owner;
mod capability;
mod engine_registration;
mod error;
mod execution;
mod extension;
mod extension_provider;
mod identity;
mod policy;
mod preparation;
mod schedule;
mod signature;
mod snapshot;
mod specialization;
mod transfer;

pub use cache::{PreparedPlanCacheLimits, PreparedPlanCacheStats, RuntimeCacheStats};
pub use cache_owner::{
    CacheOwnerError, CacheOwnerFailure, CacheOwnerId, CacheStats, RuntimeCacheError,
    RuntimeCacheOwner, RuntimeStateError,
};
pub use capability::{
    CoreCapabilityBundle, CoreCapabilityBundleBuilder, CoreCapabilityKind, CorePrepareContext,
    DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime,
    ErasedExecutionContext, IndexingPrepareRequest, IndexingRuntime, LayoutPrepareRequest,
    LayoutRuntime, PreparationKeySummary, PrepareCapability, PreparedOperation,
    PreparedOperationBinding, PreparedOperationExecutor, PreparedOperationExecutorHandle,
    PreparedOperationHandle, PreparedOperationPlan, ReductionPrepareRequest, ReductionRuntime,
    UnsupportedReason,
};
pub use engine_registration::EngineRegistration;
pub use error::{
    EngineExecutionContractError, ExecutionContextMismatch, ExecutionPolicyError,
    ExtensionModuleError, IdentityError, IdentityKind, InputSignatureError,
    InputSpecializationRequirementsError, PlacementConstraintError, PrepareError,
    ProviderContractError, RankRequirement, RegistrationKey, RuntimeConfigError,
    RuntimeReconfigureError, SpecializationError,
};
pub use execution::{ExecutionHandle, PreparedCompiledGraph};
pub use extension::{ExtensionModule, ExtensionModuleId, ExtensionModuleRegistrar};
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
pub use schedule::EventDomainId;
pub use signature::{InputSignature, InputSignatureEntry};
pub use snapshot::{
    EngineSnapshotView, Runtime, RuntimeConfigBuilder, RuntimeConfigSnapshot,
    RuntimeReconfiguration,
};
pub use specialization::{
    InputSpecializationProjection, InputSpecializationRequirements,
    InputSpecializationRequirementsBuilder, LayoutProjection, LayoutSpecialization,
    PlacementProjection, PlacementSpecialization, SpecializationProjection,
    SpecializationRequirements,
};
pub use transfer::{
    TransferError, TransferProvider, TransferProviderContractError, TransferRequest,
};

#[cfg(test)]
mod tests;
