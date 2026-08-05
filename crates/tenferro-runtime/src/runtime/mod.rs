mod cache;
mod cache_owner;
mod capability;
mod engine_registration;
mod error;
mod event_domain;
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
#[cfg(test)]
pub(crate) use engine_registration::ExecutableEngineContract;
pub use engine_registration::{
    assemble_executable_engine_registration, assemble_preparation_only_engine_registration,
    EngineRegistration, EngineRegistrationMetadata, ExecutableEngineRegistrationConfig,
    InputIngressContract, InputPlacementContract, InputSignatureContract,
    PreparationOnlyEngineRegistrationConfig, ResidentOutputContract, RuntimeInputContract,
};
pub(crate) use engine_registration::{ProviderExecutableBinding, ProviderPreparationBinding};
pub use error::{
    EngineExecutionContractError, ExecutionContextMismatch, ExecutionPolicyError,
    ExtensionModuleError, IdentityError, IdentityKind, InputIngressContractError,
    InputSignatureError, InputSpecializationRequirementsError, PlacementConstraintError,
    PrepareError, ProviderContractError, RankRequirement, RegistrationKey, RuntimeConfigError,
    RuntimeReconfigureError, SpecializationError, SubmissionError,
};
pub use event_domain::{
    EventDomainDriver, EventDomainError, EventDomainOperation, EventDomainRun, EventToken,
    ImmediateEventDomainDriver,
};
pub use execution::{
    ExecutionBundle, ExecutionHandle, ExecutionInputs, ExecutionOutcome, OutputAccessError,
    OutputExtractError, OutputMetadata, OutputRef, PreparedCompiledGraph, ScopedExecutionBundle,
    ScopedExecutionOutcome, ScopedOutput, ScopedOutputExtractError, ScopedReadBinding,
    ScopedReadInputs, ScopedSubmitRejected, SubmitError,
};
pub use extension::{ExtensionModule, ExtensionModuleId, ExtensionModuleRegistrar};
pub use extension_provider::{ExtensionEngine, ExtensionPlanningConfig, ExtensionPrepareRequest};
pub use identity::{
    EngineId, ExecutionContextIdentity, HardwareClassId, ProviderDeviceIdentity, ProviderId,
    RegistrationIdentity, RuntimeEpoch, RuntimeId,
};
pub use policy::{
    CacheInFlightBehavior, Determinism, ExecutionPolicy, LayoutClass, PrepareOptions,
    PrepareOptionsKey, ProgramPlacementConstraint, ResolvedPlanningConfig, ResolvedPlanningKey,
    ResolvedProgramPlacement, StorageClass, TransferEndpoint,
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
pub(crate) use transfer::{
    FrozenTransferRegistry, ResolvedTransferEndpoint, ResolvedTransferRoute, TransferRoute,
};
pub use transfer::{
    TransferError, TransferProvider, TransferProviderContractError, TransferRequest,
};

#[cfg(test)]
mod tests;
