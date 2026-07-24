# Phase 4: Immutable Runtime Snapshots and Preparation Substrate

## Status and provenance

**Proposed child of [#1433](https://github.com/tensor4all/tenferro-rs/issues/1433);
requires review with no Critical or Important gap.**

Authority: reconciled rules, #1433/`e2bfdde4`, Phase 3 #1449 v2.4, then this
design/plan/worklog. Amendment:
[comment 5066178110](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5066178110).
[Maintainer direction](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5066159995)
limits work to Phases 4-6 and moves audit to Phase 6. Baselines: `b5a3dcd2`,
`bb98ee28`, and one private forward adapter.

## Goal

Add immutable runtime snapshots/epochs, transactional reconfiguration, direct
core and resolved extension slots, object-safe immutable plans/context
identity, finite specialization, and bounded single-flight/negative/cycle-safe
preparation caches. Preserve current execution and the one private Phase 3
adapter. Phase 4 prepares plans; it does not schedule or execute a new graph.

## Non-goals

No public `PreparedGraph`; schedule/transfers/collectives/buffers/events/
resource admission/common executor or adapter deletion (Phase 5); extension
lowering/family migration/native N-ary einsum/changing-shape gate (Phase 6);
deferred Phases 7-9; implicit placement transfer; new crate/facade/backend/
feature/external dependency; or AD semantic change.

## Chosen boundary

Provider preparation is public; the aggregate remains crate-private to avoid
freezing Phase 5 execution contracts. `PreparedOperation` is immutable and
identity/specialization-bound with no execution method.

## Ownership and dependency constraints

`tenferro-runtime` owns snapshots, identities, slots, preparation,
specialization, cache aggregation, and private `PreparedProgram`.
`tenferro-cpu` adapts existing providers/domains through the required internal
CPU→runtime production edge; runtime→CPU stays test-only and no external
dependency is added. `tenferro-ad::EagerRuntime` owns a `Runtime`, while all AD
state/rules remain in `tenferro-ad`. Family crates may implement extension
contracts, but migration is Phase 6. `program` remains independent of runtime,
providers, caches, resources, scheduling, and AD. No registry/cache/resource
owner is global/thread-local; identity allocation is checked and non-reusing.

## Public runtime identity and configuration API

The new API lives under `tenferro_runtime::runtime` and is re-exported from the
crate root.

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeId { /* opaque nonzero identity */ }

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeEpoch { /* opaque nonzero generation */ }

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EngineId { /* validated namespaced Arc<str> */ }

impl EngineId {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HardwareClassId { /* validated namespaced Arc<str> */ }

impl HardwareClassId {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegistrationIdentity {
    /* private issuer tag plus nonzero ordinal */
}

impl RegistrationIdentity {
    pub fn ordinal(self) -> NonZeroU64;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionContextIdentity { /* TypeId plus diagnostic type name */ }

impl ExecutionContextIdentity {
    pub fn of<T: Send + Sync + 'static>() -> Self;
    pub fn type_name(&self) -> &'static str;
}

#[derive(Clone)]
pub struct Runtime { /* Arc<RuntimeState> */ }

impl Runtime {
    pub fn builder() -> Result<RuntimeConfigBuilder, RuntimeConfigError>;
    pub fn id(&self) -> RuntimeId;
    pub fn snapshot(&self) -> Result<Arc<RuntimeConfigSnapshot>, RuntimeStateError>;
    pub fn epoch(&self) -> Result<RuntimeEpoch, RuntimeStateError>;

    pub(crate) fn prepare_for(
        &self, semantic: Arc<SemanticProgram>, input_signature: InputSignature,
        options: PrepareOptions
    ) -> Result<Arc<PreparedProgram>, Arc<PrepareError>>;

    pub fn reconfigure(
        &self, edit: impl FnOnce(&mut RuntimeReconfiguration<'_>)
            -> Result<(), RuntimeConfigError>,
    ) -> Result<RuntimeEpoch, RuntimeReconfigureError>;

    pub fn prepared_cache_limits(&self)
        -> Result<PreparedPlanCacheLimits, RuntimeStateError>;
    pub fn set_prepared_cache_limits(
        &self, limits: PreparedPlanCacheLimits
    ) -> Result<(), RuntimeStateError>;
    pub fn clear_prepared_cache(&self) -> Result<(), RuntimeStateError>;
    pub fn clear_caches(&self) -> Result<(), RuntimeCacheError>;
    pub fn cache_stats(&self) -> Result<RuntimeCacheStats, RuntimeCacheError>;
}

#[derive(Clone)]
pub struct RuntimeConfigSnapshot { /* immutable private fields */ }

impl RuntimeConfigSnapshot {
    pub fn runtime_id(&self) -> RuntimeId;
    pub fn epoch(&self) -> RuntimeEpoch;
    pub fn execution_policy(&self) -> &ExecutionPolicy;
    pub fn engine_count(&self) -> usize;
    pub fn extension_module_count(&self) -> usize;
    pub fn engine(&self, id: &EngineId) -> Option<EngineSnapshotView<'_>>;
}

pub struct RuntimeConfigBuilder { /* private candidate registrations */ }

impl RuntimeConfigBuilder {
    pub fn new() -> Result<Self, RuntimeConfigError>;
    pub fn execution_policy(&mut self, value: ExecutionPolicy) -> &mut Self;
    pub fn register_engine(&mut self, value: EngineRegistration)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn replace_engine(&mut self, value: EngineRegistration)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn remove_engine(&mut self, id: &EngineId)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn install_extension_module(&mut self, value: Arc<dyn ExtensionModule>)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn replace_extension_module(&mut self, value: Arc<dyn ExtensionModule>)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn remove_extension_module(&mut self, id: &ExtensionModuleId)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn build(self) -> Result<Runtime, RuntimeConfigError>;
}

pub struct RuntimeReconfiguration<'a> { /* non-escapable candidate */ }

impl RuntimeReconfiguration<'_> {
    pub fn execution_policy(&mut self, policy: ExecutionPolicy) -> &mut Self;
    pub fn register_engine(&mut self, value: EngineRegistration)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn replace_engine(&mut self, value: EngineRegistration)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn remove_engine(&mut self, id: &EngineId)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn install_extension_module(&mut self, value: Arc<dyn ExtensionModule>)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn replace_extension_module(&mut self, value: Arc<dyn ExtensionModule>)
        -> Result<&mut Self, RuntimeConfigError>;
    pub fn remove_extension_module(&mut self, id: &ExtensionModuleId)
        -> Result<&mut Self, RuntimeConfigError>;
}

pub struct EngineSnapshotView<'a> { /* immutable borrowed slot view */ }

impl<'a> EngineSnapshotView<'a> {
    pub fn engine_id(&self) -> &'a EngineId;
    pub fn registration_identity(&self) -> RegistrationIdentity;
    pub fn context_identity(&self) -> ExecutionContextIdentity;
    pub fn hardware_class(&self) -> &'a HardwareClassId;
    pub fn capabilities(&self) -> &'a CoreCapabilityBundle;
}
```

Every public type in this issue implements `Debug`. Foundational IDs, enums,
placement/options/signature/specialization values derive it as shown; runtime,
builder, registration, request, registrar, and trait-object containers use
manual bounded output with IDs/counts only—never providers, tensors, or plan
payloads.

`EngineId` and `ExtensionModuleId` validate ASCII namespaced identifiers.
`RegistrationIdentity` has no public issuer. Consuming a fresh, non-`Clone`
builder allocates a fresh `RuntimeId`, epoch one, and private registration
issuer, then assigns identities while freezing registrations. `RuntimeState`
alone owns its checked atomic next ordinal. A reconfiguration candidate cannot
escape and has no `build`; unchanged records retain identities, while
new/replaced records receive identities only inside successful publication.
Failure allocates none. Ordinals never wrap/recycle, and no public path can
fork a runtime ID or issuer.

Before runtime identity issuance, engine equality is `(EngineId,
Arc::ptr_eq(candidate_token))`; modules/extension engines use their stable keys
plus `Arc::ptr_eq`, and planning configs use `payload_eq`. Identical repeats
are no-ops that preserve identity and skip reconfiguration; same-key unequal
values conflict; explicit replacement receives a new runtime identity.

## Execution and planning policy

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Determinism {
    Fast,
    Reproducible,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StorageClass(Arc<str>);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LayoutClass(Arc<str>);

impl StorageClass {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}
impl LayoutClass {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProgramPlacementConstraint {
    allowed_engines: Arc<[EngineId]>, /* empty means any */
    storage_class: Option<StorageClass>,
}

impl ProgramPlacementConstraint {
    pub fn any() -> Self;
    pub fn new(
        allowed_engines: impl Into<Arc<[EngineId]>>,
        storage_class: Option<StorageClass>,
    ) -> Result<Self, PlacementConstraintError>;
    pub fn allowed_engines(&self) -> &[EngineId];
    pub fn storage_class(&self) -> Option<&StorageClass>;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedProgramPlacement {
    engine_id: EngineId,
    storage_class: StorageClass,
}

impl ResolvedProgramPlacement {
    pub fn engine_id(&self) -> &EngineId;
    pub fn storage_class(&self) -> &StorageClass;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheInFlightBehavior {
    Wait,
    Refuse,
}

impl Default for CacheInFlightBehavior {
    fn default() -> Self { Self::Wait }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionPolicy {
    determinism: Determinism,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
}

impl ExecutionPolicy {
    pub fn new(
        determinism: Determinism,
        hard_workspace_limit_bytes: Option<usize>,
        planning_seed: u64,
    ) -> Result<Self, ExecutionPolicyError>;
    pub fn determinism(&self) -> Determinism;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> u64;
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PrepareOptions {
    placement: Option<ProgramPlacementConstraint>,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: Option<u64>,
    cache_in_flight: CacheInFlightBehavior,
}

impl PrepareOptions {
    pub fn new() -> Self;
    pub fn with_placement(
        self,
        placement: ProgramPlacementConstraint,
    ) -> Self;
    pub fn with_hard_workspace_limit_bytes(
        self,
        limit: Option<usize>,
    ) -> Self;
    pub fn with_planning_seed(self, seed: u64) -> Self;
    pub fn with_cache_in_flight(
        self,
        behavior: CacheInFlightBehavior,
    ) -> Self;
    pub fn placement(&self) -> Option<&ProgramPlacementConstraint>;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> Option<u64>;
    pub fn cache_in_flight(&self) -> CacheInFlightBehavior;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PrepareOptionsKey { /* runtime-created exact option identity */ }

impl PrepareOptionsKey {
    pub fn placement(&self) -> Option<&ProgramPlacementConstraint>;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> Option<u64>;
    pub fn cache_in_flight(&self) -> CacheInFlightBehavior;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedPlanningConfig {
    determinism: Determinism,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
    hardware_class: HardwareClassId,
}

impl ResolvedPlanningConfig {
    pub fn resolve(
        policy: &ExecutionPolicy,
        options: &PrepareOptions,
        hardware_class: HardwareClassId,
    ) -> Result<Self, PrepareError>;
    pub fn determinism(&self) -> Determinism;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> u64;
    pub fn hardware_class(&self) -> &HardwareClassId;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResolvedPlanningKey { /* exact bounded key projection */ }

impl ResolvedPlanningKey {
    pub fn determinism(&self) -> Determinism;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> u64;
    pub fn hardware_class(&self) -> &HardwareClassId;
}
```

Resolution happens once before capability preparation. `PrepareOptions`
overrides only fields represented as `Option`; all inherited values become
concrete in `ResolvedPlanningConfig`. A hard limit of zero is valid and means
that no workspace bytes may be planned.

Extension-specific inherited planning defaults use
`ExtensionPlanningConfig`, registered transactionally with the extension
module:

```rust
pub trait ExtensionPlanningConfig:
    Any + Debug + Send + Sync + 'static
{
    fn family_id(&self) -> ExtensionFamilyId;
    fn as_any(&self) -> &dyn Any;
    fn payload_hash(&self, state: &mut dyn Hasher);
    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool;
    fn retained_bytes(&self) -> usize;
}
```

The snapshot retains the exact config object and checks `payload_eq` after a
fingerprint collision. Extension preparation receives it with the existing
`ExtensionOp`, resolved common config, epoch, options key, specialization,
inputs, placement, and hardware class.

`ResolvedPlanningKey` is constructed only by the runtime from
`ResolvedPlanningConfig`. Extension-config fingerprints are stored beside
their exact retained objects in `PreparedProgramRoot`.

`allowed_engines` is an ordered preference list; duplicates are invalid.
Empty (`any`) means snapshot engines in ascending `EngineId` order. For each
candidate, an explicit storage class must be supported; otherwise its
registered default is used. The first eligible engine wins, so ties do not
exist. `Reproducible` and `Fast` use this same stable selection—load/free
memory may affect neither placement nor cache identity. Failure is
`PrepareError::NoEligibleEngine`. Placement resolves before cache lookup.
Engine registration validates a nonempty unique storage list and a default
member; eligibility also requires every capability needed by the program.
Duplicate preferences fail `ProgramPlacementConstraint::new` with indexed
`PlacementConstraintError::DuplicateEngine`. `EngineRegistration::new`
returns the displayed `RuntimeConfigError` storage variants. These are
malformed configuration; `NoEligibleEngine` is only for a valid constraint
that the captured snapshot cannot satisfy.

## Engine registrations and direct core capability slots

```rust
#[derive(Clone)]
pub struct EngineRegistration {
    /* private immutable fields including Arc<CandidateRegistrationToken> */
}

struct CandidateRegistrationToken { /* fresh per EngineRegistration::new */ }

impl EngineRegistration {
    pub fn new(
        engine_id: EngineId,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        capabilities: CoreCapabilityBundle,
    ) -> Result<Self, RuntimeConfigError>;
    pub fn engine_id(&self) -> &EngineId;
    pub fn context_identity(&self) -> ExecutionContextIdentity;
    pub fn hardware_class(&self) -> &HardwareClassId;
    pub fn storage_classes(&self) -> &[StorageClass];
    pub fn default_storage_class(&self) -> &StorageClass;
    pub fn capabilities(&self) -> &CoreCapabilityBundle;
    pub fn with_cache_owner(self, owner: Arc<dyn RuntimeCacheOwner>) -> Self;
}

#[derive(Clone, Default)]
pub struct CoreCapabilityBundle {
    /* five direct public slots plus crate-private reserved subgraph marker */
}

impl CoreCapabilityBundle {
    pub fn builder() -> CoreCapabilityBundleBuilder;
    pub fn elementwise(&self) -> Option<&Arc<dyn ElementwiseRuntime>>;
    pub fn reduction(&self) -> Option<&Arc<dyn ReductionRuntime>>;
    pub fn indexing(&self) -> Option<&Arc<dyn IndexingRuntime>>;
    pub fn dot_general(&self) -> Option<&Arc<dyn DotGeneralRuntime>>;
    pub fn layout(&self) -> Option<&Arc<dyn LayoutRuntime>>;
}

pub struct CoreCapabilityBundleBuilder { /* one optional direct slot each */ }

impl CoreCapabilityBundleBuilder {
    pub fn new() -> Self;
    pub fn elementwise(
        &mut self,
        capability: Arc<dyn ElementwiseRuntime>,
    ) -> Result<&mut Self, RuntimeConfigError>;
    pub fn reduction(
        &mut self,
        capability: Arc<dyn ReductionRuntime>,
    ) -> Result<&mut Self, RuntimeConfigError>;
    pub fn indexing(
        &mut self,
        capability: Arc<dyn IndexingRuntime>,
    ) -> Result<&mut Self, RuntimeConfigError>;
    pub fn dot_general(
        &mut self,
        capability: Arc<dyn DotGeneralRuntime>,
    ) -> Result<&mut Self, RuntimeConfigError>;
    pub fn layout(
        &mut self,
        capability: Arc<dyn LayoutRuntime>,
    ) -> Result<&mut Self, RuntimeConfigError>;
    pub fn build(self) -> Result<CoreCapabilityBundle, RuntimeConfigError>;
}

pub trait ElementwiseRuntime: Debug + Send + Sync + 'static {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub trait ReductionRuntime: Debug + Send + Sync + 'static {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub trait IndexingRuntime: Debug + Send + Sync + 'static {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub trait DotGeneralRuntime: Debug + Send + Sync + 'static {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub trait LayoutRuntime: Debug + Send + Sync + 'static {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub struct ElementwisePrepareRequest<'a> { /* runtime-created */ }
pub struct ReductionPrepareRequest<'a> { /* runtime-created */ }
pub struct IndexingPrepareRequest<'a> { /* runtime-created */ }
pub struct DotGeneralPrepareRequest<'a> { /* runtime-created */ }
pub struct LayoutPrepareRequest<'a> { /* runtime-created */ }

pub struct CorePrepareContext<'a> { /* runtime-created borrowed view */ }

impl<'a> CorePrepareContext<'a> {
    pub fn binding(&self) -> &'a PreparedOperationBinding;
    pub fn inputs(&self) -> &'a InputSignature;
    pub fn resolved_placement(&self) -> &'a ResolvedProgramPlacement;
    pub fn planning(&self) -> &'a ResolvedPlanningConfig;
    pub fn prepare_options(&self) -> &'a PrepareOptions;
    pub fn prepare_options_key(&self) -> &'a PrepareOptionsKey;
    pub fn specialization(&self) -> &'a SpecializationProjection;
}

impl<'a> ElementwisePrepareRequest<'a> {
    pub fn operation(&self) -> &'a SemanticOperationView<'a>;
    pub fn context(&self) -> &'a CorePrepareContext<'a>;
}
impl<'a> ReductionPrepareRequest<'a> {
    pub fn operation(&self) -> &'a SemanticOperationView<'a>;
    pub fn context(&self) -> &'a CorePrepareContext<'a>;
}
impl<'a> IndexingPrepareRequest<'a> {
    pub fn operation(&self) -> &'a SemanticOperationView<'a>;
    pub fn context(&self) -> &'a CorePrepareContext<'a>;
}
impl<'a> DotGeneralPrepareRequest<'a> {
    pub fn operation(&self) -> &'a SemanticOperationView<'a>;
    pub fn context(&self) -> &'a CorePrepareContext<'a>;
}
impl<'a> LayoutPrepareRequest<'a> {
    pub fn operation(&self) -> &'a SemanticOperationView<'a>;
    pub fn context(&self) -> &'a CorePrepareContext<'a>;
}
```

Constructors remain crate-private. The runtime rejects
family/metadata/shape/placement/determinism errors before dispatch.

The bundle contains a crate-private reserved subgraph-slot marker, but Phase 4
exports no `SubgraphCompiler`, request, accessor, or provider registration
SPI. A later accepted design must define subgraph/XLA/fusion preparation.

## Prepared-operation and execution-context contracts

```rust
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PreparedOperationBinding {
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    engine_id: EngineId,
    registration_identity: RegistrationIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
}

impl PreparedOperationBinding {
    pub(crate) fn new(
        runtime_id: RuntimeId,
        epoch: RuntimeEpoch,
        engine_id: EngineId,
        registration_identity: RegistrationIdentity,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
    ) -> Self;
    pub fn runtime_id(&self) -> RuntimeId;
    pub fn epoch(&self) -> RuntimeEpoch;
    pub fn engine_id(&self) -> &EngineId;
    pub fn registration_identity(&self) -> RegistrationIdentity;
    pub fn context_identity(&self) -> ExecutionContextIdentity;
    pub fn hardware_class(&self) -> &HardwareClassId;
}

pub trait PreparedOperation: Debug + Send + Sync + 'static {
    fn binding(&self) -> &PreparedOperationBinding;
    fn specialization(&self) -> &SpecializationProjection;
    fn retained_bytes(&self) -> usize;
}

pub type PreparedOperationHandle = Arc<dyn PreparedOperation>;

pub enum PrepareCapability {
    Prepared(PreparedOperationHandle),
    NeedsSpecialization(SpecializationRequirements),
    Unsupported(UnsupportedReason),
}

pub struct ErasedExecutionContext<'a> {
    identity: ExecutionContextIdentity,
    value: &'a mut (dyn Any + Send + Sync),
}

impl<'a> ErasedExecutionContext<'a> {
    pub fn new<T: Send + Sync + 'static>(value: &'a mut T) -> Self;
    pub fn identity(&self) -> ExecutionContextIdentity;
    pub fn downcast_mut<T: Send + Sync + 'static>(
        &mut self,
        expected: ExecutionContextIdentity,
    ) -> Result<&mut T, ExecutionContextMismatch>;
}
```

The traits are object-safe. A prepared object is immutable and contains no
mutable scratch, input/output tensor, runtime handle, resource lease,
scheduler state, or public tensor wrapper. `retained_bytes` reports logical
heap payload owned exclusively by the plan. It excludes shared engine state,
cache metadata, caller tensors, and inline object size.

The returned plan must retain the request's runtime-created binding and match
runtime/epoch/engine/registration/context/hardware; mismatch is an uncacheable
provider error. `Prepared` asserts dependence only on its projection.
`ErasedExecutionContext` performs the sole safe `TypeId` check; Phase 5 owns
invocation and may not add another plan downcast/context lookup.

## Extension modules and snapshot-local slots

```rust
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExtensionModuleId { /* validated namespaced Arc<str> */ }

impl ExtensionModuleId {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}

pub trait ExtensionModule: Debug + Send + Sync + 'static {
    fn module_id(&self) -> &ExtensionModuleId;
    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError>;
}

pub trait ExtensionEngine: Debug + Send + Sync + 'static {
    fn family_id(&self) -> ExtensionFamilyId;
    fn engine_id(&self) -> &EngineId;
    fn context_identity(&self) -> ExecutionContextIdentity;
    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

pub struct ExtensionPrepareRequest<'a> { /* runtime-created borrowed view */ }
struct ExtensionRegistrationTransaction { /* private candidate state */ }
pub struct ExtensionModuleRegistrar<'a> {
    transaction: &'a mut ExtensionRegistrationTransaction,
}

impl<'a> ExtensionPrepareRequest<'a> {
    pub fn operation(&self) -> &'a dyn ExtensionOp;
    pub fn binding(&self) -> &'a PreparedOperationBinding;
    pub fn resolved_placement(&self) -> &'a ResolvedProgramPlacement;
    pub fn hardware_class(&self) -> &'a HardwareClassId;
    pub fn planning(&self) -> &'a ResolvedPlanningConfig;
    pub fn extension_config(&self) -> &'a dyn ExtensionPlanningConfig;
    pub fn inputs(&self) -> &'a InputSignature;
    pub fn prepare_options(&self) -> &'a PrepareOptions;
    pub fn prepare_options_key(&self) -> &'a PrepareOptionsKey;
    pub fn specialization(&self) -> &'a SpecializationProjection;
}

impl ExtensionModuleRegistrar<'_> {
    pub fn register_engine(
        &mut self,
        engine: Arc<dyn ExtensionEngine>,
    ) -> Result<(), ExtensionModuleError>;
    pub fn register_planning_config(
        &mut self,
        engine_id: EngineId,
        config: Arc<dyn ExtensionPlanningConfig>,
    ) -> Result<(), ExtensionModuleError>;
    pub fn register_cache_owner(
        &mut self,
        id: CacheOwnerId,
        owner: Arc<dyn RuntimeCacheOwner>,
    ) -> Result<(), ExtensionModuleError>;
}
```

`ExtensionPrepareRequest::operation` is the existing Phase 3 `ExtensionOp`.
Its existing structural payload/hash/equality is the sole semantic
operation-local identity and policy source. Phase 4 adds no parallel semantic
policy trait, changes no Phase 3 semantic API, and migrates no operation
family. Root fingerprint collisions are resolved with exact
`SemanticProgram`/`ExtensionOp` structural equality.

Module configuration is failure-atomic; identical/conflict/replacement follows
the rules above. Freeze assigns stable private slots, eliminating steady locks,
strings, hashes, and downcasts. The registrar borrows one private transaction,
cannot escape `configure`, and drops engines/configs/cache owners together on
failure. Production lowering stays unchanged; Phase 6 P6-MIGRATE-FAMILIES
deletes temporary `register_extension` staging without promising support.

## Runtime snapshot and reconfiguration semantics

Fresh `build` consumes its builder, allocates a new nonzero `RuntimeId`,
validates all records/policy, creates the unique issuer, assigns registration
identities, freezes slots, creates epoch one/caches, and publishes once.

Reconfiguration snapshots the active `Arc`, edits/validates a non-escapable
candidate off-lock, checks the next epoch, then publishes under a short lock
only if the base pointer is still current; otherwise it returns
`ConcurrentReconfiguration`. Failure allocates no identities and publishes
nothing. Successful publication assigns identities to new/replaced records,
advances once, and clears retained cache state; running old-generation work
may finish but cannot reinsert. Preparation pins its snapshot, ID, and epoch.

A placement-bound eager context caches runtime, placement, epoch, engine slot,
and capability slots. One epoch comparison uses them or refreshes from one new
snapshot. It owns no idle resource/session/registry lock.
`EagerRuntime::on_cpu` and managed/external executor behavior remain unchanged;
no second runtime identity/backend mutex is added.

## Input signatures and finite specialization

```rust
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignature { entries: Arc<[InputSignatureEntry]> }

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignatureEntry {
    dtype: DType,
    shape: ShapeVec,
    placement: Placement,
    layout_class: LayoutClass,
    strides: StrideVec,
    alignment_log2: u8,
}

impl InputSignature {
    pub fn from_reads(inputs: &[TensorRead<'_>]) -> Result<Self, PrepareError>;
    pub fn new(
        entries: impl Into<Arc<[InputSignatureEntry]>>,
    ) -> Result<Self, InputSignatureError>;
    pub fn entries(&self) -> &[InputSignatureEntry];
}

impl InputSignatureEntry {
    pub fn new(
        dtype: DType,
        shape: ShapeVec,
        placement: Placement,
        layout_class: LayoutClass,
        strides: StrideVec,
        alignment_log2: u8,
    ) -> Result<Self, InputSignatureError>;
    pub fn dtype(&self) -> DType;
    pub fn shape(&self) -> &[usize];
    pub fn placement(&self) -> &Placement;
    pub fn layout_class(&self) -> &LayoutClass;
    pub fn strides(&self) -> &[isize];
    pub fn alignment_log2(&self) -> u8;
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PlacementSpecialization {
    None,
    StorageClass,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LayoutSpecialization {
    None,
    Class,
    ExactStrides,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSpecializationRequirements {
    dtype: bool,
    rank: bool,
    concrete_dimensions: Box<[u32]>,
    placement: PlacementSpecialization,
    layout: LayoutSpecialization,
    alignment_log2: Option<u8>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpecializationRequirements {
    inputs: Box<[InputSpecializationRequirements]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpecializationProjection {
    requirements: SpecializationRequirements,
    inputs: Box<[InputSpecializationProjection]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSpecializationProjection { /* exact selected values */ }

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PlacementProjection {
    StorageClass(StorageClass),
    Device(Placement),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LayoutProjection {
    Class(LayoutClass),
    ExactStrides(StrideVec),
}

impl InputSpecializationRequirements {
    pub fn builder() -> InputSpecializationRequirementsBuilder;
    pub fn specializes_dtype(&self) -> bool;
    pub fn specializes_rank(&self) -> bool;
    pub fn concrete_dimensions(&self) -> &[u32];
    pub fn placement(&self) -> PlacementSpecialization;
    pub fn layout(&self) -> LayoutSpecialization;
    pub fn alignment_log2(&self) -> Option<u8>;
}

pub struct InputSpecializationRequirementsBuilder { /* validated on build */ }

impl InputSpecializationRequirementsBuilder {
    pub fn new() -> Self;
    pub fn dtype(&mut self, enabled: bool) -> &mut Self;
    pub fn rank(&mut self, enabled: bool) -> &mut Self;
    pub fn concrete_dimensions(
        &mut self,
        axes: impl Into<Box<[u32]>>,
    ) -> &mut Self;
    pub fn placement(&mut self, level: PlacementSpecialization) -> &mut Self;
    pub fn layout(&mut self, level: LayoutSpecialization) -> &mut Self;
    pub fn alignment_log2(&mut self, value: Option<u8>) -> &mut Self;
    pub fn build(
        self,
    ) -> Result<InputSpecializationRequirements, SpecializationError>;
}

impl SpecializationRequirements {
    pub fn polymorphic(input_count: usize) -> Self;
    pub fn new(
        inputs: impl Into<Box<[InputSpecializationRequirements]>>,
    ) -> Result<Self, SpecializationError>;
    pub fn inputs(&self) -> &[InputSpecializationRequirements];
    pub fn strictly_widens(&self, previous: &Self) -> bool;
    pub fn project(
        &self,
        signature: &InputSignature,
    ) -> Result<SpecializationProjection, PrepareError>;
}

impl SpecializationProjection {
    pub fn requirements(&self) -> &SpecializationRequirements;
    pub fn inputs(&self) -> &[InputSpecializationProjection];
}

impl InputSpecializationProjection {
    pub fn dtype(&self) -> Option<DType>;
    pub fn rank(&self) -> Option<usize>;
    pub fn concrete_dimensions(&self) -> &[(u32, usize)];
    pub fn placement(&self) -> Option<&PlacementProjection>;
    pub fn layout(&self) -> Option<&LayoutProjection>;
    pub fn alignment_log2(&self) -> Option<u8>;
}
```

`InputSignatureEntry::alignment_log2` is the minimum guaranteed power-of-two
alignment class: class `a` guarantees the first logical byte is divisible by
`2^a`. `from_reads` checked-adds base address and byte offset; for allocated
storage it uses `min(address.trailing_zeros(), declared_alignment.trailing_zeros())`,
capped at `usize::BITS - 1`. Empty/unallocated storage uses its declared
nonzero power-of-two guarantee. Zero/non-power-of-two declarations,
address overflow, contradiction between pointer and declaration, or a class
above `usize::BITS - 1` is typed invalid metadata.
`from_reads` and projection wrap these cases as
`PrepareError::Specialization`.

A requirement `Some(k)` projects `min(actual_class, k)`; thus all inputs
guaranteeing at least `2^k` share key `k`, while weaker inputs remain distinct.
Alignment widens only
`None < Some(0) < ... < Some(usize::BITS - 1)`. Other fields widen
componentwise through their displayed finite orders. `NeedsSpecialization`
must strictly increase and never lower/incompare any component. The checked
retry bound is the sum of remaining component edges, including
`usize::BITS` alignment edges per still-polymorphic input; sum overflow is
`ProjectionOverflow`, and exhaustion is `RetryLimit`.

Construction also checks arity, axes, and rank/stride implications.
Projections/options keys are runtime-created and exclude values, pointers,
free memory, scheduler state, diagnostic strings, and private provider keys.

## Crate-private preparation aggregate

```rust
pub(crate) struct PreparedProgram {
    root: Arc<PreparedProgramRoot>,
    specialization: SpecializationProjection,
    operations: Box<[PreparedOperationHandle]>,
}

pub(crate) struct PreparedProgramRoot {
    semantic: Arc<SemanticProgram>,
    staging: Arc<ExecProgram>,
    extension_planning: Arc<[Arc<dyn ExtensionPlanningConfig>]>,
    logical_retained_bytes: usize,
}
```

Neither type owns bindings, tensor guards, eager tensors, or public tensor
wrappers. Bindings stay in `CompiledGraph`/call state. Phase 5 may borrow
`(&Arc<PreparedProgram>, &ProgramBindings)`, derive/project a fresh signature,
and require exact projection equality, but must not retain either. The root
shares one semantic program and binding-free Phase 3 staging artifact across
specializations; no reverse adapter, clone, public accessor, or sibling caller
is added.

`Runtime::prepare_for` owns its semantic `Arc`, metadata signature, and options;
pins one snapshot; resolves placement/engine/registration/context/hardware/
planning before lookup; and returns a shared binding-free plan/error. It is
crate-private to avoid publishing `PreparedGraph`. Validation, concrete-shape
lookup, and miss planning precede Phase 5 resource admission. It creates no
transfer, collective, schedule, liveness/buffer plan, resource, event,
execution, or extension lowering.

Until Phase 5, production execution continues through the existing
`GraphExecutor<B>` and private `ExecProgram` path. The new preparation
aggregate is exercised by focused runtime tests and cache integration, not by
a second public execution pipeline.

## Prepared-plan cache

The cache is owned by `RuntimeState`, never by `SemanticProgram`,
`GraphCompiler`, a thread-local, or a process-global singleton.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedPlanCacheLimits {
    pub max_entries: NonZeroUsize,
    pub max_retained_bytes: NonZeroUsize,
    pub max_in_flight_entries: NonZeroUsize,
    pub max_queued_distinct_keys: NonZeroUsize,
}

impl Default for PreparedPlanCacheLimits {
    fn default() -> Self;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedPlanCacheStats {
    pub entries: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub waits: u64,
    pub negative_hits: u64,
    pub preparations: u64,
    pub evictions: u64,
    pub redirects: u64,
    pub in_flight: usize,
    pub peak_in_flight: usize,
    pub queued_distinct_keys: usize,
    pub capacity_refusals: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheOwnerId(Arc<str>);

impl CacheOwnerId {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub entries: usize,
    pub retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub clears: u64,
}

pub trait RuntimeCacheOwner: Debug + Send + Sync + 'static {
    fn cache_stats(&self) -> Result<CacheStats, CacheOwnerError>;
    fn clear_caches(&self) -> Result<(), CacheOwnerError>;
}

#[derive(Clone, Debug)]
pub struct CacheOwnerError { /* Arc<dyn Error + Send + Sync> */ }

impl CacheOwnerError {
    pub fn new(source: Arc<dyn Error + Send + Sync>) -> Self;
    pub fn source_arc(&self) -> &Arc<dyn Error + Send + Sync>;
}

#[derive(Clone, Debug)]
pub struct CacheOwnerFailure {
    pub owner: CacheOwnerId,
    pub source: CacheOwnerError,
}

#[derive(Debug)]
pub enum RuntimeCacheError {
    Aggregate {
        runtime: Option<RuntimeStateError>,
        owners: Box<[CacheOwnerFailure]>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeCacheStats {
    pub prepared_plans: PreparedPlanCacheStats,
    pub engines: CacheStats,
    pub extensions: CacheStats,
}
```

The default is 128 retained entries, 64 MiB logical retained payload, 16
simultaneous provider preparations, and 64 queued distinct keys. Tests use
small explicit limits; production defaults are changed only through the
documented performance protocol.

Snapshot freeze registers the optional engine owner and transactionally
registered extension owners in deterministic engine-slot then
`(module_id, CacheOwnerId)` order. `cache_stats` calls every owner without a
runtime lock and returns saturated engine/extension aggregates only if all
succeed. `clear_caches` first advances/clears the prepared cache, then calls
every owner exactly once even after failures. Either operation returns one
`RuntimeCacheError::Aggregate` containing the runtime failure, if any, plus
all owner failures in snapshot order; no partial stats are returned. Owner
hooks must not call `Runtime::clear_caches` or `cache_stats`.

Registrar `CacheOwnerId` values are module-local validated names. Snapshot
canonical IDs are unambiguous: engine owners format
`engine[<id-bytes>]:<EngineId>`; extension owners format
`extension[<module-bytes>]:<ModuleId>[<local-bytes>]:<local>`.
Equality/order is the canonical byte string, and failures carry that typed ID.

The private keys are:

```rust
struct PreparedRootKey {
    semantic_fingerprint: SemanticFingerprint,
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    resolved_placement: ResolvedProgramPlacement,
    engine_id: EngineId,
    registration_identity: RegistrationIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    resolved_planning: ResolvedPlanningKey,
    prepare_options: PrepareOptionsKey,
    operation_bindings: OperationBindingKeySet,
    extension_planning: ExtensionPlanningKeySet,
}

struct PreparedEntryKey {
    root: PreparedRootKey,
    requirements: SpecializationRequirements,
    specialization: SpecializationProjection,
}
```

Each root bucket retains the exact `Arc<SemanticProgram>` and compares
`semantic_eq` after fingerprint lookup. Extension config and operation
identity use compact hashes followed by exact `payload_eq`. A collision can
increase bucket length but cannot return a wrong plan. Resolved placement,
engine, registration, execution-context identity, hardware class, common
planning key, every operation's binding identity, extension planning objects,
and the semantic program's exact `ExtensionOp` payloads are all part of root
identity; no plan crosses any of those domains.

The entry state machine is:

```rust
enum PreparedEntryState {
    Preparing,
    Ready(Arc<PreparedProgram>),
    Redirect(SpecializationRequirements),
    FailedDeterministic(Arc<PrepareError>),
    FailedTransient(Arc<PrepareError>),
}
```

Contracts:

- One producer exists per exact key; same-key callers synchronously wait for
  its shared result. Different top-level keys may prepare concurrently. No
  cache/LRU lock is held across provider, module, equality, or projection code.
- `NeedsSpecialization(next)` validates strict widening, atomically publishes
  `Redirect(next)`, and wakes waiters. Each caller reprojects its own signature
  and looks up `(root, next, projection)` in O(1). Divergent signatures branch
  without scans. Ready is inserted only under the successful key, and every
  redirect consumes the precomputed finite retry bound.
- Before taking or waiting for in-flight capacity, the scoped
  `PreparationStack` is checked. A key already on the stack returns
  `PreparationCycle`; any other request while the stack is nonempty returns
  `NestedPreparationUnsupported`. Phase 4 permits no nested distinct-key
  preparation and has no reentrant/global wait graph. Phase 6 must design
  explicit preparation dependencies separately if required.
- At most `max_in_flight_entries` top-level provider calls run. Same-key
  waiters consume no extra slot. At capacity, a new distinct key either
  returns `CacheInFlightCapacityExceeded` for `Refuse`, or waits synchronously
  in a FIFO bounded by `max_queued_distinct_keys`; a full FIFO also refuses.
  Unwind/drop guards release slots/queue records and wake the next waiter.
  This is cache capacity; resource admission remains Phase 5.
- Only unsupported capability, unsupported determinism, and stable provider
  contract rejection are negative-cached. Other failures are shared with
  current waiters and then removed. Providers cannot relabel failures.
- Ready, deterministic-negative, and unreferenced redirect entries share one
  bounded O(1) LRU. Oversized or accounting-overflow artifacts are returned
  but not retained. Limit changes evict immediately; running attempts survive.
- Clear or reconfiguration increments the cache generation, removes every
  retained ready, negative, and redirect entry, and releases roots not held by
  a running attempt. A running attempt may finish for existing callers, but
  generation mismatch prevents reinsertion.
- `entries` counts retained ready/negative/redirect entries and is zero just
  after clear. `retained_bytes` includes exact cache-owned bytes still held by
  running attempts and may remain nonzero until they finish or unwind.
  In-flight/queue gauges are instantaneous; peak/event counters are saturating
  cumulative values and clear does not reset them.

Logical retained bytes are exact, not allocated-capacity estimates. A root
charges `SemanticProgram::retained_bytes()` plus the binding-free
`ExecProgram::retained_bytes()`, exact extension-config payloads, and root
metadata exactly once while any ready, redirect, negative, or in-flight entry
references it. A ready entry then charges only its
requirements/projection, prepared-operation payloads, and cache metadata; a
negative or redirect entry charges its error or redirect payload. An active or
queued attempt additionally charges its metadata-only `InputSignature` until
completion. Shared engine/provider/cache resources are counted by their
owners. Saturating arithmetic is used only for public statistics; an insertion
whose exact checked sum overflows is returned but not retained.

The cache contains no strong edge to `Runtime`, `EagerRuntime`, `EagerTensor`,
or another public tensor wrapper. Weak sentinels prove release after clear or
runtime drop once external plan handles and running attempts are gone.

## Errors

All public failures are typed and preserve sources.

```rust
#[derive(Debug)]
pub enum PlacementConstraintError {
    DuplicateEngine {
        engine_id: EngineId,
        first_index: usize,
        duplicate_index: usize,
    },
}

#[derive(Debug)]
pub enum SpecializationError {
    WrongInputCount { expected: usize, actual: usize },
    DuplicateAxis { input: usize, axis: u32 },
    AxisOutOfRange { input: usize, axis: u32, rank: usize },
    RankRequired { input: usize },
    InvalidAlignmentClass { input: usize, alignment_log2: u8 },
    InvalidAlignmentMetadata { input: usize, declared_bytes: usize },
    AddressOverflow { input: usize },
    NonMonotonicSpecialization,
    ProjectionOverflow,
    RetryLimit { attempts: usize, limit: usize },
}

#[derive(Debug)]
pub enum RuntimeConfigError {
    IdentityExhausted,
    MalformedIdentity { kind: IdentityKind, value: Arc<str> },
    DuplicateEngine { engine_id: EngineId },
    ConflictingRegistration { key: RegistrationKey },
    MissingEngine { engine_id: EngineId },
    EmptyStorageClasses { engine_id: EngineId },
    DuplicateStorageClass {
        engine_id: EngineId,
        storage_class: StorageClass,
        first_index: usize,
        duplicate_index: usize,
    },
    DefaultStorageClassNotListed {
        engine_id: EngineId,
        default_storage_class: StorageClass,
    },
    ContextIdentityMismatch { engine_id: EngineId },
    InvalidExecutionPolicy { reason: ExecutionPolicyError },
    ExtensionModule { source: ExtensionModuleError },
}

#[derive(Debug)]
pub enum RuntimeReconfigureError {
    State { source: RuntimeStateError },
    Edit { source: RuntimeConfigError },
    ConcurrentReconfiguration {
        base: RuntimeEpoch,
        current: RuntimeEpoch,
    },
    EpochExhausted { current: RuntimeEpoch },
}

#[derive(Debug)]
pub enum PrepareError {
    RuntimeMismatch { expected: RuntimeId, actual: RuntimeId },
    StaleEpoch { prepared: RuntimeEpoch, current: RuntimeEpoch },
    InputSignature { source: InputSignatureError },
    ShapeGuard { source: ShapeGuardError },
    Specialization { source: SpecializationError },
    NoEligibleEngine { constraint: ProgramPlacementConstraint },
    Unsupported { reason: UnsupportedReason },
    DeterminismUnsupported { engine_id: EngineId },
    ProviderContract { source: ProviderContractError },
    PreparationCycle { key: PreparationKeySummary },
    NestedPreparationUnsupported {
        parent: PreparationKeySummary,
        requested: PreparationKeySummary,
    },
    CacheInFlightCapacityExceeded {
        in_flight: usize,
        queued_distinct_keys: usize,
    },
    CacheState { source: RuntimeStateError },
    Engine { source: Arc<dyn Error + Send + Sync> },
}
```

Caller misuse, foreign identities, wrong arity, invalid ranks/axes, context
mismatch, stale plans, and duplicate registration never panic. Poisoned locks
return `RuntimeStateError`; they do not fabricate empty/default state.

## Migration DAG

```text
P4-R0 rules/architecture reconciliation
 -> P4-A0 identities, inputs, options, specialization
 -> P4-A1 PreparedOperation, context, core request SPI
 -> P4-B0 immutable runtime/snapshot/reconfiguration
 -> P4-B1 transactional extension modules
 -> P4-C0 bounded cache/state machine
 -> P4-C1 binding-free PreparedProgram preparation
 -> P4-D0 tenferro-cpu registration adapter
 -> P4-D1 EagerRuntime snapshot bridge
 -> P4-E0 audits, docs, evidence
```

P4-R0 reconciles rules/architecture with Phase 3 ownership and CPU→runtime,
guarded by source tests. A0-C1 are runtime-owned/red-first; D0 adapts CPU, D1
moves configuration; E0 proves one private adapter and no stale text, later
artifact/family migration, or steady registry lookup.

## TDD tasks and acceptance tests

Required red-first coverage:

- **Identity/config:** fresh builds have distinct runtime/issuer IDs and epoch
  one; ordinals never fork/reuse; no snapshot-to-runtime build path compiles.
  Identical/conflict/replacement identity, freeze, failed/concurrent edits,
  overflow, old snapshot, stale ID, and poison contracts are covered.
- **SPI/preparation:** downstream compile fixtures construct every public
  provider type/use every accessor and run `assert_debug::<T>()` for all IDs,
  values, and manual-Debug containers. Traits are object-safe; plans are
  immutable, exact-sized, binding/resource-free. Constructors cover duplicate
  engine/storage, empty storage, and missing default. Valid placement is stable
  and unsatisfied placement returns `NoEligibleEngine` before lookup.
- **Extensions:** install/replace/remove is failure-atomic and slot order is
  deterministic; dispatch has no string/hash/lock/downcast lookup. Mocks
  receive exact op/config/placement/hardware/planning/options/input/
  specialization. No semantic API/family migration occurs.
- **Specialization:** validate every lattice field and malformed/nonmonotonic
  case, alignment pointer/metadata derivation, capped-min projection, every
  valid transition, address/sum overflow, and retry limit. Property chains for
  ranks 0-64 terminate. Shared initial projections independently re-key after
  widening; predecessor keys hold no plan.
- **Cache:** cover same-key single flight, independent top-level keys,
  fingerprint collisions, negative/transient failures, every root identity
  component, LRU/count/byte/oversize behavior, exact root-once accounting, and
  weak-sentinel release. Same-key recursion is `PreparationCycle`; nested
  distinct-key recursion is `NestedPreparationUnsupported` before capacity
  wait. Wait/refuse modes, bounded FIFO, panic/unwind/drop cleanup, clear
  semantics, gauges, counters, and stale-generation behavior are exact.
  Engine/extension cache owners freeze in stable order; stats attempts all
  owners, clear calls all despite failures, and ordered aggregate errors retain
  every failure without returning partial stats.
- **Regression/source:** `SemanticProgram` stays independent; `CompiledGraph`
  exposes only program/bindings/counts; one private adapter remains. Existing
  results/errors, executor entry, epoch refresh, extension path, and AD parity
  stay green.

## Documentation and performance gates

Document public APIs/Debug/errors and update P4-R0, provider/normative,
custom-operation/caching guides, and worklog. Dummy-dry-run, then predeclare
environment and small add/reduction/gather/`dot_general`. Steady state adds no
allocation/string/downcast/session/lock; placement is one epoch comparison and
cache/slot lookup O(1). Release evidence covers refresh, graph-size cold/hit,
concurrency, eviction, and bytes.

For a baseline median at most 10 µs, rejection requires a primary case at
least 50% slower and reproduction in a second complete paired A/B run.
`INCONCLUSIVE` is neither pass nor rejection: Phase completion remains open
until the full run succeeds or maintainers explicitly accept narrower evidence
scope. Run formatting, focused tests/doctests, ≥90% changed-file coverage,
audits, fast PR gate, and hosted matrices.

## Rollback and stop conditions

Stop for unapproved dependency/backend/family/AD change; loss of the one
binding-free private adapter; Phase 5 contract leakage; Phase 6 migration;
value/pointer/load specialization; failed convergence; global cache/steady
registry locking; repeated failed fix; or reproduced blocking regression.
Rollback retains the old snapshot/path and Phase 2/3 artifacts.

## Exit criteria

Exit requires accepted issue/plan; P4-R0 first; every contract/test above;
unchanged eager/graph/error/placement/executor behavior; no Phase 5/6
production leakage; one private adapter with Phase 5 deletion owner; and
passing docs, coverage, audits, performance evidence, and worklog review.
