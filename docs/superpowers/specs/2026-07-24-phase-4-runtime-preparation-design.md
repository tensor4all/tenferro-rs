# Phase 4: Immutable Runtime Snapshots and Preparation Substrate

## Status and provenance

**Revised Phase 4 design, child of
[#1451](https://github.com/tensor4all/tenferro-rs/issues/1451). The
implementation has not landed. Exact-commit acceptance must be recorded on
#1451 before implementation begins.**

Authority: reconciled rules, umbrella #1433/`e2bfdde4`, Phase 3 #1449 v2.4,
then Phase 4 issue #1451 and this design/plan/worklog.
[The A0 proposal comments
5068331740](https://github.com/tensor4all/tenferro-rs/issues/1451#issuecomment-5068331740),
[5068343174](https://github.com/tensor4all/tenferro-rs/issues/1451#issuecomment-5068343174),
and [conditional review
5068396967](https://github.com/tensor4all/tenferro-rs/issues/1451#issuecomment-5068396967)
are integrated here; they do not by themselves record exact-commit
acceptance. [Maintainer scope
direction](https://github.com/tensor4all/tenferro-rs/issues/1433#issuecomment-5066159995)
limits work to Phases 4-6 and moves audit to Phase 6. Baselines: `b5a3dcd2`,
`bb98ee28`, and one private forward adapter.

## Goal

Add immutable runtime snapshots/epochs, transactional reconfiguration, direct
core and resolved extension slots, object-safe plan/context-identity contracts
with immutable execution semantics, finite specialization, and bounded
single-flight/negative/cycle-safe preparation caches. Preserve current
execution and the one private Phase 3 adapter. Phase 4 prepares plans; it does
not schedule or execute a new graph.

## Non-goals

No public `PreparedGraph`; schedule/transfers/collectives/buffers/events/
resource admission/common executor or adapter deletion (Phase 5); extension
lowering/family migration/native N-ary einsum/changing-shape gate (Phase 6);
deferred Phases 7-9; implicit placement transfer; new crate/facade/backend/
feature/external dependency; or AD semantic change.

## Chosen boundary

Provider preparation is public; the aggregate remains crate-private to avoid
freezing Phase 5 execution contracts. `PreparedOperation` has immutable
execution-visible semantics, is identity/specialization-bound, and has no
execution method; a bounded derived-plan cache may mutate only behind the
owner/accounting contract below.

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
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum IdentityKind {
    Engine,
    HardwareClass,
    StorageClass,
    LayoutClass,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("malformed {kind:?} identifier")]
pub struct IdentityError {
    kind: IdentityKind,
}

impl IdentityError {
    pub fn kind(&self) -> IdentityKind {
        self.kind
    }

    fn malformed(kind: IdentityKind) -> Self {
        Self { kind }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeId(NonZeroU64);

impl RuntimeId {
    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeEpoch(NonZeroU64);

impl RuntimeEpoch {
    pub(crate) const fn one() -> Self {
        Self(NonZeroU64::MIN)
    }

    pub(crate) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }

    pub(crate) fn checked_next(self) -> Option<Self> {
        self.0
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
    }
}

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

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegistrationIdentity {
    issuer: NonZeroU64,
    ordinal: NonZeroU64,
}

impl RegistrationIdentity {
    pub(crate) const fn new(
        issuer: NonZeroU64,
        ordinal: NonZeroU64,
    ) -> Self {
        Self { issuer, ordinal }
    }

    pub fn ordinal(self) -> NonZeroU64 {
        self.ordinal
    }
}

impl std::fmt::Debug for RegistrationIdentity {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("RegistrationIdentity")
            .field("ordinal", &self.ordinal)
            .finish()
    }
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
    pub fn builder() -> RuntimeConfigBuilder;
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
    pub fn new() -> Self;
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

`IdentityError` has no source and does not retain or print the rejected string.
This bounds error retention and prevents arbitrary caller data from appearing
through `Debug`, logs, or reports. All four A0 string IDs use one
crate-private validator with the exact grammar:

```text
identifier = component "." component ("." component)*
component  = alnum | alnum *(alnum | "-" | "_") alnum
alnum      = "a".."z" | "0".."9"
```

The complete string is nonempty ASCII, has at least two dot-separated
components, and each component begins and ends with lowercase ASCII
alphanumeric. Interior bytes may additionally be `-` or `_`. Uppercase,
whitespace, non-ASCII, empty components, leading/trailing/consecutive dots,
and all other punctuation are rejected. The validator returns the original
`Arc<str>` on success.

Raw strings enter only the ID constructors, which return `IdentityError`
directly. Runtime builders and registrations accept already-valid ID values
and do not remap identity syntax failures into `RuntimeConfigError`.

The extension-module node later adds `IdentityKind::ExtensionModule`; the
cache-owner node later adds `IdentityKind::CacheOwner`. Each owning node adds
its constructor and validator call in the same commit. `#[non_exhaustive]`
supports this serial extension. Phase 4 promises no numeric discriminant,
serde representation, or stable wire encoding; a later wire format must use
explicit stable string tags.

A0 contains no global atomic, issuer, allocator, exhaustion simulation, or
public raw-integer constructor. It defines only the nonzero opaque
representations and crate-private construction used by focused tests. The
manual `RegistrationIdentity` `Debug` exposes its useful ordinal but not its
private issuer.

`Runtime::builder()` is exactly `RuntimeConfigBuilder::new()`. Both are
infallible and allocate no identity. Consuming
`RuntimeConfigBuilder::build(self)` is the sole owner of configuration
validation, fresh nonzero `RuntimeId` and registration issuer allocation,
epoch-one creation, and checked exhaustion. Exhaustion returns
`RuntimeConfigError::IdentityExhausted`; a failed build returns no `Runtime`
and never wraps or reuses an identity.

After construction, `RuntimeState` alone owns its checked next registration
ordinal. A reconfiguration candidate cannot escape and has no `build`;
unchanged records retain identities, while new/replaced records receive
identities only inside successful publication. Failure allocates none.
Ordinals never wrap/recycle, and no public path can fork a runtime ID or
issuer. `RuntimeEpoch::checked_next() == None` maps to
`RuntimeReconfigureError::EpochExhausted`.

Before runtime identity issuance, engine equality is `(EngineId,
Arc::ptr_eq(candidate_token))`; modules/extension engines use their stable keys
plus `Arc::ptr_eq`, and planning configs use `payload_eq`. Identical repeats
are no-ops that preserve identity and skip reconfiguration; same-key unequal
values conflict; explicit replacement receives a new registration identity.

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    ) -> Self;
    pub fn determinism(&self) -> Determinism;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> u64;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
pub struct PrepareOptionsKey {
    resolved_placement: ResolvedProgramPlacement,
    hard_workspace_limit_bytes: Option<usize>,
    planning_seed: u64,
}

impl PrepareOptionsKey {
    pub fn resolved_placement(&self) -> &ResolvedProgramPlacement;
    pub fn hard_workspace_limit_bytes(&self) -> Option<usize>;
    pub fn planning_seed(&self) -> u64;
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
    ) -> Self;
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
that no workspace bytes may be planned. Every accepted determinism value,
workspace limit, and seed is valid, so `ExecutionPolicy::new` and
`ResolvedPlanningConfig::resolve` are infallible and Phase 4 defines no
`ExecutionPolicyError`.

For the per-call workspace field, `None` means inherit the policy. `Some(0)`
overrides the policy with a zero byte limit, and `Some(n)` overrides it with
`n`. Calling `with_hard_workspace_limit_bytes(None)` resets the option to
inheritance. Phase 4 intentionally has no per-call override from a finite
policy limit to unlimited; unlimited resolution requires both the policy and
per-call option to be `None`. Resolution is exactly:

```rust
let hard_workspace_limit_bytes = options
    .hard_workspace_limit_bytes()
    .or(policy.hard_workspace_limit_bytes());
```

`PrepareOptions` remains the raw request-behavior object.
`PrepareOptionsKey` is normalized and created only after placement and
planning resolution. It stores the resolved placement, resolved hard
workspace, and resolved seed, with the displayed accessors; it never stores
the raw placement constraint or raw overrides. Raw `PrepareOptions` does not
implement `Hash`. `cache_in_flight` controls how the current caller reacts to
an in-flight entry and is not semantic plan identity: cache_in_flight appears
in no cache, hash, or identity key. The runtime consumes raw request behavior;
provider preparation receives only resolved planning/placement and the
normalized key, never raw `PrepareOptions`.

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
    pub fn dot_general(&self) -> Option<&Arc<dyn DotGeneralPreparation>>;
    pub fn layout(&self) -> Option<&Arc<dyn LayoutRuntime>>;
}

pub struct CoreCapabilityBundleBuilder { /* one optional direct slot each */ }

impl CoreCapabilityBundleBuilder {
    pub fn new() -> Self;
    pub fn elementwise(
        &mut self,
        capability: Arc<dyn ElementwiseRuntime>,
    ) -> &mut Self;
    pub fn reduction(
        &mut self,
        capability: Arc<dyn ReductionRuntime>,
    ) -> &mut Self;
    pub fn indexing(
        &mut self,
        capability: Arc<dyn IndexingRuntime>,
    ) -> &mut Self;
    pub fn dot_general(
        &mut self,
        capability: Arc<dyn DotGeneralPreparation>,
    ) -> &mut Self;
    pub fn layout(
        &mut self,
        capability: Arc<dyn LayoutRuntime>,
    ) -> &mut Self;
    pub fn build(self) -> CoreCapabilityBundle;
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

pub trait DotGeneralPreparation: Debug + Send + Sync + 'static {
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
Each capability setter replaces the previous value for that slot. The builder
has no invalid state, so setters and `build` are infallible.

`DotGeneralPreparation` is deliberately preparation-only. It avoids collision
with the existing CPU composite execution vocabulary. Phase 5 must migrate and
resolve this preparation-only name when it defines the execution-facing common
graph/provider boundary; Phase 4 does not pre-accept that later API.

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

The traits are object-safe. A prepared operation's execution-visible semantics
are immutable. It contains no mutable scratch, input/output tensor, runtime
handle, resource lease, scheduler state, or public tensor wrapper.

A prepared operation may reference an engine-owned derived-plan cache or own an
interior-mutable bounded derived-plan cache. Such a cache has a bounded
default, deterministic semantics, and stats/clear through its registered
`RuntimeCacheOwner` path. An engine that permits prepared-operation-owned
derived caches registers one owner that aggregates and clears those caches;
the prepared operation does not register a new snapshot owner after freeze.
The owning engine or prepared operation uses owner-specific bounded defaults
and configuration. An engine-owned cache lives with the registered engine
owner. A prepared-operation-owned cache lives no longer than its operation
handle, and the aggregate owner tracks it without extending that lifetime.
Neither kind may contain scratch buffers or tensors.
Cache mutation may affect reuse and performance only; it cannot change the
operation's binding, specialization, chosen algorithm semantics, determinism,
workspace contract, or results. `retained_bytes` reports logical heap payload
owned exclusively by the prepared operation. Derived-cache retained bytes are
charged exactly once by the registered owner and excluded from the prepared
operation's `retained_bytes`; shared engine state, cache metadata, caller
tensors, and inline object size are also excluded. Phase 4 adds no generic
derived-cache limits or capacity API. Phase 6 owns the concrete einsum cache
limits/type; any other owning phase must define its provider-specific bound
before enabling such a cache. Phase 4 does not instantiate a derived cache or
add a synthetic cache type/test seam solely to prove this future permission.

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

Phase 6 native einsum composes GEMM and layout plans through direct
engine-internal provider calls. It never re-enters `Runtime::prepare_for`
during preparation. If a later lowering needs distinct preparation
dependencies, that phase must first accept an explicit dependency model; it
must not create recursive cache entry through the Phase 4 API.

## Runtime snapshot and reconfiguration semantics

Fresh `build` consumes its builder, validates all records and policy, then
allocates a new nonzero `RuntimeId`, creates the unique issuer, assigns
registration identities, freezes slots, creates epoch one/caches, and
publishes once. Failed validation consumes no identity or registration
ordinal.

Reconfiguration snapshots the active `Arc`, edits/validates a non-escapable
candidate off-lock, checks the next epoch, then publishes under a short lock
only if the base pointer is still current; otherwise it returns
`ConcurrentReconfiguration`. Failure allocates no identities and publishes
nothing. Successful publication assigns identities to new/replaced records,
advances once, and clears retained cache state; running old-generation work
may finish but cannot reinsert. Preparation pins its snapshot, ID, and epoch.
This full clear on successful reconfiguration is deliberate. Epoch-scoped
retention is a deferred optimization and is not part of the Phase 4 cache
contract.

A placement-bound eager context caches runtime, placement, epoch, engine slot,
and capability slots. One epoch comparison uses them or refreshes from one new
snapshot. It owns no idle resource/session/registry lock.
`EagerRuntime::on_cpu` and managed/external executor behavior remain unchanged;
no second runtime identity/backend mutex is added.

## Input signatures and finite specialization

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InputSignatureError {
    #[error(
        "input signature shape rank {rank} does not match stride count {stride_count}"
    )]
    ShapeStrideRankMismatch {
        rank: usize,
        stride_count: usize,
    },

    #[error("input signature alignment class {alignment_log2} is invalid")]
    InvalidAlignmentClass {
        alignment_log2: u8,
    },

    #[error("failed to read metadata for input {input}: {source}")]
    TensorMetadata {
        input: usize,
        #[source]
        source: tenferro_tensor::Error,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum RankRequirement {
    ConcreteAxis {
        axis: u32,
    },

    ExactStrides,
}

impl std::fmt::Display for RankRequirement {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::ConcreteAxis { axis } => {
                write!(formatter, "concrete axis {axis}")
            }
            Self::ExactStrides => formatter.write_str("exact strides"),
        }
    }
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InputSpecializationRequirementsError {
    #[error(
        "specialization axis {axis} is duplicated at positions \
         {first_index} and {duplicate_index}"
    )]
    DuplicateAxis {
        axis: u32,
        first_index: usize,
        duplicate_index: usize,
    },

    #[error("rank specialization is required by {reason}")]
    RankRequired {
        reason: RankRequirement,
    },

    #[error("specialization alignment class {alignment_log2} is invalid")]
    InvalidAlignmentClass {
        alignment_log2: u8,
    },
}

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SpecializationError {
    #[error("expected {expected} inputs, got {actual}")]
    WrongInputCount {
        expected: usize,
        actual: usize,
    },

    #[error("input {input} axis {axis} is outside rank {rank}")]
    AxisOutOfRange {
        input: usize,
        axis: u32,
        rank: usize,
    },

    #[error(
        "input {input} requires alignment class {required_alignment_log2}, \
         but alignment metadata is unavailable"
    )]
    AlignmentUnavailable {
        input: usize,
        required_alignment_log2: u8,
    },

    #[error("specialization did not strictly widen")]
    NonMonotonicSpecialization,

    #[error("specialization retry-edge sum overflowed")]
    ProjectionOverflow,

    #[error("specialization retry limit {limit} exhausted after {attempts} attempts")]
    RetryLimit {
        attempts: usize,
        limit: usize,
    },
}

/// A0-complete declaration. Later nodes extend this non-exhaustive enum only
/// after their referenced types exist.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PrepareError {
    #[error("invalid input signature: {source}")]
    InputSignature {
        #[source]
        source: InputSignatureError,
    },

    #[error("invalid specialization: {source}")]
    Specialization {
        #[source]
        source: SpecializationError,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignature { entries: Arc<[InputSignatureEntry]> }

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignatureEntry {
    dtype: DType,
    shape: ShapeVec,
    placement: Placement,
    layout_class: LayoutClass,
    strides: StrideVec,
    alignment_log2: Option<u8>,
}

impl InputSignature {
    pub fn from_reads(inputs: &[TensorRead<'_>]) -> Result<Self, PrepareError>;
    pub fn new(
        entries: impl Into<Arc<[InputSignatureEntry]>>,
    ) -> Self;
    pub fn entries(&self) -> &[InputSignatureEntry];
}

impl InputSignatureEntry {
    pub fn new(
        dtype: DType,
        shape: ShapeVec,
        placement: Placement,
        layout_class: LayoutClass,
        strides: StrideVec,
        alignment_log2: Option<u8>,
    ) -> Result<Self, InputSignatureError>;
    pub fn dtype(&self) -> DType;
    pub fn shape(&self) -> &[usize];
    pub fn placement(&self) -> &Placement;
    pub fn layout_class(&self) -> &LayoutClass;
    pub fn strides(&self) -> &[isize];
    pub fn alignment_log2(&self) -> Option<u8>;
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
    ) -> Result<
        InputSpecializationRequirements,
        InputSpecializationRequirementsError,
    >;
}

impl SpecializationRequirements {
    pub fn polymorphic(input_count: usize) -> Self;
    pub fn new(
        inputs: impl Into<Box<[InputSpecializationRequirements]>>,
    ) -> Self;
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

fn checked_retry_edge_sum(
    edges: impl IntoIterator<Item = usize>,
) -> Result<usize, SpecializationError> {
    edges.into_iter().try_fold(0usize, |sum, edge| {
        sum.checked_add(edge)
            .ok_or(SpecializationError::ProjectionOverflow)
    })
}
```

`InputSignatureEntry::new` validates that shape and stride ranks match and
that `Some(a)` has `a < usize::BITS`. `None` is valid and means no alignment
guarantee is known. `Some(0)` is a known one-byte guarantee and is distinct
from `None`. `InputSignature::new` accepts only already-valid entries. Entries
have no cross-entry invariant, so aggregate construction is infallible.

`InputSignature::from_reads` retains no tensor, buffer, pointer, or value. It
copies dtype, shape, strides, placement, layout class, and the derived
alignment class. Backend-native storage has no current alignment metadata and
therefore stores `None`.

For host storage, `from_reads` dispatches by dtype and uses the already
validated host backing slice and logical offset exposed by the typed view to
obtain the actual logical pointer. A nonempty host input stores
`Some(min(address.trailing_zeros(), align_of::<T>().trailing_zeros(),
usize::BITS - 1) as u8)`. Empty host storage uses `align_of::<T>()` and stores
its bounded class without reading a logical element. View construction already
validates bounds and offset arithmetic, so Phase 4 adds no hypothetical
declaration validator, address-overflow error, or synthetic-only production
helper.

This pointer-derived alignment class is a bounded, intentional exception to
the “no pointer-derived keys” rule: keys retain only the derived alignment
class, never a pointer or address. Mutating or dropping the tensor after
signature construction cannot change the signature or retain its allocation.

Metadata-access failures are
`PrepareError::InputSignature { source:
InputSignatureError::TensorMetadata { input, source } }` and preserve the
concrete `tenferro_tensor::Error`. A manually constructed out-of-lattice
`Some(a)` returns `InputSignatureError::InvalidAlignmentClass`. Unknown
backend alignment is not an error during signature construction.

The per-input specialization builder is deliberately index-free and is the
only layer that validates duplicate axes, rank implications, and requested
alignment classes. It validates once, in this order:

1. The first duplicate concrete axis, reporting its first and duplicate
   positions.
2. The first concrete axis that requires disabled rank specialization, using
   `RankRequirement::ConcreteAxis`.
3. Exact-strides specialization with disabled rank specialization, using
   `RankRequirement::ExactStrides`.
4. An alignment class outside the finite lattice.

`SpecializationRequirements::new` accepts already-valid per-input
requirements and is infallible. Projection trusts those construction
invariants. It checks only signature-dependent facts: aggregate/signature
input count, concrete axes against actual ranks, and required alignment
against available alignment metadata. Those failures are
`WrongInputCount`, `AxisOutOfRange`, and `AlignmentUnavailable`, wrapped in
`PrepareError::Specialization`.

Alignment projection is exact:

| Requirement | Actual signature | Projection |
|---|---|---|
| `None` | `None` | `None` |
| `None` | `Some(a)` | `None` |
| `Some(k)` | `None` | `SpecializationError::AlignmentUnavailable` |
| `Some(k)` | `Some(a)` | `Some(min(k, a))` |

The finite order is
`None < Some(0) < ... < Some(usize::BITS - 1)`. The remaining alignment-edge
count is `usize::BITS` from `None`, and
`usize::BITS - 1 - usize::from(a)` from `Some(a)`. Other fields widen
componentwise through their displayed finite orders. `NeedsSpecialization`
must strictly increase and never lower or make incomparable any component.
The whole-signature retry bound uses `checked_retry_edge_sum`; overflow is
`ProjectionOverflow`, and exhaustion is `RetryLimit`.

Projections and normalized options keys are runtime-created and exclude
values, pointers, free memory, scheduler state, diagnostic strings, and
private provider keys.

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

struct CanonicalCacheOwnerId(Arc<str>);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheOwnerId(Arc<str>);

impl CacheOwnerId {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError>;
    pub fn as_str(&self) -> &str;
    fn from_canonical_owner_id(value: CanonicalCacheOwnerId) -> Self {
        Self(value.0)
    }
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
These accepted canonical strings intentionally contain `[` and `:` and do not
weaken the public namespaced grammar. `cache_owner.rs` owns the only raw
construction path:

```rust
fn canonical_engine_owner_id(
    engine_id: &EngineId,
) -> CanonicalCacheOwnerId {
    let id = engine_id.as_str();
    CanonicalCacheOwnerId(
        Arc::<str>::from(format!("engine[{}]:{id}", id.len())),
    )
}

fn canonical_extension_owner_id(
    module_id: &ExtensionModuleId,
    local: &CacheOwnerId,
) -> CanonicalCacheOwnerId {
    let module = module_id.as_str();
    let local = local.as_str();
    CanonicalCacheOwnerId(Arc::<str>::from(format!(
        "extension[{}]:{module}[{}]:{local}",
        module.len(),
        local.len(),
    )))
}

pub(crate) fn engine_cache_owner_id(
    engine_id: &EngineId,
) -> CacheOwnerId {
    CacheOwnerId::from_canonical_owner_id(
        canonical_engine_owner_id(engine_id),
    )
}

pub(crate) fn extension_cache_owner_id(
    module_id: &ExtensionModuleId,
    local: &CacheOwnerId,
) -> CacheOwnerId {
    CacheOwnerId::from_canonical_owner_id(
        canonical_extension_owner_id(module_id, local),
    )
}
```

`CanonicalCacheOwnerId`, its tuple field, both raw formatters, and the bypass
constructor are module-private. Only the two high-level functions returning a
finished `CacheOwnerId` are crate-private. Sibling snapshot modules cannot
receive raw tokens or arbitrary `Arc<str>` and call only those finished-ID
factories. Lengths are UTF-8 byte lengths; validated constituent IDs are
ASCII.

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

`PreparedRootKey` does not retain `PrepareOptionsKey`; its existing
`resolved_placement` and `resolved_planning` fields are the canonical cache
identity and avoid duplicated resolved state. The runtime-created
`PrepareOptionsKey` passed to provider preparation contains the same normalized
resolved placement, hard workspace limit, and seed, and contains no raw
placement constraint or override. If a later design retains that normalized
key beside the existing resolved fields, it must first define an explicit
consistency check. The separate `cache_in_flight` request behavior appears
nowhere in `PreparedRootKey`, `PreparedEntryKey`, any hash input, or any other
cache/identity key, so caller wait/refuse preference cannot fragment
semantically identical plans.

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
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlacementConstraintError {
    #[error(
        "engine {engine_id:?} is duplicated at positions \
         {first_index} and {duplicate_index}"
    )]
    DuplicateEngine {
        engine_id: EngineId,
        first_index: usize,
        duplicate_index: usize,
    },
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeConfigError {
    #[error("runtime identity space is exhausted")]
    IdentityExhausted,

    #[error("engine {engine_id:?} is already registered")]
    DuplicateEngine { engine_id: EngineId },

    #[error("registration conflicts for {key:?}")]
    ConflictingRegistration { key: RegistrationKey },

    #[error("engine {engine_id:?} is not registered")]
    MissingEngine { engine_id: EngineId },

    #[error("engine {engine_id:?} has no storage classes")]
    EmptyStorageClasses { engine_id: EngineId },

    #[error(
        "engine {engine_id:?} storage class {storage_class:?} is duplicated at \
         positions {first_index} and {duplicate_index}"
    )]
    DuplicateStorageClass {
        engine_id: EngineId,
        storage_class: StorageClass,
        first_index: usize,
        duplicate_index: usize,
    },

    #[error(
        "engine {engine_id:?} default storage class \
         {default_storage_class:?} is not registered"
    )]
    DefaultStorageClassNotListed {
        engine_id: EngineId,
        default_storage_class: StorageClass,
    },

    #[error("extension module configuration failed: {source}")]
    ExtensionModule {
        #[source]
        source: ExtensionModuleError,
    },
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeReconfigureError {
    #[error("runtime state unavailable: {source}")]
    State {
        #[source]
        source: RuntimeStateError,
    },

    #[error("runtime reconfiguration edit failed: {source}")]
    Edit {
        #[source]
        source: RuntimeConfigError,
    },

    #[error(
        "concurrent reconfiguration changed epoch from {base:?} to {current:?}"
    )]
    ConcurrentReconfiguration {
        base: RuntimeEpoch,
        current: RuntimeEpoch,
    },

    #[error("runtime epoch exhausted at {current:?}")]
    EpochExhausted { current: RuntimeEpoch },
}
```

The A0 `PrepareError` declaration appears before its first use in the input
signature block and contains exactly `InputSignature` and `Specialization`.
Later owning nodes extend that same non-exhaustive enum. The final post-C1
declaration, replacing the A0 milestone declaration rather than defining a
second Rust type, is:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PrepareError {
    #[error("prepared runtime {actual:?} does not match expected {expected:?}")]
    RuntimeMismatch { expected: RuntimeId, actual: RuntimeId },

    #[error("prepared epoch {prepared:?} is stale relative to {current:?}")]
    StaleEpoch { prepared: RuntimeEpoch, current: RuntimeEpoch },

    #[error("invalid input signature: {source}")]
    InputSignature {
        #[source]
        source: InputSignatureError,
    },

    #[error("shape guard failed: {source}")]
    ShapeGuard {
        #[source]
        source: ShapeGuardError,
    },

    #[error("invalid specialization: {source}")]
    Specialization {
        #[source]
        source: SpecializationError,
    },

    #[error("no engine satisfies placement {constraint:?}")]
    NoEligibleEngine { constraint: ProgramPlacementConstraint },

    #[error("operation is unsupported: {reason}")]
    Unsupported { reason: UnsupportedReason },

    #[error("engine {engine_id:?} does not support requested determinism")]
    DeterminismUnsupported { engine_id: EngineId },

    #[error("provider contract violation: {source}")]
    ProviderContract {
        #[source]
        source: ProviderContractError,
    },

    #[error("preparation cycle at {key:?}")]
    PreparationCycle { key: PreparationKeySummary },

    #[error("nested preparation from {parent:?} to {requested:?} is unsupported")]
    NestedPreparationUnsupported {
        parent: PreparationKeySummary,
        requested: PreparationKeySummary,
    },

    #[error(
        "preparation cache is at capacity: {in_flight} in flight and \
         {queued_distinct_keys} distinct keys queued"
    )]
    CacheInFlightCapacityExceeded {
        in_flight: usize,
        queued_distinct_keys: usize,
    },

    #[error("preparation cache state unavailable: {source}")]
    CacheState {
        #[source]
        source: RuntimeStateError,
    },

    #[error("engine preparation failed: {source}")]
    Engine {
        #[source]
        source: Arc<dyn Error + Send + Sync>,
    },
}
```

Caller misuse, foreign identities, wrong arity, invalid ranks/axes, context
mismatch, stale plans, and duplicate registration never panic. Poisoned locks
return `RuntimeStateError`; they do not fabricate empty/default state.

## Standalone A0 boundary

The A0 implementation must compile without later-node types. At its commit:

- `IdentityKind` contains exactly `Engine`, `HardwareClass`, `StorageClass`,
  and `LayoutClass`;
- `PrepareError` contains exactly the source-bearing `InputSignature` and
  `Specialization` arms;
- `IdentityError`, `InputSignatureError`, `RankRequirement`,
  `InputSpecializationRequirementsError`, and `SpecializationError` are
  defined before use;
- A0 contains no identity allocator, exhaustion fixture, runtime builder,
  registration key, capability/provider request, snapshot, cache owner,
  scheduling, execution, resource, buffer, transfer, collective, or event
  contract; and
- A0 public API exposes no raw numeric identity constructor/accessor, tensor or
  value owner, pointer/address, or manual cache key.

B0 owns the first `RuntimeConfigError` and `RuntimeReconfigureError`
definitions. Both builder constructors remain plain values and contain no
identity allocation; only consuming `RuntimeConfigBuilder::build` returns
`Result<Runtime, RuntimeConfigError>` and invokes checked allocation. The
cache-owner node later adds only its two high-level crate-private finished-ID
factories; sibling modules never gain access to the raw canonical token or
formatter.

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
  one; builder creation touches no allocator; consuming build alone reports
  exhaustion; ordinals never fork/reuse; `RegistrationIdentity` diagnostics
  never expose the issuer; no snapshot-to-runtime build path compiles. The
  exact ASCII grammar is shared by all four A0 IDs and invalid input is absent
  from `IdentityError` display, debug, retention, and source. Identical/
  conflict/replacement identity, freeze, failed/concurrent edits, overflow,
  old snapshot, stale ID, and poison contracts are covered.
- **Policy/key normalization:** every determinism, `u64` seed, and `None`/
  zero/maximum workspace combination constructs and resolves infallibly.
  `None` inherits and `Some(0)` overrides. Runtime-created option/planning keys
  compare every displayed resolved field; raw constraints/overrides and
  `cache_in_flight` are absent from cache identity. `PreparedRootKey` uses its
  existing resolved fields rather than duplicating `PrepareOptionsKey`.
- **SPI/preparation:** downstream compile fixtures construct every public
  provider type/use every accessor and run `assert_debug::<T>()` for all IDs,
  values, and manual-Debug containers. Traits are object-safe; plans have
  immutable execution-visible semantics, exact retained-byte accounting, and
  no call bindings or resource leases. Constructors cover duplicate engine/
  storage, empty storage, and missing default. Valid placement is stable and
  unsatisfied placement returns `NoEligibleEngine` before lookup.
  Core-capability setters prove last-write replacement and infallible build.
  Existing owner fixtures verify `RuntimeCacheOwner` aggregation/clear and,
  where naturally covered, that `PreparedOperation::retained_bytes` excludes
  bytes already reported by an owner. Phase 4 adds no concrete derived cache
  or synthetic future-cache fixture. The owning Phase 6 einsum implementation
  must test its concrete bounded default/configuration, stats/clear, lifetime,
  deterministic semantics, no tensor/scratch retention, and exactly-once byte
  accounting.
- **Extensions:** install/replace/remove is failure-atomic and slot order is
  deterministic; dispatch has no string/hash/lock/downcast lookup. Mocks
  receive exact op/config/placement/hardware/planning/normalized-options-key/
  input/specialization. Canonical cache-owner construction is reachable only
  through the two finished-ID factories. No semantic API/family migration
  occurs.
- **Specialization:** validate every reachable lattice and nonmonotonic case,
  host logical-pointer class derivation, backend-unknown versus known
  one-byte alignment, capped-min projection, `AlignmentUnavailable`, every
  valid transition, retry-sum overflow, and retry limit. Builder tests assert
  the direct index-free duplicate-position, rank-reason, and alignment errors.
  Projection tests cover only reachable signature-dependent wrong-count,
  axis-out-of-range, and alignment-unavailable errors. Property chains for
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
value/load specialization or raw pointer/address identity outside the bounded
derived alignment-class exception; failed convergence; global cache/steady
registry locking; repeated failed fix; or reproduced blocking regression.
Rollback retains the old snapshot/path and Phase 2/3 artifacts.

## Exit criteria

Exit requires accepted issue/plan; P4-R0 first; every contract/test above;
unchanged eager/graph/error/placement/executor behavior; no Phase 5/6
production leakage; one private adapter with Phase 5 deletion owner; and
passing docs, coverage, audits, performance evidence, and worklog review.
