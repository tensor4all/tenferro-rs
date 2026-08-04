use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tenferro_tensor::{Tensor, TensorValue};

use crate::graph::CompiledGraph;
use crate::program::FrozenProgram;

use super::cache::{PreparedPlanCacheLimits, RuntimeCacheSet};
use super::cache_owner::{FrozenCacheOwner, FrozenCacheOwnerKind};
use super::engine_registration::{CandidateRegistrationToken, EngineRegistrationState};
use super::execution;
#[cfg(test)]
use super::extension::ExtensionSlotFullForTest;
use super::extension::{
    bind_candidate_module, configure_module, freeze_extension_slots, BoundCandidateModuleRecord,
    CandidateModuleRecord, CandidateRegistrationIdentity, ExtensionEngineSnapshotView,
    ExtensionFamilyId, FrozenExtensionSlots,
};
use super::preparation::{PreparedEntryKey, PreparedProgram, PreparedProgramResult};
use super::schedule::EventDomainId;
use super::{
    CacheOwnerId, CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
    ExecutionPolicy, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
    FrozenTransferRegistry, HardwareClassId, InputSignature, PrepareOptions,
    ProviderDeviceIdentity, RegistrationIdentity, RegistrationKey, ResolvedTransferEndpoint,
    ResolvedTransferRoute, RuntimeCacheError, RuntimeCacheStats, RuntimeConfigError, RuntimeEpoch,
    RuntimeId, RuntimeReconfigureError, RuntimeStateError, StorageClass, TransferEndpoint,
    TransferProvider, TransferRoute,
};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REGISTRATION_ISSUER: AtomicU64 = AtomicU64::new(1);
const INITIAL_REGISTRATION_ORDINAL: NonZeroU64 = NonZeroU64::MIN;

#[derive(Clone, Debug)]
struct CandidateEngineRecord {
    registration: EngineRegistration,
    identity: CandidateRegistrationIdentity,
}

#[derive(Clone, Debug)]
struct BoundCandidateEngineRecord {
    registration: EngineRegistration,
    identity: RegistrationIdentity,
}

#[derive(Clone, Debug)]
enum CandidateTransferBinding {
    /// A route registered before its complete candidate can be validated.
    New,
    /// A route carried forward from a frozen snapshot.
    Preserved {
        source: ProviderDeviceIdentity,
        destination: ProviderDeviceIdentity,
    },
}

#[derive(Clone, Debug)]
struct CandidateTransferRecord {
    provider: Arc<dyn TransferProvider>,
    binding: CandidateTransferBinding,
}

struct BoundCandidateTransferRecord {
    provider: Arc<dyn TransferProvider>,
    source: ProviderDeviceIdentity,
    destination: ProviderDeviceIdentity,
}

#[derive(Clone, Debug)]
struct CandidateConfig {
    policy: ExecutionPolicy,
    engines: BTreeMap<EngineId, CandidateEngineRecord>,
    modules: BTreeMap<ExtensionModuleId, CandidateModuleRecord>,
    transfers: BTreeMap<TransferRoute, CandidateTransferRecord>,
}

struct BoundCandidateConfig {
    policy: ExecutionPolicy,
    engines: BTreeMap<EngineId, BoundCandidateEngineRecord>,
    modules: BTreeMap<ExtensionModuleId, BoundCandidateModuleRecord>,
    transfers: BTreeMap<TransferRoute, BoundCandidateTransferRecord>,
}

impl CandidateConfig {
    fn empty() -> Self {
        Self {
            policy: default_execution_policy(),
            engines: BTreeMap::new(),
            modules: BTreeMap::new(),
            transfers: BTreeMap::new(),
        }
    }

    fn from_snapshot(snapshot: &RuntimeConfigSnapshot) -> Result<Self, RuntimeConfigError> {
        let engines = snapshot
            .engines
            .iter()
            .map(|slot| {
                let registration = slot.to_registration()?;
                Ok((
                    registration.engine_id().clone(),
                    CandidateEngineRecord {
                        registration,
                        identity: CandidateRegistrationIdentity::Preserved(
                            slot.metadata().identity,
                        ),
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, RuntimeConfigError>>()?;
        Ok(Self {
            policy: snapshot.policy.clone(),
            engines,
            modules: snapshot.extensions.to_candidate_modules(),
            transfers: snapshot
                .transfers
                .iter()
                .map(|(resolved_route, provider)| {
                    (
                        TransferRoute::new(
                            resolved_route.source().logical().clone(),
                            resolved_route.destination().logical().clone(),
                        ),
                        CandidateTransferRecord {
                            provider: Arc::clone(provider),
                            binding: CandidateTransferBinding::Preserved {
                                source: resolved_route.source().provider_device_identity().clone(),
                                destination: resolved_route
                                    .destination()
                                    .provider_device_identity()
                                    .clone(),
                            },
                        },
                    )
                })
                .collect(),
        })
    }
}

#[derive(Clone, Debug)]
struct FrozenEngineMetadata {
    candidate_token: Arc<CandidateRegistrationToken>,
    identity: RegistrationIdentity,
    event_domain_id: EventDomainId,
}

#[derive(Clone)]
struct PreparationOnlyEngineSnapshot {
    metadata: FrozenEngineMetadata,
    binding: super::ProviderPreparationBinding,
}

#[derive(Clone, Debug)]
pub(super) struct ExecutableEngineSnapshot {
    metadata: FrozenEngineMetadata,
    binding: super::ProviderExecutableBinding,
}

#[derive(Clone)]
enum FrozenEngineSlot {
    PreparationOnly(Arc<PreparationOnlyEngineSnapshot>),
    Executable(Arc<ExecutableEngineSnapshot>),
}

impl FrozenEngineSlot {
    fn metadata(&self) -> &FrozenEngineMetadata {
        match self {
            Self::PreparationOnly(snapshot) => &snapshot.metadata,
            Self::Executable(snapshot) => &snapshot.metadata,
        }
    }

    fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.provider_device_identity(),
            Self::Executable(snapshot) => snapshot.binding.contract().provider_device_identity(),
        }
    }

    fn engine_id(&self) -> &EngineId {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.engine_id(),
            Self::Executable(snapshot) => snapshot.binding.engine_id(),
        }
    }

    fn hardware_class(&self) -> &HardwareClassId {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.hardware_class(),
            Self::Executable(snapshot) => snapshot.binding.hardware_class(),
        }
    }

    fn storage_classes(&self) -> &[StorageClass] {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.storage_classes(),
            Self::Executable(snapshot) => snapshot.binding.storage_classes(),
        }
    }

    fn default_storage_class(&self) -> &StorageClass {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.default_storage_class(),
            Self::Executable(snapshot) => snapshot.binding.default_storage_class(),
        }
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.context_identity(),
            Self::Executable(snapshot) => snapshot.binding.contract().context_identity(),
        }
    }

    fn capabilities(&self) -> &CoreCapabilityBundle {
        match self {
            Self::PreparationOnly(snapshot) => snapshot.binding.capabilities(),
            Self::Executable(snapshot) => snapshot.binding.contract().capabilities(),
        }
    }

    fn executable(&self) -> Option<&Arc<ExecutableEngineSnapshot>> {
        match self {
            Self::PreparationOnly(_) => None,
            Self::Executable(snapshot) => Some(snapshot),
        }
    }
}

impl ExecutableEngineSnapshot {
    pub(super) fn engine_id(&self) -> &EngineId {
        self.binding.engine_id()
    }

    pub(super) fn event_domain_id(&self) -> EventDomainId {
        self.metadata.event_domain_id
    }

    pub(super) fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        self.binding.contract().provider_device_identity()
    }

    #[cfg(test)]
    pub(super) fn context_identity(&self) -> ExecutionContextIdentity {
        self.binding.contract().context_identity()
    }

    pub(super) fn executor(&self) -> &Arc<dyn super::execution::ErasedTensorBackendExecutor> {
        self.binding.contract().executor()
    }

    pub(super) fn event_domain_driver(&self) -> &Arc<dyn super::EventDomainDriver> {
        self.binding.contract().event_domain_driver()
    }

    #[cfg(test)]
    pub(super) fn has_executor(&self) -> bool {
        true
    }

    #[cfg(test)]
    pub(super) fn has_event_domain_driver(&self) -> bool {
        true
    }

    pub(super) fn accepts_input_placement(
        &self,
        placement: &tenferro_tensor::Placement,
        storage_class: &StorageClass,
    ) -> bool {
        self.binding.storage_classes().contains(storage_class)
            && self
                .binding
                .contract()
                .accepts_input_placement(placement, storage_class)
    }

    pub(super) fn accepts_input_signature(
        &self,
        input: &super::InputSignatureEntry,
        storage_class: &StorageClass,
    ) -> bool {
        self.binding.storage_classes().contains(storage_class)
            && self
                .binding
                .contract()
                .accepts_input_signature(input, storage_class)
    }

    pub(super) fn accepts_runtime_input(
        &self,
        input: &tenferro_tensor::TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.binding.storage_classes().contains(storage_class)
            && self
                .binding
                .contract()
                .accepts_runtime_input(input, storage_class)
    }

    pub(super) fn owns_resident_tensor(
        &self,
        input: &tenferro_tensor::TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.binding.storage_classes().contains(storage_class)
            && self
                .binding
                .contract()
                .owns_resident_tensor(input, storage_class)
    }

    #[cfg(test)]
    pub(super) fn for_test(
        engine_id: EngineId,
        provider_device_identity: ProviderDeviceIdentity,
        event_domain_id: EventDomainId,
        storage_class: StorageClass,
    ) -> Arc<Self> {
        Self::for_test_with_driver(
            engine_id,
            provider_device_identity,
            event_domain_id,
            storage_class,
            Arc::new(super::ImmediateEventDomainDriver::new()),
        )
    }

    #[cfg(test)]
    pub(super) fn for_test_with_driver(
        engine_id: EngineId,
        provider_device_identity: ProviderDeviceIdentity,
        event_domain_id: EventDomainId,
        storage_class: StorageClass,
        event_domain_driver: Arc<dyn super::EventDomainDriver>,
    ) -> Arc<Self> {
        let ingress = super::InputIngressContract::new(
            super::InputPlacementContract::new(|_, _| true),
            super::InputSignatureContract::new(|_, _, _, _| true),
            super::RuntimeInputContract::new(|_, _| true),
            super::ResidentOutputContract::new(|_, _| true),
        );
        let contract = super::ExecutableEngineContract::new(
            provider_device_identity,
            CoreCapabilityBundle::default(),
            tenferro_cpu::CpuBackend::new(),
            event_domain_driver,
            ingress,
            None,
        );
        let binding = super::ProviderExecutableBinding::new(
            engine_id,
            HardwareClassId::new("tenferro.test.schedule.hardware").expect("test hardware class"),
            Arc::from(vec![storage_class.clone()]),
            storage_class,
            contract,
        )
        .expect("test executable binding");
        Arc::new(Self {
            metadata: FrozenEngineMetadata {
                candidate_token: Arc::new(CandidateRegistrationToken),
                identity: event_domain_id.registration_identity(),
                event_domain_id,
            },
            binding,
        })
    }
}

impl FrozenEngineSlot {
    fn to_registration(&self) -> Result<EngineRegistration, RuntimeConfigError> {
        let metadata = self.metadata();
        let registration = match self {
            Self::PreparationOnly(snapshot) => {
                EngineRegistration::from_state(EngineRegistrationState::PreparationOnly {
                    binding: snapshot.binding.clone(),
                })
            }
            Self::Executable(snapshot) => EngineRegistration::from_state(
                EngineRegistrationState::Executable(snapshot.binding.clone()),
            ),
        };
        Ok(registration.with_candidate_token(Arc::clone(&metadata.candidate_token)))
    }
}

impl fmt::Debug for FrozenEngineSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metadata = self.metadata();
        formatter
            .debug_struct("FrozenEngineSlot")
            .field("engine_id", self.engine_id())
            .field("registration_identity", &metadata.identity)
            .field("event_domain_id", &metadata.event_domain_id)
            .field("context_identity", &self.context_identity())
            .field("hardware_class", self.hardware_class())
            .field(
                "state",
                &match self {
                    Self::PreparationOnly(_) => "preparation-only",
                    Self::Executable(_) => "executable",
                },
            )
            .finish_non_exhaustive()
    }
}

/// Immutable runtime configuration snapshot.
///
/// A snapshot is readable after later reconfiguration because the runtime
/// publishes a new `Arc` instead of mutating existing slots.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::RuntimeConfigBuilder;
///
/// let runtime = RuntimeConfigBuilder::new().build()?;
/// let snapshot = runtime.snapshot()?;
/// assert_eq!(snapshot.engine_count(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RuntimeConfigSnapshot {
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    policy: ExecutionPolicy,
    engines: Arc<[FrozenEngineSlot]>,
    engine_indices: BTreeMap<EngineId, usize>,
    extensions: FrozenExtensionSlots,
    transfers: FrozenTransferRegistry,
    cache_owners: Arc<[FrozenCacheOwner]>,
}

impl RuntimeConfigSnapshot {
    /// Return the runtime identity that published this snapshot.
    pub fn runtime_id(&self) -> RuntimeId {
        self.runtime_id
    }

    /// Return the epoch for this immutable snapshot.
    pub fn epoch(&self) -> RuntimeEpoch {
        self.epoch
    }

    /// Return the execution policy captured by this snapshot.
    pub fn execution_policy(&self) -> &ExecutionPolicy {
        &self.policy
    }

    /// Return the number of direct engine slots.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Return the number of installed extension modules.
    pub fn extension_module_count(&self) -> usize {
        self.extensions.module_count()
    }

    /// Return the number of registered transfer providers.
    pub fn transfer_provider_count(&self) -> usize {
        self.transfers.len()
    }

    /// Return whether this snapshot contains an extension engine for a family.
    #[doc(hidden)]
    pub fn has_extension_family(&self, family_id: &'static str) -> bool {
        self.extensions.has_family(family_id)
    }

    /// Return an immutable view of a registered engine slot.
    pub fn engine(&self, id: &EngineId) -> Option<EngineSnapshotView<'_>> {
        self.engine_indices
            .get(id)
            .map(|&index| EngineSnapshotView {
                slot: &self.engines[index],
            })
    }

    #[cfg(test)]
    pub(crate) fn engine_ids_for_test(&self) -> impl Iterator<Item = &EngineId> {
        self.engines.iter().map(FrozenEngineSlot::engine_id)
    }

    #[cfg(test)]
    pub(crate) fn transfer_routes_for_test(&self) -> impl Iterator<Item = &ResolvedTransferRoute> {
        self.transfers.iter().map(|(route, _)| route)
    }

    pub(super) fn engine_views_for_preparation(
        &self,
    ) -> impl Iterator<Item = EngineSnapshotView<'_>> + '_ {
        self.engines.iter().map(|slot| EngineSnapshotView { slot })
    }

    pub(super) fn extension_slot_for_preparation(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<ExtensionEngineSnapshotView<'_>> {
        self.extensions.slot_for_preparation(family_id, engine_id)
    }

    pub(super) fn transfer_registry_for_preparation(&self) -> FrozenTransferRegistry {
        self.transfers.clone()
    }

    #[cfg(test)]
    pub(crate) fn extension_slots_for_test(
        &self,
    ) -> impl Iterator<
        Item = (
            &ExtensionModuleId,
            ExtensionFamilyId,
            &EngineId,
            RegistrationIdentity,
        ),
    > {
        self.extensions.slots_for_test()
    }

    #[cfg(test)]
    pub(crate) fn extension_slot_identity_for_test(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<RegistrationIdentity> {
        self.extensions.slot_identity_for_test(family_id, engine_id)
    }

    #[cfg(test)]
    pub(crate) fn extension_slot_full_for_test(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<ExtensionSlotFullForTest<'_>> {
        self.extensions.slot_full_for_test(family_id, engine_id)
    }

    #[cfg(test)]
    pub(super) fn cache_owners_for_test(&self) -> &[FrozenCacheOwner] {
        &self.cache_owners
    }

    pub(super) fn cache_owners_for_runtime(&self) -> &[FrozenCacheOwner] {
        &self.cache_owners
    }
}

impl fmt::Debug for RuntimeConfigSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeConfigSnapshot")
            .field("runtime_id", &self.runtime_id)
            .field("epoch", &self.epoch)
            .field("execution_policy", &self.policy)
            .field("engine_count", &self.engines.len())
            .field("extension_module_count", &self.extensions.module_count())
            .field("extension_engine_count", &self.extensions.engine_count())
            .field("transfer_provider_count", &self.transfers.len())
            .field("cache_owner_count", &self.cache_owners.len())
            .finish_non_exhaustive()
    }
}

struct RuntimeState {
    runtime_id: RuntimeId,
    issuer: NonZeroU64,
    next_registration_ordinal: AtomicU64,
    active: RwLock<Arc<RuntimeConfigSnapshot>>,
    published_epoch: AtomicU64,
    caches: RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
}

/// Runtime owner for immutable configuration snapshots.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::Runtime;
///
/// let runtime = Runtime::builder().build()?;
/// assert_eq!(runtime.snapshot()?.engine_count(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Runtime(Arc<RuntimeState>);

impl Runtime {
    /// Return a consuming runtime configuration builder.
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::new()
    }

    /// Return this runtime's opaque identity.
    pub fn id(&self) -> RuntimeId {
        self.0.runtime_id
    }

    /// Clone the current immutable runtime snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError::Poisoned`] when the active snapshot lock was
    /// poisoned by another thread.
    pub fn snapshot(&self) -> Result<Arc<RuntimeConfigSnapshot>, RuntimeStateError> {
        self.0
            .active
            .read()
            .map(|snapshot| Arc::clone(&snapshot))
            .map_err(|_| RuntimeStateError::Poisoned {
                lock: "runtime.active",
            })
    }

    /// Return the currently published runtime epoch without locking the active
    /// snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] only if runtime state invariants have been
    /// violated internally.
    pub fn epoch(&self) -> Result<RuntimeEpoch, RuntimeStateError> {
        match NonZeroU64::new(self.0.published_epoch.load(Ordering::Acquire)) {
            Some(value) => Ok(RuntimeEpoch::from_nonzero(value)),
            None => Err(RuntimeStateError::Poisoned {
                lock: "runtime.published_epoch",
            }),
        }
    }

    /// Return current prepared-plan cache limits.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::Runtime;
    ///
    /// let runtime = Runtime::builder().build()?;
    /// assert_eq!(runtime.prepared_cache_limits()?, Default::default());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] when the runtime-owned prepared cache state
    /// cannot be accessed.
    pub fn prepared_cache_limits(&self) -> Result<PreparedPlanCacheLimits, RuntimeStateError> {
        self.0.caches.prepared().limits()
    }

    /// Replace current prepared-plan cache limits and evict retained entries
    /// until the new limits are satisfied.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::num::NonZeroUsize;
    /// use tenferro_runtime::{PreparedPlanCacheLimits, Runtime};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// runtime.set_prepared_cache_limits(PreparedPlanCacheLimits {
    ///     max_entries: NonZeroUsize::new(1).unwrap(),
    ///     max_retained_bytes: NonZeroUsize::new(1024).unwrap(),
    ///     max_in_flight_entries: NonZeroUsize::new(1).unwrap(),
    ///     max_queued_distinct_keys: NonZeroUsize::new(1).unwrap(),
    /// })?;
    /// assert_eq!(runtime.prepared_cache_limits()?.max_entries.get(), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] when the runtime-owned prepared cache state
    /// cannot be accessed.
    pub fn set_prepared_cache_limits(
        &self,
        limits: PreparedPlanCacheLimits,
    ) -> Result<(), RuntimeStateError> {
        self.0.caches.prepared().set_limits(limits)
    }

    /// Clear the runtime-owned prepared-plan cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::Runtime;
    ///
    /// let runtime = Runtime::builder().build()?;
    /// runtime.clear_prepared_cache()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeStateError`] when the runtime-owned prepared cache state
    /// cannot be accessed.
    pub fn clear_prepared_cache(&self) -> Result<(), RuntimeStateError> {
        self.0.caches.prepared().clear()
    }

    /// Return aggregate cache statistics for the runtime and registered cache
    /// owners.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::Runtime;
    ///
    /// let runtime = Runtime::builder().build()?;
    /// assert_eq!(runtime.cache_stats()?.prepared_plans.entries, 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeCacheError`] when the runtime cache or a registered
    /// cache owner cannot report statistics.
    pub fn cache_stats(&self) -> Result<RuntimeCacheStats, RuntimeCacheError> {
        super::preparation::cache_stats(self, &self.0.caches)
    }

    /// Clear runtime-owned prepared plans and all registered engine/extension
    /// cache owners.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::Runtime;
    ///
    /// let runtime = Runtime::builder().build()?;
    /// runtime.clear_caches()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeCacheError`] when the runtime cache or a registered
    /// cache owner cannot be cleared.
    pub fn clear_caches(&self) -> Result<(), RuntimeCacheError> {
        super::preparation::clear_caches(self, &self.0.caches)
    }

    #[allow(
        dead_code,
        reason = "Phase 5 graph execution consumes crate-private prepared programs"
    )]
    pub(crate) fn prepare_for(
        &self,
        frozen: &FrozenProgram,
        signature: &InputSignature,
        options: &PrepareOptions,
    ) -> PreparedProgramResult<Arc<PreparedProgram>> {
        super::preparation::prepare_for(self, &self.0.caches, frozen, signature, options)
    }

    pub(crate) fn prepare_compiled_for(
        &self,
        program: &CompiledGraph,
        signature: &InputSignature,
        options: &PrepareOptions,
    ) -> PreparedProgramResult<Arc<PreparedProgram>> {
        super::preparation::prepare_compiled_for(self, &self.0.caches, program, signature, options)
    }

    /// Run a compiled graph synchronously with borrowed tensor inputs.
    ///
    /// The borrows remain valid until this call returns; this surface never
    /// detaches work. Asynchronous [`Self::submit`] accepts only the owning
    /// [`super::execution::ExecutionInputs`] package.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{Runtime, TracedTensor, GraphCompiler};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let program = GraphCompiler::new().compile(&x)?;
    /// let error = runtime.run_compiled(&program, &[]).unwrap_err();
    /// assert!(error.to_string().contains("no eligible engine"));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnboundPlaceholder`] when no explicit inputs are
    /// supplied and a semantic input has no bound default tensor.
    /// Returns [`crate::Error::GraphInputCountMismatch`],
    /// [`crate::Error::PlaceholderDtypeMismatch`],
    /// [`crate::Error::PlaceholderRankMismatch`],
    /// [`crate::Error::PlaceholderShapeMismatch`], or
    /// [`crate::Error::PlaceholderShapeBoundExceeded`] when ordered runtime
    /// inputs do not match the compiled graph metadata.
    /// Returns [`crate::Error::RuntimeState`] when runtime preparation, schedule
    /// validation, snapshot access, stale epoch checks, or execution-bridge
    /// resolution fails, including [`crate::PrepareError::NoInputIngress`]
    /// when no engine accepts an input's physical backend/allocation domain,
    /// [`crate::PrepareError::MissingTransferProvider`] when ingress cannot
    /// reach its first scheduled consumer, a runtime with no eligible engine,
    /// or no execution bridge for the prepared engine. Backend execution may
    /// also return concrete backend variants such as
    /// [`crate::Error::Unsupported`], [`crate::Error::Validation`], or
    /// [`crate::Error::Extension`].
    pub fn run_compiled(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> crate::Result<Vec<Tensor>> {
        super::execution::run_compiled(self, program, inputs)
    }

    /// Execute borrowed read-only inputs synchronously through retirement.
    ///
    /// Host/CPU providers may complete this call. Asynchronous device
    /// providers reject before admission and return the unchanged borrowed
    /// package through [`crate::ScopedSubmitRejected`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Unsupported`] when the selected asynchronous
    /// provider cannot execute borrowed inputs synchronously, or
    /// [`crate::ScopedSubmitRejected`] when pre-admission validation fails.
    /// Provider execution failures are reported as
    /// [`crate::runtime::execution::ScopedExecutionOutcome::RetiredFailed`].
    pub fn execute_scoped_read_only<'env>(
        &self,
        program: &CompiledGraph,
        inputs: super::execution::ScopedReadInputs<'env>,
    ) -> std::result::Result<
        super::execution::ScopedExecutionOutcome<'env>,
        super::execution::ScopedSubmitRejected<'env>,
    > {
        super::execution::execute_scoped_read_only(self, program, inputs)
    }

    /// Prepare a compiled graph for repeated execution with the same runtime.
    ///
    /// Preparation validates the supplied input metadata, selects a runtime
    /// engine, and caches the staged execution plan. Use [`Self::run_prepared`]
    /// for steady-state execution when the same compiled graph is run many
    /// times.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnboundPlaceholder`] when no explicit inputs are
    /// supplied and a semantic input has no bound default tensor.
    /// Returns [`crate::Error::GraphInputCountMismatch`],
    /// [`crate::Error::PlaceholderDtypeMismatch`],
    /// [`crate::Error::PlaceholderRankMismatch`],
    /// [`crate::Error::PlaceholderShapeMismatch`], or
    /// [`crate::Error::PlaceholderShapeBoundExceeded`] when ordered runtime
    /// inputs do not match the compiled graph metadata.
    /// Returns [`crate::Error::RuntimeState`] when runtime preparation, schedule
    /// validation, snapshot access, stale epoch checks, or execution-bridge
    /// resolution fails, including [`crate::PrepareError::NoInputIngress`]
    /// when no engine accepts an input's physical backend/allocation domain,
    /// [`crate::PrepareError::MissingTransferProvider`] when ingress cannot
    /// reach its first scheduled consumer, a runtime with no eligible engine,
    /// or no execution bridge for the prepared engine.
    pub fn prepare_compiled(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> crate::Result<super::execution::PreparedCompiledGraph> {
        super::execution::prepare_compiled(self, program, inputs)
    }

    /// Run a graph previously prepared by [`Self::prepare_compiled`].
    ///
    /// # Errors
    ///
    /// Returns metadata validation errors for incompatible inputs, a runtime
    /// state error with [`crate::InputIngressContractError`] as its typed source
    /// when an input's physical residency does not match the prepared ingress,
    /// or a runtime state error if the prepared handle belongs to a different
    /// runtime or a stale runtime epoch.
    pub fn run_prepared(
        &self,
        prepared: &super::execution::PreparedCompiledGraph,
        inputs: &[&Tensor],
    ) -> crate::Result<Vec<Tensor>> {
        super::execution::run_prepared(self, prepared, inputs)
    }

    /// Submit a compiled graph for asynchronous runtime-owned execution.
    ///
    /// Dropping the returned handle detaches the observer without blocking.
    /// Use [`super::execution::ExecutionHandle::wait`] to observe completion.
    ///
    /// # Errors
    ///
    /// Returns the same [`crate::PrepareError::InputSignature`],
    /// [`crate::PrepareError::Specialization`],
    /// [`crate::PrepareError::NoEligibleEngine`],
    /// [`crate::PrepareError::NoInputIngress`], and
    /// [`crate::PrepareError::MissingTransferProvider`] failures as
    /// [`Self::run_compiled`] before the worker is submitted. Returns a runtime
    /// state error with [`crate::SubmissionError`] as its typed source if the
    /// operating system rejects worker creation after admission.
    pub fn submit(
        &self,
        program: &CompiledGraph,
        inputs: super::execution::ExecutionInputs,
    ) -> std::result::Result<super::execution::ExecutionHandle, super::execution::SubmitError> {
        super::execution::submit(self, program, inputs)
    }

    /// Run a compiled graph and preserve lazy owned output views.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{Runtime, TracedTensor, GraphCompiler};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let program = GraphCompiler::new().compile(&x)?;
    /// let error = runtime.run_compiled_values(&program, &[]).unwrap_err();
    /// assert!(error.to_string().contains("no eligible engine"));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::UnboundPlaceholder`] when no explicit inputs are
    /// supplied and a semantic input has no bound default tensor.
    /// Returns [`crate::Error::GraphInputCountMismatch`],
    /// [`crate::Error::PlaceholderDtypeMismatch`],
    /// [`crate::Error::PlaceholderRankMismatch`],
    /// [`crate::Error::PlaceholderShapeMismatch`], or
    /// [`crate::Error::PlaceholderShapeBoundExceeded`] when ordered runtime
    /// inputs do not match the compiled graph metadata.
    /// Returns [`crate::Error::RuntimeState`] when runtime preparation, schedule
    /// validation, snapshot access, stale epoch checks, or execution-bridge
    /// resolution fails, including [`crate::PrepareError::NoInputIngress`]
    /// when no engine accepts an input's physical backend/allocation domain,
    /// [`crate::PrepareError::MissingTransferProvider`] when ingress cannot
    /// reach its first scheduled consumer, a runtime with no eligible engine,
    /// or no execution bridge for the prepared engine. Backend execution may
    /// also return concrete backend variants such as
    /// [`crate::Error::Unsupported`], [`crate::Error::Validation`], or
    /// [`crate::Error::Extension`].
    pub fn run_compiled_values(
        &self,
        program: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> crate::Result<Vec<TensorValue>> {
        super::execution::run_compiled_values(self, program, inputs)
    }

    /// Transactionally edit and publish runtime configuration.
    ///
    /// No user callback runs while the publication lock is held. If another
    /// writer publishes over the same base snapshot, this call returns
    /// [`RuntimeReconfigureError::ConcurrentReconfiguration`] and publishes
    /// nothing.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeReconfigureError`] when state access, edit validation,
    /// identity allocation, epoch advancement, or compare-and-publish fails.
    /// Invalid transfer endpoints are reported as the typed
    /// [`RuntimeConfigError::UnknownTransferEndpointEngine`] or
    /// [`RuntimeConfigError::UnsupportedTransferEndpointStorage`] source of
    /// [`RuntimeReconfigureError::Edit`].
    pub fn reconfigure(
        &self,
        edit: impl FnOnce(&mut RuntimeReconfiguration<'_>) -> Result<(), RuntimeConfigError>,
    ) -> Result<RuntimeEpoch, RuntimeReconfigureError> {
        let base = self
            .snapshot()
            .map_err(|source| RuntimeReconfigureError::State { source })?;
        let mut candidate = CandidateConfig::from_snapshot(&base)
            .map_err(|source| RuntimeReconfigureError::Edit { source })?;
        let mut changed = false;
        {
            let mut reconfiguration = RuntimeReconfiguration {
                candidate: &mut candidate,
                changed: &mut changed,
            };
            edit(&mut reconfiguration)
                .map_err(|source| RuntimeReconfigureError::Edit { source })?;
        }

        if !changed {
            return Ok(base.epoch());
        }
        let next_identity_ordinal = NonZeroU64::new(
            self.0.next_registration_ordinal.load(Ordering::SeqCst),
        )
        .ok_or(RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::IdentityExhausted,
        })?;
        let (bound_candidate, post_ordinal) =
            validate_candidate(candidate, self.0.issuer, next_identity_ordinal)
                .map_err(|source| RuntimeReconfigureError::Edit { source })?;

        let next_epoch =
            base.epoch()
                .checked_next()
                .ok_or(RuntimeReconfigureError::EpochExhausted {
                    current: base.epoch(),
                })?;

        let mut guard = self
            .0
            .active
            .write()
            .map_err(|_| RuntimeReconfigureError::State {
                source: RuntimeStateError::Poisoned {
                    lock: "runtime.active",
                },
            })?;
        if !Arc::ptr_eq(&*guard, &base) {
            return Err(RuntimeReconfigureError::ConcurrentReconfiguration {
                base: base.epoch(),
                current: guard.epoch(),
            });
        }

        let next_snapshot = Arc::new(
            freeze_candidate(self.0.runtime_id, next_epoch, bound_candidate)
                .map_err(|source| RuntimeReconfigureError::Edit { source })?,
        );

        self.0
            .next_registration_ordinal
            .store(post_ordinal.get(), Ordering::SeqCst);
        *guard = next_snapshot;
        self.0
            .published_epoch
            .store(next_epoch.get().get(), Ordering::Release);
        Ok(next_epoch)
    }

    #[cfg(test)]
    pub(crate) fn force_epoch_for_test(&self, epoch: RuntimeEpoch) {
        let mut guard = self.0.active.write().expect("test runtime lock");
        let mut replacement = (**guard).clone();
        replacement.epoch = epoch;
        *guard = Arc::new(replacement);
        self.0
            .published_epoch
            .store(epoch.get().get(), Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn force_next_registration_ordinal_for_test(&self, next: NonZeroU64) {
        self.0
            .next_registration_ordinal
            .store(next.get(), Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn poison_active_lock_for_test(&self) {
        let state = Arc::clone(&self.0);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = state.active.write().expect("test runtime lock");
            panic!("poison runtime.active for test");
        }));
    }
}

impl fmt::Debug for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Runtime")
            .field("runtime_id", &self.0.runtime_id)
            .field("published_epoch", &self.epoch().ok())
            .finish_non_exhaustive()
    }
}

/// Consuming builder for an immutable runtime configuration.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::RuntimeConfigBuilder;
///
/// let runtime = RuntimeConfigBuilder::new().build()?;
/// assert_eq!(runtime.snapshot()?.engine_count(), 0);
/// # Ok(())
/// # }
/// ```
pub struct RuntimeConfigBuilder {
    candidate: CandidateConfig,
}

impl RuntimeConfigBuilder {
    /// Create an empty runtime builder.
    pub fn new() -> Self {
        Self {
            candidate: CandidateConfig::empty(),
        }
    }

    /// Replace the execution policy in the candidate configuration.
    pub fn execution_policy(&mut self, value: ExecutionPolicy) -> &mut Self {
        self.candidate.policy = value;
        self
    }

    /// Register a new engine candidate.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::DuplicateEngine`] if a different candidate
    /// with the same engine ID is already present.
    pub fn register_engine(
        &mut self,
        value: EngineRegistration,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        register_engine_candidate(&mut self.candidate, value, &mut changed)?;
        Ok(self)
    }

    /// Explicitly replace an existing engine candidate.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingEngine`] if the engine ID is absent.
    pub fn replace_engine(
        &mut self,
        value: EngineRegistration,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        replace_engine_candidate(&mut self.candidate, value, &mut changed)?;
        Ok(self)
    }

    /// Remove an existing engine candidate.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingEngine`] if the engine ID is absent.
    pub fn remove_engine(&mut self, id: &EngineId) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        remove_engine_candidate(&mut self.candidate, id, &mut changed)?;
        Ok(self)
    }

    /// Install an extension module transaction.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ExtensionModule`] when module configuration
    /// fails or a distinct module already uses the same module ID.
    pub fn install_extension_module(
        &mut self,
        value: Arc<dyn ExtensionModule>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        install_extension_module_candidate(&mut self.candidate, value, &mut changed)?;
        Ok(self)
    }

    /// Register a transfer provider keyed by source and destination endpoints.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_runtime::{
    ///     assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
    ///     EngineRegistration, EngineRegistrationMetadata, Error, ExecutionContextIdentity,
    ///     HardwareClassId, PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity,
    ///     ProviderId, Runtime, StorageClass, TransferEndpoint, TransferProvider, TransferRequest,
    /// };
    ///
    /// #[derive(Debug)]
    /// struct ExampleProvider;
    ///
    /// impl TransferProvider for ExampleProvider {
    ///     fn transfer_blocking(
    ///         &self,
    ///         _request: TransferRequest<'_>,
    ///     ) -> tenferro_runtime::Result<tenferro_tensor::Tensor> {
    ///         Err(Error::Internal("the example does not execute a transfer".into()))
    ///     }
    /// }
    ///
    /// fn registration(
    ///     id: EngineId,
    ///     target: &str,
    ///     storage: &StorageClass,
    /// ) -> Result<EngineRegistration, tenferro_runtime::RuntimeConfigError> {
    ///     let metadata = EngineRegistrationMetadata::new(
    ///         id,
    ///         ProviderDeviceIdentity::new(ProviderId::new("example.provider")?, target)?,
    ///         HardwareClassId::new("example.hardware")?,
    ///         Arc::from([storage.clone()]),
    ///         storage.clone(),
    ///         CoreCapabilityBundle::default(),
    ///     );
    ///     assemble_preparation_only_engine_registration(
    ///         PreparationOnlyEngineRegistrationConfig::new(
    ///             metadata,
    ///             ExecutionContextIdentity::of::<()>(),
    ///         ),
    ///     )
    /// }
    ///
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let source_id = EngineId::new("example.engine.source")?;
    /// let destination_id = EngineId::new("example.engine.destination")?;
    /// let source = registration(
    ///     source_id.clone(),
    ///     "source-0",
    ///     &storage,
    /// )?;
    /// let destination = registration(
    ///     destination_id.clone(),
    ///     "destination-0",
    ///     &storage,
    /// )?;
    /// let mut builder = Runtime::builder();
    /// builder.register_engine(source)?;
    /// builder.register_engine(destination)?;
    /// builder.register_transfer_provider(
    ///     TransferEndpoint::new(source_id, storage.clone()),
    ///     TransferEndpoint::new(destination_id, storage),
    ///     Arc::new(ExampleProvider),
    /// )?;
    /// let runtime = builder.build()?;
    /// assert_eq!(runtime.snapshot()?.transfer_provider_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ConflictingRegistration`] if a different
    /// provider is already registered for the same endpoint pair. The complete
    /// endpoint pair is validated when [`Self::build`] freezes the candidate.
    pub fn register_transfer_provider(
        &mut self,
        source: TransferEndpoint,
        destination: TransferEndpoint,
        provider: Arc<dyn TransferProvider>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        register_transfer_provider_candidate(
            &mut self.candidate,
            source,
            destination,
            provider,
            &mut changed,
        )?;
        Ok(self)
    }

    /// Remove the transfer provider for an endpoint pair.
    ///
    /// This explicit removal is required before changing an engine's physical
    /// binding. The route can then be registered again against the replacement
    /// binding in the same candidate transaction.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingTransferProvider`] when the exact
    /// endpoint pair is not registered.
    pub fn remove_transfer_provider(
        &mut self,
        source: TransferEndpoint,
        destination: TransferEndpoint,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        remove_transfer_provider_candidate(&mut self.candidate, source, destination, &mut changed)?;
        Ok(self)
    }

    /// Replace an extension module transaction, installing it when absent.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ExtensionModule`] when module configuration
    /// fails.
    pub fn replace_extension_module(
        &mut self,
        value: Arc<dyn ExtensionModule>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        replace_extension_module_candidate(&mut self.candidate, value, &mut changed)?;
        Ok(self)
    }

    /// Remove an extension module candidate if present.
    ///
    /// # Errors
    ///
    /// This method currently has no failing absent-module path; it returns
    /// [`RuntimeConfigError`] only for future validated module removal failures.
    pub fn remove_extension_module(
        &mut self,
        id: &ExtensionModuleId,
    ) -> Result<&mut Self, RuntimeConfigError> {
        let mut changed = false;
        remove_extension_module_candidate(&mut self.candidate, id, &mut changed)?;
        Ok(self)
    }

    /// Build and publish the initial runtime snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::IdentityExhausted`] if runtime or
    /// registration identity allocation would wrap, or
    /// [`RuntimeConfigError::UnknownTransferEndpointEngine`] or
    /// [`RuntimeConfigError::UnsupportedTransferEndpointStorage`] if a
    /// registered transfer endpoint is invalid for the complete candidate.
    pub fn build(self) -> Result<Runtime, RuntimeConfigError> {
        let runtime_id = RuntimeId::from_nonzero(allocate_nonzero(&NEXT_RUNTIME_ID)?);
        let issuer = allocate_nonzero(&NEXT_REGISTRATION_ISSUER)?;
        let (bound_candidate, post_ordinal) =
            validate_candidate(self.candidate, issuer, INITIAL_REGISTRATION_ORDINAL)?;
        let epoch = RuntimeEpoch::one();
        let snapshot = Arc::new(freeze_candidate(runtime_id, epoch, bound_candidate)?);
        let state = RuntimeState {
            runtime_id,
            issuer,
            next_registration_ordinal: AtomicU64::new(post_ordinal.get()),
            active: RwLock::new(snapshot),
            published_epoch: AtomicU64::new(epoch.get().get()),
            caches: RuntimeCacheSet::new(PreparedPlanCacheLimits::default()),
        };
        Ok(Runtime(Arc::new(state)))
    }
}

impl Default for RuntimeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RuntimeConfigBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeConfigBuilder")
            .field("execution_policy", &self.candidate.policy)
            .field("engine_count", &self.candidate.engines.len())
            .field("extension_module_count", &self.candidate.modules.len())
            .field("transfer_provider_count", &self.candidate.transfers.len())
            .finish_non_exhaustive()
    }
}

/// Non-escapable reconfiguration edit view.
pub struct RuntimeReconfiguration<'a> {
    candidate: &'a mut CandidateConfig,
    changed: &'a mut bool,
}

impl RuntimeReconfiguration<'_> {
    /// Replace the candidate execution policy.
    pub fn execution_policy(&mut self, policy: ExecutionPolicy) -> &mut Self {
        if self.candidate.policy != policy {
            self.candidate.policy = policy;
            *self.changed = true;
        }
        self
    }

    /// Register a new engine in this reconfiguration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::DuplicateEngine`] if a different candidate
    /// with the same engine ID is already present.
    pub fn register_engine(
        &mut self,
        value: EngineRegistration,
    ) -> Result<&mut Self, RuntimeConfigError> {
        register_engine_candidate(self.candidate, value, self.changed)?;
        Ok(self)
    }

    /// Replace an existing engine in this reconfiguration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingEngine`] if the engine ID is absent.
    pub fn replace_engine(
        &mut self,
        value: EngineRegistration,
    ) -> Result<&mut Self, RuntimeConfigError> {
        replace_engine_candidate(self.candidate, value, self.changed)?;
        Ok(self)
    }

    /// Remove an existing engine in this reconfiguration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingEngine`] if the engine ID is absent.
    pub fn remove_engine(&mut self, id: &EngineId) -> Result<&mut Self, RuntimeConfigError> {
        remove_engine_candidate(self.candidate, id, self.changed)?;
        Ok(self)
    }

    /// Install an extension module in this reconfiguration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ExtensionModule`] when module configuration
    /// fails or a distinct module already uses the same module ID.
    pub fn install_extension_module(
        &mut self,
        value: Arc<dyn ExtensionModule>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        install_extension_module_candidate(self.candidate, value, self.changed)?;
        Ok(self)
    }

    /// Register a transfer provider in this reconfiguration by endpoint pair.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_runtime::{
    ///     assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
    ///     EngineRegistration, EngineRegistrationMetadata, Error, ExecutionContextIdentity,
    ///     HardwareClassId, PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity,
    ///     ProviderId, Runtime, StorageClass, TransferEndpoint, TransferProvider, TransferRequest,
    /// };
    ///
    /// #[derive(Debug)]
    /// struct ExampleProvider;
    ///
    /// impl TransferProvider for ExampleProvider {
    ///     fn transfer_blocking(
    ///         &self,
    ///         _request: TransferRequest<'_>,
    ///     ) -> tenferro_runtime::Result<tenferro_tensor::Tensor> {
    ///         Err(Error::Internal("the example does not execute a transfer".into()))
    ///     }
    /// }
    ///
    /// fn registration(
    ///     id: &str,
    ///     storage: &StorageClass,
    /// ) -> Result<EngineRegistration, tenferro_runtime::RuntimeConfigError> {
    ///     let metadata = EngineRegistrationMetadata::new(
    ///         EngineId::new(id)?,
    ///         ProviderDeviceIdentity::new(
    ///             ProviderId::new("example.provider")?,
    ///             format!("engine:{id}"),
    ///         )?,
    ///         HardwareClassId::new("example.hardware")?,
    ///         Arc::from([storage.clone()]),
    ///         storage.clone(),
    ///         CoreCapabilityBundle::default(),
    ///     );
    ///     Ok(assemble_preparation_only_engine_registration(
    ///         PreparationOnlyEngineRegistrationConfig::new(
    ///             metadata,
    ///             ExecutionContextIdentity::of::<()>(),
    ///         ),
    ///     )?)
    /// }
    ///
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let source_id = EngineId::new("example.engine.source")?;
    /// let destination_id = EngineId::new("example.engine.destination")?;
    /// let mut builder = Runtime::builder();
    /// builder.register_engine(registration(source_id.as_str(), &storage)?)?;
    /// builder.register_engine(registration(destination_id.as_str(), &storage)?)?;
    /// let runtime = builder.build()?;
    /// runtime.reconfigure(|edit| {
    ///     edit.register_transfer_provider(
    ///         TransferEndpoint::new(source_id, storage.clone()),
    ///         TransferEndpoint::new(destination_id, storage),
    ///         Arc::new(ExampleProvider),
    ///     )?;
    ///     Ok(())
    /// })?;
    /// assert_eq!(runtime.snapshot()?.transfer_provider_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ConflictingRegistration`] if a different
    /// provider is already registered for the same endpoint pair. The complete
    /// endpoint pair is validated before this reconfiguration is published.
    pub fn register_transfer_provider(
        &mut self,
        source: TransferEndpoint,
        destination: TransferEndpoint,
        provider: Arc<dyn TransferProvider>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        register_transfer_provider_candidate(
            self.candidate,
            source,
            destination,
            provider,
            self.changed,
        )?;
        Ok(self)
    }

    /// Remove the transfer provider for an endpoint pair in this
    /// reconfiguration.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::MissingTransferProvider`] when the exact
    /// endpoint pair is not registered.
    pub fn remove_transfer_provider(
        &mut self,
        source: TransferEndpoint,
        destination: TransferEndpoint,
    ) -> Result<&mut Self, RuntimeConfigError> {
        remove_transfer_provider_candidate(self.candidate, source, destination, self.changed)?;
        Ok(self)
    }

    /// Replace an extension module in this reconfiguration, installing when
    /// absent.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::ExtensionModule`] when module configuration
    /// fails.
    pub fn replace_extension_module(
        &mut self,
        value: Arc<dyn ExtensionModule>,
    ) -> Result<&mut Self, RuntimeConfigError> {
        replace_extension_module_candidate(self.candidate, value, self.changed)?;
        Ok(self)
    }

    /// Remove an extension module if present.
    ///
    /// # Errors
    ///
    /// This method currently has no failing absent-module path; it returns
    /// [`RuntimeConfigError`] only for future validated module removal failures.
    pub fn remove_extension_module(
        &mut self,
        id: &ExtensionModuleId,
    ) -> Result<&mut Self, RuntimeConfigError> {
        remove_extension_module_candidate(self.candidate, id, self.changed)?;
        Ok(self)
    }
}

impl fmt::Debug for RuntimeReconfiguration<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeReconfiguration")
            .field("engine_count", &self.candidate.engines.len())
            .field("extension_module_count", &self.candidate.modules.len())
            .field("transfer_provider_count", &self.candidate.transfers.len())
            .field("changed", &*self.changed)
            .finish_non_exhaustive()
    }
}

/// Borrowed immutable view of one engine slot in a runtime snapshot.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::EngineSnapshotView;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<EngineSnapshotView<'_>>();
/// ```
#[derive(Clone, Copy)]
pub struct EngineSnapshotView<'a> {
    slot: &'a FrozenEngineSlot,
}

impl<'a> EngineSnapshotView<'a> {
    /// Return the engine ID for this slot.
    pub fn engine_id(&self) -> &'a EngineId {
        self.slot.engine_id()
    }

    /// Return the immutable provider/device binding for this engine slot.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn inspect(view: tenferro_runtime::EngineSnapshotView<'_>) {
    /// let _ = view.provider_device_identity();
    /// # }
    /// ```
    pub fn provider_device_identity(&self) -> &'a super::ProviderDeviceIdentity {
        self.slot.provider_device_identity()
    }

    /// Return the runtime-local registration identity for this slot.
    pub fn registration_identity(&self) -> RegistrationIdentity {
        self.slot.metadata().identity
    }

    /// Return the execution-context identity required by this slot.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.slot.context_identity()
    }

    /// Return this engine's runtime event domain.
    pub fn event_domain_id(&self) -> EventDomainId {
        self.slot.metadata().event_domain_id
    }

    pub(super) fn executable_witness(&self) -> Option<&'a Arc<ExecutableEngineSnapshot>> {
        self.slot.executable()
    }

    /// Return the hardware class for this slot.
    pub fn hardware_class(&self) -> &'a HardwareClassId {
        self.slot.hardware_class()
    }

    /// Return direct core capability slots for this engine.
    pub fn capabilities(&self) -> &'a CoreCapabilityBundle {
        self.slot.capabilities()
    }

    pub(super) fn storage_classes(&self) -> &'a [StorageClass] {
        self.slot.storage_classes()
    }

    pub(super) fn default_storage_class(&self) -> &'a StorageClass {
        self.slot.default_storage_class()
    }

    pub(super) fn accepts_input_signature(
        &self,
        input: &super::InputSignatureEntry,
        storage_class: &StorageClass,
    ) -> bool {
        self.slot
            .executable()
            .is_some_and(|snapshot| snapshot.accepts_input_signature(input, storage_class))
    }

    #[cfg(test)]
    pub(crate) fn has_execution_engine_for_test(&self) -> bool {
        self.slot.executable().is_some()
    }
}

impl fmt::Debug for EngineSnapshotView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineSnapshotView")
            .field("engine_id", self.engine_id())
            .field("registration_identity", &self.registration_identity())
            .field("context_identity", &self.context_identity())
            .field("hardware_class", self.hardware_class())
            .field("capabilities", self.capabilities())
            .finish()
    }
}

fn default_execution_policy() -> ExecutionPolicy {
    ExecutionPolicy::new(super::Determinism::Fast, None, 0)
}

fn register_engine_candidate(
    candidate: &mut CandidateConfig,
    registration: EngineRegistration,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let engine_id = registration.engine_id().clone();
    match candidate.engines.get(&engine_id) {
        Some(existing) if existing.registration.candidate_identical(&registration) => Ok(()),
        Some(_) => Err(RuntimeConfigError::DuplicateEngine { engine_id }),
        None => {
            ensure_unique_provider_device_target(candidate, &registration)?;
            candidate.engines.insert(
                engine_id,
                CandidateEngineRecord {
                    registration,
                    identity: CandidateRegistrationIdentity::New,
                },
            );
            *changed = true;
            Ok(())
        }
    }
}

fn replace_engine_candidate(
    candidate: &mut CandidateConfig,
    registration: EngineRegistration,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let engine_id = registration.engine_id().clone();
    let Some(existing) = candidate.engines.get(&engine_id) else {
        return Err(RuntimeConfigError::MissingEngine { engine_id });
    };
    if existing.registration.candidate_identical(&registration) {
        return Ok(());
    }
    if existing.registration.provider_device_identity() != registration.provider_device_identity() {
        return Err(RuntimeConfigError::EngineTargetRebind {
            engine_id,
            current: existing.registration.provider_device_identity().clone(),
            replacement: registration.provider_device_identity().clone(),
        });
    }
    ensure_unique_provider_device_target_except(candidate, &registration, &engine_id)?;
    candidate.engines.insert(
        engine_id,
        CandidateEngineRecord {
            registration,
            identity: CandidateRegistrationIdentity::New,
        },
    );
    *changed = true;
    Ok(())
}

fn remove_engine_candidate(
    candidate: &mut CandidateConfig,
    id: &EngineId,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    match candidate.engines.remove(id) {
        Some(_) => {
            *changed = true;
            Ok(())
        }
        None => Err(RuntimeConfigError::MissingEngine {
            engine_id: id.clone(),
        }),
    }
}

fn install_extension_module_candidate(
    candidate: &mut CandidateConfig,
    module: Arc<dyn ExtensionModule>,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let module_id = module.module_id().clone();
    match candidate.modules.get(&module_id) {
        Some(existing) if existing.module_identical(&module) => Ok(()),
        Some(_) => Err(RuntimeConfigError::ExtensionModule {
            source: ExtensionModuleError::ConflictingModule { module_id },
        }),
        None => {
            let record = configure_module(module)
                .map_err(|source| RuntimeConfigError::ExtensionModule { source })?;
            candidate.modules.insert(module_id, record);
            *changed = true;
            Ok(())
        }
    }
}

fn replace_extension_module_candidate(
    candidate: &mut CandidateConfig,
    module: Arc<dyn ExtensionModule>,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let module_id = module.module_id().clone();
    match candidate.modules.get(&module_id) {
        Some(existing) if existing.module_identical(&module) => Ok(()),
        _ => {
            let record = configure_module(module)
                .map_err(|source| RuntimeConfigError::ExtensionModule { source })?;
            candidate.modules.insert(module_id, record);
            *changed = true;
            Ok(())
        }
    }
}

fn remove_extension_module_candidate(
    candidate: &mut CandidateConfig,
    id: &ExtensionModuleId,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    if candidate.modules.remove(id).is_some() {
        *changed = true;
    }
    Ok(())
}

fn register_transfer_provider_candidate(
    candidate: &mut CandidateConfig,
    source: TransferEndpoint,
    destination: TransferEndpoint,
    provider: Arc<dyn TransferProvider>,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let key = TransferRoute::new(source, destination);
    match candidate.transfers.get(&key) {
        Some(existing) if Arc::ptr_eq(&existing.provider, &provider) => Ok(()),
        Some(_) => Err(RuntimeConfigError::ConflictingRegistration {
            key: RegistrationKey::TransferProvider {
                source: key.source().clone(),
                destination: key.destination().clone(),
            },
        }),
        None => {
            candidate.transfers.insert(
                key,
                CandidateTransferRecord {
                    provider,
                    binding: CandidateTransferBinding::New,
                },
            );
            *changed = true;
            Ok(())
        }
    }
}

fn remove_transfer_provider_candidate(
    candidate: &mut CandidateConfig,
    source: TransferEndpoint,
    destination: TransferEndpoint,
    changed: &mut bool,
) -> Result<(), RuntimeConfigError> {
    let key = TransferRoute::new(source, destination);
    if candidate.transfers.remove(&key).is_none() {
        return Err(RuntimeConfigError::MissingTransferProvider {
            source_endpoint: key.source().clone(),
            destination: key.destination().clone(),
        });
    }
    *changed = true;
    Ok(())
}

fn ensure_unique_provider_device_target(
    candidate: &CandidateConfig,
    registration: &EngineRegistration,
) -> Result<(), RuntimeConfigError> {
    ensure_unique_provider_device_target_except(candidate, registration, registration.engine_id())
}

fn ensure_unique_provider_device_target_except(
    candidate: &CandidateConfig,
    registration: &EngineRegistration,
    ignored_engine_id: &EngineId,
) -> Result<(), RuntimeConfigError> {
    if let Some((first_engine_id, _)) = candidate.engines.iter().find(|(engine_id, record)| {
        *engine_id != ignored_engine_id
            && record.registration.provider_device_identity()
                == registration.provider_device_identity()
    }) {
        return Err(RuntimeConfigError::DuplicateProviderDeviceTarget {
            provider_device_identity: registration.provider_device_identity().clone(),
            first_engine_id: first_engine_id.clone(),
            duplicate_engine_id: registration.engine_id().clone(),
        });
    }
    Ok(())
}

fn validate_candidate(
    candidate: CandidateConfig,
    issuer: NonZeroU64,
    next_ordinal: NonZeroU64,
) -> Result<(BoundCandidateConfig, NonZeroU64), RuntimeConfigError> {
    let mut seen_targets = BTreeMap::<ProviderDeviceIdentity, EngineId>::new();
    for (engine_id, record) in &candidate.engines {
        if let Some(first_engine_id) = seen_targets.insert(
            record.registration.provider_device_identity().clone(),
            engine_id.clone(),
        ) {
            return Err(RuntimeConfigError::DuplicateProviderDeviceTarget {
                provider_device_identity: record.registration.provider_device_identity().clone(),
                first_engine_id,
                duplicate_engine_id: engine_id.clone(),
            });
        }
    }

    let mut bound_transfers = BTreeMap::new();
    for (route, record) in &candidate.transfers {
        let source_binding = validate_transfer_endpoint(&candidate, route.source())?;
        let destination_binding = validate_transfer_endpoint(&candidate, route.destination())?;
        let preserved = match &record.binding {
            CandidateTransferBinding::New => None,
            CandidateTransferBinding::Preserved {
                source,
                destination,
            } => Some((source, destination)),
        };
        if let Some((registered_source, registered_destination)) = preserved {
            if registered_source != &source_binding {
                return Err(RuntimeConfigError::StaleTransferRoute {
                    source_endpoint: route.source().clone(),
                    destination: route.destination().clone(),
                    endpoint: route.source().clone(),
                    registered: Box::new(registered_source.clone()),
                    current: Box::new(source_binding.clone()),
                });
            }
            if registered_destination != &destination_binding {
                return Err(RuntimeConfigError::StaleTransferRoute {
                    source_endpoint: route.source().clone(),
                    destination: route.destination().clone(),
                    endpoint: route.destination().clone(),
                    registered: Box::new(registered_destination.clone()),
                    current: Box::new(destination_binding.clone()),
                });
            }
        }
        bound_transfers.insert(
            route.clone(),
            BoundCandidateTransferRecord {
                provider: Arc::clone(&record.provider),
                source: source_binding,
                destination: destination_binding,
            },
        );
    }
    let mut seen = BTreeMap::<(ExtensionFamilyId, EngineId), ExtensionModuleId>::new();
    for (module_id, module) in &candidate.modules {
        for family_engine in module.engines.keys() {
            if seen
                .insert(
                    (family_engine.0, family_engine.1.clone()),
                    module_id.clone(),
                )
                .is_some()
            {
                return Err(RuntimeConfigError::ConflictingRegistration {
                    key: RegistrationKey::ExtensionEngine {
                        family: family_engine.0,
                        engine: family_engine.1.clone(),
                    },
                });
            }
        }
    }

    let CandidateConfig {
        policy,
        engines,
        modules,
        transfers: _,
    } = candidate;
    let mut allocator = RegistrationIdentityAllocator::new(issuer, next_ordinal);
    let engines = engines
        .into_iter()
        .map(|(engine_id, record)| {
            let identity = match record.identity {
                CandidateRegistrationIdentity::New => allocator.allocate()?,
                CandidateRegistrationIdentity::Preserved(identity) => identity,
            };
            Ok((
                engine_id,
                BoundCandidateEngineRecord {
                    registration: record.registration,
                    identity,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, RuntimeConfigError>>()?;
    let modules = modules
        .into_iter()
        .map(|(module_id, module)| {
            let mut allocate = || allocator.allocate();
            Ok((module_id, bind_candidate_module(module, &mut allocate)?))
        })
        .collect::<Result<BTreeMap<_, _>, RuntimeConfigError>>()?;
    Ok((
        BoundCandidateConfig {
            policy,
            engines,
            modules,
            transfers: bound_transfers,
        },
        allocator.next_ordinal(),
    ))
}

fn validate_transfer_endpoint(
    candidate: &CandidateConfig,
    endpoint: &TransferEndpoint,
) -> Result<ProviderDeviceIdentity, RuntimeConfigError> {
    let Some(engine) = candidate.engines.get(endpoint.engine_id()) else {
        return Err(RuntimeConfigError::UnknownTransferEndpointEngine {
            endpoint: endpoint.clone(),
        });
    };
    if !engine
        .registration
        .storage_classes()
        .contains(endpoint.storage_class())
    {
        return Err(RuntimeConfigError::UnsupportedTransferEndpointStorage {
            endpoint: endpoint.clone(),
        });
    }
    Ok(engine.registration.provider_device_identity().clone())
}

struct RegistrationIdentityAllocator {
    issuer: NonZeroU64,
    next: NonZeroU64,
}

impl RegistrationIdentityAllocator {
    fn new(issuer: NonZeroU64, next: NonZeroU64) -> Self {
        Self { issuer, next }
    }

    fn allocate(&mut self) -> Result<RegistrationIdentity, RuntimeConfigError> {
        let identity = RegistrationIdentity::new(self.issuer, self.next);
        let next = self
            .next
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or(RuntimeConfigError::IdentityExhausted)?;
        self.next = next;
        Ok(identity)
    }

    fn next_ordinal(&self) -> NonZeroU64 {
        self.next
    }
}

fn freeze_candidate(
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    candidate: BoundCandidateConfig,
) -> Result<RuntimeConfigSnapshot, RuntimeConfigError> {
    let mut engines = Vec::with_capacity(candidate.engines.len());
    let mut engine_indices = BTreeMap::new();
    let mut engine_locations = BTreeMap::new();
    let mut cache_owners = Vec::new();
    for (index, (engine_id, record)) in candidate.engines.into_iter().enumerate() {
        let BoundCandidateEngineRecord {
            registration,
            identity,
        } = record;
        let event_domain_id = EventDomainId::new(runtime_id, epoch, identity);
        let (state, candidate_token) = registration.into_state_and_token();
        let provider_device_identity = state.provider_device_identity().clone();
        let metadata = FrozenEngineMetadata {
            candidate_token,
            identity,
            event_domain_id,
        };
        let frozen = match state {
            EngineRegistrationState::PreparationOnly { binding } => {
                FrozenEngineSlot::PreparationOnly(Arc::new(PreparationOnlyEngineSnapshot {
                    metadata,
                    binding,
                }))
            }
            EngineRegistrationState::Executable(binding) => {
                if let Some(owner) = binding.contract().cache_owner().cloned() {
                    cache_owners.push(FrozenCacheOwner {
                        id: engine_cache_owner_id(&engine_id),
                        kind: FrozenCacheOwnerKind::Engine,
                        owner,
                    });
                }
                cache_owners.push(FrozenCacheOwner {
                    id: engine_extension_cache_owner_id(&engine_id),
                    kind: FrozenCacheOwnerKind::Extension,
                    owner: execution::extension_cache_owner(binding.contract().executor().clone()),
                });
                FrozenEngineSlot::Executable(Arc::new(ExecutableEngineSnapshot {
                    metadata,
                    binding,
                }))
            }
        };
        engine_locations.insert(
            engine_id.clone(),
            (provider_device_identity, event_domain_id),
        );
        engine_indices.insert(engine_id, index);
        engines.push(frozen);
    }
    let extensions = freeze_extension_slots(candidate.modules)?;
    for (id, owner) in extensions.cache_owner_records() {
        cache_owners.push(FrozenCacheOwner {
            id,
            kind: FrozenCacheOwnerKind::Extension,
            owner,
        });
    }
    let mut transfers = BTreeMap::new();
    for (route, record) in candidate.transfers {
        let BoundCandidateTransferRecord {
            provider,
            source: source_binding,
            destination: destination_binding,
        } = record;
        let (_, source_event_domain_id) = bound_engine_location(&engine_locations, route.source())?;
        let (_, destination_event_domain_id) =
            bound_engine_location(&engine_locations, route.destination())?;
        let resolved_route = ResolvedTransferRoute::new(
            ResolvedTransferEndpoint::new(
                route.source().clone(),
                source_binding,
                *source_event_domain_id,
            ),
            ResolvedTransferEndpoint::new(
                route.destination().clone(),
                destination_binding,
                *destination_event_domain_id,
            ),
        );
        transfers.insert(resolved_route, provider);
    }
    Ok(RuntimeConfigSnapshot {
        runtime_id,
        epoch,
        policy: candidate.policy,
        engines: engines.into(),
        engine_indices,
        extensions,
        transfers: FrozenTransferRegistry::new(transfers),
        cache_owners: cache_owners.into(),
    })
}

fn bound_engine_location<'a>(
    locations: &'a BTreeMap<EngineId, (ProviderDeviceIdentity, EventDomainId)>,
    endpoint: &TransferEndpoint,
) -> Result<&'a (ProviderDeviceIdentity, EventDomainId), RuntimeConfigError> {
    locations
        .get(endpoint.engine_id())
        .ok_or_else(|| RuntimeConfigError::BoundCandidateInvariant {
            endpoint: endpoint.clone(),
        })
}

fn engine_cache_owner_id(engine_id: &EngineId) -> CacheOwnerId {
    let id = engine_id.as_str();
    CacheOwnerId::from_canonical_owner_id(Arc::<str>::from(format!("engine[{}]:{id}", id.len())))
}

fn engine_extension_cache_owner_id(engine_id: &EngineId) -> CacheOwnerId {
    let id = engine_id.as_str();
    CacheOwnerId::from_canonical_owner_id(Arc::<str>::from(format!(
        "extension-executor[{}]:{id}",
        id.len()
    )))
}

fn allocate_nonzero(counter: &AtomicU64) -> Result<NonZeroU64, RuntimeConfigError> {
    let value = counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |next| {
            next.checked_add(1)
        })
        .map_err(|_| RuntimeConfigError::IdentityExhausted)?;
    NonZeroU64::new(value).ok_or(RuntimeConfigError::IdentityExhausted)
}

#[cfg(test)]
mod freeze_tests {
    use crate::{ProviderId, TransferRequest};

    use super::*;

    #[derive(Debug)]
    struct FreezeTestContext;

    #[derive(Debug)]
    struct FreezeTestProvider;

    impl TransferProvider for FreezeTestProvider {
        fn transfer_blocking(
            &self,
            _request: TransferRequest<'_>,
        ) -> crate::Result<tenferro_tensor::Tensor> {
            Err(crate::Error::Internal("freeze test provider".into()))
        }
    }

    fn registration(
        engine_id: &str,
        target: &str,
    ) -> Result<EngineRegistration, RuntimeConfigError> {
        let engine_id = EngineId::new(engine_id).map_err(RuntimeConfigError::from)?;
        let storage =
            StorageClass::new("tenferro.test.freeze.storage").map_err(RuntimeConfigError::from)?;
        Ok(EngineRegistration::preparation_only(
            super::super::ProviderPreparationBinding::new(
                engine_id,
                ProviderDeviceIdentity::new(
                    ProviderId::new("tenferro.test.freeze.provider")
                        .map_err(RuntimeConfigError::from)?,
                    target,
                )
                .map_err(RuntimeConfigError::from)?,
                ExecutionContextIdentity::of::<FreezeTestContext>(),
                HardwareClassId::new("tenferro.test.freeze.hardware")
                    .map_err(RuntimeConfigError::from)?,
                Arc::from(vec![storage.clone()]),
                storage,
                CoreCapabilityBundle::default(),
            )?,
        ))
    }

    fn candidate(binding: CandidateTransferBinding) -> Result<CandidateConfig, RuntimeConfigError> {
        let source_id =
            EngineId::new("tenferro.test.freeze.source").map_err(RuntimeConfigError::from)?;
        let destination_id =
            EngineId::new("tenferro.test.freeze.destination").map_err(RuntimeConfigError::from)?;
        let storage =
            StorageClass::new("tenferro.test.freeze.storage").map_err(RuntimeConfigError::from)?;
        let source_endpoint = TransferEndpoint::new(source_id.clone(), storage.clone());
        let destination_endpoint = TransferEndpoint::new(destination_id.clone(), storage);
        let mut candidate = CandidateConfig::empty();
        let mut changed = false;
        register_engine_candidate(
            &mut candidate,
            registration(source_id.as_str(), "freeze-source")?,
            &mut changed,
        )?;
        register_engine_candidate(
            &mut candidate,
            registration(destination_id.as_str(), "freeze-destination")?,
            &mut changed,
        )?;
        register_transfer_provider_candidate(
            &mut candidate,
            source_endpoint.clone(),
            destination_endpoint.clone(),
            Arc::new(FreezeTestProvider),
            &mut changed,
        )?;
        candidate
            .transfers
            .get_mut(&TransferRoute::new(source_endpoint, destination_endpoint))
            .expect("registered route")
            .binding = binding;
        Ok(candidate)
    }

    #[test]
    fn validation_owns_stale_route_rejection_and_bound_freeze_is_total() {
        let wrong_source = ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.freeze.provider").unwrap(),
            "different-source",
        )
        .unwrap();
        let preserved = CandidateTransferBinding::Preserved {
            source: wrong_source,
            destination: ProviderDeviceIdentity::new(
                ProviderId::new("tenferro.test.freeze.provider").unwrap(),
                "freeze-destination",
            )
            .unwrap(),
        };
        let result = validate_candidate(
            candidate(preserved).unwrap(),
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(1).unwrap(),
        );
        let error = match result {
            Ok(_) => panic!("candidate validation must reject stale preserved bindings"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            RuntimeConfigError::StaleTransferRoute { .. }
        ));

        let (bound, _) = validate_candidate(
            candidate(CandidateTransferBinding::New).unwrap(),
            NonZeroU64::new(1).unwrap(),
            NonZeroU64::new(1).unwrap(),
        )
        .expect("validation must produce a complete bound candidate");
        freeze_candidate(
            RuntimeId::from_nonzero(NonZeroU64::new(1).unwrap()),
            RuntimeEpoch::one(),
            bound,
        )
        .expect("a bound candidate must freeze without semantic route revalidation");
    }

    #[test]
    fn frozen_engine_slots_are_arc_sized() {
        let slot_size = std::mem::size_of::<FrozenEngineSlot>();
        let arc_size = std::mem::size_of::<Arc<()>>();

        assert!(
            slot_size <= 2 * arc_size,
            "frozen engine slots should keep immutable snapshot payloads behind Arc: slot_size={slot_size}, arc_size={arc_size}",
        );
    }
}
