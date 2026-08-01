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
use super::execution;
#[cfg(test)]
use super::extension::ExtensionSlotFullForTest;
use super::extension::{
    configure_module, freeze_extension_slots, CandidateModuleRecord, ExtensionEngineSnapshotView,
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

#[derive(Clone, Debug)]
struct CandidateEngineRecord {
    registration: EngineRegistration,
    identity: Option<RegistrationIdentity>,
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
    /// A new or preserved route bound by complete-candidate validation.
    Bound {
        source: ProviderDeviceIdentity,
        destination: ProviderDeviceIdentity,
    },
}

#[derive(Clone, Debug)]
struct CandidateTransferRecord {
    provider: Arc<dyn TransferProvider>,
    binding: CandidateTransferBinding,
}

#[derive(Clone, Debug)]
struct CandidateConfig {
    policy: ExecutionPolicy,
    engines: BTreeMap<EngineId, CandidateEngineRecord>,
    modules: BTreeMap<ExtensionModuleId, CandidateModuleRecord>,
    transfers: BTreeMap<TransferRoute, CandidateTransferRecord>,
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

    fn from_snapshot(snapshot: &RuntimeConfigSnapshot) -> Self {
        let engines = snapshot
            .engines
            .iter()
            .map(|slot| {
                (
                    slot.registration.engine_id().clone(),
                    CandidateEngineRecord {
                        registration: slot.registration.clone(),
                        identity: Some(slot.identity),
                    },
                )
            })
            .collect();
        Self {
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
        }
    }
}

#[derive(Clone)]
struct FrozenEngineSlot {
    registration: EngineRegistration,
    identity: RegistrationIdentity,
    event_domain_id: EventDomainId,
}

impl fmt::Debug for FrozenEngineSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenEngineSlot")
            .field("engine_id", self.registration.engine_id())
            .field("registration_identity", &self.identity)
            .field("event_domain_id", &self.event_domain_id)
            .field("context_identity", &self.registration.context_identity())
            .field("hardware_class", self.registration.hardware_class())
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
        self.engines
            .iter()
            .map(|slot| slot.registration.engine_id())
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

    pub(super) fn transfer_registry_for_execution(&self) -> FrozenTransferRegistry {
        self.transfers.clone()
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

    /// Run a compiled graph through runtime-owned prepared execution.
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
        inputs: &[&Tensor],
    ) -> crate::Result<super::execution::ExecutionHandle> {
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
        let mut candidate = CandidateConfig::from_snapshot(&base);
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
        validate_candidate(&mut candidate)
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

        let missing_identities = candidate
            .engines
            .values()
            .filter(|record| record.identity.is_none())
            .count()
            + candidate
                .modules
                .values()
                .flat_map(|module| module.engines.values())
                .filter(|record| record.identity.is_none())
                .count();
        let (new_identities, next_ordinal) = self
            .plan_registration_identities(missing_identities)
            .map_err(|source| RuntimeReconfigureError::Edit { source })?;
        assign_new_identities(&mut candidate, new_identities);
        let next_snapshot = Arc::new(
            freeze_candidate(self.0.runtime_id, next_epoch, candidate)
                .map_err(|source| RuntimeReconfigureError::Edit { source })?,
        );

        self.0
            .next_registration_ordinal
            .store(next_ordinal.get(), Ordering::SeqCst);
        *guard = next_snapshot;
        self.0
            .published_epoch
            .store(next_epoch.get().get(), Ordering::Release);
        Ok(next_epoch)
    }

    fn plan_registration_identities(
        &self,
        count: usize,
    ) -> Result<(Vec<RegistrationIdentity>, NonZeroU64), RuntimeConfigError> {
        let mut next = self.0.next_registration_ordinal.load(Ordering::SeqCst);
        let mut identities = Vec::with_capacity(count);
        for _ in 0..count {
            let ordinal = NonZeroU64::new(next).ok_or(RuntimeConfigError::IdentityExhausted)?;
            identities.push(RegistrationIdentity::new(self.0.issuer, ordinal));
            next = next
                .checked_add(1)
                .ok_or(RuntimeConfigError::IdentityExhausted)?;
        }
        let next_ordinal = NonZeroU64::new(next).ok_or(RuntimeConfigError::IdentityExhausted)?;
        Ok((identities, next_ordinal))
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
    ///     CoreCapabilityBundle, EngineId, EngineRegistration, Error,
    ///     ExecutionContextIdentity, HardwareClassId, Runtime, StorageClass,
    ///     TransferEndpoint, TransferProvider, TransferRequest,
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
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let source_id = EngineId::new("example.engine.source")?;
    /// let destination_id = EngineId::new("example.engine.destination")?;
    /// let source = EngineRegistration::new(
    ///     source_id.clone(),
    ///     tenferro_runtime::ProviderDeviceIdentity::new(
    ///         tenferro_runtime::ProviderId::new("example.provider")?,
    ///         "source-0",
    ///     )?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.hardware")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage.clone(),
    ///     CoreCapabilityBundle::default(),
    /// )?;
    /// let destination = EngineRegistration::new(
    ///     destination_id.clone(),
    ///     tenferro_runtime::ProviderDeviceIdentity::new(
    ///         tenferro_runtime::ProviderId::new("example.provider")?,
    ///         "destination-0",
    ///     )?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.hardware")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage.clone(),
    ///     CoreCapabilityBundle::default(),
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
    pub fn build(mut self) -> Result<Runtime, RuntimeConfigError> {
        validate_candidate(&mut self.candidate)?;
        let runtime_id = RuntimeId::from_nonzero(allocate_nonzero(&NEXT_RUNTIME_ID)?);
        let issuer = allocate_nonzero(&NEXT_REGISTRATION_ISSUER)?;
        let post_ordinal = assign_initial_identities(&mut self.candidate, issuer)?;
        let epoch = RuntimeEpoch::one();
        let snapshot = Arc::new(freeze_candidate(runtime_id, epoch, self.candidate)?);
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
    ///     CoreCapabilityBundle, EngineId, EngineRegistration, Error,
    ///     ExecutionContextIdentity, HardwareClassId, Runtime, StorageClass,
    ///     TransferEndpoint, TransferProvider, TransferRequest,
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
    ///     Ok(EngineRegistration::new(
    ///         EngineId::new(id)?,
    ///         tenferro_runtime::ProviderDeviceIdentity::new(
    ///             tenferro_runtime::ProviderId::new("example.provider")?,
    ///             format!("engine:{id}"),
    ///         )?,
    ///         ExecutionContextIdentity::of::<()>(),
    ///         HardwareClassId::new("example.hardware")?,
    ///         Arc::from(vec![storage.clone()]),
    ///         storage.clone(),
    ///         CoreCapabilityBundle::default(),
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
pub struct EngineSnapshotView<'a> {
    slot: &'a FrozenEngineSlot,
}

impl<'a> EngineSnapshotView<'a> {
    /// Return the engine ID for this slot.
    pub fn engine_id(&self) -> &'a EngineId {
        self.slot.registration.engine_id()
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
        self.slot.registration.provider_device_identity()
    }

    /// Return the runtime-local registration identity for this slot.
    pub fn registration_identity(&self) -> RegistrationIdentity {
        self.slot.identity
    }

    /// Return the execution-context identity required by this slot.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.slot.registration.context_identity()
    }

    /// Return this engine's runtime event domain.
    pub fn event_domain_id(&self) -> EventDomainId {
        self.slot.event_domain_id
    }

    pub(crate) fn event_domain_driver(&self) -> Option<&'a Arc<dyn super::EventDomainDriver>> {
        self.slot.registration.event_domain_driver()
    }

    /// Return the hardware class for this slot.
    pub fn hardware_class(&self) -> &'a HardwareClassId {
        self.slot.registration.hardware_class()
    }

    /// Return direct core capability slots for this engine.
    pub fn capabilities(&self) -> &'a CoreCapabilityBundle {
        self.slot.registration.capabilities()
    }

    pub(super) fn storage_classes(&self) -> &'a [StorageClass] {
        self.slot.registration.storage_classes()
    }

    pub(super) fn default_storage_class(&self) -> &'a StorageClass {
        self.slot.registration.default_storage_class()
    }

    pub(super) fn accepts_input_placement(
        &self,
        placement: &tenferro_tensor::Placement,
        storage_class: &StorageClass,
    ) -> bool {
        self.slot
            .registration
            .accepts_input_placement(placement, storage_class)
    }

    pub(super) fn accepts_runtime_input(
        &self,
        input: &tenferro_tensor::TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.slot
            .registration
            .accepts_runtime_input(input, storage_class)
    }

    pub(super) fn accepts_input_signature(
        &self,
        input: &super::InputSignatureEntry,
        storage_class: &StorageClass,
    ) -> bool {
        self.slot
            .registration
            .accepts_input_signature(input, storage_class)
    }

    pub(super) fn owns_resident_tensor(
        &self,
        input: &tenferro_tensor::TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.slot
            .registration
            .owns_resident_tensor(input, storage_class)
    }

    pub(super) fn execution_engine(
        &self,
    ) -> Option<&'a Arc<dyn super::execution::ErasedTensorBackendExecutor>> {
        self.slot.registration.execution_engine.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn has_execution_engine_for_test(&self) -> bool {
        self.slot.registration.has_execution_engine()
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
    validate_engine_execution_contract(&registration)?;
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
                    identity: None,
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
    validate_engine_execution_contract(&registration)?;
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
            identity: None,
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

fn validate_candidate(candidate: &mut CandidateConfig) -> Result<(), RuntimeConfigError> {
    for record in candidate.engines.values() {
        validate_engine_execution_contract(&record.registration)?;
    }

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

    let mut bindings_to_set = Vec::new();
    for (route, record) in &candidate.transfers {
        let source_binding = validate_transfer_endpoint(candidate, route.source())?;
        let destination_binding = validate_transfer_endpoint(candidate, route.destination())?;
        let preserved = match &record.binding {
            CandidateTransferBinding::New => None,
            CandidateTransferBinding::Preserved {
                source,
                destination,
            }
            | CandidateTransferBinding::Bound {
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
        bindings_to_set.push((route.clone(), source_binding, destination_binding));
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

    for (route, source_binding, destination_binding) in bindings_to_set {
        if let Some(record) = candidate.transfers.get_mut(&route) {
            record.binding = CandidateTransferBinding::Bound {
                source: source_binding,
                destination: destination_binding,
            };
        }
    }
    Ok(())
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

fn validate_engine_execution_contract(
    registration: &EngineRegistration,
) -> Result<(), RuntimeConfigError> {
    if registration.has_execution_engine() && !registration.has_input_ingress_validator() {
        return Err(RuntimeConfigError::MissingInputIngressValidator {
            engine_id: registration.engine_id().clone(),
        });
    }
    Ok(())
}

fn assign_initial_identities(
    candidate: &mut CandidateConfig,
    issuer: NonZeroU64,
) -> Result<NonZeroU64, RuntimeConfigError> {
    let mut next = 1_u64;
    for record in candidate.engines.values_mut() {
        let ordinal = NonZeroU64::new(next).ok_or(RuntimeConfigError::IdentityExhausted)?;
        record.identity = Some(RegistrationIdentity::new(issuer, ordinal));
        next = next
            .checked_add(1)
            .ok_or(RuntimeConfigError::IdentityExhausted)?;
    }
    for module in candidate.modules.values_mut() {
        for record in module.engines.values_mut() {
            let ordinal = NonZeroU64::new(next).ok_or(RuntimeConfigError::IdentityExhausted)?;
            record.identity = Some(RegistrationIdentity::new(issuer, ordinal));
            next = next
                .checked_add(1)
                .ok_or(RuntimeConfigError::IdentityExhausted)?;
        }
    }
    NonZeroU64::new(next).ok_or(RuntimeConfigError::IdentityExhausted)
}

fn assign_new_identities(
    candidate: &mut CandidateConfig,
    mut identities: Vec<RegistrationIdentity>,
) {
    identities.reverse();
    for record in candidate.engines.values_mut() {
        if record.identity.is_none() {
            record.identity = identities.pop();
        }
    }
    for module in candidate.modules.values_mut() {
        for record in module.engines.values_mut() {
            if record.identity.is_none() {
                record.identity = identities.pop();
            }
        }
    }
}

fn freeze_candidate(
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    candidate: CandidateConfig,
) -> Result<RuntimeConfigSnapshot, RuntimeConfigError> {
    let mut engines = Vec::with_capacity(candidate.engines.len());
    let mut engine_indices = BTreeMap::new();
    let mut engine_locations = BTreeMap::new();
    let mut cache_owners = Vec::new();
    for (index, (engine_id, record)) in candidate.engines.into_iter().enumerate() {
        let identity = record
            .identity
            .ok_or(RuntimeConfigError::IdentityExhausted)?;
        if let Some(owner) = record.registration.cache_owner.clone() {
            cache_owners.push(FrozenCacheOwner {
                id: engine_cache_owner_id(&engine_id),
                kind: FrozenCacheOwnerKind::Engine,
                owner,
            });
        }
        if let Some(executor) = record.registration.execution_engine.clone() {
            cache_owners.push(FrozenCacheOwner {
                id: engine_extension_cache_owner_id(&engine_id),
                kind: FrozenCacheOwnerKind::Extension,
                owner: execution::extension_cache_owner(executor),
            });
        }
        let provider_device_identity = record.registration.provider_device_identity().clone();
        let event_domain_id = EventDomainId::new(runtime_id, epoch, identity);
        engine_locations.insert(
            engine_id.clone(),
            (provider_device_identity, event_domain_id),
        );
        engine_indices.insert(engine_id, index);
        engines.push(FrozenEngineSlot {
            registration: record.registration,
            identity,
            event_domain_id,
        });
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
        let (source_binding, destination_binding) = match record.binding {
            CandidateTransferBinding::Bound {
                source,
                destination,
            } => (source, destination),
            CandidateTransferBinding::New | CandidateTransferBinding::Preserved { .. } => {
                return Err(RuntimeConfigError::UnboundTransferRoute {
                    source_endpoint: route.source().clone(),
                    destination: route.destination().clone(),
                });
            }
        };
        let (source_current, source_event_domain_id) = engine_locations
            .get(route.source().engine_id())
            .ok_or_else(|| RuntimeConfigError::UnknownTransferEndpointEngine {
                endpoint: route.source().clone(),
            })?;
        let (destination_current, destination_event_domain_id) = engine_locations
            .get(route.destination().engine_id())
            .ok_or_else(|| RuntimeConfigError::UnknownTransferEndpointEngine {
                endpoint: route.destination().clone(),
            })?;
        if &source_binding != source_current {
            return Err(RuntimeConfigError::StaleTransferRoute {
                source_endpoint: route.source().clone(),
                destination: route.destination().clone(),
                endpoint: route.source().clone(),
                registered: Box::new(source_binding),
                current: Box::new(source_current.clone()),
            });
        }
        if &destination_binding != destination_current {
            return Err(RuntimeConfigError::StaleTransferRoute {
                source_endpoint: route.source().clone(),
                destination: route.destination().clone(),
                endpoint: route.destination().clone(),
                registered: Box::new(destination_binding),
                current: Box::new(destination_current.clone()),
            });
        }
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
        transfers.insert(resolved_route, record.provider);
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
        EngineRegistration::new(
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
        )
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
        assign_initial_identities(&mut candidate, NonZeroU64::new(1).unwrap())?;
        Ok(candidate)
    }

    #[test]
    fn freeze_rejects_new_and_preserved_route_bindings_without_validation() {
        let bindings = [
            CandidateTransferBinding::New,
            CandidateTransferBinding::Preserved {
                source: ProviderDeviceIdentity::new(
                    ProviderId::new("tenferro.test.freeze.provider").unwrap(),
                    "freeze-source",
                )
                .unwrap(),
                destination: ProviderDeviceIdentity::new(
                    ProviderId::new("tenferro.test.freeze.provider").unwrap(),
                    "freeze-destination",
                )
                .unwrap(),
            },
        ];
        for binding in bindings {
            let error = freeze_candidate(
                RuntimeId::from_nonzero(NonZeroU64::new(1).unwrap()),
                RuntimeEpoch::one(),
                candidate(binding).unwrap(),
            )
            .expect_err("freeze must require validation-produced Bound state");
            assert!(matches!(
                error,
                RuntimeConfigError::UnboundTransferRoute { .. }
            ));
        }
    }
}
