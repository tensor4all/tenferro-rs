use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    CacheOwnerId, CoreCapabilityBundle, EngineId, ExecutionContextIdentity, ExecutionPolicy,
    HardwareClassId, RegistrationIdentity, RuntimeCacheOwner, RuntimeConfigError, RuntimeEpoch,
    RuntimeId, RuntimeReconfigureError, RuntimeStateError, StorageClass,
};

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REGISTRATION_ISSUER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct CandidateRegistrationToken;

/// Immutable direct engine registration candidate.
///
/// `EngineRegistration` values can be cloned and registered repeatedly. Before
/// publication, candidate identity is the pair of engine ID and an internal
/// pointer token, so reusing the same cloned candidate is idempotent while a
/// distinct value for the same engine ID is a conflict unless explicitly
/// replaced.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::sync::Arc;
/// use tenferro_runtime::{
///     CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
///     HardwareClassId, StorageClass,
/// };
///
/// let storage = StorageClass::new("tenferro.storage.host")?;
/// let registration = EngineRegistration::new(
///     EngineId::new("tenferro.cpu")?,
///     ExecutionContextIdentity::of::<()>(),
///     HardwareClassId::new("tenferro.cpu.host")?,
///     Arc::from(vec![storage.clone()]),
///     storage,
///     CoreCapabilityBundle::builder().build(),
/// )?;
/// assert_eq!(registration.engine_id().as_str(), "tenferro.cpu");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct EngineRegistration {
    engine_id: EngineId,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    capabilities: CoreCapabilityBundle,
    cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    candidate_token: Arc<CandidateRegistrationToken>,
}

impl EngineRegistration {
    /// Build an immutable engine registration candidate.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::EmptyStorageClasses`] for an empty storage
    /// list, [`RuntimeConfigError::DuplicateStorageClass`] for duplicate storage
    /// classes, or [`RuntimeConfigError::DefaultStorageClassNotListed`] when the
    /// default storage class is not supported by this engine.
    pub fn new(
        engine_id: EngineId,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        capabilities: CoreCapabilityBundle,
    ) -> Result<Self, RuntimeConfigError> {
        validate_storage_classes(&engine_id, &storage_classes, &default_storage_class)?;
        Ok(Self {
            engine_id,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            capabilities,
            cache_owner: None,
            candidate_token: Arc::new(CandidateRegistrationToken),
        })
    }

    /// Return the engine identifier.
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// Return the execution-context type identity accepted by the engine.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    /// Return the hardware class exposed by this engine.
    pub fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }

    /// Return the supported storage classes in registration order.
    pub fn storage_classes(&self) -> &[StorageClass] {
        &self.storage_classes
    }

    /// Return the default storage class.
    pub fn default_storage_class(&self) -> &StorageClass {
        &self.default_storage_class
    }

    /// Return direct core capability slots.
    pub fn capabilities(&self) -> &CoreCapabilityBundle {
        &self.capabilities
    }

    /// Attach a runtime cache owner to this registration.
    pub fn with_cache_owner(mut self, owner: Arc<dyn RuntimeCacheOwner>) -> Self {
        self.cache_owner = Some(owner);
        self
    }

    fn candidate_identical(&self, other: &Self) -> bool {
        self.engine_id == other.engine_id
            && Arc::ptr_eq(&self.candidate_token, &other.candidate_token)
    }
}

impl fmt::Debug for EngineRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineRegistration")
            .field("engine_id", &self.engine_id)
            .field("context_identity", &self.context_identity)
            .field("hardware_class", &self.hardware_class)
            .field("storage_class_count", &self.storage_classes.len())
            .field("default_storage_class", &self.default_storage_class)
            .field("capabilities", &self.capabilities)
            .field("cache_owner", &self.cache_owner.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
struct CandidateEngineRecord {
    registration: EngineRegistration,
    identity: Option<RegistrationIdentity>,
}

#[derive(Clone, Debug)]
struct CandidateConfig {
    policy: ExecutionPolicy,
    engines: BTreeMap<EngineId, CandidateEngineRecord>,
}

impl CandidateConfig {
    fn empty() -> Self {
        Self {
            policy: default_execution_policy(),
            engines: BTreeMap::new(),
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
        }
    }
}

#[derive(Clone)]
struct FrozenEngineSlot {
    registration: EngineRegistration,
    identity: RegistrationIdentity,
}

impl fmt::Debug for FrozenEngineSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenEngineSlot")
            .field("engine_id", self.registration.engine_id())
            .field("registration_identity", &self.identity)
            .field("context_identity", &self.registration.context_identity())
            .field("hardware_class", self.registration.hardware_class())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
struct FrozenExtensionSlots {
    module_count: usize,
    engine_count: usize,
}

impl FrozenExtensionSlots {
    fn empty() -> Self {
        Self {
            module_count: 0,
            engine_count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FrozenCacheOwnerKind {
    Engine,
}

#[derive(Clone)]
struct FrozenCacheOwner {
    id: CacheOwnerId,
    kind: FrozenCacheOwnerKind,
    owner: Arc<dyn RuntimeCacheOwner>,
}

impl fmt::Debug for FrozenCacheOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenCacheOwner")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("owner_strong_count", &Arc::strong_count(&self.owner))
            .finish()
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
        self.extensions.module_count
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
}

impl fmt::Debug for RuntimeConfigSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeConfigSnapshot")
            .field("runtime_id", &self.runtime_id)
            .field("epoch", &self.epoch)
            .field("execution_policy", &self.policy)
            .field("engine_count", &self.engines.len())
            .field("extension_module_count", &self.extensions.module_count)
            .field("extension_engine_count", &self.extensions.engine_count)
            .field("cache_owner_count", &self.cache_owners.len())
            .finish_non_exhaustive()
    }
}

struct RuntimeState {
    runtime_id: RuntimeId,
    issuer: NonZeroU64,
    next_registration_ordinal: AtomicU64,
    active: Mutex<Arc<RuntimeConfigSnapshot>>,
    published_epoch: AtomicU64,
    #[cfg(test)]
    snapshot_lock_calls: AtomicUsize,
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
    /// Returns [`RuntimeStateError::Poisoned`] when the active snapshot mutex was
    /// poisoned by another thread.
    pub fn snapshot(&self) -> Result<Arc<RuntimeConfigSnapshot>, RuntimeStateError> {
        #[cfg(test)]
        self.0.snapshot_lock_calls.fetch_add(1, Ordering::SeqCst);
        self.0
            .active
            .lock()
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

        let next_epoch =
            base.epoch()
                .checked_next()
                .ok_or(RuntimeReconfigureError::EpochExhausted {
                    current: base.epoch(),
                })?;

        let mut guard = self
            .0
            .active
            .lock()
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
            .count();
        let (new_identities, next_ordinal) = self
            .plan_registration_identities(missing_identities)
            .map_err(|source| RuntimeReconfigureError::Edit { source })?;
        assign_new_identities(&mut candidate, new_identities);
        let next_snapshot = Arc::new(
            freeze_candidate(
                self.0.runtime_id,
                next_epoch,
                candidate,
                base.extensions.clone(),
            )
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
    pub(crate) fn reset_snapshot_lock_calls_for_test(&self) {
        self.0.snapshot_lock_calls.store(0, Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn snapshot_lock_calls_for_test(&self) -> usize {
        self.0.snapshot_lock_calls.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn force_epoch_for_test(&self, epoch: RuntimeEpoch) {
        let mut guard = self.0.active.lock().expect("test runtime lock");
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
            let _guard = state.active.lock().expect("test runtime lock");
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

    /// Build and publish the initial runtime snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::IdentityExhausted`] if runtime or
    /// registration identity allocation would wrap.
    pub fn build(mut self) -> Result<Runtime, RuntimeConfigError> {
        let runtime_id = RuntimeId::from_nonzero(allocate_nonzero(&NEXT_RUNTIME_ID)?);
        let issuer = allocate_nonzero(&NEXT_REGISTRATION_ISSUER)?;
        let post_ordinal = assign_initial_identities(&mut self.candidate, issuer)?;
        let epoch = RuntimeEpoch::one();
        let snapshot = Arc::new(freeze_candidate(
            runtime_id,
            epoch,
            self.candidate,
            FrozenExtensionSlots::empty(),
        )?);
        let state = RuntimeState {
            runtime_id,
            issuer,
            next_registration_ordinal: AtomicU64::new(post_ordinal.get()),
            active: Mutex::new(snapshot),
            published_epoch: AtomicU64::new(epoch.get().get()),
            #[cfg(test)]
            snapshot_lock_calls: AtomicUsize::new(0),
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
}

impl fmt::Debug for RuntimeReconfiguration<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeReconfiguration")
            .field("engine_count", &self.candidate.engines.len())
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

    /// Return the runtime-local registration identity for this slot.
    pub fn registration_identity(&self) -> RegistrationIdentity {
        self.slot.identity
    }

    /// Return the execution-context identity required by this slot.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.slot.registration.context_identity()
    }

    /// Return the hardware class for this slot.
    pub fn hardware_class(&self) -> &'a HardwareClassId {
        self.slot.registration.hardware_class()
    }

    /// Return direct core capability slots for this engine.
    pub fn capabilities(&self) -> &'a CoreCapabilityBundle {
        self.slot.registration.capabilities()
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

fn validate_storage_classes(
    engine_id: &EngineId,
    storage_classes: &[StorageClass],
    default_storage_class: &StorageClass,
) -> Result<(), RuntimeConfigError> {
    if storage_classes.is_empty() {
        return Err(RuntimeConfigError::EmptyStorageClasses {
            engine_id: engine_id.clone(),
        });
    }
    for duplicate_index in 0..storage_classes.len() {
        if let Some(first_index) = (0..duplicate_index)
            .find(|&first| storage_classes[first] == storage_classes[duplicate_index])
        {
            return Err(RuntimeConfigError::DuplicateStorageClass {
                engine_id: engine_id.clone(),
                storage_class: storage_classes[duplicate_index].clone(),
                first_index,
                duplicate_index,
            });
        }
    }
    if !storage_classes
        .iter()
        .any(|storage_class| storage_class == default_storage_class)
    {
        return Err(RuntimeConfigError::DefaultStorageClassNotListed {
            engine_id: engine_id.clone(),
            default_storage_class: default_storage_class.clone(),
        });
    }
    Ok(())
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
    let engine_id = registration.engine_id().clone();
    match candidate.engines.get_mut(&engine_id) {
        Some(existing) if existing.registration.candidate_identical(&registration) => Ok(()),
        Some(existing) => {
            *existing = CandidateEngineRecord {
                registration,
                identity: None,
            };
            *changed = true;
            Ok(())
        }
        None => Err(RuntimeConfigError::MissingEngine { engine_id }),
    }
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
}

fn freeze_candidate(
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    candidate: CandidateConfig,
    extensions: FrozenExtensionSlots,
) -> Result<RuntimeConfigSnapshot, RuntimeConfigError> {
    let mut engines = Vec::with_capacity(candidate.engines.len());
    let mut engine_indices = BTreeMap::new();
    let mut cache_owners = Vec::new();
    for (index, (engine_id, record)) in candidate.engines.into_iter().enumerate() {
        let identity = record
            .identity
            .ok_or(RuntimeConfigError::IdentityExhausted)?;
        if let Some(owner) = record.registration.cache_owner.clone() {
            cache_owners.push(FrozenCacheOwner {
                id: engine_cache_owner_id(&engine_id)?,
                kind: FrozenCacheOwnerKind::Engine,
                owner,
            });
        }
        engine_indices.insert(engine_id, index);
        engines.push(FrozenEngineSlot {
            registration: record.registration,
            identity,
        });
    }
    Ok(RuntimeConfigSnapshot {
        runtime_id,
        epoch,
        policy: candidate.policy,
        engines: engines.into(),
        engine_indices,
        extensions,
        cache_owners: cache_owners.into(),
    })
}

fn engine_cache_owner_id(engine_id: &EngineId) -> Result<CacheOwnerId, RuntimeConfigError> {
    CacheOwnerId::new(format!("{}.cache-owner", engine_id.as_str()))
        .map_err(RuntimeConfigError::from)
}

fn allocate_nonzero(counter: &AtomicU64) -> Result<NonZeroU64, RuntimeConfigError> {
    let value = counter
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |next| {
            next.checked_add(1)
        })
        .map_err(|_| RuntimeConfigError::IdentityExhausted)?;
    NonZeroU64::new(value).ok_or(RuntimeConfigError::IdentityExhausted)
}
