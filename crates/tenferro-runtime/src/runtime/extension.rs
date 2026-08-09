use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::fmt;
use std::hash::Hasher;
use std::sync::Arc;

use super::identity::validate_identifier;
use super::{
    CacheOwnerId, EngineId, ExtensionEngine, ExtensionPlanningConfig, RegistrationIdentity,
    RuntimeCacheOwner,
};
use super::{
    ExecutionContextIdentity, ExtensionModuleError, IdentityError, IdentityKind, RegistrationKey,
    RuntimeConfigError,
};

pub(super) type ExtensionFamilyId = &'static str;

/// Validated extension module identifier.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::ExtensionModuleId;
///
/// assert_eq!(ExtensionModuleId::new("tenferro.module.test")?.as_str(), "tenferro.module.test");
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExtensionModuleId(Arc<str>);

impl ExtensionModuleId {
    /// Validate a lowercase ASCII namespaced extension module identifier.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] when `value` does not match the runtime
    /// identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::ExtensionModule).map(Self)
    }

    /// Borrow the validated identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Transactional extension module.
pub trait ExtensionModule: fmt::Debug + Send + Sync + 'static {
    /// Return this module's validated ID.
    fn module_id(&self) -> &ExtensionModuleId;

    /// Register extension engines, planning configs, and cache owners.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionModuleError`] when this module's transaction is
    /// internally inconsistent.
    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError>;
}

pub(super) struct CandidateModuleRecord {
    pub(super) module: Arc<dyn ExtensionModule>,
    pub(super) engines: BTreeMap<(ExtensionFamilyId, EngineId), CandidateExtensionEngine>,
    pub(super) configs: BTreeMap<EngineId, Arc<dyn ExtensionPlanningConfig>>,
    pub(super) owners: BTreeMap<CacheOwnerId, Arc<dyn RuntimeCacheOwner>>,
}

impl fmt::Debug for CandidateModuleRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandidateModuleRecord")
            .field("module_id", self.module.module_id())
            .field("engine_count", &self.engines.len())
            .field("config_count", &self.configs.len())
            .field("owner_count", &self.owners.len())
            .finish_non_exhaustive()
    }
}

impl Clone for CandidateModuleRecord {
    fn clone(&self) -> Self {
        Self {
            module: Arc::clone(&self.module),
            engines: self.engines.clone(),
            configs: self.configs.clone(),
            owners: self.owners.clone(),
        }
    }
}

impl CandidateModuleRecord {
    pub(super) fn module_identical(&self, module: &Arc<dyn ExtensionModule>) -> bool {
        Arc::ptr_eq(&self.module, module)
    }
}

pub(super) struct CandidateExtensionEngine {
    pub(super) engine: Arc<dyn ExtensionEngine>,
    pub(super) identity: CandidateRegistrationIdentity,
}

#[derive(Clone, Debug)]
pub(super) enum CandidateRegistrationIdentity {
    New,
    Preserved(RegistrationIdentity),
}

impl Clone for CandidateExtensionEngine {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            identity: self.identity.clone(),
        }
    }
}

impl fmt::Debug for CandidateExtensionEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandidateExtensionEngine")
            .field("family_id", &self.engine.family_id())
            .field("engine_id", self.engine.engine_id())
            .field("context_identity", &self.engine.context_identity())
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

pub(super) struct BoundCandidateExtensionEngine {
    pub(super) engine: Arc<dyn ExtensionEngine>,
    pub(super) identity: RegistrationIdentity,
}

pub(super) struct BoundCandidateModuleRecord {
    pub(super) module: Arc<dyn ExtensionModule>,
    pub(super) engines: BTreeMap<(ExtensionFamilyId, EngineId), BoundCandidateExtensionEngine>,
    pub(super) configs: BTreeMap<EngineId, Arc<dyn ExtensionPlanningConfig>>,
    pub(super) owners: BTreeMap<CacheOwnerId, Arc<dyn RuntimeCacheOwner>>,
}

#[derive(Clone)]
pub(super) struct FrozenExtensionEngineSlot {
    module_id: ExtensionModuleId,
    family_id: ExtensionFamilyId,
    engine_id: EngineId,
    context_identity: ExecutionContextIdentity,
    registration_identity: RegistrationIdentity,
    engine: Arc<dyn ExtensionEngine>,
    config: Arc<dyn ExtensionPlanningConfig>,
}

impl fmt::Debug for FrozenExtensionEngineSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenExtensionEngineSlot")
            .field("module_id", &self.module_id)
            .field("family_id", &self.family_id)
            .field("engine_id", &self.engine_id)
            .field("context_identity", &self.context_identity)
            .field("registration_identity", &self.registration_identity)
            .field("config_retained_bytes", &self.config.retained_bytes())
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) struct FrozenExtensionSlots {
    modules: BTreeMap<ExtensionModuleId, Arc<dyn ExtensionModule>>,
    engines: Arc<[FrozenExtensionEngineSlot]>,
    by_family_engine: BTreeMap<(ExtensionFamilyId, EngineId), usize>,
    owners: BTreeMap<(ExtensionModuleId, CacheOwnerId), Arc<dyn RuntimeCacheOwner>>,
}

pub(super) struct ExtensionEngineSnapshotView<'a> {
    slot: &'a FrozenExtensionEngineSlot,
}

impl<'a> ExtensionEngineSnapshotView<'a> {
    pub(super) fn module_id(&self) -> &'a ExtensionModuleId {
        &self.slot.module_id
    }

    pub(super) fn family_id(&self) -> ExtensionFamilyId {
        self.slot.family_id
    }

    pub(super) fn engine_id(&self) -> &'a EngineId {
        &self.slot.engine_id
    }

    pub(super) fn context_identity(&self) -> ExecutionContextIdentity {
        self.slot.context_identity
    }

    pub(super) fn registration_identity(&self) -> RegistrationIdentity {
        self.slot.registration_identity
    }

    pub(super) fn engine(&self) -> &'a Arc<dyn ExtensionEngine> {
        &self.slot.engine
    }

    pub(super) fn config(&self) -> &'a Arc<dyn ExtensionPlanningConfig> {
        &self.slot.config
    }
}

#[cfg(test)]
pub(super) type ExtensionSlotFullForTest<'a> = (
    &'a ExtensionModuleId,
    ExtensionFamilyId,
    &'a EngineId,
    RegistrationIdentity,
    &'a Arc<dyn ExtensionEngine>,
    &'a Arc<dyn ExtensionPlanningConfig>,
);

impl FrozenExtensionSlots {
    pub(super) fn module_count(&self) -> usize {
        self.modules.len()
    }

    pub(super) fn engine_count(&self) -> usize {
        self.engines.len()
    }

    pub(super) fn has_family(&self, family_id: ExtensionFamilyId) -> bool {
        self.engines.iter().any(|slot| slot.family_id == family_id)
    }

    pub(super) fn to_candidate_modules(
        &self,
    ) -> BTreeMap<ExtensionModuleId, CandidateModuleRecord> {
        let mut modules = BTreeMap::new();
        for (module_id, module) in &self.modules {
            modules.insert(
                module_id.clone(),
                CandidateModuleRecord {
                    module: Arc::clone(module),
                    engines: BTreeMap::new(),
                    configs: BTreeMap::new(),
                    owners: BTreeMap::new(),
                },
            );
        }
        for slot in self.engines.iter() {
            if let Some(record) = modules.get_mut(&slot.module_id) {
                record.engines.insert(
                    (slot.family_id, slot.engine_id.clone()),
                    CandidateExtensionEngine {
                        engine: Arc::clone(&slot.engine),
                        identity: CandidateRegistrationIdentity::Preserved(
                            slot.registration_identity,
                        ),
                    },
                );
                record
                    .configs
                    .insert(slot.engine_id.clone(), Arc::clone(&slot.config));
            }
        }
        for ((module_id, owner_id), owner) in &self.owners {
            if let Some(record) = modules.get_mut(module_id) {
                record.owners.insert(owner_id.clone(), Arc::clone(owner));
            }
        }
        modules
    }

    pub(super) fn cache_owner_records(
        &self,
    ) -> impl Iterator<Item = (CacheOwnerId, Arc<dyn RuntimeCacheOwner>)> + '_ {
        self.owners.iter().map(|((module_id, owner_id), owner)| {
            (
                extension_cache_owner_id(module_id, owner_id),
                Arc::clone(owner),
            )
        })
    }

    #[cfg(test)]
    pub(super) fn slots_for_test(
        &self,
    ) -> impl Iterator<
        Item = (
            &ExtensionModuleId,
            ExtensionFamilyId,
            &EngineId,
            RegistrationIdentity,
        ),
    > {
        self.engines.iter().map(|slot| {
            (
                &slot.module_id,
                slot.family_id,
                &slot.engine_id,
                slot.registration_identity,
            )
        })
    }

    #[cfg(test)]
    pub(super) fn slot_identity_for_test(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<RegistrationIdentity> {
        self.extension_engine_slot(family_id, engine_id)
            .map(|slot| slot.registration_identity)
    }

    #[cfg(test)]
    pub(super) fn slot_full_for_test(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<ExtensionSlotFullForTest<'_>> {
        self.extension_engine_slot(family_id, engine_id)
            .map(|slot| {
                (
                    &slot.module_id,
                    slot.family_id,
                    &slot.engine_id,
                    slot.registration_identity,
                    &slot.engine,
                    &slot.config,
                )
            })
    }

    pub(super) fn slot_for_preparation(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<ExtensionEngineSnapshotView<'_>> {
        self.extension_engine_slot(family_id, engine_id)
            .map(|slot| ExtensionEngineSnapshotView { slot })
    }

    pub(super) fn has_engine(&self, family_id: ExtensionFamilyId, engine_id: &EngineId) -> bool {
        self.by_family_engine
            .contains_key(&(family_id, engine_id.clone()))
    }

    fn extension_engine_slot(
        &self,
        family_id: ExtensionFamilyId,
        engine_id: &EngineId,
    ) -> Option<&FrozenExtensionEngineSlot> {
        self.by_family_engine
            .get(&(family_id, engine_id.clone()))
            .map(|&index| &self.engines[index])
    }
}

impl fmt::Debug for FrozenExtensionSlots {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenExtensionSlots")
            .field("module_count", &self.modules.len())
            .field("engine_count", &self.engines.len())
            .field("index_count", &self.by_family_engine.len())
            .field("owner_count", &self.owners.len())
            .finish_non_exhaustive()
    }
}

pub(super) fn freeze_extension_slots(
    modules: BTreeMap<ExtensionModuleId, BoundCandidateModuleRecord>,
) -> Result<FrozenExtensionSlots, RuntimeConfigError> {
    let mut frozen_modules = BTreeMap::new();
    let mut engines = Vec::new();
    let mut by_family_engine = BTreeMap::new();
    let mut owners = BTreeMap::new();

    for (module_id, module) in modules {
        frozen_modules.insert(module_id.clone(), Arc::clone(&module.module));
        for (owner_id, owner) in module.owners {
            owners.insert((module_id.clone(), owner_id), owner);
        }
        for ((family_id, engine_id), record) in module.engines {
            if by_family_engine
                .insert((family_id, engine_id.clone()), engines.len())
                .is_some()
            {
                return Err(RuntimeConfigError::ConflictingRegistration {
                    key: RegistrationKey::ExtensionEngine {
                        family: family_id,
                        engine: engine_id,
                    },
                });
            }
            let config = module.configs.get(&engine_id).cloned().ok_or_else(|| {
                RuntimeConfigError::ExtensionModule {
                    source: ExtensionModuleError::MissingPlanningConfig {
                        module_id: module_id.clone(),
                        engine_id: engine_id.clone(),
                    },
                }
            })?;
            engines.push(FrozenExtensionEngineSlot {
                module_id: module_id.clone(),
                family_id,
                engine_id: engine_id.clone(),
                context_identity: record.engine.context_identity(),
                registration_identity: record.identity,
                engine: record.engine,
                config,
            });
        }
    }

    Ok(FrozenExtensionSlots {
        modules: frozen_modules,
        engines: engines.into(),
        by_family_engine,
        owners,
    })
}

struct ExtensionRegistrationTransaction {
    module_id: ExtensionModuleId,
    engines: BTreeMap<(ExtensionFamilyId, EngineId), Arc<dyn ExtensionEngine>>,
    configs: BTreeMap<EngineId, Arc<dyn ExtensionPlanningConfig>>,
    owners: BTreeMap<CacheOwnerId, Arc<dyn RuntimeCacheOwner>>,
}

impl ExtensionRegistrationTransaction {
    fn new(module_id: ExtensionModuleId) -> Self {
        Self {
            module_id,
            engines: BTreeMap::new(),
            configs: BTreeMap::new(),
            owners: BTreeMap::new(),
        }
    }

    fn into_candidate(
        self,
        module: Arc<dyn ExtensionModule>,
    ) -> Result<CandidateModuleRecord, ExtensionModuleError> {
        self.validate()?;
        let engines = self
            .engines
            .into_iter()
            .map(|(key, engine)| {
                (
                    key,
                    CandidateExtensionEngine {
                        engine,
                        identity: CandidateRegistrationIdentity::New,
                    },
                )
            })
            .collect();
        Ok(CandidateModuleRecord {
            module,
            engines,
            configs: self.configs,
            owners: self.owners,
        })
    }

    fn validate(&self) -> Result<(), ExtensionModuleError> {
        for &(family_id, ref engine_id) in self.engines.keys() {
            match self.configs.get(engine_id) {
                Some(config) if config.family_id() == family_id => {}
                Some(config) => {
                    return Err(ExtensionModuleError::PlanningConfigFamilyMismatch {
                        module_id: self.module_id.clone(),
                        engine_id: engine_id.clone(),
                        expected: family_id,
                        actual: config.family_id(),
                    });
                }
                None => {
                    return Err(ExtensionModuleError::MissingPlanningConfig {
                        module_id: self.module_id.clone(),
                        engine_id: engine_id.clone(),
                    });
                }
            }
        }
        for engine_id in self.configs.keys() {
            if engine_match_count(&self.engines, engine_id) != 1 {
                return Err(ExtensionModuleError::PlanningConfigWithoutEngine {
                    module_id: self.module_id.clone(),
                    engine_id: engine_id.clone(),
                });
            }
        }
        Ok(())
    }
}

pub(super) fn bind_candidate_module(
    module: CandidateModuleRecord,
    allocate_identity: &mut impl FnMut() -> Result<RegistrationIdentity, RuntimeConfigError>,
) -> Result<BoundCandidateModuleRecord, RuntimeConfigError> {
    let engines = module
        .engines
        .into_iter()
        .map(|(key, record)| {
            let identity = match record.identity {
                CandidateRegistrationIdentity::New => allocate_identity()?,
                CandidateRegistrationIdentity::Preserved(identity) => identity,
            };
            Ok((
                key,
                BoundCandidateExtensionEngine {
                    engine: record.engine,
                    identity,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, RuntimeConfigError>>()?;
    Ok(BoundCandidateModuleRecord {
        module: module.module,
        engines,
        configs: module.configs,
        owners: module.owners,
    })
}

/// Borrowed registrar for one extension module transaction.
pub struct ExtensionModuleRegistrar<'a> {
    transaction: &'a mut ExtensionRegistrationTransaction,
}

impl ExtensionModuleRegistrar<'_> {
    /// Register one extension preparation engine.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionModuleError::ConflictingEngine`] when a distinct
    /// engine already occupies the same `(family, engine)` transaction key.
    pub fn register_engine(
        &mut self,
        engine: Arc<dyn ExtensionEngine>,
    ) -> Result<(), ExtensionModuleError> {
        let key = (engine.family_id(), engine.engine_id().clone());
        match self.transaction.engines.get(&key) {
            Some(existing) if Arc::ptr_eq(existing, &engine) => Ok(()),
            Some(_) => Err(ExtensionModuleError::ConflictingEngine {
                module_id: self.transaction.module_id.clone(),
                family_id: key.0,
                engine_id: key.1,
            }),
            None => {
                self.transaction.engines.insert(key, engine);
                Ok(())
            }
        }
    }

    /// Register the planning config for one extension engine.
    ///
    /// # Errors
    ///
    /// Returns a typed [`ExtensionModuleError`] when the target engine is absent,
    /// the config family is mismatched, or an unequal config is already present.
    pub fn register_planning_config(
        &mut self,
        engine_id: EngineId,
        config: Arc<dyn ExtensionPlanningConfig>,
    ) -> Result<(), ExtensionModuleError> {
        let Some((family_id, _)) = unique_engine_for_config(&self.transaction.engines, &engine_id)
        else {
            return Err(ExtensionModuleError::PlanningConfigWithoutEngine {
                module_id: self.transaction.module_id.clone(),
                engine_id,
            });
        };
        if config.family_id() != family_id {
            return Err(ExtensionModuleError::PlanningConfigFamilyMismatch {
                module_id: self.transaction.module_id.clone(),
                engine_id,
                expected: family_id,
                actual: config.family_id(),
            });
        }

        match self.transaction.configs.get(&engine_id) {
            Some(existing) if config_payloads_equal(existing.as_ref(), config.as_ref()) => Ok(()),
            Some(_) => Err(ExtensionModuleError::ConflictingPlanningConfig {
                module_id: self.transaction.module_id.clone(),
                engine_id,
            }),
            None => {
                self.transaction.configs.insert(engine_id, config);
                Ok(())
            }
        }
    }

    /// Register a cache owner owned by this extension module.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionModuleError::ConflictingCacheOwner`] when a distinct
    /// owner already occupies the same local owner ID.
    pub fn register_cache_owner(
        &mut self,
        id: CacheOwnerId,
        owner: Arc<dyn RuntimeCacheOwner>,
    ) -> Result<(), ExtensionModuleError> {
        match self.transaction.owners.get(&id) {
            Some(existing) if Arc::ptr_eq(existing, &owner) => Ok(()),
            Some(_) => Err(ExtensionModuleError::ConflictingCacheOwner {
                module_id: self.transaction.module_id.clone(),
                owner: id,
            }),
            None => {
                self.transaction.owners.insert(id, owner);
                Ok(())
            }
        }
    }
}

impl fmt::Debug for ExtensionModuleRegistrar<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionModuleRegistrar")
            .field("module_id", &self.transaction.module_id)
            .field("engine_count", &self.transaction.engines.len())
            .field("config_count", &self.transaction.configs.len())
            .field("owner_count", &self.transaction.owners.len())
            .finish_non_exhaustive()
    }
}

pub(super) fn configure_module(
    module: Arc<dyn ExtensionModule>,
) -> Result<CandidateModuleRecord, ExtensionModuleError> {
    let module_id = module.module_id().clone();
    let mut transaction = ExtensionRegistrationTransaction::new(module_id);
    {
        let mut registrar = ExtensionModuleRegistrar {
            transaction: &mut transaction,
        };
        module.configure(&mut registrar)?;
    }
    transaction.into_candidate(module)
}

pub(super) fn extension_cache_owner_id(
    module_id: &ExtensionModuleId,
    local: &CacheOwnerId,
) -> CacheOwnerId {
    let module = module_id.as_str();
    let local = local.as_str();
    CacheOwnerId::from_canonical_owner_id(Arc::<str>::from(format!(
        "extension[{}]:{module}[{}]:{local}",
        module.len(),
        local.len(),
    )))
}

fn unique_engine_for_config(
    engines: &BTreeMap<(ExtensionFamilyId, EngineId), Arc<dyn ExtensionEngine>>,
    engine_id: &EngineId,
) -> Option<(ExtensionFamilyId, EngineId)> {
    let mut matches = engines
        .keys()
        .filter(|(_, candidate_engine)| candidate_engine == engine_id);
    let first = matches.next()?;
    matches.next().is_none().then(|| (first.0, first.1.clone()))
}

fn engine_match_count(
    engines: &BTreeMap<(ExtensionFamilyId, EngineId), Arc<dyn ExtensionEngine>>,
    engine_id: &EngineId,
) -> usize {
    engines
        .keys()
        .filter(|(_, candidate_engine)| candidate_engine == engine_id)
        .count()
}

fn config_payloads_equal(
    left: &dyn ExtensionPlanningConfig,
    right: &dyn ExtensionPlanningConfig,
) -> bool {
    payload_hash(left) == payload_hash(right) && left.payload_eq(right)
}

fn payload_hash(config: &dyn ExtensionPlanningConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.payload_hash(&mut hasher);
    hasher.finish()
}
