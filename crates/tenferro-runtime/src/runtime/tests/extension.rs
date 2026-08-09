use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::error::Error as StdError;
use std::hash::Hasher;
use std::sync::{Arc, Mutex, Weak};

use tenferro_ops::ext_op::ExtensionOp;

use super::super::*;

#[derive(Clone, Debug, Default)]
struct SentinelSink {
    engine: Arc<Mutex<Option<Weak<()>>>>,
    config: Arc<Mutex<Option<Weak<()>>>>,
    owner: Arc<Mutex<Option<Weak<()>>>>,
}

impl SentinelSink {
    fn capture_engine(&self, sentinel: &Arc<()>) {
        *self.engine.lock().expect("engine sentinel lock") = Some(Arc::downgrade(sentinel));
    }

    fn capture_config(&self, sentinel: &Arc<()>) {
        *self.config.lock().expect("config sentinel lock") = Some(Arc::downgrade(sentinel));
    }

    fn capture_owner(&self, sentinel: &Arc<()>) {
        *self.owner.lock().expect("owner sentinel lock") = Some(Arc::downgrade(sentinel));
    }

    fn all_released(&self) -> bool {
        self.engine
            .lock()
            .expect("engine sentinel lock")
            .as_ref()
            .is_some_and(|weak| weak.upgrade().is_none())
            && self
                .config
                .lock()
                .expect("config sentinel lock")
                .as_ref()
                .is_some_and(|weak| weak.upgrade().is_none())
            && self
                .owner
                .lock()
                .expect("owner sentinel lock")
                .as_ref()
                .is_some_and(|weak| weak.upgrade().is_none())
    }
}

#[derive(Clone, Debug)]
struct TestExtensionConfig {
    family: &'static str,
    value: u64,
    retained_bytes: usize,
    forced_hash: Option<u64>,
    sentinel: Option<Arc<()>>,
}

impl ExtensionPlanningConfig for TestExtensionConfig {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.forced_hash.unwrap_or(self.value));
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family == other.family && self.value == other.value)
    }

    fn retained_bytes(&self) -> usize {
        let _ = self.sentinel.as_ref().map(Arc::strong_count);
        self.retained_bytes
    }
}

#[derive(Clone, Debug)]
struct TestExtensionEngine {
    family: &'static str,
    engine: EngineId,
    context: ExecutionContextIdentity,
    sentinel: Option<Arc<()>>,
}

impl ExtensionEngine for TestExtensionEngine {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        self.context
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        let _ = self.sentinel.as_ref().map(Arc::strong_count);
        assert_eq!(request.operation().family_id(), self.family);
        assert_eq!(request.binding().engine_id(), &self.engine);
        assert_eq!(request.hardware_class().as_str(), "tenferro.cpu.host");
        assert_eq!(request.planning().planning_seed(), 17);
        assert_eq!(request.prepare_options_key().planning_seed(), 17);
        assert_eq!(request.inputs().entries().len(), 0);
        assert_eq!(request.specialization().inputs().len(), 0);
        assert_eq!(request.extension_config().family_id(), self.family);
        Ok(PrepareCapability::Unsupported(
            UnsupportedReason::Operation {
                operation: "test-extension",
            },
        ))
    }
}

#[derive(Debug)]
struct CountingCacheOwner {
    sentinel: Option<Arc<()>>,
}

impl RuntimeCacheOwner for CountingCacheOwner {
    fn cache_stats(&self) -> Result<CacheStats, CacheOwnerError> {
        let _ = self.sentinel.as_ref().map(Arc::strong_count);
        Ok(CacheStats {
            entries: 1,
            retained_bytes: 2,
            hits: 3,
            misses: 4,
            evictions: 5,
            clears: 6,
        })
    }

    fn clear_caches(&self) -> Result<(), CacheOwnerError> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ModuleAction {
    Register(FixedTuple),
    DuplicateEngine(FixedTuple),
    DuplicateEqualConfig(FixedTuple),
    DuplicateConflictingConfig(FixedTuple),
    DuplicateOwner(FixedTuple),
    FailAfterRegisterWithSentinels(FixedTuple, SentinelSink),
}

#[derive(Clone, Debug)]
struct FixedTuple {
    family: &'static str,
    engine: EngineId,
    config_value: u64,
    config_hash: Option<u64>,
    owner: CacheOwnerId,
}

#[derive(Debug)]
struct TestModule {
    id: ExtensionModuleId,
    action: ModuleAction,
}

impl ExtensionModule for TestModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        match &self.action {
            ModuleAction::Register(tuple) => register_tuple(registrar, tuple, None),
            ModuleAction::DuplicateEngine(tuple) => {
                let engine = extension_engine(tuple, None);
                registrar.register_engine(Arc::clone(&engine))?;
                registrar.register_engine(Arc::new(TestExtensionEngine {
                    family: tuple.family,
                    engine: tuple.engine.clone(),
                    context: ExecutionContextIdentity::of::<u32>(),
                    sentinel: None,
                }))
            }
            ModuleAction::DuplicateEqualConfig(tuple) => {
                let engine = extension_engine(tuple, None);
                registrar.register_engine(engine)?;
                let config = extension_config(tuple, None);
                registrar.register_planning_config(tuple.engine.clone(), Arc::clone(&config))?;
                registrar
                    .register_planning_config(tuple.engine.clone(), extension_config(tuple, None))
            }
            ModuleAction::DuplicateConflictingConfig(tuple) => {
                let engine = extension_engine(tuple, None);
                registrar.register_engine(engine)?;
                registrar.register_planning_config(
                    tuple.engine.clone(),
                    extension_config(tuple, None),
                )?;
                let conflicting = Arc::new(TestExtensionConfig {
                    family: tuple.family,
                    value: tuple.config_value + 1,
                    retained_bytes: 0,
                    forced_hash: tuple.config_hash,
                    sentinel: None,
                });
                registrar.register_planning_config(tuple.engine.clone(), conflicting)
            }
            ModuleAction::DuplicateOwner(tuple) => {
                register_tuple(registrar, tuple, None)?;
                registrar.register_cache_owner(
                    tuple.owner.clone(),
                    Arc::new(CountingCacheOwner { sentinel: None }),
                )
            }
            ModuleAction::FailAfterRegisterWithSentinels(tuple, sink) => {
                register_tuple(registrar, tuple, Some(sink))?;
                Err(ExtensionModuleError::ConflictingCacheOwner {
                    module_id: self.id.clone(),
                    owner: tuple.owner.clone(),
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
struct TestOp {
    family: &'static str,
}

impl ExtensionOp for TestOp {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write(self.family.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.family == self.family)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        0
    }

    fn output_count(&self) -> usize {
        0
    }

    fn infer_output_meta(
        &self,
        _ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(tenferro_tensor::DType, Vec<crate::SymDim>)>> {
        Ok(Vec::new())
    }
}

fn module_id(value: &str) -> ExtensionModuleId {
    ExtensionModuleId::new(value).expect("valid module id")
}

fn engine_id(value: &str) -> EngineId {
    EngineId::new(value).expect("valid engine id")
}

fn owner_id(value: &str) -> CacheOwnerId {
    CacheOwnerId::new(value).expect("valid owner id")
}

fn fixed(module_suffix: &str, family: &'static str, engine_suffix: &str) -> FixedTuple {
    FixedTuple {
        family,
        engine: engine_id(&format!("tenferro.engine.{engine_suffix}")),
        config_value: 7,
        config_hash: None,
        owner: owner_id(&format!("tenferro.owner.{module_suffix}")),
    }
}

fn module(id: &str, action: ModuleAction) -> Arc<dyn ExtensionModule> {
    Arc::new(TestModule {
        id: module_id(id),
        action,
    })
}

fn extension_engine(tuple: &FixedTuple, sink: Option<&SentinelSink>) -> Arc<dyn ExtensionEngine> {
    let sentinel = sink.map(|sink| {
        let sentinel = Arc::new(());
        sink.capture_engine(&sentinel);
        sentinel
    });
    Arc::new(TestExtensionEngine {
        family: tuple.family,
        engine: tuple.engine.clone(),
        context: ExecutionContextIdentity::of::<u64>(),
        sentinel,
    })
}

fn extension_config(
    tuple: &FixedTuple,
    sink: Option<&SentinelSink>,
) -> Arc<dyn ExtensionPlanningConfig> {
    let sentinel = sink.map(|sink| {
        let sentinel = Arc::new(());
        sink.capture_config(&sentinel);
        sentinel
    });
    Arc::new(TestExtensionConfig {
        family: tuple.family,
        value: tuple.config_value,
        retained_bytes: 11,
        forced_hash: tuple.config_hash,
        sentinel,
    })
}

fn cache_owner(sink: Option<&SentinelSink>) -> Arc<dyn RuntimeCacheOwner> {
    let sentinel = sink.map(|sink| {
        let sentinel = Arc::new(());
        sink.capture_owner(&sentinel);
        sentinel
    });
    Arc::new(CountingCacheOwner { sentinel })
}

fn register_tuple(
    registrar: &mut ExtensionModuleRegistrar<'_>,
    tuple: &FixedTuple,
    sink: Option<&SentinelSink>,
) -> Result<(), ExtensionModuleError> {
    registrar.register_engine(extension_engine(tuple, sink))?;
    registrar.register_planning_config(tuple.engine.clone(), extension_config(tuple, sink))?;
    registrar.register_cache_owner(tuple.owner.clone(), cache_owner(sink))
}

fn build_runtime_with_module(
    module: Arc<dyn ExtensionModule>,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.install_extension_module(module)?;
    builder.build()
}

#[test]
fn install_absent_adds_module_and_sorts_slots() -> Result<(), Box<dyn StdError>> {
    let first = fixed("a", "tenferro.family.z", "z");
    let second = fixed("b", "tenferro.family.a", "a");
    let mut builder = Runtime::builder();
    builder.install_extension_module(module(
        "tenferro.module.z",
        ModuleAction::Register(first.clone()),
    ))?;
    builder.install_extension_module(module(
        "tenferro.module.a",
        ModuleAction::Register(second.clone()),
    ))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;

    assert_eq!(snapshot.extension_module_count(), 2);
    let slots: Vec<_> = snapshot
        .extension_slots_for_test()
        .map(|(module_id, family, engine_id, _identity)| {
            (
                module_id.as_str().to_owned(),
                family,
                engine_id.as_str().to_owned(),
            )
        })
        .collect();
    assert_eq!(
        slots,
        vec![
            (
                "tenferro.module.a".to_owned(),
                "tenferro.family.a",
                "tenferro.engine.a".to_owned(),
            ),
            (
                "tenferro.module.z".to_owned(),
                "tenferro.family.z",
                "tenferro.engine.z".to_owned(),
            ),
        ]
    );

    Ok(())
}

#[test]
fn snapshot_exact_extension_engine_query_distinguishes_family_and_engine(
) -> Result<(), Box<dyn StdError>> {
    let tuple = fixed(
        "snapshot-query",
        "tenferro.family.snapshot-query",
        "snapshot-query",
    );
    let runtime = build_runtime_with_module(module(
        "tenferro.module.snapshot-query",
        ModuleAction::Register(tuple.clone()),
    ))?;
    let snapshot = runtime.snapshot()?;
    assert!(snapshot.has_extension_engine(tuple.family, &tuple.engine));
    assert!(!snapshot.has_extension_engine("tenferro.family.other", &tuple.engine,));
    assert!(!snapshot.has_extension_engine(tuple.family, &engine_id("tenferro.engine.other"),));
    Ok(())
}

#[test]
fn install_same_module_arc_is_noop_and_preserves_epoch_and_identities(
) -> Result<(), Box<dyn StdError>> {
    let tuple = fixed("same", "tenferro.family.same", "same");
    let module = module(
        "tenferro.module.same",
        ModuleAction::Register(tuple.clone()),
    );
    let runtime = build_runtime_with_module(Arc::clone(&module))?;
    let before = runtime.snapshot()?;
    let before_identity = before
        .extension_slot_identity_for_test(tuple.family, &tuple.engine)
        .expect("extension identity");

    let returned = runtime.reconfigure(|edit| {
        edit.install_extension_module(module)?;
        Ok(())
    })?;

    assert_eq!(returned, before.epoch());
    let after = runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(
        after.extension_slot_identity_for_test(tuple.family, &tuple.engine),
        Some(before_identity)
    );

    Ok(())
}

#[test]
fn ensure_same_module_id_and_target_registration_is_noop_for_fresh_module_arc(
) -> Result<(), Box<dyn StdError>> {
    let tuple = fixed("ensure", "tenferro.family.ensure", "ensure");
    let runtime = build_runtime_with_module(module(
        "tenferro.module.ensure",
        ModuleAction::Register(tuple.clone()),
    ))?;
    let before = runtime.snapshot()?;
    let before_identity = before
        .extension_slot_identity_for_test(tuple.family, &tuple.engine)
        .expect("extension identity");

    let returned = runtime.reconfigure(|edit| {
        edit.ensure_extension_module_for_engine(
            module(
                "tenferro.module.ensure",
                ModuleAction::DuplicateEngine(tuple.clone()),
            ),
            tuple.family,
            &tuple.engine,
        )?;
        Ok(())
    })?;

    assert_eq!(returned, before.epoch());
    let after = runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(
        after.extension_slot_identity_for_test(tuple.family, &tuple.engine),
        Some(before_identity)
    );

    Ok(())
}

#[test]
fn ensure_same_module_id_missing_target_replaces_with_valid_module_and_advances_epoch(
) -> Result<(), Box<dyn StdError>> {
    let existing = fixed(
        "ensure-missing-existing",
        "tenferro.family.ensure-missing-existing",
        "ensure-missing-existing",
    );
    let requested = fixed(
        "ensure-missing-requested",
        "tenferro.family.ensure-missing-requested",
        "ensure-missing-requested",
    );
    let runtime = build_runtime_with_module(module(
        "tenferro.module.ensure-missing",
        ModuleAction::Register(existing.clone()),
    ))?;
    let before = runtime.snapshot()?;

    let returned = runtime.reconfigure(|edit| {
        edit.ensure_extension_module_for_engine(
            module(
                "tenferro.module.ensure-missing",
                ModuleAction::Register(requested.clone()),
            ),
            requested.family,
            &requested.engine,
        )?;
        Ok(())
    })?;

    let after = runtime.snapshot()?;
    assert!(returned > before.epoch());
    assert!(!Arc::ptr_eq(&before, &after));
    assert_eq!(after.epoch(), returned);
    assert_eq!(after.extension_module_count(), 1);
    assert!(after.has_extension_engine(requested.family, &requested.engine));
    assert!(!after.has_extension_engine(existing.family, &existing.engine));

    Ok(())
}

#[test]
fn ensure_same_module_id_missing_target_rejects_invalid_module_transactionally(
) -> Result<(), Box<dyn StdError>> {
    let existing = fixed(
        "ensure-invalid-existing",
        "tenferro.family.ensure-invalid-existing",
        "ensure-invalid-existing",
    );
    let requested = fixed(
        "ensure-invalid-requested",
        "tenferro.family.ensure-invalid-requested",
        "ensure-invalid-requested",
    );
    let invalid = fixed(
        "ensure-invalid-incoming",
        "tenferro.family.ensure-invalid-incoming",
        "ensure-invalid-incoming",
    );
    let runtime = build_runtime_with_module(module(
        "tenferro.module.ensure-invalid",
        ModuleAction::Register(existing.clone()),
    ))?;
    let before = runtime.snapshot()?;
    let before_epoch = before.epoch();

    let error = runtime
        .reconfigure(|edit| {
            edit.ensure_extension_module_for_engine(
                module(
                    "tenferro.module.ensure-invalid",
                    ModuleAction::Register(invalid),
                ),
                requested.family,
                &requested.engine,
            )?;
            Ok(())
        })
        .expect_err("incoming module without the requested registration must fail");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::MissingExtensionEngine {
                module_id: installed_module_id,
                family_id,
                engine_id,
            },
        } if installed_module_id == module_id("tenferro.module.ensure-invalid")
            && family_id == requested.family
            && engine_id == requested.engine
    ));
    let after = runtime.snapshot()?;
    assert_eq!(after.epoch(), before_epoch);
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(after.extension_module_count(), 1);
    assert!(after.has_extension_engine(existing.family, &existing.engine));
    assert!(!after.has_extension_engine(requested.family, &requested.engine));

    Ok(())
}

#[test]
fn install_unequal_same_id_is_conflicting_module_and_publishes_nothing(
) -> Result<(), Box<dyn StdError>> {
    let first = fixed("conflict-a", "tenferro.family.conflict-a", "conflict-a");
    let second = fixed("conflict-b", "tenferro.family.conflict-b", "conflict-b");
    let runtime = build_runtime_with_module(module(
        "tenferro.module.conflict",
        ModuleAction::Register(first),
    ))?;
    let before = runtime.snapshot()?;

    let error = runtime
        .reconfigure(|edit| {
            edit.install_extension_module(module(
                "tenferro.module.conflict",
                ModuleAction::Register(second),
            ))?;
            Ok(())
        })
        .expect_err("unequal module id should conflict");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::ExtensionModule {
                source: ExtensionModuleError::ConflictingModule { module_id: duplicate },
            },
        } if duplicate == module_id("tenferro.module.conflict")
    ));
    assert!(Arc::ptr_eq(&before, &runtime.snapshot()?));

    Ok(())
}

#[test]
fn targeted_replace_rejects_module_without_target_registration_even_when_another_module_matches(
) -> Result<(), Box<dyn StdError>> {
    let target = fixed("target-owner", "tenferro.family.targeted", "targeted");
    let unrelated = fixed("unrelated-owner", "tenferro.family.unrelated", "unrelated");
    let runtime = build_runtime_with_module(module(
        "tenferro.module.existing-target",
        ModuleAction::Register(target.clone()),
    ))?;
    let before = runtime.snapshot()?;

    let error = runtime
        .reconfigure(|edit| {
            edit.replace_extension_module_for_engine(
                module(
                    "tenferro.module.replacement",
                    ModuleAction::Register(unrelated),
                ),
                target.family,
                &target.engine,
            )?;
            Ok(())
        })
        .expect_err("the unrelated module must not be masked by another module");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::MissingExtensionEngine {
                module_id: installed_module_id,
                family_id,
                engine_id,
            },
        } if installed_module_id == module_id("tenferro.module.replacement")
            && family_id == target.family
            && engine_id == target.engine
    ));
    assert!(Arc::ptr_eq(&before, &runtime.snapshot()?));
    assert!(runtime
        .snapshot()?
        .has_extension_engine(target.family, &target.engine));
    assert_eq!(runtime.snapshot()?.extension_module_count(), 1);

    Ok(())
}

#[test]
fn targeted_replace_rolls_back_a_mismatched_module_and_preserves_the_previous_one(
) -> Result<(), Box<dyn StdError>> {
    let target = fixed(
        "targeted-before",
        "tenferro.family.targeted-replace",
        "targeted-replace",
    );
    let mismatched = fixed(
        "targeted-after",
        "tenferro.family.targeted-other",
        "targeted-other",
    );
    let runtime = build_runtime_with_module(module(
        "tenferro.module.targeted-replace",
        ModuleAction::Register(target.clone()),
    ))?;
    let before = runtime.snapshot()?;

    let error = runtime
        .reconfigure(|edit| {
            edit.replace_extension_module_for_engine(
                module(
                    "tenferro.module.targeted-replace",
                    ModuleAction::Register(mismatched),
                ),
                target.family,
                &target.engine,
            )?;
            Ok(())
        })
        .expect_err("a mismatched replacement must be rejected");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::MissingExtensionEngine {
                module_id: installed_module_id,
                ..
            },
        } if installed_module_id == module_id("tenferro.module.targeted-replace")
    ));
    let after = runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert!(after.has_extension_engine(target.family, &target.engine));
    assert_eq!(after.extension_module_count(), 1);

    Ok(())
}

#[test]
fn replace_absent_installs_validated_transaction_and_advances_epoch(
) -> Result<(), Box<dyn StdError>> {
    let tuple = fixed(
        "replace-absent",
        "tenferro.family.replace-absent",
        "replace-absent",
    );
    let runtime = Runtime::builder().build()?;
    let before = runtime.epoch()?;

    let after = runtime.reconfigure(|edit| {
        edit.replace_extension_module(module(
            "tenferro.module.replace-absent",
            ModuleAction::Register(tuple.clone()),
        ))?;
        Ok(())
    })?;

    assert!(after > before);
    assert_eq!(runtime.snapshot()?.extension_module_count(), 1);
    assert!(runtime
        .snapshot()?
        .extension_slot_identity_for_test(tuple.family, &tuple.engine)
        .is_some());

    Ok(())
}

#[test]
fn remove_absent_is_noop_and_preserves_snapshot() -> Result<(), Box<dyn StdError>> {
    let runtime = Runtime::builder().build()?;
    let before = runtime.snapshot()?;

    let returned = runtime.reconfigure(|edit| {
        edit.remove_extension_module(&module_id("tenferro.module.absent"))?;
        Ok(())
    })?;

    assert_eq!(returned, before.epoch());
    assert!(Arc::ptr_eq(&before, &runtime.snapshot()?));

    Ok(())
}

#[test]
fn replace_unequal_changes_replaced_identities_only() -> Result<(), Box<dyn StdError>> {
    let replaced_before = fixed("replace-before", "tenferro.family.replaced", "replaced");
    let stable = fixed("stable", "tenferro.family.stable", "stable");
    let mut builder = Runtime::builder();
    builder.install_extension_module(module(
        "tenferro.module.replaced",
        ModuleAction::Register(replaced_before.clone()),
    ))?;
    builder.install_extension_module(module(
        "tenferro.module.stable",
        ModuleAction::Register(stable.clone()),
    ))?;
    let runtime = builder.build()?;
    let before = runtime.snapshot()?;
    let replaced_identity = before
        .extension_slot_identity_for_test(replaced_before.family, &replaced_before.engine)
        .expect("replaced identity");
    let stable_identity = before
        .extension_slot_identity_for_test(stable.family, &stable.engine)
        .expect("stable identity");
    let replaced_after = fixed("replace-after", "tenferro.family.replaced", "replaced");

    runtime.reconfigure(|edit| {
        edit.replace_extension_module(module(
            "tenferro.module.replaced",
            ModuleAction::Register(replaced_after.clone()),
        ))?;
        Ok(())
    })?;

    let after = runtime.snapshot()?;
    assert_ne!(
        after
            .extension_slot_identity_for_test(replaced_after.family, &replaced_after.engine)
            .expect("new replaced identity"),
        replaced_identity
    );
    assert_eq!(
        after
            .extension_slot_identity_for_test(stable.family, &stable.engine)
            .expect("stable identity"),
        stable_identity
    );

    Ok(())
}

#[test]
fn duplicate_engine_config_and_owner_return_exact_extension_errors() {
    let duplicate_engine = Runtime::builder()
        .install_extension_module(module(
            "tenferro.module.dup-engine",
            ModuleAction::DuplicateEngine(fixed(
                "dup-engine",
                "tenferro.family.dup-engine",
                "dup-engine",
            )),
        ))
        .expect_err("duplicate extension engine");
    assert!(matches!(
        duplicate_engine,
        RuntimeConfigError::ExtensionModule {
            source: ExtensionModuleError::ConflictingEngine {
                module_id: duplicate_module,
                family_id: "tenferro.family.dup-engine",
                engine_id: duplicate_engine_id,
            },
        } if duplicate_module == module_id("tenferro.module.dup-engine")
            && duplicate_engine_id == engine_id("tenferro.engine.dup-engine")
    ));

    let mut conflicting_config = fixed("dup-config", "tenferro.family.dup-config", "dup-config");
    conflicting_config.config_hash = Some(99);
    let duplicate_config = Runtime::builder()
        .install_extension_module(module(
            "tenferro.module.dup-config",
            ModuleAction::DuplicateConflictingConfig(conflicting_config),
        ))
        .expect_err("duplicate planning config");
    assert!(matches!(
        duplicate_config,
        RuntimeConfigError::ExtensionModule {
            source: ExtensionModuleError::ConflictingPlanningConfig {
                module_id: duplicate_module,
                engine_id: duplicate_engine_id,
            },
        } if duplicate_module == module_id("tenferro.module.dup-config")
            && duplicate_engine_id == engine_id("tenferro.engine.dup-config")
    ));

    let duplicate_owner = Runtime::builder()
        .install_extension_module(module(
            "tenferro.module.dup-owner",
            ModuleAction::DuplicateOwner(fixed(
                "dup-owner",
                "tenferro.family.dup-owner",
                "dup-owner",
            )),
        ))
        .expect_err("duplicate cache owner");
    assert!(matches!(
        duplicate_owner,
        RuntimeConfigError::ExtensionModule {
            source: ExtensionModuleError::ConflictingCacheOwner { module_id: duplicate_module, owner },
        } if duplicate_module == module_id("tenferro.module.dup-owner")
            && owner == owner_id("tenferro.owner.dup-owner")
    ));
}

#[test]
fn equal_config_hash_and_payload_is_noop() -> Result<(), Box<dyn StdError>> {
    let mut tuple = fixed(
        "equal-config",
        "tenferro.family.equal-config",
        "equal-config",
    );
    tuple.config_hash = Some(123);

    let runtime = build_runtime_with_module(module(
        "tenferro.module.equal-config",
        ModuleAction::DuplicateEqualConfig(tuple.clone()),
    ))?;

    let snapshot = runtime.snapshot()?;
    assert_eq!(snapshot.extension_module_count(), 1);
    assert!(snapshot
        .extension_slot_identity_for_test(tuple.family, &tuple.engine)
        .is_some());

    Ok(())
}

#[test]
fn configure_error_releases_transaction_values() {
    let tuple = fixed("release", "tenferro.family.release", "release");
    let sink = SentinelSink::default();

    let error = Runtime::builder()
        .install_extension_module(module(
            "tenferro.module.release",
            ModuleAction::FailAfterRegisterWithSentinels(tuple, sink.clone()),
        ))
        .expect_err("configure error");

    assert!(matches!(
        error,
        RuntimeConfigError::ExtensionModule {
            source: ExtensionModuleError::ConflictingCacheOwner { .. },
        }
    ));
    assert!(
        sink.all_released(),
        "failed transaction should release engine/config/owner values"
    );
}

#[test]
fn duplicate_family_engine_across_modules_is_conflicting_registration(
) -> Result<(), Box<dyn StdError>> {
    let first = fixed("cross-a", "tenferro.family.cross", "cross");
    let second = FixedTuple {
        owner: owner_id("tenferro.owner.cross-b"),
        ..first.clone()
    };
    let runtime = build_runtime_with_module(module(
        "tenferro.module.cross-a",
        ModuleAction::Register(first.clone()),
    ))?;
    let before = runtime.snapshot()?;

    let error = runtime
        .reconfigure(|edit| {
            edit.install_extension_module(module(
                "tenferro.module.cross-b",
                ModuleAction::Register(second),
            ))?;
            Ok(())
        })
        .expect_err("duplicate family/engine across modules");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::ConflictingRegistration {
                key: RegistrationKey::ExtensionEngine { family, engine },
            },
        } if family == "tenferro.family.cross" && engine == engine_id("tenferro.engine.cross")
    ));
    assert!(Arc::ptr_eq(&before, &runtime.snapshot()?));

    Ok(())
}

#[test]
fn extension_prepare_request_preserves_exact_borrowed_inputs() -> Result<(), Box<dyn StdError>> {
    let tuple = fixed("request", "tenferro.family.request", "request");
    let runtime = build_runtime_with_module(module(
        "tenferro.module.request",
        ModuleAction::Register(tuple.clone()),
    ))?;
    let snapshot = runtime.snapshot()?;
    let (_module, _family, _engine_id, identity, engine, config) = snapshot
        .extension_slot_full_for_test(tuple.family, &tuple.engine)
        .expect("extension slot");

    let placement = ResolvedProgramPlacement::new(
        tuple.engine.clone(),
        StorageClass::new("tenferro.storage.host")?,
    );
    let planning = ResolvedPlanningConfig::resolve(
        &ExecutionPolicy::new(Determinism::Fast, None, 17),
        &PrepareOptions::new().with_planning_seed(17),
        HardwareClassId::new("tenferro.cpu.host")?,
    );
    let options_key = PrepareOptionsKey::from_resolved(placement.clone(), None, 17);
    let binding = PreparedOperationBinding::new(
        runtime.id(),
        snapshot.epoch(),
        tuple.engine.clone(),
        identity,
        ExecutionContextIdentity::of::<u64>(),
        HardwareClassId::new("tenferro.cpu.host")?,
    );
    let inputs = InputSignature::new(Vec::new());
    let specialization = SpecializationRequirements::polymorphic(0).project(&inputs)?;
    let operation = TestOp {
        family: tuple.family,
    };

    let request = ExtensionPrepareRequest::new(
        &operation,
        &binding,
        &placement,
        binding.hardware_class(),
        &planning,
        config.as_ref(),
        &inputs,
        &options_key,
        &specialization,
    );
    let result = engine.prepare(request)?;

    assert!(matches!(
        result,
        PrepareCapability::Unsupported(UnsupportedReason::Operation {
            operation: "test-extension"
        })
    ));

    Ok(())
}

#[test]
fn frozen_extension_lookup_has_no_payload_hash_lock_or_downcast() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/runtime/extension.rs"
    ))
    .expect("extension source");
    let lookup = source
        .split_once("fn extension_engine_slot")
        .and_then(|(_, rest)| rest.split_once("impl fmt::Debug for FrozenExtensionSlots"))
        .map(|(body, _)| body)
        .expect("extension lookup helper should precede FrozenExtensionSlots Debug");

    for forbidden in ["payload_hash", "payload_eq", "downcast", "Mutex", ".lock("] {
        assert!(
            !lookup.contains(forbidden),
            "frozen extension lookup must not perform {forbidden}"
        );
    }
}

#[test]
fn extension_config_payload_hash_and_equality_are_both_checked_for_duplicates() {
    let mut first = fixed("hash", "tenferro.family.hash", "hash");
    first.config_hash = Some(5);
    let first_config = extension_config(&first, None);
    let equal_config = extension_config(&first, None);
    let mut hasher = DefaultHasher::new();
    first_config.payload_hash(&mut hasher);
    let first_hash = hasher.finish();
    let mut hasher = DefaultHasher::new();
    equal_config.payload_hash(&mut hasher);
    assert_eq!(first_hash, hasher.finish());
    assert!(first_config.payload_eq(equal_config.as_ref()));

    let different = Arc::new(TestExtensionConfig {
        family: first.family,
        value: first.config_value + 1,
        retained_bytes: 0,
        forced_hash: first.config_hash,
        sentinel: None,
    });
    let mut hasher = DefaultHasher::new();
    different.payload_hash(&mut hasher);
    assert_eq!(first_hash, hasher.finish());
    assert!(!first_config.payload_eq(different.as_ref()));
}
