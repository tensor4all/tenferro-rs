use super::super::{
    CacheInFlightBehavior, Determinism, EngineId, ExecutionPolicy, HardwareClassId, PrepareOptions,
    PrepareOptionsKey, ResolvedPlanningConfig, ResolvedPlanningKey, ResolvedProgramPlacement,
    StorageClass,
};

fn engine(value: &str) -> EngineId {
    EngineId::new(value).unwrap()
}

fn storage(value: &str) -> StorageClass {
    StorageClass::new(value).unwrap()
}

fn hardware(value: &str) -> HardwareClassId {
    HardwareClassId::new(value).unwrap()
}

fn placement(engine_id: &str, storage_class: &str) -> ResolvedProgramPlacement {
    ResolvedProgramPlacement::new(engine(engine_id), storage(storage_class))
}

fn options_key(
    engine_id: &str,
    storage_class: &str,
    workspace: Option<usize>,
    seed: u64,
) -> PrepareOptionsKey {
    PrepareOptionsKey::from_resolved(placement(engine_id, storage_class), workspace, seed)
}

fn planning_key(
    determinism: Determinism,
    workspace: Option<usize>,
    seed: u64,
    hardware_class: &str,
) -> ResolvedPlanningKey {
    let policy = ExecutionPolicy::new(determinism, workspace, seed);
    let config =
        ResolvedPlanningConfig::resolve(&policy, &PrepareOptions::new(), hardware(hardware_class));
    ResolvedPlanningKey::from_config(&config)
}

#[test]
fn policy_normalized_option_key_compares_each_resolved_field() {
    let base = options_key("tenferro.cpu", "tenferro.storage.host", Some(1), 2);
    let same = options_key("tenferro.cpu", "tenferro.storage.host", Some(1), 2);

    assert_eq!(base, same);
    assert_eq!(
        base.resolved_placement().engine_id().as_str(),
        "tenferro.cpu"
    );
    assert_eq!(
        base.resolved_placement().storage_class().as_str(),
        "tenferro.storage.host"
    );
    assert_eq!(base.hard_workspace_limit_bytes(), Some(1));
    assert_eq!(base.planning_seed(), 2);

    assert_ne!(
        base,
        options_key("tenferro.gpu", "tenferro.storage.host", Some(1), 2)
    );
    assert_ne!(
        base,
        options_key("tenferro.cpu", "tenferro.storage.device", Some(1), 2)
    );
    assert_ne!(
        base,
        options_key("tenferro.cpu", "tenferro.storage.host", None, 2)
    );
    assert_ne!(
        base,
        options_key("tenferro.cpu", "tenferro.storage.host", Some(3), 2)
    );
    assert_ne!(
        base,
        options_key("tenferro.cpu", "tenferro.storage.host", Some(1), 4)
    );
}

#[test]
fn policy_normalized_planning_key_compares_each_resolved_field() {
    let base = planning_key(Determinism::Fast, Some(8), 9, "tenferro.cpu");
    let same = planning_key(Determinism::Fast, Some(8), 9, "tenferro.cpu");

    assert_eq!(base, same);
    assert_eq!(base.determinism(), Determinism::Fast);
    assert_eq!(base.hard_workspace_limit_bytes(), Some(8));
    assert_eq!(base.planning_seed(), 9);
    assert_eq!(base.hardware_class().as_str(), "tenferro.cpu");

    assert_ne!(
        base,
        planning_key(Determinism::Reproducible, Some(8), 9, "tenferro.cpu")
    );
    assert_ne!(
        base,
        planning_key(Determinism::Fast, None, 9, "tenferro.cpu")
    );
    assert_ne!(
        base,
        planning_key(Determinism::Fast, Some(10), 9, "tenferro.cpu")
    );
    assert_ne!(
        base,
        planning_key(Determinism::Fast, Some(8), 10, "tenferro.cpu")
    );
    assert_ne!(
        base,
        planning_key(Determinism::Fast, Some(8), 9, "tenferro.gpu")
    );
}

#[test]
fn policy_cache_in_flight_never_changes_normalized_key_identity() {
    let policy = ExecutionPolicy::new(Determinism::Fast, Some(8), 9);
    let wait = PrepareOptions::new().with_cache_in_flight(CacheInFlightBehavior::Wait);
    let refuse = PrepareOptions::new().with_cache_in_flight(CacheInFlightBehavior::Refuse);

    let wait_config = ResolvedPlanningConfig::resolve(&policy, &wait, hardware("tenferro.cpu"));
    let refuse_config = ResolvedPlanningConfig::resolve(&policy, &refuse, hardware("tenferro.cpu"));

    assert_eq!(
        ResolvedPlanningKey::from_config(&wait_config),
        ResolvedPlanningKey::from_config(&refuse_config)
    );
    assert_eq!(
        PrepareOptionsKey::from_resolved(
            placement("tenferro.cpu", "tenferro.storage.host"),
            wait_config.hard_workspace_limit_bytes(),
            wait_config.planning_seed(),
        ),
        PrepareOptionsKey::from_resolved(
            placement("tenferro.cpu", "tenferro.storage.host"),
            refuse_config.hard_workspace_limit_bytes(),
            refuse_config.planning_seed(),
        )
    );
}
