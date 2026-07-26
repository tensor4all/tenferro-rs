use std::path::PathBuf;

use tenferro_runtime::{
    CoreCapabilityBundle, CoreCapabilityKind, DotGeneralPreparation, ElementwiseRuntime,
    EngineRegistration, EngineSnapshotView, ExtensionEngine, ExtensionModule, ExtensionModuleError,
    ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest,
    IndexingRuntime, LayoutRuntime, PreparedOperation, PreparedOperationHandle, ReductionRuntime,
    Runtime, RuntimeConfigBuilder, RuntimeConfigSnapshot, RuntimeReconfiguration,
    UnsupportedReason,
};

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

#[test]
fn runtime_snapshot_api_is_public_debug_and_metadata_only() {
    fn takes_runtime(_: Option<&Runtime>) {}
    fn takes_builder(_: Option<&RuntimeConfigBuilder>) {}
    fn takes_snapshot(_: Option<&RuntimeConfigSnapshot>) {}
    fn takes_reconfigure(_: Option<&mut RuntimeReconfiguration<'_>>) {}
    fn takes_registration(_: Option<&EngineRegistration>) {}
    fn takes_view(_: Option<&EngineSnapshotView<'_>>) {}

    let _ = (
        takes_runtime,
        takes_builder,
        takes_snapshot,
        takes_reconfigure,
        takes_registration,
        takes_view,
        Runtime::builder,
        RuntimeConfigBuilder::new,
    );

    let source = repo_file("crates/tenferro-runtime/src/runtime/snapshot.rs");
    let snapshot_body = source
        .split_once("impl RuntimeConfigSnapshot")
        .and_then(|(_, rest)| rest.split_once("impl fmt::Debug for RuntimeConfigSnapshot"))
        .map(|(body, _)| body)
        .expect("RuntimeConfigSnapshot impl should precede its Debug impl");

    for forbidden in [
        "pub fn builder",
        "pub fn build",
        "issuer(",
        "registration_identity(",
    ] {
        assert!(
            !snapshot_body.contains(forbidden),
            "RuntimeConfigSnapshot must not expose {forbidden}"
        );
    }
}

#[test]
fn runtime_extension_module_api_is_public_debug_and_transactional() {
    fn takes_module_id(_: Option<&ExtensionModuleId>) {}
    fn takes_module(_: Option<&dyn ExtensionModule>) {}
    fn takes_registrar(_: Option<&mut ExtensionModuleRegistrar<'_>>) {}
    fn takes_module_error(_: Option<&ExtensionModuleError>) {}
    fn takes_extension_engine(_: Option<&dyn ExtensionEngine>) {}
    fn takes_extension_config(_: Option<&dyn ExtensionPlanningConfig>) {}
    fn takes_extension_request(_: Option<&ExtensionPrepareRequest<'_>>) {}

    let _ = (
        takes_module_id,
        takes_module,
        takes_registrar,
        takes_module_error,
        takes_extension_engine,
        takes_extension_config,
        takes_extension_request,
    );
    assert!(ExtensionModuleId::new("tenferro.module.api").is_ok());

    let source = repo_file("crates/tenferro-runtime/src/runtime/extension.rs");
    let registrar_body = source
        .split_once("pub struct ExtensionModuleRegistrar")
        .and_then(|(_, rest)| rest.split_once("impl fmt::Debug for ExtensionModuleRegistrar"))
        .map(|(body, _)| body)
        .expect("registrar should have bounded Debug impl");

    for forbidden in ["static mut", "thread_local!", "execute(", "PreparedGraph"] {
        assert!(
            !registrar_body.contains(forbidden),
            "extension registrar must not expose {forbidden}"
        );
    }
}

#[test]
fn prepared_operation_has_bounded_execution_contract() {
    fn takes_prepared_operation(_: Option<&dyn PreparedOperation>) {}
    fn takes_prepared_handle(_: Option<&PreparedOperationHandle>) {}
    fn takes_core_traits(
        _: Option<&dyn ElementwiseRuntime>,
        _: Option<&dyn ReductionRuntime>,
        _: Option<&dyn IndexingRuntime>,
        _: Option<&dyn DotGeneralPreparation>,
        _: Option<&dyn LayoutRuntime>,
    ) {
    }

    let _ = (
        takes_prepared_operation,
        takes_prepared_handle,
        takes_core_traits,
        CoreCapabilityBundle::builder,
        CoreCapabilityKind::Elementwise,
        UnsupportedReason::MissingCapability {
            capability: CoreCapabilityKind::Elementwise,
        },
    );

    let source = repo_file("crates/tenferro-runtime/src/runtime/capability.rs");
    let trait_body = source
        .split_once("pub trait PreparedOperation")
        .and_then(|(_, rest)| rest.split_once("pub type PreparedOperationHandle"))
        .map(|(body, _)| body)
        .expect("PreparedOperation trait should precede handle alias");

    for required in [
        "fn binding(&self) -> &PreparedOperationBinding",
        "fn specialization(&self) -> &SpecializationProjection",
        "fn retained_bytes(&self) -> usize",
        "fn execute(",
        "&mut ErasedExecutionContext<'_>",
        "&mut ExtensionCacheStore",
        "&[TensorRead<'_>]",
    ] {
        assert!(
            trait_body.contains(required),
            "PreparedOperation missing required method/signature fragment {required}"
        );
    }

    for forbidden in ["Runtime", "lease", "event", "schedule", "buffer", "scratch"] {
        assert!(
            !trait_body.contains(forbidden),
            "PreparedOperation must not expose {forbidden}"
        );
    }
}
