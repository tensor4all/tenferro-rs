use std::path::PathBuf;

use tenferro_runtime::{
    CoreCapabilityBundle, CoreCapabilityKind, DotGeneralPreparation, ElementwiseRuntime,
    IndexingRuntime, LayoutRuntime, PreparedOperation, PreparedOperationHandle, ReductionRuntime,
    UnsupportedReason,
};

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

#[test]
fn prepared_operation_has_metadata_only_contract() {
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
    ] {
        assert!(
            trait_body.contains(required),
            "PreparedOperation missing required metadata method {required}"
        );
    }

    for forbidden in [
        "execute", "Tensor", "Runtime", "lease", "event", "schedule", "buffer", "scratch",
    ] {
        assert!(
            !trait_body.contains(forbidden),
            "PreparedOperation must not expose {forbidden}"
        );
    }
}
