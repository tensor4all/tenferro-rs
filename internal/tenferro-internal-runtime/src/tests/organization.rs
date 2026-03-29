use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("internal crate should live under <repo>/internal/")
        .to_path_buf()
}

fn workspace_path(path: &str) -> PathBuf {
    workspace_root().join(path)
}

#[test]
fn lib_rs_does_not_reexport_generic_global_context_helpers() {
    let lib = std::fs::read_to_string(workspace_path("tenferro/src/lib.rs")).unwrap();
    assert!(
        !lib.contains("set_global_context")
            && !lib.contains("with_global_context")
            && !lib.contains("try_with_global_context"),
        "tenferro public API should not re-export generic global context helpers"
    );
}

#[test]
fn runtime_context_storage_is_owned_by_the_internal_runtime_crate() {
    let context =
        std::fs::read_to_string(workspace_path("tenferro/src/runtime/context.rs")).unwrap();
    let internal_context = std::fs::read_to_string(workspace_path(
        "internal/tenferro-internal-runtime/src/context.rs",
    ))
    .unwrap();
    assert!(
        context.contains("tenferro_internal_runtime"),
        "tenferro runtime context module should delegate to the internal runtime crate"
    );
    assert!(
        !context.contains("GLOBAL_CONTEXTS"),
        "tenferro runtime context shim should not keep the old generic global context map"
    );
    assert!(
        !context.contains("DEFAULT_RUNTIME"),
        "tenferro runtime context shim should not keep local default runtime storage"
    );
    assert!(
        internal_context.contains("DEFAULT_RUNTIME"),
        "internal runtime crate should own the default runtime storage slot"
    );
    assert!(
        !internal_context.contains("GLOBAL_CONTEXTS"),
        "internal runtime crate should not keep the old generic global context map"
    );
}
