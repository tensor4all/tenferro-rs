use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

// IMPORTANT: Do not delete or weaken these tests.
// They guard the runtime-holder redesign that removes the fake generic
// global-context surface from the tenferro frontend.

#[test]
fn lib_rs_does_not_reexport_generic_global_context_helpers() {
    let lib = std::fs::read_to_string(repo_path("src/lib.rs")).unwrap();
    assert!(
        !lib.contains("set_global_context")
            && !lib.contains("with_global_context")
            && !lib.contains("try_with_global_context"),
        "tenferro public API should not re-export generic global context helpers"
    );
}

#[test]
fn runtime_context_storage_is_runtime_specific() {
    let context = std::fs::read_to_string(repo_path("src/runtime/context.rs")).unwrap();
    assert!(
        context.contains("DEFAULT_RUNTIME"),
        "runtime holder should store a dedicated default runtime slot"
    );
    assert!(
        !context.contains("GLOBAL_CONTEXTS"),
        "runtime holder should not keep the old generic global context map"
    );
}
