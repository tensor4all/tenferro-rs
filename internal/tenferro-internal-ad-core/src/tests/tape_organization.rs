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
fn tape_frontend_forwards_to_internal_ad_core_after_split() {
    assert!(
        !workspace_path("tenferro/src/reverse_tape.rs").exists(),
        "reverse_tape.rs should stay removed after the tape/ split"
    );

    assert!(
        workspace_path("tenferro/src/tape/mod.rs").exists(),
        "tenferro should keep only a thin tape module shim"
    );
    for relative in [
        "tenferro/src/tape/registry.rs",
        "tenferro/src/tape/tensor_pullback.rs",
        "tenferro/src/tape/scalar_pullback.rs",
        "tenferro/src/tape/tests",
    ] {
        assert!(
            !workspace_path(relative).exists(),
            "old local tape implementation should be removed: {relative}"
        );
    }

    for relative in [
        "internal/tenferro-internal-ad-core/src/registry.rs",
        "internal/tenferro-internal-ad-core/src/tape.rs",
        "internal/tenferro-internal-ad-core/src/tests/tape_frontend.rs",
    ] {
        assert!(
            workspace_path(relative).exists(),
            "expected internal tape implementation to exist: {relative}"
        );
    }
}

#[test]
fn internal_tape_modules_stay_under_size_guideline() {
    for relative in [
        "internal/tenferro-internal-ad-core/src/registry.rs",
        "internal/tenferro-internal-ad-core/src/tape.rs",
    ] {
        let contents = std::fs::read_to_string(workspace_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

#[test]
fn tape_registry_uses_chainrules_tape_directly() {
    let registry = std::fs::read_to_string(workspace_path(
        "internal/tenferro-internal-ad-core/src/registry.rs",
    ))
    .unwrap();
    assert!(
        registry.contains("Tape<DynTensor>"),
        "tape registry should register rules against tidu::expert::Tape<DynTensor>"
    );
    assert!(
        registry.contains("tape.attach_rule"),
        "tape registry should attach reverse rules directly to the chainrules tape"
    );
    assert!(
        !registry.contains("thread_local!"),
        "tape registry should not reintroduce a frontend-local thread-local store"
    );
}
