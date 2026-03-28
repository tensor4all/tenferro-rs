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
fn ad_values_live_in_internal_ad_core_after_split() {
    assert!(
        !workspace_path("tenferro/src/ad_value.rs").exists(),
        "ad_value.rs should stay removed after the core/value split"
    );

    assert!(
        workspace_path("tenferro/src/core/value/mod.rs").exists(),
        "tenferro should keep only a thin core/value module shim"
    );
    for relative in [
        "tenferro/src/core/value/core.rs",
        "tenferro/src/core/value/tensor.rs",
        "tenferro/src/core/value/tensor/accessors.rs",
        "tenferro/src/core/value/tensor/placement.rs",
        "tenferro/src/core/value/tensor/reverse_api.rs",
        "tenferro/src/core/value/tests",
    ] {
        assert!(
            !workspace_path(relative).exists(),
            "old local AD core implementation should be removed: {relative}"
        );
    }

    for relative in [
        "internal/tenferro-internal-ad-core/src/core.rs",
        "internal/tenferro-internal-ad-core/src/tensor.rs",
        "internal/tenferro-internal-ad-core/src/registry.rs",
        "internal/tenferro-internal-ad-core/src/tape.rs",
        "internal/tenferro-internal-ad-core/src/tests/core_value.rs",
        "internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs",
    ] {
        assert!(
            workspace_path(relative).exists(),
            "expected internal AD core implementation to exist: {relative}"
        );
    }
}

#[test]
fn internal_ad_core_modules_stay_under_size_guideline() {
    for relative in [
        "internal/tenferro-internal-ad-core/src/core.rs",
        "internal/tenferro-internal-ad-core/src/tensor.rs",
        "internal/tenferro-internal-ad-core/src/registry.rs",
        "internal/tenferro-internal-ad-core/src/tape.rs",
    ] {
        let contents = std::fs::read_to_string(workspace_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 700,
            "{relative} should stay focused after the split; found {line_count} lines"
        );
    }
}
