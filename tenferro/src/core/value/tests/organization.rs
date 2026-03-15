use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn ad_values_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/ad_value.rs").exists(),
        "ad_value.rs should stay removed after the core/value split"
    );

    for relative in [
        "src/core/value/mod.rs",
        "src/core/value/core.rs",
        "src/core/value/tensor.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split ad_value module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_value_modules_stay_under_size_guideline() {
    for relative in ["src/core/value/core.rs", "src/core/value/tensor.rs"] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
