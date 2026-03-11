use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn ad_value_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/ad_value.rs").exists(),
        "ad_value.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/ad_value/mod.rs",
        "src/ad_value/core.rs",
        "src/ad_value/tensor.rs",
        "src/ad_value/scalar/mod.rs",
        "src/ad_value/scalar/shared.rs",
        "src/ad_value/scalar/unary.rs",
        "src/ad_value/scalar/binary.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split ad_value module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_value_modules_stay_under_size_guideline() {
    for relative in [
        "src/ad_value/core.rs",
        "src/ad_value/tensor.rs",
        "src/ad_value/scalar/mod.rs",
        "src/ad_value/scalar/shared.rs",
        "src/ad_value/scalar/unary.rs",
        "src/ad_value/scalar/binary.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
