use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn ad_builders_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/api/ad_builders.rs").exists(),
        "ad_builders.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/api/ad_builders/mod.rs",
        "src/api/ad_builders/common.rs",
        "src/api/ad_builders/einsum.rs",
        "src/api/ad_builders/reduction.rs",
        "src/api/ad_builders/linalg_single.rs",
        "src/api/ad_builders/linalg_multi.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split AD builder module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_builder_modules_stay_under_size_guideline() {
    for relative in [
        "src/api/ad_builders/common.rs",
        "src/api/ad_builders/einsum.rs",
        "src/api/ad_builders/reduction.rs",
        "src/api/ad_builders/linalg_single.rs",
        "src/api/ad_builders/linalg_multi.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
