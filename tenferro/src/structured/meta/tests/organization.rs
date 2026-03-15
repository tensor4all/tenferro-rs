use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn structured_meta_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/structured/meta.rs").exists(),
        "structured/meta.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/structured/meta/mod.rs",
        "src/structured/meta/types.rs",
        "src/structured/meta/planning.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split structured meta module to exist: {relative}"
        );
    }
}

#[test]
fn split_structured_meta_modules_stay_under_size_guideline() {
    for relative in [
        "src/structured/meta/types.rs",
        "src/structured/meta/planning.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
