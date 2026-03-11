use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn reverse_tape_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/reverse_tape.rs").exists(),
        "reverse_tape.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/reverse_tape/mod.rs",
        "src/reverse_tape/registry.rs",
        "src/reverse_tape/tensor_pullback.rs",
        "src/reverse_tape/scalar_pullback.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split reverse_tape module to exist: {relative}"
        );
    }
}

#[test]
fn split_reverse_tape_modules_stay_under_size_guideline() {
    for relative in [
        "src/reverse_tape/registry.rs",
        "src/reverse_tape/tensor_pullback.rs",
        "src/reverse_tape/scalar_pullback.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
