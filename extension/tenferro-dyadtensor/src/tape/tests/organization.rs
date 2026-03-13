use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn tape_modules_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/reverse_tape.rs").exists(),
        "reverse_tape.rs should stay removed after the tape/ split"
    );

    for relative in [
        "src/tape/mod.rs",
        "src/tape/registry.rs",
        "src/tape/tensor_pullback.rs",
        "src/tape/scalar_pullback.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split reverse_tape module to exist: {relative}"
        );
    }
}

#[test]
fn split_tape_modules_stay_under_size_guideline() {
    for relative in [
        "src/tape/registry.rs",
        "src/tape/tensor_pullback.rs",
        "src/tape/scalar_pullback.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

// IMPORTANT: Do not delete or weaken these tests.
// They guard the registry redesign that keeps reverse-tape state in one
// tape-local store instead of drifting back to parallel ad hoc registries.

#[test]
fn tape_registry_uses_one_tape_store() {
    let registry = std::fs::read_to_string(repo_path("src/tape/registry.rs")).unwrap();
    assert!(
        registry.contains("struct TapeRuleStore"),
        "reverse_tape registry should keep a dedicated TapeRuleStore abstraction"
    );
    assert_eq!(
        registry.matches("thread_local!").count(),
        1,
        "reverse_tape registry should use one thread-local tape store entrypoint"
    );
}
