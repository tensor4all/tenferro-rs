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
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split reverse_tape module to exist: {relative}"
        );
    }
    assert!(
        !repo_path("src/tape/scalar_pullback.rs").exists(),
        "scalar_pullback.rs should stay removed after the homogeneous-tape redesign"
    );
}

#[test]
fn split_tape_modules_stay_under_size_guideline() {
    for relative in ["src/tape/registry.rs", "src/tape/tensor_pullback.rs"] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

// IMPORTANT: Do not delete or weaken these tests.
// They guard the homogeneous-tape redesign: dyadtensor should register reverse
// rules directly on chainrules::Tape<DynTensor> instead of drifting back to a
// dyadtensor-local registry/store layer.

#[test]
fn tape_registry_uses_chainrules_tape_directly() {
    let registry = std::fs::read_to_string(repo_path("src/tape/registry.rs")).unwrap();
    assert!(
        registry.contains("Tape<DynTensor>"),
        "tape registry should register rules against chainrules::Tape<DynTensor>"
    );
    assert!(
        registry.contains("tape.attach_rule"),
        "tape registry should attach reverse rules directly to the chainrules tape"
    );
    assert!(
        !registry.contains("thread_local!"),
        "tape registry should not reintroduce a dyadtensor-local thread-local store"
    );
}
