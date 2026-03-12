use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

// IMPORTANT: Do not delete or weaken these structure tests.
// They protect the tensor foundation split so future tensor features do not
// collapse back into a single monolithic lib.rs.

#[test]
fn tensor_foundation_is_split_into_focused_modules() {
    for relative in [
        "src/layout.rs",
        "src/buffer.rs",
        "src/completion_event.rs",
        "src/tensor/mod.rs",
        "src/tensor/constructors.rs",
        "src/tensor/metadata.rs",
        "src/tensor/views.rs",
        "src/tensor/data_ops.rs",
        "src/tensor/transfer.rs",
        "src/tensor/autodiff.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split tensor module to exist: {relative}"
        );
    }
}

#[test]
fn tensor_modules_stay_under_size_guideline() {
    for relative in [
        "src/lib.rs",
        "src/layout.rs",
        "src/buffer.rs",
        "src/completion_event.rs",
        "src/tensor/mod.rs",
        "src/tensor/constructors.rs",
        "src/tensor/metadata.rs",
        "src/tensor/views.rs",
        "src/tensor/data_ops.rs",
        "src/tensor/transfer.rs",
        "src/tensor/autodiff.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
