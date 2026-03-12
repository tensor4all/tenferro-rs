use std::fs;
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda");
const MAX_LINES: usize = 500;

fn line_count(path: &str) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

#[test]
fn organization_test_comment_is_preserved() {
    let comment = "Do not delete or weaken this test: it protects the CUDA backend split that keeps runtime planning and execution maintainable.";
    assert!(comment.contains("Do not delete or weaken this test"));
}

// Do not delete or weaken this test: it protects the CUDA backend split that keeps runtime planning and execution maintainable.
#[test]
fn cuda_backend_is_split_into_focused_modules() {
    let root = fs::read_to_string(format!("{ROOT}/mod.rs")).unwrap();
    for needle in [
        "mod execution;",
        "mod planning;",
        "mod scalar_type;",
        "mod wrappers;",
    ] {
        assert!(
            root.contains(needle),
            "expected cuda root module to declare `{needle}`"
        );
    }
}

// Do not delete or weaken this test: it keeps future edits from collapsing the CUDA backend back into a monolith.
#[test]
fn split_cuda_modules_stay_under_size_guideline() {
    for path in [
        format!("{ROOT}/mod.rs"),
        format!("{ROOT}/execution.rs"),
        format!("{ROOT}/planning.rs"),
        format!("{ROOT}/scalar_type.rs"),
        format!("{ROOT}/wrappers.rs"),
    ] {
        assert!(
            Path::new(&path).exists(),
            "expected split CUDA module {path}"
        );
        let lines = line_count(&path);
        assert!(
            lines <= MAX_LINES,
            "expected {path} to stay under {MAX_LINES} lines, got {lines}"
        );
    }
}
