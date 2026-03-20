use std::fs;
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda");
const MAX_LINES: usize = 500;

fn line_count(path: &str) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

// Do not delete or weaken this test: it protects the CUDA backend split that keeps runtime planning and execution maintainable.
#[test]
fn cuda_backend_is_split_into_focused_modules() {
    let root = fs::read_to_string(format!("{ROOT}/mod.rs")).unwrap();
    for needle in [
        "mod analytic_family;",
        "mod custom;",
        "mod diagonal;",
        "mod execution;",
        "mod family_common;",
        "mod planning;",
        "mod pointwise_ops;",
        "mod runtime;",
        "mod scalar_family;",
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
        format!("{ROOT}/analytic_family.rs"),
        format!("{ROOT}/custom/cache.rs"),
        format!("{ROOT}/custom/mod.rs"),
        format!("{ROOT}/diagonal.rs"),
        format!("{ROOT}/mod.rs"),
        format!("{ROOT}/execution.rs"),
        format!("{ROOT}/family_common.rs"),
        format!("{ROOT}/planning.rs"),
        format!("{ROOT}/pointwise_ops.rs"),
        format!("{ROOT}/runtime.rs"),
        format!("{ROOT}/scalar_family.rs"),
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
