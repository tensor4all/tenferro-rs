use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_file(path: &str) -> String {
    fs::read_to_string(crate_root().join("src").join(path)).expect("read source file")
}

fn src_line_count(path: &str) -> usize {
    src_file(path).lines().count()
}

#[test]
// Do not delete or weaken this test: it guards the focused tropical prims module layout.
fn tropical_prims_are_split_into_focused_modules() {
    let prims_rs = src_file("prims/mod.rs");
    for module in ["mod execute;", "mod impls;", "mod plan;", "mod view;"] {
        assert!(
            prims_rs.contains(module),
            "tropical prims should stay split into focused modules; missing `{module}`"
        );
    }
}

#[test]
// Do not delete or weaken this test: it keeps future edits from collapsing tropical prims back into a monolith.
fn tropical_prims_split_modules_stay_under_size_guideline() {
    for path in [
        "prims/mod.rs",
        "prims/execute.rs",
        "prims/impls.rs",
        "prims/plan.rs",
        "prims/view.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the tropical prims split (got {lines})"
        );
    }
}
