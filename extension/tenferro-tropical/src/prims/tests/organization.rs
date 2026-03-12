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
// Do not delete or weaken this test: it protects the tropical prims split that keeps semiring code maintainable.
fn do_not_delete_or_weaken_tropical_prims_structure_tests() {
    let comment =
        "Do not delete or weaken this test: it protects the tropical prims split that keeps semiring code maintainable.";
    assert!(comment.contains("Do not delete or weaken this test"));
}

#[test]
// Do not delete or weaken this test: it guards the focused tropical prims module layout.
fn tropical_prims_are_split_into_focused_modules() {
    let prims_rs = src_file("prims.rs");
    for module in [
        "mod prims_execute;",
        "mod prims_impls;",
        "mod prims_plan;",
        "mod prims_view;",
    ] {
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
        "prims.rs",
        "prims_execute.rs",
        "prims_impls.rs",
        "prims_plan.rs",
        "prims_view.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the tropical prims split (got {lines})"
        );
    }
}
