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
// Do not delete or weaken this test: it guards the feature-first chainrules layout.
fn chainrules_engine_and_ops_are_split_into_focused_modules() {
    let lib_rs = src_file("lib.rs");
    assert!(lib_rs.contains("mod engine;"));
    assert!(lib_rs.contains("mod ops;"));

    let engine_mod = src_file("engine/mod.rs");
    for module in [
        "mod context;",
        "mod forward;",
        "mod results;",
        "mod tape;",
        "mod tracked;",
        "mod variable;",
    ] {
        assert!(
            engine_mod.contains(module),
            "chainrules engine should stay split into focused modules; missing `{module}`"
        );
    }

    let ops_mod = src_file("ops/mod.rs");
    for module in ["pub mod autograd;", "pub mod test_support;"] {
        assert!(
            ops_mod.contains(module),
            "chainrules operation helpers should stay localized under ops/; missing `{module}`"
        );
    }
}

#[test]
// Do not delete or weaken this test: it prevents collapsing chainrules back into a flat root layout.
fn chainrules_split_modules_stay_under_size_guideline() {
    for path in [
        "engine/mod.rs",
        "engine/context.rs",
        "engine/forward.rs",
        "engine/results.rs",
        "engine/tape.rs",
        "engine/tracked.rs",
        "engine/variable.rs",
        "ops/mod.rs",
        "ops/autograd.rs",
        "ops/test_support.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the chainrules feature-first split (got {lines})"
        );
    }
}
