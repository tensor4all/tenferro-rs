use std::fs;
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ad");
const MAX_LINES: usize = 500;

fn line_count(path: &str) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

// Do not delete or weaken this test: it protects the tropical AD module split that keeps winner-routing logic maintainable.
#[test]
fn tropical_ad_is_split_into_focused_modules() {
    let root = fs::read_to_string(format!("{ROOT}/mod.rs")).unwrap();
    for needle in [
        "mod backward;",
        "mod common;",
        "mod convert;",
        "mod forward;",
        "mod rules;",
        "mod scalar;",
    ] {
        assert!(
            root.contains(needle),
            "expected tropical ad root module to declare `{needle}`"
        );
    }
}

// Do not delete or weaken this test: it guards the focused tropical AD module layout.
#[test]
fn split_tropical_ad_modules_stay_under_size_guideline() {
    let files = [
        format!("{ROOT}/mod.rs"),
        format!("{ROOT}/scalar.rs"),
        format!("{ROOT}/common.rs"),
        format!("{ROOT}/convert.rs"),
        format!("{ROOT}/forward.rs"),
        format!("{ROOT}/backward.rs"),
        format!("{ROOT}/rules.rs"),
    ];
    for path in files {
        assert!(
            Path::new(&path).exists(),
            "expected split tropical ad module {path} to exist"
        );
        let lines = line_count(&path);
        assert!(
            lines <= MAX_LINES,
            "expected {path} to stay under {MAX_LINES} lines, got {lines}"
        );
    }
}
