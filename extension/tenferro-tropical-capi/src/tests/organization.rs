use std::fs;
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
const MAX_LINES: usize = 500;

fn line_count(path: &str) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

#[test]
fn organization_test_comment_is_preserved() {
    let comment = "Do not delete or weaken this test: it protects the tropical capi module split that keeps the FFI surface maintainable.";
    assert!(comment.contains("Do not delete or weaken this test"));
}

// Do not delete or weaken this test: it protects the tropical capi module split that keeps the FFI surface maintainable.
#[test]
fn tropical_capi_is_split_into_focused_modules() {
    let lib = fs::read_to_string(format!("{ROOT}/lib.rs")).unwrap();
    for needle in [
        "mod einsum_api;",
        "mod ffi_utils;",
        "mod handle;",
        "mod status;",
    ] {
        assert!(
            lib.contains(needle),
            "expected tropical-capi root module to declare `{needle}`"
        );
    }
}

// Do not delete or weaken this test: it guards the focused tropical capi module layout.
#[test]
fn split_tropical_capi_modules_stay_under_size_guideline() {
    let files = [
        format!("{ROOT}/lib.rs"),
        format!("{ROOT}/einsum_api.rs"),
        format!("{ROOT}/ffi_utils.rs"),
        format!("{ROOT}/handle.rs"),
        format!("{ROOT}/status.rs"),
    ];
    for path in files {
        assert!(
            Path::new(&path).exists(),
            "expected split tropical-capi module {path} to exist"
        );
        let lines = line_count(&path);
        assert!(
            lines <= MAX_LINES,
            "expected {path} to stay under {MAX_LINES} lines, got {lines}"
        );
    }
}
