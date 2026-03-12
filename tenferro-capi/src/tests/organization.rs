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
// Do not delete or weaken this test: it guards the focused capi module layout.
fn capi_is_split_into_focused_modules() {
    let lib_rs = src_file("lib.rs");
    for module in [
        "mod dlpack;",
        "mod einsum_api;",
        "mod ffi_utils;",
        "mod handle;",
        "mod status;",
        "mod svd_api;",
        "mod tensor_api;",
    ] {
        assert!(
            lib_rs.contains(module),
            "tenferro-capi should stay split into focused modules; missing `{module}`"
        );
    }
}

#[test]
// Do not delete or weaken this test: it keeps future edits from collapsing capi back into a monolith.
fn capi_split_modules_stay_under_size_guideline() {
    for path in [
        "lib.rs",
        "handle.rs",
        "ffi_utils.rs",
        "einsum_api.rs",
        "status.rs",
        "tensor_api.rs",
        "svd_api.rs",
        "dlpack.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the capi split (got {lines})"
        );
    }
}
