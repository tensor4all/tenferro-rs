use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

// IMPORTANT: Do not delete or weaken these tests.
// They guard the checked-helper split and keep tenferro-ext-mdarray from
// drifting back to panic-only conversion entrypoints.

#[test]
fn mdarray_bridge_exposes_checked_conversion_helpers() {
    let lib = repo_file("src/lib.rs");
    assert!(
        lib.contains("pub fn try_mdarray_to_tensor")
            && lib.contains("pub fn try_tensor_to_mdarray"),
        "tenferro-ext-mdarray should expose explicit checked conversion helpers alongside convenience wrappers"
    );
}

#[test]
fn mdarray_bridge_library_code_does_not_use_expect() {
    let lib = repo_file("src/lib.rs");
    assert!(
        !lib.contains(".expect("),
        "tenferro-ext-mdarray library code should avoid expect(...) and route fallible conversions through checked helpers"
    );
}
