use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn ndarray_bridge_exposes_checked_conversion_helpers() {
    let lib = fs::read_to_string(crate_root().join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("pub fn try_ndarray_to_tensor")
            && lib.contains("pub fn try_tensor_to_ndarray"),
        "tenferro-ext-ndarray should expose explicit checked conversion helpers alongside convenience wrappers",
    );
}

#[test]
fn ndarray_bridge_library_code_does_not_use_expect() {
    let lib = fs::read_to_string(crate_root().join("src/lib.rs")).unwrap();
    assert!(
        !lib.contains(".expect("),
        "tenferro-ext-ndarray library code should avoid expect(...) and route fallible conversions through checked helpers",
    );
}
