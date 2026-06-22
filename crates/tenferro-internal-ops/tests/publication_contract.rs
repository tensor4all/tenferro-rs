use std::path::PathBuf;

fn manifest() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("Cargo.toml");
    std::fs::read_to_string(path).expect("crate manifest must be readable")
}

#[test]
fn internal_ops_crate_is_publishable_implementation_crate() {
    let manifest = manifest();
    assert!(
        manifest.contains("publish.workspace = true"),
        "tenferro-internal-ops must be publishable because published tenferro crates depend on it"
    );
    assert!(
        !manifest.contains("publish = false"),
        "tenferro-internal-ops must not opt out of crates.io packaging"
    );
}
