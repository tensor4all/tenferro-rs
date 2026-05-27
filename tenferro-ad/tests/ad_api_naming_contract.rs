use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tenferro-ad should live inside the workspace")
        .to_path_buf()
}

fn source(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|err| panic!("{path} should be readable: {err}"))
}

#[test]
fn optional_ad_apis_use_optional_suffix_without_try_aliases() {
    let context = source("tenferro-ad/src/context.rs");
    assert!(context.contains("pub fn jvp_optional("));
    assert!(context.contains("pub fn vjp_optional("));
    assert!(!context.contains("pub fn try_jvp("));
    assert!(!context.contains("pub fn try_vjp("));

    let traced = source("tenferro-ad/src/traced.rs");
    assert!(traced.contains("fn jvp_optional("));
    assert!(traced.contains("fn vjp_optional("));
    assert!(!traced.contains("fn try_jvp("));
    assert!(!traced.contains("fn try_vjp("));
}
