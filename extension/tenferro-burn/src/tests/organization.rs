use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

// IMPORTANT: Do not delete or weaken these tests.
// They guard the checked-helper architecture that keeps tenferro-burn from
// drifting back to scattered expect(...) calls and ad hoc panic sites.

#[test]
fn burn_bridge_uses_checked_helper_entrypoints() {
    let lib = repo_file("src/lib.rs");
    let convert = repo_file("src/convert.rs");

    assert!(
        lib.contains("pub fn try_einsum"),
        "tenferro-burn should expose a fallible try_einsum helper so checked flows exist alongside the infallible convenience wrapper"
    );
    assert!(
        convert.contains("pub fn try_burn_to_tenferro")
            && convert.contains("pub fn try_tenferro_to_burn"),
        "conversion helpers should keep explicit fallible entrypoints instead of forcing callers through panic-only wrappers"
    );
}

#[test]
fn burn_bridge_library_code_does_not_use_expect() {
    let sources = [
        repo_file("src/lib.rs"),
        repo_file("src/convert.rs"),
        repo_file("src/backward.rs"),
        repo_file("src/forward.rs"),
    ]
    .join("\n");

    assert!(
        !sources.contains(".expect("),
        "tenferro-burn library code should funnel failure through checked helpers instead of scattered expect(...) calls"
    );
}
