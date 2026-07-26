use std::fs;
use std::path::Path;

fn rust_sources_below(root: &Path, sources: &mut Vec<String>) {
    for entry in fs::read_dir(root).unwrap_or_else(|error| {
        panic!(
            "failed to read source directory {}: {error}",
            root.display()
        )
    }) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources_below(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && path
                .file_name()
                .is_none_or(|name| name != "staging_surface_contract.rs")
        {
            sources.push(fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!("failed to read Rust source {}: {error}", path.display())
            }));
        }
    }
}

#[test]
fn ad_does_not_import_or_construct_runtime_execution_staging() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    rust_sources_below(&manifest_dir.join("src"), &mut sources);
    rust_sources_below(&manifest_dir.join("tests"), &mut sources);

    assert!(
        sources.iter().all(|source| !source.contains("ExecProgram")),
        "tenferro-ad must exercise traced/semantic public APIs, not runtime-private execution programs"
    );
    assert!(
        sources
            .iter()
            .all(|source| !source.contains("ExecInstruction")),
        "tenferro-ad must not construct runtime-private execution instructions"
    );
}
