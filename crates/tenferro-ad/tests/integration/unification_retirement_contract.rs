use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tenferro-ad should live under crates/ in the workspace")
        .to_path_buf()
}

fn read(root: &Path, path: &str) -> String {
    fs::read_to_string(root.join(path)).unwrap_or_else(|error| {
        panic!("{path} should be readable from {}: {error}", root.display())
    })
}

fn collect_rust_sources(root: &Path, relative: &Path, sources: &mut Vec<String>) {
    for entry in fs::read_dir(root.join(relative)).unwrap_or_else(|error| {
        panic!(
            "failed to read source directory {}: {error}",
            root.join(relative).display()
        )
    }) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            let relative = relative.join(path.file_name().expect("directory should have a name"));
            if relative
                .components()
                .any(|component| component.as_os_str() == "tests")
            {
                continue;
            }
            collect_rust_sources(root, &relative, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && path.file_name().is_none_or(|name| name != "tests.rs")
        {
            sources.push(
                path.strip_prefix(root)
                    .expect("source should be under repo root")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
}

fn production_paths(root: &Path) -> Vec<String> {
    let mut paths = vec![
        "Cargo.toml".to_string(),
        "crates/tenferro-ad/Cargo.toml".to_string(),
        "crates/tenferro-internal-ops/Cargo.toml".to_string(),
        "crates/tenferro-einsum/Cargo.toml".to_string(),
        "crates/tenferro-fft/Cargo.toml".to_string(),
        "crates/tenferro-linalg/Cargo.toml".to_string(),
        "ext/sparse/Cargo.toml".to_string(),
        "ext/tropical/Cargo.toml".to_string(),
    ];
    for relative in [
        "crates/tenferro-ad/src",
        "crates/tenferro-internal-ops/src",
        "crates/tenferro-einsum/src",
        "crates/tenferro-fft/src",
        "crates/tenferro-linalg/src",
        "crates/tenferro-runtime/src",
        "ext/sparse/src",
        "ext/tropical/src",
    ] {
        let path = root.join(relative);
        if path.exists() {
            collect_rust_sources(root, Path::new(relative), &mut paths);
        }
    }
    paths.sort();
    paths
}

fn assert_no_tidu_token(path: &str, source: &str) {
    for (line_index, line) in source.lines().enumerate() {
        assert!(
            !line.contains("tidu"),
            "{path}:{} must not mention the retired tidu dependency: {line}",
            line_index + 1
        );
    }
}

#[test]
fn unification_7_removes_tidu_dependency_from_production_sources_and_manifests() {
    let root = repo_root();
    for path in production_paths(&root) {
        assert_no_tidu_token(&path, &read(&root, &path));
    }
}
