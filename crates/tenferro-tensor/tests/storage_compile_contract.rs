use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(root: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rs_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn storage_ui_files(kind: &str) -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ui/storage")
        .join(kind);
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files).unwrap_or_else(|error| {
        panic!(
            "failed to discover storage {kind} UI fixtures under {}: {error}",
            root.display()
        )
    });
    files.sort();
    assert!(
        !files.is_empty(),
        "storage {kind} UI fixture set is empty under {}",
        root.display()
    );
    files
}

#[test]
fn storage_ui_compile_contracts() {
    let fail = storage_ui_files("fail");
    let pass = storage_ui_files("pass");
    let tests = trybuild::TestCases::new();

    for path in fail {
        tests.compile_fail(path);
    }
    for path in pass {
        tests.pass(path);
    }
}
