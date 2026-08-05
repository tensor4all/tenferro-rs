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

#[test]
fn p9_final_surface_has_no_parallel_tensor_owner_or_vec_result_path() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let group = fs::read_to_string(manifest.join("src/storage/group.rs"))
        .expect("group source must be readable");
    assert!(
        !group.contains("tensor_owners"),
        "AllocationGroup must not retain a parallel tensor owner table"
    );
    assert!(
        !group.contains("tensor_refs"),
        "AllocationGroup must not expose a tensor-reference side table"
    );

    let execution = fs::read_to_string(
        manifest
            .join("../tenferro-runtime/src/runtime/execution.rs")
            .canonicalize()
            .expect("runtime execution source must be present"),
    )
    .expect("runtime execution source must be readable");
    assert!(
        !execution.contains("Completed(Vec<Tensor>)"),
        "detached results must retain alias-safe group ownership"
    );
    let submit_error = execution
        .split("pub enum SubmitError")
        .nth(1)
        .and_then(|rest| rest.split("impl SubmitError").next())
        .expect("SubmitError declaration must be present");
    assert!(
        !submit_error.contains("WorkerSpawn"),
        "worker-spawn failure must recover unchanged inputs before admission"
    );
}
