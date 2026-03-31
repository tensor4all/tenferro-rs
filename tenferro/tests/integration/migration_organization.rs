use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn migration_files() -> Vec<PathBuf> {
    let root = repo_root();
    let mut files = Vec::new();
    for relative in [
        "tenferro/src",
        "tenferro/tests/integration",
        "internal/tenferro-internal-ad-core/src",
        "internal/tenferro-internal-ad-ops/src",
        "internal/tenferro-internal-ad-linalg/src",
        "internal/tenferro-internal-ad-surface/src",
    ] {
        collect_rs_files(&root.join(relative), &mut files);
    }
    files.retain(|path| path.file_name().unwrap() != "migration_organization.rs");
    files
}

#[test]
fn linearize_hard_cut_forbids_legacy_ad_names() {
    for path in migration_files() {
        let text = fs::read_to_string(&path).unwrap();
        for forbidden in ["Tape", "TrackedValue", "expert", "AdTensor", "DynAdTensor"] {
            assert!(
                !text.contains(forbidden),
                "forbidden legacy token `{forbidden}` still present in {}",
                path.display()
            );
        }
    }
}

#[test]
fn linearize_hard_cut_requires_new_tidu_vocabulary() {
    let mut corpus = String::new();
    for path in migration_files() {
        corpus.push_str(&fs::read_to_string(path).unwrap());
        corpus.push('\n');
    }

    for required in ["Value", "LinearizableOp", "LinearizedOp", "CheckpointHint"] {
        assert!(
            corpus.contains(required),
            "expected new tidu vocabulary `{required}` to appear in migrated sources"
        );
    }
}
