//! Execution-path contract.
//!
//! The prepared and unprepared paths must not diverge in *policy*: which
//! instruction runs may fuse into one command is decided by exactly one
//! segmentation function and one eligibility function, and no execution path may
//! invent its own. This contract scans the runtime sources so a future change
//! cannot add a second decision site without failing here.

use std::fs;
use std::path::{Path, PathBuf};

fn runtime_sources() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    collect(&root, &mut sources);
    sources
}

fn collect(dir: &Path, sources: &mut Vec<(PathBuf, String)>) {
    for entry in fs::read_dir(dir).expect("runtime source directory should be readable") {
        let entry = entry.expect("directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect(&path, sources);
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("runtime source should be readable");
        sources.push((path, text));
    }
}

fn files_containing(needle: &str) -> Vec<String> {
    let mut files = Vec::new();
    for (path, text) in runtime_sources() {
        if text.contains(needle) {
            files.push(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_string(),
            );
        }
    }
    files.sort();
    files
}

#[test]
fn fusion_policy_has_a_single_decision_site() {
    // Candidate extraction and eligibility live in `segment.rs`; the prepared
    // path reuses both through `region.rs`.
    let segmentation = files_containing("fn segment_exec_program(");
    assert_eq!(
        segmentation,
        vec!["segment.rs".to_string()],
        "segmentation must be defined once, found in {segmentation:?}"
    );
    let eligibility = files_containing("fn build_elementwise_fusion_plan(");
    assert_eq!(
        eligibility,
        vec!["segment.rs".to_string()],
        "fusion eligibility must be defined once, found in {eligibility:?}"
    );
    let callers = files_containing("build_elementwise_fusion_plan(");
    for expected in ["segment.rs", "region.rs"] {
        assert!(
            callers.contains(&expected.to_string()),
            "{expected} should use the shared eligibility check, found {callers:?}"
        );
    }
    assert!(
        !callers.contains(&"execution.rs".to_string()),
        "the executor must consume planned regions, not decide eligibility itself"
    );
}

/// The scheduled executor is the only production executor: the legacy
/// segmented executor's whole-program entry points are reachable only from the
/// test module, so it cannot diverge from the production path at run time.
#[test]
fn planned_regions_are_the_only_prepared_fusion_entry() {
    // The prepared path may only fuse through planned regions: the region type
    // is produced by `region.rs`, stored on the prepared root, and executed by
    // the executor.
    let region_producers = files_containing("ElementwiseRegion {");
    assert_eq!(
        region_producers,
        vec!["region.rs".to_string()],
        "regions must be constructed once, found in {region_producers:?}"
    );
    let executor = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime/execution.rs");
    let text = fs::read_to_string(&executor).expect("executor source should be readable");
    assert!(
        text.contains("execute_elementwise_fusion_slots"),
        "the executor must run planned regions through the region entry point"
    );
}
