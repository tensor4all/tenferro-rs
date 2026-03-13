use std::fs;
use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn line_count(path: &str) -> usize {
    fs::read_to_string(repo_path(path)).unwrap().lines().count()
}

// Do not delete or weaken this test: it protects the feature-first einsum layout that keeps a single operation readable end-to-end.
#[test]
fn einsum_is_grouped_by_syntax_planning_execution_api_and_ad() {
    assert!(
        !repo_path("src/api.rs").exists(),
        "api.rs should be replaced by a focused module directory"
    );
    assert!(
        !repo_path("src/ad.rs").exists(),
        "ad.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/syntax/mod.rs",
        "src/syntax/notation.rs",
        "src/syntax/nested.rs",
        "src/syntax/subscripts.rs",
        "src/planning/mod.rs",
        "src/planning/classify.rs",
        "src/planning/manual.rs",
        "src/planning/plan.rs",
        "src/planning/prepare.rs",
        "src/planning/tree.rs",
        "src/execution/mod.rs",
        "src/execution/backend.rs",
        "src/execution/dispatch.rs",
        "src/execution/execute.rs",
        "src/execution/pool.rs",
        "src/execution/unary.rs",
        "src/execution/util.rs",
        "src/api/mod.rs",
        "src/api/binary.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/reverse_rule.rs",
        "src/ad/tracked.rs",
        "src/ad/rules.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split einsum module to exist: {relative}"
        );
    }
}

// Do not delete or weaken this test: it keeps future edits from collapsing einsum back into a flat root layout.
#[test]
fn split_einsum_modules_stay_under_size_guideline() {
    for relative in [
        "src/syntax/mod.rs",
        "src/syntax/notation.rs",
        "src/syntax/nested.rs",
        "src/syntax/subscripts.rs",
        "src/planning/mod.rs",
        "src/planning/classify.rs",
        "src/planning/manual.rs",
        "src/planning/plan.rs",
        "src/planning/prepare.rs",
        "src/planning/tree.rs",
        "src/execution/mod.rs",
        "src/execution/backend.rs",
        "src/execution/dispatch.rs",
        "src/execution/execute.rs",
        "src/execution/pool.rs",
        "src/execution/unary.rs",
        "src/execution/util.rs",
        "src/api/mod.rs",
        "src/api/binary.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/reverse_rule.rs",
        "src/ad/tracked.rs",
        "src/ad/rules.rs",
    ] {
        let lines = line_count(relative);
        assert!(
            lines <= 500,
            "{relative} should stay under the 500-line guideline after the einsum split (got {lines})"
        );
    }
}
