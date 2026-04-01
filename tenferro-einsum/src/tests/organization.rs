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
fn einsum_is_grouped_by_syntax_planning_execution_api_and_math_rules() {
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
        "src/api/canonical.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/delta.rs",
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
        "src/api/canonical.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/delta.rs",
        "src/ad/rules.rs",
    ] {
        let lines = line_count(relative);
        assert!(
            lines <= 500,
            "{relative} should stay under the 500-line guideline after the einsum split (got {lines})"
        );
    }
}

#[test]
fn dispatch_cleanup_guards_against_unwraps_and_usize_flop_math() {
    let contents = fs::read_to_string(repo_path("src/execution/dispatch.rs")).unwrap();

    assert!(
        !contents.contains(
            "try_fuse_group_in_target_order(&c_dims[..n_lo], &c_strides[..n_lo]).unwrap()"
        ),
        "dispatch direct-write path should not unwrap fused-layout probes"
    );
    assert!(
        !contents.contains("let flops = 2 * batch_total * plan.m * plan.n * plan.k;"),
        "dispatch profiling should not compute FLOPS in usize arithmetic"
    );
    assert!(
        contents.contains("checked_mul") || contents.contains("saturating_mul"),
        "dispatch profiling should use checked or saturating multiply for GEMM FLOPS"
    );
}

#[test]
fn einsum_ad_surface_forbids_legacy_tidu_types_and_hvp() {
    for relative in ["src/ad/mod.rs", "src/ad/rules.rs", "src/lib.rs"] {
        let contents = fs::read_to_string(repo_path(relative)).unwrap();
        for forbidden in [
            "TrackedValue",
            "DualValue",
            "ReverseRule",
            "NodeId",
            "Tape",
            "tracked_einsum",
            "dual_einsum",
            "einsum_hvp",
        ] {
            assert!(
                !contents.contains(forbidden),
                "{relative} should not reference legacy AD surface `{forbidden}`"
            );
        }
    }

    for removed in ["src/ad/reverse_rule.rs", "src/ad/tracked.rs"] {
        assert!(
            !repo_path(removed).exists(),
            "{removed} should be deleted in the linearize-first cut"
        );
    }
}
