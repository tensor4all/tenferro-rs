use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

// IMPORTANT: Do not delete or weaken these structure tests.
// They protect the module boundaries that keep tenferro extensible as more
// ops land under #441 and later follow-up work.

#[test]
fn ad_builders_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/ops/ad_builders").exists(),
        "the temporary ops/ad_builders bucket should stay removed after the op-first split"
    );

    for relative in [
        "src/ops/einsum/mod.rs",
        "src/ops/einsum/primal.rs",
        "src/ops/einsum/ad.rs",
        "src/ops/einsum/chainrules.rs",
        "src/ops/reduction/mod.rs",
        "src/ops/reduction/ad.rs",
        "src/ops/scalar/mod.rs",
        "src/ops/scalar/primal.rs",
        "src/ops/scalar/ad/mod.rs",
        "src/ops/scalar/ad/common.rs",
        "src/ops/scalar/ad/unary.rs",
        "src/ops/scalar/ad/binary.rs",
        "src/ops/scalar/ad/reduction.rs",
        "src/ops/linalg/ad/common.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected op-first module to exist: {relative}"
        );
    }
}

#[test]
fn api_tests_are_split_into_focused_modules() {
    for relative in [
        "src/ops/tests/mod.rs",
        "src/ops/tests/support.rs",
        "src/ops/tests/runtime_surface.rs",
        "src/ops/tests/runtime_helpers.rs",
        "src/ops/tests/runtime_dispatch.rs",
        "src/ops/tests/builder_coverage.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split API test module to exist: {relative}"
        );
    }
}

#[test]
fn split_api_test_modules_stay_under_size_guideline() {
    for relative in [
        "src/ops/tests/support.rs",
        "src/ops/tests/runtime_surface.rs",
        "src/ops/tests/runtime_helpers.rs",
        "src/ops/tests/runtime_dispatch.rs",
        "src/ops/tests/builder_coverage.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

#[test]
fn ad_tests_are_split_into_focused_modules() {
    for relative in [
        "src/ops/ad/tests/mod.rs",
        "src/ops/ad/tests/support.rs",
        "src/ops/ad/tests/eager_surface.rs",
        "src/ops/ad/tests/einsum_one_stage_real.rs",
        "src/ops/ad/tests/einsum_one_stage_complex.rs",
        "src/ops/ad/tests/einsum_two_stage.rs",
        "src/ops/ad/tests/linalg_finite_difference.rs",
        "src/ops/ad/tests/builder_pullbacks.rs",
        "src/ops/ad/tests/structured_pullbacks.rs",
        "src/ops/ad/tests/scalar_generic.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split AD test module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_test_modules_stay_under_size_guideline() {
    for relative in [
        "src/ops/ad/tests/support.rs",
        "src/ops/ad/tests/eager_surface.rs",
        "src/ops/ad/tests/einsum_one_stage_real.rs",
        "src/ops/ad/tests/einsum_one_stage_complex.rs",
        "src/ops/ad/tests/einsum_two_stage.rs",
        "src/ops/ad/tests/linalg_finite_difference.rs",
        "src/ops/ad/tests/builder_pullbacks.rs",
        "src/ops/ad/tests/structured_pullbacks.rs",
        "src/ops/ad/tests/scalar_generic.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

#[test]
fn split_ad_builder_modules_stay_under_size_guideline() {
    for relative in [
        "src/ops/einsum/primal.rs",
        "src/ops/einsum/ad.rs",
        "src/ops/einsum/chainrules.rs",
        "src/ops/reduction/ad.rs",
        "src/ops/scalar/primal.rs",
        "src/ops/scalar/ad/common.rs",
        "src/ops/scalar/ad/unary.rs",
        "src/ops/scalar/ad/binary.rs",
        "src/ops/scalar/ad/reduction.rs",
        "src/ops/linalg/ad/common.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

#[test]
fn ad_entrypoints_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/api/ad.rs").exists(),
        "ad.rs should stay removed after the ops/ad split"
    );

    for relative in [
        "src/ops/ad/mod.rs",
        "src/ops/ad/layout.rs",
        "src/ops/ad/pullback.rs",
        "src/ops/ad/scalar_eager.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split AD module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_entrypoint_modules_stay_under_size_guideline() {
    for relative in [
        "src/ops/ad/layout.rs",
        "src/ops/ad/pullback.rs",
        "src/ops/ad/scalar_eager.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}

#[test]
fn linalg_builders_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/api/linalg_builders.rs").exists(),
        "linalg_builders.rs should stay removed after the ops/ split"
    );

    for relative in [
        "src/ops/linalg/mod.rs",
        "src/ops/linalg/common.rs",
        "src/ops/linalg/results.rs",
        "src/ops/linalg/primal/mod.rs",
        "src/ops/linalg/primal/factorizations.rs",
        "src/ops/linalg/primal/solve.rs",
        "src/ops/linalg/primal/spectral.rs",
        "src/ops/linalg/primal/tensorized.rs",
        "src/ops/linalg/ad/mod.rs",
        "src/ops/linalg/ad/common.rs",
        "src/ops/linalg/ad/eager.rs",
        "src/ops/linalg/ad/single.rs",
        "src/ops/linalg/ad/slogdet.rs",
        "src/ops/linalg/ad/lu_lstsq.rs",
        "src/ops/linalg/ad/spectral.rs",
        "src/ops/linalg/ad/svd_qr.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split linalg module to exist: {relative}"
        );
    }
}

#[test]
fn split_linalg_builder_modules_stay_under_size_guideline() {
    for relative in [
        "src/ops/linalg/common.rs",
        "src/ops/linalg/results.rs",
        "src/ops/linalg/primal/factorizations.rs",
        "src/ops/linalg/primal/solve.rs",
        "src/ops/linalg/primal/spectral.rs",
        "src/ops/linalg/primal/tensorized.rs",
        "src/ops/linalg/ad/common.rs",
        "src/ops/linalg/ad/eager.rs",
        "src/ops/linalg/ad/single.rs",
        "src/ops/linalg/ad/slogdet.rs",
        "src/ops/linalg/ad/lu_lstsq.rs",
        "src/ops/linalg/ad/spectral.rs",
        "src/ops/linalg/ad/svd_qr.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
