use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn result_types_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/result_types.rs").exists(),
        "result_types.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/result_types/mod.rs",
        "src/result_types/decomposition.rs",
        "src/result_types/status.rs",
        "src/result_types/spectral.rs",
        "src/result_types/cotangents.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split result_types module to exist: {relative}"
        );
    }
}

#[test]
fn split_result_type_modules_stay_under_size_guideline() {
    for relative in [
        "src/result_types/decomposition.rs",
        "src/result_types/status.rs",
        "src/result_types/spectral.rs",
        "src/result_types/cotangents.rs",
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
fn ad_helpers_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/ad_helpers.rs").exists(),
        "ad_helpers.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/ad_helpers/mod.rs",
        "src/ad_helpers/validation.rs",
        "src/ad_helpers/layout.rs",
        "src/ad_helpers/matrix_exp.rs",
        "src/ad_helpers/lu.rs",
        "src/ad_helpers/matrix_ops.rs",
        "src/ad_helpers/complex_ops.rs",
        "src/ad_helpers/backend_ops.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split ad_helpers module to exist: {relative}"
        );
    }
}

#[test]
fn split_ad_helper_modules_stay_under_size_guideline() {
    for relative in [
        "src/ad_helpers/validation.rs",
        "src/ad_helpers/layout.rs",
        "src/ad_helpers/matrix_exp.rs",
        "src/ad_helpers/lu.rs",
        "src/ad_helpers/matrix_ops.rs",
        "src/ad_helpers/complex_ops.rs",
        "src/ad_helpers/backend_ops.rs",
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
fn primal_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/primal.rs").exists(),
        "primal.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/primal/mod.rs",
        "src/primal/decompositions.rs",
        "src/primal/least_squares.rs",
        "src/primal/linear_systems.rs",
        "src/primal/spectral.rs",
        "src/primal/matrix_functions.rs",
        "src/primal/tensor_ops.rs",
        "src/primal/norms.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split primal module to exist: {relative}"
        );
    }
}

#[test]
fn split_primal_modules_stay_under_size_guideline() {
    for relative in [
        "src/primal/decompositions.rs",
        "src/primal/least_squares.rs",
        "src/primal/linear_systems.rs",
        "src/primal/spectral.rs",
        "src/primal/matrix_functions.rs",
        "src/primal/tensor_ops.rs",
        "src/primal/norms.rs",
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
fn rrules_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/rrules.rs").exists(),
        "rrules.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/rrules/mod.rs",
        "src/rrules/svd_qr.rs",
        "src/rrules/lu_eigen.rs",
        "src/rrules/least_squares.rs",
        "src/rrules/linear_systems.rs",
        "src/rrules/spectral.rs",
        "src/rrules/matrix_functions.rs",
        "src/rrules/norms.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split rrules module to exist: {relative}"
        );
    }
}

#[test]
fn split_rrule_modules_stay_under_size_guideline() {
    for relative in [
        "src/rrules/svd_qr.rs",
        "src/rrules/lu_eigen.rs",
        "src/rrules/least_squares.rs",
        "src/rrules/linear_systems.rs",
        "src/rrules/spectral.rs",
        "src/rrules/matrix_functions.rs",
        "src/rrules/norms.rs",
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
fn frules_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/frules.rs").exists(),
        "frules.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/frules/mod.rs",
        "src/frules/svd_qr.rs",
        "src/frules/lu_eigen.rs",
        "src/frules/least_squares.rs",
        "src/frules/linear_systems.rs",
        "src/frules/spectral.rs",
        "src/frules/matrix_functions.rs",
        "src/frules/norms.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split frules module to exist: {relative}"
        );
    }
}

#[test]
fn split_frule_modules_stay_under_size_guideline() {
    for relative in [
        "src/frules/svd_qr.rs",
        "src/frules/lu_eigen.rs",
        "src/frules/least_squares.rs",
        "src/frules/linear_systems.rs",
        "src/frules/spectral.rs",
        "src/frules/matrix_functions.rs",
        "src/frules/norms.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
