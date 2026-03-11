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
