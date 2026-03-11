use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn dyn_types_is_split_into_focused_modules() {
    assert!(
        !repo_path("src/dyn_types.rs").exists(),
        "dyn_types.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/dyn_types/mod.rs",
        "src/dyn_types/dyn_scalar.rs",
        "src/dyn_types/tensor_ops.rs",
        "src/dyn_types/dyn_tensor.rs",
        "src/dyn_types/dyn_ad_scalar/mod.rs",
        "src/dyn_types/dyn_ad_scalar/binary.rs",
        "src/dyn_types/dyn_ad_scalar/basics.rs",
        "src/dyn_types/dyn_ad_scalar/math.rs",
        "src/dyn_types/dyn_ad_scalar/traits.rs",
        "src/dyn_types/dyn_ad_tensor/mod.rs",
        "src/dyn_types/dyn_ad_tensor/layout.rs",
        "src/dyn_types/dyn_ad_tensor/merge.rs",
        "src/dyn_types/dyn_ad_tensor/basics.rs",
        "src/dyn_types/dyn_ad_tensor/complex.rs",
        "src/dyn_types/dyn_ad_tensor/scalar_ops.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split dyn_types module to exist: {relative}"
        );
    }
}

#[test]
fn split_dyn_types_modules_stay_under_size_guideline() {
    for relative in [
        "src/dyn_types/dyn_scalar.rs",
        "src/dyn_types/tensor_ops.rs",
        "src/dyn_types/dyn_tensor.rs",
        "src/dyn_types/dyn_ad_scalar/mod.rs",
        "src/dyn_types/dyn_ad_scalar/binary.rs",
        "src/dyn_types/dyn_ad_scalar/basics.rs",
        "src/dyn_types/dyn_ad_scalar/math.rs",
        "src/dyn_types/dyn_ad_scalar/traits.rs",
        "src/dyn_types/dyn_ad_tensor/mod.rs",
        "src/dyn_types/dyn_ad_tensor/layout.rs",
        "src/dyn_types/dyn_ad_tensor/merge.rs",
        "src/dyn_types/dyn_ad_tensor/basics.rs",
        "src/dyn_types/dyn_ad_tensor/complex.rs",
        "src/dyn_types/dyn_ad_tensor/scalar_ops.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
