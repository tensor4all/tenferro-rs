use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

#[test]
fn dynamic_types_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/dyn_types.rs").exists(),
        "dyn_types.rs should stay removed after the core/dynamic split"
    );
    assert!(
        !repo_path("src/core/dynamic/dyn_ad_scalar").exists(),
        "DynAdScalar should stay removed after the homogeneous-tape redesign"
    );

    for relative in [
        "src/core/dynamic/mod.rs",
        "src/core/dynamic/dyn_scalar.rs",
        "src/core/dynamic/tensor_ops.rs",
        "src/core/dynamic/dyn_tensor.rs",
        "src/core/dynamic/dyn_ad_tensor/mod.rs",
        "src/core/dynamic/dyn_ad_tensor/layout.rs",
        "src/core/dynamic/dyn_ad_tensor/merge.rs",
        "src/core/dynamic/dyn_ad_tensor/basics.rs",
        "src/core/dynamic/dyn_ad_tensor/complex.rs",
        "src/core/dynamic/dyn_ad_tensor/eager_scalar.rs",
        "src/core/dynamic/dyn_ad_tensor/pullback.rs",
        "src/core/dynamic/dyn_ad_tensor/scalar_ops.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split dyn_types module to exist: {relative}"
        );
    }
}

#[test]
fn split_dynamic_type_modules_stay_under_size_guideline() {
    for relative in [
        "src/core/dynamic/dyn_scalar.rs",
        "src/core/dynamic/tensor_ops.rs",
        "src/core/dynamic/dyn_tensor.rs",
        "src/core/dynamic/dyn_ad_tensor/mod.rs",
        "src/core/dynamic/dyn_ad_tensor/layout.rs",
        "src/core/dynamic/dyn_ad_tensor/merge.rs",
        "src/core/dynamic/dyn_ad_tensor/basics.rs",
        "src/core/dynamic/dyn_ad_tensor/complex.rs",
        "src/core/dynamic/dyn_ad_tensor/eager_scalar.rs",
        "src/core/dynamic/dyn_ad_tensor/pullback.rs",
        "src/core/dynamic/dyn_ad_tensor/scalar_ops.rs",
    ] {
        let contents = std::fs::read_to_string(repo_path(relative)).unwrap();
        let line_count = contents.lines().count();
        assert!(
            line_count <= 500,
            "{relative} should stay focused; found {line_count} lines"
        );
    }
}
