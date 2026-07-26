#[test]
fn einsum_eager_prototypes_are_not_public_api() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/eager_ad.rs");
    let source = std::fs::read_to_string(root).expect("read eager tensor source");

    for forbidden in [
        "pub fn einsum_whole_program_untracked(",
        "pub fn backend_broadcast_multiply_untracked(",
    ] {
        assert!(
            !source.contains(forbidden),
            "eager einsum prototype/helper leaked into the public API: {forbidden}"
        );
    }
}

#[test]
fn einsum_vjp_broadcast_uses_semantic_shape_metadata() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/extension.rs");
    let source = std::fs::read_to_string(root).expect("read einsum extension source");
    let section = source
        .split_once("fn semantic_broadcast_einsum_vjp")
        .and_then(|(_, rest)| {
            rest.split_once("fn semantic_project_repeated_labels")
                .map(|(body, _)| body)
        })
        .expect("semantic_broadcast_einsum_vjp source section should exist");

    assert!(
        section.contains("CoreSemanticOp::BroadcastInDim")
            && section.contains("semantic_project_repeated_labels"),
        "einsum VJP broadcast should use semantic shape metadata and preserve repeated labels"
    );
    assert!(
        !section.contains("active_mask: vec![true, false]"),
        "semantic einsum VJP broadcast must not reintroduce a fixed legacy active mask"
    );
}

#[test]
fn typed_einsum_view_inputs_use_read_suffix_and_typed_outputs_accept_views() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/concrete.rs");
    let source = std::fs::read_to_string(root).expect("read concrete einsum source");

    assert!(source.contains("pub trait TypedTensorReadEinsumExt"));
    assert!(source.contains("fn einsum_read<B: TensorBackend>"));
    assert!(source.contains("pub trait TypedTensorReadEinsumIntoExt"));
    assert!(source.contains("fn einsum_read_into<'out, B, O>"));
    assert!(source.contains("O: Into<TypedTensorWrite<'out, T>>"));

    assert!(
        !source.contains(
            "impl<'a, T: TensorScalar> TypedTensorEinsumExt<T> for [TypedTensorView<'a, T>]"
        ),
        "borrowed typed inputs must not implement the unsuffixed einsum surface"
    );
    assert!(
        !source.contains(
            "impl<'a, T: TensorScalar> TypedTensorEinsumIntoExt<T> for [TypedTensorView<'a, T>]"
        ),
        "borrowed typed inputs must not implement the unsuffixed einsum_into surface"
    );
}

#[test]
fn gpu_dependency_is_owned_by_opt_in_backend_features() {
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = std::fs::read_to_string(manifest_path).expect("read einsum manifest");
    let dependencies = manifest
        .split_once("[dependencies]")
        .and_then(|(_, rest)| rest.split_once("[dev-dependencies]").map(|(body, _)| body))
        .expect("manifest dependencies section");
    let dev_dependencies = manifest
        .split_once("[dev-dependencies]")
        .map(|(_, rest)| rest)
        .expect("manifest dev-dependencies section");

    assert!(
        dependencies.lines().any(|line| {
            line.starts_with("tenferro-gpu = ") && line.contains("optional = true")
        }),
        "tenferro-gpu must be an optional normal dependency"
    );
    assert!(
        !dev_dependencies
            .lines()
            .any(|line| line.starts_with("tenferro-gpu = ")),
        "tenferro-gpu must not be an unconditional dev-dependency"
    );

    for feature in ["cuda", "webgpu", "rocm"] {
        let feature_start = format!("{feature} = ");
        let feature_body = manifest
            .split_once(&feature_start)
            .and_then(|(_, rest)| rest.split_once(']').map(|(body, _)| body))
            .unwrap_or_else(|| panic!("missing or malformed {feature} feature"));
        assert!(
            feature_body.contains("dep:tenferro-gpu")
                && feature_body.contains(&format!("tenferro-gpu/{feature}")),
            "{feature} must explicitly activate tenferro-gpu and its matching backend feature"
        );
    }
}

#[test]
fn eager_extension_execution_uses_direct_context_without_legacy_registration() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/eager_ad.rs");
    let source = std::fs::read_to_string(root).expect("read eager tensor source");
    let section = source
        .split_once("pub fn einsum_subscripts")
        .and_then(|(_, rest)| {
            rest.split_once("fn try_direct_binary_dot_general")
                .map(|(body, _)| body)
        })
        .expect("eager einsum execution source section");

    assert!(section.contains("apply_eager_with_extension_context("));
    assert!(section.contains("execute_einsum_extension_reads("));
    assert!(!section.contains(".register_extension("));
    assert!(!section.contains("to_string()"));
}
