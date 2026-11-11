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

    let feature_body = manifest
        .split_once("cuda = ")
        .and_then(|(_, rest)| rest.split_once(']').map(|(body, _)| body))
        .unwrap_or_else(|| panic!("missing or malformed cuda feature"));
    assert!(
        feature_body.contains("dep:tenferro-gpu") && feature_body.contains("tenferro-gpu/cuda"),
        "cuda must explicitly activate tenferro-gpu and its matching backend feature"
    );
}
