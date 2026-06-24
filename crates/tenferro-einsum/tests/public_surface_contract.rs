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
fn einsum_vjp_broadcast_active_mask_matches_dynamic_inputs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/extension.rs");
    let source = std::fs::read_to_string(root).expect("read einsum extension source");
    let section = source
        .split_once("fn broadcast_einsum_vjp_to_input_shape")
        .and_then(|(_, rest)| {
            rest.split_once("fn map_label_occurrences")
                .map(|(body, _)| body)
        })
        .expect("broadcast_einsum_vjp_to_input_shape source section should exist");

    assert!(
        section.contains("let mut active_mask ="),
        "einsum VJP broadcast should build active_mask from the actual inputs"
    );
    assert!(
        !section.contains("active_mask: vec![true, false]"),
        "einsum VJP broadcast must not use a fixed two-input active_mask when rank-0 inputs omit shape_source"
    );
}
