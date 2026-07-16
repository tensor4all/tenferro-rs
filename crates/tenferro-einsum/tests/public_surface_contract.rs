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
            rest.split_once("fn project_repeated_labels_to_diagonal")
                .map(|(body, _)| body)
        })
        .expect("broadcast_einsum_vjp_to_input_shape source section should exist");

    assert!(
        section.contains("let source_count = shape_sources.len();")
            && section.contains("std::iter::repeat_n(false, source_count)"),
        "einsum VJP broadcast should build active_mask from the actual inputs"
    );
    assert!(
        !section.contains("active_mask: vec![true, false]"),
        "einsum VJP broadcast must not use a fixed two-input active_mask when rank-0 inputs omit shape_source"
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
