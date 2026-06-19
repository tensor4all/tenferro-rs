#[test]
fn einsum_eager_prototypes_are_not_public_api() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/eager_tensor.rs");
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
