#[test]
fn typed_tensor_helpers_do_not_erase_through_host_copies() {
    let source = include_str!("../../src/typed_tensor.rs");

    assert!(
        !source.contains("host_data().to_vec()"),
        "typed tensor helpers must use backend read APIs instead of copying host data"
    );
    assert!(
        !source.contains("fn erase("),
        "typed tensor helpers must not erase typed tensors through a local host-copy adapter"
    );
}
