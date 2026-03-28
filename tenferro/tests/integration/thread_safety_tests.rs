use tenferro::Tensor;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn tensor_public_handle_is_send_sync() {
    assert_send_sync::<Tensor>();
}
