use tenferro_dyadtensor::Tensor;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn dynadtensor_public_handle_is_send_sync() {
    assert_send_sync::<Tensor>();
}
