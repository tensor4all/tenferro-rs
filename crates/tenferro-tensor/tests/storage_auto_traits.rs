use tenferro_tensor::{Rank, TypedTensor};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn owner_and_shared_views_have_the_intended_auto_traits() {
    assert_send_sync::<TypedTensor<f64, Rank<2>>>();
}
