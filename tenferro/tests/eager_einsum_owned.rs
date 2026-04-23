use tenferro::eager_einsum::{eager_einsum, eager_einsum_owned};
use tenferro::{CpuBackend, Tensor};

#[test]
fn facade_eager_einsum_owned_matches_borrowed() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut borrowed_ctx = CpuBackend::new();
    let borrowed = eager_einsum(&mut borrowed_ctx, &[&a, &b], "ij,jk->ik").unwrap();

    let mut owned_ctx = CpuBackend::new();
    let owned = eager_einsum_owned(&mut owned_ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_eq!(owned.shape(), borrowed.shape());
    assert_eq!(owned.as_slice::<f64>(), borrowed.as_slice::<f64>());
    assert!(owned_ctx.buffer_pool_len() >= 2);
}
