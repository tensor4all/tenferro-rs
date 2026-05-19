use tenferro::tensor::{einsum, einsum_owned};
use tenferro::typed_tensor::{einsum as typed_einsum, TypedTensor};
use tenferro::{CpuBackend, Tensor};

#[test]
fn tensor_einsum_owned_matches_borrowed() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut borrowed_ctx = CpuBackend::new();
    let borrowed = einsum(&mut borrowed_ctx, &[&a, &b], "ij,jk->ik").unwrap();

    let mut owned_ctx = CpuBackend::new();
    let owned = einsum_owned(&mut owned_ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_eq!(owned.shape(), borrowed.shape());
    assert_eq!(owned.as_slice::<f64>(), borrowed.as_slice::<f64>());
    assert!(owned_ctx.buffer_pool_len() >= 2);
}

#[test]
fn typed_tensor_einsum_facade_matches_tensor_values() {
    let lhs = TypedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let rhs = TypedTensor::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();

    let out = typed_einsum(&mut backend, &[&lhs, &rhs], "ij,jk->ik").unwrap();

    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.host_data(), &[23.0, 34.0, 31.0, 46.0]);
}
