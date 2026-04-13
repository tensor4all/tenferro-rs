use tenferro::eager_einsum::eager_einsum_ad;
use tenferro::{EagerTensor, Tensor};

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

#[test]
fn eager_einsum_ad_matmul_primal_matches_expected_values() {
    let a = EagerTensor::from_tensor(Tensor::new(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = EagerTensor::from_tensor(Tensor::new(
        vec![3, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let c = eager_einsum_ad(&[&a, &b], "ij,jk->ik").unwrap();

    assert_eq!(c.data().shape(), &[2, 2]);
    assert_eq!(f64_data(c.data()), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn eager_einsum_ad_backward_populates_input_grads() {
    let a = EagerTensor::requires_grad(Tensor::new(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = EagerTensor::requires_grad(Tensor::new(
        vec![3, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let c = eager_einsum_ad(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[3, 2]);
    assert_eq!(f64_data(grad_a.as_ref()), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    assert_eq!(f64_data(grad_b.as_ref()), &[3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
}
