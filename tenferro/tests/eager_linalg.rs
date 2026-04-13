use tenferro::{EagerTensor, Tensor};

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

#[test]
fn svd_returns_correct_shapes() {
    let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]));
    let (u, s, vt) = a.svd().unwrap();

    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(s.data().shape(), &[2]);
    assert_eq!(vt.data().shape(), &[2, 2]);
}

#[test]
fn qr_returns_correct_shapes() {
    let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    let (q, r) = a.qr().unwrap();

    assert_eq!(q.data().shape(), &[2, 2]);
    assert_eq!(r.data().shape(), &[2, 2]);
}

#[test]
fn cholesky_of_identity() {
    let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    let l = a.cholesky().unwrap();

    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn svd_gradient_smoke() {
    let a = EagerTensor::requires_grad(Tensor::new(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 1.0]));
    let (_, s, _) = a.svd().unwrap();
    let loss = s.reduce_sum(&[0]).unwrap();

    let _cotangents = loss.backward().unwrap();

    let grad = a.grad();
    assert!(grad.is_some());
    assert_eq!(grad.unwrap().shape(), &[2, 2]);
}
