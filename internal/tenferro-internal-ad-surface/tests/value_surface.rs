use tenferro_internal_ad_surface::{backward, grad, BackwardOptions, GradOptions, Tensor};

#[test]
fn reverse_only_surface_uses_tidu_value_carrier() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    backward(&[&out], None, BackwardOptions::default()).unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert!((values[0] - 1.0_f64.exp()).abs() < 1.0e-12);
    assert!((values[1] - 2.0_f64.exp()).abs() < 1.0e-12);
}

#[test]
fn functional_grad_uses_value_vjp() {
    let x = Tensor::from_slice(&[1.0_f64, 0.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[2.0_f64, 3.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.add(&y).unwrap().sum().unwrap();

    let grads = grad(&[&out], &[&x, &y], None, GradOptions::default()).unwrap();

    assert_eq!(
        grads[0].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
}
