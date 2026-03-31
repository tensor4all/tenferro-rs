use tenferro::{backward, grad, BackwardOptions, GradOptions, Tensor};

#[test]
fn tensor_backward_accumulates_leaf_gradient() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let out = x.exp().unwrap().sum().unwrap();
    out.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert_eq!(values.len(), 2);
    assert!((values[0] - 1.0).abs() < 1.0e-12);
    assert!((values[1] - std::f64::consts::E).abs() < 1.0e-12);
}

#[test]
fn functional_grad_matches_additive_vjp() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let out = x.add(&y).unwrap().sum().unwrap();
    let grads = grad(&[&out], &[&x, &y], None, GradOptions::default()).unwrap();

    assert_eq!(grads.len(), 2);
    assert_eq!(
        grads[0].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
}

#[test]
fn free_backward_uses_default_seed() {
    let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    backward(&[&out], None, BackwardOptions::default()).unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert!((values[0] - 2.0_f64.exp()).abs() < 1.0e-12);
    assert!((values[1] - 3.0_f64.exp()).abs() < 1.0e-12);
}
