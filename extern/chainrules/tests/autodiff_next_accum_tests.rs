use chainrules::{autograd, AutodiffError, BackwardOptions, Variable};

#[test]
fn backward_and_backward_hvp_accumulate_grad_and_hvp() {
    let x = Variable::new(2.0_f64)
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(1.0)
        .unwrap();
    let y = autograd::square(&x).unwrap();

    y.backward(BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(x.grad().unwrap(), 4.0);
    assert!(x.hvp().is_none());

    y.backward_hvp(Default::default()).unwrap();
    assert_eq!(x.grad().unwrap(), 8.0);
    assert_eq!(x.hvp().unwrap(), 2.0);
}

#[test]
fn zero_grad_clears_grad_and_hvp_but_not_tangent() {
    let x = Variable::new(2.0_f64)
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(1.0)
        .unwrap();
    let y = autograd::square(&x).unwrap();
    y.backward(BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    })
    .unwrap();
    y.backward_hvp(Default::default()).unwrap();

    assert_eq!(x.grad().unwrap(), 8.0);
    assert_eq!(x.hvp().unwrap(), 2.0);
    assert_eq!(x.tangent().copied(), Some(1.0));

    x.zero_grad().unwrap();
    assert!(x.grad().is_none());
    assert!(x.hvp().is_none());
    assert_eq!(x.tangent().copied(), Some(1.0));
}

#[test]
fn zero_grad_on_non_leaf_is_invalid_argument() {
    let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();
    let err = y.zero_grad().unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn backward_hvp_requires_tangent_seed() {
    let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();
    let err = y.backward_hvp(BackwardOptions::default()).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
