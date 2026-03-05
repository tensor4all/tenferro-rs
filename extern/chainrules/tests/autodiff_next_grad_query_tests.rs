use chainrules::{autograd, AutodiffError, BackwardOptions, Variable};

#[test]
fn grad_tangent_is_side_effect_free() {
    let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();

    y.backward(BackwardOptions {
        retain_graph: Some(true),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(x.grad().unwrap(), 4.0);

    let grads = autograd::grad_tangent(&y, &[&x], BackwardOptions::default()).unwrap();
    assert_eq!(grads, vec![4.0]);

    // Query API must not mutate accumulator buffers.
    assert_eq!(x.grad().unwrap(), 4.0);
}

#[test]
fn grad_tangent_create_graph_is_rejected() {
    let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();
    let err = autograd::grad_tangent(
        &y,
        &[&x],
        BackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, AutodiffError::ModeNotSupported { .. }));
}

#[test]
fn grad_variable_supports_second_derivative_without_hvp() {
    let x = Variable::new(3.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();

    let gx = autograd::grad_variable(
        &y,
        &[&x],
        BackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap()[0]
        .clone();
    let gxx = autograd::grad_variable(&gx, &[&x], BackwardOptions::default()).unwrap()[0].clone();

    assert_eq!(*gx.value(), 6.0);
    assert_eq!(*gxx.value(), 2.0);
}
