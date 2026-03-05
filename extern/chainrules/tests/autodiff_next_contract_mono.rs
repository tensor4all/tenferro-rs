use std::sync::Arc;

use chainrules::{autograd, AutodiffError, AutogradContext, BackwardOptions, Variable};

#[test]
fn ad_next_004_second_derivative_without_hvp() {
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
    assert_eq!(*gxx.value(), 2.0);
}

#[test]
fn ad_next_025_accumulation_invariant_backward_and_hvp() {
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
    assert_eq!(x.grad(), Some(8.0));
    assert_eq!(x.hvp(), Some(2.0));
}

#[test]
fn ad_next_028_context_merge_single_context_with_constants() {
    let ctx = AutogradContext::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let c = Variable::new(2.0_f64);
    let out = autograd::add(&a, &c).unwrap();
    assert_eq!(out.context_id(), a.context_id());
}

#[test]
fn ad_next_041_shared_context_multi_leaf_success() {
    let ctx = AutogradContext::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let y = autograd::add(&a, &b).unwrap();
    y.backward(Default::default()).unwrap();
    assert_eq!(a.grad(), Some(1.0));
    assert_eq!(b.grad(), Some(1.0));
}

#[test]
fn ad_next_043_requires_grad_false_keeps_context_linkage() {
    let ctx = AutogradContext::<f64>::new();
    let x = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let x = x.requires_grad_(false).unwrap();
    assert!(!x.requires_grad());
    assert_eq!(x.context_id(), Some(ctx.lock().unwrap().id()));
}

#[test]
fn ad_next_047_zero_grad_leaf_only_manual_reset() {
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
    x.zero_grad().unwrap();
    assert!(x.grad().is_none());
    assert!(x.hvp().is_none());

    let err = y.zero_grad().unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
