use std::sync::Arc;

use chainrules::{autograd, AutodiffError, AutogradGraph, BackwardOptions, Tape, Variable};

#[test]
fn new_in_shared_context_allows_binary_ops() {
    let ctx = AutogradGraph::<f64>::new();
    let ctx_id = ctx.lock().unwrap().id();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let c = autograd::add(&a, &b).unwrap();
    assert_eq!(c.context_id(), Some(ctx_id));
    assert!(c.requires_grad());
}

#[test]
fn mixed_contexts_fail() {
    let a = Variable::new(1.0_f64).requires_grad_(true).unwrap();
    let b = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let out = autograd::add(&a, &b);
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn all_requires_grad_false_drops_output_context() {
    let ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(false)
        .unwrap();
    let b = Variable::new(2.0_f64);
    let c = autograd::add(&a, &b).unwrap();
    assert_eq!(c.context_id(), None);
}

#[test]
fn foreign_context_constant_is_ignored_for_binary_ops() {
    let tracked_ctx = AutogradGraph::<f64>::new();
    let foreign_ctx = AutogradGraph::<f64>::new();
    let tracked = Variable::new_in(1.0_f64, Arc::clone(&tracked_ctx))
        .requires_grad_(true)
        .unwrap();
    let constant = Variable::new_in(2.0_f64, Arc::clone(&foreign_ctx))
        .requires_grad_(false)
        .unwrap();

    let out = autograd::add(&tracked, &constant).unwrap();
    assert_eq!(out.context_id(), tracked.context_id());
    assert!(out.requires_grad());
}

#[test]
fn variable_api_surface_exists() {
    let _ = BackwardOptions::<f64>::default();
    let _ = Tape::<f64>::new();

    let v = Variable::new(1.0_f64);
    assert!(!v.requires_grad());
    assert!(v.node_id().is_none());
}

#[test]
fn tape_pullback_returns_leaf_gradient_for_tracked_variable() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0_f64);
    assert!(x.requires_grad());

    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}
