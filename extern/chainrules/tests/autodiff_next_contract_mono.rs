use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use chainrules::{autograd, AutodiffError, AutogradGraph, BackwardOptions, Variable};

fn poison_graph<V: chainrules::Differentiable>(graph: &Arc<Mutex<AutogradGraph<V>>>) {
    let graph = Arc::clone(graph);
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let _guard = graph.lock().unwrap();
        panic!("poison graph mutex");
    }));
}

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
    let ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let c = Variable::new(2.0_f64);
    let out = autograd::add(&a, &c).unwrap();
    assert_eq!(out.context_id(), a.context_id());
}

#[test]
fn ad_next_041_shared_context_multi_leaf_success() {
    let ctx = AutogradGraph::<f64>::new();
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
    let ctx = AutogradGraph::<f64>::new();
    let x = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let x = x.requires_grad_(false).unwrap();
    assert!(!x.requires_grad());
    assert_eq!(x.context_id(), Some(ctx.lock().unwrap().id()));
}

#[test]
fn ad_next_044_requires_grad_false_foreign_context_is_treated_as_constant() {
    let tracked_ctx = AutogradGraph::<f64>::new();
    let foreign_ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&tracked_ctx))
        .requires_grad_(true)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&foreign_ctx))
        .requires_grad_(true)
        .unwrap()
        .requires_grad_(false)
        .unwrap();

    let y = autograd::add(&a, &b).unwrap();
    assert_eq!(y.context_id(), a.context_id());
    assert!(y.requires_grad());

    y.backward(Default::default()).unwrap();
    assert_eq!(a.grad(), Some(1.0));
    assert_eq!(b.grad(), None);
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

#[test]
fn ad_next_050_add_preserves_single_input_tangent() {
    let ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(3.5)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();

    let y = autograd::add(&a, &b).unwrap();
    assert_eq!(y.tangent(), Some(&3.5));
}

#[test]
fn ad_next_051_add_hvp_runs_for_two_tracked_inputs() {
    let ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(3.0)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap()
        .with_tangent_(5.0)
        .unwrap();

    let y = autograd::add(&a, &b).unwrap();
    y.backward_hvp(BackwardOptions::default()).unwrap();

    assert_eq!(a.grad(), Some(1.0));
    assert_eq!(b.grad(), Some(1.0));
    assert!(a.hvp().is_some());
    assert!(b.hvp().is_some());
}

#[test]
fn ad_next_052_poisoned_graph_rejects_add_and_square() {
    let ctx = AutogradGraph::<f64>::new();
    let a = Variable::new_in(1.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    let b = Variable::new_in(2.0_f64, Arc::clone(&ctx))
        .requires_grad_(true)
        .unwrap();
    poison_graph(&ctx);

    let add_err = autograd::add(&a, &b).err().unwrap();
    assert!(matches!(add_err, AutodiffError::InvalidArgument(msg) if msg.contains("poisoned")));

    let square_err = autograd::square(&a).err().unwrap();
    assert!(matches!(square_err, AutodiffError::InvalidArgument(msg) if msg.contains("poisoned")));
}
