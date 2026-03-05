use chainrules::{autograd, AutodiffError, BackwardOptions, DynHvpOptions, DynTape, Variable};

#[test]
fn ad_next_005_grad_tangent_create_graph_rejected() {
    let x = Variable::new(1.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();
    let out = autograd::grad_tangent(
        &y,
        &[&x],
        BackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    );
    assert!(matches!(out, Err(AutodiffError::ModeNotSupported { .. })));
}

#[test]
fn ad_next_013_mixed_contexts_invalid_argument() {
    let a = Variable::new(1.0_f64).requires_grad_(true).unwrap();
    let b = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let out = autograd::add(&a, &b);
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn ad_next_034_hvp_missing_direction_seed_invalid_argument() {
    let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();
    let out = y.backward_hvp(Default::default());
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn ad_next_044_dyn_hvp_freed_graph_error() {
    let tape = DynTape::new();
    let loss = tape.leaf(1.0_f64);
    let _ = tape.hvp(&loss, DynHvpOptions::default()).unwrap();
    let out = tape.hvp(&loss, DynHvpOptions::default());
    assert!(matches!(out, Err(AutodiffError::GraphFreed)));
}
