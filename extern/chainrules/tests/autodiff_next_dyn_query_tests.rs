use chainrules::{autograd, AutodiffError, DynBackwardOptions, DynHvpOptions, DynTangent, DynTape};

#[test]
fn grad_dyn_tangent_create_graph_rejected() {
    let tape = DynTape::new();
    let x = tape.leaf(1.0_f64);
    let out = autograd::grad_dyn_tangent(
        &[&x],
        &[&x],
        DynBackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    );
    assert!(matches!(out, Err(AutodiffError::ModeNotSupported { .. })));
}

#[test]
fn dyn_hvp_requires_seed_for_non_scalar_loss() {
    let tape = DynTape::new();
    let loss = tape.leaf(vec![1.0_f64, 2.0_f64]);
    let out = tape.hvp(&loss, DynHvpOptions::default());
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn dyn_cross_tape_operands_error() {
    let t1 = DynTape::new();
    let t2 = DynTape::new();
    let a = t1.leaf(1.0_f64);
    let b = t2.leaf(2.0_f64);
    let out = autograd::grad_dyn_tangent(&[&a], &[&b], DynBackwardOptions::default());
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn dyn_hvp_graph_freed_after_default_retain_policy() {
    let tape = DynTape::new();
    let loss = tape.leaf(1.0_f64);
    let _ = tape.hvp(&loss, DynHvpOptions::default()).unwrap();
    let out = tape.hvp(&loss, DynHvpOptions::default());
    assert!(matches!(out, Err(AutodiffError::GraphFreed)));
}

#[test]
fn grad_dyn_variable_create_graph_keeps_graph_by_default() {
    let tape = DynTape::new();
    let x = tape.leaf(1.0_f64);
    let g1 = autograd::grad_dyn_variable(
        &[&x],
        &[&x],
        DynBackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(g1[0].context_id(), tape.id());
    assert!(g1[0].requires_grad());

    let g2 = autograd::grad_dyn_variable(
        &[&x],
        &[&x],
        DynBackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(g2.len(), 1);
}

#[test]
fn grad_dyn_tangent_non_scalar_requires_seed_grads() {
    let tape = DynTape::new();
    let y = tape.leaf(vec![1.0_f64, 2.0_f64]);
    let x = tape.leaf(1.0_f64);
    let out = autograd::grad_dyn_tangent(&[&y], &[&x], DynBackwardOptions::default());
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));

    let ok = autograd::grad_dyn_tangent(
        &[&y],
        &[&x],
        DynBackwardOptions {
            seed_grads: Some(vec![DynTangent::new(vec![1.0_f64, 1.0_f64])]),
            ..Default::default()
        },
    );
    assert!(ok.is_ok());
}
