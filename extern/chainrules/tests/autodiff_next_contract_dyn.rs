use chainrules::{autograd, AutodiffError, DynBackwardOptions, DynHvpOptions, DynTangent, DynTape};

#[test]
fn ad_next_020_grad_dyn_tangent_create_graph_rejected() {
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
fn ad_next_023_grad_dyn_variable_retain_default_free() {
    let tape = DynTape::new();
    let x = tape.leaf(1.0_f64);
    let _ = autograd::grad_dyn_variable(&[&x], &[&x], DynBackwardOptions::default()).unwrap();
    let out = autograd::grad_dyn_variable(&[&x], &[&x], DynBackwardOptions::default());
    assert!(matches!(out, Err(AutodiffError::GraphFreed)));
}

#[test]
fn ad_next_027_grad_dyn_variable_create_graph_implicit_keep() {
    let tape = DynTape::new();
    let x = tape.leaf(1.0_f64);
    let _ = autograd::grad_dyn_variable(
        &[&x],
        &[&x],
        DynBackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap();
    let out = autograd::grad_dyn_variable(
        &[&x],
        &[&x],
        DynBackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    );
    assert!(out.is_ok());
}

#[test]
fn ad_next_037_dyn_hvp_seed_none_invalid() {
    let tape = DynTape::new();
    let loss = tape.leaf(vec![1.0_f64, 2.0_f64]);
    let out = tape.hvp(&loss, DynHvpOptions::default());
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn ad_next_039_dyn_hvp_create_graph_unsupported() {
    let tape = DynTape::new();
    let loss = tape.leaf(1.0_f64);
    let out = tape.hvp(
        &loss,
        DynHvpOptions {
            create_graph: true,
            ..Default::default()
        },
    );
    assert!(matches!(out, Err(AutodiffError::ModeNotSupported { .. })));
}

#[test]
fn ad_next_049_cross_tape_context_mismatch() {
    let t1 = DynTape::new();
    let t2 = DynTape::new();
    let a = t1.leaf(1.0_f64);
    let b = t2.leaf(2.0_f64);
    let out = autograd::grad_dyn_tangent(
        &[&a],
        &[&b],
        DynBackwardOptions {
            seed_grads: Some(vec![DynTangent::new(1.0_f64)]),
            ..Default::default()
        },
    );
    assert!(matches!(out, Err(AutodiffError::InvalidArgument(_))));
}
