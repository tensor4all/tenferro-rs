use chainrules::{autograd, AutodiffError, BackwardOptions};

#[test]
fn retain_none_create_false_frees_graph() {
    let (_x, loss) = chainrules::test_support::square_graph().unwrap();
    loss.backward(Default::default()).unwrap();
    let err = loss.backward(Default::default()).unwrap_err();
    assert!(matches!(err, AutodiffError::GraphFreed));
}

#[test]
fn retain_none_create_true_keeps_graph() {
    let (x, loss) = chainrules::test_support::square_graph().unwrap();
    let opts = BackwardOptions {
        create_graph: true,
        ..Default::default()
    };
    let _ = autograd::grad_variable(&loss, &[&x], opts).unwrap();
    let _ = autograd::grad_variable(
        &loss,
        &[&x],
        BackwardOptions {
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn retain_some_false_overrides_create_true_and_frees_graph() {
    let (x, loss) = chainrules::test_support::square_graph().unwrap();
    let _ = autograd::grad_variable(
        &loss,
        &[&x],
        BackwardOptions {
            retain_graph: Some(false),
            create_graph: true,
            ..Default::default()
        },
    )
    .unwrap();

    let out = autograd::grad_variable(&loss, &[&x], BackwardOptions::default());
    assert!(matches!(out, Err(AutodiffError::GraphFreed)));
}
