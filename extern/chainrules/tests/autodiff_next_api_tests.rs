use chainrules::{
    BackwardOptions, DynBackwardOptions, DynHvpOptions, DynTape, DynVariable, Variable,
};

#[test]
fn variable_api_surface_exists() {
    let _ = BackwardOptions::<f64>::default();
    let _ = DynBackwardOptions::default();
    let _ = DynHvpOptions::default();

    let v = Variable::new(1.0_f64);
    assert!(!v.requires_grad());
    assert!(v.node_id().is_none());
}

#[test]
fn dyn_api_surface_exists() {
    let tape = DynTape::new();
    let x: DynVariable = tape.leaf(3.0_f64);
    assert!(x.requires_grad());
    assert_eq!(x.context_id(), tape.id());
    assert!(x.value_as::<f64>().is_ok());
}
