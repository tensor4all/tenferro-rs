use chainrules::{AutodiffError, DynTape};

#[test]
fn dyn_tape_leafs_are_tracked_and_have_context_id() {
    let tape = DynTape::new();
    let x = tape.leaf(3.0_f64);
    assert!(x.requires_grad());
    assert!(x.is_scalar());
    assert_eq!(x.context_id(), tape.id());
}

#[test]
fn dyn_value_as_wrong_type_errors() {
    let tape = DynTape::new();
    let x = tape.leaf(3.0_f64);
    assert!(matches!(
        x.value_as::<i32>(),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
