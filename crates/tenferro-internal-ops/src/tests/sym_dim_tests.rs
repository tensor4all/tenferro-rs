use crate::SymDim;

#[test]
fn constant_value_evaluates_constant_expressions() {
    let expr = (SymDim::from(2usize).max(3usize) * SymDim::from(4usize)) / 2usize;

    assert_eq!(expr.constant_value(), Some(6));
}

#[test]
fn constant_value_returns_none_for_symbolic_expressions() {
    let expr = SymDim::tensor_axis(7, 0).max(3usize);

    assert_eq!(expr.constant_value(), None);
}
