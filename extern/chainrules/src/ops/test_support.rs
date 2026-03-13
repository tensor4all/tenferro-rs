use crate::{autograd, AdResult, Variable};

/// Builds `loss = x * x` with a tracked scalar leaf.
///
/// # Examples
///
/// ```
/// let (_x, loss) = chainrules::test_support::square_graph().unwrap();
/// assert!(loss.requires_grad());
/// ```
pub fn square_graph() -> AdResult<(Variable<f64>, Variable<f64>)> {
    let x = Variable::new(2.0_f64).requires_grad_(true)?;
    let loss = autograd::square(&x)?;
    Ok((x, loss))
}
