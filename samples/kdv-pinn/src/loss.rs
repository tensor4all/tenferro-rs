//! PINN loss helpers: mean-squared error and the composite KdV loss.

use tenferro_runtime::{Result, TracedTensor};

/// Mean squared error between `pred` and `target`, scaled by `1 / n`.
///
/// Both tensors are expected to have shape `[n, 1]`. The result is a scalar.
pub(crate) fn mean_square(
    pred: &TracedTensor,
    target: &TracedTensor,
    n: usize,
) -> Result<TracedTensor> {
    assert!(n > 0, "mean_square count must be positive");
    let neg_target = target.neg()?;
    let diff = pred.add(&neg_target)?;
    let sq = diff.mul(&diff)?;
    let sum = sq.reduce_sum(&[0, 1])?;
    sum.scale_real(1.0 / n as f64)
}

/// Mean squared error of a single tensor against zero, scaled by `1 / n`.
///
/// The tensor is expected to have shape `[n, 1]`. The result is a scalar.
pub(crate) fn mean_square_single(tensor: &TracedTensor, n: usize) -> Result<TracedTensor> {
    assert!(n > 0, "mean_square_single count must be positive");
    let sq = tensor.mul(tensor)?;
    let sum = sq.reduce_sum(&[0, 1])?;
    sum.scale_real(1.0 / n as f64)
}

/// Composite KdV PINN loss.
///
/// Combines the PDE residual loss with initial- and boundary-condition losses
/// using scalar weights `lambda_pde`, `lambda_ic`, and `lambda_bc`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn total_loss(
    residual: &TracedTensor,
    u_ic: &TracedTensor,
    u_ic_true: &TracedTensor,
    u_bc: &TracedTensor,
    u_bc_true: &TracedTensor,
    n_col: usize,
    n_ic: usize,
    n_bc: usize,
    lambda_pde: f64,
    lambda_ic: f64,
    lambda_bc: f64,
) -> Result<TracedTensor> {
    let loss_pde = mean_square_single(residual, n_col)?.scale_real(lambda_pde)?;
    let loss_ic = mean_square(u_ic, u_ic_true, n_ic)?.scale_real(lambda_ic)?;
    let loss_bc = mean_square(u_bc, u_bc_true, n_bc)?.scale_real(lambda_bc)?;
    loss_pde.add(&loss_ic)?.add(&loss_bc)
}

#[cfg(test)]
mod tests;
