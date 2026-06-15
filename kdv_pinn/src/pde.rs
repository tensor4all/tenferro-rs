//! KdV residual and differentiation helpers.
//!
//! This module builds the PDE residual terms for the Korteweg–de Vries equation
//! and provides small wrappers around the traced-graph automatic-differentiation
//! APIs so the PINN code can request spatial and temporal derivatives concisely.

use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{Error, Result, TracedTensor};

/// Compute the gradient of `output` with respect to `input` on a traced graph.
///
/// This is a thin helper over [`TracedTensorAdExt::grad`] so that PDE residual
/// code can request derivatives without importing the AD extension trait
/// directly.
// TODO(kdv-pinn): remove #[allow(dead_code)] once training loop or loss code wires this helper.
#[allow(dead_code)]
pub(crate) fn grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor> {
    output.grad(input)
}

/// Compute the KdV residual `u_t + u * u_x + u_xxx`.
///
/// `u`, `x`, and `t` must have concrete shapes. `x` and `t` are the
/// independent-variable placeholders with respect to which derivatives are
/// taken. The returned tensor has the same shape as `u`.
pub(crate) fn kdv_residual(
    u: &TracedTensor,
    x: &TracedTensor,
    t: &TracedTensor,
) -> Result<TracedTensor> {
    let ones_x = ones_like(x)?;
    let ones_t = ones_like(t)?;
    let u_t = u.jvp(t, &ones_t)?;
    let u_x = u.jvp(x, &ones_x)?;
    let u_xx = u_x.jvp(x, &ones_x)?;
    let u_xxx = u_xx.jvp(x, &ones_x)?;
    let nonlinear = u.mul(&u_x);
    Ok(u_t.add(&nonlinear).add(&u_xxx))
}

/// Return a constant tensor of ones with the same shape and dtype as `tensor`.
fn ones_like(tensor: &TracedTensor) -> Result<TracedTensor> {
    let shape = tensor
        .try_concrete_shape()
        .ok_or_else(|| Error::Internal("placeholder shape must be concrete".to_string()))?;
    let len = shape.iter().product::<usize>();
    let ones = match tensor.dtype {
        tenferro_runtime::DType::F64 => {
            TracedTensor::from_vec_col_major(shape.clone(), vec![1.0_f64; len])
        }
        dtype => {
            return Err(Error::Internal(format!(
                "ones_like only supports F64, got {:?}",
                dtype
            )))
        }
    };
    Ok(ones)
}

#[cfg(test)]
mod tests;
