//! Fused real-float activations computed in one elementwise traversal.
//!
//! The shared composite formulation in `tenferro_runtime::composite` evaluates
//! the same functions as a chain of separately materialized elementwise ops.
//! For `F32`/`F64` the CPU backend can instead fold the whole expression into a
//! single `map`, which removes the intermediate allocations and passes without
//! changing the result.
//!
//! This module is private: it backs the optional
//! [`tenferro_tensor::BackendSession::fused_activation_read`] fast path only.

use tenferro_tensor::{ActivationOp, Tensor, TensorRead, TensorView, TypedTensorView};

use crate::analytic::typed_unary_view_tensor_with_pool;
use crate::buffer_pool::{BufferPool, PoolScalar};

/// `sqrt(2 / pi)`, the GELU tanh-approximation scale.
const SQRT_2_OVER_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;

/// Cubic coefficient of the GELU tanh approximation.
const GELU_TANH_CUBIC: f64 = 0.044_715;

/// Scalar evaluation of one [`ActivationOp`].
///
/// The formulas mirror `tenferro_runtime::composite` so the fused result
/// matches the composite path elementwise up to the normal float rounding of a
/// different association order.
trait FusedActivationElem: Copy + PoolScalar {
    fn apply(op: ActivationOp, x: Self) -> Self;
}

macro_rules! impl_fused_activation_real {
    ($ty:ty, $erf:path) => {
        impl FusedActivationElem for $ty {
            fn apply(op: ActivationOp, x: Self) -> Self {
                let sqrt_2_over_pi = SQRT_2_OVER_PI as $ty;
                let cubic = GELU_TANH_CUBIC as $ty;
                let half = 0.5 as $ty;
                let one = 1 as $ty;
                // Overflow-free `1 / (1 + exp(-x))`: the small exponential is
                // the numerator on the negative branch, so a large negative
                // input cannot overflow it.
                let sigmoid = |x: $ty| {
                    let e = (-x.abs()).exp();
                    if x > 0 as $ty {
                        one / (one + e)
                    } else {
                        e / (one + e)
                    }
                };
                match op {
                    ActivationOp::Sigmoid => sigmoid(x),
                    ActivationOp::Silu => x * sigmoid(x),
                    ActivationOp::Softplus => {
                        let tail = (-x.abs()).exp().ln_1p();
                        if x > 0 as $ty {
                            x + tail
                        } else {
                            tail
                        }
                    }
                    ActivationOp::Gelu => {
                        let erf = $erf(x * (std::f64::consts::FRAC_1_SQRT_2 as $ty));
                        x * (one + erf) * half
                    }
                    ActivationOp::GeluTanh => {
                        let inner = sqrt_2_over_pi * (x + cubic * x * x * x);
                        x * (one + inner.tanh()) * half
                    }
                }
            }
        }
    };
}

impl_fused_activation_real!(f32, libm::erff);
impl_fused_activation_real!(f64, libm::erf);

/// Evaluate `op` on a borrowed typed view into a fresh pooled output.
fn fused_typed<T, R>(
    buffers: &mut BufferPool,
    op: ActivationOp,
    view: &TypedTensorView<'_, T, R>,
) -> crate::Result<Tensor>
where
    T: FusedActivationElem,
    R: tenferro_tensor::TensorRank,
{
    typed_unary_view_tensor_with_pool(op.label(), buffers, view, |x| T::apply(op, x))
}

/// Fused activation fast path for the CPU backend.
///
/// Returns `Ok(None)` for a dtype without a fused kernel so the caller keeps the
/// shared composite formulation.
pub(crate) fn fused_activation_read_with_pool(
    buffers: &mut BufferPool,
    op: ActivationOp,
    input: TensorRead<'_>,
) -> crate::Result<Option<Tensor>> {
    match input {
        TensorRead::Tensor(tensor) => {
            if let Some(v) = tensor.as_typed::<f32>() {
                return Ok(Some(fused_typed(buffers, op, &v.as_view())?));
            }
            if let Some(v) = tensor.as_typed::<f64>() {
                return Ok(Some(fused_typed(buffers, op, &v.as_view())?));
            }
            Ok(None)
        }
        TensorRead::View(TensorView::F32(view)) => Ok(Some(fused_typed(buffers, op, &view)?)),
        TensorRead::View(TensorView::F64(view)) => Ok(Some(fused_typed(buffers, op, &view)?)),
        TensorRead::View(_) => Ok(None),
    }
}
