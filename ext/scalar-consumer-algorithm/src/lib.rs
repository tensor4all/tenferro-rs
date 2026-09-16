//! The algorithm role of the external-scalar consumer.
//!
//! The algorithm states the capabilities it needs and never names a scalar, a provider,
//! or a dtype-specific entry point. The application supplies the bindings, so the same
//! source runs with canonical standard support and with an application-added scalar
//! contribution.

use tenferro_ad::AdContext;
use tenferro_runtime::{Error, ErrorPhase, TracedTensor};

/// The scalar and operation capabilities the algorithm requires from its binding.
///
/// A binding supplies a factorization of the supported scalar and a way to present a
/// value of that scalar as an ordinary `f64` tensor, which is what the loss and its
/// derivative are written against.
pub trait ScalarSupport {
    /// Factor a matrix into `(Q, R)`, where `R` has a positive diagonal.
    ///
    /// # Errors
    ///
    /// Returns the binding's typed error when the input is not a full-column-rank matrix
    /// the binding supports.
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error>;

    /// Present a supported value as an ordinary `f64` tensor.
    ///
    /// # Errors
    ///
    /// Returns the binding's typed error when the value is not one it supports.
    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error>;
}

/// The loss the connected program minimizes: the squared norm of the triangular factor.
///
/// The loss is ordinary `f64` work, so only [`ScalarSupport::to_f64`] stands between the
/// factorization's scalar and the objective.
///
/// # Errors
///
/// Returns the binding's typed error when the factorization or the presentation fails.
pub fn squared_factor_norm<S: ScalarSupport + ?Sized>(
    input: &TracedTensor,
    support: &S,
) -> Result<TracedTensor, Error> {
    let (_q, r) = support.qr(input)?;
    let narrowed = support.to_f64(&r)?;
    let squared = narrowed.mul(&narrowed)?;
    squared.reduce_sum(None)
}

/// The objective's gradient with respect to the program input.
///
/// # Errors
///
/// Returns the binding's typed error when the program cannot be built, or the AD
/// context's error when the reverse pass cannot be run.
pub fn factor_norm_gradient<S: ScalarSupport + ?Sized>(
    ad: &AdContext,
    input: &TracedTensor,
    seed: &TracedTensor,
    support: &S,
) -> Result<TracedTensor, Error> {
    let loss = squared_factor_norm(input, support)?;
    let gradients = ad.vjp_many(&loss, &[input], seed)?;
    gradients.into_iter().next().flatten().ok_or_else(|| {
        Error::runtime_state(
            "factor_norm_gradient",
            ErrorPhase::GraphBuild,
            "the input is inactive",
        )
    })
}

#[cfg(test)]
mod tests;
