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
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{Error, ErrorPhase, TracedTensor};
/// # use tenferro_scalar_consumer_algorithm::{squared_factor_norm, ScalarSupport};
/// struct Refusing;
///
/// impl ScalarSupport for Refusing {
///     fn qr(&self, _input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
///         Err(Error::runtime_state("Refusing::qr", ErrorPhase::GraphBuild, "unsupported"))
///     }
///     fn to_f64(&self, _input: &TracedTensor) -> Result<TracedTensor, Error> {
///         Err(Error::runtime_state("Refusing::to_f64", ErrorPhase::GraphBuild, "unsupported"))
///     }
/// }
///
/// let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[2, 1])?;
/// // The factorization is the first capability the algorithm needs.
/// assert!(squared_factor_norm(&input, &Refusing).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait ScalarSupport {
    /// Factor a matrix into `(Q, R)`, where `R` has a positive diagonal.
    ///
    /// # Errors
    ///
    /// Returns the binding's typed error when the input is not a full-column-rank matrix
    /// the binding supports.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::{Error, ErrorPhase, TracedTensor};
    /// # use tenferro_scalar_consumer_algorithm::ScalarSupport;
    /// struct Identity;
    ///
    /// impl ScalarSupport for Identity {
    ///     fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
    ///         // A binding decides what a factorization means for its scalar.
    ///         Ok((input.clone(), input.clone()))
    ///     }
    ///     fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
    ///         Ok(input.clone())
    ///     }
    /// }
    ///
    /// let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[2, 1])?;
    /// let (_q, r) = Identity.qr(&input)?;
    /// assert_eq!(r.rank, 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error>;

    /// Present a supported value as an ordinary `f64` tensor.
    ///
    /// # Errors
    ///
    /// Returns the binding's typed error when the value is not one it supports.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::{Error, TracedTensor};
    /// # use tenferro_scalar_consumer_algorithm::ScalarSupport;
    /// struct AlreadyF64;
    ///
    /// impl ScalarSupport for AlreadyF64 {
    ///     fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
    ///         Ok((input.clone(), input.clone()))
    ///     }
    ///     fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
    ///         // The supported scalar is already ordinary `f64`, so no conversion is needed.
    ///         Ok(input.clone())
    ///     }
    /// }
    ///
    /// let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[1])?;
    /// assert_eq!(AlreadyF64.to_f64(&input)?.dtype(), tenferro_tensor::DType::F64);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error>;
}

/// The loss the connected program minimizes: the squared norm of the triangular factor.
///
/// The loss is ordinary `f64` work, so only [`ScalarSupport::to_f64`] stands between the
/// factorization's scalar and the objective.
///
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{Error, TracedTensor};
/// # use tenferro_scalar_consumer_algorithm::{squared_factor_norm, ScalarSupport};
/// struct PassThrough;
///
/// impl ScalarSupport for PassThrough {
///     fn qr(&self, input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
///         Ok((input.clone(), input.clone()))
///     }
///     fn to_f64(&self, input: &TracedTensor) -> Result<TracedTensor, Error> {
///         Ok(input.clone())
///     }
/// }
///
/// let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[2, 2])?;
/// // The loss is the squared norm of the factor the binding returned, as ordinary f64.
/// let loss = squared_factor_norm(&input, &PassThrough)?;
/// assert_eq!(loss.rank, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
/// # Examples
///
/// ```rust
/// # use tenferro_runtime::{Error, ErrorPhase, TracedTensor};
/// # use tenferro_scalar_consumer_algorithm::{factor_norm_gradient, ScalarSupport};
/// struct Refusing;
///
/// impl ScalarSupport for Refusing {
///     fn qr(&self, _input: &TracedTensor) -> Result<(TracedTensor, TracedTensor), Error> {
///         Err(Error::runtime_state("Refusing::qr", ErrorPhase::GraphBuild, "unsupported"))
///     }
///     fn to_f64(&self, _input: &TracedTensor) -> Result<TracedTensor, Error> {
///         Err(Error::runtime_state("Refusing::to_f64", ErrorPhase::GraphBuild, "unsupported"))
///     }
/// }
///
/// let ad = tenferro_ad::AdContext::builder().build()?;
/// let input = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[2, 1])?;
/// let seed = TracedTensor::input_concrete_shape(tenferro_tensor::DType::F64, &[])?;
/// // A binding that cannot factor stops the program before the reverse pass.
/// assert!(factor_norm_gradient(&ad, &input, &seed, &Refusing).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
