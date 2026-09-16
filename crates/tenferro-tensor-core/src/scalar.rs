//! Open scalar contracts for the host tensor data model.
//!
//! The preset scalar types are ordinary members of these contracts, not special
//! cases: the same table that enumerates them for the tag also declares their
//! scalar properties. A downstream crate implements [`Scalar`] for its own type
//! to take part in the same machinery.
//!
//! Scalars are classified by the algebra their ordinary arithmetic belongs to.
//! The host tensor data model stores values of any [`Scalar`]; differentiation
//! is a separate question answered by [`ad_admission`].

/// Algebra that a scalar's ordinary arithmetic belongs to.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Scalar, ScalarDomain};
///
/// assert_eq!(<f64 as Scalar>::DOMAIN, ScalarDomain::Field);
/// assert_eq!(<bool as Scalar>::DOMAIN, ScalarDomain::NonField);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarDomain {
    /// Addition, subtraction, multiplication, and multiplication by a negative
    /// value follow the ordinary real or complex field rules, so the canonical
    /// mathematical derivative definitions apply.
    Field,
    /// Any other algebra: tropical or min-plus, boolean, saturating, or another
    /// semiring. Such a scalar can still be stored and computed with, but the
    /// canonical field derivative rules are not valid for it.
    NonField,
}

/// A scalar the host tensor data model can store and move.
///
/// This contract carries only storage and representation properties. It does
/// not require arithmetic, a dtype tag, or any operation support, so declaring
/// a scalar never forces unrelated implementations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{Scalar, ScalarDomain};
///
/// fn domain_of<T: Scalar>() -> ScalarDomain {
///     T::DOMAIN
/// }
///
/// assert_eq!(domain_of::<i32>(), ScalarDomain::Field);
/// ```
pub trait Scalar: Copy + Send + Sync + 'static {
    /// Algebra of this scalar's ordinary arithmetic.
    const DOMAIN: ScalarDomain;
}

/// Arithmetic a scalar supports under its own rules.
///
/// The operations use the scalar's own semantics, so an integer implementation
/// wraps exactly as the existing integer kernels do. Complex scalars qualify as
/// [`ScalarDomain::Field`]. `bool` does not implement this trait: it is a
/// storable scalar without arithmetic.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::ScalarArithmetic;
///
/// assert_eq!(<f64 as ScalarArithmetic>::scalar_add(1.0, 2.0), 3.0);
/// assert_eq!(<i32 as ScalarArithmetic>::scalar_add(i32::MAX, 1), i32::MIN);
/// ```
pub trait ScalarArithmetic: Scalar {
    /// Additive identity.
    fn scalar_zero() -> Self;

    /// Multiplicative identity.
    fn scalar_one() -> Self;

    /// Sum under this scalar's own semantics.
    fn scalar_add(self, rhs: Self) -> Self;

    /// Difference under this scalar's own semantics.
    fn scalar_sub(self, rhs: Self) -> Self;

    /// Product under this scalar's own semantics.
    fn scalar_mul(self, rhs: Self) -> Self;
}

/// Why a scalar may not be differentiated at the requested order.
///
/// This is the query result of [`ad_admission`]; it never represents a gradient.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ad_admission, AdAdmissionError};
///
/// assert_eq!(
///     ad_admission::<f64>(2),
///     Err(AdAdmissionError::UnsupportedAdOrder { order: 2 })
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AdAdmissionError {
    /// Only first-order differentiation is admitted for a non-preset scalar.
    #[error("differentiation order {order} is not supported")]
    UnsupportedAdOrder {
        /// Requested derivative order.
        order: u32,
    },
    /// The scalar's arithmetic is not a field, so the canonical field
    /// derivative rules do not apply.
    #[error("the scalar's arithmetic is not an ordinary field")]
    NonFieldScalar,
    /// The scalar is admissible in principle, but no derivative rules exist for
    /// it in this build.
    #[error("no derivative rules are available for this scalar")]
    AdRuleUnavailable,
}

/// Answer whether the shared differentiation paths may differentiate `T` at
/// `order`.
///
/// This is a query. It does not run kernels, register rules, or change existing
/// differentiation, and a rejection is always explicit: an unsupported scalar
/// or order is never silently treated as a zero gradient.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{ad_admission, AdAdmissionError};
///
/// // A non-field scalar is rejected even at first order.
/// assert_eq!(ad_admission::<bool>(1), Err(AdAdmissionError::NonFieldScalar));
///
/// // A field scalar with no rules yet is rejected as unavailable, not as zero.
/// assert_eq!(
///     ad_admission::<f64>(1),
///     Err(AdAdmissionError::AdRuleUnavailable)
/// );
/// ```
///
/// # Errors
///
/// Returns [`AdAdmissionError::UnsupportedAdOrder`] for any order other than
/// one, [`AdAdmissionError::NonFieldScalar`] for a scalar whose arithmetic is
/// not an ordinary field, and [`AdAdmissionError::AdRuleUnavailable`] when no
/// rules exist for an otherwise admissible scalar.
pub fn ad_admission<T: Scalar>(order: u32) -> Result<(), AdAdmissionError> {
    if order != 1 {
        return Err(AdAdmissionError::UnsupportedAdOrder { order });
    }
    match T::DOMAIN {
        ScalarDomain::Field => Err(AdAdmissionError::AdRuleUnavailable),
        ScalarDomain::NonField => Err(AdAdmissionError::NonFieldScalar),
    }
}
