//! External-scalar proof crate: a scalar type that tenferro does not define.
//!
//! This crate exists to show that the ordinary CPU numerical path in
//! `tenferro-cpu` does not require the preset scalar types. It defines its own
//! real scalar as a two-`f64` expansion, stores it in
//! [`tenferro_tensor_core::HostTensor`], and executes through
//! [`tenferro_cpu::scalar_binary_into`] and [`tenferro_cpu::scalar_fold`] using
//! the same traversal the preset scalars use.
//!
//! The type is deliberately small. It backs four arithmetic operations with an
//! exact two-sum so that low-order information survives accumulation, which an
//! `f64` round trip destroys. `xprec::Df64` is the production-shaped equivalent
//! and can replace this type without changing the tenferro side.

#![deny(missing_docs)]

pub mod ad;
pub mod conversion;
pub mod extension;

tenferro_tensor_core::define_scalar_set! {
    /// Tag for the external extended-precision set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::ExtendedTag;
    ///
    /// assert_ne!(ExtendedTag::F64, ExtendedTag::Df64);
    /// ```
    pub enum ExtendedTag {
        /// Standard double precision.
        F64 => f64 : Float 0 64,
        /// The external two-component scalar, ranked above double precision.
        Df64 => Df64 : Float 1 64,
    }
    /// Value enum for the external extended-precision set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::{Df64, ExtendedSet, ExtendedTag};
    /// use tenferro_tensor_core::{HostTensor, ScalarSet};
    ///
    /// let value = ExtendedSet::Df64(
    ///     HostTensor::from_vec_col_major(vec![1], vec![Df64::from_f64(2.0)])?,
    /// );
    /// assert_eq!(value.tag(), ExtendedTag::Df64);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub enum ExtendedSet;
}

/// A real scalar carrying a high and a low `f64` component.
///
/// `hi` holds the rounded value and `lo` the exact residual, so
/// `hi + lo` is the represented real number.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::Df64;
///
/// let carried = Df64 { hi: 1.0, lo: 2f64.powi(-80) };
/// assert_eq!(carried.narrow_to_f64(), 1.0);
/// assert_ne!(carried, Df64::from_f64(1.0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Df64 {
    /// Rounded component.
    pub hi: f64,
    /// Exact residual component.
    pub lo: f64,
}

/// Exact sum of two `f64` values as a high and low component (Knuth's two-sum).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let error = (a - (sum - b_virtual)) + (b - b_virtual);
    (sum, error)
}

impl Df64 {
    /// Build a scalar with no low component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::Df64;
    ///
    /// assert_eq!(Df64::from_f64(1.5), Df64 { hi: 1.5, lo: 0.0 });
    /// ```
    #[inline]
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Additive identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::Df64;
    ///
    /// assert_eq!(Df64::zero(), Df64 { hi: 0.0, lo: 0.0 });
    /// ```
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// Explicitly narrow to `f64`, discarding the low component.
    ///
    /// This is a numerical conversion, not a reinterpretation: it allocates
    /// nothing and reads only the rounded component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::Df64;
    ///
    /// // The low component is deliberately dropped, and is not recovered.
    /// assert_eq!(Df64 { hi: 1.0, lo: 2f64.powi(-80) }.narrow_to_f64(), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn narrow_to_f64(self) -> f64 {
        self.hi
    }
}

/// Exact product of two `f64` values as a high and low component (Dekker's
/// two-product), used by the expansion product.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let factor = 134_217_729.0_f64;
    let c = factor * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    let error = ((a_hi * b_hi - product) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (product, error)
}

impl tenferro_tensor_core::Scalar for Df64 {
    const DOMAIN: tenferro_tensor_core::ScalarDomain = tenferro_tensor_core::ScalarDomain::Field;
}

impl tenferro_tensor_core::ScalarArithmetic for Df64 {
    fn scalar_zero() -> Self {
        Self::zero()
    }

    fn scalar_one() -> Self {
        Self::from_f64(1.0)
    }

    fn scalar_add(self, rhs: Self) -> Self {
        std::ops::Add::add(self, rhs)
    }

    fn scalar_sub(self, rhs: Self) -> Self {
        std::ops::Sub::sub(self, rhs)
    }

    /// First-order expansion product: the exact leading product plus the
    /// first-order correction terms.
    fn scalar_mul(self, rhs: Self) -> Self {
        let (leading, trailing) = two_product(self.hi, rhs.hi);
        let correction = trailing + self.hi * rhs.lo + self.lo * rhs.hi;
        let (hi, lo) = two_sum(leading, correction);
        Self { hi, lo }
    }
}

/// The external contribution's addition operation.
///
/// The operation is a type in the crate that owns the scalar, so tenferro's
/// kernels are instantiated once per element type and operation rather than once
/// per call site.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::{Df64, Df64Add};
/// use tenferro_cpu::BinaryScalarOp;
///
/// let sum = <Df64Add as BinaryScalarOp<Df64>>::apply(
///     Df64::from_f64(1.0),
///     Df64::from_f64(2.0),
/// );
/// assert_eq!(sum, Df64::from_f64(3.0));
/// ```
pub struct Df64Add;

impl tenferro_cpu::BinaryScalarOp<Df64> for Df64Add {
    fn apply(lhs: Df64, rhs: Df64) -> Df64 {
        std::ops::Add::add(lhs, rhs)
    }
}

impl std::ops::Add for Df64 {
    type Output = Self;

    /// Exact addition of two expansions.
    #[inline]
    fn add(self, other: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, other.hi);
        let (t1, t2) = two_sum(self.lo, other.lo);
        let (s2, s3) = two_sum(s2, t1);
        let lo = s3 + t2;
        let (hi, lo) = two_sum(s1, s2 + lo);
        Self { hi, lo }
    }
}

impl std::ops::Neg for Df64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl std::ops::Sub for Df64 {
    type Output = Self;

    /// Exact subtraction.
    #[inline]
    fn sub(self, other: Self) -> Self {
        std::ops::Add::add(self, std::ops::Neg::neg(other))
    }
}
