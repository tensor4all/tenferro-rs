//! Standard bfloat16 carried through the external-scalar boundary.
//!
//! #1785 asks for `half::bf16` as a standard scalar representation, with construction, views,
//! materialization, explicit bf16/f32 conversions, basic forward arithmetic, and sum reduction
//! through the same public boundary the other external scalars use. This crate supplies exactly
//! that: the representation *is* [`half::bf16`], and the thin wrapper exists only because a
//! foreign type cannot implement tenferro's local scalar traits, which #1785 accepts for an
//! external contribution.
//!
//! # Declared arithmetic, accumulation, and rounding
//!
//! The behaviour is specified rather than implied, which is what #1785 asks for:
//!
//! - **Storage** is `half::bf16`, so a stored value is the nearest bfloat16 to what was written.
//! - **A single operation** is computed in `f32` and rounded back to bfloat16 once, so the result
//!   carries the rounding error of the operation and nothing else.
//! - **A reduction** accumulates in `f32` and rounds once at the end, so the accumulation does not
//!   quantize at every step. [`reduction::sum_in_f32_accumulation`] states this, and
//!   `tests/reduction_precision.rs` distinguishes that contract from repeated bfloat16 rounding,
//!   which is what #1785 requires of a promise like this one.
//!
//! Repeated bfloat16 rounding is a different and weaker contract, so the tests measure the
//! difference rather than asserting only that the sum is close.

#![deny(missing_docs)]

pub mod conversion;
pub mod reduction;

/// The standard bfloat16 representation this contribution stores.
pub use half::bf16;

/// A bfloat16 scalar in tenferro's scalar contract.
///
/// The representation is [`bf16`], and the wrapper carries no extra state.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::Bf16;
///
/// let value = Bf16::from_f32(1.5);
/// assert_eq!(value.to_f32(), 1.5);
/// assert_eq!(value.narrow(), half::bf16::from_f32(1.5));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bf16(pub bf16);

impl Bf16 {
    /// Wrap a bfloat16 value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::of(half::bf16::from_f32(2.0)).to_f32(), 2.0);
    /// ```
    #[must_use]
    pub const fn of(value: bf16) -> Self {
        Self(value)
    }

    /// The stored bfloat16 value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::from_f32(1.0).narrow(), half::bf16::from_f32(1.0));
    /// ```
    #[must_use]
    pub const fn narrow(self) -> bf16 {
        self.0
    }

    /// Round an `f64` to bfloat16, which is the only way a stored value is produced.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// // The spacing of bfloat16 on [1, 2) is 2^-8, so 1.001 is nearer to 1.0 than to the next
    /// // representable value above it.
    /// let stored = Bf16::from_f64(1.001);
    /// assert_eq!(stored.to_f32(), 1.0);
    /// ```
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self(bf16::from_f64(value))
    }

    /// Round an `f32` to bfloat16.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::from_f32(0.5).to_f32(), 0.5);
    /// ```
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }

    /// Widen the stored value to `f32`, which is exact.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// let stored = Bf16::from_f64(1.0 + 2f64.powi(-10));
    /// assert_eq!(stored.to_f32(), 1.0);
    /// ```
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    /// Widen the stored value to `f64`, which is exact.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::from_f32(2.5).to_f64(), 2.5);
    /// ```
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from(self.0.to_f32())
    }

    /// Additive identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::zero().to_f32(), 0.0);
    /// ```
    #[must_use]
    pub const fn zero() -> Self {
        Self(bf16::from_bits(0x0000))
    }

    /// Multiplicative identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!(Bf16::one().to_f32(), 1.0);
    /// ```
    #[must_use]
    pub fn one() -> Self {
        Self::from_f32(1.0)
    }
}

impl std::ops::Add for Bf16 {
    type Output = Self;

    /// Add in `f32` and round the sum once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!((Bf16::from_f32(1.0) + Bf16::from_f32(2.0)).to_f32(), 3.0);
    /// ```
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl std::ops::Sub for Bf16 {
    type Output = Self;

    /// Subtract in `f32` and round the difference once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!((Bf16::from_f32(3.0) - Bf16::from_f32(2.0)).to_f32(), 1.0);
    /// ```
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl std::ops::Mul for Bf16 {
    type Output = Self;

    /// Multiply in `f32` and round the product once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!((Bf16::from_f32(3.0) * Bf16::from_f32(4.0)).to_f32(), 12.0);
    /// ```
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl std::ops::Neg for Bf16 {
    type Output = Self;

    /// Negate, which flips the sign bit and needs no rounding.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16;
    ///
    /// assert_eq!((-Bf16::from_f32(2.0)).to_f32(), -2.0);
    /// ```
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl tenferro_tensor_core::Scalar for Bf16 {
    const DOMAIN: tenferro_tensor_core::ScalarDomain = tenferro_tensor_core::ScalarDomain::Field;
}

impl tenferro_tensor_core::ScalarArithmetic for Bf16 {
    fn scalar_zero() -> Self {
        Self::zero()
    }

    fn scalar_one() -> Self {
        Self::one()
    }

    fn scalar_add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn scalar_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn scalar_mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

/// bfloat16 addition as an operation type.
///
/// The operation is a type rather than a closure, so tenferro's kernels are instantiated once per
/// element type and operation instead of once per call site.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{Bf16, Bf16Add};
/// use tenferro_cpu::BinaryScalarOp;
///
/// let sum = <Bf16Add as BinaryScalarOp<Bf16>>::apply(Bf16::from_f32(1.0), Bf16::from_f32(2.0));
/// assert_eq!(sum.to_f32(), 3.0);
/// ```
pub struct Bf16Add;

impl tenferro_cpu::BinaryScalarOp<Bf16> for Bf16Add {
    fn apply(lhs: Bf16, rhs: Bf16) -> Bf16 {
        lhs + rhs
    }
}

/// bfloat16 subtraction as an operation type.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{Bf16, Bf16Sub};
/// use tenferro_cpu::BinaryScalarOp;
///
/// let difference =
///     <Bf16Sub as BinaryScalarOp<Bf16>>::apply(Bf16::from_f32(3.0), Bf16::from_f32(2.0));
/// assert_eq!(difference.to_f32(), 1.0);
/// ```
pub struct Bf16Sub;

impl tenferro_cpu::BinaryScalarOp<Bf16> for Bf16Sub {
    fn apply(lhs: Bf16, rhs: Bf16) -> Bf16 {
        lhs - rhs
    }
}

/// bfloat16 multiplication as an operation type.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{Bf16, Bf16Mul};
/// use tenferro_cpu::BinaryScalarOp;
///
/// let product =
///     <Bf16Mul as BinaryScalarOp<Bf16>>::apply(Bf16::from_f32(3.0), Bf16::from_f32(4.0));
/// assert_eq!(product.to_f32(), 12.0);
/// ```
pub struct Bf16Mul;

impl tenferro_cpu::BinaryScalarOp<Bf16> for Bf16Mul {
    fn apply(lhs: Bf16, rhs: Bf16) -> Bf16 {
        lhs * rhs
    }
}

tenferro_tensor_core::define_scalar_set! {
    /// Tag for the set that pairs the standard narrow type with `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::Bf16Tag;
    ///
    /// assert_ne!(Bf16Tag::F32, Bf16Tag::Bf16);
    /// ```
    pub enum Bf16Tag {
        /// Standard single precision.
        F32 => f32 : Float 0 32,
        /// Bfloat16, ranked below single precision.
        Bf16 => Bf16 : Float 1 16,
    }
    /// Value enum for the set that carries bfloat16 beside `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::{Bf16, Bf16Set, Bf16Tag};
    /// use tenferro_tensor_core::{HostTensor, ScalarSet};
    ///
    /// let value = Bf16Set::Bf16(HostTensor::from_vec_col_major(vec![1], vec![Bf16::from_f32(2.0)])?);
    /// assert_eq!(value.tag(), Bf16Tag::Bf16);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    pub enum Bf16Set;
}
