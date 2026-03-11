//! Tropical algebra markers, [`HasAlgebra`], and [`Semiring`] implementations.
//!
//! Each zero-sized struct identifies a tropical algebra for use with
//! semiring-family execution traits such as
//! [`TensorSemiringCore<Alg>`](tenferro_prims::TensorSemiringCore). The orphan
//! rule is satisfied because the algebra markers are defined in this crate.
//!
//! | Algebra marker | Scalar wrapper | ⊕ | ⊗ |
//! |----------------|---------------|---|---|
//! | [`MaxPlusAlgebra<T>`] | [`MaxPlus<T>`](crate::MaxPlus) | max | + |
//! | [`MinPlusAlgebra<T>`] | [`MinPlus<T>`](crate::MinPlus) | min | + |
//! | [`MaxMulAlgebra<T>`] | [`MaxMul<T>`](crate::MaxMul) | max | × |

use std::marker::PhantomData;

use tenferro_algebra::{Algebra, HasAlgebra, Semiring};

use crate::scalar::{MaxMul, MaxPlus, MinPlus};

/// Generates a tropical algebra marker struct with HasAlgebra and Semiring impls.
///
/// For each invocation, this creates:
/// - A generic zero-sized marker struct `$marker<T>`
/// - `HasAlgebra` impls mapping `$wrapper<f32>` and `$wrapper<f64>` to the marker
/// - `Semiring` impls for `$marker<f32>` and `$marker<f64>`
///
/// The `$add_fn` and `$mul_fn` closures define the semiring operations and must
/// be valid for both f32 and f64.
macro_rules! define_tropical_algebra {
    (
        $(#[$meta:meta])*
        $marker:ident, $wrapper:ident,
        zero_f32: $z32:expr, one_f32: $o32:expr,
        zero_f64: $z64:expr, one_f64: $o64:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy)]
        pub struct $marker<T>(PhantomData<T>);

        impl HasAlgebra for $wrapper<f32> {
            type Algebra = $marker<f32>;
        }

        impl HasAlgebra for $wrapper<f64> {
            type Algebra = $marker<f64>;
        }

        impl Algebra for $marker<f32> {
            type Scalar = $wrapper<f32>;
        }

        impl Algebra for $marker<f64> {
            type Scalar = $wrapper<f64>;
        }

        impl Semiring for $marker<f32> {
            fn zero() -> Self::Scalar { $z32 }
            fn one() -> Self::Scalar { $o32 }
            fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar { a + b }
            fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar { a * b }
        }

        impl Semiring for $marker<f64> {
            fn zero() -> Self::Scalar { $z64 }
            fn one() -> Self::Scalar { $o64 }
            fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar { a + b }
            fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar { a * b }
        }
    };
}

define_tropical_algebra!(
    /// Algebra marker for the max-plus tropical semiring (⊕ = max, ⊗ = +).
    ///
    /// Generic over the inner scalar type `T` (typically `f32` or `f64`).
    /// Used as the algebra parameter `Alg` in semiring-family execution traits
    /// such as
    /// [`TensorSemiringCore<MaxPlusAlgebra<T>>`](tenferro_prims::TensorSemiringCore).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_algebra::Semiring;
    /// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
    ///
    /// let z = <MaxPlusAlgebra<f64> as Semiring>::zero();
    /// assert_eq!(z, MaxPlus(f64::NEG_INFINITY));
    /// let z32 = <MaxPlusAlgebra<f32> as Semiring>::zero();
    /// assert_eq!(z32, MaxPlus(f32::NEG_INFINITY));
    /// ```
    MaxPlusAlgebra, MaxPlus,
    zero_f32: MaxPlus(f32::NEG_INFINITY), one_f32: MaxPlus(0.0f32),
    zero_f64: MaxPlus(f64::NEG_INFINITY), one_f64: MaxPlus(0.0f64)
);

define_tropical_algebra!(
    /// Algebra marker for the min-plus tropical semiring (⊕ = min, ⊗ = +).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_algebra::Semiring;
    /// use tenferro_tropical::{MinPlus, MinPlusAlgebra};
    ///
    /// let z = <MinPlusAlgebra<f64> as Semiring>::zero();
    /// assert_eq!(z, MinPlus(f64::INFINITY));
    /// ```
    MinPlusAlgebra, MinPlus,
    zero_f32: MinPlus(f32::INFINITY), one_f32: MinPlus(0.0f32),
    zero_f64: MinPlus(f64::INFINITY), one_f64: MinPlus(0.0f64)
);

define_tropical_algebra!(
    /// Algebra marker for the max-times tropical semiring (⊕ = max, ⊗ = ×).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_algebra::Semiring;
    /// use tenferro_tropical::{MaxMul, MaxMulAlgebra};
    ///
    /// let z = <MaxMulAlgebra<f64> as Semiring>::zero();
    /// assert_eq!(z, MaxMul(0.0f64));
    /// ```
    MaxMulAlgebra, MaxMul,
    zero_f32: MaxMul(0.0f32), one_f32: MaxMul(1.0f32),
    zero_f64: MaxMul(0.0f64), one_f64: MaxMul(1.0f64)
);

#[cfg(test)]
mod tests;
