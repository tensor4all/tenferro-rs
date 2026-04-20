//! Scalar newtypes for tropical (max-plus / min-plus / max-mul) semirings.
//!
//! These newtypes are intended for eager `TypedTensor<T>` T-generic kernels,
//! so that `TypedTensor<MaxPlus<f64>>::add(a, b)` can compute max-plus
//! addition element-wise by relying on the standard Rust arithmetic traits.
//!
//! # Current reachability (design_v3 Stage 4a)
//!
//! The `tenferro-tensor` public facade's `TensorScalar` trait is **sealed**
//! (implemented only for `f32`, `f64`, `Complex32`, `Complex64`). This means
//! `TypedTensor<MaxPlus<T>>` cannot currently flow through the `Tensor`
//! enum, `TensorBackend` ops, or the `TracedTensor` / `Engine` pipeline
//! through the public facade.
//!
//! The eager T-generic path is therefore scheduled for a later eager
//! integration stage (after Stage 7). Until that stage lands, the scalar
//! newtypes here are useful for direct scalar arithmetic and as a
//! specification of the intended eager algebraic semantics.
//!
//! # Semantics summary
//!
//! | Newtype       | `⊕` (Add)  | `⊗` (Mul) | additive identity |
//! |---------------|------------|-----------|-------------------|
//! | `MaxPlus<T>`  | `max(a,b)` | `a + b`   | `-∞`              |
//! | `MinPlus<T>`  | `min(a,b)` | `a + b`   | `+∞`              |
//! | `MaxMul<T>`   | `max(a,b)` | `a * b`   | `0`               |
//!
//! # Examples
//!
//! ```
//! use tenferro_ext_tropical::newtype::MaxPlus;
//!
//! let a = MaxPlus(3.0_f64);
//! let b = MaxPlus(5.0_f64);
//! // ⊕ (max)
//! assert_eq!(a + b, MaxPlus(5.0));
//! // ⊗ (ordinary +)
//! assert_eq!(a * b, MaxPlus(8.0));
//! ```

use std::ops::{Add, Mul};

/// Max-plus semiring scalar: `⊕ = max`, `⊗ = +`, additive identity `-∞`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::newtype::MaxPlus;
/// assert_eq!(MaxPlus(3.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
/// assert_eq!(MaxPlus(3.0_f64) * MaxPlus(5.0_f64), MaxPlus(8.0_f64));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxPlus<T>(pub T);

/// Min-plus semiring scalar: `⊕ = min`, `⊗ = +`, additive identity `+∞`.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::newtype::MinPlus;
/// assert_eq!(MinPlus(3.0_f64) + MinPlus(5.0_f64), MinPlus(3.0_f64));
/// assert_eq!(MinPlus(3.0_f64) * MinPlus(5.0_f64), MinPlus(8.0_f64));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinPlus<T>(pub T);

/// Max-times semiring scalar: `⊕ = max`, `⊗ = *`, additive identity `0`.
///
/// Useful for probabilities or tropical-of-exp transforms.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::newtype::MaxMul;
/// assert_eq!(MaxMul(0.3_f64) + MaxMul(0.5_f64), MaxMul(0.5_f64));
/// assert_eq!(MaxMul(0.3_f64) * MaxMul(0.5_f64), MaxMul(0.15_f64));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxMul<T>(pub T);

// ---------------------------------------------------------------------------
// MaxPlus impls
// ---------------------------------------------------------------------------

impl<T: num_traits::Float> Add for MaxPlus<T> {
    type Output = MaxPlus<T>;
    /// `⊕` in max-plus is `max`.
    fn add(self, rhs: MaxPlus<T>) -> MaxPlus<T> {
        MaxPlus(T::max(self.0, rhs.0))
    }
}

impl<T: Add<Output = T>> Mul for MaxPlus<T> {
    type Output = MaxPlus<T>;
    /// `⊗` in max-plus is ordinary `+`.
    fn mul(self, rhs: MaxPlus<T>) -> MaxPlus<T> {
        MaxPlus(self.0 + rhs.0)
    }
}

impl<T: num_traits::Float> Default for MaxPlus<T> {
    /// Additive identity for max-plus is `-∞`.
    fn default() -> Self {
        MaxPlus(T::neg_infinity())
    }
}

// ---------------------------------------------------------------------------
// MinPlus impls
// ---------------------------------------------------------------------------

impl<T: num_traits::Float> Add for MinPlus<T> {
    type Output = MinPlus<T>;
    /// `⊕` in min-plus is `min`.
    fn add(self, rhs: MinPlus<T>) -> MinPlus<T> {
        MinPlus(T::min(self.0, rhs.0))
    }
}

impl<T: Add<Output = T>> Mul for MinPlus<T> {
    type Output = MinPlus<T>;
    /// `⊗` in min-plus is ordinary `+`.
    fn mul(self, rhs: MinPlus<T>) -> MinPlus<T> {
        MinPlus(self.0 + rhs.0)
    }
}

impl<T: num_traits::Float> Default for MinPlus<T> {
    /// Additive identity for min-plus is `+∞`.
    fn default() -> Self {
        MinPlus(T::infinity())
    }
}

// ---------------------------------------------------------------------------
// MaxMul impls
// ---------------------------------------------------------------------------

impl<T: num_traits::Float> Add for MaxMul<T> {
    type Output = MaxMul<T>;
    /// `⊕` in max-times is `max`.
    fn add(self, rhs: MaxMul<T>) -> MaxMul<T> {
        MaxMul(T::max(self.0, rhs.0))
    }
}

impl<T: Mul<Output = T>> Mul for MaxMul<T> {
    type Output = MaxMul<T>;
    /// `⊗` in max-times is ordinary `*`.
    fn mul(self, rhs: MaxMul<T>) -> MaxMul<T> {
        MaxMul(self.0 * rhs.0)
    }
}

impl<T: num_traits::Zero> Default for MaxMul<T> {
    /// Additive identity for max-times is `0`.
    fn default() -> Self {
        MaxMul(T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MaxPlus -----------------------------------------------------------

    #[test]
    fn max_plus_add_takes_max() {
        assert_eq!(MaxPlus(3.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
        assert_eq!(MaxPlus(7.0_f64) + MaxPlus(2.0_f64), MaxPlus(7.0_f64));
    }

    #[test]
    fn max_plus_mul_is_ordinary_plus() {
        assert_eq!(MaxPlus(3.0_f64) * MaxPlus(5.0_f64), MaxPlus(8.0_f64));
        assert_eq!(MaxPlus(-1.0_f64) * MaxPlus(4.0_f64), MaxPlus(3.0_f64));
    }

    #[test]
    fn max_plus_default_is_neg_infinity() {
        let z: MaxPlus<f64> = MaxPlus::default();
        assert_eq!(z, MaxPlus(f64::NEG_INFINITY));
        // Additive identity: z ⊕ x = x.
        assert_eq!(z + MaxPlus(3.0_f64), MaxPlus(3.0_f64));
        assert_eq!(MaxPlus(3.0_f64) + z, MaxPlus(3.0_f64));
    }

    #[test]
    fn max_plus_works_with_f32() {
        assert_eq!(MaxPlus(1.0_f32) + MaxPlus(2.0_f32), MaxPlus(2.0_f32));
        assert_eq!(MaxPlus(1.0_f32) * MaxPlus(2.0_f32), MaxPlus(3.0_f32));
    }

    // ---- MinPlus -----------------------------------------------------------

    #[test]
    fn min_plus_add_takes_min() {
        assert_eq!(MinPlus(3.0_f64) + MinPlus(5.0_f64), MinPlus(3.0_f64));
        assert_eq!(MinPlus(7.0_f64) + MinPlus(2.0_f64), MinPlus(2.0_f64));
    }

    #[test]
    fn min_plus_mul_is_ordinary_plus() {
        assert_eq!(MinPlus(3.0_f64) * MinPlus(5.0_f64), MinPlus(8.0_f64));
    }

    #[test]
    fn min_plus_default_is_infinity() {
        let z: MinPlus<f64> = MinPlus::default();
        assert_eq!(z, MinPlus(f64::INFINITY));
        // Additive identity: z ⊕ x = x.
        assert_eq!(z + MinPlus(3.0_f64), MinPlus(3.0_f64));
    }

    // ---- MaxMul ------------------------------------------------------------

    #[test]
    fn max_mul_add_takes_max() {
        assert_eq!(MaxMul(0.3_f64) + MaxMul(0.5_f64), MaxMul(0.5_f64));
    }

    #[test]
    fn max_mul_mul_is_ordinary_times() {
        assert_eq!(MaxMul(0.3_f64) * MaxMul(0.5_f64), MaxMul(0.15_f64));
        assert_eq!(MaxMul(2.0_f64) * MaxMul(3.0_f64), MaxMul(6.0_f64));
    }

    #[test]
    fn max_mul_default_is_zero() {
        let z: MaxMul<f64> = MaxMul::default();
        assert_eq!(z, MaxMul(0.0_f64));
        // Additive identity: z ⊕ x = x when x >= 0.
        assert_eq!(z + MaxMul(0.4_f64), MaxMul(0.4_f64));
    }
}
