//! Tropical scalar wrapper types.
//!
//! Each newtype redefines `Add` and `Mul` so that the [`Scalar`](tenferro_algebra::Scalar)
//! blanket impl is satisfied with tropical semantics:
//!
//! | Type | `+` (⊕) | `*` (⊗) | Zero | One |
//! |------|---------|---------|------|-----|
//! | [`MaxPlus<T>`] | max | + | −∞ | 0 |
//! | [`MinPlus<T>`] | min | + | +∞ | 0 |
//! | [`MaxMul<T>`] | max | × | 0 | 1 |
//!
//! All arithmetic bodies use `todo!()` (POC skeleton).

use std::fmt;
use std::ops;

/// Max-plus tropical scalar: ⊕ = max, ⊗ = +.
///
/// The most common tropical semiring, used for shortest-path and
/// optimal-alignment problems. Wraps any ordered numeric type `T`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlus;
///
/// let a = MaxPlus(3.0_f64);
/// let b = MaxPlus(5.0_f64);
///
/// // Tropical addition = max
/// let c = a + b;   // MaxPlus(5.0)
///
/// // Tropical multiplication = ordinary addition
/// let d = a * b;   // MaxPlus(8.0)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxPlus<T>(pub T);

/// Min-plus tropical scalar: ⊕ = min, ⊗ = +.
///
/// Used for shortest-path problems (Dijkstra, Bellman–Ford).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MinPlus;
///
/// let a = MinPlus(3.0_f64);
/// let b = MinPlus(5.0_f64);
///
/// // Tropical addition = min
/// let c = a + b;   // MinPlus(3.0)
///
/// // Tropical multiplication = ordinary addition
/// let d = a * b;   // MinPlus(8.0)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MinPlus<T>(pub T);

/// Max-times tropical scalar: ⊕ = max, ⊗ = ×.
///
/// Used for probabilistic models where you want the most likely
/// path (Viterbi algorithm with probabilities in [0, 1]).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxMul;
///
/// let a = MaxMul(0.3_f64);
/// let b = MaxMul(0.7_f64);
///
/// // Tropical addition = max
/// let c = a + b;   // MaxMul(0.7)
///
/// // Tropical multiplication = ordinary multiplication
/// let d = a * b;   // MaxMul(0.21)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxMul<T>(pub T);

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for MaxPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MaxPlus({})", self.0)
    }
}

impl<T: fmt::Display> fmt::Display for MinPlus<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinPlus({})", self.0)
    }
}

impl<T: fmt::Display> fmt::Display for MaxMul<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MaxMul({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// MaxPlus<T>: Add (= max), Mul (= +), Zero (= −∞), One (= 0)
// ---------------------------------------------------------------------------

impl<T: Copy + PartialOrd> ops::Add for MaxPlus<T> {
    type Output = Self;

    /// Tropical addition: max(a, b).
    fn add(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            self
        } else {
            rhs
        }
    }
}

impl<T: Copy + ops::Add<Output = T>> ops::Mul for MaxPlus<T> {
    type Output = Self;

    /// Tropical multiplication: a + b (ordinary addition).
    fn mul(self, rhs: Self) -> Self {
        MaxPlus(self.0 + rhs.0)
    }
}

impl<T: num_traits::Float> num_traits::Zero for MaxPlus<T> {
    /// Additive identity: −∞ (negative infinity).
    fn zero() -> Self {
        MaxPlus(T::neg_infinity())
    }

    fn is_zero(&self) -> bool {
        self.0 == T::neg_infinity()
    }
}

impl<T: num_traits::Float> num_traits::One for MaxPlus<T> {
    /// Multiplicative identity: 0.
    fn one() -> Self {
        MaxPlus(T::zero())
    }
}

// ---------------------------------------------------------------------------
// MinPlus<T>: Add (= min), Mul (= +), Zero (= +∞), One (= 0)
// ---------------------------------------------------------------------------

impl<T: Copy + PartialOrd> ops::Add for MinPlus<T> {
    type Output = Self;

    /// Tropical addition: min(a, b).
    fn add(self, rhs: Self) -> Self {
        if self.0 <= rhs.0 {
            self
        } else {
            rhs
        }
    }
}

impl<T: Copy + ops::Add<Output = T>> ops::Mul for MinPlus<T> {
    type Output = Self;

    /// Tropical multiplication: a + b (ordinary addition).
    fn mul(self, rhs: Self) -> Self {
        MinPlus(self.0 + rhs.0)
    }
}

impl<T: num_traits::Float> num_traits::Zero for MinPlus<T> {
    /// Additive identity: +∞ (positive infinity).
    fn zero() -> Self {
        MinPlus(T::infinity())
    }

    fn is_zero(&self) -> bool {
        self.0 == T::infinity()
    }
}

impl<T: num_traits::Float> num_traits::One for MinPlus<T> {
    /// Multiplicative identity: 0.
    fn one() -> Self {
        MinPlus(T::zero())
    }
}

// ---------------------------------------------------------------------------
// MaxMul<T>: Add (= max), Mul (= ×), Zero (= 0), One (= 1)
// ---------------------------------------------------------------------------

impl<T: Copy + PartialOrd> ops::Add for MaxMul<T> {
    type Output = Self;

    /// Tropical addition: max(a, b).
    fn add(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            self
        } else {
            rhs
        }
    }
}

impl<T: Copy + ops::Mul<Output = T>> ops::Mul for MaxMul<T> {
    type Output = Self;

    /// Tropical multiplication: a × b (ordinary multiplication).
    fn mul(self, rhs: Self) -> Self {
        MaxMul(self.0 * rhs.0)
    }
}

impl<T: num_traits::Float> num_traits::Zero for MaxMul<T> {
    /// Additive identity: 0 (for max, 0 is the identity when values are non-negative).
    fn zero() -> Self {
        MaxMul(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0 == T::zero()
    }
}

impl<T: num_traits::Float> num_traits::One for MaxMul<T> {
    /// Multiplicative identity: 1.
    fn one() -> Self {
        MaxMul(T::one())
    }
}

// ---------------------------------------------------------------------------
// Unsafe marker traits — MaxPlus/MinPlus/MaxMul are Send + Sync if T is.
// These are automatically derived from the inner T via #[repr(transparent)].
// ---------------------------------------------------------------------------

// Send and Sync are auto-derived since the wrapper is transparent over T.
