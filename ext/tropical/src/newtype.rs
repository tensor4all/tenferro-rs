//! Scalar newtypes for tropical semirings.
//!
//! The newtypes implement ordinary Rust arithmetic traits with tropical
//! semantics. They are useful for scalar specification tests today and provide
//! the intended eager typed-tensor direction once tenferro supports external
//! scalar element types in tensor storage.
//!
//! # Examples
//!
//! ```
//! use tenferro_ext_tropical::{MaxMul, MaxPlus, MinPlus};
//!
//! assert_eq!(MaxPlus(3.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
//! assert_eq!(MinPlus(3.0_f64) + MinPlus(5.0_f64), MinPlus(3.0_f64));
//! assert_eq!(MaxMul(0.3_f64) * MaxMul(0.5_f64), MaxMul(0.15_f64));
//! ```

use std::ops::{Add, Mul};

use num_traits::{Float, Zero};

/// Max-plus semiring scalar: `+` takes `max`, and `*` performs ordinary
/// addition.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::MaxPlus;
///
/// assert_eq!(MaxPlus(3.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
/// assert_eq!(MaxPlus(3.0_f64) * MaxPlus(5.0_f64), MaxPlus(8.0_f64));
/// assert_eq!(MaxPlus(3.0_f64).value(), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxPlus<T>(
    /// Wrapped scalar value.
    pub T,
);

impl<T> MaxPlus<T> {
    /// Return the wrapped scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::MaxPlus;
    ///
    /// assert_eq!(MaxPlus(2.0_f64).value(), 2.0);
    /// ```
    #[must_use]
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Float> Add for MaxPlus<T> {
    type Output = MaxPlus<T>;

    fn add(self, rhs: MaxPlus<T>) -> MaxPlus<T> {
        MaxPlus(self.0.max(rhs.0))
    }
}

// In max-plus algebra, semiring multiplication is ordinary addition.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Add<Output = T>> Mul for MaxPlus<T> {
    type Output = MaxPlus<T>;

    fn mul(self, rhs: MaxPlus<T>) -> MaxPlus<T> {
        MaxPlus(self.0 + rhs.0)
    }
}

impl<T: Float> Default for MaxPlus<T> {
    fn default() -> Self {
        MaxPlus(T::neg_infinity())
    }
}

/// Min-plus semiring scalar: `+` takes `min`, and `*` performs ordinary
/// addition.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::MinPlus;
///
/// assert_eq!(MinPlus(3.0_f64) + MinPlus(5.0_f64), MinPlus(3.0_f64));
/// assert_eq!(MinPlus(3.0_f64) * MinPlus(5.0_f64), MinPlus(8.0_f64));
/// assert_eq!(MinPlus(3.0_f64).value(), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinPlus<T>(
    /// Wrapped scalar value.
    pub T,
);

impl<T> MinPlus<T> {
    /// Return the wrapped scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::MinPlus;
    ///
    /// assert_eq!(MinPlus(3.0_f64).value(), 3.0);
    /// ```
    #[must_use]
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Float> Add for MinPlus<T> {
    type Output = MinPlus<T>;

    fn add(self, rhs: MinPlus<T>) -> MinPlus<T> {
        MinPlus(self.0.min(rhs.0))
    }
}

// In min-plus algebra, semiring multiplication is ordinary addition.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Add<Output = T>> Mul for MinPlus<T> {
    type Output = MinPlus<T>;

    fn mul(self, rhs: MinPlus<T>) -> MinPlus<T> {
        MinPlus(self.0 + rhs.0)
    }
}

impl<T: Float> Default for MinPlus<T> {
    fn default() -> Self {
        MinPlus(T::infinity())
    }
}

/// Max-times semiring scalar: `+` takes `max`, and `*` performs ordinary
/// multiplication.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::MaxMul;
///
/// assert_eq!(MaxMul(0.3_f64) + MaxMul(0.5_f64), MaxMul(0.5_f64));
/// assert_eq!(MaxMul(0.3_f64) * MaxMul(0.5_f64), MaxMul(0.15_f64));
/// assert_eq!(MaxMul(0.3_f64).value(), 0.3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxMul<T>(
    /// Wrapped scalar value.
    pub T,
);

impl<T> MaxMul<T> {
    /// Return the wrapped scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ext_tropical::MaxMul;
    ///
    /// assert_eq!(MaxMul(0.3_f64).value(), 0.3);
    /// ```
    #[must_use]
    pub fn value(self) -> T {
        self.0
    }
}

impl<T: Float> Add for MaxMul<T> {
    type Output = MaxMul<T>;

    fn add(self, rhs: MaxMul<T>) -> MaxMul<T> {
        MaxMul(self.0.max(rhs.0))
    }
}

impl<T: Mul<Output = T>> Mul for MaxMul<T> {
    type Output = MaxMul<T>;

    fn mul(self, rhs: MaxMul<T>) -> MaxMul<T> {
        MaxMul(self.0 * rhs.0)
    }
}

impl<T: Zero> Default for MaxMul<T> {
    fn default() -> Self {
        MaxMul(T::zero())
    }
}
