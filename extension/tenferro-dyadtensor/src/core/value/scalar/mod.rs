mod binary;
mod unary;

use super::core::{AdMode, AdValue};

/// Scalar newtype carrying AD mode information.
///
/// Binary operator overloads (`+`, `-`, `*`, `/`) are fallible and return
/// [`crate::Result`] so mixed reverse-tape validation is reported as a
/// recoverable error instead of panicking.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, core::AdScalar};
///
/// let x: AdScalar<f64> = 2.0_f64.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AdScalar<T>(pub AdValue<T>);

impl<T> AdScalar<T> {
    /// Creates a primal scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, core::AdScalar};
    ///
    /// let x = AdScalar::new_primal(1.5_f64);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn new_primal(value: T) -> Self {
        Self(AdValue::primal(value))
    }

    /// Creates a forward-mode scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, core::AdScalar};
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(primal: T, tangent: T) -> Self {
        Self(AdValue::forward(primal, tangent))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, core::AdScalar};
    ///
    /// let x = AdScalar::new_primal(2.0_f64);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        self.0.mode()
    }

    /// Returns reference to underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::core::{AdScalar, AdValue};
    ///
    /// let x = AdScalar::new_primal(2.0_f64);
    /// assert!(matches!(x.as_value(), AdValue::Primal(_)));
    /// ```
    pub fn as_value(&self) -> &AdValue<T> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::core::{AdScalar, AdValue};
    ///
    /// let x = AdScalar::new_primal(2.0_f64).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<T> {
        self.0
    }

    /// Consumes this wrapper and returns only the primal scalar value.
    ///
    /// This is an explicit AD-drop API for intentional metadata discard.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::core::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.into_primal(), 2.0);
    /// ```
    pub fn into_primal(self) -> T {
        match self.0 {
            AdValue::Primal(primal) => primal,
            AdValue::Forward { primal, .. } => primal,
        }
    }

    /// Returns primal scalar reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::core::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.primal(), &2.0);
    /// ```
    pub fn primal(&self) -> &T {
        self.0.primal_ref()
    }

    /// Returns tangent scalar reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::core::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.tangent(), Some(&1.0));
    /// ```
    pub fn tangent(&self) -> Option<&T> {
        self.0.tangent_ref()
    }
}

impl<T> From<T> for AdScalar<T> {
    fn from(value: T) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T> TryFrom<AdValue<T>> for AdScalar<T> {
    type Error = crate::Error;

    fn try_from(value: AdValue<T>) -> crate::Result<Self> {
        match value {
            AdValue::Primal(primal) => Ok(Self::new_primal(primal)),
            AdValue::Forward { primal, tangent } => Ok(Self::new_forward(primal, tangent)),
        }
    }
}

impl<T> From<AdScalar<T>> for AdValue<T> {
    fn from(value: AdScalar<T>) -> Self {
        value.0
    }
}
