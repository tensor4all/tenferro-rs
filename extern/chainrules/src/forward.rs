use crate::{AdResult, Differentiable};

/// Value wrapper for forward-mode AD.
///
/// # Examples
///
/// ```
/// use chainrules::DualTensor;
/// let dual = DualTensor::new(3.14_f64);
/// assert!(!dual.has_tangent());
/// ```
pub struct DualTensor<V: Differentiable> {
    primal: V,
    tangent: Option<V::Tangent>,
}

impl<V: Differentiable> DualTensor<V> {
    /// Creates a dual value with zero tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(!x.has_tangent());
    /// ```
    pub fn new(primal: V) -> Self {
        Self {
            primal,
            tangent: None,
        }
    }

    /// Creates a dual value with explicit tangent.
    ///
    /// # Errors
    ///
    /// Returns [`chainrules_core::AutodiffError::TangentShapeMismatch`] if
    /// shapes differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0_f64).unwrap();
    /// assert!(x.has_tangent());
    /// assert_eq!(*x.tangent().unwrap(), 1.0);
    /// ```
    pub fn with_tangent(primal: V, tangent: V::Tangent) -> AdResult<Self> {
        Ok(Self {
            primal,
            tangent: Some(tangent),
        })
    }

    /// Returns the primal value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert_eq!(*x.primal(), 3.14);
    /// ```
    pub fn primal(&self) -> &V {
        &self.primal
    }

    /// Returns the tangent, or `None` for zero tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(x.tangent().is_none());
    /// ```
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this dual value has a non-zero tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(!x.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Consumes and returns `(primal, tangent)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    /// let (p, t) = x.into_parts();
    /// assert_eq!(p, 3.14);
    /// assert_eq!(t, Some(1.0));
    /// ```
    pub fn into_parts(self) -> (V, Option<V::Tangent>) {
        (self.primal, self.tangent)
    }

    /// Consumes and returns a dual value with tangent removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    /// let c = x.detach_tangent();
    /// assert!(!c.has_tangent());
    /// assert_eq!(*c.primal(), 3.14);
    /// ```
    pub fn detach_tangent(self) -> Self {
        Self {
            primal: self.primal,
            tangent: None,
        }
    }
}
