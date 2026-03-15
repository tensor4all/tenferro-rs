pub use chainrules_core::NodeId;

/// Automatic differentiation mode.
///
/// # Examples
///
/// ```ignore
/// use tenferro::AdMode;
///
/// assert_eq!(AdMode::Primal, AdMode::Primal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdMode {
    /// Plain evaluation without derivative propagation.
    Primal,
    /// Forward-mode value carrying tangent information.
    Forward,
    /// Reverse-mode value carrying graph metadata.
    Reverse,
}

/// Generic AD value that can wrap any user-defined payload type `T`.
///
/// This is the primary extension point of the crate for primal/forward-mode
/// values. Reverse-mode graph values are represented by [`crate::AdTensor`]
/// on a homogeneous `chainrules::Tape<crate::DynTensor>`.
///
/// # Examples
///
/// ```text
/// use tenferro::{AdMode, core::AdValue};
///
/// let primal = AdValue::primal(3.0_f64);
/// assert_eq!(primal.mode(), AdMode::Primal);
///
/// let dual = AdValue::forward(3.0_f64, 1.0_f64);
/// assert_eq!(dual.mode(), AdMode::Forward);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AdValue<T> {
    /// Primal-only value.
    Primal(T),
    /// Forward-mode value and tangent.
    Forward { primal: T, tangent: T },
}

impl<T> AdValue<T> {
    /// Creates a primal-only value.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let x = AdValue::primal(2_i32);
    /// assert!(matches!(x, AdValue::Primal(2)));
    /// ```
    pub fn primal(value: T) -> Self {
        Self::Primal(value)
    }

    /// Creates a forward-mode value.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 1.0_f64);
    /// assert!(matches!(x, AdValue::Forward { .. }));
    /// ```
    pub fn forward(primal: T, tangent: T) -> Self {
        Self::Forward { primal, tangent }
    }

    /// Returns the AD mode.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::{AdMode, core::AdValue};
    ///
    /// let x = AdValue::forward(1.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::Primal(_) => AdMode::Primal,
            Self::Forward { .. } => AdMode::Forward,
        }
    }

    /// Returns a reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let x = AdValue::forward(10_i32, 1_i32);
    /// assert_eq!(x.primal_ref(), &10);
    /// ```
    pub fn primal_ref(&self) -> &T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
        }
    }

    /// Returns a mutable reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let mut x = AdValue::primal(1_i32);
    /// *x.primal_mut() = 7;
    /// assert_eq!(x.primal_ref(), &7);
    /// ```
    pub fn primal_mut(&mut self) -> &mut T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
        }
    }

    /// Returns a reference to tangent payload when available.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 3.0_f64);
    /// assert_eq!(x.tangent_ref(), Some(&3.0));
    /// ```
    pub fn tangent_ref(&self) -> Option<&T> {
        match self {
            Self::Primal(_) => None,
            Self::Forward { tangent, .. } => Some(tangent),
        }
    }

    /// Maps the payload type while preserving AD metadata.
    ///
    /// In reverse mode this preserves the existing `node` / `tape` metadata, so
    /// it is only appropriate for identity-same-cotangent-space transforms.
    /// Dtype-changing or otherwise graph-changing reverse transforms must
    /// register explicit pullback rules instead of using this helper directly.
    ///
    /// # Examples
    ///
    /// ```text
    /// use tenferro::core::AdValue;
    ///
    /// let x = AdValue::forward(2_i32, 3_i32);
    /// let y = x.map_preserving_metadata(|v| v as f64);
    /// assert_eq!(y.primal_ref(), &2.0_f64);
    /// assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    /// ```
    pub fn map_preserving_metadata<U>(self, mut f: impl FnMut(T) -> U) -> AdValue<U> {
        match self {
            Self::Primal(value) => AdValue::Primal(f(value)),
            Self::Forward { primal, tangent } => AdValue::Forward {
                primal: f(primal),
                tangent: f(tangent),
            },
        }
    }
}

impl<T> From<T> for AdValue<T> {
    fn from(value: T) -> Self {
        Self::Primal(value)
    }
}
