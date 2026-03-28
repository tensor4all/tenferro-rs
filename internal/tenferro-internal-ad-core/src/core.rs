pub use chainrules_core::NodeId;

/// Automatic differentiation mode.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_ad_core::AdMode;
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

/// Generic AD value used by unit tests to validate metadata-preserving helpers.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_ad_core::{AdMode, AdValue};
///
/// let primal = AdValue::primal(3.0_f64);
/// assert_eq!(primal.mode(), AdMode::Primal);
///
/// let dual = AdValue::forward(3.0_f64, 1.0_f64);
/// assert_eq!(dual.mode(), AdMode::Forward);
/// ```
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub enum AdValue<T> {
    Primal(T),
    Forward { primal: T, tangent: T },
}

impl<T> AdValue<T> {
    pub fn primal(value: T) -> Self {
        Self::Primal(value)
    }

    pub fn forward(primal: T, tangent: T) -> Self {
        Self::Forward { primal, tangent }
    }

    pub fn mode(&self) -> AdMode {
        match self {
            Self::Primal(_) => AdMode::Primal,
            Self::Forward { .. } => AdMode::Forward,
        }
    }

    pub fn primal_ref(&self) -> &T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
        }
    }

    pub fn tangent_ref(&self) -> Option<&T> {
        match self {
            Self::Primal(_) => None,
            Self::Forward { tangent, .. } => Some(tangent),
        }
    }

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
