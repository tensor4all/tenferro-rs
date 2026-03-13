/// Automatic differentiation mode.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::AdMode;
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

/// Opaque identifier of a reverse-mode graph node.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::NodeId;
///
/// let node = NodeId(7);
/// assert_eq!(node.0, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Opaque identifier of a tape instance.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::TapeId;
///
/// let tape = TapeId(2);
/// assert_eq!(tape.0, 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeId(pub u64);

/// Generic AD value that can wrap any user-defined payload type `T`.
///
/// This is the primary extension point of the crate.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdValue, NodeId, TapeId};
///
/// let primal = AdValue::primal(3.0_f64);
/// assert_eq!(primal.mode(), AdMode::Primal);
///
/// let dual = AdValue::forward(3.0_f64, 1.0_f64);
/// assert_eq!(dual.mode(), AdMode::Forward);
///
/// let tracked = AdValue::reverse(3.0_f64, NodeId(1), TapeId(9), None);
/// assert_eq!(tracked.mode(), AdMode::Reverse);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AdValue<T> {
    /// Primal-only value.
    Primal(T),
    /// Forward-mode value and tangent.
    Forward { primal: T, tangent: T },
    /// Reverse-mode value with graph metadata.
    Reverse {
        primal: T,
        node: NodeId,
        tape: TapeId,
        tangent: Option<T>,
    },
}

impl<T> AdValue<T> {
    /// Creates a primal-only value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
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
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 1.0_f64);
    /// assert!(matches!(x, AdValue::Forward { .. }));
    /// ```
    pub fn forward(primal: T, tangent: T) -> Self {
        Self::Forward { primal, tangent }
    }

    /// Creates a reverse-mode value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(2.0_f64, NodeId(3), TapeId(5), Some(0.1));
    /// assert!(matches!(x, AdValue::Reverse { .. }));
    /// ```
    pub fn reverse(primal: T, node: NodeId, tape: TapeId, tangent: Option<T>) -> Self {
        Self::Reverse {
            primal,
            node,
            tape,
            tangent,
        }
    }

    /// Returns the AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdValue};
    ///
    /// let x = AdValue::forward(1.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::Primal(_) => AdMode::Primal,
            Self::Forward { .. } => AdMode::Forward,
            Self::Reverse { .. } => AdMode::Reverse,
        }
    }

    /// Returns a reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(10_i32, 1_i32);
    /// assert_eq!(x.primal_ref(), &10);
    /// ```
    pub fn primal_ref(&self) -> &T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a mutable reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let mut x = AdValue::primal(1_i32);
    /// *x.primal_mut() = 7;
    /// assert_eq!(x.primal_ref(), &7);
    /// ```
    pub fn primal_mut(&mut self) -> &mut T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a reference to tangent payload when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 3.0_f64);
    /// assert_eq!(x.tangent_ref(), Some(&3.0));
    /// ```
    pub fn tangent_ref(&self) -> Option<&T> {
        match self {
            Self::Primal(_) => None,
            Self::Forward { tangent, .. } => Some(tangent),
            Self::Reverse { tangent, .. } => tangent.as_ref(),
        }
    }

    /// Returns reverse-mode node id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.node_id(), Some(NodeId(4)));
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::Reverse { node, .. } => Some(*node),
            _ => None,
        }
    }

    /// Returns reverse-mode tape id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.tape_id(), Some(TapeId(6)));
    /// ```
    pub fn tape_id(&self) -> Option<TapeId> {
        match self {
            Self::Reverse { tape, .. } => Some(*tape),
            _ => None,
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
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
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
            Self::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => AdValue::Reverse {
                primal: f(primal),
                node,
                tape,
                tangent: tangent.map(f),
            },
        }
    }
}

impl<T> From<T> for AdValue<T> {
    fn from(value: T) -> Self {
        Self::Primal(value)
    }
}
