use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

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

    /// Maps the payload type while preserving AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2_i32, 3_i32);
    /// let y = x.map(|v| v as f64);
    /// assert_eq!(y.primal_ref(), &2.0_f64);
    /// assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    /// ```
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> AdValue<U> {
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

/// Scalar newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdScalar};
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
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
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
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(primal: T, tangent: T) -> Self {
        Self(AdValue::forward(primal, tangent))
    }

    /// Creates a reverse-mode scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar, NodeId, TapeId};
    ///
    /// let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(2), Some(0.4));
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse(primal: T, node: NodeId, tape: TapeId, tangent: Option<T>) -> Self {
        Self(AdValue::reverse(primal, node, tape, tangent))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
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
    /// use tenferro_dyadtensor::{AdScalar, AdValue};
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
    /// use tenferro_dyadtensor::{AdScalar, AdValue};
    ///
    /// let x = AdScalar::new_primal(2.0_f64).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<T> {
        self.0
    }

    /// Returns primal scalar reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
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
    /// use tenferro_dyadtensor::AdScalar;
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

impl<T> From<AdValue<T>> for AdScalar<T> {
    fn from(value: AdValue<T>) -> Self {
        Self(value)
    }
}

impl<T> From<AdScalar<T>> for AdValue<T> {
    fn from(value: AdScalar<T>) -> Self {
        value.0
    }
}

/// Tensor newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: AdTensor<f64> = t.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Clone)]
pub struct AdTensor<T: Scalar>(pub AdValue<Tensor<T>>);

impl<T: Scalar> AdTensor<T> {
    /// Creates a primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[1]);
    /// ```
    pub fn new_primal(tensor: Tensor<T>) -> Self {
        Self(AdValue::primal(tensor))
    }

    /// Creates a forward-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.1], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(primal: Tensor<T>, tangent: Tensor<T>) -> Self {
        Self(AdValue::forward(primal, tangent))
    }

    /// Creates a reverse-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor, NodeId, TapeId};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_reverse(primal, NodeId(8), TapeId(3), None);
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse(
        primal: Tensor<T>,
        node: NodeId,
        tape: TapeId,
        tangent: Option<Tensor<T>>,
    ) -> Self {
        Self(AdValue::reverse(primal, node, tape, tangent))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
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
    /// use tenferro_dyadtensor::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(matches!(x.as_value(), AdValue::Primal(_)));
    /// ```
    pub fn as_value(&self) -> &AdValue<Tensor<T>> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<Tensor<T>> {
        self.0
    }

    /// Returns primal tensor reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.primal().dims(), &[2]);
    /// ```
    pub fn primal(&self) -> &Tensor<T> {
        self.0.primal_ref()
    }

    /// Returns tangent tensor reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.5], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.tangent().unwrap().dims(), &[1]);
    /// ```
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.0.tangent_ref()
    }

    /// Returns dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.primal().dims()
    }

    /// Returns number of dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Scalar> From<Tensor<T>> for AdTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T: Scalar> From<AdValue<Tensor<T>>> for AdTensor<T> {
    fn from(value: AdValue<Tensor<T>>) -> Self {
        Self(value)
    }
}

impl<T: Scalar> From<AdTensor<T>> for AdValue<Tensor<T>> {
    fn from(value: AdTensor<T>) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn ad_value_map_preserves_mode() {
        let x = AdValue::forward(2_i32, 3_i32);
        let y = x.map(|v| v as f64);
        assert_eq!(y.mode(), AdMode::Forward);
        assert_eq!(y.primal_ref(), &2.0_f64);
        assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    }

    #[test]
    fn ad_tensor_metadata() {
        let tensor =
            Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let ad = AdTensor::new_primal(tensor);
        assert_eq!(ad.mode(), AdMode::Primal);
        assert_eq!(ad.dims(), &[2]);
        assert_eq!(ad.ndim(), 1);
        assert_eq!(ad.len(), 2);
    }
}
