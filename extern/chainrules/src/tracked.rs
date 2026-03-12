use crate::tape::Tape;
use crate::{Differentiable, NodeId};

/// Value wrapper for reverse-mode AD.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::ones(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// assert!(a.requires_grad());
/// ```
pub struct TrackedTensor<V: Differentiable> {
    pub(crate) value: V,
    pub(crate) node_id: Option<NodeId>,
    pub(crate) tape: Option<Tape<V>>,
    pub(crate) requires_grad: bool,
    pub(crate) tangent: Option<V::Tangent>,
}

impl<V: Differentiable> TrackedTensor<V> {
    /// Creates a tracked value with `requires_grad = false` (no tape).
    pub fn new(value: V) -> Self {
        Self {
            value,
            node_id: None,
            tape: None,
            requires_grad: false,
            tangent: None,
        }
    }

    /// Returns the underlying value.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes and returns the underlying value.
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns whether this value participates in gradient propagation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the graph node ID when this value is connected to a tape.
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Returns the tangent for HVP, or `None` if not set.
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this tracked value has a tangent for HVP.
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Returns a reference to the tape this value is connected to, if any.
    pub fn tape(&self) -> Option<&Tape<V>> {
        self.tape.as_ref()
    }

    /// Consumes and returns a detached value that does not require gradients.
    pub fn detach(self) -> Self {
        Self {
            value: self.value,
            node_id: None,
            tape: None,
            requires_grad: false,
            tangent: None,
        }
    }
}
