use std::marker::PhantomData;

use crate::{AdResult, AutodiffError, Differentiable, NodeId, TrackedValue};

/// Accumulated gradients indexed by [`NodeId`].
///
/// # Examples
///
/// ```
/// use chainrules::{Gradients, NodeId};
///
/// let mut grads = Gradients::<f64>::new();
/// grads.accumulate(NodeId::new(0), 3.0).unwrap();
/// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 3.0);
/// ```
pub struct Gradients<V: Differentiable> {
    entries: Vec<(NodeId, V::Tangent)>,
}

impl<V: Differentiable> Gradients<V> {
    /// Creates an empty gradient container.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Gradients;
    /// let grads = Gradients::<f64>::new();
    /// assert!(grads.entries().is_empty());
    /// ```
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    pub(crate) fn push_entry(&mut self, node: NodeId, grad: V::Tangent) {
        self.entries.push((node, grad));
    }

    /// Returns the gradient for `node`, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 5.0).unwrap();
    /// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
    /// assert!(grads.get(NodeId::new(1)).is_none());
    /// ```
    pub fn get(&self, node: NodeId) -> Option<&V::Tangent> {
        self.entries
            .iter()
            .find(|(id, _)| *id == node)
            .map(|(_, grad)| grad)
    }

    /// Inserts or accumulates a gradient for `node`.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 2.0).unwrap();
    /// grads.accumulate(NodeId::new(0), 3.0).unwrap();
    /// assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
    /// ```
    pub fn accumulate(&mut self, node: NodeId, grad: V::Tangent) -> AdResult<()> {
        if let Some(entry) = self.entries.iter_mut().find(|(id, _)| *id == node) {
            let existing = entry.1.clone();
            entry.1 = V::accumulate_tangent(existing, &grad);
        } else {
            self.entries.push((node, grad));
        }
        Ok(())
    }

    /// Returns all `(node, grad)` entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{Gradients, NodeId};
    ///
    /// let mut grads = Gradients::<f64>::new();
    /// grads.accumulate(NodeId::new(0), 1.0).unwrap();
    /// assert_eq!(grads.entries().len(), 1);
    /// ```
    pub fn entries(&self) -> &[(NodeId, V::Tangent)] {
        &self.entries
    }
}

impl<V: Differentiable> Default for Gradients<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled pullback execution plan.
///
/// # Examples
///
/// ```ignore
/// let plan = chainrules::PullbackPlan::<MyType>::build(&loss).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PullbackPlan<V: Differentiable> {
    loss: NodeId,
    _marker: PhantomData<V>,
}

impl<V: Differentiable> PullbackPlan<V> {
    /// Builds a pullback plan from a loss value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{PullbackPlan, Tape};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// assert_eq!(plan.loss_node().index(), 0);
    /// ```
    pub fn build(loss: &TrackedValue<V>) -> AdResult<Self> {
        let node_id = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        Ok(Self {
            loss: node_id,
            _marker: PhantomData,
        })
    }

    /// Executes the pre-built pullback plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{PullbackPlan, Tape};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// let grads = plan.execute(&x).unwrap();
    /// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    /// ```
    pub fn execute(&self, loss: &TrackedValue<V>) -> AdResult<Gradients<V>> {
        let tape = loss.tape.as_ref().ok_or(AutodiffError::MissingNode)?;
        tape.pullback(loss)
    }

    /// Returns loss node ID for this plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{NodeId, PullbackPlan};
    /// let _id_fn: fn(&PullbackPlan<f64>) -> NodeId = PullbackPlan::loss_node;
    /// ```
    pub fn loss_node(&self) -> NodeId {
        self.loss
    }
}

/// Result of a forward-over-reverse HVP computation.
///
/// Contains both the standard gradient and the Hessian-vector
/// product H*v, where v is the tangent direction set on leaf values
/// via [`crate::Tape::leaf_with_tangent`].
///
/// # Examples
///
/// ```ignore
/// use chainrules::{HvpResult, Tape};
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use tenferro_algebra::Standard;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
/// let x = tape.leaf_with_tangent(
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
/// ).unwrap();
/// let loss =
///     tracked_einsum::<Standard<f64>, CpuBackend>(ctx, "i,i->", &[&x, &x]).unwrap();
/// let result: HvpResult<Tensor<f64>> = tape.hvp(&loss).unwrap();
/// let _grad = result.gradients.get(x.node_id().unwrap());
/// let _hv = result.hvp.get(x.node_id().unwrap());
/// ```
pub struct HvpResult<V: Differentiable> {
    /// Gradients.
    pub gradients: Gradients<V>,
    /// Hessian-vector product: H*v.
    pub hvp: Gradients<V>,
}
