use std::sync::{Arc, Mutex, MutexGuard};

use crate::engine::{AutogradGraph, Gradients, TrackedValue, VariableNodeKind};
use crate::{AdResult, AutodiffError, Differentiable, HvpResult, NodeId, ReverseRule};

/// Reverse-mode AD tape.
///
/// The tape records operations performed on [`TrackedValue`] values and
/// enables gradient computation via [`Tape::pullback`] or HVP via
/// [`Tape::hvp`].
///
/// `Tape` is cheaply cloneable (internally reference-counted). Multiple
/// clones refer to the same underlying autograd graph.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use std::sync::{Arc, Mutex};
/// use tenferro_algebra::Standard;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
/// let a = tape.leaf(Tensor::ones(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let b = tape.leaf(Tensor::ones(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let c =
///     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss =
///     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
pub struct Tape<V: Differentiable> {
    inner: Arc<Mutex<AutogradGraph<V>>>,
}

impl<V: Differentiable> Tape<V> {
    fn lock_graph(&self) -> MutexGuard<'_, AutogradGraph<V>> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Creates a new empty tape.
    pub fn new() -> Self {
        Self {
            inner: AutogradGraph::new(),
        }
    }

    /// Returns `true` if `self` and `other` are the same tape.
    pub fn same_tape(&self, other: &Tape<V>) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns a stable process-local identifier for this tape.
    pub fn id(&self) -> usize {
        self.lock_graph().id() as usize
    }

    /// Returns the current number of nodes recorded on this tape.
    pub fn node_count(&self) -> usize {
        self.lock_graph().node_count()
    }

    /// Creates a leaf value requiring gradient on this tape.
    pub fn leaf(&self, value: V) -> TrackedValue<V> {
        let node_id = self.lock_graph().record_leaf(None);
        TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: None,
        }
    }

    /// Creates a leaf value with a tangent for HVP computation.
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedValue<V>> {
        let node_id = self.lock_graph().record_leaf(Some(tangent.clone()));
        Ok(TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: Some(tangent),
        })
    }

    /// Records an output value on the tape before attaching its reverse rule.
    pub fn placeholder(&self, value: V, tangent: Option<V::Tangent>) -> TrackedValue<V> {
        let node_id = self.lock_graph().record_placeholder(tangent.clone());
        TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent,
        }
    }

    /// Reconstructs a tracked handle for an existing node already recorded on
    /// this tape.
    pub fn tracked_existing(
        &self,
        node_id: NodeId,
        value: V,
        tangent: Option<V::Tangent>,
    ) -> AdResult<TrackedValue<V>> {
        let guard = self.lock_graph();
        if !guard.has_node(node_id) {
            return Err(AutodiffError::InvalidArgument(format!(
                "node {} is not present on this tape",
                node_id.index()
            )));
        }
        Ok(TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent,
        })
    }

    /// Records an operation on the tape, returning a tracked output.
    pub fn record_op(
        &self,
        output_value: V,
        rule: Box<dyn ReverseRule<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedValue<V> {
        let node_id =
            self.lock_graph()
                .record_op(rule, output_tangent.clone(), VariableNodeKind::Custom);
        TrackedValue {
            value: output_value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: output_tangent,
        }
    }

    /// Attaches or replaces the reverse rule for an existing output node.
    pub fn attach_rule(&self, node_id: NodeId, rule: Box<dyn ReverseRule<V>>) -> AdResult<()> {
        self.lock_graph().attach_rule(node_id, rule)
    }

    /// Runs reverse-mode pullback from a scalar loss value.
    pub fn pullback(&self, loss: &TrackedValue<V>) -> AdResult<Gradients<V>> {
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        self.pullback_with_seed(loss, loss.value.seed_cotangent())
    }

    /// Runs reverse-mode pullback from an arbitrary output cotangent seed.
    pub fn pullback_with_seed(
        &self,
        output: &TrackedValue<V>,
        seed: V::Tangent,
    ) -> AdResult<Gradients<V>> {
        let output_node = output.node_id.ok_or(AutodiffError::MissingNode)?;
        let guard = self.lock_graph();
        guard.ensure_alive()?;
        guard.pullback_from(output_node, seed)
    }

    /// Computes gradient and Hessian-vector product via forward-over-reverse.
    pub fn hvp(&self, loss: &TrackedValue<V>) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Differentiable<Tangent = V::Tangent>,
    {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        let guard = self.lock_graph();
        guard.ensure_alive()?;
        guard.hvp_from(
            loss_node,
            loss.value.seed_cotangent(),
            loss.value.zero_tangent(),
        )
    }

    /// Marks the current graph as freed.
    pub fn free_graph(&self) {
        self.lock_graph().free_graph();
    }
}

impl<V: Differentiable> Clone for Tape<V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<V: Differentiable> Default for Tape<V> {
    fn default() -> Self {
        Self::new()
    }
}
