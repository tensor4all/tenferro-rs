//! AD engine: tape-based reverse-mode and dual-number forward-mode.
//!
//! This crate provides the AD execution engine, built on top of
//! [`chainrules_core`] traits. It is analogous to Zygote.jl in the Julia
//! ecosystem: a concrete AD engine that uses ChainRulesCore.jl interfaces.
//!
//! - Reverse-mode AD via [`TrackedTensor`] and [`pullback`]
//! - Forward-mode AD via [`DualTensor`]
//! - Forward-over-reverse HVP via [`hvp`]
//!
//! Operation-specific AD rules (e.g., einsum rrule/frule) live in the crate
//! that defines the operation. See `tenferro-einsum` for einsum AD functions.
//!
//! Bodies are intentionally `todo!()` in the current POC phase.
//!
//! # Examples
//!
//! Reverse-mode usage (with operation-specific AD functions from other crates):
//!
//! ```ignore
//! use chainrules::{TrackedTensor, pullback, clear_tape};
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! clear_tape::<Tensor<f64>>();
//! let a = TrackedTensor::leaf(Tensor::ones(
//!     &[2, 3],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let b = TrackedTensor::leaf(Tensor::ones(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let c = tracked_einsum("ij,jk->ik", &[&a, &b]).unwrap();
//! let loss = tracked_einsum("ij,ij->", &[&c, &c]).unwrap();
//! let grads = pullback(&loss).unwrap();
//! let _ga = grads.get(a.node_id().unwrap()).unwrap();
//! ```
//!
//! Forward-mode usage:
//!
//! ```ignore
//! use chainrules::DualTensor;
//! use tenferro_einsum::dual_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
//! let da = Tensor::<f64>::ones(&[2, 2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let b = Tensor::<f64>::ones(&[2, 2], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//!
//! let a_dual = DualTensor::with_tangent(a, da).unwrap();
//! let b_dual = DualTensor::new(b);
//! let c_dual = dual_einsum("ij,jk->ik", &[&a_dual, &b_dual]).unwrap();
//! let _jvp = c_dual.tangent();
//! ```
//!
//! Forward-over-reverse HVP (Hessian-vector product):
//!
//! ```ignore
//! use chainrules::{TrackedTensor, hvp, clear_tape};
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! clear_tape::<Tensor<f64>>();
//! let x = TrackedTensor::leaf_with_tangent(
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),  // direction v
//! ).unwrap();
//! let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();  // f(x) = x·x
//! let result = hvp(&loss).unwrap();
//! let _grad = result.gradients;  // ∇f(x) = 2x
//! let _hv = result.hvp;          // H·v = 2v
//! ```

// Re-export all core traits so downstream can depend on just `chainrules`.
pub use chainrules_core::*;

/// Value wrapper for reverse-mode AD.
///
/// Wraps any [`Differentiable`] value and connects it to the reverse-mode
/// tape for gradient computation.
///
/// # Examples
///
/// ```ignore
/// use chainrules::TrackedTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let a = TrackedTensor::leaf(Tensor::<f64>::ones(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// assert!(a.requires_grad());
/// ```
pub struct TrackedTensor<V: Differentiable> {
    value: V,
    node_id: Option<NodeId>,
    requires_grad: bool,
    tangent: Option<V::Tangent>,
}

impl<V: Differentiable> TrackedTensor<V> {
    /// Creates a tracked value with `requires_grad = false`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(value);
    /// assert!(!x.requires_grad());
    /// ```
    pub fn new(value: V) -> Self {
        Self {
            value,
            node_id: None,
            requires_grad: false,
            tangent: None,
        }
    }

    /// Creates a leaf value requiring gradient.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::leaf(value);
    /// assert!(x.requires_grad());
    /// ```
    pub fn leaf(_value: V) -> Self {
        todo!()
    }

    /// Creates a tracked value with an explicit `requires_grad` flag.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::with_requires_grad(value, true);
    /// ```
    pub fn with_requires_grad(_value: V, _requires_grad: bool) -> Self {
        todo!()
    }

    /// Returns the underlying value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let v = tracked.value();
    /// ```
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes and returns the underlying value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let raw = tracked.into_value();
    /// ```
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns whether this value participates in gradient propagation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert!(tracked.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the graph node ID when this value is connected to the tape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(id) = tracked.node_id() {
    ///     println!("node = {}", id.index());
    /// }
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Creates a leaf value requiring gradient, with a tangent for HVP.
    ///
    /// The tangent defines the perturbation direction *v* used in
    /// forward-over-reverse Hessian-vector product computation.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::TangentShapeMismatch`] if shapes differ.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::leaf_with_tangent(value, tangent).unwrap();
    /// assert!(x.requires_grad());
    /// assert!(x.has_tangent());
    /// ```
    pub fn leaf_with_tangent(_value: V, _tangent: V::Tangent) -> AdResult<Self> {
        todo!()
    }

    /// Returns the tangent for HVP, or `None` if not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(t) = tracked.tangent() {
    ///     // use tangent
    /// }
    /// ```
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this tracked value has a tangent for HVP.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert!(tracked.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Returns a detached value that does not require gradients.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let detached = tracked.detach();
    /// assert!(!detached.requires_grad());
    /// ```
    pub fn detach(&self) -> Self {
        todo!()
    }
}

/// Value wrapper for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// let dual = DualTensor::new(primal);
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
    /// ```ignore
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(primal);
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
    /// Returns [`AutodiffError::TangentShapeMismatch`] if shapes differ.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(primal, tangent).unwrap();
    /// ```
    pub fn with_tangent(_primal: V, _tangent: V::Tangent) -> AdResult<Self> {
        todo!()
    }

    /// Returns the primal value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let p = dual.primal();
    /// ```
    pub fn primal(&self) -> &V {
        &self.primal
    }

    /// Returns the tangent, or `None` for zero tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let maybe_t = dual.tangent();
    /// ```
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this dual value has a non-zero tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert!(dual.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Consumes and returns `(primal, tangent)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let (p, t) = dual.into_parts();
    /// ```
    pub fn into_parts(self) -> (V, Option<V::Tangent>) {
        (self.primal, self.tangent)
    }

    /// Returns a dual value with tangent detached (set to zero).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let c = dual.detach_tangent();
    /// assert!(!c.has_tangent());
    /// ```
    pub fn detach_tangent(&self) -> Self {
        todo!()
    }
}

/// Accumulated gradients indexed by [`NodeId`].
///
/// # Examples
///
/// ```ignore
/// use chainrules::{Gradients, Differentiable};
/// // V::Tangent is the gradient type
/// let mut grads = Gradients::<MyType>::new();
/// ```
pub struct Gradients<V: Differentiable> {
    entries: Vec<(NodeId, V::Tangent)>,
}

impl<V: Differentiable> Gradients<V> {
    /// Creates an empty gradient container.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Gradients;
    /// let grads = Gradients::<MyType>::new();
    /// ```
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    /// Returns the gradient for `node`, if present.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(g) = grads.get(node) {
    ///     // use gradient
    /// }
    /// ```
    pub fn get(&self, _node: NodeId) -> Option<&V::Tangent> {
        todo!()
    }

    /// Inserts or accumulates a gradient for `node`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// grads.accumulate(node, grad);
    /// ```
    pub fn accumulate(&mut self, _node: NodeId, _grad: V::Tangent) -> AdResult<()> {
        todo!()
    }

    /// Returns all `(node, grad)` entries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for (node, grad) in grads.entries() {
    ///     println!("{}", node.index());
    /// }
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
    _marker: std::marker::PhantomData<V>,
}

impl<V: Differentiable> PullbackPlan<V> {
    /// Builds a pullback plan from a loss value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let plan = chainrules::PullbackPlan::build(&loss).unwrap();
    /// ```
    pub fn build(_loss: &TrackedTensor<V>) -> AdResult<Self> {
        todo!()
    }

    /// Executes the pre-built pullback plan.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let grads = plan.execute(&loss).unwrap();
    /// ```
    pub fn execute(&self, _loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
        todo!()
    }

    /// Returns loss node ID for this plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{PullbackPlan, NodeId};
    /// let _id_fn: fn(&PullbackPlan<f64>) -> NodeId = PullbackPlan::loss_node;
    /// ```
    pub fn loss_node(&self) -> NodeId {
        self.loss
    }
}

/// Clears the current reverse-mode tape/graph for type `V`.
///
/// # Examples
///
/// ```ignore
/// chainrules::clear_tape::<Tensor<f64>>();
/// ```
pub fn clear_tape<V: Differentiable>() {
    let _ = std::marker::PhantomData::<V>;
    todo!()
}

/// Runs reverse-mode pullback from a scalar loss value.
///
/// # Errors
///
/// Returns [`AutodiffError::NonScalarLoss`] for non-scalar losses.
///
/// # Examples
///
/// ```ignore
/// let grads = chainrules::pullback(&loss).unwrap();
/// ```
pub fn pullback<V: Differentiable>(_loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
    todo!()
}

/// Result of a forward-over-reverse HVP computation.
///
/// Contains both the standard gradient and the Hessian-vector
/// product H*v, where v is the tangent direction set on leaf values
/// via [`TrackedTensor::leaf_with_tangent`].
///
/// # Examples
///
/// ```ignore
/// use chainrules::{TrackedTensor, hvp, HvpResult};
/// use tenferro_einsum::tracked_einsum;
///
/// let result: HvpResult<Tensor<f64>> = hvp(&loss).unwrap();
/// let _grad = result.gradients.get(x.node_id().unwrap());
/// let _hv = result.hvp.get(x.node_id().unwrap());
/// ```
pub struct HvpResult<V: Differentiable> {
    /// Gradients.
    pub gradients: Gradients<V>,
    /// Hessian-vector product: H*v.
    pub hvp: Gradients<V>,
}

/// Computes gradient and Hessian-vector product via forward-over-reverse.
///
/// Leaf values with tangents (created via [`TrackedTensor::leaf_with_tangent`])
/// define the direction *v*. The function runs pullback through the tape,
/// propagating both cotangents and cotangent-tangents at each node.
///
/// Returns both the gradient (in [`HvpResult::gradients`]) and H*v (in
/// [`HvpResult::hvp`]).
///
/// # Errors
///
/// Returns [`AutodiffError::NonScalarLoss`] for non-scalar losses.
/// Returns [`AutodiffError::HvpNotSupported`] if any ReverseRule on the tape
/// does not implement `pullback_with_tangents`.
///
/// # Examples
///
/// ```ignore
/// use chainrules::{TrackedTensor, hvp, clear_tape};
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// clear_tape::<Tensor<f64>>();
/// let x = TrackedTensor::leaf_with_tangent(
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
/// ).unwrap();
/// let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();  // f(x) = x*x
/// let result = hvp(&loss).unwrap();
/// let _grad = result.gradients;
/// let _hv = result.hvp;
/// ```
pub fn hvp<V: Differentiable>(_loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>> {
    todo!()
}
