//! AD engine: tape-based reverse-mode and dual-number forward-mode.
//!
//! This crate provides the AD execution engine, built on top of
//! [`chainrules_core`] traits. It is analogous to Zygote.jl in the Julia
//! ecosystem: a concrete AD engine that uses ChainRulesCore.jl interfaces.
//!
//! - Reverse-mode AD via [`Tape`], [`TrackedTensor`], and [`Tape::pullback`]
//! - Forward-mode AD via [`DualTensor`]
//! - Forward-over-reverse HVP via [`Tape::hvp`]
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
//! use chainrules::{Tape, TrackedTensor};
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let tape = Tape::<Tensor<f64>>::new();
//! let a = tape.leaf(Tensor::ones(
//!     &[2, 3],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let b = tape.leaf(Tensor::ones(
//!     &[3, 4],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ));
//! let c = tracked_einsum("ij,jk->ik", &[&a, &b]).unwrap();
//! let loss = tracked_einsum("ij,ij->", &[&c, &c]).unwrap();
//! let grads = tape.pullback(&loss).unwrap();
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
//! use chainrules::Tape;
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let tape = Tape::<Tensor<f64>>::new();
//! let x = tape.leaf_with_tangent(
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),  // direction v
//! ).unwrap();
//! let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();  // f(x) = x·x
//! let result = tape.hvp(&loss).unwrap();
//! let _grad = result.gradients;  // ∇f(x) = 2x
//! let _hv = result.hvp;          // H·v = 2v
//! ```

// Re-export all core traits so downstream can depend on just `chainrules`.
pub use chainrules_core::*;

use std::marker::PhantomData;

/// Reverse-mode AD tape.
///
/// The tape records operations performed on [`TrackedTensor`] values and
/// enables gradient computation via [`Tape::pullback`] or HVP via
/// [`Tape::hvp`].
///
/// Create leaf values with [`Tape::leaf`], perform operations using
/// AD-aware functions (e.g., `tracked_einsum`), then call
/// [`Tape::pullback`] on the scalar loss to compute gradients.
///
/// `Tape` is cheaply cloneable (internally reference-counted). Multiple
/// clones refer to the same underlying tape.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
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
/// let c = tracked_einsum("ij,jk->ik", &[&a, &b]).unwrap();
/// let loss = tracked_einsum("ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
pub struct Tape<V: Differentiable> {
    _marker: PhantomData<V>,
}

impl<V: Differentiable> Tape<V> {
    /// Creates a new empty tape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Creates a leaf value requiring gradient on this tape.
    ///
    /// The returned [`TrackedTensor`] is connected to this tape and
    /// will participate in gradient computation via [`Tape::pullback`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(3.14);
    /// assert!(x.requires_grad());
    /// ```
    pub fn leaf(&self, _value: V) -> TrackedTensor<V> {
        todo!()
    }

    /// Creates a leaf value with a tangent for HVP computation.
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
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf_with_tangent(3.14, 1.0).unwrap();
    /// assert!(x.requires_grad());
    /// assert!(x.has_tangent());
    /// ```
    pub fn leaf_with_tangent(&self, _value: V, _tangent: V::Tangent) -> AdResult<TrackedTensor<V>> {
        todo!()
    }

    /// Runs reverse-mode pullback from a scalar loss value.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::NonScalarLoss`] for non-scalar losses.
    /// Returns [`AutodiffError::MissingNode`] if the loss is not connected
    /// to this tape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// // ... compute loss from x ...
    /// let grads = tape.pullback(&x).unwrap();
    /// ```
    pub fn pullback(&self, _loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
        todo!()
    }

    /// Computes gradient and Hessian-vector product via forward-over-reverse.
    ///
    /// Leaf values with tangents (created via [`Tape::leaf_with_tangent`])
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
    /// use chainrules::Tape;
    /// use tenferro_einsum::tracked_einsum;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let tape = Tape::<Tensor<f64>>::new();
    /// let x = tape.leaf_with_tangent(
    ///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
    ///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
    /// ).unwrap();
    /// let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();
    /// let result = tape.hvp(&loss).unwrap();
    /// let _grad = result.gradients;
    /// let _hv = result.hvp;
    /// ```
    pub fn hvp(&self, _loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>> {
        todo!()
    }
}

impl<V: Differentiable> Clone for Tape<V> {
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<V: Differentiable> Default for Tape<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Value wrapper for reverse-mode AD.
///
/// Wraps any [`Differentiable`] value and connects it to a [`Tape`]
/// for gradient computation.
///
/// Created via [`Tape::leaf`] for gradient-tracked values, or
/// [`TrackedTensor::new`] for values that do not require gradients.
///
/// # Examples
///
/// ```ignore
/// use chainrules::{Tape, TrackedTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
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
    value: V,
    node_id: Option<NodeId>,
    tape: Option<Tape<V>>,
    requires_grad: bool,
    tangent: Option<V::Tangent>,
}

impl<V: Differentiable> TrackedTensor<V> {
    /// Creates a tracked value with `requires_grad = false` (no tape).
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
            tape: None,
            requires_grad: false,
            tangent: None,
        }
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

    /// Returns the graph node ID when this value is connected to a tape.
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

    /// Consumes and returns a detached value that does not require gradients.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let detached = tracked.detach();
    /// assert!(!detached.requires_grad());
    /// ```
    pub fn detach(self) -> Self {
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

    /// Consumes and returns a dual value with tangent removed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let c = dual.detach_tangent();
    /// assert!(!c.has_tangent());
    /// ```
    pub fn detach_tangent(self) -> Self {
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
    _marker: PhantomData<V>,
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

/// Result of a forward-over-reverse HVP computation.
///
/// Contains both the standard gradient and the Hessian-vector
/// product H*v, where v is the tangent direction set on leaf values
/// via [`Tape::leaf_with_tangent`].
///
/// # Examples
///
/// ```ignore
/// use chainrules::{Tape, HvpResult};
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let x = tape.leaf_with_tangent(
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
/// ).unwrap();
/// let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();
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
