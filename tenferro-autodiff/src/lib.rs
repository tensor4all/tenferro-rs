//! Generic automatic differentiation framework.
//!
//! This crate provides type-agnostic AD infrastructure, independent of any
//! specific tensor or array type. It follows the design of Julia's
//! ChainRulesCore.jl: the [`Differentiable`] trait defines the tangent space
//! for any value type, while [`ReverseRule`] and [`ForwardRule`] define
//! per-operation AD rules.
//!
//! - Reverse-mode AD (rrule/pullback) via [`TrackedTensor`]
//! - Forward-mode AD (frule/pushforward) via [`DualTensor`]
//! - Rule extension traits ([`ReverseRule`], [`ForwardRule`])
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
//! use tenferro_autodiff::{TrackedTensor, pullback, clear_tape};
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
//! use tenferro_autodiff::DualTensor;
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
//! use tenferro_autodiff::{TrackedTensor, hvp, clear_tape};
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

/// Trait defining the tangent space for a differentiable type.
///
/// This is the core abstraction of the AD framework, analogous to Julia's
/// ChainRulesCore.jl tangent type system. Any type that participates in
/// automatic differentiation must implement this trait.
///
/// The tangent type represents infinitesimal perturbations of the value.
/// For most tensor types, `Tangent = Self` (e.g., the tangent of a matrix
/// is another matrix of the same shape).
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::Differentiable;
///
/// // Tensor<f64> implements Differentiable with Tangent = Tensor<f64>
/// // (defined in tenferro-tensor crate)
/// fn example<V: Differentiable>(x: &V) {
///     let zero = x.zero_tangent();
///     let _acc = V::accumulate_tangent(zero.clone(), &x.zero_tangent());
/// }
/// ```
pub trait Differentiable: Clone {
    /// The tangent type for this value.
    ///
    /// For most types, this is `Self` (e.g., tangent of a tensor is a tensor).
    type Tangent: Clone;

    /// Returns the zero tangent for this value (additive identity).
    fn zero_tangent(&self) -> Self::Tangent;

    /// Accumulates (adds) two tangents: `a + b`.
    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent;
}

/// AD-specific error type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::AutodiffError;
///
/// let err = AutodiffError::NonScalarLoss { num_elements: 8 };
/// assert!(format!("{err}").contains("scalar"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum AutodiffError {
    /// Loss tensor for pullback must contain exactly one element.
    #[error("pullback() requires scalar loss, got {num_elements} elements")]
    NonScalarLoss { num_elements: usize },
    /// Attempted pullback on a tensor not connected to AD tape.
    #[error("tensor is not connected to AD tape")]
    MissingNode,
    /// Tangent shape must match primal shape.
    #[error("tangent shape mismatch: expected {expected}, got {got}")]
    TangentShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },
    /// A ReverseRule does not support HVP (pullback_with_tangents).
    #[error("HVP not supported by this ReverseRule implementation")]
    HvpNotSupported,
    /// Generic AD argument error.
    #[error("invalid autodiff argument: {0}")]
    InvalidArgument(String),
}

/// Result alias for AD APIs.
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::{AdResult, TrackedTensor, Differentiable};
///
/// fn takes_ad_result<V: Differentiable>(_x: AdResult<TrackedTensor<V>>) {}
/// ```
pub type AdResult<T> = std::result::Result<T, AutodiffError>;

/// Stable identifier of an AD graph node.
///
/// # Examples
///
/// ```
/// use tenferro_autodiff::NodeId;
///
/// let id = NodeId::new(7);
/// assert_eq!(id.index(), 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// Creates a node ID from an integer index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_autodiff::NodeId;
    ///
    /// let id = NodeId::new(42);
    /// assert_eq!(id.index(), 42);
    /// ```
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the numeric index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_autodiff::NodeId;
    ///
    /// let id = NodeId::new(3);
    /// assert_eq!(id.index(), 3);
    /// ```
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Saved-tensor retention policy for reverse-mode rules.
///
/// # Examples
///
/// ```
/// use tenferro_autodiff::SavePolicy;
///
/// let p = SavePolicy::SaveForPullback;
/// assert_eq!(p, SavePolicy::SaveForPullback);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavePolicy {
    /// Keep forward tensors for exact pullback formulas.
    SaveForPullback,
    /// Discard forward tensors and require recomputation/checkpointing later.
    RecomputeOnPullback,
}

/// Value wrapper for reverse-mode AD.
///
/// Wraps any [`Differentiable`] value and connects it to the reverse-mode
/// tape for gradient computation.
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::TrackedTensor;
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
    /// use tenferro_autodiff::TrackedTensor;
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
    /// use tenferro_autodiff::TrackedTensor;
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
    /// use tenferro_autodiff::TrackedTensor;
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
    /// use tenferro_autodiff::TrackedTensor;
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
/// use tenferro_autodiff::DualTensor;
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
    /// use tenferro_autodiff::DualTensor;
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
    /// use tenferro_autodiff::DualTensor;
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
/// use tenferro_autodiff::{Gradients, Differentiable};
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
    /// use tenferro_autodiff::Gradients;
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

/// Reverse-mode AD rule interface (rrule).
///
/// Implemented by operation-specific nodes (einsum, reduce, permute, ...).
/// Named after Julia's ChainRules.jl convention: `rrule` returns a pullback.
///
/// The type parameter `V` is the differentiable value type (e.g., `Tensor<f64>`).
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::{ReverseRule, Differentiable, AdResult, NodeId};
///
/// struct MyRule;
/// impl<V: Differentiable> ReverseRule<V> for MyRule {
///     fn pullback(&self, cotangent: &V::Tangent)
///         -> AdResult<Vec<(NodeId, V::Tangent)>> {
///         todo!()
///     }
///     fn inputs(&self) -> Vec<NodeId> { vec![] }
/// }
/// ```
pub trait ReverseRule<V: Differentiable> {
    /// Computes input cotangents from an output cotangent (pullback).
    fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>>;

    /// Returns input node IDs this rule depends on.
    fn inputs(&self) -> Vec<NodeId>;

    /// Computes pullback with tangent propagation for HVP.
    ///
    /// Given an output cotangent and its tangent, returns
    /// `(node_id, input_cotangent, input_cotangent_tangent)` triples.
    ///
    /// The default implementation returns [`AutodiffError::HvpNotSupported`].
    /// Operations that support forward-over-reverse HVP override this method.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Called internally by hvp(); users rarely call this directly.
    /// let results = rule.pullback_with_tangents(&cotangent, &cotangent_tangent)?;
    /// for (node_id, grad, grad_tangent) in results {
    ///     // grad: standard cotangent for this input
    ///     // grad_tangent: cotangent tangent for HVP
    /// }
    /// ```
    fn pullback_with_tangents(
        &self,
        cotangent: &V::Tangent,
        cotangent_tangent: &V::Tangent,
    ) -> AdResult<Vec<(NodeId, V::Tangent, V::Tangent)>> {
        let _ = (cotangent, cotangent_tangent);
        Err(AutodiffError::HvpNotSupported)
    }
}

/// Forward-mode AD rule interface (frule).
///
/// Named after Julia's ChainRules.jl convention: `frule` computes pushforward.
///
/// The type parameter `V` is the differentiable value type (e.g., `Tensor<f64>`).
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::{ForwardRule, Differentiable, AdResult};
///
/// struct MyFrule;
/// impl<V: Differentiable> ForwardRule<V> for MyFrule {
///     fn pushforward(&self, tangents: &[Option<&V::Tangent>])
///         -> AdResult<V::Tangent> {
///         todo!()
///     }
/// }
/// ```
pub trait ForwardRule<V: Differentiable> {
    /// Computes output tangent from input tangents (pushforward).
    fn pushforward(&self, tangents: &[Option<&V::Tangent>]) -> AdResult<V::Tangent>;
}

/// Compiled pullback execution plan.
///
/// # Examples
///
/// ```ignore
/// let plan = tenferro_autodiff::PullbackPlan::<MyType>::build(&loss).unwrap();
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
    /// let plan = tenferro_autodiff::PullbackPlan::build(&loss).unwrap();
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
    /// use tenferro_autodiff::{PullbackPlan, NodeId};
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
/// tenferro_autodiff::clear_tape::<Tensor<f64>>();
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
/// let grads = tenferro_autodiff::pullback(&loss).unwrap();
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
/// use tenferro_autodiff::{TrackedTensor, hvp, HvpResult};
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
/// use tenferro_autodiff::{TrackedTensor, hvp, clear_tape};
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

// ============================================================================
// Differentiable impl for f64 (enables PullbackPlan doc test)
// ============================================================================

impl Differentiable for f64 {
    type Tangent = f64;

    fn zero_tangent(&self) -> f64 {
        0.0
    }

    fn accumulate_tangent(a: f64, b: &f64) -> f64 {
        a + b
    }
}

impl Differentiable for f32 {
    type Tangent = f32;

    fn zero_tangent(&self) -> f32 {
        0.0
    }

    fn accumulate_tangent(a: f32, b: &f32) -> f32 {
        a + b
    }
}
