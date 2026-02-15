//! Automatic differentiation framework for tenferro.
//!
//! This crate provides the core AD infrastructure:
//! - reverse-mode AD (rrule/pullback) via [`TrackedTensor`]
//! - forward-mode AD (frule/pushforward) via [`DualTensor`]
//! - rule extension traits ([`ReverseRule`], [`ForwardRule`])
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
//! use tenferro_autodiff::{TrackedTensor, backward, clear_tape};
//! use tenferro_einsum::tracked_einsum;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//! use tenferro_device::LogicalMemorySpace;
//!
//! clear_tape::<f64>();
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
//! let grads = backward(&loss).unwrap();
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
//! clear_tape::<f64>();
//! let x = TrackedTensor::leaf_with_tangent(
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
//!     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),  // direction v
//! ).unwrap();
//! let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();  // f(x) = x·x
//! let result = hvp(&loss).unwrap();
//! let _grad = result.gradients;  // ∇f(x) = 2x
//! let _hv = result.hvp;          // H·v = 2v
//! ```

use strided_traits::ScalarBase;
use tenferro_algebra::HasAlgebra;
use tenferro_device::Error as DeviceError;
use tenferro_tensor::Tensor;

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
    /// Wrapped error from tenferro shared device/result layer.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// Loss tensor for backward pass must contain exactly one element.
    #[error("backward() requires scalar loss, got {num_elements} elements")]
    NonScalarLoss { num_elements: usize },
    /// Attempted backward on a tensor not connected to AD tape.
    #[error("tensor is not connected to AD tape")]
    MissingNode,
    /// Tangent shape must match primal shape.
    #[error("tangent shape mismatch: expected {expected:?}, got {got:?}")]
    TangentShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
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
/// use tenferro_autodiff::{AdResult, TrackedTensor};
///
/// fn takes_ad_result(_x: AdResult<TrackedTensor<f64>>) {}
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
/// let p = SavePolicy::SaveForBackward;
/// assert_eq!(p, SavePolicy::SaveForBackward);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavePolicy {
    /// Keep forward tensors for exact backward formulas.
    SaveForBackward,
    /// Discard forward tensors and require recomputation/checkpointing later.
    RecomputeOnBackward,
}

/// Tensor wrapper for reverse-mode AD.
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
pub struct TrackedTensor<T: ScalarBase> {
    tensor: Tensor<T>,
    node_id: Option<NodeId>,
    requires_grad: bool,
    tangent: Option<Tensor<T>>,
}

impl<T: ScalarBase> TrackedTensor<T> {
    /// Creates a tracked tensor with `requires_grad = false`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_autodiff::TrackedTensor;
    /// let x = TrackedTensor::new(tensor);
    /// assert!(!x.requires_grad());
    /// ```
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            tensor,
            node_id: None,
            requires_grad: false,
            tangent: None,
        }
    }

    /// Creates a leaf tensor requiring gradient.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_autodiff::TrackedTensor;
    /// let x = TrackedTensor::leaf(tensor);
    /// assert!(x.requires_grad());
    /// ```
    pub fn leaf(_tensor: Tensor<T>) -> Self {
        todo!()
    }

    /// Creates a tracked tensor with an explicit `requires_grad` flag.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_autodiff::TrackedTensor;
    /// let x = TrackedTensor::with_requires_grad(tensor, true);
    /// ```
    pub fn with_requires_grad(_tensor: Tensor<T>, _requires_grad: bool) -> Self {
        todo!()
    }

    /// Returns the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tracked.tensor().tensor_view();
    /// ```
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Consumes and returns the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let raw = tracked.into_tensor();
    /// ```
    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    /// Returns whether this tensor participates in gradient propagation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert!(tracked.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the graph node ID when this tensor is connected to the tape.
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

    /// Creates a leaf tensor requiring gradient, with a tangent for HVP.
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
    /// let x = TrackedTensor::leaf_with_tangent(tensor, tangent).unwrap();
    /// assert!(x.requires_grad());
    /// assert!(x.has_tangent());
    /// ```
    pub fn leaf_with_tangent(_tensor: Tensor<T>, _tangent: Tensor<T>) -> AdResult<Self> {
        todo!()
    }

    /// Returns the tangent tensor for HVP, or `None` if not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(t) = tracked.tangent() {
    ///     println!("tangent shape: {:?}", t.dims());
    /// }
    /// ```
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.tangent.as_ref()
    }

    /// Returns whether this tracked tensor has a tangent for HVP.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert!(tracked.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Returns a detached tensor that does not require gradients.
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

/// Tensor wrapper for forward-mode AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::DualTensor;
/// let dual = DualTensor::new(primal);
/// assert!(!dual.has_tangent());
/// ```
pub struct DualTensor<T: ScalarBase> {
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
}

impl<T: ScalarBase> DualTensor<T> {
    /// Creates a dual tensor with zero tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_autodiff::DualTensor;
    /// let x = DualTensor::new(primal);
    /// ```
    pub fn new(primal: Tensor<T>) -> Self {
        Self {
            primal,
            tangent: None,
        }
    }

    /// Creates a dual tensor with explicit tangent.
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
    pub fn with_tangent(_primal: Tensor<T>, _tangent: Tensor<T>) -> AdResult<Self> {
        todo!()
    }

    /// Returns the primal value tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let p = dual.primal();
    /// ```
    pub fn primal(&self) -> &Tensor<T> {
        &self.primal
    }

    /// Returns the tangent tensor, or `None` for zero tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let maybe_t = dual.tangent();
    /// ```
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.tangent.as_ref()
    }

    /// Returns whether this dual tensor has a non-zero tangent.
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
    pub fn into_parts(self) -> (Tensor<T>, Option<Tensor<T>>) {
        (self.primal, self.tangent)
    }

    /// Returns a dual tensor with tangent detached (set to zero).
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
/// use tenferro_autodiff::Gradients;
/// let mut grads = Gradients::<f64>::new();
/// ```
pub struct Gradients<T: ScalarBase> {
    entries: Vec<(NodeId, Tensor<T>)>,
}

impl<T: ScalarBase> Gradients<T> {
    /// Creates an empty gradient container.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_autodiff::Gradients;
    /// let grads = Gradients::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self { entries: vec![] }
    }

    /// Returns the gradient tensor for `node`, if present.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if let Some(g) = grads.get(node) {
    ///     println!("{:?}", g.dims());
    /// }
    /// ```
    pub fn get(&self, _node: NodeId) -> Option<&Tensor<T>> {
        todo!()
    }

    /// Inserts or accumulates a gradient tensor for `node`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// grads.accumulate(node, grad);
    /// ```
    pub fn accumulate(&mut self, _node: NodeId, _grad: Tensor<T>) -> AdResult<()> {
        todo!()
    }

    /// Returns all `(node, grad)` entries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for (node, grad) in grads.entries() {
    ///     println!("{} {:?}", node.index(), grad.dims());
    /// }
    /// ```
    pub fn entries(&self) -> &[(NodeId, Tensor<T>)] {
        &self.entries
    }
}

impl<T: ScalarBase> Default for Gradients<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reverse-mode AD rule interface (rrule).
///
/// Implemented by operation-specific nodes (einsum, reduce, permute, ...).
/// Named after Julia's ChainRules.jl convention: `rrule` returns a pullback.
///
/// # Examples
///
/// ```ignore
/// struct MyRule;
/// impl tenferro_autodiff::ReverseRule<f64> for MyRule {
///     fn pullback(&self, cotangent: &tenferro_tensor::Tensor<f64>)
///         -> tenferro_autodiff::AdResult<Vec<(tenferro_autodiff::NodeId, tenferro_tensor::Tensor<f64>)>> {
///         todo!()
///     }
///     fn inputs(&self) -> Vec<tenferro_autodiff::NodeId> { vec![] }
/// }
/// ```
pub trait ReverseRule<T: ScalarBase + HasAlgebra> {
    /// Computes input cotangents from an output cotangent (pullback).
    fn pullback(&self, cotangent: &Tensor<T>) -> AdResult<Vec<(NodeId, Tensor<T>)>>;

    /// Returns input node IDs this rule depends on.
    fn inputs(&self) -> Vec<NodeId>;

    /// Computes pullback with tangent propagation for HVP.
    ///
    /// Given an output cotangent ḡ and its tangent dḡ, returns
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
        cotangent: &Tensor<T>,
        cotangent_tangent: &Tensor<T>,
    ) -> AdResult<Vec<(NodeId, Tensor<T>, Tensor<T>)>> {
        let _ = (cotangent, cotangent_tangent);
        Err(AutodiffError::HvpNotSupported)
    }
}

/// Forward-mode AD rule interface (frule).
///
/// Named after Julia's ChainRules.jl convention: `frule` computes pushforward.
///
/// # Examples
///
/// ```ignore
/// struct MyFrule;
/// impl tenferro_autodiff::ForwardRule<f64> for MyFrule {
///     fn pushforward(&self, tangents: &[Option<&tenferro_tensor::Tensor<f64>>])
///         -> tenferro_autodiff::AdResult<tenferro_tensor::Tensor<f64>> {
///         todo!()
///     }
/// }
/// ```
pub trait ForwardRule<T: ScalarBase + HasAlgebra> {
    /// Computes output tangent from input tangents (pushforward).
    fn pushforward(&self, tangents: &[Option<&Tensor<T>>]) -> AdResult<Tensor<T>>;
}

/// Compiled backward execution plan.
///
/// # Examples
///
/// ```ignore
/// let plan = tenferro_autodiff::BackwardPlan::<f64>::build(&loss).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BackwardPlan<T: ScalarBase + HasAlgebra> {
    loss: NodeId,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ScalarBase + HasAlgebra> BackwardPlan<T> {
    /// Builds a backward plan from a loss tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let plan = tenferro_autodiff::BackwardPlan::<f64>::build(&loss).unwrap();
    /// ```
    pub fn build(_loss: &TrackedTensor<T>) -> AdResult<Self> {
        todo!()
    }

    /// Executes the pre-built backward plan.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let grads = plan.execute(&loss).unwrap();
    /// ```
    pub fn execute(&self, _loss: &TrackedTensor<T>) -> AdResult<Gradients<T>> {
        todo!()
    }

    /// Returns loss node ID for this plan.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_autodiff::{BackwardPlan, NodeId};
    /// let _id_fn: fn(&BackwardPlan<f64>) -> NodeId = BackwardPlan::loss_node;
    /// ```
    pub fn loss_node(&self) -> NodeId {
        self.loss
    }
}

/// Clears the current reverse-mode tape/graph for type `T`.
///
/// # Examples
///
/// ```ignore
/// tenferro_autodiff::clear_tape::<f64>();
/// ```
pub fn clear_tape<T: ScalarBase>() {
    let _ = std::marker::PhantomData::<T>;
    todo!()
}

/// Runs reverse-mode backward from a scalar loss tensor.
///
/// # Errors
///
/// Returns [`AutodiffError::NonScalarLoss`] for non-scalar losses.
///
/// # Examples
///
/// ```ignore
/// let grads = tenferro_autodiff::backward(&loss).unwrap();
/// ```
pub fn backward<T: ScalarBase + HasAlgebra>(_loss: &TrackedTensor<T>) -> AdResult<Gradients<T>> {
    todo!()
}

/// Result of a forward-over-reverse HVP computation.
///
/// Contains both the standard gradient ∇f(x) and the Hessian-vector
/// product H·v, where v is the tangent direction set on leaf tensors
/// via [`TrackedTensor::leaf_with_tangent`].
///
/// # Examples
///
/// ```ignore
/// use tenferro_autodiff::{TrackedTensor, hvp, HvpResult};
/// use tenferro_einsum::tracked_einsum;
///
/// let result: HvpResult<f64> = hvp(&loss).unwrap();
/// let _grad = result.gradients.get(x.node_id().unwrap());  // ∇f(x)
/// let _hv = result.hvp.get(x.node_id().unwrap());          // H·v
/// ```
pub struct HvpResult<T: ScalarBase> {
    /// Gradients: ∇f(x).
    pub gradients: Gradients<T>,
    /// Hessian-vector product: H·v.
    pub hvp: Gradients<T>,
}

/// Computes gradient and Hessian-vector product via forward-over-reverse.
///
/// Leaf tensors with tangents (created via [`TrackedTensor::leaf_with_tangent`])
/// define the direction *v*. The function runs backward through the tape,
/// propagating both cotangents (ḡ) and cotangent-tangents (dḡ) at each node.
///
/// Returns both ∇f(x) (in [`HvpResult::gradients`]) and H·v (in
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
/// clear_tape::<f64>();
/// let x = TrackedTensor::leaf_with_tangent(
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
/// ).unwrap();
/// let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();  // f(x) = x·x
/// let result = hvp(&loss).unwrap();
/// let _grad = result.gradients;  // ∇f(x) = 2x
/// let _hv = result.hvp;          // H·v = 2v
/// ```
pub fn hvp<T: ScalarBase + HasAlgebra>(_loss: &TrackedTensor<T>) -> AdResult<HvpResult<T>> {
    todo!()
}
