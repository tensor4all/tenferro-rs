//! Automatic differentiation API skeleton for tenferro.
//!
//! This crate defines the public API for:
//! - reverse-mode AD (VJP/backward) via [`TrackedTensor`]
//! - forward-mode AD (JVP) via [`DualTensor`]
//! - primitive derivative interfaces ([`VjpRule`], [`JvpRule`])
//!
//! Bodies are intentionally `todo!()` in the current POC phase.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_autodiff::{TrackedTensor, backward, clear_tape};
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
//! let c = tenferro_autodiff::tracked_einsum("ij,jk->ik", &[&a, &b]).unwrap();
//! let loss = tenferro_autodiff::tracked_einsum("ij,ij->", &[&c, &c]).unwrap();
//! let grads = backward(&loss).unwrap();
//! let _ga = grads.get(a.node_id().unwrap()).unwrap();
//! ```
//!
//! ```ignore
//! use tenferro_autodiff::{DualTensor, dual_einsum};
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

use strided_traits::ScalarBase;
use tenferro_algebra::HasAlgebra;
use tenferro_device::{Error as DeviceError, Result as DeviceResult};
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

/// Backward rule interface for reverse-mode AD.
///
/// Implemented by operation-specific nodes (einsum, reduce, permute, ...).
///
/// # Examples
///
/// ```ignore
/// struct MyRule;
/// impl tenferro_autodiff::VjpRule<f64> for MyRule {
///     fn backward(&self, cotangent: &tenferro_tensor::Tensor<f64>)
///         -> tenferro_autodiff::AdResult<Vec<(tenferro_autodiff::NodeId, tenferro_tensor::Tensor<f64>)>> {
///         todo!()
///     }
///     fn inputs(&self) -> Vec<tenferro_autodiff::NodeId> { vec![] }
/// }
/// ```
pub trait VjpRule<T: ScalarBase + HasAlgebra> {
    /// Computes input cotangents from an output cotangent.
    fn backward(&self, cotangent: &Tensor<T>) -> AdResult<Vec<(NodeId, Tensor<T>)>>;

    /// Returns input node IDs this rule depends on.
    fn inputs(&self) -> Vec<NodeId>;
}

/// Forward rule interface for JVP propagation.
///
/// # Examples
///
/// ```ignore
/// struct MyJvp;
/// impl tenferro_autodiff::JvpRule<f64> for MyJvp {
///     fn forward(&self, tangents: &[Option<&tenferro_tensor::Tensor<f64>>])
///         -> tenferro_autodiff::AdResult<tenferro_tensor::Tensor<f64>> {
///         todo!()
///     }
/// }
/// ```
pub trait JvpRule<T: ScalarBase + HasAlgebra> {
    /// Computes output tangent from input tangents.
    fn forward(&self, tangents: &[Option<&Tensor<T>>]) -> AdResult<Tensor<T>>;
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

/// Tracked einsum (reverse-mode).
///
/// This is the AD-aware counterpart of [`tenferro_einsum::einsum`].
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_autodiff::tracked_einsum("ij,jk->ik", &[&a, &b]).unwrap();
/// ```
pub fn tracked_einsum<T: ScalarBase + HasAlgebra>(
    _subscripts: &str,
    _operands: &[&TrackedTensor<T>],
) -> AdResult<TrackedTensor<T>> {
    todo!()
}

/// Dual einsum (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_autodiff::dual_einsum("ij,jk->ik", &[&a, &b]).unwrap();
/// ```
pub fn dual_einsum<T: ScalarBase + HasAlgebra>(
    _subscripts: &str,
    _operands: &[&DualTensor<T>],
) -> AdResult<DualTensor<T>> {
    todo!()
}

/// Local VJP rule for einsum without building a global tape.
///
/// This API is intended for language interop (`custom_vjp`) and manual AD.
///
/// # Examples
///
/// ```ignore
/// let grads = tenferro_autodiff::einsum_vjp("ij,jk->ik", &[&a, &b], &grad_c).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn einsum_vjp<T: ScalarBase + HasAlgebra>(
    _subscripts: &str,
    _operands: &[&Tensor<T>],
    _cotangent: &Tensor<T>,
) -> DeviceResult<Vec<Tensor<T>>> {
    todo!()
}

/// Local JVP rule for einsum without building a global tape.
///
/// Inputs without tangent should use `None`.
///
/// # Examples
///
/// ```ignore
/// let dc = tenferro_autodiff::einsum_jvp(
///     "ij,jk->ik",
///     &[&a, &b],
///     &[Some(&da), None],
/// ).unwrap();
/// ```
pub fn einsum_jvp<T: ScalarBase + HasAlgebra>(
    _subscripts: &str,
    _primals: &[&Tensor<T>],
    _tangents: &[Option<&Tensor<T>>],
) -> DeviceResult<Tensor<T>> {
    todo!()
}
