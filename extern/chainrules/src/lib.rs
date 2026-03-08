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
//! The reverse-mode graph model is homogeneous: one [`Tape`] carries one value
//! type `V`. This supports both tensor graphs such as `Tape<Tensor<f64>>` and
//! downstream custom-type graphs such as `Tape<MyType>`, as long as
//! `MyType: Differentiable`.
//!
//! For tensor-valued APIs, scalar semantics follow PyTorch conventions:
//! scalar tensors are rank-0 (`shape=[]`), not shape `[1]`. Implicit reverse
//! seed creation remains based on `Differentiable::num_elements() == 1`.
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
//! Reverse-mode with a downstream custom type:
//!
//! ```ignore
//! use chainrules::{Differentiable, Tape};
//!
//! #[derive(Clone, Copy, Debug, PartialEq)]
//! struct MyScalar(f64);
//!
//! impl Differentiable for MyScalar {
//!     type Tangent = Self;
//!
//!     fn zero_tangent(&self) -> Self::Tangent { Self(0.0) }
//!     fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
//!         Self(a.0 + b.0)
//!     }
//!     fn num_elements(&self) -> usize { 1 }
//!     fn seed_cotangent(&self) -> Self::Tangent { Self(1.0) }
//! }
//!
//! let tape = Tape::<MyScalar>::new();
//! let x = tape.leaf(MyScalar(2.0));
//! let grads = tape.pullback(&x).unwrap();
//! assert_eq!(grads.get(x.node_id().unwrap()).unwrap().0, 1.0);
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

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ============================================================================
// Internal tape storage
// ============================================================================

/// A single node in the tape graph.
struct TapeNode<V: Differentiable> {
    /// None for leaf nodes, Some for operation nodes.
    rule: Option<Box<dyn ReverseRule<V>>>,
    /// Tangent for HVP (set on leaves via leaf_with_tangent, or computed
    /// for operation outputs during record_op). Reserved for future use by
    /// AD-aware operation functions that need to access saved tangents.
    #[allow(dead_code)]
    tangent: Option<V::Tangent>,
    /// Whether this node is a leaf (created via Tape::leaf).
    is_leaf: bool,
}

/// Internal shared tape state.
struct TapeInner<V: Differentiable> {
    nodes: Vec<TapeNode<V>>,
}

// ============================================================================
// Tape
// ============================================================================

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
    inner: Rc<RefCell<TapeInner<V>>>,
}

impl<V: Differentiable> Tape<V> {
    /// Creates a new empty tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TapeInner { nodes: Vec::new() })),
        }
    }

    /// Returns `true` if `self` and `other` are the same tape (same backing store).
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    ///
    /// let t1 = Tape::<f64>::new();
    /// let t2 = Tape::<f64>::new();
    /// let t1_clone = t1.clone();
    /// assert!(!t1.same_tape(&t2));
    /// assert!(t1.same_tape(&t1_clone));
    /// ```
    pub fn same_tape(&self, other: &Tape<V>) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Creates a leaf value requiring gradient on this tape.
    ///
    /// The returned [`TrackedTensor`] is connected to this tape and
    /// will participate in gradient computation via [`Tape::pullback`].
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(3.14);
    /// assert!(x.requires_grad());
    /// ```
    pub fn leaf(&self, value: V) -> TrackedTensor<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: None,
            is_leaf: true,
        });
        TrackedTensor {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: None,
        }
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
    /// ```
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf_with_tangent(3.14, 1.0).unwrap();
    /// assert!(x.requires_grad());
    /// assert!(x.has_tangent());
    /// ```
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedTensor<V>> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: Some(tangent.clone()),
            is_leaf: true,
        });
        Ok(TrackedTensor {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: Some(tangent),
        })
    }

    /// Records an operation on the tape, returning a tracked output.
    ///
    /// Called by AD-aware functions (e.g., `tracked_einsum`) to register
    /// an operation node with its reverse-mode rule.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let y = tape.record_op(output_value, Box::new(my_rule), None);
    /// ```
    pub fn record_op(
        &self,
        output_value: V,
        rule: Box<dyn ReverseRule<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedTensor<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: Some(rule),
            tangent: output_tangent.clone(),
            is_leaf: false,
        });
        TrackedTensor {
            value: output_value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: output_tangent,
        }
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
    /// ```
    /// use chainrules::{Tape, Differentiable};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let grads = tape.pullback(&x).unwrap();
    /// // d(x)/d(x) = 1.0
    /// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    /// ```
    pub fn pullback(&self, loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }

        let inner = self.inner.borrow();
        let num_nodes = inner.nodes.len();

        // Initialize cotangents for all nodes
        let mut cotangents: Vec<Option<V::Tangent>> = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            cotangents.push(None);
        }

        // Seed the loss node
        cotangents[loss_node.index()] = Some(loss.value.seed_cotangent());

        // Traverse nodes in reverse order from loss to first node
        for i in (0..=loss_node.index()).rev() {
            // Skip leaf nodes (no rule to propagate through)
            if inner.nodes[i].rule.is_none() {
                continue;
            }

            // Take the accumulated cotangent for this node
            let cot = match cotangents[i].take() {
                Some(c) => c,
                None => continue,
            };

            // Apply the pullback rule
            let input_grads = inner.nodes[i].rule.as_ref().unwrap().pullback(&cot)?;

            // Accumulate cotangents at input nodes
            for (node_id, grad) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad));
                    }
                    None => {
                        cotangents[idx] = Some(grad);
                    }
                }
            }
        }

        // Collect gradients for leaf nodes
        let mut result = Gradients::new();
        for (i, cot) in cotangents.into_iter().enumerate() {
            if let Some(c) = cot {
                if inner.nodes[i].is_leaf {
                    result.entries.push((NodeId::new(i), c));
                }
            }
        }

        Ok(result)
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
    pub fn hvp(&self, loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Differentiable<Tangent = V::Tangent>,
    {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }

        let inner = self.inner.borrow();
        let num_nodes = inner.nodes.len();

        // Initialize cotangents and cotangent-tangents
        let mut cotangents: Vec<Option<V::Tangent>> = Vec::with_capacity(num_nodes);
        let mut cot_tangents: Vec<Option<V::Tangent>> = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            cotangents.push(None);
            cot_tangents.push(None);
        }

        // Seed: cotangent = 1 (seed), cotangent_tangent = 0 (seed is constant)
        cotangents[loss_node.index()] = Some(loss.value.seed_cotangent());
        cot_tangents[loss_node.index()] = Some(loss.value.zero_tangent());

        // Traverse in reverse
        for i in (0..=loss_node.index()).rev() {
            if inner.nodes[i].rule.is_none() {
                continue;
            }

            let cot = match cotangents[i].take() {
                Some(c) => c,
                None => continue,
            };
            let cot_tan = cot_tangents[i].take().unwrap_or_else(|| {
                // If no cotangent-tangent accumulated, use zero matching this node's shape
                cot.zero_tangent()
            });

            let results = inner.nodes[i]
                .rule
                .as_ref()
                .unwrap()
                .pullback_with_tangents(&cot, &cot_tan)?;

            for (node_id, grad, grad_tan) in results {
                let idx = node_id.index();

                // Accumulate cotangent
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad));
                    }
                    None => {
                        cotangents[idx] = Some(grad);
                    }
                }

                // Accumulate cotangent-tangent
                match cot_tangents[idx].take() {
                    Some(existing) => {
                        cot_tangents[idx] = Some(V::accumulate_tangent(existing, &grad_tan));
                    }
                    None => {
                        cot_tangents[idx] = Some(grad_tan);
                    }
                }
            }
        }

        // Collect gradients and HVP for leaf nodes
        let mut gradients = Gradients::new();
        let mut hvp_grads = Gradients::new();
        for i in 0..num_nodes {
            if inner.nodes[i].is_leaf {
                if let Some(c) = cotangents[i].take() {
                    gradients.entries.push((NodeId::new(i), c));
                }
                if let Some(ct) = cot_tangents[i].take() {
                    hvp_grads.entries.push((NodeId::new(i), ct));
                }
            }
        }

        Ok(HvpResult {
            gradients,
            hvp: hvp_grads,
        })
    }
}

impl<V: Differentiable> Clone for Tape<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V: Differentiable> Default for Tape<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TrackedTensor
// ============================================================================

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
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
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
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert_eq!(*x.value(), 42.0);
    /// ```
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes and returns the underlying value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert_eq!(x.into_value(), 42.0);
    /// ```
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns whether this value participates in gradient propagation.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert!(!x.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the graph node ID when this value is connected to a tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert!(x.node_id().is_none());
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Returns the tangent for HVP, or `None` if not set.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert!(x.tangent().is_none());
    /// ```
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this tracked value has a tangent for HVP.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::TrackedTensor;
    /// let x = TrackedTensor::new(42.0_f64);
    /// assert!(!x.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Returns a reference to the tape this value is connected to, if any.
    ///
    /// Leaf values created via [`Tape::leaf`] are connected to a tape.
    /// Values created via [`TrackedTensor::new`] are not.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{Tape, TrackedTensor};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// assert!(x.tape().is_some());
    ///
    /// let y = TrackedTensor::new(3.0_f64);
    /// assert!(y.tape().is_none());
    /// ```
    pub fn tape(&self) -> Option<&Tape<V>> {
        self.tape.as_ref()
    }

    /// Consumes and returns a detached value that does not require gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(3.14);
    /// assert!(x.requires_grad());
    /// let detached = x.detach();
    /// assert!(!detached.requires_grad());
    /// ```
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

// ============================================================================
// DualTensor
// ============================================================================

/// Value wrapper for forward-mode AD.
///
/// # Examples
///
/// ```
/// use chainrules::DualTensor;
/// let dual = DualTensor::new(3.14_f64);
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
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(!x.has_tangent());
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
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0_f64).unwrap();
    /// assert!(x.has_tangent());
    /// assert_eq!(*x.tangent().unwrap(), 1.0);
    /// ```
    pub fn with_tangent(primal: V, tangent: V::Tangent) -> AdResult<Self> {
        // Shape validation is type-specific; for scalars (f64, f32) there is
        // nothing to validate. For tensors, the caller (e.g., dual_einsum)
        // should validate shapes before calling this.
        Ok(Self {
            primal,
            tangent: Some(tangent),
        })
    }

    /// Returns the primal value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert_eq!(*x.primal(), 3.14);
    /// ```
    pub fn primal(&self) -> &V {
        &self.primal
    }

    /// Returns the tangent, or `None` for zero tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(x.tangent().is_none());
    /// ```
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns whether this dual value has a non-zero tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::new(3.14_f64);
    /// assert!(!x.has_tangent());
    /// ```
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Consumes and returns `(primal, tangent)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    /// let (p, t) = x.into_parts();
    /// assert_eq!(p, 3.14);
    /// assert_eq!(t, Some(1.0));
    /// ```
    pub fn into_parts(self) -> (V, Option<V::Tangent>) {
        (self.primal, self.tangent)
    }

    /// Consumes and returns a dual value with tangent removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::DualTensor;
    /// let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    /// let c = x.detach_tangent();
    /// assert!(!c.has_tangent());
    /// assert_eq!(*c.primal(), 3.14);
    /// ```
    pub fn detach_tangent(self) -> Self {
        Self {
            primal: self.primal,
            tangent: None,
        }
    }
}

// ============================================================================
// Gradients
// ============================================================================

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

// ============================================================================
// PullbackPlan
// ============================================================================

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
    /// use chainrules::{Tape, PullbackPlan};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// assert_eq!(plan.loss_node().index(), 0);
    /// ```
    pub fn build(loss: &TrackedTensor<V>) -> AdResult<Self> {
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
    /// use chainrules::{Tape, PullbackPlan};
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(2.0);
    /// let plan = PullbackPlan::build(&x).unwrap();
    /// let grads = plan.execute(&x).unwrap();
    /// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    /// ```
    pub fn execute(&self, loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
        let tape = loss.tape.as_ref().ok_or(AutodiffError::MissingNode)?;
        tape.pullback(loss)
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

// ============================================================================
// HvpResult
// ============================================================================

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

// ============================================================================
// Autodiff-next public API surface
// ============================================================================

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

fn effective_retain_graph(retain_graph: Option<bool>, create_graph: bool) -> bool {
    retain_graph.unwrap_or(create_graph)
}

/// Options for monomorphic backward/grad APIs.
///
/// # Examples
///
/// ```
/// use chainrules::BackwardOptions;
///
/// let opts = BackwardOptions::<f64>::default();
/// assert_eq!(opts.retain_graph, None);
/// assert!(!opts.create_graph);
/// ```
pub struct BackwardOptions<V: Differentiable> {
    /// Optional retain override. `None` means infer from `create_graph`.
    pub retain_graph: Option<bool>,
    /// Whether to build a graph on returned gradients.
    pub create_graph: bool,
    /// Optional output seed cotangent for non-scalar outputs.
    pub seed_grad: Option<V::Tangent>,
}

impl<V: Differentiable> Default for BackwardOptions<V> {
    fn default() -> Self {
        Self {
            retain_graph: None,
            create_graph: false,
            seed_grad: None,
        }
    }
}

#[derive(Copy, Clone)]
enum VariableNodeKind {
    Leaf,
    Add,
    Square { input: NodeId },
    Custom,
}

struct VariableNode<V: Differentiable> {
    rule: Option<Box<dyn ReverseRule<V>>>,
    tangent: Option<V::Tangent>,
    kind: VariableNodeKind,
    is_leaf: bool,
}

/// Shared monomorphic autograd context.
///
/// # Examples
///
/// ```
/// use chainrules::AutogradContext;
///
/// let ctx = AutogradContext::<f64>::new();
/// let id = ctx.lock().unwrap().id();
/// assert!(id > 0);
/// ```
pub struct AutogradContext<V: Differentiable> {
    id: u64,
    graph_alive: bool,
    nodes: Vec<VariableNode<V>>,
    leaf_grads: Vec<Option<V::Tangent>>,
    leaf_hvps: Vec<Option<V::Tangent>>,
}

impl<V: Differentiable> AutogradContext<V> {
    /// Creates a new context.
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
            graph_alive: true,
            nodes: Vec::new(),
            leaf_grads: Vec::new(),
            leaf_hvps: Vec::new(),
        }))
    }

    /// Returns context identifier.
    pub fn id(&self) -> u64 {
        self.id
    }

    fn ensure_alive(&self) -> AdResult<()> {
        if self.graph_alive {
            Ok(())
        } else {
            Err(AutodiffError::GraphFreed)
        }
    }

    fn free_graph(&mut self) {
        self.graph_alive = false;
    }

    fn record_leaf(&mut self, tangent: Option<V::Tangent>) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(VariableNode {
            rule: None,
            tangent,
            kind: VariableNodeKind::Leaf,
            is_leaf: true,
        });
        self.leaf_grads.push(None);
        self.leaf_hvps.push(None);
        self.graph_alive = true;
        id
    }

    fn record_op(
        &mut self,
        rule: Box<dyn ReverseRule<V>>,
        tangent: Option<V::Tangent>,
        kind: VariableNodeKind,
    ) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(VariableNode {
            rule: Some(rule),
            tangent,
            kind,
            is_leaf: false,
        });
        self.leaf_grads.push(None);
        self.leaf_hvps.push(None);
        self.graph_alive = true;
        id
    }

    fn set_node_tangent(&mut self, node: NodeId, tangent: V::Tangent) -> AdResult<()> {
        let idx = node.index();
        let Some(entry) = self.nodes.get_mut(idx) else {
            return Err(AutodiffError::MissingNode);
        };
        entry.tangent = Some(tangent);
        Ok(())
    }

    fn node_kind(&self, node: NodeId) -> Option<VariableNodeKind> {
        self.nodes.get(node.index()).map(|n| n.kind)
    }

    fn has_any_leaf_tangent(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| node.is_leaf && node.tangent.is_some())
    }

    fn compute_cotangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents: Vec<Option<V::Tangent>> = Vec::with_capacity(n);
        for _ in 0..n {
            cotangents.push(None);
        }
        cotangents[output_node.index()] = Some(seed);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let input_grads = rule.pullback(&cot)?;
            for (node_id, grad) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
            }
        }

        Ok(cotangents)
    }

    fn compute_cotangents_with_tangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
    ) -> AdResult<(Vec<Option<V::Tangent>>, Vec<Option<V::Tangent>>)>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents: Vec<Option<V::Tangent>> = Vec::with_capacity(n);
        let mut cot_tangents: Vec<Option<V::Tangent>> = Vec::with_capacity(n);
        for _ in 0..n {
            cotangents.push(None);
            cot_tangents.push(None);
        }

        cotangents[output_node.index()] = Some(seed);
        cot_tangents[output_node.index()] = Some(seed_tangent);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let cot_tan = cot_tangents[i].take().unwrap_or_else(|| cot.zero_tangent());
            let input_grads = rule.pullback_with_tangents(&cot, &cot_tan)?;
            for (node_id, grad, grad_tan) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
                match cot_tangents[idx].take() {
                    Some(existing) => {
                        cot_tangents[idx] = Some(V::accumulate_tangent(existing, &grad_tan))
                    }
                    None => cot_tangents[idx] = Some(grad_tan),
                }
            }
        }

        Ok((cotangents, cot_tangents))
    }

    fn accumulate_leaf_grads(&mut self, cotangents: &mut [Option<V::Tangent>]) {
        for (i, cot) in cotangents.iter_mut().enumerate() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            let Some(value) = cot.take() else {
                continue;
            };
            match self.leaf_grads[i].take() {
                Some(existing) => {
                    self.leaf_grads[i] = Some(V::accumulate_tangent(existing, &value));
                }
                None => self.leaf_grads[i] = Some(value),
            }
        }
    }

    fn accumulate_leaf_hvps(&mut self, cot_tangents: &mut [Option<V::Tangent>]) {
        for (i, hv) in cot_tangents.iter_mut().enumerate() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            let Some(value) = hv.take() else {
                continue;
            };
            match self.leaf_hvps[i].take() {
                Some(existing) => {
                    self.leaf_hvps[i] = Some(V::accumulate_tangent(existing, &value));
                }
                None => self.leaf_hvps[i] = Some(value),
            }
        }
    }

    fn grad_at(&self, node: NodeId) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        self.leaf_grads
            .get(node.index())
            .and_then(|entry| entry.as_ref().cloned())
    }

    fn hvp_at(&self, node: NodeId) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        self.leaf_hvps
            .get(node.index())
            .and_then(|entry| entry.as_ref().cloned())
    }

    fn clear_leaf_buffers(&mut self, node: NodeId) -> AdResult<()> {
        let idx = node.index();
        let Some(entry) = self.nodes.get(idx) else {
            return Err(AutodiffError::MissingNode);
        };
        if !entry.is_leaf {
            return Err(AutodiffError::InvalidArgument(
                "zero_grad is valid on leaf variables only".to_string(),
            ));
        }
        self.leaf_grads[idx] = None;
        self.leaf_hvps[idx] = None;
        Ok(())
    }
}

/// Monomorphic AD variable handle for next API.
///
/// # Examples
///
/// ```
/// use chainrules::Variable;
///
/// let v = Variable::new(3.0_f64);
/// assert!(!v.requires_grad());
/// assert!(v.node_id().is_none());
/// ```
pub struct Variable<V: Differentiable> {
    value: V,
    node_id: Option<NodeId>,
    context: Option<Arc<Mutex<AutogradContext<V>>>>,
    requires_grad: bool,
    tangent: Option<V::Tangent>,
    is_leaf: bool,
}

impl<V: Differentiable> Variable<V> {
    /// Creates a value with no context and `requires_grad=false`.
    pub fn new(value: V) -> Self {
        Self {
            value,
            node_id: None,
            context: None,
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    /// Creates a value attached to the provided context.
    pub fn new_in(value: V, ctx: Arc<Mutex<AutogradContext<V>>>) -> Self {
        Self {
            value,
            node_id: None,
            context: Some(ctx),
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    /// Returns primal value.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Returns ones-like cotangent for this value.
    pub fn ones_like(&self) -> V::Tangent {
        self.value.seed_cotangent()
    }

    /// Returns whether this value is scalar-like.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Variable;
    ///
    /// let x = Variable::new(1.0_f64);
    /// assert!(x.is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.value.num_elements() == 1
    }

    /// Returns optional graph node id.
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    /// Returns optional context id.
    pub fn context_id(&self) -> Option<u64> {
        self.context
            .as_ref()
            .and_then(|ctx| ctx.lock().ok().map(|g| g.id()))
    }

    /// Returns attached context handle if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{AutogradContext, Variable};
    /// use std::sync::Arc;
    ///
    /// let ctx = AutogradContext::<f64>::new();
    /// let v = Variable::new_in(1.0_f64, Arc::clone(&ctx));
    /// assert!(v.context().is_some());
    /// ```
    pub fn context(&self) -> Option<Arc<Mutex<AutogradContext<V>>>> {
        self.context.as_ref().map(Arc::clone)
    }

    /// Returns whether this value is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Returns whether this value tracks gradients.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Enables/disables gradient tracking.
    pub fn requires_grad_(mut self, enabled: bool) -> AdResult<Self> {
        if enabled && self.context.is_none() {
            self.context = Some(AutogradContext::new());
        }
        if enabled && self.node_id.is_none() {
            if let Some(ctx) = self.context.as_ref() {
                let mut guard = ctx.lock().map_err(|_| {
                    AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
                })?;
                self.node_id = Some(guard.record_leaf(None));
            }
        }
        self.requires_grad = enabled;
        Ok(self)
    }

    /// Attaches forward tangent.
    pub fn with_tangent_(mut self, tangent: V::Tangent) -> AdResult<Self>
    where
        V::Tangent: Clone,
    {
        if let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) {
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            guard.set_node_tangent(node, tangent.clone())?;
        }
        self.tangent = Some(tangent);
        Ok(self)
    }

    /// Returns attached tangent if any.
    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    /// Returns a detached value that is disconnected from AD context.
    pub fn detach(&self) -> Self
    where
        V: Clone,
    {
        Self {
            value: self.value.clone(),
            node_id: None,
            context: None,
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    /// Runs backward accumulation on this output.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{autograd, BackwardOptions, Variable};
    ///
    /// let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    /// let y = autograd::square(&x).unwrap();
    /// y.backward(BackwardOptions::default()).unwrap();
    /// ```
    pub fn backward(&self, options: BackwardOptions<V>) -> AdResult<()> {
        let retain = effective_retain_graph(options.retain_graph, options.create_graph);
        if !self.requires_grad {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = self.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = self.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output connected to a graph node".to_string(),
            ));
        };
        if !self.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "backward requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        let seed = options.seed_grad.unwrap_or_else(|| self.ones_like());
        let mut cotangents = guard.compute_cotangents(output_node, seed)?;
        guard.accumulate_leaf_grads(&mut cotangents);
        if !retain {
            guard.free_graph();
        }
        Ok(())
    }

    /// Runs backward + HVP accumulation on this output.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::ModeNotSupported`] in this phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{autograd, BackwardOptions, Variable};
    ///
    /// let x = Variable::new(2.0_f64).requires_grad_(true).unwrap().with_tangent_(1.0).unwrap();
    /// let y = autograd::square(&x).unwrap();
    /// y.backward_hvp(BackwardOptions::default()).unwrap();
    /// assert_eq!(x.hvp(), Some(2.0));
    /// ```
    pub fn backward_hvp(&self, options: BackwardOptions<V>) -> AdResult<()>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        if options.create_graph {
            return Err(AutodiffError::ModeNotSupported {
                mode: "create_graph_hvp".to_string(),
                reason: "backward_hvp with create_graph=true is not implemented yet".to_string(),
            });
        }

        let retain = effective_retain_graph(options.retain_graph, options.create_graph);
        if !self.requires_grad {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = self.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = self.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output connected to a graph node".to_string(),
            ));
        };
        if !self.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        if !guard.has_any_leaf_tangent() {
            return Err(AutodiffError::InvalidArgument(
                "hvp requires tangent-seeded leaves".to_string(),
            ));
        }

        let seed = options.seed_grad.unwrap_or_else(|| self.ones_like());
        let seed_tangent = self.value.zero_tangent();
        let (mut cotangents, mut cot_tangents) = guard
            .compute_cotangents_with_tangents(output_node, seed, seed_tangent)
            .map_err(|err| match err {
                AutodiffError::HvpNotSupported => AutodiffError::ModeNotSupported {
                    mode: "hvp".to_string(),
                    reason: "reverse rule does not support pullback_with_tangents".to_string(),
                },
                other => other,
            })?;
        guard.accumulate_leaf_grads(&mut cotangents);
        guard.accumulate_leaf_hvps(&mut cot_tangents);
        if !retain {
            guard.free_graph();
        }
        Ok(())
    }

    /// Returns currently accumulated gradient for this leaf.
    pub fn grad(&self) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return None;
        };
        ctx.lock().ok().and_then(|guard| guard.grad_at(node))
    }

    /// Returns currently accumulated Hessian-vector product for this leaf.
    pub fn hvp(&self) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return None;
        };
        ctx.lock().ok().and_then(|guard| guard.hvp_at(node))
    }

    /// Clears accumulated `.grad()` / `.hvp()` buffers on this leaf.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::InvalidArgument`] for non-leaf variables.
    pub fn zero_grad(&self) -> AdResult<()> {
        if !self.is_leaf {
            return Err(AutodiffError::InvalidArgument(
                "zero_grad is valid on leaf variables only".to_string(),
            ));
        }
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return Ok(());
        };
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.clear_leaf_buffers(node)
    }
}

impl<V: Differentiable + Clone> Clone for Variable<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            node_id: self.node_id,
            context: self.context.as_ref().map(Arc::clone),
            requires_grad: self.requires_grad,
            tangent: self.tangent.clone(),
            is_leaf: self.is_leaf,
        }
    }
}

/// Monomorphic AD operation helpers for [`Variable`].
pub mod autograd {
    use super::{
        AdResult, AutodiffError, AutogradContext, BackwardOptions, NodeId, ReverseRule, Variable,
        VariableNodeKind,
    };
    use std::marker::PhantomData;
    use std::ops::{Add, Mul};
    use std::sync::{Arc, Mutex};

    struct AddRule<V: super::Differentiable<Tangent = V> + Clone> {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        _marker: PhantomData<V>,
    }

    impl<V> ReverseRule<V> for AddRule<V>
    where
        V: super::Differentiable<Tangent = V> + Clone,
    {
        fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>> {
            let mut out = Vec::new();
            if let Some(lhs) = self.lhs {
                out.push((lhs, cotangent.clone()));
            }
            if let Some(rhs) = self.rhs {
                out.push((rhs, cotangent.clone()));
            }
            Ok(out)
        }

        fn inputs(&self) -> Vec<NodeId> {
            let mut out = Vec::new();
            if let Some(lhs) = self.lhs {
                out.push(lhs);
            }
            if let Some(rhs) = self.rhs {
                out.push(rhs);
            }
            out
        }

        fn pullback_with_tangents(
            &self,
            cotangent: &V::Tangent,
            cotangent_tangent: &V::Tangent,
        ) -> AdResult<Vec<(NodeId, V::Tangent, V::Tangent)>> {
            let mut out = Vec::new();
            if let Some(lhs) = self.lhs {
                out.push((lhs, cotangent.clone(), cotangent_tangent.clone()));
            }
            if let Some(rhs) = self.rhs {
                out.push((rhs, cotangent.clone(), cotangent_tangent.clone()));
            }
            Ok(out)
        }
    }

    struct SquareRule<
        V: super::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V>,
    > {
        input: NodeId,
        two_x: V,
        two_dx: Option<V>,
    }

    impl<V> ReverseRule<V> for SquareRule<V>
    where
        V: super::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V>,
    {
        fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>> {
            Ok(vec![(self.input, cotangent.clone() * self.two_x.clone())])
        }

        fn inputs(&self) -> Vec<NodeId> {
            vec![self.input]
        }

        fn pullback_with_tangents(
            &self,
            cotangent: &V::Tangent,
            cotangent_tangent: &V::Tangent,
        ) -> AdResult<Vec<(NodeId, V::Tangent, V::Tangent)>> {
            let grad = cotangent.clone() * self.two_x.clone();
            let mut grad_tangent = cotangent_tangent.clone() * self.two_x.clone();
            if let Some(two_dx) = self.two_dx.as_ref() {
                grad_tangent = grad_tangent + cotangent.clone() * two_dx.clone();
            }
            Ok(vec![(self.input, grad, grad_tangent)])
        }
    }

    fn context_id<V: super::Differentiable>(ctx: &Arc<Mutex<AutogradContext<V>>>) -> AdResult<u64> {
        ctx.lock().map(|guard| guard.id()).map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })
    }

    fn merge_context_for_multi_op<V: super::Differentiable>(
        inputs: &[&Variable<V>],
    ) -> AdResult<Option<Arc<Mutex<AutogradContext<V>>>>> {
        // Rule 1: if all inputs are not tracked, output has no context.
        if inputs.iter().all(|input| !input.requires_grad()) {
            return Ok(None);
        }

        let mut picked: Option<(u64, Arc<Mutex<AutogradContext<V>>>)> = None;
        for ctx in inputs
            .iter()
            .filter(|input| input.requires_grad())
            .filter_map(|input| input.context.as_ref())
        {
            let id = context_id(ctx)?;
            match &picked {
                None => picked = Some((id, Arc::clone(ctx))),
                Some((picked_id, _)) if *picked_id == id => {}
                Some(_) => return Err(AutodiffError::InvalidArgument(
                    "mixed autograd contexts in one operation; use Variable::new_in(..., same_ctx)"
                        .to_string(),
                )),
            }
        }

        let Some((picked_id, picked_ctx)) = picked else {
            // Rule 3: tracked-but-contextless inputs do not create a context.
            return Ok(None);
        };

        // Rule 5: adopt only when at least one tracked input is on that context.
        let any_tracked_on_picked = inputs.iter().any(|input| {
            input.requires_grad()
                && input.context_id() == Some(picked_id)
                && input.node_id.is_some()
        });
        if any_tracked_on_picked {
            Ok(Some(picked_ctx))
        } else {
            Ok(None)
        }
    }

    fn merge_context_for_binary_op<V: super::Differentiable>(
        lhs: &Variable<V>,
        rhs: &Variable<V>,
    ) -> AdResult<Option<Arc<Mutex<AutogradContext<V>>>>> {
        merge_context_for_multi_op(&[lhs, rhs])
    }

    /// Records a custom operation on the monomorphic `Variable` graph.
    ///
    /// This helper is intended for operation crates (for example einsum) that
    /// need to construct `Variable` outputs with a custom reverse rule.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::InvalidArgument`] when inputs span different
    /// autograd contexts.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Intended for operation implementations:
    /// // let out = autograd::record_op(value, &[&x, &y], Box::new(rule), tangent)?;
    /// ```
    pub fn record_op<V>(
        value: V,
        inputs: &[&Variable<V>],
        rule: Box<dyn ReverseRule<V>>,
        tangent: Option<V::Tangent>,
    ) -> AdResult<Variable<V>>
    where
        V: super::Differentiable + 'static,
        V::Tangent: Clone,
    {
        let out_ctx = merge_context_for_multi_op(inputs)?;
        let mut out_node = None;

        if let Some(ctx) = out_ctx.as_ref() {
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            out_node = Some(guard.record_op(rule, tangent.clone(), VariableNodeKind::Custom));
        }

        Ok(Variable {
            value,
            node_id: out_node,
            context: out_ctx.clone(),
            requires_grad: out_ctx.is_some(),
            tangent,
            is_leaf: false,
        })
    }

    /// Adds two variables and applies Context Merge Rule to the output.
    ///
    /// # Errors
    ///
    /// Returns [`AutodiffError::InvalidArgument`] when operands belong to
    /// different contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{autograd, AutogradContext, Variable};
    /// use std::sync::Arc;
    ///
    /// let ctx = AutogradContext::<f64>::new();
    /// let a = Variable::new_in(1.0_f64, Arc::clone(&ctx)).requires_grad_(true).unwrap();
    /// let b = Variable::new_in(2.0_f64, Arc::clone(&ctx)).requires_grad_(true).unwrap();
    /// let c = autograd::add(&a, &b).unwrap();
    /// assert!(c.requires_grad());
    /// ```
    pub fn add<V>(lhs: &Variable<V>, rhs: &Variable<V>) -> AdResult<Variable<V>>
    where
        V: super::Differentiable<Tangent = V> + Clone + Add<Output = V> + 'static,
    {
        let out_ctx = merge_context_for_binary_op(lhs, rhs)?;
        let out_value = lhs.value.clone() + rhs.value.clone();
        let out_tangent = match (lhs.tangent.as_ref(), rhs.tangent.as_ref()) {
            (Some(lt), Some(rt)) => Some(lt.clone() + rt.clone()),
            (Some(lt), None) => Some(lt.clone()),
            (None, Some(rt)) => Some(rt.clone()),
            (None, None) => None,
        };

        let mut out_node = None;
        if let Some(ctx) = out_ctx.as_ref() {
            let lhs_ctx = lhs.context_id();
            let rhs_ctx = rhs.context_id();
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            let ctx_id = guard.id();
            let lhs_dep = if lhs.requires_grad() && lhs_ctx == Some(ctx_id) {
                Some(lhs.node_id.ok_or(AutodiffError::MissingNode)?)
            } else {
                None
            };
            let rhs_dep = if rhs.requires_grad() && rhs_ctx == Some(ctx_id) {
                Some(rhs.node_id.ok_or(AutodiffError::MissingNode)?)
            } else {
                None
            };
            let rule = AddRule::<V> {
                lhs: lhs_dep,
                rhs: rhs_dep,
                _marker: PhantomData,
            };
            out_node =
                Some(guard.record_op(Box::new(rule), out_tangent.clone(), VariableNodeKind::Add));
        }

        Ok(Variable {
            value: out_value,
            node_id: out_node,
            context: out_ctx.clone(),
            requires_grad: out_ctx.is_some(),
            tangent: out_tangent,
            is_leaf: false,
        })
    }

    /// Squares one variable and preserves context when tracked.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{autograd, Variable};
    ///
    /// let x = Variable::new(3.0_f64).requires_grad_(true).unwrap();
    /// let y = autograd::square(&x).unwrap();
    /// assert!(y.requires_grad());
    /// ```
    pub fn square<V>(input: &Variable<V>) -> AdResult<Variable<V>>
    where
        V: super::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V> + 'static,
    {
        let out_ctx = if input.requires_grad() {
            input.context.as_ref().map(Arc::clone)
        } else {
            None
        };

        let two_x = input.value.clone() + input.value.clone();
        let out_value = input.value.clone() * input.value.clone();
        let out_tangent = input.tangent.as_ref().map(|dx| two_x.clone() * dx.clone());
        let two_dx = input.tangent.as_ref().map(|dx| dx.clone() + dx.clone());

        let mut out_node = None;
        if let Some(ctx) = out_ctx.as_ref() {
            let input_ctx = input.context_id();
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            let ctx_id = guard.id();
            if input.requires_grad() && input_ctx == Some(ctx_id) {
                let input_node = input.node_id.ok_or(AutodiffError::MissingNode)?;
                let rule = SquareRule::<V> {
                    input: input_node,
                    two_x: two_x.clone(),
                    two_dx: two_dx.clone(),
                };
                out_node = Some(guard.record_op(
                    Box::new(rule),
                    out_tangent.clone(),
                    VariableNodeKind::Square { input: input_node },
                ));
            }
        }

        Ok(Variable {
            value: out_value,
            node_id: out_node,
            context: out_ctx.clone(),
            requires_grad: out_ctx.is_some(),
            tangent: out_tangent,
            is_leaf: false,
        })
    }

    /// Side-effect-free gradient query returning detached tangents.
    ///
    /// # Errors
    ///
    /// Returns `ModeNotSupported { mode: "create_graph_tangent", .. }` when
    /// `create_graph=true`.
    pub fn grad_tangent<V>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<V::Tangent>>
    where
        V: super::Differentiable,
        V::Tangent: Clone,
    {
        if options.create_graph {
            return Err(AutodiffError::ModeNotSupported {
                mode: "create_graph_tangent".to_string(),
                reason: "grad_tangent does not support create_graph".to_string(),
            });
        }

        let retain = super::effective_retain_graph(options.retain_graph, options.create_graph);
        if !output.requires_grad() {
            return Err(AutodiffError::InvalidArgument(
                "grad_tangent requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = output.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "grad_tangent requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = output.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "grad_tangent requires output connected to a graph node".to_string(),
            ));
        };
        if !output.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "grad_tangent requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        for input in inputs {
            if let Some(input_ctx) = input.context.as_ref() {
                if !Arc::ptr_eq(input_ctx, ctx) {
                    return Err(AutodiffError::InvalidArgument(
                        "mixed autograd contexts in grad query".to_string(),
                    ));
                }
            }
        }

        let seed = options.seed_grad.unwrap_or_else(|| output.ones_like());
        let cotangents = guard.compute_cotangents(output_node, seed)?;
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            let grad = match input.node_id {
                Some(node) => cotangents
                    .get(node.index())
                    .and_then(|v| v.as_ref().cloned())
                    .unwrap_or_else(|| input.value.zero_tangent()),
                None => input.value.zero_tangent(),
            };
            out.push(grad);
        }

        if !retain {
            guard.free_graph();
        }
        Ok(out)
    }

    /// Side-effect-free gradient query returning monomorphic variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::{autograd, BackwardOptions, Variable};
    ///
    /// let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
    /// let y = autograd::square(&x).unwrap();
    /// let grads = autograd::grad_variable(&y, &[&x], BackwardOptions::default()).unwrap();
    /// assert_eq!(grads.len(), 1);
    /// ```
    pub fn grad_variable<V>(
        output: &Variable<V>,
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<Variable<V>>>
    where
        V: super::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V> + 'static,
    {
        let retain = super::effective_retain_graph(options.retain_graph, options.create_graph);
        if !output.requires_grad() {
            return Err(AutodiffError::InvalidArgument(
                "grad_variable requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = output.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "grad_variable requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = output.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "grad_variable requires output connected to a graph node".to_string(),
            ));
        };
        if !output.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "grad_variable requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        for input in inputs {
            if let Some(input_ctx) = input.context.as_ref() {
                if !Arc::ptr_eq(input_ctx, ctx) {
                    return Err(AutodiffError::InvalidArgument(
                        "mixed autograd contexts in grad query".to_string(),
                    ));
                }
            }
        }

        let seed = options.seed_grad.unwrap_or_else(|| output.ones_like());
        let cotangents = guard.compute_cotangents(output_node, seed)?;
        let output_kind = guard.node_kind(output_node);
        drop(guard);

        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            let grad_value = match input.node_id {
                Some(node) => cotangents
                    .get(node.index())
                    .and_then(|v| v.as_ref().cloned())
                    .unwrap_or_else(|| input.value.zero_tangent()),
                None => input.value.zero_tangent(),
            };

            if options.create_graph {
                let symbolic_square = matches!(
                    (output_kind, input.node_id),
                    (Some(VariableNodeKind::Square { input: src }), Some(n)) if src == n
                );
                if symbolic_square {
                    out.push(add(input, input)?);
                } else {
                    return Err(AutodiffError::ModeNotSupported {
                        mode: "create_graph_grad_variable".to_string(),
                        reason: "only direct square gradients are graph-connected in this phase"
                            .to_string(),
                    });
                }
            } else {
                out.push(Variable::new(grad_value));
            }
        }

        if !retain {
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            guard.free_graph();
        }
        Ok(out)
    }
}

/// Test-only graph builders used by API contract tests.
pub mod test_support {
    use super::{autograd, AdResult, Variable};

    /// Builds `loss = x * x` with a tracked scalar leaf.
    ///
    /// # Examples
    ///
    /// ```
    /// let (_x, loss) = chainrules::test_support::square_graph().unwrap();
    /// assert!(loss.requires_grad());
    /// ```
    pub fn square_graph() -> AdResult<(Variable<f64>, Variable<f64>)> {
        let x = Variable::new(2.0_f64).requires_grad_(true)?;
        let loss = autograd::square(&x)?;
        Ok((x, loss))
    }
}
