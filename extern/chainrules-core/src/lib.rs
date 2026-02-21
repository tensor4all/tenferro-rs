//! Core AD trait definitions (like Julia's ChainRulesCore.jl).
//!
//! This crate defines the interface for automatic differentiation without
//! providing an AD engine. It contains:
//!
//! - [`Differentiable`] — tangent space definition for any value type
//! - [`ReverseRule`] — per-operation reverse-mode rule (rrule/pullback)
//! - [`ForwardRule`] — per-operation forward-mode rule (frule/pushforward)
//! - Error types ([`AutodiffError`], [`AdResult`])
//! - [`NodeId`], [`SavePolicy`] — graph node identifier and save strategy
//!
//! The AD engine (`TrackedTensor`, `DualTensor`, `pullback`, `hvp`) lives in
//! the [`chainrules`](https://docs.rs/chainrules) crate.
//!
//! Operation-specific AD rules (e.g., einsum rrule/frule) live in the crate
//! that defines the operation.
//!
//! # Examples
//!
//! Implementing `Differentiable` for a custom type:
//!
//! ```ignore
//! use chainrules_core::Differentiable;
//!
//! #[derive(Clone)]
//! struct MyVec(Vec<f64>);
//!
//! impl Differentiable for MyVec {
//!     type Tangent = MyVec;
//!     fn zero_tangent(&self) -> MyVec {
//!         MyVec(vec![0.0; self.0.len()])
//!     }
//!     fn accumulate_tangent(a: MyVec, b: &MyVec) -> MyVec {
//!         MyVec(a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect())
//!     }
//! }
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
/// Note: This trait intentionally does **not** require `Clone` on the primal
/// type. `Clone` is only required on `Tangent` (for gradient accumulation).
/// Large values (e.g., tensors) may be expensive to clone; the AD engine
/// avoids cloning primals by taking ownership where needed.
///
/// # Examples
///
/// ```ignore
/// use chainrules_core::Differentiable;
///
/// // Tensor<f64> implements Differentiable with Tangent = Tensor<f64>
/// // (defined in tenferro-tensor crate)
/// fn example<V: Differentiable>(x: &V) {
///     let zero = x.zero_tangent();
///     let _acc = V::accumulate_tangent(zero.clone(), &x.zero_tangent());
/// }
/// ```
pub trait Differentiable {
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
/// use chainrules_core::AutodiffError;
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
    /// The requested AD mode is not supported for the given algebra or operation.
    ///
    /// For example, tropical einsum does not support frule (JVP) or hvp —
    /// only rrule (VJP) via the argmax route is available.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules_core::AutodiffError;
    ///
    /// let err = AutodiffError::ModeNotSupported {
    ///     mode: "frule".into(),
    ///     reason: "tropical einsum supports rrule only (max is not smooth)".into(),
    /// };
    /// ```
    #[error("AD mode not supported: {mode} — {reason}")]
    ModeNotSupported {
        /// The unsupported mode (e.g., "frule", "hvp").
        mode: String,
        /// Explanation of why this mode is not supported.
        reason: String,
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
/// use chainrules_core::AdResult;
///
/// fn returns_ad_result() -> AdResult<()> { Ok(()) }
/// ```
pub type AdResult<T> = std::result::Result<T, AutodiffError>;

/// Stable identifier of an AD graph node.
///
/// # Examples
///
/// ```
/// use chainrules_core::NodeId;
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
    /// use chainrules_core::NodeId;
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
    /// use chainrules_core::NodeId;
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
/// use chainrules_core::SavePolicy;
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
/// use chainrules_core::{ReverseRule, Differentiable, AdResult, NodeId};
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
/// use chainrules_core::{ForwardRule, Differentiable, AdResult};
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

// ============================================================================
// Differentiable impls for primitive types
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
