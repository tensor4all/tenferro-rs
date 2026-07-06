//! Out-of-tree extension-operation mechanism.
//!
//! This module implements the [`ExtensionOp`] trait and its process-local
//! registry. Together they let external crates contribute fused primitives
//! that participate in the [`crate::std_tensor_op::StdTensorOp`] graph through
//! the single carrier variant
//! `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`.
//!
//! See `docs/spec/extension-op.md` for the normative contract. Key points:
//!
//! - Identity / hashing / equality are expressed on the trait so the
//!   type-erased `Arc<dyn ExtensionOp>` carrier can satisfy
//!   `Clone + Hash + Eq + Send + Sync + 'static` (computegraph's
//!   `GraphOperation` requirements).
//! - AD rules are owned by explicit [`ExtensionRuleSet`] values. A rule may
//!   emit core [`StdTensorOp`] values and registered `Extension` values so
//!   out-of-tree operations remain in the same graph.
//! - Extension ops themselves do not require process-global registration.
//!   Frontends carry them directly as `Arc<dyn ExtensionOp>`.

use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
#[cfg(not(feature = "autodiff"))]
use computegraph::types::ValueRef;
#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{DType, Tensor};
#[cfg(feature = "autodiff")]
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

#[cfg(feature = "autodiff")]
use crate::ad::context::ShapeGuardContext;
#[cfg(feature = "autodiff")]
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;
#[cfg(feature = "autodiff")]
use std::collections::HashMap;

/// Error returned when an extension cannot expand itself into standard ops.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::ExtensionLoweringError;
///
/// let err = ExtensionLoweringError::new("example extension cannot lower");
/// assert!(err.to_string().contains("cannot lower"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ExtensionLoweringError {
    message: String,
}

impl ExtensionLoweringError {
    /// Create a lowering error with a human-readable diagnostic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::ExtensionLoweringError;
    ///
    /// let err = ExtensionLoweringError::new("shape must be static");
    /// assert_eq!(err.to_string(), "shape must be static");
    /// ```
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Result returned by [`ExtensionOp::lower_to_standard_ops`].
pub type ExtensionLoweringResult =
    std::result::Result<Option<Vec<ValueRef<StdTensorOp>>>, ExtensionLoweringError>;

/// Host/reference implementation for an extension family.
///
/// This capability is optional. Backend-only extension families can omit it
/// and still implement [`ExtensionOp`]; runtimes that specifically delegate to
/// host reference execution must report a typed capability-missing error when
/// this hook is absent.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::HostReference;
/// use tenferro_tensor::Tensor;
///
/// #[derive(Debug)]
/// struct IdentityHost;
///
/// impl HostReference for IdentityHost {
///     fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
///         Ok(vec![inputs[0].clone()])
///     }
/// }
///
/// let input = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
/// let output = IdentityHost.execute(&[&input]).unwrap();
/// assert_eq!(output[0].as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub trait HostReference: Debug + Send + Sync + 'static {
    /// Execute the extension op on host/reference tensors.
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>>;
}

/// The contract every out-of-tree extension primitive must satisfy.
///
/// Implementations appear in the core graph as
/// `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`. Every method is part of the
/// `ExtensionOp` spec (`docs/spec/extension-op.md`); the short form:
///
/// - identity via [`family_id`][Self::family_id] + [`payload_hash`][Self::payload_hash]
///   + [`payload_eq`][Self::payload_eq];
/// - fixed arity via [`input_count`][Self::input_count] / [`output_count`][Self::output_count];
/// - shape / dtype inference via [`infer_output_meta`][Self::infer_output_meta];
/// - optional host/reference forward execution via
///   [`host_reference`][Self::host_reference];
/// - optional fixed-shape standard-op expansion via
///   [`lower_to_standard_ops`][Self::lower_to_standard_ops] for peer lowerers
///   such as XLA that cannot execute extension runtimes;
/// - AD via separately registered role-specific extension rules.
///
/// # Downcast convention
///
/// Implementations MUST also implement [`Any`] so that
/// [`ExtensionOp::payload_eq`] can downcast a trait-object reference to
/// the concrete type. The helper [`ExtensionOp::as_any`] returns
/// `&dyn Any` for this purpose. Implementations usually define it as
/// `fn as_any(&self) -> &dyn Any { self }`.
///
/// # Examples
///
/// ```
/// # use std::any::Any;
/// use std::sync::Arc;
/// use tenferro_ops::ext_op::{ExtensionOp, HostReference};
/// use tenferro_ops::SymDim;
/// use tenferro_tensor::{DType, Tensor};
///
/// #[derive(Clone, Debug)]
/// struct IdentityExt;
///
/// impl ExtensionOp for IdentityExt {
///     fn family_id(&self) -> &'static str { "example.identity.v1" }
///     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
///     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
///         other.as_any().downcast_ref::<IdentityExt>().is_some()
///     }
///     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { Arc::new(self.clone()) }
///     fn as_any(&self) -> &dyn Any { self }
///     fn input_count(&self) -> usize { 1 }
///     fn output_count(&self) -> usize { 1 }
///     fn infer_output_meta(
///         &self,
///         dtypes: &[DType],
///         shapes: &[&[SymDim]],
///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
///         Ok(vec![(dtypes[0], shapes[0].to_vec())])
///     }
///     fn host_reference(&self) -> Option<&dyn HostReference> {
///         Some(self)
///     }
/// }
///
/// impl HostReference for IdentityExt {
///     fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
///         Ok(vec![inputs[0].clone()])
///     }
/// }
///
/// let op: Arc<dyn ExtensionOp> = Arc::new(IdentityExt);
/// assert_eq!(op.input_count(), 1);
/// ```
pub trait ExtensionOp: Debug + Send + Sync + 'static {
    // ----- Identity, hashing, equality (spec Section 5) -----

    /// Stable, process-independent family identifier.
    ///
    /// MUST be unique per extension *family* (payload schema), not per
    /// *instance*, and MUST follow the reserved format
    /// `"<crate-name>.<op-name>.v<major>"`.
    fn family_id(&self) -> &'static str;

    /// Hash the payload (everything except `family_id`).
    ///
    /// Implementations MUST be pure and deterministic across calls on the same
    /// value. Hashes MUST NOT include transient state such as allocation
    /// addresses or atomically updated counters.
    fn payload_hash(&self, hasher: &mut dyn Hasher);

    /// Structural equality against another extension value.
    ///
    /// The carrier's `PartialEq` impl first compares `family_id`s. When the
    /// family IDs match, it calls `payload_eq`. Implementations MUST return
    /// `true` iff the payloads are semantically equal AND
    /// `other.family_id() == self.family_id()`.
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool;

    /// Deep-clone the payload behind an `Arc`.
    ///
    /// The carrier's `Clone` impl uses `Arc::clone` on the fast path; this
    /// method exists for rare cases that need a second independent `Arc`.
    fn clone_arc(&self) -> Arc<dyn ExtensionOp>;

    /// Upcast this extension to `&dyn Any` for downcasting in `payload_eq`.
    ///
    /// Implementations SHOULD return `self` verbatim. The method is
    /// object-safe (no `Self: Sized` bound) so it can be called on an
    /// `&dyn ExtensionOp`; that's what makes
    /// `other.as_any().downcast_ref::<ConcreteType>()` work from
    /// [`Self::payload_eq`] implementations.
    fn as_any(&self) -> &dyn Any;

    // ----- Arity (spec Section 6) -----

    /// Number of primal inputs. MUST be constant for any given
    /// `Arc<dyn ExtensionOp>` value.
    fn input_count(&self) -> usize;

    /// Number of outputs. MUST match the length of the vector returned by a
    /// successful [`Self::infer_output_meta`] call.
    fn output_count(&self) -> usize;

    // ----- Shape and dtype inference (spec Section 7) -----

    /// Infer output dtypes and shapes for each output slot.
    ///
    /// Implementations MUST validate arity, rank, dtype, axis, and other
    /// input-derived metadata before indexing shape arrays. Invalid public
    /// input must return a typed error rather than an empty sentinel or panic.
    ///
    /// On success, the returned vector MUST have length `self.output_count()`,
    /// one `(dtype, shape)` entry per output slot. Shapes use [`SymDim`] so
    /// extension ops compose with graph-global symbolic metadata.
    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>>;

    // ----- Optional host/reference execution (spec Section 8) -----

    /// Return the host/reference implementation for this payload, when the
    /// family has one.
    ///
    /// Backend-only extension families may leave the default `None`. Runtime
    /// execution never silently falls back to this hook; a runtime must opt in
    /// explicitly by using this capability.
    fn host_reference(&self) -> Option<&dyn HostReference> {
        None
    }

    /// Optionally expand this extension into standard tensor graph operations.
    ///
    /// Peer lowerers call this when all input metadata is known and extension
    /// runtime dispatch is not available. Return `Ok(Some(outputs))` after
    /// adding only standard [`StdTensorOp`] operations to `builder`. Return
    /// `Ok(None)` when this extension family has no standard-op lowering for
    /// the supplied metadata; strict lowerers should surface that as an
    /// explicit unsupported-extension error. Return [`ExtensionLoweringError`]
    /// when the payload is malformed or the lowering detects invalid metadata.
    ///
    /// The default implementation returns `Ok(None)` so existing extension
    /// runtimes keep their native dispatch behavior until their owning crate
    /// deliberately implements this hook.
    fn lower_to_standard_ops(
        &self,
        _builder: &mut GraphBuilder<StdTensorOp>,
        _inputs: &[ValueRef<StdTensorOp>],
        _input_dtypes: &[DType],
        _input_shapes: &[&[SymDim]],
    ) -> ExtensionLoweringResult {
        Ok(None)
    }

    /// Optionally return an equivalent op that produces only live outputs.
    ///
    /// `live_outputs` is aligned with this op's current output slots. Return
    /// `None` when the family does not support output pruning. Return
    /// `Some(op)` only when the new op's outputs are exactly the live output
    /// slots, in ascending slot order, and `op.output_count()` equals the
    /// number of `true` entries in `live_outputs`.
    fn prune_outputs(&self, _live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
        None
    }

    // AD rules are registered separately; see the role-specific rule traits.
}

/// Definitional JVP rule provider for an extension family.
///
/// Required only for families that appear in primal graphs and must be
/// differentiable.
#[cfg(feature = "autodiff")]
pub trait ExtensionLinearizeRule: Debug + Send + Sync + 'static {
    /// The extension family this rule handles.
    fn family_id(&self) -> &'static str;

    /// Emit the linear (JVP) rule.
    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

/// Adjoint rule for an extension op viewed as a linear map in active inputs.
///
/// This rule is used only while `linear_transpose` walks a linearized graph.
#[cfg(feature = "autodiff")]
pub trait ExtensionLinearTransposeRule: Debug + Send + Sync + 'static {
    /// The extension family this rule handles.
    fn family_id(&self) -> &'static str;

    /// Emit cotangents for active linear inputs.
    fn linear_transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        active_mask: &[bool],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

/// Optional direct primal VJP rule provider for an extension family.
///
/// This is an opt-in performance or compatibility escape hatch. The default
/// reverse-mode path remains `linearize` followed by `linear_transpose`.
#[cfg(feature = "autodiff")]
pub trait ExtensionPrimalVjpRule: Debug + Send + Sync + 'static {
    /// The extension family this rule handles.
    fn family_id(&self) -> &'static str;

    /// Emit a direct VJP from the primal op.
    fn primal_vjp(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

/// Extension AD rule role.
#[cfg(feature = "autodiff")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtensionRuleRole {
    /// Definitional JVP.
    Linearize,
    /// Transpose of an op as a linear map.
    LinearTranspose,
    /// Direct VJP from the primal graph.
    PrimalVjp,
}

/// Errors returned from extension registries.
#[cfg(feature = "autodiff")]
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRegistryError {
    /// An AD rule with the same `family_id` was already registered for one role.
    #[error("extension AD {role:?} rule for family_id {family_id:?} already registered")]
    DuplicateRule {
        /// Duplicate extension family id.
        family_id: &'static str,
        /// Duplicate AD rule role.
        role: ExtensionRuleRole,
    },
    /// The `family_id` does not match the namespaced format
    /// `"<crate-name>.<op-name>.v<major>"`.
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}

#[cfg(feature = "autodiff")]
type LinearizeRuleMap = HashMap<&'static str, Arc<dyn ExtensionLinearizeRule>>;
#[cfg(feature = "autodiff")]
type LinearTransposeRuleMap = HashMap<&'static str, Arc<dyn ExtensionLinearTransposeRule>>;
#[cfg(feature = "autodiff")]
type PrimalVjpRuleMap = HashMap<&'static str, Arc<dyn ExtensionPrimalVjpRule>>;

/// Explicit, owned set of extension AD rules.
///
/// This is the rule container used by higher-level AD contexts. Extension AD
/// intentionally has no process-global fallback; callers must pass the rule set
/// that their graph needs.
#[cfg(feature = "autodiff")]
#[derive(Clone, Default)]
pub struct ExtensionRuleSet {
    linearize_rules: Arc<LinearizeRuleMap>,
    linear_transpose_rules: Arc<LinearTransposeRuleMap>,
    primal_vjp_rules: Arc<PrimalVjpRuleMap>,
}

#[cfg(feature = "autodiff")]
impl std::fmt::Debug for ExtensionRuleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut linearize: Vec<_> = self.linearize_rules.keys().copied().collect();
        let mut linear_transpose: Vec<_> = self.linear_transpose_rules.keys().copied().collect();
        let mut primal_vjp: Vec<_> = self.primal_vjp_rules.keys().copied().collect();
        linearize.sort_unstable();
        linear_transpose.sort_unstable();
        primal_vjp.sort_unstable();
        f.debug_struct("ExtensionRuleSet")
            .field("linearize", &linearize)
            .field("linear_transpose", &linear_transpose)
            .field("primal_vjp", &primal_vjp)
            .finish()
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionRuleSet {
    /// Create an empty extension rule set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let rules = ExtensionRuleSet::new();
    /// assert!(!rules.is_linearize_registered("example.missing.v1"));
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one linearize rule to this owned set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tidu::ADRuleResult;
    /// use computegraph::types::{LocalValueId, ValueKey};
    /// use tenferro_ops::ad::PrimitiveRuleBuilder;
    /// use tenferro_ops::ext_op::{ExtensionLinearizeRule, ExtensionOp};
    /// use tenferro_ops::{ExtensionRuleSet, ShapeGuardContext};
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// #[derive(Debug)]
    /// struct Rule;
    ///
    /// impl ExtensionLinearizeRule for Rule {
    ///     fn family_id(&self) -> &'static str { "example.register_linearize.v1" }
    ///     fn linearize(
    ///         &self,
    ///         _op: &dyn ExtensionOp,
    ///         _builder: &mut dyn PrimitiveRuleBuilder,
    ///         _primal_in: &[ValueKey<StdTensorOp>],
    ///         _primal_out: &[ValueKey<StdTensorOp>],
    ///         tangent_in: &[Option<LocalValueId>],
    ///         _ctx: &mut ShapeGuardContext,
    ///     ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(tangent_in.to_vec())
    ///     }
    /// }
    ///
    /// let mut rules = ExtensionRuleSet::new();
    /// rules.register_linearize(Arc::new(Rule)).unwrap();
    /// assert!(rules.lookup_linearize("example.register_linearize.v1").is_some());
    /// ```
    pub fn register_linearize(
        &mut self,
        rule: Arc<dyn ExtensionLinearizeRule>,
    ) -> Result<(), ExtensionRegistryError> {
        let family_id = rule.family_id();
        validate_linearize_insert(&self.linearize_rules, family_id)?;
        let rules = Arc::make_mut(&mut self.linearize_rules);
        rules.insert(family_id, rule);
        Ok(())
    }

    /// Add one linear-transpose rule to this owned set.
    pub fn register_linear_transpose(
        &mut self,
        rule: Arc<dyn ExtensionLinearTransposeRule>,
    ) -> Result<(), ExtensionRegistryError> {
        let family_id = rule.family_id();
        validate_linear_transpose_insert(&self.linear_transpose_rules, family_id)?;
        let rules = Arc::make_mut(&mut self.linear_transpose_rules);
        rules.insert(family_id, rule);
        Ok(())
    }

    /// Add one primal-VJP rule to this owned set.
    pub fn register_primal_vjp(
        &mut self,
        rule: Arc<dyn ExtensionPrimalVjpRule>,
    ) -> Result<(), ExtensionRegistryError> {
        let family_id = rule.family_id();
        validate_primal_vjp_insert(&self.primal_vjp_rules, family_id)?;
        let rules = Arc::make_mut(&mut self.primal_vjp_rules);
        rules.insert(family_id, rule);
        Ok(())
    }

    /// Return a new rule set containing a linearize rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tidu::ADRuleResult;
    /// use computegraph::types::{LocalValueId, ValueKey};
    /// use tenferro_ops::ad::PrimitiveRuleBuilder;
    /// use tenferro_ops::ext_op::{ExtensionLinearizeRule, ExtensionOp};
    /// use tenferro_ops::{ExtensionRuleSet, ShapeGuardContext};
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// #[derive(Debug)]
    /// struct Rule;
    ///
    /// impl ExtensionLinearizeRule for Rule {
    ///     fn family_id(&self) -> &'static str { "example.with_linearize.v1" }
    ///     fn linearize(
    ///         &self,
    ///         _op: &dyn ExtensionOp,
    ///         _builder: &mut dyn PrimitiveRuleBuilder,
    ///         _primal_in: &[ValueKey<StdTensorOp>],
    ///         _primal_out: &[ValueKey<StdTensorOp>],
    ///         tangent_in: &[Option<LocalValueId>],
    ///         _ctx: &mut ShapeGuardContext,
    ///     ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(tangent_in.to_vec())
    ///     }
    /// }
    ///
    /// let rules = ExtensionRuleSet::new().with_linearize(Arc::new(Rule)).unwrap();
    /// assert!(rules.is_linearize_registered("example.with_linearize.v1"));
    /// ```
    pub fn with_linearize(
        mut self,
        rule: Arc<dyn ExtensionLinearizeRule>,
    ) -> Result<Self, ExtensionRegistryError> {
        self.register_linearize(rule)?;
        Ok(self)
    }

    /// Return a new rule set containing a linear-transpose rule.
    pub fn with_linear_transpose(
        mut self,
        rule: Arc<dyn ExtensionLinearTransposeRule>,
    ) -> Result<Self, ExtensionRegistryError> {
        self.register_linear_transpose(rule)?;
        Ok(self)
    }

    /// Return a new rule set containing a primal-VJP rule.
    pub fn with_primal_vjp(
        mut self,
        rule: Arc<dyn ExtensionPrimalVjpRule>,
    ) -> Result<Self, ExtensionRegistryError> {
        self.register_primal_vjp(rule)?;
        Ok(self)
    }

    /// Merge another owned rule set into this one.
    ///
    /// The merge is atomic: if any rule in `other` is invalid or duplicates an
    /// existing family, `self` is left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let mut rules = ExtensionRuleSet::new();
    /// rules.merge(ExtensionRuleSet::new()).unwrap();
    /// assert!(!rules.is_linearize_registered("example.missing.v1"));
    /// ```
    pub fn merge(&mut self, other: ExtensionRuleSet) -> Result<(), ExtensionRegistryError> {
        let mut linearize_rules = (*self.linearize_rules).clone();
        let mut linear_transpose_rules = (*self.linear_transpose_rules).clone();
        let mut primal_vjp_rules = (*self.primal_vjp_rules).clone();
        for rule in other.linearize_rules.values() {
            insert_linearize_rule(&mut linearize_rules, Arc::clone(rule))?;
        }
        for rule in other.linear_transpose_rules.values() {
            insert_linear_transpose_rule(&mut linear_transpose_rules, Arc::clone(rule))?;
        }
        for rule in other.primal_vjp_rules.values() {
            insert_primal_vjp_rule(&mut primal_vjp_rules, Arc::clone(rule))?;
        }
        self.linearize_rules = Arc::new(linearize_rules);
        self.linear_transpose_rules = Arc::new(linear_transpose_rules);
        self.primal_vjp_rules = Arc::new(primal_vjp_rules);
        Ok(())
    }

    /// Look up a linearize rule in this set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let rules = ExtensionRuleSet::new();
    /// assert!(rules.lookup_linearize("example.missing.v1").is_none());
    /// ```
    pub fn lookup_linearize(&self, family_id: &str) -> Option<Arc<dyn ExtensionLinearizeRule>> {
        self.linearize_rules.get(family_id).cloned()
    }

    /// Look up a linear-transpose rule in this set.
    pub fn lookup_linear_transpose(
        &self,
        family_id: &str,
    ) -> Option<Arc<dyn ExtensionLinearTransposeRule>> {
        self.linear_transpose_rules.get(family_id).cloned()
    }

    /// Look up a primal-VJP rule in this set.
    pub fn lookup_primal_vjp(&self, family_id: &str) -> Option<Arc<dyn ExtensionPrimalVjpRule>> {
        self.primal_vjp_rules.get(family_id).cloned()
    }

    /// Return whether `family_id` has a linearize rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let rules = ExtensionRuleSet::new();
    /// assert!(!rules.is_linearize_registered("example.missing.v1"));
    /// ```
    pub fn is_linearize_registered(&self, family_id: &str) -> bool {
        self.linearize_rules.contains_key(family_id)
    }

    /// Return whether `family_id` has a linear-transpose rule.
    pub fn is_linear_transpose_registered(&self, family_id: &str) -> bool {
        self.linear_transpose_rules.contains_key(family_id)
    }

    /// Return whether `family_id` has a primal-VJP rule.
    pub fn is_primal_vjp_registered(&self, family_id: &str) -> bool {
        self.primal_vjp_rules.contains_key(family_id)
    }
}

#[cfg(feature = "autodiff")]
fn is_valid_family_id(family_id: &str) -> bool {
    // Required shape: `<crate>.<op>.v<major>` with at least one non-empty
    // `<crate>` chunk, at least one non-empty `<op>` chunk (which may itself
    // contain `.`), and a final `.v<integer>` component. `<crate>` and
    // `<op>` segments must be ASCII with no whitespace.
    let mut parts = family_id.rsplitn(2, '.');
    let Some(version_part) = parts.next() else {
        return false;
    };
    let Some(prefix) = parts.next() else {
        return false;
    };
    if !version_part.starts_with('v') {
        return false;
    }
    let digits = &version_part[1..];
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let Some((crate_name, op_name)) = prefix.split_once('.') else {
        return false;
    };
    if crate_name.is_empty() || op_name.is_empty() {
        return false;
    }
    let any_invalid = |s: &str| s.chars().any(|c| c.is_whitespace() || !c.is_ascii());
    if any_invalid(crate_name) || any_invalid(op_name) {
        return false;
    }
    true
}

#[cfg(feature = "autodiff")]
fn insert_linearize_rule(
    rules: &mut LinearizeRuleMap,
    rule: Arc<dyn ExtensionLinearizeRule>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    validate_linearize_insert(rules, family_id)?;
    rules.insert(family_id, rule);
    Ok(())
}

#[cfg(feature = "autodiff")]
fn insert_linear_transpose_rule(
    rules: &mut LinearTransposeRuleMap,
    rule: Arc<dyn ExtensionLinearTransposeRule>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    validate_linear_transpose_insert(rules, family_id)?;
    rules.insert(family_id, rule);
    Ok(())
}

#[cfg(feature = "autodiff")]
fn insert_primal_vjp_rule(
    rules: &mut PrimalVjpRuleMap,
    rule: Arc<dyn ExtensionPrimalVjpRule>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    validate_primal_vjp_insert(rules, family_id)?;
    rules.insert(family_id, rule);
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_rule_insert(
    contains_family: bool,
    family_id: &'static str,
    role: ExtensionRuleRole,
) -> Result<(), ExtensionRegistryError> {
    if !is_valid_family_id(family_id) {
        return Err(ExtensionRegistryError::MalformedFamilyId { family_id });
    }
    if contains_family {
        return Err(ExtensionRegistryError::DuplicateRule { family_id, role });
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_linearize_insert(
    rules: &LinearizeRuleMap,
    family_id: &'static str,
) -> Result<(), ExtensionRegistryError> {
    validate_rule_insert(
        rules.contains_key(family_id),
        family_id,
        ExtensionRuleRole::Linearize,
    )
}

#[cfg(feature = "autodiff")]
fn validate_linear_transpose_insert(
    rules: &LinearTransposeRuleMap,
    family_id: &'static str,
) -> Result<(), ExtensionRegistryError> {
    validate_rule_insert(
        rules.contains_key(family_id),
        family_id,
        ExtensionRuleRole::LinearTranspose,
    )
}

#[cfg(feature = "autodiff")]
fn validate_primal_vjp_insert(
    rules: &PrimalVjpRuleMap,
    family_id: &'static str,
) -> Result<(), ExtensionRegistryError> {
    validate_rule_insert(
        rules.contains_key(family_id),
        family_id,
        ExtensionRuleRole::PrimalVjp,
    )
}

/// Emit a registered extension linearization rule.
#[cfg(feature = "autodiff")]
pub fn linearize_extension_rule(
    op: &dyn ExtensionOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    match ctx.extension_linearize_rule_for(op.family_id()) {
        Some(rule) => rule.linearize(op, builder, primal_in, primal_out, tangent_in, ctx),
        None => Err(ADRuleError::unsupported(op.family_id(), ADRuleKind::Jvp)),
    }
}

/// Emit a registered extension transpose rule.
#[cfg(feature = "autodiff")]
pub fn transpose_extension_rule(
    op: &dyn ExtensionOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    match mode {
        OperationRole::Linearized { active_mask } => {
            match ctx.extension_linear_transpose_rule_for(op.family_id()) {
                Some(rule) => {
                    rule.linear_transpose(op, builder, cotangent_out, inputs, active_mask, ctx)
                }
                None => Err(ADRuleError::unsupported(
                    op.family_id(),
                    ADRuleKind::Transpose,
                )),
            }
        }
        OperationRole::Primary => match ctx.extension_primal_vjp_rule_for(op.family_id()) {
            Some(rule) => rule.primal_vjp(op, builder, cotangent_out, inputs, ctx),
            None => Err(ADRuleError::unsupported(
                op.family_id(),
                ADRuleKind::Transpose,
            )),
        },
    }
}

/// Thin adapter that lets a generic `H: Hasher` satisfy the object-safe
/// `&mut dyn Hasher` signature required by [`ExtensionOp::payload_hash`].
///
/// Only `write` and `finish` are load-bearing from the generic hasher; the
/// various `write_u8` / `write_u16` default implementations in `Hasher`
/// delegate to `write`. The adapter preserves that behaviour.
pub(crate) struct DynHasherProxy<'a, H: Hasher + ?Sized> {
    inner: &'a mut H,
}

impl<'a, H: Hasher + ?Sized> DynHasherProxy<'a, H> {
    pub(crate) fn new(inner: &'a mut H) -> Self {
        Self { inner }
    }
}

impl<H: Hasher + ?Sized> Hasher for DynHasherProxy<'_, H> {
    fn finish(&self) -> u64 {
        self.inner.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.inner.write(bytes);
    }
}

/// Hash an `Arc<dyn ExtensionOp>` payload using the extension's
/// [`ExtensionOp::family_id`] plus [`ExtensionOp::payload_hash`]. Shared
/// between the `StdTensorOp::Extension` carrier's `Hash` impl and callers
/// that need to fingerprint an `ExtensionOp` independently.
pub(crate) fn hash_extension<H: Hasher>(op: &(dyn ExtensionOp + '_), state: &mut H) {
    op.family_id().as_bytes().hash(state);
    op.payload_hash(&mut DynHasherProxy::new(state));
}

/// Structural equality used by the `StdTensorOp::Extension` carrier.
///
/// Short-circuits on `family_id` inequality so two extensions with
/// accidentally similar payloads but different families cannot be unified
/// by the op interner.
pub(crate) fn ext_op_eq(a: &dyn ExtensionOp, b: &dyn ExtensionOp) -> bool {
    a.family_id() == b.family_id() && a.payload_eq(b)
}
