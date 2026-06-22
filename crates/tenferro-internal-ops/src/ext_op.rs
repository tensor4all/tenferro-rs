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
/// - host/reference forward execution via [`eager_execute`][Self::eager_execute];
///   runtime-owned eager and compiled paths dispatch through registered
///   extension runtimes instead of falling back to this method;
/// - optional fixed-shape standard-op expansion via
///   [`lower_to_standard_ops`][Self::lower_to_standard_ops] for peer lowerers
///   such as XLA that cannot execute extension runtimes;
/// - AD via a separately registered [`ExtensionAdRule`].
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
/// use tenferro_ops::ext_op::ExtensionOp;
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
///     fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
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

    // ----- Forward execution dispatch (spec Section 8) -----

    /// Eager forward execution; called from the eager path and indirectly
    /// from the compiled path.
    ///
    /// Input tensors are on the device the caller already arranged. Output
    /// tensors MUST have shapes matching [`Self::infer_output_meta`] and MUST
    /// be placed on a device the caller can consume.
    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>>;

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

    // AD rules are registered separately; see [`ExtensionAdRule`].
}

/// AD rule provider for an extension family.
///
/// Rules are registered independently from the primal operation so an
/// out-of-tree crate can provide forward execution without AD, or gate AD
/// support behind an optional feature. Rule methods receive the concrete
/// [`ExtensionOp`] payload as a trait object; implementations should downcast
/// through [`ExtensionOp::as_any`] when they need payload-specific parameters.
#[cfg(feature = "autodiff")]
pub trait ExtensionAdRule: Debug + Send + Sync + 'static {
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

    /// Emit the transpose (VJP) rule.
    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

/// Errors returned from extension registries.
#[cfg(feature = "autodiff")]
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRegistryError {
    /// An AD rule with the same `family_id` was already registered.
    #[error("AD rule for family_id {family_id:?} already registered")]
    DuplicateRule { family_id: &'static str },
    /// The `family_id` does not match the namespaced format
    /// `"<crate-name>.<op-name>.v<major>"`.
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}

#[cfg(feature = "autodiff")]
type RuleMap = HashMap<&'static str, Arc<dyn ExtensionAdRule>>;

/// Explicit, owned set of extension AD rules.
///
/// This is the rule container used by higher-level AD contexts. Extension AD
/// intentionally has no process-global fallback; callers must pass the rule set
/// that their graph needs.
#[cfg(feature = "autodiff")]
#[derive(Clone, Default)]
pub struct ExtensionRuleSet {
    rules: Arc<RuleMap>,
}

#[cfg(feature = "autodiff")]
impl std::fmt::Debug for ExtensionRuleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut families: Vec<_> = self.rules.keys().copied().collect();
        families.sort_unstable();
        f.debug_struct("ExtensionRuleSet")
            .field("families", &families)
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
    /// assert!(!rules.is_rule_registered("example.missing.v1"));
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one rule to this owned set.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tidu::ADRuleResult;
    /// use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
    /// use tenferro_ops::ad::PrimitiveRuleBuilder;
    /// use tenferro_ops::ext_op::{ExtensionAdRule, ExtensionOp};
    /// use tenferro_ops::{ExtensionRuleSet, ShapeGuardContext};
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// #[derive(Debug)]
    /// struct Rule;
    ///
    /// impl ExtensionAdRule for Rule {
    ///     fn family_id(&self) -> &'static str { "example.register_rule.v1" }
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
    ///     fn transpose_rule(
    ///         &self,
    ///         _op: &dyn ExtensionOp,
    ///         _builder: &mut dyn PrimitiveRuleBuilder,
    ///         cotangent_out: &[Option<LocalValueId>],
    ///         _inputs: &[ValueRef<StdTensorOp>],
    ///         _mode: &OperationRole,
    ///         _ctx: &mut ShapeGuardContext,
    ///     ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(cotangent_out.to_vec())
    ///     }
    /// }
    ///
    /// let mut rules = ExtensionRuleSet::new();
    /// rules.register_rule(Arc::new(Rule)).unwrap();
    /// assert!(rules.lookup_rule("example.register_rule.v1").is_some());
    /// ```
    pub fn register_rule(
        &mut self,
        rule: Arc<dyn ExtensionAdRule>,
    ) -> Result<(), ExtensionRegistryError> {
        let family_id = rule.family_id();
        validate_rule_insert(&self.rules, family_id)?;
        let rules = Arc::make_mut(&mut self.rules);
        rules.insert(family_id, rule);
        Ok(())
    }

    /// Return a new rule set containing `rule`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tidu::ADRuleResult;
    /// use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
    /// use tenferro_ops::ad::PrimitiveRuleBuilder;
    /// use tenferro_ops::ext_op::{ExtensionAdRule, ExtensionOp};
    /// use tenferro_ops::{ExtensionRuleSet, ShapeGuardContext};
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// #[derive(Debug)]
    /// struct Rule;
    ///
    /// impl ExtensionAdRule for Rule {
    ///     fn family_id(&self) -> &'static str { "example.with_rule.v1" }
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
    ///     fn transpose_rule(
    ///         &self,
    ///         _op: &dyn ExtensionOp,
    ///         _builder: &mut dyn PrimitiveRuleBuilder,
    ///         cotangent_out: &[Option<LocalValueId>],
    ///         _inputs: &[ValueRef<StdTensorOp>],
    ///         _mode: &OperationRole,
    ///         _ctx: &mut ShapeGuardContext,
    ///     ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    ///         Ok(cotangent_out.to_vec())
    ///     }
    /// }
    ///
    /// let rules = ExtensionRuleSet::new().with_rule(Arc::new(Rule)).unwrap();
    /// assert!(rules.is_rule_registered("example.with_rule.v1"));
    /// ```
    pub fn with_rule(
        mut self,
        rule: Arc<dyn ExtensionAdRule>,
    ) -> Result<Self, ExtensionRegistryError> {
        self.register_rule(rule)?;
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
    /// assert!(!rules.is_rule_registered("example.missing.v1"));
    /// ```
    pub fn merge(&mut self, other: ExtensionRuleSet) -> Result<(), ExtensionRegistryError> {
        let mut rules = (*self.rules).clone();
        for rule in other.rules.values() {
            insert_rule(&mut rules, Arc::clone(rule))?;
        }
        self.rules = Arc::new(rules);
        Ok(())
    }

    /// Look up a rule in this set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let rules = ExtensionRuleSet::new();
    /// assert!(rules.lookup_rule("example.missing.v1").is_none());
    /// ```
    pub fn lookup_rule(&self, family_id: &str) -> Option<Arc<dyn ExtensionAdRule>> {
        self.rules.get(family_id).cloned()
    }

    /// Return whether `family_id` is present in this set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ExtensionRuleSet;
    ///
    /// let rules = ExtensionRuleSet::new();
    /// assert!(!rules.is_rule_registered("example.missing.v1"));
    /// ```
    pub fn is_rule_registered(&self, family_id: &str) -> bool {
        self.rules.contains_key(family_id)
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
fn insert_rule(
    rules: &mut RuleMap,
    rule: Arc<dyn ExtensionAdRule>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    validate_rule_insert(rules, family_id)?;
    rules.insert(family_id, rule);
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_rule_insert(
    rules: &RuleMap,
    family_id: &'static str,
) -> Result<(), ExtensionRegistryError> {
    if !is_valid_family_id(family_id) {
        return Err(ExtensionRegistryError::MalformedFamilyId { family_id });
    }
    if rules.contains_key(family_id) {
        return Err(ExtensionRegistryError::DuplicateRule { family_id });
    }
    Ok(())
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
    match ctx.extension_rule_for(op.family_id()) {
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
    match ctx.extension_rule_for(op.family_id()) {
        Some(rule) => rule.transpose_rule(op, builder, cotangent_out, inputs, mode, ctx),
        None => Err(ADRuleError::unsupported(
            op.family_id(),
            ADRuleKind::Transpose,
        )),
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
