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
//!   `GraphOp` requirements).
//! - AD rules are registered separately through [`ExtensionAdRule`] and
//!   [`register_extension_rule`]. A rule may emit core [`StdTensorOp`] values
//!   and registered `Extension` values so out-of-tree operations remain in the
//!   same graph.
//! - [`ExtensionFactory`] is registered at program start via
//!   [`register_extension`]; the registry is an
//!   `OnceLock<RwLock<HashMap<&'static str, Arc<dyn ExtensionFactory>>>>`
//!   keyed by [`ExtensionOp::family_id`].
//!
//! # Examples
//!
//! ```ignore
//! use std::sync::Arc;
//! use tenferro_ops::ext_op::{register_extension, ExtensionFactory};
//!
//! # struct MyFactory;
//! # impl ExtensionFactory for MyFactory {
//! #     fn family_id(&self) -> &'static str { "my-crate.my_op.v1" }
//! #     fn version(&self) -> u32 { 1 }
//! # }
//! let factory: Arc<dyn ExtensionFactory> = Arc::new(MyFactory);
//! register_extension(factory).expect("registration");
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, RwLock};

use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::{DType, Tensor};

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

/// The contract every out-of-tree extension primitive must satisfy.
///
/// Implementations appear in the core graph as
/// `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`. Every method is part of the
/// `ExtensionOp` spec (`docs/spec/extension-op.md`); the short form:
///
/// - identity via [`family_id`][Self::family_id] + [`payload_hash`][Self::payload_hash]
///   + [`payload_eq`][Self::payload_eq];
/// - fixed arity via [`n_inputs`][Self::n_inputs] / [`n_outputs`][Self::n_outputs];
/// - shape / dtype inference via [`infer_output_meta`][Self::infer_output_meta];
/// - forward dispatch via [`eager_execute`][Self::eager_execute] (used by both
///   the eager and compiled paths);
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
/// ```ignore
/// use std::sync::Arc;
/// use tenferro_ops::ext_op::ExtensionOp;
///
/// # struct MyExt;
/// # impl std::fmt::Debug for MyExt { fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) } }
/// # impl ExtensionOp for MyExt { /* ... */
/// #     fn family_id(&self) -> &'static str { "my-crate.my_op.v1" }
/// #     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
/// #     fn payload_eq(&self, _other: &dyn ExtensionOp) -> bool { true }
/// #     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { unimplemented!() }
/// #     fn n_inputs(&self) -> usize { 0 }
/// #     fn n_outputs(&self) -> usize { 0 }
/// #     fn infer_output_meta(&self, _: &[tenferro_tensor::DType], _: &[&[tenferro_ops::SymDim]]) -> Vec<(tenferro_tensor::DType, Vec<tenferro_ops::SymDim>)> { vec![] }
/// #     fn eager_execute(&self, _: &[&tenferro_tensor::Tensor]) -> tenferro_tensor::Result<Vec<tenferro_tensor::Tensor>> { unimplemented!() }
/// #     fn linearize(
/// #         &self,
/// #         _: &mut computegraph::fragment::FragmentBuilder<tenferro_ops::std_tensor_op::StdTensorOp>,
/// #         _: &[computegraph::types::GlobalValKey<tenferro_ops::std_tensor_op::StdTensorOp>],
/// #         _: &[computegraph::types::GlobalValKey<tenferro_ops::std_tensor_op::StdTensorOp>],
/// #         _: &[Option<computegraph::types::LocalValId>],
/// #         _: &mut tenferro_ops::ShapeGuardContext,
/// #     ) -> Vec<Option<computegraph::types::LocalValId>> { vec![] }
/// #     fn transpose_rule(
/// #         &self,
/// #         _: &mut dyn computegraph::OpEmitter<tenferro_ops::std_tensor_op::StdTensorOp>,
/// #         _: &[Option<computegraph::types::LocalValId>],
/// #         _: &[computegraph::types::ValRef<tenferro_ops::std_tensor_op::StdTensorOp>],
/// #         _: &computegraph::types::OpMode,
/// #         _: &mut tenferro_ops::ShapeGuardContext,
/// #     ) -> Vec<Option<computegraph::types::LocalValId>> { vec![] }
/// # }
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
    fn n_inputs(&self) -> usize;

    /// Number of outputs. MUST match the length of the vector returned by
    /// [`Self::infer_output_meta`].
    fn n_outputs(&self) -> usize;

    // ----- Shape and dtype inference (spec Section 7) -----

    /// Infer output dtypes and shapes for each output slot.
    ///
    /// `input_dtypes.len()` and `input_shapes.len()` both equal
    /// `self.n_inputs()`. The returned vector MUST have length
    /// `self.n_outputs()`, one `(dtype, shape)` entry per output slot.
    /// Shapes use [`SymDim`] so extension ops compose with graph-global
    /// symbolic metadata.
    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)>;

    // ----- Forward execution dispatch (spec Section 8) -----

    /// Eager forward execution; called from the eager path and indirectly
    /// from the compiled path.
    ///
    /// Input tensors are on the device the caller already arranged. Output
    /// tensors MUST have shapes matching [`Self::infer_output_meta`] and MUST
    /// be placed on a device the caller can consume.
    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>>;

    // ----- AD rules (spec Section 10) -----

    /// Emit the linear (JVP) rule.
    ///
    /// This legacy inline hook is retained so existing impl blocks remain
    /// source-compatible. AD dispatch uses registered [`ExtensionAdRule`]
    /// providers; new extension crates should register a rule instead of
    /// relying on this method.
    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        _tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        panic!(
            "extension family {:?} has no inline linearize rule; register an ExtensionAdRule",
            self.family_id()
        )
    }

    /// Emit the transpose (VJP) rule.
    ///
    /// This legacy inline hook is retained so existing impl blocks remain
    /// source-compatible. AD dispatch uses registered [`ExtensionAdRule`]
    /// providers; new extension crates should register a rule instead of
    /// relying on this method.
    fn transpose_rule(
        &self,
        _emitter: &mut dyn OpEmitter<StdTensorOp>,
        _cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        panic!(
            "extension family {:?} has no inline transpose rule; register an ExtensionAdRule",
            self.family_id()
        )
    }
}

/// AD rule provider for an extension family.
///
/// Rules are registered independently from [`ExtensionFactory`] so an
/// out-of-tree crate can provide a primal operation and AD behavior as separate
/// components. Rule methods receive the concrete [`ExtensionOp`] payload as a
/// trait object; implementations should downcast through [`ExtensionOp::as_any`]
/// when they need payload-specific parameters.
pub trait ExtensionAdRule: Debug + Send + Sync + 'static {
    /// The extension family this rule handles.
    fn family_id(&self) -> &'static str;

    /// Emit the linear (JVP) rule.
    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>;

    /// Emit the transpose (VJP) rule.
    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>;
}

/// Factory trait used at registration time.
///
/// An [`ExtensionFactory`] identifies a single extension family by its
/// [`ExtensionOp::family_id`] string and advertises the currently-registered
/// `version`. Serialization consumers use `version` to detect drift against
/// the on-wire `family_id`.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use tenferro_ops::ext_op::{register_extension, ExtensionFactory, ExtensionOp};
///
/// struct MyFactory;
/// impl ExtensionFactory for MyFactory {
///     fn family_id(&self) -> &'static str { "my-crate.my_op.v1" }
///     fn version(&self) -> u32 { 1 }
/// }
///
/// let _ = register_extension(Arc::new(MyFactory));
/// ```
pub trait ExtensionFactory: Send + Sync + 'static {
    /// The [`ExtensionOp::family_id`] that this factory is registered under.
    fn family_id(&self) -> &'static str;

    /// Current in-process version for this family.
    fn version(&self) -> u32;

    /// Optional: produce a default / zero-payload [`ExtensionOp`] instance
    /// for diagnostic or cross-process reconstruction purposes.
    fn instantiate_default(&self) -> Option<Arc<dyn ExtensionOp>> {
        None
    }
}

/// Errors returned from [`register_extension`].
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRegistryError {
    /// A factory with the same `family_id` was already registered.
    #[error("family_id {family_id:?} already registered")]
    Duplicate { family_id: &'static str },
    /// An AD rule with the same `family_id` was already registered.
    #[error("AD rule for family_id {family_id:?} already registered")]
    DuplicateRule { family_id: &'static str },
    /// The `family_id` does not match the namespaced format
    /// `"<crate-name>.<op-name>.v<major>"`.
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}

type FactoryMap = HashMap<&'static str, Arc<dyn ExtensionFactory>>;
type RuleMap = HashMap<&'static str, Arc<dyn ExtensionAdRule>>;

fn registry() -> &'static RwLock<FactoryMap> {
    static REG: OnceLock<RwLock<FactoryMap>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(HashMap::new()))
}

fn rule_registry() -> &'static RwLock<RuleMap> {
    static REG: OnceLock<RwLock<RuleMap>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(HashMap::new()))
}

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

/// Register a new extension factory.
///
/// The factory's `family_id` MUST follow the reserved format
/// `"<crate-name>.<op-name>.v<major>"` (spec Section 5) and MUST NOT collide
/// with any already-registered family. Returns
/// [`ExtensionRegistryError::Duplicate`] on collision and
/// [`ExtensionRegistryError::MalformedFamilyId`] on format violation.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use tenferro_ops::ext_op::{register_extension, ExtensionFactory};
///
/// struct MyFactory;
/// impl ExtensionFactory for MyFactory {
///     fn family_id(&self) -> &'static str { "my-crate.my_op.v1" }
///     fn version(&self) -> u32 { 1 }
/// }
///
/// register_extension(Arc::new(MyFactory)).expect("first registration");
/// ```
pub fn register_extension(
    factory: Arc<dyn ExtensionFactory>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = factory.family_id();
    if !is_valid_family_id(family_id) {
        return Err(ExtensionRegistryError::MalformedFamilyId { family_id });
    }
    let mut guard = registry()
        .write()
        .expect("extension registry RwLock poisoned");
    if guard.contains_key(family_id) {
        return Err(ExtensionRegistryError::Duplicate { family_id });
    }
    guard.insert(family_id, factory);
    Ok(())
}

/// Register a new extension AD rule.
///
/// The rule's `family_id` uses the same validation as
/// [`register_extension`]. Registering a rule does not require registering a
/// factory first; this lets crates split primal construction and AD support
/// across modules or optional features.
pub fn register_extension_rule(
    rule: Arc<dyn ExtensionAdRule>,
) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    if !is_valid_family_id(family_id) {
        return Err(ExtensionRegistryError::MalformedFamilyId { family_id });
    }
    let mut guard = rule_registry()
        .write()
        .expect("extension rule registry RwLock poisoned");
    if guard.contains_key(family_id) {
        return Err(ExtensionRegistryError::DuplicateRule { family_id });
    }
    guard.insert(family_id, rule);
    Ok(())
}

/// Look up a factory by `family_id`.
///
/// Returns `None` if no factory is registered for the given identifier.
/// Callers decide how to surface the absence (spec Section 12).
///
/// # Examples
///
/// ```ignore
/// use tenferro_ops::ext_op::lookup_extension_factory;
///
/// assert!(lookup_extension_factory("unknown.op.v1").is_none());
/// ```
pub fn lookup_extension_factory(family_id: &str) -> Option<Arc<dyn ExtensionFactory>> {
    registry()
        .read()
        .expect("extension registry RwLock poisoned")
        .get(family_id)
        .cloned()
}

/// Look up an extension AD rule by `family_id`.
pub fn lookup_extension_rule(family_id: &str) -> Option<Arc<dyn ExtensionAdRule>> {
    rule_registry()
        .read()
        .expect("extension rule registry RwLock poisoned")
        .get(family_id)
        .cloned()
}

/// Emit a registered extension linearization rule.
pub fn linearize_extension_rule(
    op: &dyn ExtensionOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    match lookup_extension_rule(op.family_id()) {
        Some(rule) => rule.linearize(op, builder, primal_in, primal_out, tangent_in, ctx),
        None => Err(ADRuleError::unsupported(
            op.family_id(),
            ADRuleKind::Linearize,
        )),
    }
}

/// Emit a registered extension transpose rule.
pub fn transpose_extension_rule(
    op: &dyn ExtensionOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    match lookup_extension_rule(op.family_id()) {
        Some(rule) => rule.transpose_rule(op, emitter, cotangent_out, inputs, mode, ctx),
        None => Err(ADRuleError::unsupported(
            op.family_id(),
            ADRuleKind::Transpose,
        )),
    }
}

/// Returns `true` when a factory with `family_id` is currently registered.
///
/// # Examples
///
/// ```ignore
/// use tenferro_ops::ext_op::is_extension_registered;
///
/// assert!(!is_extension_registered("unknown.op.v1"));
/// ```
pub fn is_extension_registered(family_id: &str) -> bool {
    lookup_extension_factory(family_id).is_some()
}

/// Returns `true` when an AD rule with `family_id` is currently registered.
pub fn is_extension_rule_registered(family_id: &str) -> bool {
    lookup_extension_rule(family_id).is_some()
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
