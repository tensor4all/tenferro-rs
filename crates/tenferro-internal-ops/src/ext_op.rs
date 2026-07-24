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
//! - AD rules are owned by the semantic AD registry in `tenferro-ad`; this
//!   extension-operation contract does not import an AD engine.
//! - Extension ops themselves do not require process-global registration.
//!   Frontends carry them directly as `Arc<dyn ExtensionOp>`.

use std::any::Any;
use std::error::Error as StdError;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueRef;
use tenferro_tensor::{DType, ErrorKind, Tensor, ValidationKind};

use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;
use crate::ExtensionShapeContext;

#[doc(hidden)]
pub use crate::shape_constraint::ExtensionShapeConstraint;

/// Canonical result of one extension metadata inference callback.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::ext_op::ExtensionShapeInference;
///
/// let inferred = ExtensionShapeInference {
///     output_metas: Vec::new(),
///     constraints: Vec::new(),
/// };
/// assert!(inferred.output_metas.is_empty());
/// assert!(inferred.constraints.is_empty());
/// ```
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionShapeInference {
    /// Output dtype and symbolic-shape metadata in output-slot order.
    pub output_metas: Vec<(DType, Vec<SymDim>)>,
    /// Shape requirements recorded by the callback.
    pub constraints: Vec<ExtensionShapeConstraint>,
}

/// Invoke an extension metadata callback after validating its declared arity.
///
/// # Examples
///
/// ```rust
/// use std::any::Any;
/// use std::sync::Arc;
/// use tenferro_ops::ext_op::{invoke_extension_shape_inference, ExtensionOp};
/// use tenferro_ops::{ExtensionShapeContext, SymDim};
/// use tenferro_tensor::DType;
///
/// #[derive(Clone, Debug)]
/// struct Identity;
///
/// impl ExtensionOp for Identity {
///     fn family_id(&self) -> &'static str { "example.identity.v1" }
///     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
///     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
///         other.as_any().downcast_ref::<Self>().is_some()
///     }
///     fn clone_arc(&self) -> Arc<dyn ExtensionOp> { Arc::new(self.clone()) }
///     fn as_any(&self) -> &dyn Any { self }
///     fn input_count(&self) -> usize { 1 }
///     fn output_count(&self) -> usize { 1 }
///     fn infer_output_meta(
///         &self,
///         ctx: &mut ExtensionShapeContext<'_>,
///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
///         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
///     }
/// }
///
/// let shape = [SymDim::from(3usize)];
/// let inferred = invoke_extension_shape_inference(
///     &Identity,
///     &[DType::F64],
///     &[&shape],
/// ).unwrap();
/// assert_eq!(inferred.output_metas, vec![(DType::F64, shape.to_vec())]);
/// ```
#[doc(hidden)]
pub fn invoke_extension_shape_inference(
    op: &dyn ExtensionOp,
    input_dtypes: &[DType],
    input_shapes: &[&[SymDim]],
) -> tenferro_tensor::Result<ExtensionShapeInference> {
    let expected_inputs = op.input_count();
    if input_dtypes.len() != expected_inputs || input_shapes.len() != expected_inputs {
        return Err(tenferro_tensor::Error::invalid_argument(
            "extension",
            "input metadata",
            format!(
                "family_id={:?}: infer_output_meta expects {expected_inputs} input metadata entries, got {} dtypes and {} shapes",
                op.family_id(),
                input_dtypes.len(),
                input_shapes.len()
            ),
        ));
    }

    let mut ctx =
        ExtensionShapeContext::new_for_inference(op.family_id(), input_dtypes, input_shapes);
    let output_metas = op.infer_output_meta(&mut ctx)?;
    if output_metas.len() != op.output_count() {
        return Err(tenferro_tensor::Error::invalid_argument(
            "extension",
            "output metadata",
            format!(
                "family_id={:?}: infer_output_meta produced {} output metadata entries; op declared {} outputs",
                op.family_id(),
                output_metas.len(),
                op.output_count()
            ),
        ));
    }

    Ok(ExtensionShapeInference {
        output_metas,
        constraints: ctx.into_constraints(),
    })
}

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
#[derive(Debug, thiserror::Error)]
pub enum ExtensionLoweringError {
    /// A lowering failure that has no typed source.
    #[error("{message}")]
    Message {
        /// Human-readable lowering detail.
        message: String,
        /// Coarse classification supplied by the extension owner.
        kind: ErrorKind,
    },
    /// A lowering failure retaining the domain source that caused it.
    #[error("{source}")]
    Source {
        /// Coarse classification supplied by the extension owner.
        kind: ErrorKind,
        /// Original typed lowering source.
        #[source]
        source: Box<dyn StdError + Send + Sync + 'static>,
    },
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
        Self::new_with_kind(
            ErrorKind::Validation(ValidationKind::InvalidArgument),
            message,
        )
    }

    /// Create a lowering error with an explicit coarse classification.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::ExtensionLoweringError;
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let err = ExtensionLoweringError::new_with_kind(
    ///     ErrorKind::Unsupported,
    ///     "extension is not supported by this lowering target",
    /// );
    /// assert_eq!(err.kind(), ErrorKind::Unsupported);
    /// ```
    pub fn new_with_kind(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
            kind,
        }
    }

    /// Create a lowering error while retaining a typed source.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error as _;
    /// use tenferro_ops::ext_op::ExtensionLoweringError;
    ///
    /// let source = std::io::Error::new(std::io::ErrorKind::Other, "shape unavailable");
    /// let err = ExtensionLoweringError::from_source(source);
    /// assert!(err.source().is_some());
    /// ```
    pub fn from_source<E>(source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::from_source_with_kind(
            ErrorKind::Validation(ValidationKind::InvalidArgument),
            source,
        )
    }

    /// Create a lowering error with a typed source and explicit classification.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::ExtensionLoweringError;
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let err = ExtensionLoweringError::from_source_with_kind(
    ///     ErrorKind::BackendFailure,
    ///     std::io::Error::other("backend rejected lowering"),
    /// );
    /// assert_eq!(err.kind(), ErrorKind::BackendFailure);
    /// assert!(std::error::Error::source(&err).is_some());
    /// ```
    pub fn from_source_with_kind<E>(kind: ErrorKind, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Source {
            kind,
            source: Box::new(source),
        }
    }

    /// Return the stable classification carried by this lowering failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::ExtensionLoweringError;
    /// use tenferro_tensor::ErrorKind;
    ///
    /// let error = ExtensionLoweringError::new_with_kind(
    ///     ErrorKind::Unsupported,
    ///     "target has no lowering",
    /// );
    /// assert_eq!(error.kind(), ErrorKind::Unsupported);
    /// ```
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Message { kind, .. } | Self::Source { kind, .. } => *kind,
        }
    }
}

/// Result returned by [`ExtensionOp::lower_to_standard_ops`].
pub type ExtensionLoweringResult =
    std::result::Result<Option<Vec<ValueRef<StdTensorOp>>>, ExtensionLoweringError>;

/// Typed result of trying to lower an extension into standard tensor ops.
///
/// This is the Phase 6 compatibility boundary for the legacy
/// [`ExtensionOp::lower_to_standard_ops`] hook. New callers should branch on
/// this enum instead of treating `Ok(None)` as a capability protocol.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::ExtensionStandardLowering;
///
/// let outcome = ExtensionStandardLowering::Unsupported;
/// assert!(matches!(outcome, ExtensionStandardLowering::Unsupported));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtensionStandardLowering {
    /// The extension emitted standard tensor graph outputs.
    Lowered(Vec<ValueRef<StdTensorOp>>),
    /// The extension has no standard-op lowering for the supplied metadata.
    Unsupported,
}

impl ExtensionStandardLowering {
    /// Convert the legacy optional lowering result into an explicit outcome.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::ExtensionStandardLowering;
    ///
    /// assert_eq!(
    ///     ExtensionStandardLowering::from_legacy(None),
    ///     ExtensionStandardLowering::Unsupported,
    /// );
    /// ```
    pub fn from_legacy(value: Option<Vec<ValueRef<StdTensorOp>>>) -> Self {
        match value {
            Some(outputs) => Self::Lowered(outputs),
            None => Self::Unsupported,
        }
    }
}

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
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for invalid input
    /// shapes/dtypes or output arity, [`tenferro_tensor::Error::Unsupported`]
    /// when the reference implementation does not support an operation, or a
    /// typed [`tenferro_tensor::Error::BackendSource`] failure from execution.
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
/// use tenferro_ops::{ExtensionShapeContext, SymDim};
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
///         ctx: &mut ExtensionShapeContext<'_>,
///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
///         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
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

    /// Declare observable semantic effects for this extension payload.
    ///
    /// The compatibility default is deliberately `Undeclared`, not pure.
    /// Semantic-program construction rejects an undeclared payload so an
    /// extension cannot silently acquire purity during migration.
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Undeclared
    }

    /// Declare semantic output aliasing for this extension payload.
    ///
    /// The compatibility default is deliberately `Undeclared`, not fresh.
    /// Execution-only users may continue to carry an older payload, while
    /// semantic-program construction requires an explicit declaration.
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::Undeclared
    }

    // ----- Shape and dtype inference (spec Section 7) -----

    /// Infer output dtypes and shapes for each output slot.
    ///
    /// The canonical inference driver validates arity before invoking this
    /// callback. Implementations MUST validate rank, dtype, axis, and other
    /// input-derived metadata through `ctx` before using it. Invalid public
    /// input must return a typed error rather than an empty sentinel or panic.
    ///
    /// On success, the returned vector MUST have length `self.output_count()`,
    /// one `(dtype, shape)` entry per output slot. Shapes use [`SymDim`] so
    /// extension ops compose with graph-global symbolic metadata.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for invalid rank, axis,
    /// or dtype metadata, or [`tenferro_tensor::Error::RuntimeState`] when the
    /// output contract cannot be inferred from unavailable metadata.
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
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
    /// This is the legacy compatibility hook. New lowering callers should call
    /// [`Self::lower_to_standard_ops_typed`] so the unsupported case is explicit.
    /// Return `Ok(Some(outputs))` after adding only standard [`StdTensorOp`]
    /// operations to `builder`. Return `Ok(None)` when this extension family has
    /// no standard-op lowering for the supplied metadata. Return
    /// [`ExtensionLoweringError`] when the payload is malformed or the lowering
    /// detects invalid metadata.
    ///
    /// The default implementation returns `Ok(None)` so existing extension
    /// runtimes keep their native dispatch behavior until their owning crate
    /// deliberately implements this hook.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionLoweringError`] when the payload or input metadata
    /// cannot be lowered safely.
    fn lower_to_standard_ops(
        &self,
        _builder: &mut GraphBuilder<StdTensorOp>,
        _inputs: &[ValueRef<StdTensorOp>],
        _input_dtypes: &[DType],
        _input_shapes: &[&[SymDim]],
    ) -> ExtensionLoweringResult {
        Ok(None)
    }

    /// Try to expand this extension into standard tensor graph operations.
    ///
    /// This method preserves existing extension implementations while removing
    /// `Ok(None)` from new call sites. [`ExtensionStandardLowering::Unsupported`]
    /// means a lowerer may try a configured fallback; an
    /// [`ExtensionLoweringError`] remains a real lowering failure and must not be
    /// converted into a capability miss.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionLoweringError`] when the extension payload or input
    /// metadata cannot be lowered safely.
    fn lower_to_standard_ops_typed(
        &self,
        builder: &mut GraphBuilder<StdTensorOp>,
        inputs: &[ValueRef<StdTensorOp>],
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Result<ExtensionStandardLowering, ExtensionLoweringError> {
        self.lower_to_standard_ops(builder, inputs, input_dtypes, input_shapes)
            .map(ExtensionStandardLowering::from_legacy)
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

    // AD rules are registered separately in `tenferro-ad`.
}

/// Access mode for one extension-declared semantic resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtensionEffectAccess {
    /// Read-only access.
    Read,
    /// Mutating access.
    Write,
}

/// Backend-neutral resource access declared by an extension payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtensionEffect {
    /// Stable versioned resource family.
    pub family: &'static str,
    /// Family-local resource identity.
    pub key: u64,
    /// Read or write access.
    pub access: ExtensionEffectAccess,
}

/// Explicit effect declaration returned by an extension payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionEffectDeclaration<'a> {
    /// The payload has not been migrated to the semantic contract.
    Undeclared,
    /// Complete ordered effect list; an empty slice explicitly means pure.
    Declared(&'a [ExtensionEffect]),
}

/// One extension-declared output alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtensionAlias {
    /// The output is semantically fresh.
    Fresh {
        /// Operation-local output index.
        output: usize,
    },
    /// The output is a view of an input.
    ViewOf {
        /// Operation-local output index.
        output: usize,
        /// Operation-local input index.
        input: usize,
    },
    /// The output must alias an input.
    MustAlias {
        /// Operation-local output index.
        output: usize,
        /// Operation-local input index.
        input: usize,
    },
    /// The output aliases an external typed resource.
    ExternalAlias {
        /// Operation-local output index.
        output: usize,
        /// Stable versioned resource family.
        family: &'static str,
        /// Family-local resource identity.
        key: u64,
    },
}

/// Explicit alias declaration returned by an extension payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionAliasDeclaration<'a> {
    /// The payload has not been migrated to the semantic contract.
    Undeclared,
    /// Every output is semantically fresh.
    AllFresh,
    /// Complete ordered alias list.
    Declared(&'a [ExtensionAlias]),
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
