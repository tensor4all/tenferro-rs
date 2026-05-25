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
//! - AD rules are registered separately through [`ExtensionChainRule`] and
//!   [`register_extension_chain_rule`] for ChainRules-style code, or through
//!   the lower-level [`ExtensionAdRule`] / [`register_extension_rule`] adapter
//!   interface. A rule may emit core [`StdTensorOp`] values and registered
//!   `Extension` values so out-of-tree operations remain in the same graph.
//! - Extension ops themselves do not require process-global registration.
//!   Frontends carry them directly as `Arc<dyn ExtensionOp>`.

use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
#[cfg(feature = "autodiff")]
use computegraph::fragment::FragmentBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::GraphOp;
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;
use tenferro_tensor::{DType, Tensor};

#[cfg(feature = "autodiff")]
use crate::ad::context::ShapeGuardContext;
#[cfg(feature = "autodiff")]
use crate::dim_expr::DimExpr;
#[cfg(feature = "autodiff")]
use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;
#[cfg(feature = "autodiff")]
use std::collections::HashMap;
#[cfg(feature = "autodiff")]
use std::sync::{OnceLock, RwLock};

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
/// - AD via a separately registered [`ExtensionChainRule`] or
///   [`ExtensionAdRule`].
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
///     fn n_inputs(&self) -> usize { 1 }
///     fn n_outputs(&self) -> usize { 1 }
///     fn infer_output_meta(
///         &self,
///         dtypes: &[DType],
///         shapes: &[&[SymDim]],
///     ) -> Vec<(DType, Vec<SymDim>)> {
///         vec![(dtypes[0], shapes[0].to_vec())]
///     }
///     fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
///         Ok(vec![inputs[0].clone()])
///     }
/// }
///
/// let op: Arc<dyn ExtensionOp> = Arc::new(IdentityExt);
/// assert_eq!(op.n_inputs(), 1);
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

    // AD rules are registered separately; see [`ExtensionChainRule`].
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

/// Opaque value handle used by extension `frule` and `rrule` builders.
///
/// Extension rule authors receive `AdValue`s from [`FruleBuilder::tangent`],
/// [`FruleBuilder::primal`], [`RRuleBuilder::cotangent`], and
/// [`RRuleBuilder::primal`], then feed them into builder helper methods. This
/// keeps `LocalValId` construction inside tenferro while still allowing rules
/// to emit core and extension ops.
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::AdValue;
///
/// fn inspect(value: &AdValue) -> bool {
///     value.is_active()
/// }
/// ```
#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdValue {
    value: ValRef<StdTensorOp>,
}

#[cfg(feature = "autodiff")]
impl AdValue {
    fn local(local_id: LocalValId) -> Self {
        Self {
            value: ValRef::Local(local_id),
        }
    }

    fn external(key: GlobalValKey<StdTensorOp>) -> Self {
        Self {
            value: ValRef::External(key),
        }
    }

    fn as_val_ref(&self) -> &ValRef<StdTensorOp> {
        &self.value
    }

    fn into_val_ref(self) -> ValRef<StdTensorOp> {
        self.value
    }

    fn local_id(&self) -> Option<LocalValId> {
        match self.value {
            ValRef::Local(local_id) => Some(local_id),
            ValRef::External(_) => None,
        }
    }

    /// Returns `true` when this value is part of the active linear graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::ext_op::AdValue;
    ///
    /// fn is_linear_value(value: &AdValue) -> bool {
    ///     value.is_active()
    /// }
    /// ```
    pub fn is_active(&self) -> bool {
        self.local_id().is_some()
    }
}

/// ChainRules-style AD rule provider for an extension family.
///
/// Implement [`Self::frule`] for JVP and [`Self::rrule`] for VJP. The builder
/// arguments provide named accessors such as `tangent(0)` and `cotangent(0)`
/// plus helper methods that emit `StdTensorOp` or registered extension ops.
///
/// # Examples
///
/// ```
/// use chainrules_core::ADRuleResult;
/// use tenferro_ops::ext_op::{ExtensionChainRule, ExtensionOp, FruleBuilder, RRuleBuilder};
///
/// #[derive(Debug)]
/// struct IdentityRule;
///
/// impl ExtensionChainRule for IdentityRule {
///     fn family_id(&self) -> &'static str { "example.identity.v1" }
///     fn frule(&self, _op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
///         let dx = cx.tangent(0)?;
///         cx.set_output_tangent(0, dx)
///     }
///     fn rrule(&self, _op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
///         let dy = cx.cotangent(0)?;
///         cx.set_input_cotangent(0, dy)
///     }
/// }
/// ```
#[cfg(feature = "autodiff")]
pub trait ExtensionChainRule: Debug + Send + Sync + 'static {
    /// The extension family this rule handles.
    fn family_id(&self) -> &'static str;

    /// Emit the forward-mode rule.
    fn frule(&self, op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()>;

    /// Emit the reverse-mode rule.
    fn rrule(&self, op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()>;
}

/// Builder passed to [`ExtensionChainRule::frule`].
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::FruleBuilder;
///
/// fn accepts_builder(_cx: &mut FruleBuilder<'_>) {}
/// ```
#[cfg(feature = "autodiff")]
pub struct FruleBuilder<'a> {
    family_id: &'static str,
    builder: &'a mut FragmentBuilder<StdTensorOp>,
    primal_in: &'a [GlobalValKey<StdTensorOp>],
    primal_out: &'a [GlobalValKey<StdTensorOp>],
    tangent_in: &'a [Option<LocalValId>],
    ctx: &'a mut ShapeGuardContext,
    output_tangents: Vec<Option<LocalValId>>,
}

#[cfg(feature = "autodiff")]
impl<'a> FruleBuilder<'a> {
    fn new(
        family_id: &'static str,
        op: &dyn ExtensionOp,
        builder: &'a mut FragmentBuilder<StdTensorOp>,
        primal_in: &'a [GlobalValKey<StdTensorOp>],
        primal_out: &'a [GlobalValKey<StdTensorOp>],
        tangent_in: &'a [Option<LocalValId>],
        ctx: &'a mut ShapeGuardContext,
    ) -> Self {
        Self {
            family_id,
            builder,
            primal_in,
            primal_out,
            tangent_in,
            ctx,
            output_tangents: vec![None; op.n_outputs()],
        }
    }

    /// Return the tangent for primal input `input_index`, if active.
    pub fn tangent(&self, input_index: usize) -> ADRuleResult<Option<AdValue>> {
        self.tangent_in
            .get(input_index)
            .copied()
            .map(|tangent| tangent.map(AdValue::local))
            .ok_or_else(|| self.error(ADRuleKind::Linearize, "input tangent index out of bounds"))
    }

    /// Return the primal input value for `input_index`.
    pub fn primal(&self, input_index: usize) -> ADRuleResult<AdValue> {
        self.primal_in
            .get(input_index)
            .cloned()
            .map(AdValue::external)
            .ok_or_else(|| self.error(ADRuleKind::Linearize, "primal input index out of bounds"))
    }

    /// Return the primal output value for `output_index`.
    pub fn output(&self, output_index: usize) -> ADRuleResult<AdValue> {
        self.primal_out
            .get(output_index)
            .cloned()
            .map(AdValue::external)
            .ok_or_else(|| self.error(ADRuleKind::Linearize, "primal output index out of bounds"))
    }

    /// Return the rank of primal input `input_index`.
    pub fn input_rank(&mut self, input_index: usize) -> ADRuleResult<usize> {
        let primal = self.primal(input_index)?;
        self.ctx
            .try_shape_of(primal.as_val_ref())
            .map(|shape| shape.len())
            .ok_or_else(|| self.error(ADRuleKind::Linearize, "missing input shape metadata"))
    }

    /// Emit an arbitrary standard tensor op.
    pub fn emit(&mut self, op: StdTensorOp, inputs: Vec<AdValue>) -> ADRuleResult<Vec<AdValue>> {
        emit_with_builder(
            self.family_id,
            ADRuleKind::Linearize,
            self.builder,
            op,
            inputs,
        )
    }

    /// Emit `lhs + rhs`.
    pub fn add(&mut self, lhs: AdValue, rhs: AdValue) -> ADRuleResult<AdValue> {
        self.emit_one(StdTensorOp::Add, vec![lhs, rhs])
    }

    /// Emit `lhs * rhs`.
    pub fn mul(&mut self, lhs: AdValue, rhs: AdValue) -> ADRuleResult<AdValue> {
        self.emit_one(StdTensorOp::Mul, vec![lhs, rhs])
    }

    /// Emit `sum(value)` over axes `0..rank`.
    pub fn reduce_sum_all(&mut self, value: AdValue, rank: usize) -> ADRuleResult<AdValue> {
        self.emit_one(
            StdTensorOp::ReduceSum {
                axes: (0..rank).collect(),
            },
            vec![value],
        )
    }

    /// Broadcast a scalar-like value to the shape of primal input `input_index`.
    pub fn broadcast_scalar_to_input(
        &mut self,
        value: AdValue,
        input_index: usize,
    ) -> ADRuleResult<AdValue> {
        let primal = self.primal(input_index)?;
        let shape = self.input_shape_as_dim_exprs(input_index, 1)?;
        let inputs = if shape.is_empty() {
            vec![value]
        } else {
            vec![value, primal]
        };
        self.emit_one(
            StdTensorOp::BroadcastInDim {
                shape,
                dims: Vec::new(),
            },
            inputs,
        )
    }

    /// Emit a registered extension op inside this rule.
    pub fn apply_extension(
        &mut self,
        op: Arc<dyn ExtensionOp>,
        inputs: Vec<AdValue>,
    ) -> ADRuleResult<Vec<AdValue>> {
        self.emit(StdTensorOp::Extension(op), inputs)
    }

    /// Set the tangent for output `output_index`.
    pub fn set_output_tangent(
        &mut self,
        output_index: usize,
        tangent: Option<AdValue>,
    ) -> ADRuleResult<()> {
        let local = tangent
            .map(|value| {
                value.local_id().ok_or_else(|| {
                    self.error(
                        ADRuleKind::Linearize,
                        "output tangent must be an emitted local value",
                    )
                })
            })
            .transpose()?;
        if output_index >= self.output_tangents.len() {
            return Err(self.error(ADRuleKind::Linearize, "output tangent index out of bounds"));
        }
        self.output_tangents[output_index] = local;
        Ok(())
    }

    fn emit_one(&mut self, op: StdTensorOp, inputs: Vec<AdValue>) -> ADRuleResult<AdValue> {
        let mut outputs = self.emit(op, inputs)?;
        debug_assert_eq!(outputs.len(), 1);
        Ok(outputs.remove(0))
    }

    fn input_shape_as_dim_exprs(
        &mut self,
        input_index: usize,
        source_idx: usize,
    ) -> ADRuleResult<Vec<DimExpr>> {
        let rank = self.input_rank(input_index)?;
        Ok((0..rank)
            .map(|axis| DimExpr::InputDim {
                input_idx: source_idx,
                axis,
            })
            .collect())
    }

    fn finish(self) -> Vec<Option<LocalValId>> {
        self.output_tangents
    }

    fn error(&self, rule: ADRuleKind, message: &str) -> ADRuleError {
        chain_rule_error(self.family_id, rule, message)
    }
}

/// Builder passed to [`ExtensionChainRule::rrule`].
///
/// # Examples
///
/// ```
/// use tenferro_ops::ext_op::RRuleBuilder;
///
/// fn accepts_builder(_cx: &mut RRuleBuilder<'_>) {}
/// ```
#[cfg(feature = "autodiff")]
pub struct RRuleBuilder<'a> {
    family_id: &'static str,
    emitter: &'a mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &'a [Option<LocalValId>],
    inputs: &'a [ValRef<StdTensorOp>],
    ctx: &'a mut ShapeGuardContext,
    input_cotangents: Vec<Option<LocalValId>>,
}

#[cfg(feature = "autodiff")]
impl<'a> RRuleBuilder<'a> {
    fn new(
        family_id: &'static str,
        op: &dyn ExtensionOp,
        emitter: &'a mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &'a [Option<LocalValId>],
        inputs: &'a [ValRef<StdTensorOp>],
        ctx: &'a mut ShapeGuardContext,
    ) -> Self {
        Self {
            family_id,
            emitter,
            cotangent_out,
            inputs,
            ctx,
            input_cotangents: vec![None; op.n_inputs()],
        }
    }

    /// Return the cotangent for primal output `output_index`, if active.
    pub fn cotangent(&self, output_index: usize) -> ADRuleResult<Option<AdValue>> {
        self.cotangent_out
            .get(output_index)
            .copied()
            .map(|cotangent| cotangent.map(AdValue::local))
            .ok_or_else(|| {
                self.error(
                    ADRuleKind::Transpose,
                    "output cotangent index out of bounds",
                )
            })
    }

    /// Return the primal input value for `input_index`.
    pub fn primal(&self, input_index: usize) -> ADRuleResult<AdValue> {
        self.inputs
            .get(input_index)
            .cloned()
            .map(|value| AdValue { value })
            .ok_or_else(|| self.error(ADRuleKind::Transpose, "primal input index out of bounds"))
    }

    /// Return the rank of primal input `input_index`.
    pub fn input_rank(&mut self, input_index: usize) -> ADRuleResult<usize> {
        let primal = self.primal(input_index)?;
        self.ctx
            .try_shape_of(primal.as_val_ref())
            .map(|shape| shape.len())
            .ok_or_else(|| self.error(ADRuleKind::Transpose, "missing input shape metadata"))
    }

    /// Emit an arbitrary standard tensor op.
    pub fn emit(&mut self, op: StdTensorOp, inputs: Vec<AdValue>) -> ADRuleResult<Vec<AdValue>> {
        emit_with_emitter(
            self.family_id,
            ADRuleKind::Transpose,
            self.emitter,
            op,
            inputs,
        )
    }

    /// Emit `lhs + rhs`.
    pub fn add(&mut self, lhs: AdValue, rhs: AdValue) -> ADRuleResult<AdValue> {
        self.emit_one(StdTensorOp::Add, vec![lhs, rhs])
    }

    /// Emit `lhs * rhs`.
    pub fn mul(&mut self, lhs: AdValue, rhs: AdValue) -> ADRuleResult<AdValue> {
        self.emit_one(StdTensorOp::Mul, vec![lhs, rhs])
    }

    /// Emit `sum(value)` over axes `0..rank`.
    pub fn reduce_sum_all(&mut self, value: AdValue, rank: usize) -> ADRuleResult<AdValue> {
        self.emit_one(
            StdTensorOp::ReduceSum {
                axes: (0..rank).collect(),
            },
            vec![value],
        )
    }

    /// Broadcast a scalar-like value to the shape of primal input `input_index`.
    pub fn broadcast_scalar_to_input(
        &mut self,
        value: AdValue,
        input_index: usize,
    ) -> ADRuleResult<AdValue> {
        let primal = self.primal(input_index)?;
        let shape = self.input_shape_as_dim_exprs(input_index, 1)?;
        let inputs = if shape.is_empty() {
            vec![value]
        } else {
            vec![value, primal]
        };
        self.emit_one(
            StdTensorOp::BroadcastInDim {
                shape,
                dims: Vec::new(),
            },
            inputs,
        )
    }

    /// Emit a registered extension op inside this rule.
    pub fn apply_extension(
        &mut self,
        op: Arc<dyn ExtensionOp>,
        inputs: Vec<AdValue>,
    ) -> ADRuleResult<Vec<AdValue>> {
        self.emit(StdTensorOp::Extension(op), inputs)
    }

    /// Set the cotangent for input `input_index`.
    pub fn set_input_cotangent(
        &mut self,
        input_index: usize,
        cotangent: Option<AdValue>,
    ) -> ADRuleResult<()> {
        let local = cotangent
            .map(|value| {
                value.local_id().ok_or_else(|| {
                    self.error(
                        ADRuleKind::Transpose,
                        "input cotangent must be an emitted local value",
                    )
                })
            })
            .transpose()?;
        if input_index >= self.input_cotangents.len() {
            return Err(self.error(ADRuleKind::Transpose, "input cotangent index out of bounds"));
        }
        self.input_cotangents[input_index] = local;
        Ok(())
    }

    fn emit_one(&mut self, op: StdTensorOp, inputs: Vec<AdValue>) -> ADRuleResult<AdValue> {
        let mut outputs = self.emit(op, inputs)?;
        debug_assert_eq!(outputs.len(), 1);
        Ok(outputs.remove(0))
    }

    fn input_shape_as_dim_exprs(
        &mut self,
        input_index: usize,
        source_idx: usize,
    ) -> ADRuleResult<Vec<DimExpr>> {
        let rank = self.input_rank(input_index)?;
        Ok((0..rank)
            .map(|axis| DimExpr::InputDim {
                input_idx: source_idx,
                axis,
            })
            .collect())
    }

    fn finish(self) -> Vec<Option<LocalValId>> {
        self.input_cotangents
    }

    fn error(&self, rule: ADRuleKind, message: &str) -> ADRuleError {
        chain_rule_error(self.family_id, rule, message)
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct ChainRuleAdapter {
    rule: Arc<dyn ExtensionChainRule>,
}

#[cfg(feature = "autodiff")]
impl ExtensionAdRule for ChainRuleAdapter {
    fn family_id(&self) -> &'static str {
        self.rule.family_id()
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let mut cx = FruleBuilder::new(
            self.family_id(),
            op,
            builder,
            primal_in,
            primal_out,
            tangent_in,
            ctx,
        );
        self.rule.frule(op, &mut cx)?;
        Ok(cx.finish())
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let mut cx = RRuleBuilder::new(self.family_id(), op, emitter, cotangent_out, inputs, ctx);
        self.rule.rrule(op, &mut cx)?;
        Ok(cx.finish())
    }
}

#[cfg(feature = "autodiff")]
fn emit_with_builder(
    family_id: &'static str,
    rule: ADRuleKind,
    builder: &mut FragmentBuilder<StdTensorOp>,
    op: StdTensorOp,
    inputs: Vec<AdValue>,
) -> ADRuleResult<Vec<AdValue>> {
    let input_count = op.n_inputs();
    if inputs.len() != input_count {
        return Err(chain_rule_error(
            family_id,
            rule,
            "emitted op input count does not match op arity",
        ));
    }
    let mode = mode_for(&inputs);
    let inputs = inputs.into_iter().map(AdValue::into_val_ref).collect();
    Ok(builder
        .add_op(op, inputs, mode)
        .into_iter()
        .map(AdValue::local)
        .collect())
}

#[cfg(feature = "autodiff")]
fn emit_with_emitter(
    family_id: &'static str,
    rule: ADRuleKind,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    op: StdTensorOp,
    inputs: Vec<AdValue>,
) -> ADRuleResult<Vec<AdValue>> {
    let input_count = op.n_inputs();
    if inputs.len() != input_count {
        return Err(chain_rule_error(
            family_id,
            rule,
            "emitted op input count does not match op arity",
        ));
    }
    let mode = mode_for(&inputs);
    let inputs = inputs.into_iter().map(AdValue::into_val_ref).collect();
    Ok(emitter
        .add_op(op, inputs, mode)
        .into_iter()
        .map(AdValue::local)
        .collect())
}

#[cfg(feature = "autodiff")]
fn mode_for(inputs: &[AdValue]) -> OpMode {
    let active_mask = inputs.iter().map(AdValue::is_active).collect::<Vec<_>>();
    if active_mask.iter().any(|&active| active) {
        OpMode::Linear { active_mask }
    } else {
        OpMode::Primal
    }
}

#[cfg(feature = "autodiff")]
fn chain_rule_error(family_id: &'static str, rule: ADRuleKind, message: &str) -> ADRuleError {
    ADRuleError::unsupported(format!("{family_id}: {message}"), rule)
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

#[cfg(feature = "autodiff")]
fn rule_registry() -> &'static RwLock<RuleMap> {
    static REG: OnceLock<RwLock<RuleMap>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(HashMap::new()))
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

/// Register a new extension AD rule.
///
/// The rule's `family_id` MUST follow the reserved format
/// `"<crate-name>.<op-name>.v<major>"`. Registering a rule does not require
/// registering the primal op first; frontends carry the op payload directly.
#[cfg(feature = "autodiff")]
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

/// Register a ChainRules-style extension AD rule.
///
/// This is the preferred API for new extension crates. Internally it adapts
/// [`ExtensionChainRule::frule`] and [`ExtensionChainRule::rrule`] to the
/// lower-level [`ExtensionAdRule`] registry used by the AD dispatcher.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use chainrules_core::ADRuleResult;
/// use tenferro_ops::ext_op::{
///     register_extension_chain_rule, ExtensionChainRule, ExtensionOp, FruleBuilder, RRuleBuilder,
/// };
///
/// #[derive(Debug)]
/// struct IdentityRule;
///
/// impl ExtensionChainRule for IdentityRule {
///     fn family_id(&self) -> &'static str { "example.register_chain_rule.v1" }
///     fn frule(&self, _op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
///         let dx = cx.tangent(0)?;
///         cx.set_output_tangent(0, dx)
///     }
///     fn rrule(&self, _op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
///         let dy = cx.cotangent(0)?;
///         cx.set_input_cotangent(0, dy)
///     }
/// }
///
/// let _ = register_extension_chain_rule(Arc::new(IdentityRule));
/// ```
#[cfg(feature = "autodiff")]
pub fn register_extension_chain_rule(
    rule: Arc<dyn ExtensionChainRule>,
) -> Result<(), ExtensionRegistryError> {
    register_extension_rule(Arc::new(ChainRuleAdapter { rule }))
}

/// Look up an extension AD rule by `family_id`.
#[cfg(feature = "autodiff")]
pub fn lookup_extension_rule(family_id: &str) -> Option<Arc<dyn ExtensionAdRule>> {
    rule_registry()
        .read()
        .expect("extension rule registry RwLock poisoned")
        .get(family_id)
        .cloned()
}

/// Emit a registered extension linearization rule.
#[cfg(feature = "autodiff")]
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
#[cfg(feature = "autodiff")]
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

/// Returns `true` when an AD rule with `family_id` is currently registered.
#[cfg(feature = "autodiff")]
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
