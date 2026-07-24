//! Automatic differentiation rules for [`StdTensorOp`].
//!
//! `linearize` and `transpose_rule` are separate graph-level contracts.
//! Core ops keep their rules here; extension ops own their own AD support
//! through the extension trait.

pub mod context;

#[cfg(feature = "autodiff")]
mod analytic;
#[cfg(feature = "autodiff")]
mod contraction;
#[cfg(feature = "autodiff")]
mod diagonal;
#[cfg(feature = "autodiff")]
mod dynamic;
#[cfg(feature = "autodiff")]
mod elementwise;
#[cfg(feature = "autodiff")]
mod indexing;
#[cfg(feature = "autodiff")]
pub(crate) mod registry;
#[cfg(feature = "autodiff")]
mod semiring;
#[cfg(feature = "autodiff")]
mod structural;
#[cfg(feature = "autodiff")]
#[doc(hidden)]
pub mod support;
#[cfg(feature = "autodiff")]
#[doc(hidden)]
pub mod transpose_input;
#[cfg(feature = "autodiff")]
mod zeros;

#[cfg(feature = "autodiff")]
use computegraph::graph::GraphBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
#[cfg(feature = "autodiff")]
use tidu::{
    ADRuleError, ADRuleKind, ADRuleResult, PrimitiveBuilder, PrimitiveTransposeInput,
    PrimitiveValue,
};

#[cfg(feature = "autodiff")]
use crate::ext_op::{dispatch_extension_linearize, dispatch_extension_transpose};
#[cfg(feature = "autodiff")]
use crate::std_tensor_op::StdTensorOp;

#[cfg(feature = "autodiff")]
fn missing_primitive_kind(op: &StdTensorOp, rule: ADRuleKind) -> ADRuleError {
    ADRuleError::invalid_input(
        "tenferro-internal-ops primitive AD dispatch",
        rule,
        format!("non-extension operation has no primitive kind: {op:?}"),
    )
}

/// Builder interface used by tenferro AD rules.
///
/// # Examples
///
/// ```
/// use computegraph::graph::GraphBuilder;
/// use computegraph::{OperationRole, ValueRef};
/// use tenferro_ops::ad::PrimitiveRuleBuilder;
/// use tenferro_ops::input_key::TensorInputKey;
/// use tenferro_ops::std_tensor_op::StdTensorOp;
///
/// let mut builder = GraphBuilder::<StdTensorOp>::new();
/// let x = builder.add_input(TensorInputKey::User { id: 1 });
/// let out = PrimitiveRuleBuilder::add_operation(
///     &mut builder,
///     StdTensorOp::Neg,
///     vec![ValueRef::Local(x)],
///     OperationRole::Primary,
/// );
/// assert_eq!(out.len(), 1);
/// ```
#[cfg(feature = "autodiff")]
pub trait PrimitiveRuleBuilder {
    /// Add one primitive graph operation and return local ids for its outputs.
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId>;
}

#[cfg(feature = "autodiff")]
impl<B> PrimitiveRuleBuilder for B
where
    B: PrimitiveBuilder<StdTensorOp> + ?Sized,
{
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        let inputs = inputs.into_iter().map(PrimitiveValue::from).collect();
        PrimitiveBuilder::add_primitive(self, operation, inputs, role)
    }
}

#[cfg(feature = "autodiff")]
impl PrimitiveRuleBuilder for GraphBuilder<StdTensorOp> {
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        GraphBuilder::add_operation(self, operation, inputs, role)
    }
}

/// Forward-mode AD (JVP) for `StdTensorOp`: given the primal op and its
/// tangent inputs, emit the linearized graph into `builder` and return
/// the output tangents.
///
/// Rules per op live in the category submodules (`semiring`, `analytic`,
/// `elementwise`, `structural`, `contraction`, `indexing`, `diagonal`,
/// `dynamic`). `StdTensorOp::Extension(_)` delegates to the trait.
///
/// # Errors
///
/// Returns [`ADRuleError::InvalidInput`] when the operation is not a known
/// primitive or when a registered rule rejects the graph metadata. Returns
/// [`ADRuleError::Unsupported`] when no AD rule is registered for the
/// operation. Errors returned by an extension rule are propagated unchanged.
#[cfg(feature = "autodiff")]
pub fn linearize(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut context::ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if let StdTensorOp::Extension(ext) = op {
        return dispatch_extension_linearize(
            ext.as_ref(),
            builder,
            primal_in,
            primal_out,
            tangent_in,
            ctx,
        );
    }

    let kind = op
        .primitive_kind()
        .ok_or_else(|| missing_primitive_kind(op, ADRuleKind::Jvp))?;
    let rule = registry::primitive_ad_rule(kind)
        .ok_or_else(|| registry::missing_rule(kind, ADRuleKind::Jvp))?;
    rule.linearize(op, builder, primal_in, primal_out, tangent_in, ctx)
}

/// Reverse-mode AD (VJP) for `StdTensorOp`: given the primal op, its
/// inputs, and the output cotangent, emit the transposed graph and
/// return the input cotangents.
///
/// See [`linearize`] for the category split; the same categories appear
/// here.
///
/// # Errors
///
/// Returns [`ADRuleError::InvalidInput`] when the operation, transpose inputs,
/// or graph metadata are inconsistent. Returns [`ADRuleError::Unsupported`]
/// when the required primitive or extension transpose rule is unavailable.
/// Errors returned by a registered rule are propagated unchanged.
#[cfg(feature = "autodiff")]
pub fn transpose_rule(
    op: &StdTensorOp,
    builder: &mut impl PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[PrimitiveTransposeInput<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut context::ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if let StdTensorOp::Extension(ext) = op {
        let builder_dyn: &mut dyn PrimitiveRuleBuilder = builder;
        return dispatch_extension_transpose(
            ext.as_ref(),
            builder_dyn,
            cotangent_out,
            inputs,
            mode,
            ctx,
        );
    }

    let transpose_inputs = inputs
        .iter()
        .map(transpose_input::TransposeInputRef::new)
        .collect::<Vec<_>>();
    let kind = op
        .primitive_kind()
        .ok_or_else(|| missing_primitive_kind(op, ADRuleKind::Transpose))?;
    let rule = registry::primitive_ad_rule(kind)
        .ok_or_else(|| registry::missing_rule(kind, ADRuleKind::Transpose))?;
    let builder_dyn: &mut dyn PrimitiveRuleBuilder = builder;
    rule.transpose_rule(op, builder_dyn, cotangent_out, &transpose_inputs, mode, ctx)
}

#[cfg(test)]
mod tests;
