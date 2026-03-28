use ::tidu::{ReverseRule, Tape};
use chainrules_core::{AdResult, AutodiffError, NodeId};

use crate::core::DynTensorTyped;
use crate::structured::StructuredTensor;
use crate::{DynTensor, Result};

/// VJP-only closure adapter that implements `ReverseRule<DynTensor>` directly.
///
/// HVP methods return `HvpNotSupported`. The closure receives the full
/// `StructuredTensor<T>` cotangent (preserving structured layout).
///
/// # Examples
///
/// ```ignore
/// let rule = ClosureRuleAdapter::<f64> {
///     pullback_fn: Box::new(|cotangent| Ok(vec![(node, structured)])),
///     input_node_ids: vec![node],
/// };
/// ```
struct ClosureRuleAdapter<T: DynTensorTyped> {
    pullback_fn: Box<
        dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
            + Send
            + Sync
            + 'static,
    >,
    input_node_ids: Vec<NodeId>,
}

impl<T: DynTensorTyped> ReverseRule<DynTensor> for ClosureRuleAdapter<T> {
    fn pullback(&self, cotangent: &DynTensor) -> AdResult<Vec<(NodeId, DynTensor)>> {
        let cotangent = T::structured_ref(cotangent).ok_or_else(|| {
            AutodiffError::InvalidArgument("cotangent dtype did not match registered rule".into())
        })?;
        (self.pullback_fn)(cotangent)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|(node, grad)| (node, T::into_dyn(grad)))
                    .collect()
            })
            .map_err(|err| AutodiffError::InvalidArgument(err.to_string()))
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.clone()
    }
}

struct TensorRuleAdapter<T: DynTensorTyped> {
    rule: Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>,
}

struct MixedTensorRuleAdapter<TOut: DynTensorTyped, TIn: DynTensorTyped> {
    rule: Box<
        dyn Fn(&StructuredTensor<TOut>) -> Result<Vec<(NodeId, StructuredTensor<TIn>)>>
            + Send
            + Sync
            + 'static,
    >,
    input_node_ids: Vec<NodeId>,
}

impl<T: DynTensorTyped> ReverseRule<DynTensor> for TensorRuleAdapter<T> {
    fn pullback(&self, cotangent: &DynTensor) -> AdResult<Vec<(NodeId, DynTensor)>> {
        let cotangent = T::structured_ref(cotangent).ok_or_else(|| {
            AutodiffError::InvalidArgument("cotangent dtype did not match registered rule".into())
        })?;
        let results = self.rule.pullback(cotangent.payload())?;
        Ok(results
            .into_iter()
            .map(|(node, grad)| (node, T::into_dyn(grad.into())))
            .collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.rule.inputs()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DynTensor>,
    ) -> AdResult<Option<DynTensor>>
    where
        DynTensor: 't,
    {
        let typed_fn = |node: NodeId| -> Option<&'t tenferro_tensor::Tensor<T>> {
            input_tangents(node)
                .and_then(T::structured_ref)
                .map(|s| s.payload())
        };
        match self.rule.forward_tangents(&typed_fn)? {
            Some(t) => Ok(Some(T::into_dyn(t.into()))),
            None => Ok(None),
        }
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &DynTensor,
        cotangent_tangent: &DynTensor,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DynTensor>,
    ) -> AdResult<Vec<(NodeId, DynTensor, DynTensor)>>
    where
        DynTensor: 't,
    {
        let cotangent_typed = T::structured_ref(cotangent)
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "cotangent dtype did not match registered rule".into(),
                )
            })?
            .payload();
        let cotangent_tangent_typed = T::structured_ref(cotangent_tangent)
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "cotangent_tangent dtype did not match registered rule".into(),
                )
            })?
            .payload();
        let typed_fn = |node: NodeId| -> Option<&'t tenferro_tensor::Tensor<T>> {
            input_tangents(node)
                .and_then(T::structured_ref)
                .map(|s| s.payload())
        };
        let results = self.rule.pullback_with_tangents(
            cotangent_typed,
            cotangent_tangent_typed,
            &typed_fn,
        )?;
        Ok(results
            .into_iter()
            .map(|(node, g, gt)| {
                (
                    node,
                    T::into_dyn(StructuredTensor::from(g)),
                    T::into_dyn(StructuredTensor::from(gt)),
                )
            })
            .collect())
    }
}

impl<TOut: DynTensorTyped, TIn: DynTensorTyped> ReverseRule<DynTensor>
    for MixedTensorRuleAdapter<TOut, TIn>
{
    fn pullback(&self, cotangent: &DynTensor) -> AdResult<Vec<(NodeId, DynTensor)>> {
        let cotangent = TOut::structured_ref(cotangent).ok_or_else(|| {
            AutodiffError::InvalidArgument("cotangent dtype did not match registered rule".into())
        })?;
        (self.rule)(cotangent)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|(node, grad)| (node, TIn::into_dyn(grad)))
                    .collect()
            })
            .map_err(|err| AutodiffError::InvalidArgument(err.to_string()))
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.clone()
    }
}

pub(crate) fn register_rule<T: DynTensorTyped>(
    tape: &Tape<DynTensor>,
    node: NodeId,
    rule: Box<dyn ReverseRule<tenferro_tensor::Tensor<T>>>,
) {
    tape.attach_rule(node, Box::new(TensorRuleAdapter { rule }))
        .unwrap_or_else(|err| {
            unreachable!("reverse output node should exist before rule registration: {err}")
        });
}

pub(crate) fn register_closure_rule<T: DynTensorTyped>(
    tape: &Tape<DynTensor>,
    node: NodeId,
    input_node_ids: Vec<NodeId>,
    rule: Box<
        dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
            + Send
            + Sync
            + 'static,
    >,
) {
    tape.attach_rule(
        node,
        Box::new(ClosureRuleAdapter {
            pullback_fn: rule,
            input_node_ids,
        }),
    )
    .unwrap_or_else(|err| {
        unreachable!("reverse output node should exist before rule registration: {err}")
    });
}

pub(crate) fn register_mixed_rule<TOut: DynTensorTyped, TIn: DynTensorTyped>(
    tape: &Tape<DynTensor>,
    node: NodeId,
    input_node_ids: Vec<NodeId>,
    rule: Box<
        dyn Fn(&StructuredTensor<TOut>) -> Result<Vec<(NodeId, StructuredTensor<TIn>)>>
            + Send
            + Sync
            + 'static,
    >,
) {
    tape.attach_rule(
        node,
        Box::new(MixedTensorRuleAdapter {
            rule,
            input_node_ids,
        }),
    )
    .unwrap_or_else(|err| {
        unreachable!("reverse output node should exist before rule registration: {err}")
    });
}

#[cfg(test)]
mod tests;
