use ::chainrules::{ReverseRule, Tape};
use chainrules_core::AutodiffError;

use crate::core::DynTensorTyped;
use crate::structured::StructuredTensor;
use crate::{DynTensor, NodeId, Result};

pub(crate) type PullbackRule<T> = Box<
    dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
        + Send
        + Sync
        + 'static,
>;

pub(crate) type MixedPullbackRule<TOut, TIn> = Box<
    dyn Fn(&StructuredTensor<TOut>) -> Result<Vec<(NodeId, StructuredTensor<TIn>)>>
        + Send
        + Sync
        + 'static,
>;

struct TensorRuleAdapter<T: DynTensorTyped> {
    rule: PullbackRule<T>,
}

struct MixedTensorRuleAdapter<TOut: DynTensorTyped, TIn: DynTensorTyped> {
    rule: MixedPullbackRule<TOut, TIn>,
}

impl<T: DynTensorTyped> ReverseRule<DynTensor> for TensorRuleAdapter<T> {
    fn pullback(
        &self,
        cotangent: &DynTensor,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DynTensor)>> {
        let cotangent = T::structured_ref(cotangent).ok_or_else(|| {
            AutodiffError::InvalidArgument("cotangent dtype did not match registered rule".into())
        })?;
        (self.rule)(cotangent)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|(node, grad)| (node, T::into_dyn(grad)))
                    .collect()
            })
            .map_err(|err| AutodiffError::InvalidArgument(err.to_string()))
    }

    fn pullback_with_tangents(
        &self,
        _cotangent: &DynTensor,
        _cotangent_tangent: &DynTensor,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DynTensor, DynTensor)>> {
        Err(AutodiffError::HvpNotSupported)
    }

    fn inputs(&self) -> Vec<NodeId> {
        Vec::new()
    }
}

impl<TOut: DynTensorTyped, TIn: DynTensorTyped> ReverseRule<DynTensor>
    for MixedTensorRuleAdapter<TOut, TIn>
{
    fn pullback(
        &self,
        cotangent: &DynTensor,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DynTensor)>> {
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

    fn pullback_with_tangents(
        &self,
        _cotangent: &DynTensor,
        _cotangent_tangent: &DynTensor,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DynTensor, DynTensor)>> {
        Err(AutodiffError::HvpNotSupported)
    }

    fn inputs(&self) -> Vec<NodeId> {
        Vec::new()
    }
}

pub(crate) fn register_rule<T: DynTensorTyped>(
    tape: &Tape<DynTensor>,
    node: NodeId,
    rule: PullbackRule<T>,
) {
    tape.attach_rule(node, Box::new(TensorRuleAdapter { rule }))
        .unwrap_or_else(|err| {
            unreachable!("reverse output node should exist before rule registration: {err}")
        });
}

pub(crate) fn register_mixed_rule<TOut: DynTensorTyped, TIn: DynTensorTyped>(
    tape: &Tape<DynTensor>,
    node: NodeId,
    rule: MixedPullbackRule<TOut, TIn>,
) {
    tape.attach_rule(node, Box::new(MixedTensorRuleAdapter { rule }))
        .unwrap_or_else(|err| {
            unreachable!("reverse output node should exist before rule registration: {err}")
        });
}

#[cfg(test)]
mod tests;
