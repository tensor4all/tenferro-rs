use ::chainrules::{ReverseRule, Tape};
use chainrules_core::AutodiffError;
use tenferro_algebra::Scalar;

use crate::structured::StructuredTensor;
use crate::{NodeId, Result};

pub(crate) type PullbackRule<T> =
    Box<dyn Fn(&StructuredTensor<T>) -> Result<Vec<(NodeId, StructuredTensor<T>)>> + 'static>;

struct TensorRuleAdapter<T: Scalar + 'static> {
    rule: PullbackRule<T>,
}

impl<T: Scalar + 'static> ReverseRule<StructuredTensor<T>> for TensorRuleAdapter<T> {
    fn pullback(
        &self,
        cotangent: &StructuredTensor<T>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, StructuredTensor<T>)>> {
        (self.rule)(cotangent).map_err(|err| AutodiffError::InvalidArgument(err.to_string()))
    }

    fn pullback_with_tangents(
        &self,
        _cotangent: &StructuredTensor<T>,
        _cotangent_tangent: &StructuredTensor<T>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, StructuredTensor<T>, StructuredTensor<T>)>> {
        Err(AutodiffError::HvpNotSupported)
    }

    fn inputs(&self) -> Vec<NodeId> {
        Vec::new()
    }
}

pub(crate) fn register_rule<T: Scalar + 'static>(
    tape: &Tape<StructuredTensor<T>>,
    node: NodeId,
    rule: PullbackRule<T>,
) {
    tape.attach_rule(node, Box::new(TensorRuleAdapter { rule }))
        .unwrap_or_else(|err| {
            unreachable!("reverse output node should exist before rule registration: {err}")
        });
}
