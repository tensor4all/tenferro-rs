use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::{AdValueRecord, EagerTensor, Error, Result, StdTensorOp, TracedTensor};
use crate::semantic_extension::ResidualSpec;

#[derive(Clone, Debug)]
pub(crate) struct EagerTrace(Arc<ResidualNode>);

struct ResidualNode {
    parents: Vec<EagerTrace>,
    saved: OnceLock<Vec<SavedValue>>,
}

impl std::fmt::Debug for ResidualNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualNode")
            .field("parents", &self.parents.len())
            .field("saved", &self.saved.get().map(Vec::len))
            .finish()
    }
}

#[derive(Clone)]
pub(super) struct SavedValue {
    pub(super) trace: TracedTensor,
    pub(super) value: Arc<AdValueRecord>,
}

impl EagerTrace {
    pub(super) fn new(inputs: &[&EagerTensor]) -> Self {
        Self(Arc::new(ResidualNode {
            parents: inputs
                .iter()
                .filter_map(|input| input.trace.clone())
                .collect(),
            saved: OnceLock::new(),
        }))
    }

    pub(super) fn collect(&self) -> Vec<SavedValue> {
        let mut stack = vec![self];
        let mut visited = HashSet::new();
        let mut keys = HashSet::new();
        let mut values = Vec::new();
        while let Some(trace) = stack.pop() {
            if !visited.insert(Arc::as_ptr(&trace.0)) {
                continue;
            }
            if let Some(saved) = trace.0.saved.get() {
                for value in saved {
                    let key = value.trace.graph().values()[value.trace.val].key.clone();
                    if keys.insert(key) {
                        values.push(value.clone());
                    }
                }
            }
            stack.extend(trace.0.parents.iter().rev());
        }
        values
    }
}

// Share only declared tensor values, never EagerTensor records (which own the
// runtime and gradient slots). This also keeps residuals alive after callers
// drop unused decomposition output handles, without copying their buffers.
pub(crate) fn finish_residuals(
    op: &StdTensorOp,
    inputs: &[&EagerTensor],
    outputs: &[&EagerTensor],
) -> Result<()> {
    let Some(first) = outputs.first() else {
        return Ok(());
    };
    let Some(trace) = &first.trace else {
        return Ok(());
    };
    let mask = match op {
        StdTensorOp::Extension(op) => {
            let rules = &first.ctx.semantic_extension_rules;
            if let Some(rule) = rules.lookup_primal_vjp(op.family_id()) {
                rule.residual_mask()
            } else if let Some(rule) = rules.lookup_linear_transpose(op.family_id()) {
                rule.residual_mask()
            } else {
                ResidualSpec::none()
            }
        }
        _ => crate::semantic_transform::eager_core_residual_spec(op),
    };
    let mut saved = Vec::new();
    for (index, tensor) in inputs.iter().enumerate().chain(
        outputs
            .iter()
            .enumerate()
            .map(|(i, t)| (inputs.len() + i, t)),
    ) {
        let declared = if index < inputs.len() {
            mask.declares_input(index)
        } else {
            mask.declares_output(index - inputs.len())
        };
        if !declared {
            continue;
        }
        let Some(semantic) = &tensor.semantic_trace else {
            continue;
        };
        // Leaf values are already bound by the semantic source.
        if semantic.input_key().is_some() {
            continue;
        }
        saved.push(SavedValue {
            trace: semantic.clone(),
            value: Arc::clone(&tensor._record.value),
        });
    }
    trace
        .0
        .saved
        .set(saved)
        .map_err(|_| Error::Internal("eager residual node was already initialized".into()))
}
