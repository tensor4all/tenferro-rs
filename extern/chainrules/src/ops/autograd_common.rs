use std::sync::{Arc, Mutex};

use crate::{engine::AutogradGraph, engine::Variable, AdResult, AutodiffError, Differentiable};

pub(crate) fn context_id<V: Differentiable>(ctx: &Arc<Mutex<AutogradGraph<V>>>) -> AdResult<u64> {
    ctx.lock()
        .map(|guard| guard.id())
        .map_err(|_| AutodiffError::InvalidArgument("autograd graph lock is poisoned".to_string()))
}

pub(crate) fn merge_context_for_multi_op<V: Differentiable>(
    inputs: &[&Variable<V>],
) -> AdResult<Option<Arc<Mutex<AutogradGraph<V>>>>> {
    if inputs.iter().all(|input| !input.requires_grad()) {
        return Ok(None);
    }

    let mut picked: Option<(u64, Arc<Mutex<AutogradGraph<V>>>)> = None;
    for ctx in inputs
        .iter()
        .filter(|input| input.requires_grad())
        .filter_map(|input| input.context.as_ref())
    {
        let id = context_id(ctx)?;
        match &picked {
            None => picked = Some((id, Arc::clone(ctx))),
            Some((picked_id, _)) if *picked_id == id => {}
            Some(_) => {
                return Err(AutodiffError::InvalidArgument(
                    "mixed autograd graphs in one operation; use Variable::new_in(..., same_graph)"
                        .to_string(),
                ));
            }
        }
    }

    let Some((picked_id, picked_ctx)) = picked else {
        return Ok(None);
    };

    let any_tracked_on_picked = inputs.iter().any(|input| {
        input.requires_grad() && input.context_id() == Some(picked_id) && input.node_id.is_some()
    });
    if any_tracked_on_picked {
        Ok(Some(picked_ctx))
    } else {
        Ok(None)
    }
}

pub(crate) fn merge_context_for_binary_op<V: Differentiable>(
    lhs: &Variable<V>,
    rhs: &Variable<V>,
) -> AdResult<Option<Arc<Mutex<AutogradGraph<V>>>>> {
    merge_context_for_multi_op(&[lhs, rhs])
}
