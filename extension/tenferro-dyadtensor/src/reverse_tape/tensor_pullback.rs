use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

use super::registry::{bridge_pullback, is_no_tensor_rules_error, with_tensor_rules};

fn accumulate_into<T: Scalar>(
    totals: &mut HashMap<NodeId, Tensor<T>>,
    node: NodeId,
    delta: Tensor<T>,
) {
    if let Some(prev) = totals.remove(&node) {
        totals.insert(node, Tensor::<T>::accumulate_tangent(prev, &delta));
    } else {
        totals.insert(node, delta);
    }
}

pub(crate) fn pullback<T: Scalar + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    with_tensor_rules::<T, _>(tape, |state| {
        if !state.rules.contains_key(&output_node) {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "no reverse rule registered for output node {} on tape {}",
                    output_node.0, tape.0
                ),
            });
        }

        let mut totals: HashMap<NodeId, Tensor<T>> = HashMap::new();
        let mut worklist: Vec<(NodeId, Tensor<T>)> = vec![(output_node, cotangent.clone())];

        while let Some((node, delta)) = worklist.pop() {
            accumulate_into(&mut totals, node, delta.clone());

            if let Some(rule) = state.rules.get(&node) {
                let input_deltas = rule(&delta)?;
                for (in_node, in_delta) in input_deltas {
                    worklist.push((in_node, in_delta));
                }
            }
        }

        Ok(totals)
    })
}

pub(crate) fn pullback_wrt_mixed<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
    wrt_nodes: &[Option<NodeId>],
) -> Result<Vec<Option<Tensor<TIn>>>> {
    let all_out_grads = match pullback::<TOut>(tape, output_node, cotangent) {
        Ok(grads) => grads,
        Err(e) if is_no_tensor_rules_error(&e) => {
            let mut seed = HashMap::new();
            seed.insert(output_node, cotangent.clone());
            seed
        }
        Err(e) => return Err(e),
    };

    let mut seed_in: HashMap<NodeId, Tensor<TIn>> = HashMap::new();
    for (node, delta_out) in &all_out_grads {
        let bridged = bridge_pullback::<TOut, TIn>(tape, *node, delta_out)?;
        for (in_node, in_delta) in bridged {
            accumulate_into(&mut seed_in, in_node, in_delta);
        }
    }

    let mut all_in_grads: HashMap<NodeId, Tensor<TIn>> = HashMap::new();
    for (seed_node, seed_delta) in seed_in {
        let propagated = match pullback::<TIn>(tape, seed_node, &seed_delta) {
            Ok(grads) => grads,
            Err(e) if is_no_tensor_rules_error(&e) => {
                let mut seed = HashMap::new();
                seed.insert(seed_node, seed_delta.clone());
                seed
            }
            Err(e) => return Err(e),
        };
        for (node, grad) in propagated {
            accumulate_into(&mut all_in_grads, node, grad);
        }
    }

    Ok(wrt_nodes
        .iter()
        .map(|node| node.and_then(|n| all_in_grads.get(&n).cloned()))
        .collect())
}
