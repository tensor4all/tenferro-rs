use std::any::TypeId;
use std::collections::HashMap;

use chainrules_scalarops::ScalarAd;
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

use super::registry::{
    bridge_pullback_scalar, bridge_pullback_scalar_mixed, is_no_scalar_rules_error,
    is_no_tensor_rules_error, with_scalar_rules,
};

fn accumulate_scalar_into<T: ScalarAd>(totals: &mut HashMap<NodeId, T>, node: NodeId, delta: T) {
    totals
        .entry(node)
        .and_modify(|existing| *existing = *existing + delta)
        .or_insert(delta);
}

fn collect_tensor_scalar_bridge_seeds<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    all_out_grads: &HashMap<NodeId, Tensor<TOut>>,
) -> Result<HashMap<NodeId, TIn>> {
    let mut seed_in: HashMap<NodeId, TIn> = HashMap::new();
    for (node, delta_out) in all_out_grads {
        let bridged = bridge_pullback_scalar::<TOut, TIn>(tape, *node, delta_out)?;
        for (in_node, in_delta) in bridged {
            accumulate_scalar_into(&mut seed_in, in_node, in_delta);
        }
    }
    Ok(seed_in)
}

fn propagate_scalar_seeds<T: ScalarAd + 'static>(
    tape: TapeId,
    seed_in: HashMap<NodeId, T>,
) -> Result<HashMap<NodeId, T>> {
    let mut all_in_grads: HashMap<NodeId, T> = HashMap::new();
    for (seed_node, seed_delta) in seed_in {
        let propagated = match pullback_scalar::<T>(tape, seed_node, &seed_delta) {
            Ok(grads) => grads,
            Err(e) if is_no_scalar_rules_error(&e) => {
                let mut seed = HashMap::new();
                seed.insert(seed_node, seed_delta);
                seed
            }
            Err(e) => return Err(e),
        };
        for (node, grad) in propagated {
            accumulate_scalar_into(&mut all_in_grads, node, grad);
        }
    }
    Ok(all_in_grads)
}

fn pullback_scalar_wrt_mixed<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &TOut,
) -> Result<HashMap<NodeId, TIn>> {
    let all_out_grads = match pullback_scalar::<TOut>(tape, output_node, cotangent) {
        Ok(grads) => grads,
        Err(e) if is_no_scalar_rules_error(&e) => {
            let mut seed = HashMap::new();
            seed.insert(output_node, *cotangent);
            seed
        }
        Err(e) => return Err(e),
    };

    let mut seed_in: HashMap<NodeId, TIn> = HashMap::new();
    for (node, delta_out) in &all_out_grads {
        let bridged = bridge_pullback_scalar_mixed::<TOut, TIn>(tape, *node, delta_out)?;
        for (in_node, in_delta) in bridged {
            accumulate_scalar_into(&mut seed_in, in_node, in_delta);
        }
    }

    propagate_scalar_seeds(tape, seed_in)
}

fn accumulate_tensor_scalar_mixed_path<
    TOut: Scalar + 'static,
    TSeed: ScalarAd + 'static,
    TIn: ScalarAd + 'static,
>(
    tape: TapeId,
    all_out_grads: &HashMap<NodeId, Tensor<TOut>>,
    all_in_grads: &mut HashMap<NodeId, TIn>,
) -> Result<()> {
    if TypeId::of::<TSeed>() == TypeId::of::<TIn>() {
        return Ok(());
    }

    let seed_in = collect_tensor_scalar_bridge_seeds::<TOut, TSeed>(tape, all_out_grads)?;
    for (seed_node, seed_delta) in seed_in {
        let propagated = pullback_scalar_wrt_mixed::<TSeed, TIn>(tape, seed_node, &seed_delta)?;
        for (node, grad) in propagated {
            accumulate_scalar_into(all_in_grads, node, grad);
        }
    }
    Ok(())
}

pub(crate) fn pullback_wrt_scalars<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
    wrt_nodes: &[Option<NodeId>],
) -> Result<Vec<Option<TIn>>> {
    let all_out_grads = match super::tensor_pullback::pullback::<TOut>(tape, output_node, cotangent)
    {
        Ok(grads) => grads,
        Err(e) if is_no_tensor_rules_error(&e) => {
            let mut seed = HashMap::new();
            seed.insert(output_node, cotangent.clone());
            seed
        }
        Err(e) => return Err(e),
    };

    let mut all_in_grads: HashMap<NodeId, TIn> = HashMap::new();
    let direct_seed_in = collect_tensor_scalar_bridge_seeds::<TOut, TIn>(tape, &all_out_grads)?;
    for (node, grad) in propagate_scalar_seeds(tape, direct_seed_in)? {
        accumulate_scalar_into(&mut all_in_grads, node, grad);
    }

    accumulate_tensor_scalar_mixed_path::<TOut, f32, TIn>(tape, &all_out_grads, &mut all_in_grads)?;
    accumulate_tensor_scalar_mixed_path::<TOut, f64, TIn>(tape, &all_out_grads, &mut all_in_grads)?;
    accumulate_tensor_scalar_mixed_path::<TOut, Complex32, TIn>(
        tape,
        &all_out_grads,
        &mut all_in_grads,
    )?;
    accumulate_tensor_scalar_mixed_path::<TOut, Complex64, TIn>(
        tape,
        &all_out_grads,
        &mut all_in_grads,
    )?;

    Ok(wrt_nodes
        .iter()
        .map(|node| node.and_then(|n| all_in_grads.get(&n).copied()))
        .collect())
}

pub(crate) fn pullback_scalar<T: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &T,
) -> Result<HashMap<NodeId, T>> {
    with_scalar_rules::<T, _>(tape, |state| {
        if !state.rules.contains_key(&output_node) {
            return Err(Error::InvalidAdScalar {
                message: format!(
                    "no reverse scalar rule registered for output node {} on tape {}",
                    output_node.0, tape.0
                ),
            });
        }

        let mut totals: HashMap<NodeId, T> = HashMap::new();
        let mut worklist: Vec<(NodeId, T)> = vec![(output_node, *cotangent)];

        while let Some((node, delta)) = worklist.pop() {
            accumulate_scalar_into(&mut totals, node, delta);

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
