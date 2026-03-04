use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

type PullbackRule<T> = Box<dyn Fn(&Tensor<T>) -> Result<Vec<(NodeId, Tensor<T>)>> + 'static>;
type BridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, Tensor<TIn>)>> + 'static>;

struct TapeRules<T: Scalar> {
    rules: HashMap<NodeId, PullbackRule<T>>,
}

impl<T: Scalar> TapeRules<T> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeBridgeRules<TOut: Scalar, TIn: Scalar> {
    rules: HashMap<NodeId, BridgeRule<TOut, TIn>>,
}

impl<TOut: Scalar, TIn: Scalar> TapeBridgeRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

type RuleRegistry = HashMap<(u64, TypeId), Box<dyn Any>>;
type BridgeRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;

thread_local! {
    static REVERSE_RULE_REGISTRY: RefCell<RuleRegistry> = RefCell::new(HashMap::new());
    static REVERSE_BRIDGE_REGISTRY: RefCell<BridgeRegistry> = RefCell::new(HashMap::new());
}

pub(crate) fn register_rule<T: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: PullbackRule<T>,
) -> Result<()> {
    REVERSE_RULE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<T>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeRules::<T>::new()));
        let typed = entry
            .downcast_mut::<TapeRules<T>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_bridge_rule<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: BridgeRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_BRIDGE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeBridgeRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape bridge registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

fn bridge_pullback<TOut: Scalar + 'static, TIn: Scalar + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
) -> Result<Vec<(NodeId, Tensor<TIn>)>> {
    REVERSE_BRIDGE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "reverse tape bridge registry type mismatch".to_string(),
            })?;
        let Some(rule) = state.rules.get(&output_node) else {
            return Ok(Vec::new());
        };
        rule(cotangent)
    })
}

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
    REVERSE_RULE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<T>());
        let state_any = registry.get(&key).ok_or_else(|| Error::InvalidAdTensor {
            message: format!("no reverse rules registered for tape {}", tape.0),
        })?;
        let state =
            state_any
                .downcast_ref::<TapeRules<T>>()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "reverse tape registry type mismatch".to_string(),
                })?;

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
            if let Some(prev) = totals.remove(&node) {
                totals.insert(node, Tensor::<T>::accumulate_tangent(prev, &delta));
            } else {
                totals.insert(node, delta.clone());
            }

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
        Err(Error::InvalidAdTensor { message })
            if message.starts_with("no reverse rules registered for tape")
                || message.starts_with("no reverse rule registered for output node") =>
        {
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
            Err(Error::InvalidAdTensor { message })
                if message.starts_with("no reverse rules registered for tape")
                    || message.starts_with("no reverse rule registered for output node") =>
            {
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
