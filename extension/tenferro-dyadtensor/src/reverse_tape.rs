use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use chainrules_scalarops::ScalarAd;
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

type PullbackRule<T> = Box<dyn Fn(&Tensor<T>) -> Result<Vec<(NodeId, Tensor<T>)>> + 'static>;
type BridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, Tensor<TIn>)>> + 'static>;
type ScalarBridgeRule<TOut, TIn> =
    Box<dyn Fn(&Tensor<TOut>) -> Result<Vec<(NodeId, TIn)>> + 'static>;
type ScalarMixedRule<TOut, TIn> = Box<dyn Fn(&TOut) -> Result<Vec<(NodeId, TIn)>> + 'static>;
type ScalarPullbackRule<T> = Box<dyn Fn(&T) -> Result<Vec<(NodeId, T)>> + 'static>;

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

struct TapeScalarBridgeRules<TOut: Scalar, TIn: ScalarAd> {
    rules: HashMap<NodeId, ScalarBridgeRule<TOut, TIn>>,
}

impl<TOut: Scalar, TIn: ScalarAd> TapeScalarBridgeRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeScalarMixedRules<TOut: ScalarAd, TIn: ScalarAd> {
    rules: HashMap<NodeId, ScalarMixedRule<TOut, TIn>>,
}

impl<TOut: ScalarAd, TIn: ScalarAd> TapeScalarMixedRules<TOut, TIn> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

struct TapeScalarRules<T: ScalarAd> {
    rules: HashMap<NodeId, ScalarPullbackRule<T>>,
}

impl<T: ScalarAd> TapeScalarRules<T> {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
}

type RuleRegistry = HashMap<(u64, TypeId), Box<dyn Any>>;
type BridgeRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarBridgeRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarMixedRegistry = HashMap<(u64, TypeId, TypeId), Box<dyn Any>>;
type ScalarRuleRegistry = HashMap<(u64, TypeId), Box<dyn Any>>;

/// Check whether an `InvalidAdTensor` error indicates that no reverse rules
/// are registered (tape-level or node-level). This matches on error message
/// prefixes — fragile if the messages in `pullback()` change. Kept as a
/// named helper so the coupling is explicit and easy to find.
fn is_no_tensor_rules_error(err: &Error) -> bool {
    matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.starts_with("no reverse rules registered for tape")
                || message.starts_with("no reverse rule registered for output node")
    )
}

/// Same as [`is_no_tensor_rules_error`] but for scalar-typed rules.
fn is_no_scalar_rules_error(err: &Error) -> bool {
    matches!(
        err,
        Error::InvalidAdScalar { message }
            if message.starts_with("no reverse scalar rules registered for tape")
                || message.starts_with("no reverse scalar rule registered for output node")
    )
}

thread_local! {
    static REVERSE_RULE_REGISTRY: RefCell<RuleRegistry> = RefCell::new(HashMap::new());
    static REVERSE_BRIDGE_REGISTRY: RefCell<BridgeRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_BRIDGE_REGISTRY: RefCell<ScalarBridgeRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_MIXED_REGISTRY: RefCell<ScalarMixedRegistry> = RefCell::new(HashMap::new());
    static REVERSE_SCALAR_RULE_REGISTRY: RefCell<ScalarRuleRegistry> = RefCell::new(HashMap::new());
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

pub(crate) fn register_scalar_bridge_rule<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarBridgeRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_SCALAR_BRIDGE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarBridgeRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeScalarBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar bridge registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_scalar_mixed_rule<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarMixedRule<TOut, TIn>,
) -> Result<()> {
    REVERSE_SCALAR_MIXED_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarMixedRules::<TOut, TIn>::new()));
        let typed = entry
            .downcast_mut::<TapeScalarMixedRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar mixed registry type mismatch".to_string(),
            })?;
        typed.rules.insert(node, rule);
        Ok(())
    })
}

pub(crate) fn register_scalar_rule<T: ScalarAd + 'static>(
    tape: TapeId,
    node: NodeId,
    rule: ScalarPullbackRule<T>,
) -> Result<()> {
    REVERSE_SCALAR_RULE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let key = (tape.0, TypeId::of::<T>());
        let entry = registry
            .entry(key)
            .or_insert_with(|| Box::new(TapeScalarRules::<T>::new()));
        let typed =
            entry
                .downcast_mut::<TapeScalarRules<T>>()
                .ok_or_else(|| Error::InvalidAdScalar {
                    message: "reverse scalar tape registry type mismatch".to_string(),
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

fn bridge_pullback_scalar<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
) -> Result<Vec<(NodeId, TIn)>> {
    REVERSE_SCALAR_BRIDGE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeScalarBridgeRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar bridge registry type mismatch".to_string(),
            })?;
        let Some(rule) = state.rules.get(&output_node) else {
            return Ok(Vec::new());
        };
        rule(cotangent)
    })
}

fn bridge_pullback_scalar_mixed<TOut: ScalarAd + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &TOut,
) -> Result<Vec<(NodeId, TIn)>> {
    REVERSE_SCALAR_MIXED_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<TOut>(), TypeId::of::<TIn>());
        let Some(state_any) = registry.get(&key) else {
            return Ok(Vec::new());
        };
        let state = state_any
            .downcast_ref::<TapeScalarMixedRules<TOut, TIn>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar mixed registry type mismatch".to_string(),
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

fn accumulate_scalar_into<T: ScalarAd>(totals: &mut HashMap<NodeId, T>, node: NodeId, delta: T) {
    totals
        .entry(node)
        .and_modify(|existing| *existing = *existing + delta)
        .or_insert(delta);
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
    // Limitation: when pullback returns a "no rules" error we fall back to
    // using the output cotangent as a seed. This means gradient paths that
    // were not recorded on the tape are silently treated as identity,
    // which may cause gradient path loss for unregistered nodes.
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

pub(crate) fn pullback_wrt_scalars<TOut: Scalar + 'static, TIn: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &Tensor<TOut>,
    wrt_nodes: &[Option<NodeId>],
) -> Result<Vec<Option<TIn>>> {
    let all_out_grads = match pullback::<TOut>(tape, output_node, cotangent) {
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

    // Supported scalar types for mixed-path gradient propagation.
    // If new scalar types are added to the crate (beyond f32, f64, Complex32,
    // Complex64), a corresponding `accumulate_tensor_scalar_mixed_path` call
    // must be added here to ensure gradients flow through the new type.
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

pub(crate) fn pullback_scalar<T: ScalarAd + 'static>(
    tape: TapeId,
    output_node: NodeId,
    cotangent: &T,
) -> Result<HashMap<NodeId, T>> {
    REVERSE_SCALAR_RULE_REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let key = (tape.0, TypeId::of::<T>());
        let state_any = registry.get(&key).ok_or_else(|| Error::InvalidAdScalar {
            message: format!("no reverse scalar rules registered for tape {}", tape.0),
        })?;
        let state = state_any
            .downcast_ref::<TapeScalarRules<T>>()
            .ok_or_else(|| Error::InvalidAdScalar {
                message: "reverse scalar tape registry type mismatch".to_string(),
            })?;

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
            totals
                .entry(node)
                .and_modify(|existing| *existing = *existing + delta)
                .or_insert(delta);

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

#[cfg(test)]
mod tests;
