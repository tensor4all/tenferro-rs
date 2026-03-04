use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, NodeId, Result, TapeId};

type PullbackRule<T> = Box<dyn Fn(&Tensor<T>) -> Result<Vec<(NodeId, Tensor<T>)>> + 'static>;

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

thread_local! {
    static REVERSE_RULE_REGISTRY: RefCell<HashMap<(u64, TypeId), Box<dyn Any>>> =
        RefCell::new(HashMap::new());
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
