use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::resolve::{ResolvedView, ValueDef};
use computegraph::types::{OperationKey, ValueKey, ValueRef};
use computegraph::{GraphOperation, LocalValueId, OperationRole};
use tenferro_ops::ad::transpose_rule;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tidu::{ADKey, ADRuleError, ADRuleKind, ADRuleResult, PrimitiveValue};

/// Reverse-mode graph built from primary transpose rules on the forward graph.
pub(super) struct PrimalTransposeGraph {
    graph: computegraph::graph::Graph<StdTensorOp>,
    cotangent_inputs: Vec<(TensorInputKey, LocalValueId)>,
    cotangent_outputs: Vec<Option<LocalValueId>>,
}

impl PrimalTransposeGraph {
    pub(super) fn as_graph(&self) -> &computegraph::graph::Graph<StdTensorOp> {
        &self.graph
    }

    pub(super) fn into_graph(self) -> computegraph::graph::Graph<StdTensorOp> {
        self.graph
    }

    pub(super) fn tangent_inputs(&self) -> &[(TensorInputKey, LocalValueId)] {
        &self.cotangent_inputs
    }

    pub(super) fn tangent_outputs(&self) -> &[Option<LocalValueId>] {
        &self.cotangent_outputs
    }
}

fn output_keys_for_operation(
    op_key: &Arc<OperationKey<StdTensorOp>>,
) -> Vec<ValueKey<StdTensorOp>> {
    let output_count = GraphOperation::output_count(op_key.operation());
    (0..output_count)
        .map(|slot| ValueKey::Derived {
            operation: Arc::clone(op_key),
            output_slot: slot as u8,
        })
        .collect()
}

fn forward_operation_order(
    view: &ResolvedView<StdTensorOp>,
    outputs: &[ValueKey<StdTensorOp>],
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
) -> Vec<Arc<OperationKey<StdTensorOp>>> {
    fn visit_key(
        key: &ValueKey<StdTensorOp>,
        view: &ResolvedView<StdTensorOp>,
        aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
        visited: &mut HashSet<ValueKey<StdTensorOp>>,
        order: &mut Vec<ValueKey<StdTensorOp>>,
    ) {
        if !visited.insert(key.clone()) {
            return;
        }

        match view.resolve_value(key) {
            Some(ValueDef::Produced { input_keys, .. }) => {
                for input_key in input_keys {
                    visit_key(&input_key, view, aliases, visited, order);
                }
            }
            Some(ValueDef::Input { key: input_key }) => {
                if let Some(aliased_key) = aliases.get(&input_key) {
                    visit_key(aliased_key, view, aliases, visited, order);
                }
            }
            None => {}
        }

        order.push(key.clone());
    }

    let mut visited = HashSet::new();
    let mut key_order = Vec::new();
    for output_key in outputs {
        visit_key(output_key, view, aliases, &mut visited, &mut key_order);
    }

    let mut seen_ops = HashSet::new();
    let mut op_order = Vec::new();
    for key in key_order {
        let ValueKey::Derived { operation, .. } = key else {
            continue;
        };
        if seen_ops.insert(operation.fingerprint()) {
            op_order.push(operation);
        }
    }
    op_order
}

fn cotangent_seed_key(
    wrt_keys: &[TensorInputKey],
    index: usize,
    pass_id: u64,
) -> ADRuleResult<TensorInputKey> {
    let base = wrt_keys.get(index).ok_or_else(|| {
        ADRuleError::invalid_input(
            "tenferro-ad.primal_transpose",
            ADRuleKind::Transpose,
            format!(
                "cannot derive cotangent seed {index} from {} wrt inputs",
                wrt_keys.len()
            ),
        )
    })?;
    let index = u64::try_from(index).map_err(|_| {
        ADRuleError::invalid_input(
            "tenferro-ad.primal_transpose",
            ADRuleKind::Transpose,
            "cotangent seed index does not fit in a diff pass id",
        )
    })?;
    let seed_pass = pass_id.checked_add(index).ok_or_else(|| {
        ADRuleError::invalid_input(
            "tenferro-ad.primal_transpose",
            ADRuleKind::Transpose,
            "cotangent seed diff pass id overflowed",
        )
    })?;
    Ok(base.tangent_of(seed_pass))
}

fn value_depends_on_wrt(
    key: &ValueKey<StdTensorOp>,
    view: &ResolvedView<StdTensorOp>,
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    wrt_value_keys: &HashSet<ValueKey<StdTensorOp>>,
    memo: &mut HashMap<ValueKey<StdTensorOp>, bool>,
) -> bool {
    if let Some(depends) = memo.get(key) {
        return *depends;
    }
    if wrt_value_keys.contains(key) {
        memo.insert(key.clone(), true);
        return true;
    }

    let depends = match view.resolve_value(key) {
        Some(ValueDef::Produced { input_keys, .. }) => input_keys
            .into_iter()
            .any(|input_key| value_depends_on_wrt(&input_key, view, aliases, wrt_value_keys, memo)),
        Some(ValueDef::Input { key: input_key }) => {
            aliases.get(&input_key).is_some_and(|aliased_key| {
                value_depends_on_wrt(aliased_key, view, aliases, wrt_value_keys, memo)
            })
        }
        None => false,
    };
    memo.insert(key.clone(), depends);
    depends
}

fn transpose_active_mask(
    op_key: &Arc<OperationKey<StdTensorOp>>,
    view: &ResolvedView<StdTensorOp>,
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    wrt_value_keys: &HashSet<ValueKey<StdTensorOp>>,
    memo: &mut HashMap<ValueKey<StdTensorOp>, bool>,
) -> Vec<bool> {
    op_key
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, input_key)| {
            let depends = value_depends_on_wrt(input_key, view, aliases, wrt_value_keys, memo);
            match op_key.role() {
                OperationRole::Primary => depends,
                OperationRole::Linearized { active_mask } => {
                    depends && active_mask.get(index).copied().unwrap_or(false)
                }
            }
        })
        .collect()
}

fn unsupported_primal_transpose(op: &StdTensorOp) -> ADRuleError {
    ADRuleError::unsupported(
        format!("primal transpose could not fully transpose {op:?}"),
        ADRuleKind::Transpose,
    )
}

/// Transpose a primal computation graph for reverse-mode AD (VJP).
///
/// Unlike [`tidu::try_linear_transpose`], this walks the forward graph and
/// applies each operation's primary transpose rule, allowing extension rules
/// such as `Eigh` to reuse forward eigenvectors.
pub(super) fn try_primal_transpose(
    view: &ResolvedView<StdTensorOp>,
    output_keys: &[ValueKey<StdTensorOp>],
    wrt_keys: &[TensorInputKey],
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    ctx: &mut ShapeGuardContext,
    pass_id: u64,
) -> ADRuleResult<PrimalTransposeGraph> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut cotangent_env: HashMap<ValueKey<StdTensorOp>, LocalValueId> = HashMap::new();
    let mut cotangent_seed_inputs = Vec::new();
    let wrt_value_keys: Vec<ValueKey<StdTensorOp>> =
        wrt_keys.iter().cloned().map(ValueKey::Input).collect();
    let wrt_value_key_set: HashSet<ValueKey<StdTensorOp>> =
        wrt_value_keys.iter().cloned().collect();
    let mut dependency_memo = HashMap::new();

    for (index, output_key) in output_keys.iter().enumerate() {
        let seed_key = cotangent_seed_key(wrt_keys, index, pass_id)?;
        let seed_id = builder.add_input(seed_key.clone());
        cotangent_env.insert(output_key.clone(), seed_id);
        cotangent_seed_inputs.push((seed_key, seed_id));
    }

    let op_order = forward_operation_order(view, output_keys, aliases);
    for op_key in op_order.iter().rev() {
        let output_keys = output_keys_for_operation(op_key);
        let cotangent_out: Vec<Option<LocalValueId>> = output_keys
            .iter()
            .map(|key| cotangent_env.get(key).copied())
            .collect();
        if cotangent_out.iter().all(Option::is_none) {
            continue;
        }

        let rule_inputs: Vec<PrimitiveValue<StdTensorOp>> = op_key
            .inputs()
            .iter()
            .map(|key| PrimitiveValue::External(key.clone()))
            .collect();
        let inputs: Vec<ValueRef<StdTensorOp>> = rule_inputs
            .iter()
            .map(|value| match value {
                PrimitiveValue::External(key) => ValueRef::External(key.clone()),
                PrimitiveValue::Local(_) => {
                    unreachable!("primal transpose rule inputs are external refs")
                }
            })
            .collect();

        let active_mask = transpose_active_mask(
            op_key,
            view,
            aliases,
            &wrt_value_key_set,
            &mut dependency_memo,
        );
        let transpose_mode = if matches!(op_key.operation(), StdTensorOp::Extension(_)) {
            OperationRole::Primary
        } else {
            OperationRole::Linearized {
                active_mask: active_mask.clone(),
            }
        };

        ctx.set_transpose_primal_outputs(Some(output_keys.clone()));
        let cotangent_in = transpose_rule(
            op_key.operation(),
            &mut builder,
            &cotangent_out,
            &inputs,
            &transpose_mode,
            ctx,
        );
        ctx.set_transpose_primal_outputs(None);
        let cotangent_in = cotangent_in?;

        if cotangent_in.len() != rule_inputs.len() {
            return Err(ADRuleError::invalid_input(
                "tenferro-ad.primal_transpose",
                ADRuleKind::Transpose,
                format!(
                    "transpose_rule for {:?} returned {} cotangents for {} inputs",
                    op_key.operation(),
                    cotangent_in.len(),
                    rule_inputs.len()
                ),
            ));
        }
        if active_mask.iter().any(|active| *active)
            && !cotangent_in
                .iter()
                .zip(active_mask.iter())
                .any(|(cotangent, active)| *active && cotangent.is_some())
        {
            return Err(unsupported_primal_transpose(op_key.operation()));
        }

        for (input, maybe_cotangent) in rule_inputs.iter().zip(cotangent_in) {
            let Some(cotangent_id) = maybe_cotangent else {
                continue;
            };
            let PrimitiveValue::External(input_key) = input else {
                continue;
            };

            match cotangent_env.get(input_key).copied() {
                Some(existing_id) => {
                    let sum = builder.add_operation(
                        StdTensorOp::Add,
                        vec![ValueRef::Local(existing_id), ValueRef::Local(cotangent_id)],
                        OperationRole::Linearized {
                            active_mask: vec![true, true],
                        },
                    );
                    cotangent_env.insert(input_key.clone(), sum[0]);
                }
                None => {
                    cotangent_env.insert(input_key.clone(), cotangent_id);
                }
            }
        }
    }

    let tangent_outputs: Vec<Option<LocalValueId>> = wrt_value_keys
        .iter()
        .map(|key| cotangent_env.get(key).copied())
        .collect();
    let active_outputs: Vec<LocalValueId> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    Ok(PrimalTransposeGraph {
        graph: builder.build(),
        cotangent_inputs: cotangent_seed_inputs,
        cotangent_outputs: tangent_outputs,
    })
}
