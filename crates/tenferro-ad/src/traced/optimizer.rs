use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;

use crate::transform_cache::CachedOptimizedLinearGraph;

pub(super) struct OptimizedLinearGraph {
    graph: Graph<StdTensorOp>,
    tangent_inputs: Vec<(TensorInputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
}

impl OptimizedLinearGraph {
    pub(super) fn from_tidu(linear: tidu::LinearizedGraph<StdTensorOp>) -> Self {
        let tangent_inputs = linear.tangent_inputs().to_vec();
        let tangent_outputs = linear.tangent_outputs().to_vec();
        optimize_graph(linear.into_graph(), tangent_inputs, tangent_outputs)
    }

    pub(super) fn into_cached(self) -> CachedOptimizedLinearGraph {
        CachedOptimizedLinearGraph::new(self.graph, self.tangent_inputs, self.tangent_outputs)
    }

    #[cfg(test)]
    pub(super) fn tangent_outputs(&self) -> &[Option<LocalValueId>] {
        &self.tangent_outputs
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarConst {
    Zero,
    One,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalFact {
    Neg(LocalValueId),
    Conj(LocalValueId),
    ScalarConst(ScalarConst),
}

fn optimize_graph(
    graph: Graph<StdTensorOp>,
    tangent_inputs: Vec<(TensorInputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
) -> OptimizedLinearGraph {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    for parent in graph.parents() {
        builder.add_parent(Arc::clone(parent));
    }

    let mut remap: Vec<Option<ValueRef<StdTensorOp>>> = vec![None; graph.values().len()];
    let mut facts: HashMap<LocalValueId, LocalFact> = HashMap::new();

    for &input in graph.inputs() {
        let ValueKey::Input(key) = &graph.values()[input].key else {
            continue;
        };
        let new_input = builder.add_input(key.clone());
        remap[input] = Some(ValueRef::Local(new_input));
    }

    for op_node in graph.operations() {
        let inputs: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| remap_value(input, &remap))
            .collect();

        if op_node.outputs.len() == 1 {
            if let Some(alias) =
                canonical_alias(&op_node.operation, &inputs, op_node.role.clone(), &facts)
            {
                remap[op_node.outputs[0]] = Some(alias);
                continue;
            }
        }

        let outputs = builder.add_operation(
            op_node.operation.clone(),
            inputs.clone(),
            op_node.role.clone(),
        );
        for (&old_output, &new_output) in op_node.outputs.iter().zip(outputs.iter()) {
            remap[old_output] = Some(ValueRef::Local(new_output));
        }
        if op_node.outputs.len() == 1 {
            if let Some(fact) = local_fact(&op_node.operation, &inputs) {
                facts.insert(outputs[0], fact);
            }
        }
    }

    let tangent_inputs = tangent_inputs
        .into_iter()
        .filter_map(|(key, old_id)| remap_local(old_id, &remap).map(|new_id| (key, new_id)))
        .collect();
    let tangent_outputs: Vec<_> = tangent_outputs
        .into_iter()
        .map(|maybe_id| maybe_id.and_then(|old_id| remap_local(old_id, &remap)))
        .collect();
    let active_outputs: Vec<_> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    prune_unreachable_graph(builder.build(), tangent_inputs, tangent_outputs)
}

fn prune_unreachable_graph(
    graph: Graph<StdTensorOp>,
    tangent_inputs: Vec<(TensorInputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
) -> OptimizedLinearGraph {
    let mut needed_values = HashSet::new();
    let mut needed_ops = HashSet::new();
    let mut stack: Vec<_> = tangent_outputs.iter().filter_map(|id| *id).collect();

    while let Some(value_id) = stack.pop() {
        if !needed_values.insert(value_id) {
            continue;
        }
        let Some((op_id, _slot)) = graph.values()[value_id].producer else {
            continue;
        };
        if needed_ops.insert(op_id) {
            for input in &graph.operations()[op_id].inputs {
                if let ValueRef::Local(input_id) = input {
                    stack.push(*input_id);
                }
            }
        }
    }
    let live_output_masks = live_output_masks(&graph, &needed_values, &needed_ops);

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    for parent in graph.parents() {
        builder.add_parent(Arc::clone(parent));
    }

    let mut remap: Vec<Option<ValueRef<StdTensorOp>>> = vec![None; graph.values().len()];
    for &input in graph.inputs() {
        let ValueKey::Input(key) = &graph.values()[input].key else {
            continue;
        };
        let new_input = builder.add_input(key.clone());
        remap[input] = Some(ValueRef::Local(new_input));
    }

    for (op_id, op_node) in graph.operations().iter().enumerate() {
        if !needed_ops.contains(&op_id) {
            continue;
        }
        let inputs: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| remap_value(input, &remap))
            .collect();
        if let Some((operation, kept_slots)) =
            pruned_operation_outputs(&op_node.operation, &live_output_masks[op_id])
        {
            let outputs = builder.add_operation(operation, inputs, op_node.role.clone());
            for (new_slot, old_slot) in kept_slots.into_iter().enumerate() {
                remap[op_node.outputs[old_slot]] = Some(ValueRef::Local(outputs[new_slot]));
            }
        } else {
            let outputs =
                builder.add_operation(op_node.operation.clone(), inputs, op_node.role.clone());
            for (&old_output, &new_output) in op_node.outputs.iter().zip(outputs.iter()) {
                remap[old_output] = Some(ValueRef::Local(new_output));
            }
        }
    }

    let tangent_inputs = tangent_inputs
        .into_iter()
        .filter_map(|(key, old_id)| remap_local(old_id, &remap).map(|new_id| (key, new_id)))
        .collect();
    let tangent_outputs: Vec<_> = tangent_outputs
        .into_iter()
        .map(|maybe_id| maybe_id.and_then(|old_id| remap_local(old_id, &remap)))
        .collect();
    let active_outputs: Vec<_> = tangent_outputs.iter().filter_map(|id| *id).collect();
    if !active_outputs.is_empty() {
        builder.set_outputs(active_outputs);
    }

    OptimizedLinearGraph {
        graph: builder.build(),
        tangent_inputs,
        tangent_outputs,
    }
}

fn live_output_masks(
    graph: &Graph<StdTensorOp>,
    needed_values: &HashSet<LocalValueId>,
    needed_ops: &HashSet<usize>,
) -> Vec<Vec<bool>> {
    let mut masks: Vec<Vec<bool>> = graph
        .operations()
        .iter()
        .map(|op| vec![false; op.outputs.len()])
        .collect();
    for &value_id in needed_values {
        let Some((op_id, output_slot)) = graph.values()[value_id].producer else {
            continue;
        };
        if needed_ops.contains(&op_id) {
            masks[op_id][output_slot] = true;
        }
    }
    masks
}

fn pruned_operation_outputs(
    operation: &StdTensorOp,
    live_outputs: &[bool],
) -> Option<(StdTensorOp, Vec<usize>)> {
    let kept_slots: Vec<_> = live_outputs
        .iter()
        .enumerate()
        .filter_map(|(slot, live)| live.then_some(slot))
        .collect();
    if kept_slots.is_empty() || kept_slots.len() == live_outputs.len() {
        return None;
    }

    let pruned = prune_operation_outputs(operation, live_outputs)?;
    match &pruned {
        StdTensorOp::Extension(op) if op.output_count() == kept_slots.len() => {
            Some((pruned, kept_slots))
        }
        _ => None,
    }
}

fn prune_operation_outputs(operation: &StdTensorOp, live_outputs: &[bool]) -> Option<StdTensorOp> {
    match operation {
        StdTensorOp::Extension(op) => op.prune_outputs(live_outputs).map(StdTensorOp::Extension),
        _ => None,
    }
}

fn remap_value(
    input: &ValueRef<StdTensorOp>,
    remap: &[Option<ValueRef<StdTensorOp>>],
) -> ValueRef<StdTensorOp> {
    match input {
        ValueRef::Local(local_id) => remap
            .get(*local_id)
            .and_then(Clone::clone)
            .unwrap_or(ValueRef::Local(*local_id)),
        ValueRef::External(key) => ValueRef::External(key.clone()),
    }
}

fn remap_local(
    old_id: LocalValueId,
    remap: &[Option<ValueRef<StdTensorOp>>],
) -> Option<LocalValueId> {
    match remap.get(old_id).and_then(Clone::clone)? {
        ValueRef::Local(local_id) => Some(local_id),
        ValueRef::External(_) => None,
    }
}

fn canonical_alias(
    operation: &StdTensorOp,
    inputs: &[ValueRef<StdTensorOp>],
    role: OperationRole,
    facts: &HashMap<LocalValueId, LocalFact>,
) -> Option<ValueRef<StdTensorOp>> {
    match (operation, inputs) {
        (StdTensorOp::Neg, [ValueRef::Local(input)]) => match facts.get(input).copied() {
            Some(LocalFact::Neg(inner)) => Some(ValueRef::Local(inner)),
            _ => None,
        },
        (StdTensorOp::Conj, [ValueRef::Local(input)]) => match facts.get(input).copied() {
            Some(LocalFact::Conj(inner)) => Some(ValueRef::Local(inner)),
            _ => None,
        },
        (StdTensorOp::Convert { from, to }, [input]) if from == to => Some(input.clone()),
        (StdTensorOp::Transpose { perm }, [input]) if is_identity_perm(perm) => Some(input.clone()),
        (StdTensorOp::Add, [lhs, rhs]) => {
            match (scalar_const(lhs, facts), scalar_const(rhs, facts)) {
                (Some(ScalarConst::Zero), _) if input_can_alias(&role, 1) => Some(rhs.clone()),
                (_, Some(ScalarConst::Zero)) if input_can_alias(&role, 0) => Some(lhs.clone()),
                _ => None,
            }
        }
        (StdTensorOp::Mul, [lhs, rhs]) => {
            match (scalar_const(lhs, facts), scalar_const(rhs, facts)) {
                (Some(ScalarConst::One), _) if input_can_alias(&role, 1) => Some(rhs.clone()),
                (_, Some(ScalarConst::One)) if input_can_alias(&role, 0) => Some(lhs.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn local_fact(operation: &StdTensorOp, inputs: &[ValueRef<StdTensorOp>]) -> Option<LocalFact> {
    match (operation, inputs) {
        (StdTensorOp::Neg, [ValueRef::Local(input)]) => Some(LocalFact::Neg(*input)),
        (StdTensorOp::Conj, [ValueRef::Local(input)]) => Some(LocalFact::Conj(*input)),
        (StdTensorOp::Constant { dtype, bytes }, _) => {
            scalar_const_bytes(*dtype, bytes).map(LocalFact::ScalarConst)
        }
        _ => None,
    }
}

fn scalar_const(
    input: &ValueRef<StdTensorOp>,
    facts: &HashMap<LocalValueId, LocalFact>,
) -> Option<ScalarConst> {
    match input {
        ValueRef::Local(local_id) => match facts.get(local_id).copied() {
            Some(LocalFact::ScalarConst(value)) => Some(value),
            _ => None,
        },
        ValueRef::External(_) => None,
    }
}

fn scalar_const_bytes(dtype: DType, bytes: &[u8]) -> Option<ScalarConst> {
    if bytes == zero_bytes(dtype).as_slice() {
        Some(ScalarConst::Zero)
    } else if bytes == one_bytes(dtype).as_slice() {
        Some(ScalarConst::One)
    } else {
        None
    }
}

fn zero_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 0.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 0.0_f64.to_le_bytes().to_vec(),
        DType::I32 => 0_i32.to_le_bytes().to_vec(),
        DType::I64 => 0_i64.to_le_bytes().to_vec(),
        DType::Bool => vec![0],
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    }
}

fn one_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 1.0_f64.to_le_bytes().to_vec(),
        DType::I32 => 1_i32.to_le_bytes().to_vec(),
        DType::I64 => 1_i64.to_le_bytes().to_vec(),
        DType::Bool => vec![1],
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&1.0_f64.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    }
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter()
        .enumerate()
        .all(|(index, &value)| index == value)
}

fn input_can_alias(role: &OperationRole, input_index: usize) -> bool {
    match role {
        OperationRole::Primary => true,
        OperationRole::Linearized { active_mask } => {
            active_mask.get(input_index).copied().unwrap_or(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::*;
    use tenferro_ops::ext_op::ExtensionOp;
    use tenferro_ops::SymDim;

    #[derive(Clone, Debug)]
    struct PrunableTestOp {
        kept_outputs: Vec<usize>,
    }

    impl ExtensionOp for PrunableTestOp {
        fn family_id(&self) -> &'static str {
            "tenferro-tests.prunable-output.v1"
        }

        fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
            hasher.write_usize(self.kept_outputs.len());
            for output in &self.kept_outputs {
                hasher.write_usize(*output);
            }
        }

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .is_some_and(|that| that.kept_outputs == self.kept_outputs)
        }

        fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
            Arc::new(self.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn input_count(&self) -> usize {
            1
        }

        fn output_count(&self) -> usize {
            self.kept_outputs.len()
        }

        fn infer_output_meta(
            &self,
            input_dtypes: &[DType],
            input_shapes: &[&[SymDim]],
        ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
            Ok(self
                .kept_outputs
                .iter()
                .map(|_| (input_dtypes[0], input_shapes[0].to_vec()))
                .collect())
        }

        fn prune_outputs(&self, live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
            let kept_outputs = live_outputs
                .iter()
                .enumerate()
                .filter_map(|(slot, live)| live.then_some(self.kept_outputs[slot]))
                .collect::<Vec<_>>();
            Some(Arc::new(Self { kept_outputs }))
        }
    }

    fn scalar_constant(dtype: DType, value: f64) -> StdTensorOp {
        match dtype {
            DType::F64 => StdTensorOp::Constant {
                dtype,
                bytes: value.to_le_bytes().to_vec(),
            },
            _ => unreachable!("test helper only needs f64"),
        }
    }

    #[test]
    fn optimizer_canonicalizes_ad_identity_chains() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let x = builder.add_input(TensorInputKey::User { id: 1 });

        let neg = builder.add_operation(
            StdTensorOp::Neg,
            vec![ValueRef::Local(x)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];
        let double_neg = builder.add_operation(
            StdTensorOp::Neg,
            vec![ValueRef::Local(neg)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];

        let conj = builder.add_operation(
            StdTensorOp::Conj,
            vec![ValueRef::Local(double_neg)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];
        let double_conj = builder.add_operation(
            StdTensorOp::Conj,
            vec![ValueRef::Local(conj)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];

        let zero = builder.add_operation(
            scalar_constant(DType::F64, 0.0),
            vec![],
            OperationRole::Primary,
        )[0];
        let added_zero = builder.add_operation(
            StdTensorOp::Add,
            vec![ValueRef::Local(double_conj), ValueRef::Local(zero)],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        )[0];

        let one = builder.add_operation(
            scalar_constant(DType::F64, 1.0),
            vec![],
            OperationRole::Primary,
        )[0];
        let mul_one = builder.add_operation(
            StdTensorOp::Mul,
            vec![ValueRef::Local(added_zero), ValueRef::Local(one)],
            OperationRole::Linearized {
                active_mask: vec![true, false],
            },
        )[0];

        builder.set_outputs(vec![mul_one]);
        let graph = builder.build();
        let optimized = optimize_graph(
            graph,
            vec![(TensorInputKey::User { id: 1 }, x)],
            vec![Some(mul_one)],
        );

        let operations = optimized.graph.operations();
        assert!(
            operations.is_empty(),
            "all identity ops should fold away, got {} ops",
            operations.len()
        );
        assert_eq!(optimized.tangent_outputs(), &[Some(0)]);
    }

    #[test]
    fn optimizer_prunes_unused_multi_output_slots_with_extension_hook() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let x = builder.add_input(TensorInputKey::User { id: 7 });
        let outputs = builder.add_operation(
            StdTensorOp::Extension(Arc::new(PrunableTestOp {
                kept_outputs: vec![0, 1, 2],
            })),
            vec![ValueRef::Local(x)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        );
        builder.set_outputs(vec![outputs[1]]);

        let optimized = optimize_graph(
            builder.build(),
            vec![(TensorInputKey::User { id: 7 }, x)],
            vec![Some(outputs[1])],
        );

        let operations = optimized.graph.operations();
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].outputs.len(), 1);
        let StdTensorOp::Extension(op) = &operations[0].operation else {
            panic!("expected pruned extension op");
        };
        let pruned = op.as_any().downcast_ref::<PrunableTestOp>().unwrap();
        assert_eq!(pruned.kept_outputs, vec![1]);
        assert_eq!(
            optimized.tangent_outputs(),
            &[Some(operations[0].outputs[0])]
        );
    }
}
