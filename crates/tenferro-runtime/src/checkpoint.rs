use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::types::ValueKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::Tensor;

#[derive(Clone)]
pub struct CheckpointNode {
    pub graph: Arc<Graph<StdTensorOp>>,
    pub alias_key: TensorInputKey,
    pub alias_target: ValueKey<StdTensorOp>,
    pub old_inputs: HashMap<TensorInputKey, Arc<Tensor>>,
    pub prev: Option<Arc<CheckpointNode>>,
}

impl CheckpointNode {
    pub fn collect_aliases(&self) -> HashMap<TensorInputKey, ValueKey<StdTensorOp>> {
        let mut aliases = HashMap::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            aliases.insert(node.alias_key.clone(), node.alias_target.clone());
            current = node.prev.as_deref();
        }
        aliases
    }

    pub fn collect_graphs(&self) -> Vec<Arc<Graph<StdTensorOp>>> {
        let mut graphs = Vec::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            graphs.push(node.graph.clone());
            current = node.prev.as_deref();
        }
        graphs
    }

    pub fn collect_inputs(&self) -> HashMap<TensorInputKey, Arc<Tensor>> {
        let mut inputs = HashMap::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            inputs.extend(
                node.old_inputs
                    .iter()
                    .map(|(key, tensor)| (key.clone(), tensor.clone())),
            );
            current = node.prev.as_deref();
        }
        inputs
    }

    /// Merge two checkpoint chains into a single linked list.
    ///
    /// The lhs chain is reconstructed on top of the rhs chain so that
    /// `collect_aliases`, `collect_graphs`, and `collect_inputs`
    /// traverse both sides during the AD pass.
    pub(crate) fn merge_chains(
        lhs: Option<Arc<CheckpointNode>>,
        rhs: Option<Arc<CheckpointNode>>,
    ) -> Option<Arc<CheckpointNode>> {
        match (lhs, rhs) {
            (None, rhs) => rhs,
            (lhs, None) => lhs,
            (Some(lhs_head), Some(rhs_head)) => {
                let mut nodes: Vec<&CheckpointNode> = Vec::new();
                let mut current: Option<&CheckpointNode> = Some(&lhs_head);
                while let Some(node) = current {
                    nodes.push(node);
                    current = node.prev.as_deref();
                }
                let mut prev: Option<Arc<CheckpointNode>> = Some(rhs_head);
                for node in nodes.into_iter().rev() {
                    prev = Some(Arc::new(CheckpointNode {
                        graph: node.graph.clone(),
                        alias_key: node.alias_key.clone(),
                        alias_target: node.alias_target.clone(),
                        old_inputs: node.old_inputs.clone(),
                        prev,
                    }));
                }
                prev
            }
        }
    }
}
