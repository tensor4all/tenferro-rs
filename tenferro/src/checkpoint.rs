use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::Fragment;
use computegraph::types::GlobalValKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::Tensor;

#[derive(Clone)]
pub(crate) struct CheckpointNode {
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub alias_key: TensorInputKey,
    pub alias_target: GlobalValKey<StdTensorOp>,
    pub old_inputs: HashMap<TensorInputKey, Arc<Tensor>>,
    pub prev: Option<Arc<CheckpointNode>>,
}

impl CheckpointNode {
    pub(crate) fn collect_aliases(&self) -> HashMap<TensorInputKey, GlobalValKey<StdTensorOp>> {
        let mut aliases = HashMap::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            aliases.insert(node.alias_key.clone(), node.alias_target.clone());
            current = node.prev.as_deref();
        }
        aliases
    }

    pub(crate) fn collect_fragments(&self) -> Vec<Arc<Fragment<StdTensorOp>>> {
        let mut fragments = Vec::new();
        let mut current: Option<&CheckpointNode> = Some(self);
        while let Some(node) = current {
            fragments.push(node.fragment.clone());
            current = node.prev.as_deref();
        }
        fragments
    }

    pub(crate) fn collect_inputs(&self) -> HashMap<TensorInputKey, Arc<Tensor>> {
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
}
