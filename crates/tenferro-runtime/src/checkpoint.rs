use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::types::ValueKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{AllocationGroup, DType, DescriptorSlot, Tensor, TensorRead, TensorValue};

use crate::error::{Error, ErrorPhase, Result};

/// A read-only retained tensor handle backed by one allocation group owner.
///
/// Cloning the handle retains the group container and descriptor metadata; it
/// never clones a physical allocation. Creating another physical value remains
/// an explicit tensor operation at the caller's boundary.
#[derive(Clone)]
pub struct RetainedValue {
    container: Arc<RetentionContainer>,
    slot: DescriptorSlot,
    dtype: DType,
    shape: Box<[usize]>,
}

#[derive(Debug)]
struct RetentionContainer {
    group: AllocationGroup,
}

impl fmt::Debug for RetainedValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedValue")
            .field("slot", &self.slot)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl RetainedValue {
    /// Move an owned tensor value into a retained group-backed handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the value's descriptor cannot be
    /// transferred into a retention allocation group.
    pub fn from_tensor_value(value: TensorValue) -> Result<Self> {
        let (group, slot, dtype, shape) = value.try_into_group_parts().map_err(|_| {
            Error::runtime_state(
                "RetainedValue::from_tensor_value",
                ErrorPhase::Execution,
                "a TensorValue could not be transferred into its retention group",
            )
        })?;
        Ok(Self {
            container: Arc::new(RetentionContainer { group }),
            slot,
            dtype,
            shape: shape.into_boxed_slice(),
        })
    }

    /// Move an owned compact tensor into a retained group-backed handle.
    ///
    /// # Panics
    ///
    /// Panics only if the validated compact tensor cannot produce its valid
    /// allocation-group descriptor, which indicates an internal invariant
    /// violation.
    pub fn from_tensor(tensor: Tensor) -> Self {
        // A compact tensor always has a valid descriptor, so this conversion
        // cannot fail after the tensor owner has been constructed.
        Self::from_tensor_value(TensorValue::from_tensor(tensor))
            .expect("compact tensor retention must have a valid descriptor")
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the retained descriptor for one read-only execution boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained descriptor is vacant,
    /// out of bounds, or otherwise invalid in its allocation group.
    pub fn tensor_read(&self) -> Result<TensorRead<'_>> {
        self.container.group.read_view(self.slot).map_err(|error| {
            Error::runtime_state(
                "RetainedValue::tensor_read",
                ErrorPhase::Execution,
                error.to_string(),
            )
        })
    }
}

pub type RetainedInputMap = HashMap<TensorInputKey, Arc<RetainedValue>>;

#[derive(Clone)]
pub struct CheckpointNode {
    pub graph: Arc<Graph<StdTensorOp>>,
    pub alias_key: TensorInputKey,
    pub alias_target: ValueKey<StdTensorOp>,
    pub old_inputs: Arc<RetainedInputMap>,
    pub prev: Option<Arc<CheckpointNode>>,
}

impl fmt::Debug for CheckpointNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckpointNode")
            .field("alias_key", &self.alias_key)
            .field("alias_target", &self.alias_target)
            .field("old_inputs_len", &self.old_inputs.len())
            .field("has_prev", &self.prev.is_some())
            .finish_non_exhaustive()
    }
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

    pub fn collect_inputs(&self) -> RetainedInputMap {
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
                        old_inputs: Arc::clone(&node.old_inputs),
                        prev,
                    }));
                }
                prev
            }
        }
    }
}
