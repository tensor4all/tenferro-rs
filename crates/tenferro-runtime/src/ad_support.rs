//! Internal support surface used by `tenferro-ad`.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::types::{LocalValueId, ValueKey};
pub use tenferro_ops::ad::context::GlobalMetadataScope;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, Tensor, TypedTensor};

pub use crate::checkpoint::CheckpointNode;
use crate::metadata::MetadataScopeChain;
pub use crate::metadata::{
    metadata_scopes_for_scope, metadata_scopes_with_new, metadata_scopes_with_scope,
    push_metadata_scope, register_scoped_graph_analysis, register_scoped_graph_metadata,
    register_scoped_live_graph_metadata, register_scoped_metadata_batch,
    register_scoped_value_metadata, registered_meta, tensor_meta_from_tensor,
    RegisteredGraphAnalysis,
};
use crate::shape_constraint::ConstraintScopeChain;
pub use crate::shape_constraint::ShapeConstraintScope;
use crate::sym_dim::SymDim;
use crate::traced::{next_input_key, next_traced_id, TracedTensor};
use crate::{Error, Result};

/// Opaque, persistent shape-constraint history transferred across AD graphs.
///
/// Cloning a transfer is constant-time. Combining transfers shares their
/// existing histories and defers pointer de-duplication to one materialization
/// walk when the compiler needs the scopes.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ad_support::ConstraintScopeTransfer;
/// use tenferro_runtime::{DType, TracedTensor};
///
/// let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
/// let transfer = ConstraintScopeTransfer::from_tensor(&input);
/// assert!(transfer.is_empty());
/// ```
#[derive(Clone)]
pub struct ConstraintScopeTransfer {
    chain: ConstraintScopeChain,
}

impl ConstraintScopeTransfer {
    /// Create an empty transfer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::ConstraintScopeTransfer;
    ///
    /// assert!(ConstraintScopeTransfer::empty().is_empty());
    /// ```
    pub fn empty() -> Self {
        Self {
            chain: ConstraintScopeChain::empty(),
        }
    }

    /// Borrow a traced tensor's constraint history through an opaque transfer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::ConstraintScopeTransfer;
    /// use tenferro_runtime::{DType, TracedTensor};
    ///
    /// let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// assert!(ConstraintScopeTransfer::from_tensor(&input).is_empty());
    /// ```
    pub fn from_tensor(tensor: &TracedTensor) -> Self {
        Self {
            chain: tensor.constraint_scopes.clone(),
        }
    }

    /// Add one analyzed graph scope above inherited persistent histories.
    ///
    /// Empty analyzed scopes are not retained. Existing histories remain
    /// shared, including when the same parent is inherited more than once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::{
    ///     register_scoped_graph_analysis, ConstraintScopeTransfer,
    /// };
    /// use tenferro_runtime::{DType, TracedTensor};
    ///
    /// let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    /// let parent = ConstraintScopeTransfer::from_tensor(&input);
    /// let analysis = register_scoped_graph_analysis(input.graph(), []).unwrap();
    /// let transfer = ConstraintScopeTransfer::with_new(analysis.constraints, [&parent]);
    /// assert!(transfer.is_empty());
    /// ```
    pub fn with_new<'a>(
        scope: ShapeConstraintScope,
        inherited: impl IntoIterator<Item = &'a ConstraintScopeTransfer>,
    ) -> Self {
        let parents: Vec<_> = inherited
            .into_iter()
            .map(|transfer| &transfer.chain)
            .collect();
        let chain = if scope.is_empty() {
            ConstraintScopeChain::merge(parents)
        } else {
            ConstraintScopeChain::with_scope(Arc::new(scope), parents)
        };
        Self { chain }
    }

    /// Merge inherited persistent histories without adding a graph scope.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::ConstraintScopeTransfer;
    ///
    /// let parent = ConstraintScopeTransfer::empty();
    /// let merged = ConstraintScopeTransfer::merge([&parent, &parent]);
    /// assert!(merged.is_empty());
    /// ```
    pub fn merge<'a>(inherited: impl IntoIterator<Item = &'a ConstraintScopeTransfer>) -> Self {
        Self {
            chain: ConstraintScopeChain::merge(
                inherited.into_iter().map(|transfer| &transfer.chain),
            ),
        }
    }

    /// Return whether the transferred history contains no constraint scopes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::ad_support::ConstraintScopeTransfer;
    ///
    /// assert!(ConstraintScopeTransfer::empty().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.chain.as_slice().is_empty()
    }

    #[cfg(test)]
    fn test_scope_and_node_counts(&self) -> (usize, usize) {
        let (scopes, visited_nodes) = self.chain.materialize_with_visit_count();
        (scopes.len(), visited_nodes)
    }
}

impl fmt::Debug for ConstraintScopeTransfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstraintScopeTransfer")
            .finish_non_exhaustive()
    }
}

/// Parts required to construct a traced tensor from an AD transform.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use computegraph::graph::GraphBuilder;
/// use tenferro_runtime::ad_support::{
///     allocate_input_key, tensor_from_parts, ConstraintScopeTransfer, TracedTensorParts,
/// };
/// use tenferro_runtime::{DType, SymDim};
///
/// let key = allocate_input_key();
/// let mut builder = GraphBuilder::new();
/// let value = builder.add_input(key);
/// builder.set_outputs(vec![value]);
/// let tensor = tensor_from_parts(TracedTensorParts {
///     rank: 1,
///     dtype: DType::F64,
///     graph: Arc::new(builder.build()),
///     val: value,
///     data: None,
///     shape_hint: Some(vec![SymDim::from(2)]),
///     inputs_map: Arc::new(HashMap::new()),
///     extra_roots: Vec::new(),
///     checkpoint_chain: None,
///     metadata_scopes: Vec::new(),
///     constraint_scope_transfer: ConstraintScopeTransfer::empty(),
/// });
/// assert_eq!(tensor.rank, 1);
/// ```
pub struct TracedTensorParts {
    pub rank: usize,
    pub dtype: DType,
    pub graph: Arc<Graph<StdTensorOp>>,
    pub val: LocalValueId,
    pub data: Option<Arc<Tensor>>,
    pub shape_hint: Option<Vec<SymDim>>,
    pub inputs_map: Arc<HashMap<TensorInputKey, Arc<Tensor>>>,
    pub extra_roots: Vec<Arc<Graph<StdTensorOp>>>,
    pub checkpoint_chain: Option<Arc<CheckpointNode>>,
    pub metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    pub constraint_scope_transfer: ConstraintScopeTransfer,
}

impl fmt::Debug for TracedTensorParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracedTensorParts")
            .field("rank", &self.rank)
            .field("dtype", &self.dtype)
            .field("val", &self.val)
            .field("has_data", &self.data.is_some())
            .field("shape_hint", &self.shape_hint)
            .field("inputs_len", &self.inputs_map.len())
            .field("extra_roots_len", &self.extra_roots.len())
            .field("has_checkpoint_chain", &self.checkpoint_chain.is_some())
            .field("metadata_scopes_len", &self.metadata_scopes.len())
            .field("constraint_scope_transfer", &self.constraint_scope_transfer)
            .finish_non_exhaustive()
    }
}

/// Builds a traced tensor from validated AD transform output.
///
/// See [`TracedTensorParts`] for a complete runnable example.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use computegraph::graph::GraphBuilder;
/// use tenferro_runtime::ad_support::{
///     allocate_input_key, tensor_from_parts, ConstraintScopeTransfer, TracedTensorParts,
/// };
/// use tenferro_runtime::{DType, SymDim};
///
/// let key = allocate_input_key();
/// let mut builder = GraphBuilder::new();
/// let value = builder.add_input(key);
/// builder.set_outputs(vec![value]);
/// let tensor = tensor_from_parts(TracedTensorParts {
///     rank: 1,
///     dtype: DType::F64,
///     graph: Arc::new(builder.build()),
///     val: value,
///     data: None,
///     shape_hint: Some(vec![SymDim::from(1)]),
///     inputs_map: Arc::new(HashMap::new()),
///     extra_roots: Vec::new(),
///     checkpoint_chain: None,
///     metadata_scopes: Vec::new(),
///     constraint_scope_transfer: ConstraintScopeTransfer::empty(),
/// });
/// assert_eq!(tensor.dtype, DType::F64);
/// ```
pub fn tensor_from_parts(parts: TracedTensorParts) -> TracedTensor {
    TracedTensor {
        id: next_traced_id(),
        rank: parts.rank,
        dtype: parts.dtype,
        graph: parts.graph,
        val: parts.val,
        data: parts.data,
        shape_hint: parts.shape_hint,
        inputs_map: parts.inputs_map,
        extra_roots: parts.extra_roots,
        checkpoint_chain: parts.checkpoint_chain,
        metadata_scopes: MetadataScopeChain::from_materialized(parts.metadata_scopes),
        constraint_scopes: parts.constraint_scope_transfer.chain,
    }
}

pub fn shape_hint(tensor: &TracedTensor) -> Option<Vec<SymDim>> {
    tensor.shape_hint.clone()
}

pub fn inputs_map(tensor: &TracedTensor) -> Arc<HashMap<TensorInputKey, Arc<Tensor>>> {
    Arc::clone(&tensor.inputs_map)
}

pub fn extra_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>> {
    tensor.extra_roots.clone()
}

pub fn checkpoint_chain(tensor: &TracedTensor) -> Option<Arc<CheckpointNode>> {
    tensor.checkpoint_chain.clone()
}

pub fn metadata_scopes(tensor: &TracedTensor) -> &[Arc<GlobalMetadataScope>] {
    tensor.metadata_scopes.as_slice()
}

pub fn resolve_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>> {
    tensor.resolve_roots()
}

pub fn checkpoint_tensor(tensor: &mut TracedTensor, data: Arc<Tensor>) -> Result<()> {
    let old_graph = tensor.graph.clone();
    let old_output_key = old_graph.values()[tensor.val].key.clone();
    let old_inputs = Arc::clone(&tensor.inputs_map);
    let concrete_meta = tensor_meta_from_tensor(data.as_ref());
    let new_key = next_input_key();
    let mut builder = computegraph::graph::GraphBuilder::new();
    let leaf_val = builder.add_input(new_key.clone());
    builder.set_outputs(vec![leaf_val]);
    let new_graph = Arc::new(builder.build());
    let new_metadata_scope = register_scoped_value_metadata(
        new_graph.values()[leaf_val].key.clone(),
        concrete_meta.clone(),
    )?;
    let old_output_metadata_scope =
        register_scoped_value_metadata(old_output_key.clone(), concrete_meta)?;
    let node = CheckpointNode {
        graph: old_graph,
        alias_key: new_key.clone(),
        alias_target: old_output_key,
        old_inputs,
        prev: tensor.checkpoint_chain.take(),
    };
    tensor.graph = new_graph;
    tensor.val = leaf_val;
    tensor.extra_roots.clear();
    tensor.data = Some(Arc::clone(&data));
    tensor.shape_hint = Some(data.shape().iter().copied().map(SymDim::from).collect());
    tensor.checkpoint_chain = Some(Arc::new(node));
    let mut metadata_scopes = tensor.metadata_scopes.materialize();
    push_metadata_scope(&mut metadata_scopes, Arc::new(new_metadata_scope));
    push_metadata_scope(&mut metadata_scopes, Arc::new(old_output_metadata_scope));
    tensor.metadata_scopes = MetadataScopeChain::from_materialized(metadata_scopes);

    let mut merged = HashMap::new();
    if let Some(chain) = &tensor.checkpoint_chain {
        merged.extend(chain.collect_inputs());
    }
    merged.insert(new_key, data);
    tensor.inputs_map = Arc::new(merged);
    Ok(())
}

pub fn allocate_input_key() -> TensorInputKey {
    next_input_key()
}

pub fn leaf_input_key(tensor: &TracedTensor) -> Result<TensorInputKey> {
    match &tensor.graph.values()[tensor.val].key {
        ValueKey::Input(key) => Ok(key.clone()),
        other => Err(Error::InvalidGraphBuild {
            op: "ad_support::leaf_input_key",
            message: format!("expected traced leaf input, got {other:?}"),
        }),
    }
}

pub fn linear_input_key(
    graph: &Graph<StdTensorOp>,
    local_id: LocalValueId,
) -> Result<TensorInputKey> {
    match &graph.values()[local_id].key {
        ValueKey::Input(key) => Ok(key.clone()),
        other => Err(Error::InvalidGraphBuild {
            op: "ad_support::linear_input_key",
            message: format!("expected linear graph input, got {other:?}"),
        }),
    }
}

pub fn ones_tensor(dtype: DType, shape: Vec<usize>) -> Result<Tensor> {
    match dtype {
        DType::F32 => Ok(Tensor::F32(TypedTensor::ones(shape)?)),
        DType::F64 => Ok(Tensor::F64(TypedTensor::ones(shape)?)),
        DType::I32 => Ok(Tensor::I32(TypedTensor::ones(shape)?)),
        DType::I64 => Ok(Tensor::I64(TypedTensor::ones(shape)?)),
        DType::Bool => {
            let len =
                tenferro_tensor::validate::checked_shape_product("ones_tensor", "shape", &shape)?;
            Ok(Tensor::Bool(TypedTensor::from_vec_col_major(
                shape,
                vec![true; len],
            )?))
        }
        DType::C32 => Ok(Tensor::C32(TypedTensor::ones(shape)?)),
        DType::C64 => Ok(Tensor::C64(TypedTensor::ones(shape)?)),
    }
}

#[cfg(test)]
mod tests;
