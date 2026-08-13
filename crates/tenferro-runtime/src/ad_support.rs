//! Internal support surface used by `tenferro-ad`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::types::{LocalValueId, ValueKey};
pub use tenferro_ops::ad::context::GlobalMetadataScope;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, Tensor, TypedTensor};

pub use crate::checkpoint::{CheckpointNode, RetainedValue};
use crate::error::ErrorPhase;
use crate::metadata::MetadataScopeChain;
use tenferro_ops::ad::context::TensorMeta;

use crate::metadata::concrete_tensor_meta;
pub use crate::metadata::{
    metadata_scopes_for_scope, metadata_scopes_with_new, metadata_scopes_with_scope,
    push_metadata_scope, register_scoped_graph_analysis, register_scoped_graph_metadata,
    register_scoped_live_graph_metadata, register_scoped_metadata_batch,
    register_scoped_value_metadata, registered_meta, tensor_meta_from_tensor,
    RegisteredGraphAnalysis,
};
use crate::program::FrozenProgram;
use crate::shape_constraint::ConstraintScopeChain;
pub use crate::shape_constraint::ShapeConstraintScope;
use crate::sym_dim::SymDim;
use crate::traced::{next_input_key, next_traced_id, TracedTensor};
use crate::{CompiledGraph, Error, GraphCompiler, Result};

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
///     leaf_metas: Arc::new(HashMap::new()),
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
    pub data: Option<Arc<RetainedValue>>,
    pub shape_hint: Option<Vec<SymDim>>,
    pub inputs_map: Arc<HashMap<TensorInputKey, Arc<RetainedValue>>>,
    /// Retained construction-time metadata per bound leaf input key; see
    /// [`TracedTensor`]'s `leaf_metas` field.
    pub leaf_metas: Arc<HashMap<TensorInputKey, TensorMeta>>,
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
            .field("leaf_metas_len", &self.leaf_metas.len())
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
///     leaf_metas: Arc::new(HashMap::new()),
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
        leaf_metas: parts.leaf_metas,
        extra_roots: parts.extra_roots,
        checkpoint_chain: parts.checkpoint_chain,
        metadata_scopes: MetadataScopeChain::from_materialized(parts.metadata_scopes),
        constraint_scopes: parts.constraint_scope_transfer.chain,
    }
}

pub fn shape_hint(tensor: &TracedTensor) -> Option<Vec<SymDim>> {
    tensor.shape_hint.clone()
}

pub fn inputs_map(tensor: &TracedTensor) -> Arc<HashMap<TensorInputKey, Arc<RetainedValue>>> {
    Arc::clone(&tensor.inputs_map)
}

/// Return the retained construction-time leaf metadata map of a traced tensor.
#[doc(hidden)]
pub fn leaf_metas(tensor: &TracedTensor) -> Arc<HashMap<TensorInputKey, TensorMeta>> {
    Arc::clone(&tensor.leaf_metas)
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

/// Merge the input-value maps of several traced tensors into one bindings map.
///
/// The eager-AD raw carrier needs the same merged leaf bindings (`inputs_map`)
/// that [`crate::extension::apply`] attaches to analyzed traces, so the
/// deferred `compile_ad_source` can bind concrete values without re-walking.
#[doc(hidden)]
pub fn merge_traced_inputs_map<'a>(
    inputs: impl IntoIterator<Item = &'a TracedTensor>,
) -> Arc<HashMap<TensorInputKey, Arc<RetainedValue>>> {
    crate::traced::merge_traced_inputs_map(inputs)
}

/// Merge the retained construction-time leaf metadata maps of several traced
/// tensors into one map, covering the same leaf keys as the merged bindings.
#[doc(hidden)]
pub fn merge_traced_leaf_metas<'a>(
    inputs: impl IntoIterator<Item = &'a TracedTensor>,
) -> Arc<HashMap<TensorInputKey, TensorMeta>> {
    crate::traced::merge_traced_leaf_metas(inputs)
}

/// Run the deferred graph analysis over an eagerly appended raw semantic trace
/// once, at the first AD request, and return the analyzed twin.
///
/// The eager forward records ops via [`crate::extension::append_raw_op`]
/// without running `infer_output_meta` / metadata registration / constraint
/// inference. This runs that analysis pass over the whole parent graph chain
/// (idempotently, post-order so parents are registered before dependents) and
/// attaches the resulting metadata and constraint scopes, so downstream
/// `compile_ad_source` sees the same scoped graph the eager forward used to
/// build.
///
/// Leaf metadata is not read from the global registry: it is seeded from the
/// raw carrier's own bindings (`inputs_map`) and retained leaf metadata
/// (`leaf_metas`, the construction-time `TensorMeta` each symbolic leaf
/// registered), so constants materialized at forward time (untracked inputs
/// feeding tracked ops) stay analyzable even after their leaf scopes are
/// dropped. Seeding from the retained symbolic leaf metas keeps the compiled
/// program's semantic fingerprint symbolic-consistent with the traced path
/// (concrete extents are binding data, not part of the fingerprint). The
/// seeded registrations live in the scopes attached to the returned trace,
/// keeping them alive through `compile_ad_source`.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] or
/// [`crate::Error::RuntimeStateSource`] when analysis fails for any live graph
/// in the chain, or [`crate::Error::Internal`] when a leaf input key has no
/// retained leaf metadata (a leaf constructor that does not retain its
/// registration-time `TensorMeta` fed the eager AD path).
#[doc(hidden)]
pub fn analyze_deferred_semantic_trace(raw: &TracedTensor) -> Result<TracedTensor> {
    let mut graphs = Vec::new();
    let mut seen = HashSet::new();
    for root in raw.resolve_roots() {
        collect_chain_graphs(&root, &mut graphs, &mut seen);
    }

    // Canonical symbolic leaves: seed each bound input key from the leaf's
    // retained construction-time metadata (the same `symbolic_input_meta`
    // registered at leaf construction), not from concrete extents derived from
    // the bound value. Concrete extents are binding data and must not leak
    // into the semantic fingerprint.
    let mut leaf_keys = HashSet::new();
    for graph in &graphs {
        if !graph.operations().is_empty() {
            continue;
        }
        for input in graph.inputs() {
            if let ValueKey::Input(key) = &graph.values()[*input].key {
                leaf_keys.insert(key.clone());
            }
        }
    }
    let seeded: Vec<(ValueKey<StdTensorOp>, TensorMeta)> = raw
        .inputs_map
        .iter()
        .map(|(key, value)| {
            let meta = match raw.leaf_metas.get(key) {
                Some(meta) => meta.clone(),
                None => {
                    if leaf_keys.contains(key) {
                        return Err(Error::Internal(format!(
                            "analyze_deferred_semantic_trace: leaf input {key:?} has no retained \
                             leaf metadata; leaf constructors must retain their construction-time \
                             TensorMeta"
                        )));
                    }
                    // Non-leaf bound keys (e.g. derivative-program seed inputs
                    // of a gradient tensor feeding this op) have no retained
                    // leaf meta; concrete extents match the traced path's
                    // shape-specialized derivative seeding.
                    concrete_tensor_meta(value.dtype(), value.shape())
                }
            };
            Ok((ValueKey::Input(key.clone()), meta))
        })
        .collect::<Result<_>>()?;

    let mut metadata_scopes = Vec::with_capacity(graphs.len());
    let mut constraint_scopes = Vec::with_capacity(graphs.len());
    for graph in &graphs {
        let analysis = register_scoped_graph_analysis(graph, seeded.iter().cloned())?;
        metadata_scopes.push(Arc::new(analysis.metadata));
        if !analysis.constraints.is_empty() {
            constraint_scopes.push(Arc::new(analysis.constraints));
        }
    }

    let mut analyzed = raw.clone();
    analyzed.metadata_scopes = MetadataScopeChain::from_materialized(metadata_scopes);
    analyzed.constraint_scopes = constraint_chain_from_materialized(constraint_scopes);
    Ok(analyzed)
}

fn collect_chain_graphs(
    root: &Arc<Graph<StdTensorOp>>,
    graphs: &mut Vec<Arc<Graph<StdTensorOp>>>,
    seen: &mut HashSet<*const Graph<StdTensorOp>>,
) {
    if !seen.insert(Arc::as_ptr(root)) {
        return;
    }
    for parent in root.parents() {
        collect_chain_graphs(parent, graphs, seen);
    }
    graphs.push(Arc::clone(root));
}

fn constraint_chain_from_materialized(
    scopes: Vec<Arc<ShapeConstraintScope>>,
) -> ConstraintScopeChain {
    let mut chain = ConstraintScopeChain::empty();
    for scope in scopes.into_iter().rev() {
        chain = ConstraintScopeChain::with_scope(scope, [&chain]);
    }
    chain
}

/// Compile a traced tensor as an AD source program.
///
/// # Errors
///
/// Returns [`Error::Validation`] for invalid graph metadata or shape
/// constraints, [`Error::RuntimeState`] for missing/inconsistent metadata or
/// cache state, and [`Error::Internal`] when the graph violates a compiler
/// invariant. Extension lowering failures retain their typed
/// [`Error::Extension`] source.
pub fn compile_ad_source(
    compiler: &mut GraphCompiler,
    output: &TracedTensor,
) -> Result<CompiledGraph> {
    compiler.compile_ad_source(output)
}

///
/// # Errors
///
/// Returns [`Error::RuntimeStateSource`] when either metadata scope cannot be
/// registered because the global registry is poisoned. The tensor metadata is
/// derived from the already-valid `Tensor`; no dtype/shape validation is
/// deferred by this operation.
pub fn checkpoint_tensor(tensor: &mut TracedTensor, data: Arc<RetainedValue>) -> Result<()> {
    let old_graph = tensor.graph.clone();
    let old_output_key = old_graph.values()[tensor.val].key.clone();
    let old_inputs = Arc::clone(&tensor.inputs_map);
    let concrete_meta = tenferro_ops::TensorMeta::exact(
        data.dtype(),
        data.shape().iter().copied().map(SymDim::from).collect(),
    );
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
        register_scoped_value_metadata(old_output_key.clone(), concrete_meta.clone())?;
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
    merged.insert(new_key.clone(), data);
    tensor.inputs_map = Arc::new(merged);

    let mut merged_metas = (*tensor.leaf_metas).clone();
    merged_metas.insert(new_key, concrete_meta);
    tensor.leaf_metas = Arc::new(merged_metas);
    Ok(())
}

pub fn allocate_input_key() -> TensorInputKey {
    next_input_key()
}

pub fn allocate_shape_tensor_id() -> u64 {
    next_traced_id()
}

pub fn frozen_input_value(
    frozen: &FrozenProgram,
    input_index: usize,
) -> Option<Arc<RetainedValue>> {
    let input = *frozen.program.inputs().get(input_index)?;
    frozen.bindings.tensor_for_input(input)
}

///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` when
/// `tensor` is an operation result rather than a placeholder input.
pub fn leaf_input_key(tensor: &TracedTensor) -> Result<TensorInputKey> {
    match &tensor.graph.values()[tensor.val].key {
        ValueKey::Input(key) => Ok(key.clone()),
        other => Err(Error::invalid_argument(
            "ad_support::leaf_input_key",
            ErrorPhase::GraphBuild,
            "tensor",
            format!("expected traced leaf input, got {other:?}"),
        )),
    }
}

///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` when
/// `local_id` does not refer to an input value in the linearized graph.
pub fn linear_input_key(
    graph: &Graph<StdTensorOp>,
    local_id: LocalValueId,
) -> Result<TensorInputKey> {
    match &graph.values()[local_id].key {
        ValueKey::Input(key) => Ok(key.clone()),
        other => Err(Error::invalid_argument(
            "ad_support::linear_input_key",
            ErrorPhase::Compile,
            "graph",
            format!("expected linear graph input, got {other:?}"),
        )),
    }
}

///
/// # Errors
///
/// Returns [`Error::TensorRuntime`] containing `ValidationError::IntegerOverflow`
/// when the shape product overflows, or the typed tensor's validation source
/// when it cannot be constructed from `shape`.
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
