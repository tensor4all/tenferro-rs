use computegraph::graph::Graph;
use computegraph::types::{LocalValueId, ValueKey, ValueRef};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
pub(crate) mod test_support;

use tenferro_ops::ad::context::{
    lookup_global_metadata, register_scoped_global_metadata_batch, GlobalMetadataScope, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_tensor::DType;
use tenferro_tensor::Tensor;

use crate::shape_constraint::{ScopedShapeConstraint, ShapeConstraintScope};
use crate::shape_infer::{
    infer_extension_output_meta_with_constraints, infer_output_dtype, infer_output_extents,
};
use crate::{Error, Result};

#[derive(Clone)]
pub(crate) struct MetadataScopeChain {
    node: Arc<MetadataScopeChainNode>,
}

struct MetadataScopeChainNode {
    scope: Option<Arc<GlobalMetadataScope>>,
    parents: Vec<MetadataScopeChain>,
    materialized: OnceLock<Vec<Arc<GlobalMetadataScope>>>,
}

impl MetadataScopeChain {
    pub(crate) fn empty() -> Self {
        Self {
            node: Arc::new(MetadataScopeChainNode {
                scope: None,
                parents: Vec::new(),
                materialized: OnceLock::new(),
            }),
        }
    }

    pub(crate) fn from_scope(scope: GlobalMetadataScope) -> Self {
        Self::with_scope(Arc::new(scope), [])
    }

    pub(crate) fn with_new<'a>(
        scope: GlobalMetadataScope,
        inherited: impl IntoIterator<Item = &'a MetadataScopeChain>,
    ) -> Self {
        Self::with_scope(Arc::new(scope), inherited)
    }

    pub(crate) fn with_scope<'a>(
        scope: Arc<GlobalMetadataScope>,
        inherited: impl IntoIterator<Item = &'a MetadataScopeChain>,
    ) -> Self {
        Self {
            node: Arc::new(MetadataScopeChainNode {
                scope: Some(scope),
                parents: inherited.into_iter().cloned().collect(),
                materialized: OnceLock::new(),
            }),
        }
    }

    pub(crate) fn from_materialized(scopes: Vec<Arc<GlobalMetadataScope>>) -> Self {
        let mut chain = Self::empty();
        for scope in scopes.into_iter().rev() {
            chain = Self::with_scope(scope, [&chain]);
        }
        chain
    }

    pub(crate) fn materialize(&self) -> Vec<Arc<GlobalMetadataScope>> {
        self.as_slice().to_vec()
    }

    pub(crate) fn as_slice(&self) -> &[Arc<GlobalMetadataScope>] {
        self.node
            .materialized
            .get_or_init(|| {
                let mut scopes = Vec::new();
                let mut seen = HashSet::new();
                self.extend_materialized(&mut scopes, &mut seen);
                scopes
            })
            .as_slice()
    }

    fn extend_materialized(
        &self,
        scopes: &mut Vec<Arc<GlobalMetadataScope>>,
        seen: &mut HashSet<*const GlobalMetadataScope>,
    ) {
        if let Some(scope) = &self.node.scope {
            push_metadata_scope_seen(scopes, seen, Arc::clone(scope));
        }
        for parent in &self.node.parents {
            parent.extend_materialized(scopes, seen);
        }
    }
}

pub(crate) fn tensor_meta(dtype: DType, shape: Vec<SymDim>) -> TensorMeta {
    TensorMeta::exact(dtype, shape)
}

pub(crate) fn concrete_tensor_meta(dtype: DType, shape: &[usize]) -> TensorMeta {
    tensor_meta(dtype, shape.iter().copied().map(SymDim::from).collect())
}

pub(crate) fn symbolic_input_meta(dtype: DType, tensor_id: u64, rank: usize) -> TensorMeta {
    tensor_meta(
        dtype,
        (0..rank)
            .map(|axis| SymDim::tensor_axis(tensor_id, axis))
            .collect(),
    )
}

pub fn tensor_meta_from_tensor(tensor: &Tensor) -> TensorMeta {
    concrete_tensor_meta(tensor.dtype(), tensor.shape())
}

pub fn register_scoped_value_metadata(
    key: ValueKey<StdTensorOp>,
    meta: TensorMeta,
) -> Result<GlobalMetadataScope> {
    register_scoped_global_metadata_batch([(key, meta)])
        .map_err(|err| metadata_error(err.to_string()))
}

pub fn register_scoped_metadata_batch(
    entries: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<GlobalMetadataScope> {
    register_scoped_global_metadata_batch(entries).map_err(|err| metadata_error(err.to_string()))
}

pub fn registered_meta(key: &ValueKey<StdTensorOp>) -> Result<TensorMeta> {
    lookup_global_metadata(key)
        .map_err(|err| metadata_error(err.to_string()))?
        .ok_or_else(|| metadata_error(format!("missing registered metadata for {:?}", key)))
}

pub fn register_scoped_graph_metadata(
    graph: &Graph<StdTensorOp>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<GlobalMetadataScope> {
    Ok(register_scoped_graph_analysis(graph, seeded)?.metadata)
}

/// Metadata and shape constraints discovered by one graph-analysis walk.
///
/// The public container is used only by [`crate::ad_support`]; constraint
/// representation remains private to `tenferro-runtime`.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ad_support::register_scoped_graph_analysis;
/// use tenferro_runtime::{DType, TracedTensor};
///
/// let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
/// let analysis = register_scoped_graph_analysis(input.graph(), []).unwrap();
/// assert!(analysis.constraints.is_empty());
/// ```
pub struct RegisteredGraphAnalysis {
    /// Graph-scoped metadata registered by the analysis walk.
    pub metadata: GlobalMetadataScope,
    /// Shape constraints recorded by extension nodes in the analyzed graph.
    pub constraints: ShapeConstraintScope,
}

impl std::fmt::Debug for RegisteredGraphAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredGraphAnalysis")
            .field("constraints", &self.constraints)
            .finish_non_exhaustive()
    }
}

/// Analyze a graph once and register both output metadata and scoped shape constraints.
///
/// # Examples
///
/// ```rust
/// use tenferro_runtime::ad_support::register_scoped_graph_analysis;
/// use tenferro_runtime::{DType, TracedTensor};
///
/// let input = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
/// let analysis = register_scoped_graph_analysis(input.graph(), []).unwrap();
/// assert!(analysis.constraints.is_empty());
/// ```
pub fn register_scoped_graph_analysis(
    graph: &Graph<StdTensorOp>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<RegisteredGraphAnalysis> {
    let analysis = graph_analysis_registrations(graph, None, seeded)?;
    let metadata = register_scoped_global_metadata_batch(analysis.metadata)
        .map_err(|err| metadata_error(err.to_string()))?;
    Ok(RegisteredGraphAnalysis {
        metadata,
        constraints: ShapeConstraintScope::new(analysis.constraints),
    })
}

pub fn register_scoped_live_graph_metadata(
    graph: &Graph<StdTensorOp>,
    live_values: &HashSet<LocalValueId>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<GlobalMetadataScope> {
    register_scoped_global_metadata_batch(
        graph_analysis_registrations(graph, Some(live_values), seeded)?.metadata,
    )
    .map_err(|err| metadata_error(err.to_string()))
}

pub fn metadata_scopes_with_new<'a>(
    scope: GlobalMetadataScope,
    inherited: impl IntoIterator<Item = &'a [Arc<GlobalMetadataScope>]>,
) -> Vec<Arc<GlobalMetadataScope>> {
    let scope = Arc::new(scope);
    metadata_scopes_with_scope(scope, inherited)
}

pub fn metadata_scopes_for_scope(scope: GlobalMetadataScope) -> Vec<Arc<GlobalMetadataScope>> {
    vec![Arc::new(scope)]
}

pub fn metadata_scopes_with_scope<'a>(
    scope: Arc<GlobalMetadataScope>,
    inherited: impl IntoIterator<Item = &'a [Arc<GlobalMetadataScope>]>,
) -> Vec<Arc<GlobalMetadataScope>> {
    let mut scopes = Vec::new();
    let mut seen = HashSet::new();
    push_metadata_scope_seen(&mut scopes, &mut seen, scope);
    extend_metadata_scopes(&mut scopes, &mut seen, inherited);
    scopes
}

pub fn push_metadata_scope(
    scopes: &mut Vec<Arc<GlobalMetadataScope>>,
    scope: Arc<GlobalMetadataScope>,
) {
    if scopes.iter().all(|existing| !Arc::ptr_eq(existing, &scope)) {
        scopes.push(scope);
    }
}

fn push_metadata_scope_seen(
    scopes: &mut Vec<Arc<GlobalMetadataScope>>,
    seen: &mut HashSet<*const GlobalMetadataScope>,
    scope: Arc<GlobalMetadataScope>,
) {
    if seen.insert(Arc::as_ptr(&scope)) {
        scopes.push(scope);
    }
}

fn extend_metadata_scopes<'a>(
    scopes: &mut Vec<Arc<GlobalMetadataScope>>,
    seen: &mut HashSet<*const GlobalMetadataScope>,
    inherited: impl IntoIterator<Item = &'a [Arc<GlobalMetadataScope>]>,
) {
    for source in inherited {
        for scope in source {
            push_metadata_scope_seen(scopes, seen, Arc::clone(scope));
        }
    }
}

struct GraphAnalysisRegistrations {
    metadata: Vec<(ValueKey<StdTensorOp>, TensorMeta)>,
    constraints: Vec<ScopedShapeConstraint>,
}

fn graph_analysis_registrations(
    graph: &Graph<StdTensorOp>,
    live_values: Option<&HashSet<LocalValueId>>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<GraphAnalysisRegistrations> {
    let seeded: Vec<_> = seeded.into_iter().collect();
    // Start from just the seeded inputs. External keys not in `seeded` are
    // resolved on demand via a single-key lookup against the global
    // registry — crucially, we do NOT clone the entire global map. The
    // global registry grows monotonically across a process, so a full-map
    // snapshot per graph construction is quadratic in the total number
    // of registered ops and dominated oracle_replay runtime.
    let mut known: HashMap<ValueKey<StdTensorOp>, TensorMeta> = seeded.iter().cloned().collect();

    let mut registrations = seeded;
    let mut constraints = Vec::new();
    let mut visited = HashSet::new();
    append_graph_metadata_registrations(
        graph,
        live_values,
        true,
        &mut known,
        &mut registrations,
        &mut constraints,
        &mut visited,
    )?;

    Ok(GraphAnalysisRegistrations {
        metadata: registrations,
        constraints,
    })
}

fn append_graph_metadata_registrations(
    graph: &Graph<StdTensorOp>,
    live_values: Option<&HashSet<LocalValueId>>,
    collect_constraints: bool,
    known: &mut HashMap<ValueKey<StdTensorOp>, TensorMeta>,
    registrations: &mut Vec<(ValueKey<StdTensorOp>, TensorMeta)>,
    constraints: &mut Vec<ScopedShapeConstraint>,
    visited: &mut HashSet<*const Graph<StdTensorOp>>,
) -> Result<()> {
    let graph_ptr: *const Graph<StdTensorOp> = graph;
    if !visited.insert(graph_ptr) {
        return Ok(());
    }
    #[cfg(test)]
    test_support::record_graph_visit();

    for op_node in graph.operations() {
        #[cfg(test)]
        test_support::record_operation_visit();
        if let Some(live_values) = live_values {
            if !op_node
                .outputs
                .iter()
                .any(|output_id| live_values.contains(output_id))
            {
                continue;
            }
        }

        if !collect_constraints {
            let mut all_outputs_registered = true;
            for &output_id in &op_node.outputs {
                let key = graph.values()[output_id].key.clone();
                let Some(meta) =
                    lookup_global_metadata(&key).map_err(|err| metadata_error(err.to_string()))?
                else {
                    all_outputs_registered = false;
                    break;
                };
                known.insert(key, meta);
            }
            if all_outputs_registered {
                continue;
            }
        }

        let input_keys: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| match input {
                ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                ValueRef::External(key) => key,
            })
            .cloned()
            .collect();
        let mut input_metas = Vec::with_capacity(input_keys.len());
        for key in &input_keys {
            if let Some(meta) = known.get(key).cloned() {
                input_metas.push(meta);
                continue;
            }
            if let Some(meta) =
                lookup_global_metadata(key).map_err(|err| metadata_error(err.to_string()))?
            {
                known.insert(key.clone(), meta.clone());
                input_metas.push(meta);
                continue;
            }

            // Traced construction normally keeps parent metadata scopes alive,
            // so this is only the compatibility path for manually assembled or
            // otherwise unregistered parent graphs.
            let Some(parent) = graph
                .parents()
                .iter()
                .find(|parent| parent.values().iter().any(|value| value.key == *key))
            else {
                return Err(metadata_error(format!(
                    "missing input metadata for {:?}",
                    key
                )));
            };
            append_graph_metadata_registrations(
                parent,
                None,
                false,
                known,
                registrations,
                constraints,
                visited,
            )?;
            input_metas.push(
                known.get(key).cloned().ok_or_else(|| {
                    metadata_error(format!("missing input metadata for {:?}", key))
                })?,
            );
        }

        let inferred = infer_output_metas(&op_node.operation, &input_metas)?;
        let origin_keys: Vec<_> = op_node
            .outputs
            .iter()
            .map(|&output_id| graph.values()[output_id].key.clone())
            .collect();
        if collect_constraints {
            constraints.extend(inferred.constraints.into_iter().map(|local| {
                ScopedShapeConstraint {
                    origins: origin_keys.clone(),
                    inputs: input_keys.clone(),
                    local,
                }
            }));
        }
        for (&output_id, meta) in op_node.outputs.iter().zip(inferred.output_metas) {
            let key = graph.values()[output_id].key.clone();
            // INVARIANT: both owners are required: `known` feeds later local
            // inference in this walk, while `registrations` is returned for
            // scoped metadata publication after the walk.
            known.insert(key.clone(), meta.clone());
            registrations.push((key, meta));
        }
    }

    Ok(())
}

struct InferredGraphOutput {
    output_metas: Vec<TensorMeta>,
    constraints: Vec<crate::shape_constraint::LocalShapeConstraint>,
}

fn infer_output_metas(op: &StdTensorOp, input_metas: &[TensorMeta]) -> Result<InferredGraphOutput> {
    let input_shape_exprs: Vec<Vec<DimExpr>> = input_metas
        .iter()
        .enumerate()
        .map(|(input_idx, meta)| DimExpr::input_shape(input_idx, meta.rank()))
        .collect();
    let input_shape_refs: Vec<&[DimExpr]> = input_shape_exprs.iter().map(Vec::as_slice).collect();
    let input_dtypes: Vec<DType> = input_metas.iter().map(|meta| meta.dtype).collect();
    let resolved_inputs = resolved_bound_shapes(input_metas)?;
    let resolved_input_refs: Vec<&[SymDim]> = resolved_inputs.iter().map(Vec::as_slice).collect();

    if let StdTensorOp::Extension(ext) = op {
        let inferred = infer_extension_output_meta_with_constraints(
            ext.as_ref(),
            &input_dtypes,
            &input_shape_refs,
        )?;
        let output_metas = inferred
            .output_metas
            .into_iter()
            .map(|(dtype, shape)| {
                tensor_meta(
                    dtype,
                    shape
                        .iter()
                        .map(|dim| SymDim::from_dim_expr(dim, &resolved_input_refs))
                        .collect(),
                )
            })
            .collect();
        return Ok(InferredGraphOutput {
            output_metas,
            constraints: inferred.constraints,
        });
    }

    let output_dtype = infer_output_dtype(op, &input_dtypes)?;
    let output_metas = infer_output_extents(op, &input_shape_refs)?
        .into_iter()
        .map(|extents| {
            let resolved_extents = extents
                .into_iter()
                .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, &resolved_input_refs)))
                .collect();
            TensorMeta::with_extents(output_dtype, resolved_extents)
        })
        .collect();
    Ok(InferredGraphOutput {
        output_metas,
        constraints: Vec::new(),
    })
}

fn resolved_bound_shapes(input_metas: &[TensorMeta]) -> Result<Vec<Vec<SymDim>>> {
    input_metas
        .iter()
        .map(|meta| {
            meta.bound_shape().ok_or_else(|| {
                metadata_error(
                    "metadata contains an unknown shape extent; cannot resolve output metadata",
                )
            })
        })
        .collect()
}

fn metadata_error(message: impl Into<String>) -> Error {
    Error::InvalidCompiledGraph {
        message: format!("metadata registration: {}", message.into()),
    }
}
