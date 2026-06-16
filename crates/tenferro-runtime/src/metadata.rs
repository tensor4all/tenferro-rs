use computegraph::graph::Graph;
use computegraph::types::{LocalValueId, ValueKey, ValueRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tenferro_ops::ad::context::{
    lookup_global_metadata, register_scoped_global_metadata_batch, GlobalMetadataScope, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_tensor::DType;
use tenferro_tensor::Tensor;

use crate::shape_infer::{infer_extension_output_meta, infer_output_dtype, infer_output_extents};
use crate::{Error, Result};

pub type MetadataScope = GlobalMetadataScope;

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
) -> MetadataScope {
    register_scoped_global_metadata_batch([(key, meta)])
}

pub fn register_scoped_metadata_batch(
    entries: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> MetadataScope {
    register_scoped_global_metadata_batch(entries)
}

pub fn registered_meta(key: &ValueKey<StdTensorOp>) -> TensorMeta {
    lookup_global_metadata(key)
        .unwrap_or_else(|| panic!("metadata lookup: missing registered metadata for {:?}", key))
}

pub fn register_scoped_graph_metadata(
    graph: &Graph<StdTensorOp>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> MetadataScope {
    try_register_scoped_graph_metadata(graph, seeded).unwrap_or_else(|err| panic!("{err}"))
}

pub(crate) fn try_register_scoped_graph_metadata(
    graph: &Graph<StdTensorOp>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<MetadataScope> {
    Ok(register_scoped_global_metadata_batch(
        graph_metadata_registrations(graph, None, seeded)?,
    ))
}

pub fn register_scoped_live_graph_metadata(
    graph: &Graph<StdTensorOp>,
    live_values: &HashSet<LocalValueId>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> MetadataScope {
    try_register_scoped_live_graph_metadata(graph, live_values, seeded)
        .unwrap_or_else(|err| panic!("{err}"))
}

pub(crate) fn try_register_scoped_live_graph_metadata(
    graph: &Graph<StdTensorOp>,
    live_values: &HashSet<LocalValueId>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<MetadataScope> {
    Ok(register_scoped_global_metadata_batch(
        graph_metadata_registrations(graph, Some(live_values), seeded)?,
    ))
}

pub fn metadata_scopes_with_new<'a>(
    scope: MetadataScope,
    inherited: impl IntoIterator<Item = &'a [Arc<MetadataScope>]>,
) -> Vec<Arc<MetadataScope>> {
    let scope = Arc::new(scope);
    metadata_scopes_with_scope(scope, inherited)
}

pub fn metadata_scopes_for_scope(scope: MetadataScope) -> Vec<Arc<MetadataScope>> {
    vec![Arc::new(scope)]
}

pub fn metadata_scopes_with_scope<'a>(
    scope: Arc<MetadataScope>,
    inherited: impl IntoIterator<Item = &'a [Arc<MetadataScope>]>,
) -> Vec<Arc<MetadataScope>> {
    let mut scopes = Vec::new();
    let mut seen = HashSet::new();
    push_metadata_scope_seen(&mut scopes, &mut seen, scope);
    extend_metadata_scopes(&mut scopes, &mut seen, inherited);
    scopes
}

pub fn push_metadata_scope(scopes: &mut Vec<Arc<MetadataScope>>, scope: Arc<MetadataScope>) {
    if scopes.iter().all(|existing| !Arc::ptr_eq(existing, &scope)) {
        scopes.push(scope);
    }
}

fn push_metadata_scope_seen(
    scopes: &mut Vec<Arc<MetadataScope>>,
    seen: &mut HashSet<*const MetadataScope>,
    scope: Arc<MetadataScope>,
) {
    if seen.insert(Arc::as_ptr(&scope)) {
        scopes.push(scope);
    }
}

fn extend_metadata_scopes<'a>(
    scopes: &mut Vec<Arc<MetadataScope>>,
    seen: &mut HashSet<*const MetadataScope>,
    inherited: impl IntoIterator<Item = &'a [Arc<MetadataScope>]>,
) {
    for source in inherited {
        for scope in source {
            push_metadata_scope_seen(scopes, seen, Arc::clone(scope));
        }
    }
}

fn graph_metadata_registrations(
    graph: &Graph<StdTensorOp>,
    live_values: Option<&HashSet<LocalValueId>>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> Result<Vec<(ValueKey<StdTensorOp>, TensorMeta)>> {
    let seeded: Vec<_> = seeded.into_iter().collect();
    // Start from just the seeded inputs. External keys not in `seeded` are
    // resolved on demand via a single-key lookup against the global
    // registry — crucially, we do NOT clone the entire global map. The
    // global registry grows monotonically across a process, so a full-map
    // snapshot per graph construction is quadratic in the total number
    // of registered ops and dominated oracle_replay runtime.
    let mut known: HashMap<ValueKey<StdTensorOp>, TensorMeta> = seeded.iter().cloned().collect();

    let mut registrations = seeded;
    for op_node in graph.operations() {
        if let Some(live_values) = live_values {
            if !op_node
                .outputs
                .iter()
                .any(|output_id| live_values.contains(output_id))
            {
                continue;
            }
        }

        let input_metas: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| {
                let key = match input {
                    ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                    ValueRef::External(key) => key,
                };
                known
                    .get(key)
                    .cloned()
                    .or_else(|| lookup_global_metadata(key))
                    .ok_or_else(|| metadata_error(format!("missing input metadata for {:?}", key)))
            })
            .collect::<Result<_>>()?;

        let output_metas = infer_output_metas(&op_node.operation, &input_metas)?;
        for (&output_id, meta) in op_node.outputs.iter().zip(output_metas) {
            let key = graph.values()[output_id].key.clone();
            known.insert(key.clone(), meta.clone());
            registrations.push((key, meta));
        }
    }

    Ok(registrations)
}

fn infer_output_metas(op: &StdTensorOp, input_metas: &[TensorMeta]) -> Result<Vec<TensorMeta>> {
    let input_shape_exprs: Vec<Vec<DimExpr>> = input_metas
        .iter()
        .enumerate()
        .map(|(input_idx, meta)| DimExpr::input_shape(input_idx, meta.shape.len()))
        .collect();
    let input_shape_refs: Vec<&[DimExpr]> = input_shape_exprs.iter().map(Vec::as_slice).collect();
    let input_dtypes: Vec<DType> = input_metas.iter().map(|meta| meta.dtype).collect();

    if let StdTensorOp::Extension(ext) = op {
        let metas = infer_extension_output_meta(ext.as_ref(), &input_dtypes, &input_shape_refs)?;
        let resolved_inputs: Vec<&[SymDim]> = input_metas
            .iter()
            .map(|meta| meta.shape.as_slice())
            .collect();
        return Ok(metas
            .into_iter()
            .map(|(dtype, shape)| {
                tensor_meta(
                    dtype,
                    shape
                        .iter()
                        .map(|dim| SymDim::from_dim_expr(dim, &resolved_inputs))
                        .collect(),
                )
            })
            .collect());
    }

    let output_dtype = infer_output_dtype(op, &input_dtypes);
    Ok(infer_output_extents(op, &input_shape_refs)?
        .into_iter()
        .map(|extents| {
            let resolved_inputs: Vec<&[SymDim]> = input_metas
                .iter()
                .map(|meta| meta.shape.as_slice())
                .collect();
            let resolved_extents = extents
                .into_iter()
                .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, &resolved_inputs)))
                .collect();
            TensorMeta::with_extents(output_dtype, resolved_extents)
        })
        .collect())
}

fn metadata_error(message: impl Into<String>) -> Error {
    Error::InvalidCompiledGraph {
        message: format!("metadata registration: {}", message.into()),
    }
}
