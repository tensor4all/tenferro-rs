use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::types::ValueRef;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_runtime::error::{Error, Result};
use tenferro_runtime::extension::{self, ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::{GraphCompiler, SymDim, TracedTensor};

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
use crate::builder::build_einsum_graph_dim_expr;
use crate::cache::{
    einsum_subscripts_retained_bytes, ParsedEinsum, EINSUM_EXTENSION_FAMILY_ID, EINSUM_PARSE_CACHE,
    EINSUM_STATIC_PLANS_CACHE,
};
#[cfg(feature = "autodiff")]
use crate::extension::ensure_einsum_extension_rule_registered;
use crate::extension::EinsumExtensionOp;
use crate::optimize::{
    hash_einsum_plan_spec, plan_spec_from_optimize, resolve_einsum_strategy_with_spec,
    resolve_plan_spec, EinsumPlanSpec,
};
use crate::{
    parse_einsum_subscripts, ContractionTree, EinsumOptimize, EinsumSubscripts,
    Error as EinsumError, Result as EinsumResult, Subscripts,
};

/// N-ary einsum with default time-optimized automatic planning.
///
/// The default optimizer is resolved into a shape-independent plan
/// specification stored in the extension payload. That payload identity
/// participates in traced extension-op equality and in compile/runtime einsum
/// plan caches.
pub fn einsum(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    einsum_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum using integer labels and the default contraction strategy.
///
/// The default optimizer is resolved into a shape-independent plan
/// specification stored in the extension payload. That payload identity
/// participates in traced extension-op equality and in compile/runtime einsum
/// plan caches.
pub fn einsum_subscripts(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<TracedTensor> {
    einsum_subscripts_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum with explicit contraction strategy.
///
/// `optimize` is converted to a shape-independent plan specification carried
/// by the extension payload. `EinsumOptimize::Path` uses JAX-style positions
/// over the current shrinking operand list, so it works with symbolic traced
/// inputs. `EinsumOptimize::Tree` requires concrete shapes for N-ary extension
/// execution; binary trees that lower exactly to `dot_general` bypass the
/// extension path and may use symbolic traced inputs.
///
/// Planner options, explicit paths, and fixed plan identities affect traced
/// extension payload identity and the einsum compile/runtime plan caches.
/// Different options or paths are therefore not treated as identical extension
/// ops.
pub fn einsum_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    let parsed = cached_subscripts(compiler.extension_caches_mut(), subscripts)?;
    einsum_subscripts_with(compiler, inputs, &parsed.subscripts, optimize)
}

/// N-ary einsum with integer labels and explicit contraction strategy.
///
/// `optimize` is converted to a shape-independent plan specification carried
/// by the extension payload. `EinsumOptimize::Path` uses JAX-style positions
/// over the current shrinking operand list, so it works with symbolic traced
/// inputs. `EinsumOptimize::Tree` requires concrete shapes for N-ary extension
/// execution; binary trees that lower exactly to `dot_general` bypass the
/// extension path and may use symbolic traced inputs.
///
/// Planner options, explicit paths, and fixed plan identities affect traced
/// extension payload identity and the einsum compile/runtime plan caches.
/// Different options or paths are therefore not treated as identical extension
/// ops.
pub fn einsum_subscripts_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "einsum requires at least one input tensor".into(),
        ));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(Error::ContractionError(format!(
            "einsum subscripts expect {} inputs, got {}",
            subscripts.inputs.len(),
            inputs.len()
        )));
    }

    let output_shape_hint = infer_symbolic_output_shape(subscripts, inputs)?;
    if let Some(result) = try_direct_binary_dot_general(inputs, subscripts, &optimize)? {
        return Ok(result);
    }

    #[cfg(feature = "autodiff")]
    ensure_einsum_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))?;

    let subs = Subscripts::from(subscripts);

    let (plan_spec, static_tree) = if let Some(shapes) = concrete_shapes(inputs) {
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let (plan_spec, tree) = match optimize {
            EinsumOptimize::Tree(tree) => {
                let (plan_spec, tree) = resolve_einsum_strategy_with_spec(
                    EinsumOptimize::Tree(tree),
                    &subs,
                    &shape_refs,
                )
                .map_err(to_tenferro_error)?;
                let tree = cached_static_tree(
                    compiler.extension_caches_mut(),
                    subscripts,
                    &plan_spec,
                    &shapes,
                    || Ok(tree),
                )?;
                (plan_spec, tree)
            }
            optimize => {
                let plan_spec =
                    plan_spec_from_optimize(optimize, &subs).map_err(to_tenferro_error)?;
                let tree = cached_static_tree(
                    compiler.extension_caches_mut(),
                    subscripts,
                    &plan_spec,
                    &shapes,
                    || resolve_plan_spec(&plan_spec, &subs, &shape_refs),
                )?;
                (plan_spec, tree)
            }
        };
        (plan_spec, Some(tree))
    } else {
        let plan_spec = plan_spec_from_optimize(optimize, &subs).map_err(to_tenferro_error)?;
        let tree = symbolic_fixed_path_tree(&plan_spec, &subs, inputs)?;
        (plan_spec, tree.map(Arc::new))
    };

    if let Some(tree) = static_tree {
        return expand_traced_einsum_graph(inputs, subscripts, tree.as_ref(), output_shape_hint);
    }

    let op =
        EinsumExtensionOp::with_output_shape_hint(subscripts.clone(), output_shape_hint, plan_spec);
    let outputs = extension::apply(Arc::new(op), inputs);
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("einsum extension produced no output".into()))
}

fn expand_traced_einsum_graph(
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    tree: &ContractionTree,
    output_shape_hint: Vec<SymDim>,
) -> Result<TracedTensor> {
    let op = EinsumExtensionOp::with_output_shape_hint(
        subscripts.clone(),
        output_shape_hint,
        EinsumPlanSpec::LeftToRight,
    );
    let input_dtypes: Vec<_> = inputs.iter().map(|tensor| tensor.dtype).collect();
    let input_sym_shapes: Vec<Vec<SymDim>> = inputs
        .iter()
        .map(|tensor| {
            tensor
                .sym_shape()
                .map(|shape| shape.to_vec())
                .unwrap_or_else(|| {
                    (0..tensor.rank)
                        .map(|axis| tensor.axis_sym_dim(axis))
                        .collect()
                })
        })
        .collect();
    let input_sym_shape_refs: Vec<_> = input_sym_shapes.iter().map(Vec::as_slice).collect();
    let output_metas = op.infer_output_meta(&input_dtypes, &input_sym_shape_refs);
    let input_dim_shapes = traced_dim_expr_shapes(inputs);

    let outputs = extension::apply_expanded_graph(inputs, output_metas, |builder, input_refs| {
        let result = build_einsum_graph_dim_expr(builder, tree, input_refs, &input_dim_shapes)
            .map_err(|err| Error::ContractionError(err.to_string()))?;
        let ValueRef::Local(local) = result else {
            return Err(Error::Internal(
                "expanded einsum returned an external value".into(),
            ));
        };
        Ok(vec![local])
    })?;

    outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("expanded einsum produced no output".into()))
}

fn traced_dim_expr_shapes(inputs: &[&TracedTensor]) -> Vec<Vec<DimExpr>> {
    inputs
        .iter()
        .map(|tensor| DimExpr::input_shape(0, tensor.rank))
        .collect()
}

fn symbolic_fixed_path_tree(
    plan_spec: &EinsumPlanSpec,
    subs: &Subscripts,
    inputs: &[&TracedTensor],
) -> Result<Option<ContractionTree>> {
    if matches!(plan_spec, EinsumPlanSpec::Auto(_)) {
        return Ok(None);
    }
    let dummy_shapes = symbolic_dummy_shapes(inputs);
    let shape_refs: Vec<&[usize]> = dummy_shapes.iter().map(Vec::as_slice).collect();
    resolve_plan_spec(plan_spec, subs, &shape_refs)
        .map(Some)
        .map_err(to_tenferro_error)
}

fn symbolic_dummy_shapes(inputs: &[&TracedTensor]) -> Vec<Vec<usize>> {
    inputs.iter().map(|tensor| vec![1; tensor.rank]).collect()
}

fn try_direct_binary_dot_general(
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: &EinsumOptimize,
) -> Result<Option<TracedTensor>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return Ok(None);
    }
    if !optimize_allows_direct_binary_dot(optimize)? {
        return Ok(None);
    }

    let lhs_labels = &subscripts.inputs[0];
    let rhs_labels = &subscripts.inputs[1];
    if lhs_labels.len() != inputs[0].rank || rhs_labels.len() != inputs[1].rank {
        return Ok(None);
    }
    validate_direct_binary_dot_label_dims(inputs, subscripts)?;

    let Some(plan) =
        try_build_exact_output_binary_dot_plan(lhs_labels, rhs_labels, &subscripts.output)
    else {
        return Ok(None);
    };

    let result = match plan.operand_order {
        BinaryDotOperandOrder::Original => inputs[0].dot_general(inputs[1], plan.config),
        BinaryDotOperandOrder::Swapped => inputs[1].dot_general(inputs[0], plan.config),
    };
    Ok(Some(result))
}

fn validate_direct_binary_dot_label_dims(
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<()> {
    let mut label_dims = std::collections::HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs.iter()) {
        let Some(shape) = tensor.sym_shape() else {
            continue;
        };
        for (&label, dim) in labels.iter().zip(shape.iter()) {
            let Some(dim) = dim.constant_value() else {
                continue;
            };
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(Error::ContractionError(format!(
                        "einsum label {label} has inconsistent dimensions {existing} and {dim}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn optimize_allows_direct_binary_dot(optimize: &EinsumOptimize) -> Result<bool> {
    match optimize {
        EinsumOptimize::Auto(options) => {
            options.validate().map_err(to_tenferro_error)?;
            Ok(true)
        }
        EinsumOptimize::False => Ok(true),
        EinsumOptimize::Tree(tree) => {
            Ok(tree.step_count() == 1 && matches!(tree.step_pair(0), Some((0, 1)) | Some((1, 0))))
        }
        EinsumOptimize::Nested(_) | EinsumOptimize::Path(_) => Ok(false),
    }
}

fn cached_subscripts(
    caches: &mut ExtensionCacheStore,
    notation: &str,
) -> Result<Arc<ParsedEinsum>> {
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_PARSE_CACHE,
        hash_value(&notation),
    );
    if let Some(cached) = caches.get::<Arc<ParsedEinsum>>(&key) {
        return Ok(Arc::clone(cached));
    }

    let parsed = Arc::new(ParsedEinsum {
        subscripts: parse_einsum_subscripts(notation).map_err(to_tenferro_error)?,
    });
    let retained_bytes = notation.len() + einsum_subscripts_retained_bytes(&parsed.subscripts);
    caches.put(key, Arc::clone(&parsed), retained_bytes);
    Ok(parsed)
}

fn cached_static_tree(
    caches: &mut ExtensionCacheStore,
    subscripts: &EinsumSubscripts,
    plan_spec: &EinsumPlanSpec,
    shapes: &[Vec<usize>],
    build: impl FnOnce() -> EinsumResult<ContractionTree>,
) -> Result<Arc<ContractionTree>> {
    let mut plan_hasher = DefaultHasher::new();
    hash_einsum_plan_spec(plan_spec, &mut plan_hasher);
    let key_data = (subscripts.clone(), shapes.to_vec(), plan_hasher.finish());
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_STATIC_PLANS_CACHE,
        hash_value(&key_data),
    );
    if let Some(cached) = caches.get::<Arc<ContractionTree>>(&key) {
        return Ok(Arc::clone(cached));
    }

    let tree = Arc::new(build().map_err(to_tenferro_error)?);
    let retained_bytes = einsum_subscripts_retained_bytes(subscripts)
        + shapes
            .iter()
            .map(|shape| shape.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
        + std::mem::size_of::<u64>()
        + tree.retained_bytes_for_cache_stats();
    caches.put(key, Arc::clone(&tree), retained_bytes);
    Ok(tree)
}

fn concrete_shapes(inputs: &[&TracedTensor]) -> Option<Vec<Vec<usize>>> {
    inputs
        .iter()
        .map(|tensor| {
            tensor
                .sym_shape()?
                .iter()
                .map(|dim| dim.constant_value())
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

fn infer_symbolic_output_shape(
    subscripts: &EinsumSubscripts,
    inputs: &[&TracedTensor],
) -> Result<Vec<SymDim>> {
    let mut label_dims = std::collections::HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs.iter()) {
        let shape: Vec<_> = tensor
            .sym_shape()
            .map(|shape| shape.to_vec())
            .unwrap_or_else(|| {
                (0..tensor.rank)
                    .map(|axis| tensor.axis_sym_dim(axis))
                    .collect()
            });
        if labels.len() != shape.len() {
            return Err(Error::ContractionError(format!(
                "einsum input rank mismatch: labels={}, shape={}",
                labels.len(),
                shape.len()
            )));
        }
        for (&label, dim) in labels.iter().zip(shape) {
            label_dims.entry(label).or_insert(dim);
        }
    }
    subscripts
        .output
        .iter()
        .map(|label| {
            label_dims.get(label).cloned().ok_or_else(|| {
                Error::ContractionError(format!(
                    "einsum output label {label} is missing from inputs"
                ))
            })
        })
        .collect()
}

fn to_tenferro_error(error: EinsumError) -> Error {
    Error::ContractionError(error.to_string())
}

fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use tenferro_ops::std_tensor_op::StdTensorOp;
    use tenferro_runtime::{DType, GraphCompiler, TracedTensor};

    use super::{einsum, einsum_with};
    use crate::EinsumOptimize;

    #[test]
    fn concrete_traced_nary_einsum_expands_to_standard_graph() {
        let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
        let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
        let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]);
        let mut compiler = GraphCompiler::new();

        let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

        assert!(out
            .graph
            .operations()
            .iter()
            .all(|node| { !matches!(node.operation, StdTensorOp::Extension(_)) }));
        assert!(out
            .graph
            .operations()
            .iter()
            .any(|node| { matches!(node.operation, StdTensorOp::DotGeneral { .. }) }));
    }

    #[test]
    fn symbolic_path_traced_nary_einsum_expands_to_standard_graph() {
        let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let c = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let mut compiler = GraphCompiler::new();

        let out = einsum_with(
            &mut compiler,
            &[&a, &b, &c],
            "ij,jk,kl->il",
            EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
        )
        .unwrap();

        assert!(out
            .graph
            .operations()
            .iter()
            .all(|node| { !matches!(node.operation, StdTensorOp::Extension(_)) }));
        assert!(out
            .graph
            .operations()
            .iter()
            .any(|node| { matches!(node.operation, StdTensorOp::DotGeneral { .. }) }));
    }

    #[test]
    fn symbolic_auto_traced_nary_einsum_remains_extension() {
        let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let c = TracedTensor::input_symbolic_shape(DType::F64, 2);
        let mut compiler = GraphCompiler::new();

        let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

        assert!(out
            .graph
            .operations()
            .iter()
            .any(|node| { matches!(node.operation, StdTensorOp::Extension(_)) }));
    }
}
