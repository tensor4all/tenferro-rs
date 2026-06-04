use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_runtime::error::{Error, Result};
use tenferro_runtime::extension::{self, ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::{GraphCompiler, SymDim, TracedTensor};

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
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
        (
            plan_spec_from_optimize(optimize, &subs).map_err(to_tenferro_error)?,
            None,
        )
    };

    let mut op =
        EinsumExtensionOp::with_output_shape_hint(subscripts.clone(), output_shape_hint, plan_spec);
    if let Some(tree) = static_tree {
        op = op.with_static_tree_hint(tree);
    }

    let outputs = extension::apply(Arc::new(op), inputs);
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("einsum extension produced no output".into()))
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
