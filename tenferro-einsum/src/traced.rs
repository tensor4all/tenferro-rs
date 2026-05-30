use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_device::{Error as DeviceError, Result as DeviceResult};
use tenferro_runtime::error::{Error, Result};
use tenferro_runtime::extension::{self, ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::{GraphCompiler, SymDim, TracedTensor};

use crate::cache::{
    einsum_subscripts_retained_bytes, ParsedEinsum, EINSUM_EXTENSION_FAMILY_ID, EINSUM_PARSE_CACHE,
    EINSUM_STATIC_PLANS_CACHE,
};
#[cfg(feature = "autodiff")]
use crate::extension::ensure_einsum_extension_rule_registered;
use crate::extension::EinsumExtensionOp;
use crate::optimize::{is_default_auto_options, resolve_einsum_strategy};
use crate::{
    parse_einsum_subscripts, ContractionTree, EinsumOptimize, EinsumSubscripts, Subscripts,
};

/// N-ary einsum with default FLOPS-first optimization.
pub fn einsum(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor> {
    einsum_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum using integer labels and the default contraction strategy.
pub fn einsum_subscripts(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<TracedTensor> {
    einsum_subscripts_with(compiler, inputs, subscripts, EinsumOptimize::default())
}

/// N-ary einsum with explicit contraction strategy.
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
pub fn einsum_subscripts_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
    #[cfg(feature = "autodiff")]
    ensure_einsum_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))?;

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
    let mut op = EinsumExtensionOp::with_output_shape_hint(subscripts.clone(), output_shape_hint);

    if let Some(shapes) = concrete_shapes(inputs) {
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let subs = Subscripts::from(subscripts);
        let tree = match optimize {
            EinsumOptimize::Auto(opts) if is_default_auto_options(&opts) => {
                let key = (subscripts.clone(), shapes.clone());
                cached_static_tree(compiler.extension_caches_mut(), &key, || {
                    resolve_einsum_strategy(EinsumOptimize::Auto(opts), &subs, &shape_refs)
                })?
            }
            optimize => Arc::new(
                resolve_einsum_strategy(optimize, &subs, &shape_refs).map_err(to_tenferro_error)?,
            ),
        };
        op = op.with_static_tree_hint(tree);
    } else {
        validate_symbolic_optimize(&optimize)?;
    }

    let outputs = extension::apply(Arc::new(op), inputs);
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("einsum extension produced no output".into()))
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
    key_data: &(EinsumSubscripts, Vec<Vec<usize>>),
    build: impl FnOnce() -> DeviceResult<ContractionTree>,
) -> Result<Arc<ContractionTree>> {
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_STATIC_PLANS_CACHE,
        hash_value(key_data),
    );
    if let Some(cached) = caches.get::<Arc<ContractionTree>>(&key) {
        return Ok(Arc::clone(cached));
    }

    let tree = Arc::new(build().map_err(to_tenferro_error)?);
    let retained_bytes = einsum_subscripts_retained_bytes(&key_data.0)
        + key_data
            .1
            .iter()
            .map(|shape| shape.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
        + tree.retained_bytes_for_cache_stats();
    caches.put(key, Arc::clone(&tree), retained_bytes);
    Ok(tree)
}

fn validate_symbolic_optimize(optimize: &EinsumOptimize) -> Result<()> {
    match optimize {
        EinsumOptimize::Auto(opts) if is_default_auto_options(opts) => Ok(()),
        _ => Err(Error::ContractionError(
            "symbolic einsum supports only default automatic optimization".into(),
        )),
    }
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

fn to_tenferro_error(error: DeviceError) -> Error {
    Error::ContractionError(error.to_string())
}

fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
