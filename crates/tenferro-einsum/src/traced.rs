use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::extension::{ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::program::ProgramBuildError;
use tenferro_runtime::{TraceContext, TraceValue, TracedTensor};
use tenferro_tensor::{ErrorKind, ValidationKind};

use crate::cache::{
    einsum_notation_retained_bytes, saturating_sum, ParsedEinsum, EINSUM_EXTENSION_FAMILY_ID,
    EINSUM_PARSE_CACHE,
};
use crate::ellipsis::{notation_without_ellipsis, resolve_einsum_notation};
use crate::extension::EinsumExtensionOp;
use crate::optimize::{plan_spec_from_optimize, resolve_einsum_strategy_with_spec};
use crate::{
    parse_einsum_notation, EinsumAxis, EinsumNotation, EinsumOptimize, EinsumSubscripts, Error,
    Result, Subscripts, TensorDotAxes,
};

/// Traced einsum of a #2552 [`Spec`](crate::Spec) on trace values; the same
/// recording as the former `TraceContext` entries.
pub(crate) fn trace_einsum_spec(
    trace: &TraceContext,
    inputs: &[TraceValue],
    spec: crate::Spec<'_>,
    optimize: EinsumOptimize,
) -> Result<TraceValue> {
    let mut trace = trace.clone();
    match spec {
        crate::Spec::Text(text) => {
            let parsed = trace
                .with_extension_caches(|caches| cached_subscripts(caches, text))
                .map_err(semantic_trace_error)??;
            trace_context_einsum_notation_with(&mut trace, inputs, &parsed.notation, optimize)
        }
        crate::Spec::Notation(notation) => {
            trace_context_einsum_notation_with(&mut trace, inputs, notation, optimize)
        }
        crate::Spec::Subscripts(subscripts) => {
            trace_context_einsum_subscripts_with(&mut trace, inputs, subscripts, optimize, false)
        }
    }
}

fn trace_context_einsum_notation_with(
    trace: &mut TraceContext,
    inputs: &[TraceValue],
    notation: &EinsumNotation,
    optimize: EinsumOptimize,
) -> Result<TraceValue> {
    let ellipsis = notation
        .inputs
        .iter()
        .chain(std::iter::once(&notation.output))
        .any(|term| term.contains(&EinsumAxis::Ellipsis));
    let subscripts = if ellipsis {
        let shapes = trace_concrete_shapes(trace, inputs)?.ok_or_else(|| {
            Error::planning("ellipsis einsum requires concrete input dimensions during tracing")
        })?;
        let shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
        resolve_einsum_notation(notation, &shape_refs)?.into()
    } else {
        notation_without_ellipsis(notation)?.into()
    };
    trace_context_einsum_subscripts_with(trace, inputs, &subscripts, optimize, ellipsis)
}

fn trace_context_einsum_subscripts_with(
    trace: &mut TraceContext,
    inputs: &[TraceValue],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
    ellipsis: bool,
) -> Result<TraceValue> {
    if inputs.is_empty() {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            "einsum requires at least one input tensor",
        ));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            format!(
                "einsum subscripts expect {} inputs, got {}",
                subscripts.inputs.len(),
                inputs.len()
            ),
        ));
    }

    let subs = Subscripts::from(subscripts);
    let allow_broadcast = ellipsis || trace_requires_broadcast(trace, inputs, subscripts)?;
    let op = match optimize {
        EinsumOptimize::Tree(tree) => {
            let shapes = trace_concrete_shapes(trace, inputs)?.ok_or_else(|| {
                Error::planning("precomputed contraction tree requires concrete input shapes")
            })?;
            let shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
            let (plan_spec, _tree) =
                resolve_einsum_strategy_with_spec(EinsumOptimize::Tree(tree), &subs, &shape_refs)?;
            EinsumExtensionOp::with_plan_spec_and_broadcast(
                subscripts.clone(),
                plan_spec,
                allow_broadcast,
            )
        }
        optimize => EinsumExtensionOp::with_plan_spec_and_broadcast(
            subscripts.clone(),
            plan_spec_from_optimize(optimize, &subs)?,
            allow_broadcast,
        ),
    };
    let outputs = trace
        .add_extension(Arc::new(op), inputs)
        .map_err(semantic_trace_error)?;
    outputs.first().copied().ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::Internal(
            "einsum semantic extension produced no output".into(),
        ))
    })
}

fn trace_requires_broadcast(
    trace: &TraceContext,
    inputs: &[TraceValue],
    subscripts: &EinsumSubscripts,
) -> Result<bool> {
    // A symbolic extent is skipped rather than guessed, so a partially symbolic
    // trace still decides every pair of extents it does know.
    let shapes = trace_known_shapes(trace, inputs)?;
    let shape_refs: Vec<&[Option<usize>]> = shapes.iter().map(Vec::as_slice).collect();
    Ok(crate::broadcast::requires_broadcast(
        &shape_refs,
        &subscripts.inputs,
    ))
}

/// Per-axis input extents that are constant at trace time; `None` is symbolic.
fn trace_known_shapes(
    trace: &TraceContext,
    inputs: &[TraceValue],
) -> Result<Vec<Vec<Option<usize>>>> {
    let mut shapes = Vec::with_capacity(inputs.len());
    for &value in inputs {
        let metadata = trace.value_metadata(value).map_err(semantic_trace_error)?;
        shapes.push(
            metadata
                .shape()
                .iter()
                .map(|extent| match extent.as_exact() {
                    Some(DimExpr::Const(value)) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        );
    }
    Ok(shapes)
}

fn trace_concrete_shapes(
    trace: &TraceContext,
    inputs: &[TraceValue],
) -> Result<Option<Vec<Vec<usize>>>> {
    Ok(trace_known_shapes(trace, inputs)?
        .into_iter()
        .map(|shape| shape.into_iter().collect::<Option<Vec<_>>>())
        .collect())
}

fn semantic_trace_error(source: ProgramBuildError) -> Error {
    Error::Runtime(tenferro_runtime::Error::extension(
        "einsum",
        tenferro_runtime::ErrorPhase::GraphBuild,
        EINSUM_EXTENSION_FAMILY_ID,
        ErrorKind::Validation(ValidationKind::InvalidArgument),
        source,
    ))
}

pub(crate) fn tensordot(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    axes: TensorDotAxes<'_>,
) -> Result<TracedTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.rank(), rhs.rank())?;
    crate::tensordot::validate_traced_contract_dims(lhs, rhs, &config)?;
    lhs.dot_general(rhs, config).map_err(Error::Runtime)
}

struct ParsedEinsumCacheEntry {
    notation: String,
    parsed: Arc<ParsedEinsum>,
}

impl ParsedEinsumCacheEntry {
    fn matches_notation(&self, notation: &str) -> bool {
        self.notation == notation
    }
}

fn cached_subscripts(
    caches: &mut ExtensionCacheStore,
    notation: &str,
) -> Result<Arc<ParsedEinsum>> {
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_PARSE_CACHE,
        hash_value(notation),
    );
    if let Some(cached) = caches.get::<ParsedEinsumCacheEntry>(&key) {
        if cached.matches_notation(notation) {
            return Ok(Arc::clone(&cached.parsed));
        }
    }

    let parsed = Arc::new(ParsedEinsum {
        notation: parse_einsum_notation(notation)?,
    });
    let entry = ParsedEinsumCacheEntry {
        notation: notation.to_owned(),
        parsed: Arc::clone(&parsed),
    };
    let retained_bytes = saturating_sum([
        entry.notation.len(),
        einsum_notation_retained_bytes(&parsed.notation),
    ]);
    caches.put(key, entry, retained_bytes);
    Ok(parsed)
}

fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests;
