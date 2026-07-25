use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::extension::{ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::program::ProgramBuildError;
use tenferro_runtime::{TraceContext, TraceValue, TracedTensor};
use tenferro_tensor::{ErrorKind, ValidationKind};

use crate::cache::{
    einsum_subscripts_retained_bytes, saturating_sum, ParsedEinsum, EINSUM_EXTENSION_FAMILY_ID,
    EINSUM_PARSE_CACHE,
};
use crate::extension::EinsumExtensionOp;
use crate::optimize::{plan_spec_from_optimize, resolve_einsum_strategy_with_spec};
use crate::{
    parse_einsum_subscripts, EinsumOptimize, EinsumSubscripts, Error, Result, Subscripts,
    TensorDotAxes,
};

/// Backend-neutral einsum tracing methods for [`TraceContext`].
///
/// Each method records one semantic einsum extension operation. Contraction
/// path materialization and provider selection remain compiler/runtime work.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::TraceContextEinsumExt;
/// use tenferro_ops::dim_expr::DimExpr;
/// use tenferro_runtime::program::ProgramInputSpec;
/// use tenferro_runtime::TraceContext;
/// use tenferro_tensor::DType;
///
/// let matrix = || {
///     ProgramInputSpec::new(
///         DType::F64,
///         [DimExpr::Const(2), DimExpr::Const(2)],
///     )
/// };
/// let mut trace = TraceContext::new();
/// let lhs = trace.input(matrix()).unwrap();
/// let rhs = trace.input(matrix()).unwrap();
/// let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
/// let graph = trace.finish(&[output]).unwrap();
/// assert_eq!(graph.program().operations().count(), 1);
/// ```
pub trait TraceContextEinsumExt {
    /// Trace textual einsum notation using the default optimizer policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for invalid input metadata, [`Error::Planning`]
    /// for an invalid optimizer policy, or [`Error::Runtime`] when semantic
    /// program construction fails.
    fn einsum(&mut self, inputs: &[TraceValue], subscripts: &str) -> Result<TraceValue>;

    /// Trace parsed einsum notation using the default optimizer policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid input metadata,
    /// [`Error::Planning`] for an invalid optimizer policy, or
    /// [`Error::Runtime`] when semantic program construction fails.
    fn einsum_subscripts(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &EinsumSubscripts,
    ) -> Result<TraceValue>;

    /// Trace textual einsum notation with an explicit optimizer policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for invalid input metadata, [`Error::Planning`]
    /// for an invalid optimizer policy, or [`Error::Runtime`] when semantic
    /// program construction fails.
    fn einsum_with(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &str,
        optimize: EinsumOptimize,
    ) -> Result<TraceValue>;

    /// Trace parsed einsum notation with an explicit optimizer policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid input metadata,
    /// [`Error::Planning`] for an invalid optimizer policy, or
    /// [`Error::Runtime`] when semantic program construction fails.
    fn einsum_subscripts_with(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &EinsumSubscripts,
        optimize: EinsumOptimize,
    ) -> Result<TraceValue>;
}

impl TraceContextEinsumExt for TraceContext {
    fn einsum(&mut self, inputs: &[TraceValue], subscripts: &str) -> Result<TraceValue> {
        self.einsum_with(inputs, subscripts, EinsumOptimize::default())
    }

    fn einsum_subscripts(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &EinsumSubscripts,
    ) -> Result<TraceValue> {
        self.einsum_subscripts_with(inputs, subscripts, EinsumOptimize::default())
    }

    fn einsum_with(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &str,
        optimize: EinsumOptimize,
    ) -> Result<TraceValue> {
        let parsed = cached_subscripts(self.extension_caches_mut(), subscripts)?;
        self.einsum_subscripts_with(inputs, &parsed.subscripts, optimize)
    }

    fn einsum_subscripts_with(
        &mut self,
        inputs: &[TraceValue],
        subscripts: &EinsumSubscripts,
        optimize: EinsumOptimize,
    ) -> Result<TraceValue> {
        trace_context_einsum_subscripts_with(self, inputs, subscripts, optimize)
    }
}

fn trace_context_einsum_subscripts_with(
    trace: &mut TraceContext,
    inputs: &[TraceValue],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
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
    let op = match optimize {
        EinsumOptimize::Tree(tree) => {
            let shapes = trace_concrete_shapes(trace, inputs)?.ok_or_else(|| {
                Error::planning("precomputed contraction tree requires concrete input shapes")
            })?;
            let shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
            let (plan_spec, _tree) =
                resolve_einsum_strategy_with_spec(EinsumOptimize::Tree(tree), &subs, &shape_refs)?;
            EinsumExtensionOp::with_plan_spec(subscripts.clone(), plan_spec)
        }
        optimize => EinsumExtensionOp::with_plan_spec(
            subscripts.clone(),
            plan_spec_from_optimize(optimize, &subs)?,
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

fn trace_concrete_shapes(
    trace: &TraceContext,
    inputs: &[TraceValue],
) -> Result<Option<Vec<Vec<usize>>>> {
    let mut shapes = Vec::with_capacity(inputs.len());
    for &value in inputs {
        let metadata = trace.value_metadata(value).map_err(semantic_trace_error)?;
        let Some(shape) = metadata
            .shape()
            .iter()
            .map(|extent| match extent.as_exact() {
                Some(DimExpr::Const(value)) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        shapes.push(shape);
    }
    Ok(Some(shapes))
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

/// Traced tensor contraction-sugar methods.
pub trait TracedTensorEinsumExt {
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with rank, axis, duplicate-axis, or
    /// contracted-dimension payloads for invalid axes, or [`Error::Runtime`]
    /// for graph-build failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic contracted-dimension equalities are checked during compilation
    /// or execution and retain the runtime
    /// [`ErrorPhase`](tenferro_runtime::ErrorPhase).
    fn tensordot(&self, rhs: &TracedTensor, axes: TensorDotAxes<'_>) -> Result<TracedTensor>;
}

impl TracedTensorEinsumExt for TracedTensor {
    fn tensordot(&self, rhs: &TracedTensor, axes: TensorDotAxes<'_>) -> Result<TracedTensor> {
        tensordot(self, rhs, axes)
    }
}

fn tensordot(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    axes: TensorDotAxes<'_>,
) -> Result<TracedTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.rank, rhs.rank)?;
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
        subscripts: parse_einsum_subscripts(notation)?,
    });
    let entry = ParsedEinsumCacheEntry {
        notation: notation.to_owned(),
        parsed: Arc::clone(&parsed),
    };
    let retained_bytes = saturating_sum([
        entry.notation.len(),
        einsum_subscripts_retained_bytes(&parsed.subscripts),
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
