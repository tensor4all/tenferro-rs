use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::types::ValueRef;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::extension::{self, ExtensionCacheKey, ExtensionCacheStore};
use tenferro_runtime::{GraphCompiler, SymDim, TracedTensor};
use tenferro_tensor::{ShapeMismatch, ValidationError};

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
use crate::builder::build_einsum_graph_dim_expr;
use crate::cache::{
    einsum_subscripts_retained_bytes, saturating_sum, vec_retained_bytes, ParsedEinsum,
    EINSUM_EXTENSION_FAMILY_ID, EINSUM_PARSE_CACHE, EINSUM_STATIC_PLANS_CACHE,
};
use crate::extension::EinsumExtensionOp;
use crate::optimize::{
    hash_einsum_plan_spec, plan_spec_from_optimize, plan_specs_equal,
    resolve_einsum_strategy_with_spec, resolve_plan_spec, EinsumPlanSpec,
};
use crate::{
    parse_einsum_subscripts, ContractionTree, EinsumOptimize, EinsumSubscripts, Error, Result,
    Subscripts, TensorDotAxes,
};

/// Traced einsum extension methods for [`GraphCompiler`].
pub trait GraphCompilerEinsumExt {
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] with `InvalidArgument`, `RankMismatch`, or
    /// `ShapeMismatch` for input-count/rank/label inconsistencies,
    /// [`Error::Planning`] when no contraction plan is available, or
    /// [`Error::Runtime`] for graph-build/lowering failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic label-dimension equalities that are not decidable during
    /// graph construction are checked during compilation or execution and
    /// retain the runtime [`ErrorPhase`](tenferro_runtime::ErrorPhase).
    fn einsum(&mut self, inputs: &[&TracedTensor], subscripts: &str) -> Result<TracedTensor>;
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `InvalidArgument`, `RankMismatch`,
    /// or `ShapeMismatch` for input-count/rank/label inconsistencies,
    /// [`Error::Planning`] when no contraction plan is available, or
    /// [`Error::Runtime`] for graph-build/lowering failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic label-dimension equalities that are not decidable during
    /// graph construction are checked during compilation or execution and
    /// retain the runtime [`ErrorPhase`](tenferro_runtime::ErrorPhase).
    fn einsum_subscripts(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &EinsumSubscripts,
    ) -> Result<TracedTensor>;
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Planning`] when the requested strategy cannot be resolved,
    /// [`Error::Validation`] with `InvalidArgument`, `RankMismatch`, or
    /// `ShapeMismatch` for input/label inconsistencies, or [`Error::Runtime`]
    /// for graph-build/lowering failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic label-dimension equalities that are not decidable during
    /// graph construction are checked during compilation or execution and
    /// retain the runtime [`ErrorPhase`](tenferro_runtime::ErrorPhase).
    fn einsum_with(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &str,
        optimize: EinsumOptimize,
    ) -> Result<TracedTensor>;
    ///
    /// # Errors
    ///
    /// Returns [`Error::Planning`] when the requested strategy cannot be
    /// resolved, [`Error::Validation`] with `InvalidArgument`, `RankMismatch`,
    /// or `ShapeMismatch` for input/label inconsistencies, or [`Error::Runtime`]
    /// for graph-build/lowering failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic label-dimension equalities that are not decidable during
    /// graph construction are checked during compilation or execution and
    /// retain the runtime [`ErrorPhase`](tenferro_runtime::ErrorPhase).
    fn einsum_subscripts_with(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &EinsumSubscripts,
        optimize: EinsumOptimize,
    ) -> Result<TracedTensor>;
}

impl GraphCompilerEinsumExt for GraphCompiler {
    fn einsum(&mut self, inputs: &[&TracedTensor], subscripts: &str) -> Result<TracedTensor> {
        einsum(self, inputs, subscripts)
    }

    fn einsum_subscripts(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &EinsumSubscripts,
    ) -> Result<TracedTensor> {
        einsum_subscripts(self, inputs, subscripts)
    }

    fn einsum_with(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &str,
        optimize: EinsumOptimize,
    ) -> Result<TracedTensor> {
        einsum_with(self, inputs, subscripts, optimize)
    }

    fn einsum_subscripts_with(
        &mut self,
        inputs: &[&TracedTensor],
        subscripts: &EinsumSubscripts,
        optimize: EinsumOptimize,
    ) -> Result<TracedTensor> {
        einsum_subscripts_with(self, inputs, subscripts, optimize)
    }
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

/// N-ary einsum with default time-optimized automatic planning.
///
/// The default optimizer is resolved into a shape-independent plan
/// specification stored in the extension payload. That payload identity
/// participates in traced extension-op equality and in compile/runtime einsum
/// plan caches.
///
/// # Errors
///
/// Returns [`Error::InvalidSubscripts`] for malformed notation,
/// [`Error::Validation`] for input-count or symbolic shape mismatches, or
/// [`Error::Runtime`] for graph-build/lowering failures.
///
/// # Deferred errors
///
/// Symbolic shape constraints that cannot be decided during graph construction
/// are checked during compile or execution and retain their runtime phase.
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
///
/// # Errors
///
/// Returns [`Error::Validation`] for input-count or symbolic shape mismatches,
/// [`Error::Planning`] when no contraction plan can be built, or
/// [`Error::Runtime`] for graph-build/lowering failures.
///
/// # Deferred errors
///
/// Symbolic shape constraints that cannot be decided during graph construction
/// are checked during compile or execution and retain their runtime phase.
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
///
/// # Errors
///
/// Returns [`Error::InvalidSubscripts`] for malformed notation,
/// [`Error::Validation`] for input-count or symbolic shape mismatches,
/// [`Error::Planning`] when the strategy cannot be resolved, or
/// [`Error::Runtime`] for graph-build/lowering failures.
///
/// # Deferred errors
///
/// Symbolic shape constraints that cannot be decided during graph construction
/// are checked during compile or execution and retain their runtime phase.
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
///
/// # Errors
///
/// Returns [`Error::Validation`] for input-count or symbolic shape mismatches,
/// [`Error::Planning`] when the strategy cannot be resolved, or
/// [`Error::Runtime`] for graph-build/lowering failures.
///
/// # Deferred errors
///
/// Symbolic shape constraints that cannot be decided during graph construction
/// are checked during compile or execution and retain their runtime phase.
pub fn einsum_subscripts_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor> {
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

    let output_shape_hint = infer_symbolic_output_shape(subscripts, inputs)?;
    if let Some(result) = try_direct_binary_dot_general(inputs, subscripts, &optimize)? {
        let contract_op = EinsumExtensionOp::with_output_shape_hint(
            subscripts.clone(),
            output_shape_hint,
            EinsumPlanSpec::LeftToRight,
        );
        return extension::attach_expanded_shape_contract(&contract_op, inputs, result)
            .map_err(Error::Runtime);
    }

    let subs = Subscripts::from(subscripts);

    let (plan_spec, static_tree) = if let Some(shapes) = concrete_shapes(inputs) {
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let (plan_spec, tree) = match optimize {
            EinsumOptimize::Tree(tree) => {
                let (plan_spec, tree) = resolve_einsum_strategy_with_spec(
                    EinsumOptimize::Tree(tree),
                    &subs,
                    &shape_refs,
                )?;
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
                let plan_spec = plan_spec_from_optimize(optimize, &subs)?;
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
        let plan_spec = plan_spec_from_optimize(optimize, &subs)?;
        let tree = symbolic_fixed_path_tree(&plan_spec, &subs, inputs)?;
        (plan_spec, tree.map(Arc::new))
    };

    if let Some(tree) = static_tree {
        return expand_traced_einsum_graph(inputs, subscripts, tree.as_ref(), output_shape_hint);
    }

    let op =
        EinsumExtensionOp::with_output_shape_hint(subscripts.clone(), output_shape_hint, plan_spec);
    let outputs = extension::apply(Arc::new(op), inputs)?;
    outputs.into_iter().next().ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::Internal(
            "einsum extension produced no output".into(),
        ))
    })
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
    let input_dim_shapes = traced_dim_expr_shapes(inputs);

    let outputs =
        extension::apply_expanded_graph_with_shape_contract(&op, inputs, |builder, input_refs| {
            let result = build_einsum_graph_dim_expr(builder, tree, input_refs, &input_dim_shapes)
                .map_err(|error| {
                    tenferro_runtime::Error::extension(
                        "einsum",
                        tenferro_runtime::ErrorPhase::GraphBuild,
                        EINSUM_EXTENSION_FAMILY_ID,
                        error.kind(),
                        error,
                    )
                })?;
            let ValueRef::Local(local) = result else {
                return Err(tenferro_runtime::Error::Internal(
                    "expanded einsum returned an external value".into(),
                ));
            };
            Ok(vec![local])
        })?;

    outputs.into_iter().next().ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::Internal(
            "expanded einsum produced no output".into(),
        ))
    })
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
    resolve_plan_spec(plan_spec, subs, &shape_refs).map(Some)
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
        BinaryDotOperandOrder::Original => inputs[0].dot_general(inputs[1], plan.config)?,
        BinaryDotOperandOrder::Swapped => inputs[1].dot_general(inputs[0], plan.config)?,
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
                    return Err(Error::validation(
                        "einsum",
                        ShapeMismatch::ExpectedActual {
                            expected: tenferro_tensor::ShapeVec::from_vec(vec![existing]),
                            actual: tenferro_tensor::ShapeVec::from_vec(vec![dim]),
                        }
                        .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn optimize_allows_direct_binary_dot(optimize: &EinsumOptimize) -> Result<bool> {
    match optimize {
        EinsumOptimize::Auto(options) => {
            options.validate()?;
            Ok(true)
        }
        EinsumOptimize::False => Ok(true),
        EinsumOptimize::Tree(tree) => {
            Ok(tree.step_count() == 1 && matches!(tree.step_pair(0), Some((0, 1)) | Some((1, 0))))
        }
        EinsumOptimize::Nested(_) | EinsumOptimize::Path(_) => Ok(false),
    }
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

#[derive(Clone)]
struct StaticTreeCacheKeyData {
    subscripts: EinsumSubscripts,
    shapes: Vec<Vec<usize>>,
    plan_spec: EinsumPlanSpec,
}

impl StaticTreeCacheKeyData {
    fn new(
        subscripts: &EinsumSubscripts,
        shapes: &[Vec<usize>],
        plan_spec: &EinsumPlanSpec,
    ) -> Self {
        Self {
            subscripts: subscripts.clone(),
            shapes: shapes.to_vec(),
            plan_spec: plan_spec.clone(),
        }
    }

    fn matches_static_tree(
        &self,
        subscripts: &EinsumSubscripts,
        shapes: &[Vec<usize>],
        plan_spec: &EinsumPlanSpec,
    ) -> bool {
        self.subscripts == *subscripts
            && self.shapes.as_slice() == shapes
            && plan_specs_equal(&self.plan_spec, plan_spec)
    }

    fn retained_bytes(&self) -> usize {
        saturating_sum([
            einsum_subscripts_retained_bytes(&self.subscripts),
            saturating_sum(self.shapes.iter().map(vec_retained_bytes)),
            plan_spec_retained_bytes(&self.plan_spec),
        ])
    }
}

struct CachedStaticTree {
    key_data: StaticTreeCacheKeyData,
    tree: Arc<ContractionTree>,
}

fn cached_static_tree(
    caches: &mut ExtensionCacheStore,
    subscripts: &EinsumSubscripts,
    plan_spec: &EinsumPlanSpec,
    shapes: &[Vec<usize>],
    build: impl FnOnce() -> Result<ContractionTree>,
) -> Result<Arc<ContractionTree>> {
    let plan_hash = plan_spec_hash(plan_spec);
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_STATIC_PLANS_CACHE,
        static_tree_cache_discriminator(subscripts, shapes, plan_hash),
    );
    if let Some(cached) = caches.get::<CachedStaticTree>(&key) {
        let key_data = &cached.key_data;
        if key_data.matches_static_tree(subscripts, shapes, plan_spec) {
            return Ok(Arc::clone(&cached.tree));
        }
    }

    let tree = Arc::new(build()?);
    let key_data = StaticTreeCacheKeyData::new(subscripts, shapes, plan_spec);
    let retained_bytes = saturating_sum([
        key_data.retained_bytes(),
        tree.retained_bytes_for_cache_stats(),
    ]);
    caches.put(
        key,
        CachedStaticTree {
            key_data,
            tree: Arc::clone(&tree),
        },
        retained_bytes,
    );
    Ok(tree)
}

fn static_tree_cache_discriminator(
    subscripts: &EinsumSubscripts,
    shapes: &[Vec<usize>],
    plan_hash: u64,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    subscripts.hash(&mut hasher);
    shapes.hash(&mut hasher);
    plan_hash.hash(&mut hasher);
    hasher.finish()
}

fn plan_spec_hash(plan_spec: &EinsumPlanSpec) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_einsum_plan_spec(plan_spec, &mut hasher);
    hasher.finish()
}

fn plan_spec_retained_bytes(plan_spec: &EinsumPlanSpec) -> usize {
    match plan_spec {
        EinsumPlanSpec::Auto(options) => saturating_sum([
            std::mem::size_of::<EinsumPlanSpec>(),
            vec_retained_bytes(&options.betas),
        ]),
        EinsumPlanSpec::LeftToRight => std::mem::size_of::<EinsumPlanSpec>(),
        EinsumPlanSpec::Path(path) | EinsumPlanSpec::FixedPairs(path) => saturating_sum([
            std::mem::size_of::<EinsumPlanSpec>(),
            vec_retained_bytes(path),
        ]),
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
        let shape: Vec<_> = match tensor.sym_shape() {
            Some(shape) => shape.to_vec(),
            None => (0..tensor.rank)
                .map(|axis| tensor.axis_sym_dim(axis).map_err(Error::Runtime))
                .collect::<Result<_>>()?,
        };
        if labels.len() != shape.len() {
            return Err(Error::validation(
                "einsum",
                ValidationError::RankMismatch {
                    expected: labels.len(),
                    actual: shape.len(),
                },
            ));
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
                Error::invalid_argument(
                    "einsum",
                    "output",
                    format!("einsum output label {label} is missing from inputs"),
                )
            })
        })
        .collect()
}

fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests;
