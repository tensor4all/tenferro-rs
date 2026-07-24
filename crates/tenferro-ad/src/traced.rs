use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::ad_rule_error::{ad_rule_error, ad_rule_error_with_context};
use computegraph::graph::Graph;
use computegraph::resolve::resolve;
use computegraph::resolve::{ResolvedView, ValueDef};
use computegraph::types::ValueKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ExtensionAdDispatcher, ShapeGuardContext};
use tenferro_runtime::ad_support::{
    checkpoint_chain as tensor_checkpoint_chain, checkpoint_tensor,
    extra_roots as tensor_extra_roots, inputs_map as tensor_inputs_map, leaf_input_key,
    linear_input_key, metadata_scopes as tensor_metadata_scopes, metadata_scopes_with_new,
    ones_tensor, push_metadata_scope, register_scoped_graph_analysis, registered_meta,
    resolve_roots as tensor_resolve_roots, shape_hint as tensor_shape_hint, tensor_from_parts,
    tensor_meta_from_tensor, ConstraintScopeTransfer, RegisteredGraphAnalysis, TracedTensorParts,
};
use tenferro_runtime::{Error, ErrorPhase, GraphCompiler, GraphExecutor, Result, TracedTensor};
use tenferro_tensor::TensorBackend;
use tidu::{linear_transpose, linearize, ADRuleError};

#[path = "traced/optimizer.rs"]
mod optimizer;
#[path = "traced/primal_transpose.rs"]
mod primal_transpose;

use optimizer::OptimizedLinearGraph;
use primal_transpose::{try_primal_transpose, PrimalTransposeGraph};

use crate::transform_cache::{
    AdTransformCache, CachedTracedVjpTransform, TracedAdTransformCacheKey, TracedAdTransformKind,
};

static NEXT_DIFF_PASS_ID: AtomicU64 = AtomicU64::new(0);

fn next_pass_id() -> u64 {
    NEXT_DIFF_PASS_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn next_input_key() -> TensorInputKey {
    tenferro_runtime::ad_support::allocate_input_key()
}

fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .try_concrete_shape()
        .unwrap_or_else(|| vec![0; tensor.rank])
}

fn shape_guard_context(
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    active_values: Option<Arc<HashSet<ValueKey<StdTensorOp>>>>,
    roots: &[Arc<Graph<StdTensorOp>>],
) -> ShapeGuardContext {
    let mut ctx = ShapeGuardContext::with_global_metadata();
    register_shape_sources(&mut ctx, roots);
    let ctx = match extension_ad_dispatcher {
        Some(dispatcher) => ctx.with_extension_ad_dispatcher(Arc::clone(dispatcher)),
        None => ctx,
    };
    match active_values {
        Some(keys) => ctx.with_linearize_active_values(keys),
        None => ctx,
    }
}

fn register_shape_sources(ctx: &mut ShapeGuardContext, roots: &[Arc<Graph<StdTensorOp>>]) {
    let mut seen = HashSet::new();
    for graph in roots {
        register_graph_shape_sources(ctx, graph, &mut seen);
    }
}

fn register_graph_shape_sources(
    ctx: &mut ShapeGuardContext,
    graph: &Arc<Graph<StdTensorOp>>,
    seen: &mut HashSet<*const Graph<StdTensorOp>>,
) {
    if !seen.insert(Arc::as_ptr(graph)) {
        return;
    }
    for parent in graph.parents() {
        register_graph_shape_sources(ctx, parent, seen);
    }
    for &input_id in graph.inputs() {
        let key = graph.values()[input_id].key.clone();
        let Ok(meta) = registered_meta(&key) else {
            continue;
        };
        let Some(shape) = meta.bound_shape() else {
            continue;
        };
        for tensor_id in shape
            .iter()
            .flat_map(|dim| dim.referenced_tensor_ids().into_iter())
        {
            ctx.insert_shape_source(tensor_id, key.clone());
        }
    }
}

fn linearize_active_value_keys(
    view: &ResolvedView<StdTensorOp>,
    outputs: &[ValueKey<StdTensorOp>],
    aliases: &std::collections::HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
) -> Arc<HashSet<ValueKey<StdTensorOp>>> {
    let mut active = HashSet::new();
    let mut stack: Vec<ValueKey<StdTensorOp>> = outputs.to_vec();
    while let Some(key) = stack.pop() {
        if !active.insert(key.clone()) {
            continue;
        }
        let Some(val_def) = view.resolve_value(&key) else {
            continue;
        };
        match val_def {
            ValueDef::Produced { input_keys, .. } => {
                for input_key in input_keys {
                    stack.push(input_key.clone());
                }
            }
            ValueDef::Input { key: input_key } => {
                if let Some(aliased) = aliases.get(&input_key) {
                    stack.push(aliased.clone());
                }
            }
        }
    }
    Arc::new(active)
}

fn graph_has_registered_primal_vjp(
    view: &ResolvedView<StdTensorOp>,
    outputs: &[ValueKey<StdTensorOp>],
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
) -> bool {
    let Some(extension_ad_dispatcher) = extension_ad_dispatcher else {
        return false;
    };
    let mut seen = HashSet::new();
    let mut stack = outputs.to_vec();
    while let Some(key) = stack.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        if let ValueKey::Derived { operation, .. } = &key {
            if let StdTensorOp::Extension(ext) = operation.operation() {
                if extension_ad_dispatcher.has_primal_vjp(ext.family_id()) {
                    return true;
                }
            }
        }
        let Some(val_def) = view.resolve_value(&key) else {
            continue;
        };
        match val_def {
            ValueDef::Produced { input_keys, .. } => {
                for input_key in input_keys {
                    stack.push(input_key);
                }
            }
            ValueDef::Input { key: input_key } => {
                if let Some(aliased) = aliases.get(&input_key) {
                    stack.push(aliased.clone());
                }
            }
        }
    }
    false
}

fn is_not_applicable_custom_vjp(err: &ADRuleError) -> bool {
    matches!(err, ADRuleError::Unsupported { .. })
}

pub(crate) fn grad_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    grad_with_optional_rules(output, wrt, extension_ad_dispatcher, ad_transform_cache)
}

pub(crate) fn jvp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    jvp_optional_impl(
        output,
        wrt,
        tangent,
        extension_ad_dispatcher,
        ad_transform_cache,
    )?
    .ok_or_else(|| Error::Internal(format!("jvp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn grad_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    if output.rank != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }

    let ones = ones_tensor(output.dtype, vec![])?;
    let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
    vjp_optional_impl(
        output,
        wrt,
        &seed,
        extension_ad_dispatcher,
        "grad",
        ad_transform_cache,
    )
}

pub(crate) fn jvp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    jvp_optional_impl(
        output,
        wrt,
        tangent,
        extension_ad_dispatcher,
        ad_transform_cache,
    )
}

pub(crate) fn vjp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    vjp_optional_impl(
        output,
        wrt,
        cotangent,
        extension_ad_dispatcher,
        "vjp",
        ad_transform_cache,
    )?
    .ok_or_else(|| Error::Internal(format!("vjp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn vjp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    vjp_optional_impl(
        output,
        wrt,
        cotangent,
        extension_ad_dispatcher,
        "vjp",
        ad_transform_cache,
    )
}

fn grad_with_optional_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    if output.rank != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }

    let ones = ones_tensor(output.dtype, vec![])?;
    let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
    let wrt_input_key = leaf_input_key(wrt)?;
    vjp_optional_impl(
        output,
        wrt,
        &seed,
        extension_ad_dispatcher,
        "grad",
        ad_transform_cache,
    )?
    .ok_or_else(|| Error::Internal(format!("grad output is inactive for {:?}", wrt_input_key)))
}

/// Automatic differentiation helpers for [`TracedTensor`].
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::TracedTensorAdExt;
/// use tenferro_runtime::TracedTensor;
///
/// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
/// let loss = x.scale_real(2.0).unwrap();
/// let maybe_dx = loss.grad_optional(&x).unwrap();
/// assert!(maybe_dx.is_some());
/// ```
pub trait TracedTensorAdExt {
    /// Gradient of a scalar output with respect to a traced input.
    ///
    /// For complex scalar outputs, tenferro returns the Hermitian-adjoint
    /// cotangent. To compare seed-`1` scalar gradients with JAX's public
    /// `grad` values, use the complex conjugate of this result. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut executor = GraphExecutor::new(CpuBackend::new());
    ///     executor.run(&program).unwrap()
    /// }
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let loss = (&x * &x).unwrap();
    /// let dx = loss.grad(&x).unwrap();
    ///
    /// assert_eq!(eval(&dx).as_slice::<f64>().unwrap(), &[6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] for a non-scalar
    /// output, [`tenferro_runtime::Error::UnsupportedAdRule`] when an AD rule
    /// is unavailable, or a typed validation/backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor>;

    /// Like [`grad`](Self::grad), but returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let loss = (&y * &y).unwrap();
    ///
    /// assert!(loss.grad_optional(&x).unwrap().is_none());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] for a non-scalar
    /// output, [`Error::UnsupportedAdRule`] when an AD rule is unavailable, or
    /// a typed validation/backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>>;

    /// Evaluate this tensor and replace its graph with a concrete leaf while
    /// preserving the previous graph for AD replay.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let mut executor = GraphExecutor::new(CpuBackend::new());
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let mut y = (&x * &x).unwrap();
    ///
    /// y.checkpoint(&mut compiler, &mut executor).unwrap();
    ///
    /// let value = y.attached_data().unwrap();
    /// assert_eq!(value.as_slice::<f64>().unwrap(), &[9.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] when checkpoint metadata
    /// is invalid, [`Error::RuntimeState`] when graph metadata or executor
    /// state is unavailable, or a typed backend error from evaluation.
    fn checkpoint<B: TensorBackend>(
        &mut self,
        compiler: &mut GraphCompiler,
        executor: &mut GraphExecutor<B>,
    ) -> Result<()>;

    /// Forward-mode Jacobian-vector product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut executor = GraphExecutor::new(CpuBackend::new());
    ///     executor.run(&program).unwrap()
    /// }
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let tangent = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// let dy = y.jvp(&x, &tangent).unwrap();
    ///
    /// assert_eq!(eval(&dy).as_slice::<f64>().unwrap(), &[12.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when a JVP rule
    /// is unavailable, [`Error::Validation`] for incompatible tangent metadata,
    /// or a typed backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Result<TracedTensor>;

    /// Like [`jvp`](Self::jvp), but returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let loss = (&y * &y).unwrap();
    ///
    /// assert!(loss.jvp_optional(&x, &tangent).unwrap().is_none());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when a JVP rule
    /// is unavailable, [`Error::Validation`] for incompatible tangent metadata,
    /// or a typed backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn jvp_optional(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>>;

    /// Reverse-mode vector-Jacobian product.
    ///
    /// Complex cotangents use tenferro's Hermitian real-inner-product
    /// convention. Non-real complex cotangent seeds therefore need an explicit
    /// seed-convention comparison when matching JAX. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut executor = GraphExecutor::new(CpuBackend::new());
    ///     executor.run(&program).unwrap()
    /// }
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let cotangent = TracedTensor::from_vec_col_major(vec![], vec![0.5_f64]).unwrap();
    /// let y = (&x * &x).unwrap();
    /// let dx = y.vjp(&x, &cotangent).unwrap();
    ///
    /// assert_eq!(eval(&dx).as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when a VJP rule
    /// is unavailable, [`Error::Validation`] for incompatible cotangent
    /// metadata, or a typed backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Result<TracedTensor>;

    /// Like [`vjp`](Self::vjp), but returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let loss = (&y * &y).unwrap();
    ///
    /// assert!(loss.vjp_optional(&x, &cotangent).unwrap().is_none());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::UnsupportedAdRule`] when a VJP rule
    /// is unavailable, [`Error::Validation`] for incompatible cotangent
    /// metadata, or a typed backend/runtime-state error.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints can later produce
    /// [`tenferro_runtime::Error::ShapeConstraintViolation`] or
    /// [`tenferro_runtime::Error::ShapeConstraintEvaluation`] during compile
    /// or execution.
    fn vjp_optional(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>>;
}

impl TracedTensorAdExt for TracedTensor {
    fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor> {
        grad_with_optional_rules(self, wrt, None, None)
    }

    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![])?;
        let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
        vjp_optional_impl(self, wrt, &seed, None, "grad", None)
    }

    fn checkpoint<B: TensorBackend>(
        &mut self,
        compiler: &mut GraphCompiler,
        executor: &mut GraphExecutor<B>,
    ) -> Result<()> {
        let data = if let Some(data) = self.attached_data() {
            Arc::clone(data)
        } else {
            let program = compiler.compile(self)?;
            Arc::new(executor.run(&program)?)
        };
        checkpoint_tensor(self, data)?;
        Ok(())
    }

    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Result<TracedTensor> {
        let wrt_input_key = leaf_input_key(wrt)?;
        self.jvp_optional(wrt, tangent)?.ok_or_else(|| {
            Error::Internal(format!("jvp output is inactive for {:?}", wrt_input_key))
        })
    }

    fn jvp_optional(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        jvp_optional_impl(self, wrt, tangent, None, None)
    }

    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Result<TracedTensor> {
        let wrt_input_key = leaf_input_key(wrt)?;
        self.vjp_optional(wrt, cotangent)?.ok_or_else(|| {
            Error::Internal(format!("vjp output is inactive for {:?}", wrt_input_key))
        })
    }

    fn vjp_optional(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        vjp_optional_impl(self, wrt, cotangent, None, "vjp", None)
    }
}

fn jvp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt)?;
    let output_key = output.graph().values()[output.val].key.clone();
    let checkpoint_chain = tensor_checkpoint_chain(output);
    let aliases = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_aliases())
        .unwrap_or_default();
    let checkpoint_graphs = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_graphs())
        .unwrap_or_default();
    let mut roots = tensor_resolve_roots(output);
    roots.extend(checkpoint_graphs.iter().cloned());
    let view = resolve(roots);
    let active_values =
        linearize_active_value_keys(&view, std::slice::from_ref(&output_key), &aliases);
    let cache_key = ad_transform_cache.map(|_| {
        TracedAdTransformCacheKey::new(
            TracedAdTransformKind::Jvp,
            &view.roots,
            &output_key,
            &wrt_input_key,
            &aliases,
        )
    });
    let linear = match (ad_transform_cache, cache_key) {
        (Some(cache), Some(key)) => {
            if let Some(linear) = cache.get_traced_linearized(&key)? {
                linear
            } else {
                let mut ad_ctx =
                    shape_guard_context(extension_ad_dispatcher, Some(active_values), &view.roots);
                let linear = linearize(
                    &view,
                    std::slice::from_ref(&output_key),
                    std::slice::from_ref(&wrt_input_key),
                    next_pass_id(),
                    &mut ad_ctx,
                    &aliases,
                )
                .map_err(|err| ad_rule_error_with_context("jvp", err, &mut ad_ctx))?;
                let linear = Arc::new(OptimizedLinearGraph::from_tidu(linear).into_cached());
                cache.put_traced_linearized(key, Arc::clone(&linear))?;
                linear
            }
        }
        _ => {
            let mut ad_ctx =
                shape_guard_context(extension_ad_dispatcher, Some(active_values), &view.roots);
            let linear = linearize(
                &view,
                std::slice::from_ref(&output_key),
                std::slice::from_ref(&wrt_input_key),
                next_pass_id(),
                &mut ad_ctx,
                &aliases,
            )
            .map_err(|err| ad_rule_error_with_context("jvp", err, &mut ad_ctx))?;
            Arc::new(OptimizedLinearGraph::from_tidu(linear).into_cached())
        }
    };
    let Some(tangent_output) = linear.tangent_outputs()[0] else {
        return Ok(None);
    };
    let tangent_input_key = linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1)?;
    let tangent_data = tangent.attached_data().cloned().ok_or_else(|| {
        Error::invalid_argument(
            "jvp",
            ErrorPhase::GraphBuild,
            "tangent",
            "jvp tangent must have concrete tensor data",
        )
    })?;
    let analysis = register_scoped_graph_analysis(
        linear.as_graph(),
        vec![(
            ValueKey::Input(tangent_input_key.clone()),
            tensor_meta_from_tensor(tangent_data.as_ref()),
        )],
    )?;

    let mut inputs_map = (*tensor_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(tangent_input_key, tangent_data);

    let mut extra_roots = vec![Arc::clone(output.graph())];
    extra_roots.extend(checkpoint_graphs);
    extra_roots.extend(tensor_extra_roots(output));
    let inherited_constraint_scopes = [
        ConstraintScopeTransfer::from_tensor(output),
        ConstraintScopeTransfer::from_tensor(wrt),
        ConstraintScopeTransfer::from_tensor(tangent),
    ];

    Ok(Some(tensor_from_parts(TracedTensorParts {
        rank: output.rank,
        dtype: output.dtype,
        graph: Arc::clone(linear.graph()),
        val: tangent_output,
        data: None,
        shape_hint: tensor_shape_hint(output),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: metadata_scopes_with_new(
            analysis.metadata,
            [
                tensor_metadata_scopes(output),
                tensor_metadata_scopes(wrt),
                tensor_metadata_scopes(tangent),
            ],
        ),
        constraint_scope_transfer: ConstraintScopeTransfer::with_new(
            analysis.constraints,
            inherited_constraint_scopes.iter(),
        ),
    })))
}

enum VjpTransposeGraph {
    Primal(PrimalTransposeGraph),
    Linear(Arc<CachedTracedVjpTransform>),
}

struct ActiveLinearVjp {
    transposed: Arc<CachedTracedVjpTransform>,
    residual_analysis: RegisteredGraphAnalysis,
}

impl VjpTransposeGraph {
    fn as_graph(&self) -> &computegraph::graph::Graph<StdTensorOp> {
        match self {
            Self::Primal(graph) => graph.as_graph(),
            Self::Linear(graph) => graph.transposed().as_graph(),
        }
    }

    fn tangent_inputs(&self) -> &[(TensorInputKey, computegraph::LocalValueId)] {
        match self {
            Self::Primal(graph) => graph.tangent_inputs(),
            Self::Linear(graph) => graph.transposed().tangent_inputs(),
        }
    }

    fn tangent_outputs(&self) -> &[Option<computegraph::LocalValueId>] {
        match self {
            Self::Primal(graph) => graph.tangent_outputs(),
            Self::Linear(graph) => graph.transposed().tangent_outputs(),
        }
    }

    fn into_graph_arc(self) -> Arc<computegraph::graph::Graph<StdTensorOp>> {
        match self {
            Self::Primal(graph) => Arc::new(graph.into_graph()),
            Self::Linear(graph) => Arc::clone(graph.transposed().graph()),
        }
    }
}

fn compute_linear_vjp_transform(
    view: &ResolvedView<StdTensorOp>,
    output_key: &ValueKey<StdTensorOp>,
    wrt_input_key: &TensorInputKey,
    aliases: &HashMap<TensorInputKey, ValueKey<StdTensorOp>>,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    active_values: Arc<HashSet<ValueKey<StdTensorOp>>>,
    wrt: &TracedTensor,
) -> Result<tidu::ADRuleResult<Option<ActiveLinearVjp>>> {
    let mut linear_ad_ctx =
        shape_guard_context(extension_ad_dispatcher, Some(active_values), &view.roots);
    let linear = match linearize(
        view,
        std::slice::from_ref(output_key),
        std::slice::from_ref(wrt_input_key),
        next_pass_id(),
        &mut linear_ad_ctx,
        aliases,
    ) {
        Ok(linear) => linear,
        Err(err) => {
            if let Some(source) = linear_ad_ctx.take_deferred_shape_error() {
                return Err(Error::ad_rule_source("vjp", source));
            }
            return Ok(Err(err));
        }
    };
    if linear.tangent_outputs()[0].is_none() {
        return Ok(Ok(None));
    }

    let linear_seed_key = linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1)?;
    let linear_analysis = register_scoped_graph_analysis(
        linear.as_graph(),
        vec![(
            ValueKey::Input(linear_seed_key),
            registered_meta(&wrt.graph().values()[wrt.val].key)?,
        )],
    )?;
    linear_ad_ctx.refresh_global_metadata();
    let transposed = match linear_transpose(&linear, &mut linear_ad_ctx) {
        Ok(transposed) => OptimizedLinearGraph::from_tidu(transposed).into_cached(),
        Err(err) => {
            if let Some(source) = linear_ad_ctx.take_deferred_shape_error() {
                return Err(Error::ad_rule_source("vjp", source));
            }
            return Ok(Err(err));
        }
    };
    let (_linear_graph, residual_graph) = linear.into_graphs();
    let residual_analysis =
        register_scoped_graph_analysis(residual_graph.as_ref(), std::iter::empty())?;
    drop(linear_analysis);
    Ok(Ok(Some(ActiveLinearVjp {
        transposed: Arc::new(CachedTracedVjpTransform::new(residual_graph, transposed)),
        residual_analysis,
    })))
}

fn vjp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_ad_dispatcher: Option<&Arc<dyn ExtensionAdDispatcher>>,
    transform: &'static str,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt)?;
    let output_key = output.graph().values()[output.val].key.clone();
    let checkpoint_chain = tensor_checkpoint_chain(output);
    let aliases = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_aliases())
        .unwrap_or_default();
    let checkpoint_graphs = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_graphs())
        .unwrap_or_default();
    let mut roots = tensor_resolve_roots(output);
    roots.extend(checkpoint_graphs.iter().cloned());
    let view = resolve(roots);

    let active_values =
        linearize_active_value_keys(&view, std::slice::from_ref(&output_key), &aliases);
    let cache_key = ad_transform_cache.map(|_| {
        TracedAdTransformCacheKey::new(
            TracedAdTransformKind::Vjp,
            &view.roots,
            &output_key,
            &wrt_input_key,
            &aliases,
        )
    });
    if graph_has_registered_primal_vjp(
        &view,
        std::slice::from_ref(&output_key),
        &aliases,
        extension_ad_dispatcher,
    ) {
        let mut primal_ad_ctx = shape_guard_context(extension_ad_dispatcher, None, &view.roots);
        primal_ad_ctx.refresh_global_metadata();
        match try_primal_transpose(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            &aliases,
            &mut primal_ad_ctx,
            next_pass_id(),
        ) {
            Ok(transposed) => {
                if transposed
                    .tangent_outputs()
                    .first()
                    .and_then(|slot| *slot)
                    .is_some()
                {
                    let transposed = VjpTransposeGraph::Primal(transposed);
                    return build_vjp_tensor(
                        output,
                        wrt,
                        cotangent,
                        transposed,
                        None,
                        checkpoint_chain,
                        checkpoint_graphs,
                    );
                }
                return Ok(None);
            }
            Err(err) if !is_not_applicable_custom_vjp(&err) => {
                return Err(ad_rule_error_with_context(
                    transform,
                    err,
                    &mut primal_ad_ctx,
                ));
            }
            Err(err) => {
                if let Some(source) = primal_ad_ctx.take_deferred_shape_error() {
                    return Err(Error::ad_rule_source(transform, source));
                }
                let _ = err;
            }
        }
    }

    let linear_attempt = match (ad_transform_cache, cache_key) {
        (Some(cache), Some(key)) => {
            if let Some(cached) = cache.get_traced_vjp(&key)? {
                let residual_analysis =
                    register_scoped_graph_analysis(cached.residual_graph(), std::iter::empty())?;
                Ok(Some(ActiveLinearVjp {
                    transposed: cached,
                    residual_analysis,
                }))
            } else {
                let computed = compute_linear_vjp_transform(
                    &view,
                    &output_key,
                    &wrt_input_key,
                    &aliases,
                    extension_ad_dispatcher,
                    active_values,
                    wrt,
                )?;
                if let Ok(Some(active)) = &computed {
                    cache.put_traced_vjp(key, Arc::clone(&active.transposed))?;
                }
                computed
            }
        }
        _ => compute_linear_vjp_transform(
            &view,
            &output_key,
            &wrt_input_key,
            &aliases,
            extension_ad_dispatcher,
            active_values,
            wrt,
        )?,
    };

    let (transposed, residual_analysis) = match linear_attempt {
        Ok(None) => return Ok(None),
        Ok(Some(active)) => (
            VjpTransposeGraph::Linear(active.transposed),
            Some(active.residual_analysis),
        ),
        Err(linear_err) => return Err(ad_rule_error(transform, linear_err)),
    };

    build_vjp_tensor(
        output,
        wrt,
        cotangent,
        transposed,
        residual_analysis,
        checkpoint_chain,
        checkpoint_graphs,
    )
}

fn build_vjp_tensor(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    transposed: VjpTransposeGraph,
    residual_analysis: Option<RegisteredGraphAnalysis>,
    checkpoint_chain: Option<Arc<tenferro_runtime::ad_support::CheckpointNode>>,
    checkpoint_graphs: Vec<Arc<Graph<StdTensorOp>>>,
) -> Result<Option<TracedTensor>> {
    let cotangent_input_key =
        linear_input_key(transposed.as_graph(), transposed.tangent_inputs()[0].1)?;
    let cotangent_data = cotangent.attached_data().cloned().ok_or_else(|| {
        Error::invalid_argument(
            "vjp",
            ErrorPhase::GraphBuild,
            "cotangent",
            "vjp cotangent must have concrete tensor data",
        )
    })?;
    let transposed_analysis = register_scoped_graph_analysis(
        transposed.as_graph(),
        vec![(
            ValueKey::Input(cotangent_input_key.clone()),
            tensor_meta_from_tensor(cotangent_data.as_ref()),
        )],
    )?;
    let Some(cotangent_output) = transposed.tangent_outputs()[0] else {
        return Ok(None);
    };

    let mut inputs_map = (*tensor_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(cotangent_input_key.clone(), cotangent_data);

    let mut extra_roots = vec![Arc::clone(output.graph())];
    if let VjpTransposeGraph::Linear(cached) = &transposed {
        extra_roots.push(Arc::clone(cached.residual_graph()));
    }
    extra_roots.extend(checkpoint_graphs);
    extra_roots.extend(tensor_extra_roots(output));

    let (residual_metadata_scope, residual_constraint_scope) = match residual_analysis {
        Some(analysis) => (Some(analysis.metadata), Some(analysis.constraints)),
        None => (None, None),
    };
    let RegisteredGraphAnalysis {
        metadata: transposed_metadata_scope,
        constraints: transposed_constraint_scope,
    } = transposed_analysis;
    let inherited_constraint_scopes = [
        ConstraintScopeTransfer::from_tensor(output),
        ConstraintScopeTransfer::from_tensor(wrt),
        ConstraintScopeTransfer::from_tensor(cotangent),
    ];
    let inherited_constraint_scope = match residual_constraint_scope {
        Some(scope) => ConstraintScopeTransfer::with_new(scope, inherited_constraint_scopes.iter()),
        None => ConstraintScopeTransfer::merge(inherited_constraint_scopes.iter()),
    };
    let constraint_scope_transfer = ConstraintScopeTransfer::with_new(
        transposed_constraint_scope,
        [&inherited_constraint_scope],
    );

    Ok(Some(tensor_from_parts(TracedTensorParts {
        rank: wrt.rank,
        dtype: wrt.dtype,
        graph: transposed.into_graph_arc(),
        val: cotangent_output,
        data: None,
        shape_hint: tensor_shape_hint(wrt),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: {
            let mut scopes = if let Some(scope) = residual_metadata_scope {
                metadata_scopes_with_new(
                    scope,
                    [
                        tensor_metadata_scopes(output),
                        tensor_metadata_scopes(wrt),
                        tensor_metadata_scopes(cotangent),
                    ],
                )
            } else {
                let mut scopes: Vec<Arc<crate::metadata::GlobalMetadataScope>> = Vec::new();
                for inherited in [
                    tensor_metadata_scopes(output),
                    tensor_metadata_scopes(wrt),
                    tensor_metadata_scopes(cotangent),
                ] {
                    for scope in inherited {
                        scopes.push(Arc::clone(scope));
                    }
                }
                scopes
            };
            push_metadata_scope(&mut scopes, Arc::new(transposed_metadata_scope));
            scopes
        },
        constraint_scope_transfer,
    })))
}
