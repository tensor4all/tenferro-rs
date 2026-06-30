use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::ad_rule_error::ad_rule_error;
use computegraph::resolve::resolve;
use computegraph::resolve::{ResolvedView, ValueDef};
use computegraph::types::ValueKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ExtensionRuleSet;
use tenferro_ops::ShapeGuardContext;
use tenferro_runtime::ad_support::{
    checkpoint_chain as tensor_checkpoint_chain, checkpoint_tensor,
    extra_roots as tensor_extra_roots, inputs_map as tensor_inputs_map, leaf_input_key,
    linear_input_key, metadata_scopes as tensor_metadata_scopes, metadata_scopes_with_new,
    ones_tensor, push_metadata_scope, register_scoped_graph_metadata, registered_meta,
    resolve_roots as tensor_resolve_roots, shape_hint as tensor_shape_hint, tensor_from_parts,
    tensor_meta_from_tensor, TracedTensorParts,
};
use tenferro_runtime::{Error, GraphCompiler, GraphExecutor, Result, TracedTensor};
use tenferro_tensor::TensorBackend;
use tidu::{linear_transpose, linearize};

use crate::primal_transpose::{try_primal_transpose, PrimalTransposeGraph};

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
    extension_rules: Option<&ExtensionRuleSet>,
    active_values: Option<Arc<HashSet<ValueKey<StdTensorOp>>>>,
) -> ShapeGuardContext {
    let ctx = ShapeGuardContext::with_global_metadata();
    let ctx = match extension_rules {
        Some(rules) => ctx.with_extension_rules(rules.clone()),
        None => ctx,
    };
    match active_values {
        Some(keys) => ctx.with_linearize_active_values(keys),
        None => ctx,
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

pub(crate) fn grad_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<TracedTensor> {
    grad_with_optional_rules(output, wrt, Some(extension_rules))
}

pub(crate) fn jvp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    jvp_optional_impl(output, wrt, tangent, Some(extension_rules))?
        .ok_or_else(|| Error::Internal(format!("jvp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn grad_optional_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    if output.rank != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }

    let ones = ones_tensor(output.dtype, vec![])?;
    let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
    vjp_optional_impl(output, wrt, &seed, Some(extension_rules), "grad")
}

pub(crate) fn jvp_optional_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    jvp_optional_impl(output, wrt, tangent, Some(extension_rules))
}

pub(crate) fn vjp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    vjp_optional_impl(output, wrt, cotangent, Some(extension_rules), "vjp")?
        .ok_or_else(|| Error::Internal(format!("vjp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn vjp_optional_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    vjp_optional_impl(output, wrt, cotangent, Some(extension_rules), "vjp")
}

fn grad_with_optional_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
) -> Result<TracedTensor> {
    if output.rank != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }

    let ones = ones_tensor(output.dtype, vec![])?;
    let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
    let wrt_input_key = leaf_input_key(wrt)?;
    vjp_optional_impl(output, wrt, &seed, extension_rules, "grad")?
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
    fn vjp_optional(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>>;
}

impl TracedTensorAdExt for TracedTensor {
    fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor> {
        grad_with_optional_rules(self, wrt, None)
    }

    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![])?;
        let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
        vjp_optional_impl(self, wrt, &seed, None, "grad")
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
        jvp_optional_impl(self, wrt, tangent, None)
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
        vjp_optional_impl(self, wrt, cotangent, None, "vjp")
    }
}

fn jvp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
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
    let active_values = Arc::from(linearize_active_value_keys(
        &view,
        std::slice::from_ref(&output_key),
        &aliases,
    ));
    let mut ad_ctx = shape_guard_context(extension_rules, Some(active_values));
    let linear = linearize(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    )
    .map_err(|err| ad_rule_error("jvp", err))?;
    let Some(tangent_output) = linear.tangent_outputs()[0] else {
        return Ok(None);
    };
    let tangent_input_key = linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1)?;
    let tangent_data =
        tangent
            .attached_data()
            .cloned()
            .ok_or_else(|| Error::InvalidGraphBuild {
                op: "jvp",
                message: "jvp tangent must have concrete tensor data".to_string(),
            })?;
    let metadata_scope = register_scoped_graph_metadata(
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

    Ok(Some(tensor_from_parts(TracedTensorParts {
        rank: output.rank,
        dtype: output.dtype,
        graph: Arc::new(linear.into_graph()),
        val: tangent_output,
        data: None,
        shape_hint: tensor_shape_hint(output),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: metadata_scopes_with_new(
            metadata_scope,
            [
                tensor_metadata_scopes(output),
                tensor_metadata_scopes(wrt),
                tensor_metadata_scopes(tangent),
            ],
        ),
    })))
}

enum VjpTransposeGraph {
    Primal(PrimalTransposeGraph),
    Linear(tidu::LinearizedGraph<StdTensorOp>),
}

impl VjpTransposeGraph {
    fn as_graph(&self) -> &computegraph::graph::Graph<StdTensorOp> {
        match self {
            Self::Primal(graph) => graph.as_graph(),
            Self::Linear(graph) => graph.as_graph(),
        }
    }

    fn tangent_inputs(&self) -> &[(TensorInputKey, computegraph::LocalValueId)] {
        match self {
            Self::Primal(graph) => graph.tangent_inputs(),
            Self::Linear(graph) => graph.tangent_inputs(),
        }
    }

    fn tangent_outputs(&self) -> &[Option<computegraph::LocalValueId>] {
        match self {
            Self::Primal(graph) => graph.tangent_outputs(),
            Self::Linear(graph) => graph.tangent_outputs(),
        }
    }

    fn into_graph(self) -> computegraph::graph::Graph<StdTensorOp> {
        match self {
            Self::Primal(graph) => graph.into_graph(),
            Self::Linear(graph) => graph.into_graph(),
        }
    }
}

fn vjp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
    transform: &'static str,
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
    let mut ad_ctx = shape_guard_context(extension_rules, None);
    ad_ctx.refresh_global_metadata();

    let (transposed, linear_metadata_scope, linear_graph) = match try_primal_transpose(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        &aliases,
        &mut ad_ctx,
        next_pass_id(),
    ) {
        Ok(transposed)
            if transposed
                .tangent_outputs()
                .first()
                .and_then(|slot| *slot)
                .is_some() =>
        {
            (VjpTransposeGraph::Primal(transposed), None, None)
        }
        _ => {
            let active_values = Arc::from(linearize_active_value_keys(
                &view,
                std::slice::from_ref(&output_key),
                &aliases,
            ));
            let mut ad_ctx = shape_guard_context(extension_rules, Some(active_values));
            let linear = linearize(
                &view,
                std::slice::from_ref(&output_key),
                std::slice::from_ref(&wrt_input_key),
                next_pass_id(),
                &mut ad_ctx,
                &aliases,
            )
            .map_err(|err| ad_rule_error(transform, err))?;
            if linear.tangent_outputs()[0].is_none() {
                return Ok(None);
            }
            let linear_seed_key =
                linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1)?;
            let linear_metadata_scope = Some(register_scoped_graph_metadata(
                linear.as_graph(),
                vec![(
                    ValueKey::Input(linear_seed_key),
                    registered_meta(&wrt.graph().values()[wrt.val].key)?,
                )],
            )?);
            ad_ctx.refresh_global_metadata();
            let transposed = linear_transpose(&linear, &mut ad_ctx)
                .map_err(|err| ad_rule_error(transform, err))?;
            (
                VjpTransposeGraph::Linear(transposed),
                linear_metadata_scope,
                Some(Arc::new(linear.into_graph())),
            )
        }
    };

    let cotangent_input_key =
        linear_input_key(transposed.as_graph(), transposed.tangent_inputs()[0].1)?;
    let cotangent_data =
        cotangent
            .attached_data()
            .cloned()
            .ok_or_else(|| Error::InvalidGraphBuild {
                op: "vjp",
                message: "vjp cotangent must have concrete tensor data".to_string(),
            })?;
    let transposed_metadata_scope = register_scoped_graph_metadata(
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
    if let Some(linear_graph) = linear_graph {
        extra_roots.push(linear_graph);
    }
    extra_roots.extend(checkpoint_graphs);
    extra_roots.extend(tensor_extra_roots(output));

    Ok(Some(tensor_from_parts(TracedTensorParts {
        rank: wrt.rank,
        dtype: wrt.dtype,
        graph: Arc::new(transposed.into_graph()),
        val: cotangent_output,
        data: None,
        shape_hint: tensor_shape_hint(wrt),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: {
            let mut scopes = if let Some(scope) = linear_metadata_scope {
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
    })))
}
