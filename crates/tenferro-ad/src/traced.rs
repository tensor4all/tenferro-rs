use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::resolve::resolve;
use computegraph::types::ValueKey;
use tenferro_ops::input_key::TensorInputKey;
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
use tidu::{try_linear_transpose, try_linearize};

static NEXT_DIFF_PASS_ID: AtomicU64 = AtomicU64::new(0);

fn next_pass_id() -> u64 {
    NEXT_DIFF_PASS_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn next_input_key() -> TensorInputKey {
    tenferro_runtime::ad_support::fresh_input_key()
}

fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .try_concrete_shape()
        .unwrap_or_else(|| vec![0; tensor.rank])
}

fn shape_guard_context(extension_rules: Option<&ExtensionRuleSet>) -> ShapeGuardContext {
    let ctx = ShapeGuardContext::with_global_metadata();
    match extension_rules {
        Some(rules) => ctx.with_extension_rules(rules.clone()),
        None => ctx,
    }
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
    jvp_optional_result_with_rules(output, wrt, tangent, Some(extension_rules))?.ok_or_else(|| {
        Error::Internal(format!(
            "jvp output is inactive for {:?}",
            leaf_input_key(wrt)
        ))
    })
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

    let ones = ones_tensor(output.dtype, vec![]);
    let seed = TracedTensor::from_tensor_concrete_shape(ones);
    vjp_optional_result_with_rules(output, wrt, &seed, Some(extension_rules))
}

pub(crate) fn jvp_optional_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    jvp_optional_result_with_rules(output, wrt, tangent, Some(extension_rules))
}

pub(crate) fn vjp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<TracedTensor> {
    vjp_optional_result_with_rules(output, wrt, cotangent, Some(extension_rules))?.ok_or_else(
        || {
            Error::Internal(format!(
                "vjp output is inactive for {:?}",
                leaf_input_key(wrt)
            ))
        },
    )
}

pub(crate) fn vjp_optional_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    vjp_optional_result_with_rules(output, wrt, cotangent, Some(extension_rules))
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

    let ones = ones_tensor(output.dtype, vec![]);
    let seed = TracedTensor::from_tensor_concrete_shape(ones);
    vjp_optional_result_with_rules(output, wrt, &seed, extension_rules)?.ok_or_else(|| {
        Error::Internal(format!(
            "grad output is inactive for {:?}",
            leaf_input_key(wrt)
        ))
    })
}

/// Automatic differentiation helpers for [`TracedTensor`].
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::TracedTensorAdExt;
/// use tenferro_runtime::TracedTensor;
///
/// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
/// let loss = x.scale_real(2.0);
/// let maybe_dx = loss.grad_optional(&x).unwrap();
/// assert!(maybe_dx.is_some());
/// ```
pub trait TracedTensorAdExt {
    /// Gradient of a scalar output with respect to a traced input.
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let loss = &x * &x;
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]);
    /// let loss = &y * &y;
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let mut y = &x * &x;
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let tangent = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]);
    /// let y = &x * &x;
    /// let dy = y.jvp(&x, &tangent);
    ///
    /// assert_eq!(eval(&dy).as_slice::<f64>().unwrap(), &[12.0]);
    /// ```
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor;

    /// Like [`jvp`](Self::jvp), but returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]);
    /// let tangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]);
    /// let loss = &y * &y;
    ///
    /// assert!(loss.jvp_optional(&x, &tangent).is_none());
    /// ```
    fn jvp_optional(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor>;

    /// Fallible forward-mode Jacobian-vector product.
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let tangent = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]);
    /// let y = &x * &x;
    /// let dy = y.jvp_optional_result(&x, &tangent).unwrap().unwrap();
    ///
    /// assert_eq!(eval(&dy).as_slice::<f64>().unwrap(), &[12.0]);
    /// ```
    fn jvp_optional_result(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>>;

    /// Reverse-mode vector-Jacobian product.
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let cotangent = TracedTensor::from_vec_col_major(vec![], vec![0.5_f64]);
    /// let y = &x * &x;
    /// let dx = y.vjp(&x, &cotangent);
    ///
    /// assert_eq!(eval(&dx).as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor;

    /// Like [`vjp`](Self::vjp), but returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::TracedTensor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let y = TracedTensor::from_vec_col_major(vec![], vec![4.0_f64]);
    /// let cotangent = TracedTensor::from_vec_col_major(vec![], vec![1.0_f64]);
    /// let loss = &y * &y;
    ///
    /// assert!(loss.vjp_optional(&x, &cotangent).is_none());
    /// ```
    fn vjp_optional(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Option<TracedTensor>;

    /// Fallible reverse-mode vector-Jacobian product.
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
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    /// let cotangent = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]);
    /// let y = &x * &x;
    /// let dx = y.vjp_optional_result(&x, &cotangent).unwrap().unwrap();
    ///
    /// assert_eq!(eval(&dx).as_slice::<f64>().unwrap(), &[12.0]);
    /// ```
    fn vjp_optional_result(
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

        let ones = ones_tensor(self.dtype, vec![]);
        let seed = TracedTensor::from_tensor_concrete_shape(ones);
        vjp_optional_result_with_rules(self, wrt, &seed, None)
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
        checkpoint_tensor(self, data);
        Ok(())
    }

    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor {
        self.jvp_optional(wrt, tangent)
            .unwrap_or_else(|| panic!("jvp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    fn jvp_optional(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor> {
        self.jvp_optional_result(wrt, tangent)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn jvp_optional_result(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        jvp_optional_result_with_rules(self, wrt, tangent, None)
    }

    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor {
        self.vjp_optional(wrt, cotangent)
            .unwrap_or_else(|| panic!("vjp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    fn vjp_optional(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Option<TracedTensor> {
        self.vjp_optional_result(wrt, cotangent)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn vjp_optional_result(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        vjp_optional_result_with_rules(self, wrt, cotangent, None)
    }
}

fn jvp_optional_result_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt);
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
    let mut ad_ctx = shape_guard_context(extension_rules);
    let linear = try_linearize(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    )
    .map_err(|err| Error::Internal(err.to_string()))?;
    let Some(tangent_output) = linear.tangent_outputs()[0] else {
        return Ok(None);
    };
    let tangent_input_key = linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1);
    let metadata_scope = register_scoped_graph_metadata(
        linear.as_graph(),
        vec![(
            ValueKey::Input(tangent_input_key.clone()),
            tensor_meta_from_tensor(
                tangent
                    .attached_data()
                    .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data"))
                    .as_ref(),
            ),
        )],
    );

    let mut inputs_map = (*tensor_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(
        tangent_input_key,
        tangent
            .attached_data()
            .cloned()
            .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data")),
    );

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

fn vjp_optional_result_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt);
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
    let mut ad_ctx = shape_guard_context(extension_rules);
    let linear = try_linearize(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    )
    .map_err(|err| Error::Internal(err.to_string()))?;
    if linear.tangent_outputs()[0].is_none() {
        return Ok(None);
    }
    let linear_seed_key = linear_input_key(linear.as_graph(), linear.tangent_inputs()[0].1);
    let linear_metadata_scope = register_scoped_graph_metadata(
        linear.as_graph(),
        vec![(
            ValueKey::Input(linear_seed_key),
            registered_meta(&wrt.graph().values()[wrt.val].key),
        )],
    );
    ad_ctx.refresh_global_metadata();
    let transposed = try_linear_transpose(&linear, &mut ad_ctx)
        .map_err(|err| Error::Internal(err.to_string()))?;
    let cotangent_input_key =
        linear_input_key(transposed.as_graph(), transposed.tangent_inputs()[0].1);
    let transposed_metadata_scope = register_scoped_graph_metadata(
        transposed.as_graph(),
        vec![(
            ValueKey::Input(cotangent_input_key.clone()),
            tensor_meta_from_tensor(
                cotangent
                    .attached_data()
                    .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data"))
                    .as_ref(),
            ),
        )],
    );
    let linear_graph = Arc::new(linear.into_graph());
    let Some(cotangent_output) = transposed.tangent_outputs()[0] else {
        return Ok(None);
    };

    let mut inputs_map = (*tensor_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(
        cotangent_input_key.clone(),
        cotangent
            .attached_data()
            .cloned()
            .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data")),
    );

    let mut extra_roots = vec![Arc::clone(output.graph()), linear_graph];
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
            let mut scopes = metadata_scopes_with_new(
                linear_metadata_scope,
                [
                    tensor_metadata_scopes(output),
                    tensor_metadata_scopes(wrt),
                    tensor_metadata_scopes(cotangent),
                ],
            );
            push_metadata_scope(&mut scopes, Arc::new(transposed_metadata_scope));
            scopes
        },
    })))
}
