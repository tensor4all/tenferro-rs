use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::resolve::resolve;
use computegraph::types::GlobalValKey;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::ExtensionRuleSet;
use tenferro_ops::ShapeGuardContext;
use tenferro_runtime::ad_support::{
    checkpoint_traced_tensor, leaf_input_key, linear_input_key, metadata_scopes_with_new,
    ones_tensor, push_metadata_scope, register_scoped_fragment_metadata, registered_meta,
    tensor_meta_from_tensor, traced_checkpoint_chain, traced_extra_roots, traced_inputs_map,
    traced_metadata_scopes, traced_resolve_roots, traced_shape_hint, traced_tensor_from_parts,
    TracedTensorParts,
};
use tenferro_runtime::{Error, GraphCompiler, GraphExecutor, Result, TracedTensor};
use tenferro_tensor::TensorBackend;
use tidu::{try_differentiate, try_transpose};

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
    try_jvp_result_with_optional_rules(output, wrt, tangent, Some(extension_rules))?.ok_or_else(
        || {
            Error::Internal(format!(
                "jvp output is inactive for {:?}",
                leaf_input_key(wrt)
            ))
        },
    )
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
    try_vjp_result_with_optional_rules(output, wrt, &seed, Some(extension_rules))
}

pub(crate) fn try_jvp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    try_jvp_result_with_optional_rules(output, wrt, tangent, Some(extension_rules))
}

pub(crate) fn vjp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<TracedTensor> {
    try_vjp_result_with_optional_rules(output, wrt, cotangent, Some(extension_rules))?.ok_or_else(
        || {
            Error::Internal(format!(
                "vjp output is inactive for {:?}",
                leaf_input_key(wrt)
            ))
        },
    )
}

pub(crate) fn try_vjp_with_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: &ExtensionRuleSet,
) -> Result<Option<TracedTensor>> {
    try_vjp_result_with_optional_rules(output, wrt, cotangent, Some(extension_rules))
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
    try_vjp_result_with_optional_rules(output, wrt, &seed, extension_rules)?.ok_or_else(|| {
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
    fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor>;

    /// Like [`grad`](Self::grad), but returns `None` when `wrt` is inactive.
    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>>;

    /// Evaluate this tensor and replace its graph with a concrete leaf while
    /// preserving the previous graph for AD replay.
    fn checkpoint<B: TensorBackend>(
        &mut self,
        compiler: &mut GraphCompiler,
        executor: &mut GraphExecutor<B>,
    ) -> Result<()>;

    /// Forward-mode Jacobian-vector product.
    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor;

    /// Like [`jvp`](Self::jvp), but returns `None` when `wrt` is inactive.
    fn try_jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor>;

    /// Fallible forward-mode Jacobian-vector product.
    fn try_jvp_result(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>>;

    /// Reverse-mode vector-Jacobian product.
    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor;

    /// Fallible reverse-mode vector-Jacobian product.
    fn try_vjp_result(
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
        try_vjp_result_with_optional_rules(self, wrt, &seed, None)
    }

    fn checkpoint<B: TensorBackend>(
        &mut self,
        compiler: &mut GraphCompiler,
        executor: &mut GraphExecutor<B>,
    ) -> Result<()> {
        let data = if let Some(data) = &self.data {
            Arc::clone(data)
        } else {
            let program = compiler.compile(self)?;
            Arc::new(executor.run(&program)?)
        };
        checkpoint_traced_tensor(self, data);
        Ok(())
    }

    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor {
        self.try_jvp(wrt, tangent)
            .unwrap_or_else(|| panic!("jvp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    fn try_jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor> {
        self.try_jvp_result(wrt, tangent)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn try_jvp_result(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        try_jvp_result_with_optional_rules(self, wrt, tangent, None)
    }

    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor {
        match self.try_vjp_result(wrt, cotangent) {
            Ok(Some(vjp)) => vjp,
            Ok(None) => panic!("vjp output is inactive for {:?}", leaf_input_key(wrt)),
            Err(err) => panic!("{err}"),
        }
    }

    fn try_vjp_result(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        try_vjp_result_with_optional_rules(self, wrt, cotangent, None)
    }
}

fn try_jvp_result_with_optional_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt);
    let output_key = output.fragment.vals()[output.val].key.clone();
    let checkpoint_chain = traced_checkpoint_chain(output);
    let aliases = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_aliases())
        .unwrap_or_default();
    let checkpoint_fragments = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_fragments())
        .unwrap_or_default();
    let mut roots = traced_resolve_roots(output);
    roots.extend(checkpoint_fragments.iter().cloned());
    let view = resolve(roots);
    let mut ad_ctx = shape_guard_context(extension_rules);
    let linear = try_differentiate(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    )
    .map_err(|err| Error::Internal(err.to_string()))?;
    let Some(tangent_output) = linear.tangent_outputs[0] else {
        return Ok(None);
    };
    let tangent_input_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);
    let metadata_scope = register_scoped_fragment_metadata(
        &linear.fragment,
        vec![(
            GlobalValKey::Input(tangent_input_key.clone()),
            tensor_meta_from_tensor(
                tangent
                    .data
                    .as_ref()
                    .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data"))
                    .as_ref(),
            ),
        )],
    );

    let mut inputs_map = (*traced_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(
        tangent_input_key,
        tangent
            .data
            .clone()
            .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data")),
    );

    let mut extra_roots = vec![output.fragment.clone()];
    extra_roots.extend(checkpoint_fragments);
    extra_roots.extend(traced_extra_roots(output));

    Ok(Some(traced_tensor_from_parts(TracedTensorParts {
        rank: output.rank,
        dtype: output.dtype,
        fragment: Arc::new(linear.fragment),
        val: tangent_output,
        data: None,
        shape_hint: traced_shape_hint(output),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: metadata_scopes_with_new(
            metadata_scope,
            [
                traced_metadata_scopes(output),
                traced_metadata_scopes(wrt),
                traced_metadata_scopes(tangent),
            ],
        ),
    })))
}

fn try_vjp_result_with_optional_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    extension_rules: Option<&ExtensionRuleSet>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt);
    let output_key = output.fragment.vals()[output.val].key.clone();
    let checkpoint_chain = traced_checkpoint_chain(output);
    let aliases = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_aliases())
        .unwrap_or_default();
    let checkpoint_fragments = checkpoint_chain
        .as_ref()
        .map(|chain| chain.collect_fragments())
        .unwrap_or_default();
    let mut roots = traced_resolve_roots(output);
    roots.extend(checkpoint_fragments.iter().cloned());
    let view = resolve(roots);
    let mut ad_ctx = shape_guard_context(extension_rules);
    let linear = try_differentiate(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&wrt_input_key),
        next_pass_id(),
        &mut ad_ctx,
        &aliases,
    )
    .map_err(|err| Error::Internal(err.to_string()))?;
    if linear.tangent_outputs[0].is_none() {
        return Ok(None);
    }
    let linear_seed_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);
    let linear_metadata_scope = register_scoped_fragment_metadata(
        &linear.fragment,
        vec![(
            GlobalValKey::Input(linear_seed_key),
            registered_meta(&wrt.fragment.vals()[wrt.val].key),
        )],
    );
    ad_ctx.refresh_global_metadata();
    let transposed =
        try_transpose(&linear, &mut ad_ctx).map_err(|err| Error::Internal(err.to_string()))?;
    let cotangent_input_key =
        linear_input_key(&transposed.fragment, transposed.tangent_inputs[0].1);
    let transposed_metadata_scope = register_scoped_fragment_metadata(
        &transposed.fragment,
        vec![(
            GlobalValKey::Input(cotangent_input_key.clone()),
            tensor_meta_from_tensor(
                cotangent
                    .data
                    .as_ref()
                    .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data"))
                    .as_ref(),
            ),
        )],
    );
    let linear_fragment = Arc::new(linear.fragment);
    let Some(cotangent_output) = transposed.tangent_outputs[0] else {
        return Ok(None);
    };

    let mut inputs_map = (*traced_inputs_map(output)).clone();
    if let Some(chain) = &checkpoint_chain {
        inputs_map.extend(chain.collect_inputs());
    }
    inputs_map.insert(
        cotangent_input_key.clone(),
        cotangent
            .data
            .clone()
            .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data")),
    );

    let mut extra_roots = vec![output.fragment.clone(), linear_fragment];
    extra_roots.extend(checkpoint_fragments);
    extra_roots.extend(traced_extra_roots(output));

    Ok(Some(traced_tensor_from_parts(TracedTensorParts {
        rank: wrt.rank,
        dtype: wrt.dtype,
        fragment: Arc::new(transposed.fragment),
        val: cotangent_output,
        data: None,
        shape_hint: traced_shape_hint(wrt),
        inputs_map: Arc::new(inputs_map),
        extra_roots,
        checkpoint_chain,
        metadata_scopes: {
            let mut scopes = metadata_scopes_with_new(
                linear_metadata_scope,
                [
                    traced_metadata_scopes(output),
                    traced_metadata_scopes(wrt),
                    traced_metadata_scopes(cotangent),
                ],
            );
            push_metadata_scope(&mut scopes, Arc::new(transposed_metadata_scope));
            scopes
        },
    })))
}
