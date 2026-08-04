use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::{LocalValueId, OperationRole, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{SymDim, TensorMeta};
use tenferro_runtime::ad_support::{
    allocate_input_key, allocate_shape_tensor_id, checkpoint_tensor, compile_ad_source,
    frozen_input_value, inputs_map as tensor_inputs_map, leaf_input_key,
    metadata_scopes as tensor_metadata_scopes, metadata_scopes_with_new, ones_tensor,
    register_scoped_graph_analysis, shape_hint as tensor_shape_hint, tensor_from_parts,
    ConstraintScopeTransfer, RetainedValue, TracedTensorParts,
};
use tenferro_runtime::program::{FrozenProgram, ProgramValue, ProgramValueMetadata, SemanticOpRef};
use tenferro_runtime::{
    CompiledGraph, Error, ErrorPhase, GraphCompiler, Result, Runtime, Tensor, TracedTensor,
};

use crate::semantic_extension::{SemanticAdError, SemanticExtensionRuleSet};
use crate::semantic_transform::{
    semantic_jvp, semantic_vjp, SemanticAdProgram, SemanticAdTransformError,
};
use crate::transform_cache::{AdTransformCache, SemanticAdTransformCacheKey};

pub(crate) fn next_input_key() -> TensorInputKey {
    tenferro_runtime::ad_support::allocate_input_key()
}

fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .try_concrete_shape()
        .unwrap_or_else(|| vec![0; tensor.rank])
}

pub(crate) fn grad_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    grad_with_optional_rules(output, wrt, rules, ad_transform_cache)
}

pub(crate) fn jvp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    jvp_optional_impl(output, wrt, tangent, rules, ad_transform_cache)?
        .ok_or_else(|| Error::Internal(format!("jvp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn grad_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    if output.rank != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }

    let ones = ones_tensor(output.dtype, vec![])?;
    let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
    vjp_optional_impl(output, wrt, &seed, rules, "grad", ad_transform_cache)
}

pub(crate) fn jvp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    jvp_optional_impl(output, wrt, tangent, rules, ad_transform_cache)
}

pub(crate) fn vjp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<TracedTensor> {
    let wrt_input_key = leaf_input_key(wrt)?;
    vjp_optional_impl(output, wrt, cotangent, rules, "vjp", ad_transform_cache)?
        .ok_or_else(|| Error::Internal(format!("vjp output is inactive for {:?}", wrt_input_key)))
}

pub(crate) fn vjp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    vjp_optional_impl(output, wrt, cotangent, rules, "vjp", ad_transform_cache)
}

fn grad_with_optional_rules(
    output: &TracedTensor,
    wrt: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
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
    vjp_optional_impl(output, wrt, &seed, rules, "grad", ad_transform_cache)?
        .ok_or_else(|| Error::Internal(format!("grad output is inactive for {:?}", wrt_input_key)))
}

fn single_runtime_output(mut outputs: Vec<Tensor>, op: &'static str) -> Result<Tensor> {
    let actual = outputs.len();
    if actual != 1 {
        return Err(Error::runtime_state(
            op,
            ErrorPhase::Execution,
            format!("expected one runtime output, got {actual}"),
        ));
    }
    outputs.pop().ok_or_else(|| {
        Error::runtime_state(
            op,
            ErrorPhase::Execution,
            "runtime returned no output after successful output-count validation",
        )
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
    /// use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let backend = CpuBackend::new();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     runtime.run_compiled(&program, &[]).unwrap().pop().unwrap()
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
    /// use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let backend = CpuBackend::new();
    /// let mut builder = Runtime::builder();
    /// builder
    ///     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///     .unwrap();
    /// let runtime = builder.build().unwrap();
    /// let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let mut y = (&x * &x).unwrap();
    ///
    /// y.checkpoint(&mut compiler, &runtime).unwrap();
    ///
    /// let value = y.attached_value().unwrap();
    /// assert_eq!(value.as_slice::<f64>().unwrap(), &[9.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] when checkpoint metadata
    /// is invalid, [`Error::RuntimeState`] when graph metadata or runtime
    /// state is unavailable, or a typed backend error from evaluation.
    fn checkpoint(&mut self, compiler: &mut GraphCompiler, runtime: &Runtime) -> Result<()>;

    /// Forward-mode Jacobian-vector product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let backend = CpuBackend::new();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     runtime.run_compiled(&program, &[]).unwrap().pop().unwrap()
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
    /// use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let backend = CpuBackend::new();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     runtime.run_compiled(&program, &[]).unwrap().pop().unwrap()
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
        let rules = SemanticExtensionRuleSet::default();
        grad_with_optional_rules(self, wrt, &rules, None)
    }

    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![])?;
        let seed = TracedTensor::from_tensor_concrete_shape(ones)?;
        let rules = SemanticExtensionRuleSet::default();
        vjp_optional_impl(self, wrt, &seed, &rules, "grad", None)
    }

    fn checkpoint(&mut self, compiler: &mut GraphCompiler, runtime: &Runtime) -> Result<()> {
        let data = if let Some(data) = self.attached_value() {
            Arc::clone(data)
        } else {
            let program = compiler.compile(self)?;
            Arc::new(RetainedValue::from_tensor(single_runtime_output(
                runtime.run_compiled(&program, &[])?,
                "TracedTensorAdExt::checkpoint",
            )?))
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
        let rules = SemanticExtensionRuleSet::default();
        jvp_optional_impl(self, wrt, tangent, &rules, None)
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
        let rules = SemanticExtensionRuleSet::default();
        vjp_optional_impl(self, wrt, cotangent, &rules, "vjp", None)
    }
}

fn jvp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt)?;
    let tangent_data = tangent.attached_value().cloned().ok_or_else(|| {
        Error::invalid_argument(
            "jvp",
            ErrorPhase::GraphBuild,
            "tangent",
            "jvp tangent must have concrete tensor data",
        )
    })?;
    let mut compiler = GraphCompiler::new();
    let source = compile_ad_source(&mut compiler, output)?;
    let Some(wrt_input_index) = source.input_key_index(&wrt_input_key) else {
        return Ok(None);
    };

    let mut active_inputs = vec![false; source.input_count()];
    active_inputs[wrt_input_index] = true;
    let derivative = semantic_jvp_with_cache(
        source.frozen_program(),
        &active_inputs,
        rules,
        ad_transform_cache,
    )?;
    let Some(seed_input_index) = derivative
        .derivative_input_indices()
        .get(wrt_input_index)
        .copied()
        .flatten()
    else {
        return Ok(None);
    };
    let Some(derivative_output_index) = derivative
        .derivative_output_indices()
        .first()
        .copied()
        .flatten()
    else {
        return Ok(None);
    };

    derivative_tensor_from_program(
        &source,
        &derivative,
        derivative_output_index,
        &[(seed_input_index, tangent_data)],
        [output, wrt, tangent],
        tensor_shape_hint(output),
        "jvp",
    )
    .map(Some)
}

fn vjp_optional_impl(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    transform: &'static str,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<Option<TracedTensor>> {
    let wrt_input_key = leaf_input_key(wrt)?;
    let cotangent_data = cotangent.attached_value().cloned().ok_or_else(|| {
        Error::invalid_argument(
            transform,
            ErrorPhase::GraphBuild,
            "cotangent",
            "vjp cotangent must have concrete tensor data",
        )
    })?;
    let mut compiler = GraphCompiler::new();
    let source = compile_ad_source(&mut compiler, output)?;
    let Some(wrt_input_index) = source.input_key_index(&wrt_input_key) else {
        return Ok(None);
    };

    let mut active_inputs = vec![false; source.input_count()];
    active_inputs[wrt_input_index] = true;
    let active_outputs = vec![true; source.output_count()];
    let derivative = semantic_vjp_with_cache(
        source.frozen_program(),
        &active_inputs,
        &active_outputs,
        rules,
        ad_transform_cache,
    )?;
    let Some(seed_input_index) = derivative
        .derivative_input_indices()
        .first()
        .copied()
        .flatten()
    else {
        return Ok(None);
    };
    let Some(derivative_output_index) = derivative
        .derivative_output_indices()
        .get(wrt_input_index)
        .copied()
        .flatten()
    else {
        return Ok(None);
    };

    derivative_tensor_from_program(
        &source,
        &derivative,
        derivative_output_index,
        &[(seed_input_index, cotangent_data)],
        [output, wrt, cotangent],
        tensor_shape_hint(wrt),
        transform,
    )
    .map(Some)
}

fn semantic_jvp_with_cache(
    source: &FrozenProgram,
    active_inputs: &[bool],
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<SemanticAdProgram> {
    let key = SemanticAdTransformCacheKey::jvp(source, active_inputs);
    if let Some(cache) = ad_transform_cache {
        if let Some(cached) = cache.get_semantic(&key, source)? {
            return cached
                .as_ref()
                .with_input_prefix_bindings_from(source)
                .map_err(|source| {
                    Error::runtime_state_source(
                        "semantic traced jvp cache",
                        ErrorPhase::GraphBuild,
                        source,
                    )
                });
        }
    }
    let derivative =
        semantic_jvp(source, active_inputs, rules).map_err(semantic_transform_error("jvp"))?;
    if let Some(cache) = ad_transform_cache {
        cache.put_semantic(key, source, Arc::new(derivative.clone()))?;
    }
    Ok(derivative)
}

fn semantic_vjp_with_cache(
    source: &FrozenProgram,
    active_inputs: &[bool],
    active_outputs: &[bool],
    rules: &SemanticExtensionRuleSet,
    ad_transform_cache: Option<&AdTransformCache>,
) -> Result<SemanticAdProgram> {
    let key = SemanticAdTransformCacheKey::vjp(source, active_inputs, active_outputs);
    if let Some(cache) = ad_transform_cache {
        if let Some(cached) = cache.get_semantic(&key, source)? {
            return cached
                .as_ref()
                .with_input_prefix_bindings_from(source)
                .map_err(|source| {
                    Error::runtime_state_source(
                        "semantic traced vjp cache",
                        ErrorPhase::GraphBuild,
                        source,
                    )
                });
        }
    }
    let derivative = semantic_vjp(source, active_inputs, active_outputs, rules)
        .map_err(semantic_transform_error("vjp"))?;
    if let Some(cache) = ad_transform_cache {
        cache.put_semantic(key, source, Arc::new(derivative.clone()))?;
    }
    Ok(derivative)
}

fn semantic_transform_error(
    transform: &'static str,
) -> impl FnOnce(SemanticAdTransformError) -> Error {
    move |source| {
        semantic_transform_validation_error(transform, &source).unwrap_or_else(|| {
            Error::runtime_state_source(transform, ErrorPhase::GraphBuild, source)
        })
    }
}

fn semantic_transform_validation_error(
    transform: &'static str,
    source: &SemanticAdTransformError,
) -> Option<Error> {
    if let SemanticAdTransformError::Extension(
        SemanticAdError::Unsupported { family_id, .. }
        | SemanticAdError::MissingRule { family_id, .. },
    ) = source
    {
        return Some(Error::UnsupportedAdRule {
            transform,
            op: (*family_id).to_owned(),
        });
    }

    let SemanticAdTransformError::Extension(SemanticAdError::Rule { source, .. }) = source else {
        return None;
    };
    let tenferro_ops::ad::ADRuleError::InvalidInput { op, message, .. } =
        source.downcast_ref::<tenferro_ops::ad::ADRuleError>()?
    else {
        return None;
    };
    Some(Error::invalid_argument(
        transform,
        ErrorPhase::GraphBuild,
        "semantic_ad_rule",
        format!("{op}: {message}"),
    ))
}

fn derivative_tensor_from_program(
    source: &CompiledGraph,
    derivative: &SemanticAdProgram,
    derivative_output_index: usize,
    seed_tensors: &[(usize, Arc<RetainedValue>)],
    inherited_tensors: [&TracedTensor; 3],
    fallback_shape_hint: Option<Vec<SymDim>>,
    transform: &'static str,
) -> Result<TracedTensor> {
    derivative_trace_from_frozen_program(
        source,
        derivative.frozen(),
        derivative_output_index,
        seed_tensors,
        &inherited_tensors,
        fallback_shape_hint,
        transform,
    )
}

pub(crate) fn derivative_trace_from_frozen_program(
    source: &CompiledGraph,
    frozen: &FrozenProgram,
    derivative_output_index: usize,
    seed_tensors: &[(usize, Arc<RetainedValue>)],
    inherited_tensors: &[&TracedTensor],
    fallback_shape_hint: Option<Vec<SymDim>>,
    transform: &'static str,
) -> Result<TracedTensor> {
    let input_shapes = symbolic_input_shapes(frozen)?;
    let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
    let input_metas = frozen
        .program
        .inputs()
        .iter()
        .copied()
        .map(|value| tensor_meta_for_value(frozen, value, &input_shape_refs, transform))
        .collect::<Result<Vec<_>>>()?;

    let output_value = *frozen
        .program
        .outputs()
        .get(derivative_output_index)
        .ok_or_else(|| {
            Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                format!(
                    "derivative output index {derivative_output_index} is outside {} outputs",
                    frozen.program.outputs().len()
                ),
            )
        })?;
    let output_meta = tensor_meta_for_value(frozen, output_value, &input_shape_refs, transform)?;

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut value_map = HashMap::<ProgramValue, LocalValueId>::new();
    let mut input_keys = Vec::with_capacity(frozen.program.inputs().len());
    for (input_index, input) in frozen.program.inputs().iter().copied().enumerate() {
        let key = if input_index < source.input_keys().len() {
            source.input_keys()[input_index].clone()
        } else {
            allocate_input_key()
        };
        let local = builder.add_input(key.clone());
        value_map.insert(input, local);
        input_keys.push(key);
    }

    for operation in frozen.program.operations() {
        let inputs = operation
            .inputs()
            .iter()
            .copied()
            .map(|value| {
                value_map
                    .get(&value)
                    .copied()
                    .map(ValueRef::Local)
                    .ok_or_else(|| missing_program_value(transform, "operation input"))
            })
            .collect::<Result<Vec<_>>>()?;
        let op = match operation.op() {
            SemanticOpRef::Core(op) => StdTensorOp::from(op),
            SemanticOpRef::Extension(op) => StdTensorOp::Extension(op.clone_arc()),
            _ => {
                return Err(Error::runtime_state(
                    transform,
                    ErrorPhase::GraphBuild,
                    "unsupported semantic operation variant in derivative graph",
                ));
            }
        };
        let outputs = builder.add_operation(op, inputs, OperationRole::Primary);
        if outputs.len() != operation.outputs().len() {
            return Err(Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                format!(
                    "semantic operation expected {} outputs, graph builder produced {}",
                    operation.outputs().len(),
                    outputs.len()
                ),
            ));
        }
        for (value, local) in operation.outputs().iter().copied().zip(outputs) {
            value_map.insert(value, local);
        }
    }

    let graph_outputs = frozen
        .program
        .outputs()
        .iter()
        .copied()
        .map(|value| {
            value_map
                .get(&value)
                .copied()
                .ok_or_else(|| missing_program_value(transform, "program output"))
        })
        .collect::<Result<Vec<_>>>()?;
    let val = *graph_outputs.get(derivative_output_index).ok_or_else(|| {
        Error::runtime_state(
            transform,
            ErrorPhase::GraphBuild,
            "derivative output index missing after graph conversion",
        )
    })?;
    builder.set_outputs(graph_outputs);
    let graph = Arc::new(builder.build());

    let Some(primary_tensor) = inherited_tensors.first() else {
        return Err(Error::runtime_state(
            transform,
            ErrorPhase::GraphBuild,
            "derivative trace construction requires inherited source tensors",
        ));
    };
    let mut inputs_map = (*tensor_inputs_map(primary_tensor)).clone();
    for (input_index, key) in input_keys.iter().enumerate() {
        if let Some(tensor) = frozen_input_value(frozen, input_index) {
            inputs_map.insert(key.clone(), tensor);
        }
    }
    for (seed_input_index, tensor) in seed_tensors {
        let meta = input_metas.get(*seed_input_index).ok_or_else(|| {
            Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                format!("seed input index {seed_input_index} is outside derivative inputs"),
            )
        })?;
        validate_seed_tensor(transform, *seed_input_index, tensor.as_ref(), meta)?;
        let key = input_keys.get(*seed_input_index).ok_or_else(|| {
            Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                format!("seed input key {seed_input_index} is outside derivative inputs"),
            )
        })?;
        inputs_map.insert(key.clone(), Arc::clone(tensor));
    }

    let source_input_count = source.input_keys().len();
    let graph_input_metadata = graph
        .inputs()
        .iter()
        .copied()
        .zip(input_metas.iter().cloned())
        .enumerate()
        .filter_map(|(input_index, (input, meta))| {
            // Source input keys are reused so derivative traces compose with the
            // original eager tensor. Keep those keys owned by the source tensor's
            // metadata scopes; derivative input metadata may be shape-specialized
            // for the current run and must not shadow the source while the VJP/JVP
            // result stays alive.
            if input_index < source_input_count {
                None
            } else {
                Some((graph.values()[input].key.clone(), meta))
            }
        });
    let analysis = register_scoped_graph_analysis(graph.as_ref(), graph_input_metadata)?;
    let inherited_constraint_scopes = inherited_tensors
        .iter()
        .map(|tensor| ConstraintScopeTransfer::from_tensor(tensor))
        .collect::<Vec<_>>();

    Ok(tensor_from_parts(TracedTensorParts {
        rank: output_meta.rank(),
        dtype: output_meta.dtype,
        graph,
        val,
        data: None,
        shape_hint: output_meta.exact_shape().or(fallback_shape_hint),
        inputs_map: Arc::new(inputs_map),
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: metadata_scopes_with_new(
            analysis.metadata,
            inherited_tensors
                .iter()
                .map(|tensor| tensor_metadata_scopes(tensor)),
        ),
        constraint_scope_transfer: ConstraintScopeTransfer::with_new(
            analysis.constraints,
            inherited_constraint_scopes.iter(),
        ),
    }))
}

fn missing_program_value(transform: &'static str, role: &'static str) -> Error {
    Error::runtime_state(
        transform,
        ErrorPhase::GraphBuild,
        format!("semantic derivative graph references missing {role}"),
    )
}

fn symbolic_input_shapes(frozen: &FrozenProgram) -> Result<Vec<Vec<SymDim>>> {
    frozen
        .program
        .inputs()
        .iter()
        .copied()
        .map(|value| {
            let meta = frozen.program.value_metadata(value).map_err(|source| {
                Error::runtime_state_source(
                    "semantic traced AD input metadata",
                    ErrorPhase::GraphBuild,
                    source,
                )
            })?;
            let tensor_id = allocate_shape_tensor_id();
            Ok((0..meta.shape().len())
                .map(|axis| SymDim::tensor_axis(tensor_id, axis))
                .collect())
        })
        .collect()
}

fn tensor_meta_for_value(
    frozen: &FrozenProgram,
    value: ProgramValue,
    input_shapes: &[&[SymDim]],
    transform: &'static str,
) -> Result<TensorMeta> {
    let meta = frozen
        .program
        .value_metadata(value)
        .map_err(|source| Error::runtime_state_source(transform, ErrorPhase::GraphBuild, source))?;
    Ok(program_metadata_to_tensor_meta(meta, input_shapes))
}

fn program_metadata_to_tensor_meta(
    metadata: &ProgramValueMetadata,
    input_shapes: &[&[SymDim]],
) -> TensorMeta {
    let extents = metadata
        .shape()
        .iter()
        .cloned()
        .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, input_shapes)))
        .collect();
    TensorMeta::with_extents(metadata.dtype(), extents)
}

fn validate_seed_tensor(
    transform: &'static str,
    input_index: usize,
    tensor: &RetainedValue,
    expected: &TensorMeta,
) -> Result<()> {
    let actual_dtype = tensor.dtype();
    if actual_dtype != expected.dtype {
        return Err(Error::invalid_argument(
            transform,
            ErrorPhase::GraphBuild,
            "seed",
            format!(
                "seed input {input_index} dtype mismatch: expected {:?}, got {:?}",
                expected.dtype, actual_dtype
            ),
        ));
    }
    let actual_shape = tensor.shape();
    if actual_shape.len() != expected.rank() {
        return Err(Error::invalid_argument(
            transform,
            ErrorPhase::GraphBuild,
            "seed",
            format!(
                "seed input {input_index} rank mismatch: expected {}, got {}",
                expected.rank(),
                actual_shape.len()
            ),
        ));
    }
    if let Some(expected_shape) = expected
        .exact_shape()
        .filter(|shape| shape.iter().all(|dim| dim.constant_value().is_some()))
        .map(|shape| {
            shape
                .into_iter()
                .map(|dim| dim.constant_value().expect("filtered constant shape"))
                .collect::<Vec<_>>()
        })
    {
        if expected_shape != actual_shape {
            return Err(Error::invalid_argument(
                transform,
                ErrorPhase::GraphBuild,
                "seed",
                format!(
                    "seed input {input_index} shape mismatch: expected {:?}, got {:?}",
                    expected_shape, actual_shape
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod semantic_transform_error_tests {
    use super::*;
    use crate::semantic_extension::SemanticAdRuleRole;

    #[test]
    fn unsupported_semantic_rule_maps_to_public_transform_error() {
        let source = SemanticAdTransformError::Extension(SemanticAdError::Unsupported {
            family_id: "tenferro-tests.unsupported.v1",
            role: SemanticAdRuleRole::LinearTranspose,
            message: "unsupported test payload".into(),
        });

        let error = semantic_transform_validation_error("vjp", &source)
            .expect("semantic rejection must map to a public unsupported-rule error");

        assert!(matches!(
            error,
            Error::UnsupportedAdRule { transform: "vjp", ref op }
                if op == "tenferro-tests.unsupported.v1"
        ));
    }

    #[test]
    fn missing_semantic_rule_maps_to_public_transform_error() {
        let source = SemanticAdTransformError::Extension(SemanticAdError::MissingRule {
            family_id: "tenferro-tests.missing.v1",
            role: SemanticAdRuleRole::Linearize,
        });

        let error = semantic_transform_validation_error("jvp", &source)
            .expect("missing semantic rule must map to a public unsupported-rule error");

        assert!(matches!(
            error,
            Error::UnsupportedAdRule { transform: "jvp", ref op }
                if op == "tenferro-tests.missing.v1"
        ));
    }
}
