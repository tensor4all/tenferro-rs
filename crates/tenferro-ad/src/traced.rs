//! Traced reverse- and forward-mode AD on the semantic program of a trace context.
//!
//! A differentiated output is snapshotted out of its [`TraceContext`], the
//! program-level transform runs on that snapshot, and the derivative program is
//! imported back into the same context, so gradients compose with further traced
//! operations and with higher-order transforms.

use std::collections::HashSet;
use std::sync::Arc;

use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::TensorMeta;
use tenferro_runtime::ad_support::{ones_tensor, RetainedValue};
use tenferro_runtime::program::{FrozenProgram, ProgramImport, ProgramValue};
use tenferro_runtime::{
    Error, ErrorPhase, ExecutionEngine, GraphCompiler, Result, Runtime, Tensor, TraceValue,
    TracedTensor,
};

use crate::semantic_extension::{SemanticAdError, SemanticExtensionRuleSet};
use crate::semantic_transform::{SemanticAdProgram, SemanticAdTransformError};
use crate::transform_cache::AdTransformCache;
use crate::AdContext;

fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .try_concrete_shape()
        .unwrap_or_else(|| vec![0; tensor.rank()])
}

/// Typed error for the documented, user-reachable inactive-`wrt` state.
///
/// The `_optional` siblings expose this state as `Ok(None)`; the strict wrapper
/// must not report a reachable input condition as an internal invariant.
pub(crate) fn inactive_wrt_error(op: &'static str, wrt_key: impl std::fmt::Debug) -> Error {
    Error::invalid_argument(
        op,
        ErrorPhase::GraphBuild,
        "wrt",
        format!(
            "{op} is inactive for {wrt_key:?}; use {op}_optional to observe the inactive state"
        ),
    )
}

fn snapshot_error(source: tenferro_runtime::program::ProgramFinishError) -> Error {
    Error::runtime_state_source("traced AD snapshot", ErrorPhase::GraphBuild, source)
}

fn import_error(source: tenferro_runtime::program::ProgramBuildError) -> Error {
    Error::runtime_state_source("traced AD import", ErrorPhase::GraphBuild, source)
}

/// One differentiated output's snapshot plus its live input interface.
struct AdSource {
    frozen: FrozenProgram,
    /// Context inputs in program input order.
    inputs: Vec<(TensorInputKey, TraceValue)>,
}

impl AdSource {
    fn of(output: &TracedTensor) -> Result<Self> {
        let (frozen, inputs) = output
            .context()
            .ad_snapshot_with_inputs(&[output.trace_value()])
            .map_err(snapshot_error)?;
        Ok(Self { frozen, inputs })
    }

    /// Program input index of `wrt`, when `wrt` is an input of this program.
    fn slot_of(&self, wrt: &TracedTensor) -> Result<Option<usize>> {
        let Some(key) = wrt.input_key() else {
            return Ok(None);
        };
        Ok(self
            .inputs
            .iter()
            .position(|(candidate, _)| *candidate == key))
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }
}

fn leaf_input_key(tensor: &TracedTensor) -> Result<TensorInputKey> {
    tensor.input_key().ok_or_else(|| {
        Error::invalid_argument(
            "traced AD",
            ErrorPhase::GraphBuild,
            "wrt",
            "the differentiated tensor is not a trace input",
        )
    })
}

fn seed_leaf(
    output: &TracedTensor,
    seeded_value: TraceValue,
    data: &Arc<RetainedValue>,
    op: &'static str,
) -> Result<TraceValue> {
    // A seed of an externally defined scalar must carry the same canonical scalar
    // identity as the value it seeds, or the program cannot express it.
    let identity = output.context().value_scalar_identity(seeded_value);
    output
        .context()
        .bound_input_concrete(Arc::clone(data), identity)
        .map_err(|source| {
            Error::invalid_argument(
                op,
                ErrorPhase::GraphBuild,
                "seed",
                format!("a seed value could not be bound into the trace: {source}"),
            )
        })
}

/// Import derivative roots back into the output's context.
///
/// The derivative program's inputs are the source inputs followed by the seeds
/// the transform introduced; the mapping reuses the live inputs so the imported
/// operations continue on the same leaves.
fn import_derivative(
    output: &TracedTensor,
    source: &AdSource,
    derivative: &SemanticAdProgram,
    seeds: &[(usize, TraceValue)],
    derivative_output_indices: &[usize],
    transform: &'static str,
) -> Result<Vec<TracedTensor>> {
    let frozen = derivative.frozen();
    let derivative_inputs = frozen.program.inputs();
    let source_input_count = source.input_count();
    let mut input_map: Vec<(ProgramValue, ProgramValue)> =
        Vec::with_capacity(source_input_count + seeds.len());
    for (index, (_, live)) in source.inputs.iter().enumerate() {
        let Some(&derivative_input) = derivative_inputs.get(index) else {
            return Err(Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                "the derivative program has fewer inputs than the source program",
            ));
        };
        input_map.push((derivative_input, live.program_value()));
    }
    for (seed_input_index, seed) in seeds {
        if *seed_input_index < source_input_count {
            return Err(Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                "a derivative seed input overlaps the source inputs",
            ));
        }
        let Some(&derivative_input) = derivative_inputs.get(*seed_input_index) else {
            return Err(Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                "a derivative seed input is outside the derivative program",
            ));
        };
        input_map.push((derivative_input, seed.program_value()));
    }
    let mut roots = Vec::with_capacity(derivative_output_indices.len());
    for &index in derivative_output_indices {
        let Some(&root) = frozen.program.outputs().get(index) else {
            return Err(Error::runtime_state(
                transform,
                ErrorPhase::GraphBuild,
                "a derivative output index is outside the derivative program",
            ));
        };
        roots.push(root);
    }
    let imported = output
        .context()
        .import_values(ProgramImport {
            program: frozen.program.as_ref(),
            bindings: None,
            roots: &roots,
            input_map: &input_map,
        })
        .map_err(import_error)?;
    imported
        .into_iter()
        .map(|value| {
            output.context().derive_tensor(value).map_err(|source| {
                Error::runtime_state_source(transform, ErrorPhase::GraphBuild, source)
            })
        })
        .collect()
}

/// Declared dtype and symbolic extents of a traced value.
fn tensor_meta_of(tensor: &TracedTensor) -> Result<TensorMeta> {
    let extents = tensor
        .sym_shape()
        .map(|shape| shape.to_vec())
        .unwrap_or_else(|| {
            (0..tensor.rank())
                .map(|axis| tenferro_ops::SymDim::tensor_axis(axis as u64, axis))
                .collect()
        });
    Ok(TensorMeta::exact(tensor.dtype(), extents))
}

/// Metadata of the differentiated output, used to validate a cotangent seed.
fn source_output_metadata(source: &AdSource, transform: &'static str) -> Result<TensorMeta> {
    let Some(&output) = source.frozen.program.outputs().first() else {
        return Err(Error::runtime_state(
            transform,
            ErrorPhase::GraphBuild,
            "the AD source program has no output",
        ));
    };
    let metadata = source
        .frozen
        .program
        .value_metadata(output)
        .map_err(|source| Error::runtime_state_source(transform, ErrorPhase::GraphBuild, source))?;
    program_metadata_to_tensor_meta(metadata, transform)
}

fn validate_seed_against_program(
    transform: &'static str,
    _source: &AdSource,
    data: &Arc<RetainedValue>,
    expected: TensorMeta,
) -> Result<()> {
    validate_seed_tensor(transform, 0, data, &expected)
}

fn program_metadata_to_tensor_meta(
    metadata: &tenferro_runtime::program::ProgramValueMetadata,
    transform: &'static str,
) -> Result<TensorMeta> {
    let extents = metadata
        .shape()
        .iter()
        .map(|extent| match extent {
            tenferro_ops::shape_extent::ShapeExtent::Exact(expr) => {
                tenferro_runtime::dim_expr_to_sym_dim_public(expr)
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::invalid_argument(
                transform,
                ErrorPhase::GraphBuild,
                "seed",
                "the differentiated value has no concrete dtype and rank metadata",
            )
        })?;
    Ok(TensorMeta::exact(metadata.dtype(), extents))
}

fn ad_context(
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> AdContext {
    match cache {
        Some(cache) => AdContext::with_rules_and_transform_cache(rules.clone(), Arc::clone(cache)),
        None => AdContext::builder()
            .with_semantic_extension_rules(rules.clone())
            .expect("registering semantic extension rules is infallible")
            .build()
            .expect("building an AD context with a rule set is infallible"),
    }
}

fn vjp_many_impl(
    output: &TracedTensor,
    wrts: &[&TracedTensor],
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
    transform: &'static str,
) -> Result<Vec<Option<TracedTensor>>> {
    if wrts.is_empty() {
        return Ok(Vec::new());
    }
    let source = AdSource::of(output)?;
    let slots: Vec<Option<usize>> = wrts
        .iter()
        .map(|wrt| source.slot_of(wrt))
        .collect::<Result<_>>()?;
    let mut active_inputs = vec![false; source.input_count()];
    for slot in slots.iter().flatten().copied() {
        active_inputs[slot] = true;
    }
    if !active_inputs.iter().any(|active| *active) {
        return Ok(vec![None; wrts.len()]);
    }
    let cotangent_data = cotangent.attached_value().cloned().ok_or_else(|| {
        Error::invalid_argument(
            transform,
            ErrorPhase::GraphBuild,
            "cotangent",
            format!("{transform} cotangent must have concrete tensor data"),
        )
    })?;
    validate_seed_against_program(
        transform,
        &source,
        &cotangent_data,
        source_output_metadata(&source, transform)?,
    )?;
    let seed = seed_leaf(output, output.trace_value(), &cotangent_data, transform)?;
    let active_outputs = vec![true; source.frozen.program.outputs().len()];
    let derivative = ad_context(rules, cache)
        .vjp_program(&source.frozen, &active_inputs, &active_outputs)
        .map_err(semantic_transform_error(transform))?;
    let Some(seed_input_index) = derivative
        .derivative_input_indices()
        .first()
        .copied()
        .flatten()
    else {
        return Ok(vec![None; wrts.len()]);
    };
    let mut requested = Vec::with_capacity(wrts.len());
    let mut roots = Vec::new();
    for (index, slot) in slots.iter().enumerate() {
        let derivative_output = slot.and_then(|slot| {
            derivative
                .derivative_output_indices()
                .get(slot)
                .copied()
                .flatten()
        });
        requested.push((index, derivative_output));
        if let Some(derivative_output) = derivative_output
            && !roots.contains(&derivative_output)
        {
            roots.push(derivative_output);
        }
    }
    if roots.is_empty() {
        return Ok(vec![None; wrts.len()]);
    }
    let imported = import_derivative(
        output,
        &source,
        &derivative,
        &[(seed_input_index, seed)],
        &roots,
        transform,
    )?;
    let mut results = vec![None; wrts.len()];
    for (index, derivative_output) in requested {
        if let Some(derivative_output) = derivative_output
            && let Some(position) = roots.iter().position(|root| *root == derivative_output)
        {
            results[index] = imported.get(position).cloned();
        }
    }
    Ok(results)
}

fn jvp_many_impl(
    output: &TracedTensor,
    wrt_tangents: &[(&TracedTensor, &TracedTensor)],
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
    transform: &'static str,
) -> Result<Option<TracedTensor>> {
    if wrt_tangents.is_empty() {
        return Ok(None);
    }
    let source = AdSource::of(output)?;
    let mut seen = HashSet::with_capacity(wrt_tangents.len());
    let mut active_inputs = vec![false; source.input_count()];
    let mut tangents = Vec::with_capacity(wrt_tangents.len());
    for (wrt, tangent) in wrt_tangents {
        let key = leaf_input_key(wrt)?;
        if !seen.insert(key.clone()) {
            return Err(Error::invalid_argument(
                transform,
                ErrorPhase::GraphBuild,
                "wrt_tangents",
                "wrt_tangents contains duplicate wrt leaves",
            ));
        }
        let tangent_data = tangent.attached_value().cloned().ok_or_else(|| {
            Error::invalid_argument(
                transform,
                ErrorPhase::GraphBuild,
                "tangent",
                format!("{transform} tangent must have concrete tensor data"),
            )
        })?;
        tangents.push((wrt, tangent_data));
    }
    for (index, (wrt, _)) in tangents.iter().enumerate() {
        let Some(slot) = source.slot_of(wrt)? else {
            continue;
        };
        active_inputs[slot] = true;
        let _ = index;
    }
    if !active_inputs.iter().any(|active| *active) {
        return Ok(None);
    }
    let derivative = ad_context(rules, cache)
        .jvp_program(&source.frozen, &active_inputs)
        .map_err(semantic_transform_error(transform))?;
    let Some(derivative_output_index) = derivative
        .derivative_output_indices()
        .first()
        .copied()
        .flatten()
    else {
        return Ok(None);
    };
    let mut seeds = Vec::with_capacity(tangents.len());
    for (wrt, data) in tangents {
        let Some(slot) = source.slot_of(wrt)? else {
            continue;
        };
        let Some(seed_input_index) = derivative
            .derivative_input_indices()
            .get(slot)
            .copied()
            .flatten()
        else {
            continue;
        };
        validate_seed_against_program(transform, &source, &data, tensor_meta_of(wrt)?)?;
        let seed = seed_leaf(output, wrt.trace_value(), &data, transform)?;
        seeds.push((seed_input_index, seed));
    }
    if seeds.is_empty() {
        return Ok(None);
    }
    let mut imported = import_derivative(
        output,
        &source,
        &derivative,
        &seeds,
        &[derivative_output_index],
        transform,
    )?;
    Ok(imported.pop())
}

pub(crate) fn grad_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<TracedTensor> {
    let key = leaf_input_key(wrt)?;
    grad_optional_with_rules_and_cache(output, wrt, rules, cache)
        .and_then(|gradient| gradient.ok_or_else(|| inactive_wrt_error("grad", &key)))
}

pub(crate) fn grad_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<Option<TracedTensor>> {
    if output.rank() != 0 {
        return Err(Error::NonScalarGrad {
            shape: error_shape_hint(output),
        });
    }
    let ones = ones_tensor(output.dtype(), vec![])?;
    let seed = output
        .context()
        .from_tensor_concrete_shape(ones)
        .map_err(|source| Error::runtime_state_source("grad", ErrorPhase::GraphBuild, source))?;
    vjp_many_impl(output, &[wrt], &seed, rules, cache, "grad")
        .map(|mut results| results.pop().flatten())
}

pub(crate) fn jvp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<TracedTensor> {
    let key = leaf_input_key(wrt)?;
    jvp_many_impl(output, &[(wrt, tangent)], rules, cache, "jvp")
        .and_then(|result| result.ok_or_else(|| inactive_wrt_error("jvp", &key)))
}

pub(crate) fn jvp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<Option<TracedTensor>> {
    jvp_many_impl(output, &[(wrt, tangent)], rules, cache, "jvp")
}

pub(crate) fn jvp_many_with_rules_and_cache(
    output: &TracedTensor,
    wrt_tangents: &[(&TracedTensor, &TracedTensor)],
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<Option<TracedTensor>> {
    jvp_many_impl(output, wrt_tangents, rules, cache, "jvp")
}

pub(crate) fn vjp_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<TracedTensor> {
    let key = leaf_input_key(wrt)?;
    vjp_optional_with_rules_and_cache(output, wrt, cotangent, rules, cache)
        .and_then(|gradient| gradient.ok_or_else(|| inactive_wrt_error("vjp", &key)))
}

pub(crate) fn vjp_optional_with_rules_and_cache(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<Option<TracedTensor>> {
    vjp_many_impl(output, &[wrt], cotangent, rules, cache, "vjp")
        .map(|mut results| results.pop().flatten())
}

pub(crate) fn vjp_many_with_rules_and_cache(
    output: &TracedTensor,
    wrts: &[&TracedTensor],
    cotangent: &TracedTensor,
    rules: &SemanticExtensionRuleSet,
    cache: Option<&Arc<AdTransformCache>>,
) -> Result<Vec<Option<TracedTensor>>> {
    vjp_many_impl(output, wrts, cotangent, rules, cache, "vjp")
}

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
    /// use tenferro_runtime::{TraceContext, GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut engine = tenferro_cpu::CpuEngine::serial();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     let target = tenferro_runtime::ExecutionEngine::target(&engine);
    ///     let prepared = runtime.prepare(&program, &[], &[target]).unwrap();
    ///     let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[]).unwrap();
    ///     outputs.pop().unwrap().into_owned().unwrap()
    /// }
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
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
    /// is unavailable, or a typed validation/backend/runtime-state error. An
    /// inactive `wrt` returns [`Error::Validation`] with `argument: "wrt"`; use
    /// [`grad_optional`](Self::grad_optional) to observe that state.
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
    /// use tenferro_runtime::{TraceContext, TracedTensor};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = trace.from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
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
    /// use tenferro_runtime::{TraceContext, GraphCompiler, Runtime, TracedTensor};
    ///
    /// let mut compiler = GraphCompiler::new();
    /// let backend = tenferro_cpu::CpuEngine::serial();
    /// let mut builder = Runtime::builder();
    /// builder
    ///     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
    ///     .unwrap();
    /// let runtime = builder.build().unwrap();
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let mut y = (&x * &x).unwrap();
    ///
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// y.checkpoint(&mut compiler, &runtime, &mut engine).unwrap();
    ///
    /// let value = y.attached_value().unwrap();
    /// let materialized = value.tensor_read().unwrap().tensor_view().duplicate().unwrap();
    /// assert_eq!(materialized.as_slice::<f64>().unwrap(), &[9.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::Validation`] when checkpoint metadata
    /// is invalid, [`Error::RuntimeState`] when graph metadata or runtime
    /// state is unavailable, or a typed backend error from evaluation.
    fn checkpoint(
        &mut self,
        compiler: &mut GraphCompiler,
        runtime: &Runtime,
        engine: &mut dyn ExecutionEngine,
    ) -> Result<()>;

    /// Forward-mode Jacobian-vector product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::{TraceContext, GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut engine = tenferro_cpu::CpuEngine::serial();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     let target = tenferro_runtime::ExecutionEngine::target(&engine);
    ///     let prepared = runtime.prepare(&program, &[], &[target]).unwrap();
    ///     let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[]).unwrap();
    ///     outputs.pop().unwrap().into_owned().unwrap()
    /// }
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let tangent = trace.from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
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
    /// or a typed backend/runtime-state error. An inactive `wrt` returns
    /// [`Error::Validation`] with `argument: "wrt"`; use
    /// [`jvp_optional`](Self::jvp_optional) to observe that state.
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
    /// use tenferro_runtime::{TraceContext, TracedTensor};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = trace.from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let tangent = trace.from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
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
    /// use tenferro_runtime::{TraceContext, GraphCompiler, Runtime, TracedTensor};
    ///
    /// fn eval(tensor: &TracedTensor) -> tenferro_runtime::Tensor {
    ///     let mut compiler = GraphCompiler::new();
    ///     let program = compiler.compile(tensor).unwrap();
    ///     let mut engine = tenferro_cpu::CpuEngine::serial();
    ///     let mut builder = Runtime::builder();
    ///     builder
    ///         .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
    ///         .unwrap();
    ///     let runtime = builder.build().unwrap();
    ///     let target = tenferro_runtime::ExecutionEngine::target(&engine);
    ///     let prepared = runtime.prepare(&program, &[], &[target]).unwrap();
    ///     let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[]).unwrap();
    ///     outputs.pop().unwrap().into_owned().unwrap()
    /// }
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let cotangent = trace.from_vec_col_major(vec![], vec![0.5_f64]).unwrap();
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
    /// metadata, or a typed backend/runtime-state error. An inactive `wrt`
    /// returns [`Error::Validation`] with `argument: "wrt"`; use
    /// [`vjp_optional`](Self::vjp_optional) to observe that state.
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
    /// use tenferro_runtime::{TraceContext, TracedTensor};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = trace.from_vec_col_major(vec![], vec![4.0_f64]).unwrap();
    /// let cotangent = trace.from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
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

    /// Reverse-mode gradients for several leaves from one transform.
    ///
    /// One snapshot, one program-level transform and one import cover the whole
    /// `wrt` set, so the backward is a single sweep instead of one per leaf.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::{TraceContext, DType};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let w = trace.from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    /// let loss = (&(&x * &w)?).reduce_sum(None)?;
    /// let grads = loss.grad_many(&[&x, &w])?;
    /// assert_eq!(grads.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonScalarGrad`] for a non-scalar output,
    /// [`Error::UnsupportedAdRule`] when a rule is unavailable, or a typed
    /// validation/runtime-state error from the snapshot or the transform.
    fn grad_many(&self, wrts: &[&TracedTensor]) -> Result<Vec<Option<TracedTensor>>>;

    /// The scalar output together with reverse-mode gradients for several leaves.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::TracedTensorAdExt;
    /// use tenferro_runtime::{TraceContext, DType};
    ///
    /// let trace = TraceContext::new();
    /// let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let loss = (&(&x * &x)?).reduce_sum(None)?;
    /// let (value, grads) = loss.value_and_grad(&[&x])?;
    /// assert_eq!(value.rank(), 0);
    /// assert!(grads[0].is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`grad_many`](Self::grad_many).
    fn value_and_grad(
        &self,
        wrts: &[&TracedTensor],
    ) -> Result<(TracedTensor, Vec<Option<TracedTensor>>)>;

    /// Forward-mode JVPs for several leaves from one transform.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedAdRule`] when a rule is unavailable, or a
    /// typed validation/runtime-state error from the snapshot or the transform.
    fn jvp_many(
        &self,
        wrt_tangents: &[(&TracedTensor, &TracedTensor)],
    ) -> Result<Option<TracedTensor>>;

    /// Reverse-mode VJPs for several leaves from one transform.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedAdRule`] when a rule is unavailable, or a
    /// typed validation/runtime-state error from the snapshot or the transform.
    fn vjp_many(
        &self,
        wrts: &[&TracedTensor],
        cotangent: &TracedTensor,
    ) -> Result<Vec<Option<TracedTensor>>>;
}

impl TracedTensorAdExt for TracedTensor {
    fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor> {
        grad_with_rules_and_cache(self, wrt, &SemanticExtensionRuleSet::default(), None)
    }

    fn grad_optional(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        grad_optional_with_rules_and_cache(self, wrt, &SemanticExtensionRuleSet::default(), None)
    }

    fn checkpoint(
        &mut self,
        compiler: &mut GraphCompiler,
        runtime: &Runtime,
        engine: &mut dyn ExecutionEngine,
    ) -> Result<()> {
        let data = if let Some(data) = self.attached_value() {
            Arc::clone(data)
        } else {
            let program = compiler.compile(self)?;
            let target = engine.target();
            let prepared = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
            let outputs = crate::eager::run_prepared_on(runtime, engine, &prepared, &[])?;
            Arc::new(RetainedValue::from_tensor(single_runtime_output(
                outputs,
                "TracedTensorAdExt::checkpoint",
            )?))
        };
        let value = self
            .context()
            .checkpoint_value(self.trace_value(), Arc::clone(&data))
            .map_err(|source| {
                Error::runtime_state_source(
                    "TracedTensorAdExt::checkpoint",
                    ErrorPhase::GraphBuild,
                    source,
                )
            })?;
        self.rebind_bound(value, data)
    }

    fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Result<TracedTensor> {
        jvp_with_rules_and_cache(
            self,
            wrt,
            tangent,
            &SemanticExtensionRuleSet::default(),
            None,
        )
    }

    fn jvp_optional(
        &self,
        wrt: &TracedTensor,
        tangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        jvp_optional_with_rules_and_cache(
            self,
            wrt,
            tangent,
            &SemanticExtensionRuleSet::default(),
            None,
        )
    }

    fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Result<TracedTensor> {
        vjp_with_rules_and_cache(
            self,
            wrt,
            cotangent,
            &SemanticExtensionRuleSet::default(),
            None,
        )
    }

    fn vjp_optional(
        &self,
        wrt: &TracedTensor,
        cotangent: &TracedTensor,
    ) -> Result<Option<TracedTensor>> {
        vjp_optional_with_rules_and_cache(
            self,
            wrt,
            cotangent,
            &SemanticExtensionRuleSet::default(),
            None,
        )
    }

    fn grad_many(&self, wrts: &[&TracedTensor]) -> Result<Vec<Option<TracedTensor>>> {
        if self.rank() != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }
        let ones = ones_tensor(self.dtype(), vec![])?;
        let seed = self
            .context()
            .from_tensor_concrete_shape(ones)
            .map_err(|source| {
                Error::runtime_state_source("grad_many", ErrorPhase::GraphBuild, source)
            })?;
        vjp_many_impl(
            self,
            wrts,
            &seed,
            &SemanticExtensionRuleSet::default(),
            None,
            "grad_many",
        )
    }

    fn value_and_grad(
        &self,
        wrts: &[&TracedTensor],
    ) -> Result<(TracedTensor, Vec<Option<TracedTensor>>)> {
        let gradients = self.grad_many(wrts)?;
        Ok((self.clone(), gradients))
    }

    fn jvp_many(
        &self,
        wrt_tangents: &[(&TracedTensor, &TracedTensor)],
    ) -> Result<Option<TracedTensor>> {
        jvp_many_impl(
            self,
            wrt_tangents,
            &SemanticExtensionRuleSet::default(),
            None,
            "jvp_many",
        )
    }

    fn vjp_many(
        &self,
        wrts: &[&TracedTensor],
        cotangent: &TracedTensor,
    ) -> Result<Vec<Option<TracedTensor>>> {
        vjp_many_with_rules_and_cache(
            self,
            wrts,
            cotangent,
            &SemanticExtensionRuleSet::default(),
            None,
        )
    }
}

fn single_runtime_output(mut outputs: Vec<Tensor>, op: &'static str) -> Result<Tensor> {
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "{op}: expected exactly one execution output, got {}",
            outputs.len()
        )));
    }
    Ok(outputs.remove(0))
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
        && expected_shape != actual_shape
    {
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
