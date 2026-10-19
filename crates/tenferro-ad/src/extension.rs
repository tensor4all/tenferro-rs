//! Eager AD support for out-of-tree extension primitives.

use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{
    Error, ErrorPhase, ExtensionModule, InputSignature, PrepareCapability, PrepareError,
    PreparedOperationExecutorHandle, Result, Runtime, RuntimeConfigError,
};
use tenferro_tensor::{Tensor, TensorRead, TensorValue};

use crate::eager::{
    eager_capture_active, eager_grad_recording_enabled, execute_prepared_extension_on,
    EagerRuntime, EagerSession, EagerTensor,
};
use tenferro_runtime::ExecutionEngine;

pub use tenferro_runtime::extension::{
    apply, ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
    ExtensionExecutionContext, ExtensionFamilyId, ExtensionOp,
};

/// Closed backend kind selected by the eager runtime owner for an extension.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::{EagerExtensionBackendKind, EagerExtensionTarget};
/// use tenferro_runtime::EngineId;
///
/// let target = EagerExtensionTarget {
///     engine_id: EngineId::new("example.engine")?,
///     backend_kind: EagerExtensionBackendKind::Cpu,
/// };
/// assert!(matches!(
///     target.backend_kind,
///     EagerExtensionBackendKind::Cpu
/// ));
/// assert_eq!(target.engine_id.as_str(), "example.engine");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EagerExtensionBackendKind {
    /// The eager runtime owns a CPU backend.
    Cpu,
    /// The eager runtime owns a CUDA backend.
    #[cfg(feature = "cuda")]
    Cuda,
    /// The eager runtime owns a WebGPU backend.
    #[cfg(feature = "webgpu")]
    WebGpu,
}

/// Exact engine target selected by the eager runtime owner.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::{EagerExtensionBackendKind, EagerExtensionTarget};
/// use tenferro_runtime::EngineId;
///
/// let target = EagerExtensionTarget {
///     engine_id: EngineId::new("example.engine")?,
///     backend_kind: EagerExtensionBackendKind::Cpu,
/// };
/// assert_eq!(target.backend_kind, EagerExtensionBackendKind::Cpu);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerExtensionTarget {
    /// Exact runtime engine selected for this eager context.
    pub engine_id: tenferro_runtime::EngineId,
    /// Closed backend kind selected for this eager context.
    pub backend_kind: EagerExtensionBackendKind,
}

#[cfg(test)]
mod tests;

/// Validate an untracked eager input for in-place execution on `engine` and
/// install the extension module the in-place kernel needs.
///
/// # Errors
///
/// Returns a runtime-state error for a tracked input or inside trace capture,
/// and the runtime's typed error when the engine cannot accept the input or
/// the module cannot be installed.
#[doc(hidden)]
pub fn prepare_eager_in_place_input(
    engine: &mut dyn ExecutionEngine,
    input: &EagerTensor,
    family_id: &'static str,
    module_factory: impl FnOnce(EagerExtensionTarget) -> Result<Arc<dyn ExtensionModule>>,
) -> Result<()> {
    if input.requires_grad || input.trace.is_some() || eager_capture_active() {
        return Err(Error::runtime_state(
            "eager in-place",
            ErrorPhase::Execution,
            "in-place execution requires an untracked value outside trace capture",
        ));
    }
    let target = input.ctx.eager_extension_target(engine)?;
    validate_eager_extension_input_signature(&input.ctx, &target, &[input.tensor_read()])?;
    let module = module_factory(target.clone())?;
    input
        .ctx
        .ensure_extension_module_for_engine(module, family_id, &target.engine_id)?;
    Ok(())
}

/// Adopt a runtime value produced by an extension kernel as an untracked eager
/// tensor of `ctx`.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, Tensor};
/// use tenferro_tensor::TensorValue;
///
/// let ctx = EagerRuntime::new()?;
/// let value = TensorValue::from_tensor(Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?);
/// let tensor = tenferro_ad::extension::adopt_untracked_eager_value(ctx, value)?;
/// assert!(!tensor.tracks_grad());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns a runtime-state error when the value cannot be registered.
#[must_use = "the adopted eager tensor carries the runtime value"]
pub fn adopt_untracked_eager_value(
    ctx: Arc<EagerRuntime>,
    value: TensorValue,
) -> Result<EagerTensor> {
    EagerTensor::new_untracked_value_result(ctx, value)
}

/// Apply an extension op to eager tensors on `engine`, recording it for AD.
///
/// # Errors
///
/// Returns an input-count or context-mismatch error, a runtime error when no
/// installed extension module serves the op on this engine, or the extension
/// kernel's own typed error.
pub fn apply_eager(
    engine: &mut dyn ExecutionEngine,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ctx.with_eager_session(engine, |session| {
        apply_eager_in_session(session, op, inputs)
    })
}

/// Apply an extension op inside an eager session, preferring a prepared
/// session executor and falling back to a compiled one-op program.
fn apply_eager_in_session(
    session: &mut EagerSession<'_>,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ensure_session_runtime(session, &ctx)?;
    let std_op = StdTensorOp::Extension(Arc::clone(&op));
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    let outputs = match try_prepared_eager_extension(session, &std_op, &input_reads)? {
        Some(outputs) => outputs,
        None => ctx.exec_extension_outputs_read(session.engine(), &op, &input_reads)?,
    };
    finish_eager_extension_outputs(ctx, std_op, inputs, outputs, session)
}

fn ensure_session_runtime(session: &EagerSession<'_>, ctx: &Arc<EagerRuntime>) -> Result<()> {
    if !Arc::ptr_eq(session.runtime(), ctx) {
        return Err(Error::ContextMismatch {
            lhs: session.runtime().id(),
            rhs: ctx.id(),
        });
    }
    Ok(())
}

/// Run a prepared extension executor on `engine`: through the session-capable
/// entry when it has one, otherwise through the typed engine escape.
fn execute_prepared_executor(
    engine: &mut dyn ExecutionEngine,
    executor: &dyn tenferro_runtime::PreparedOperationExecutor,
    input_reads: &[TensorRead<'_>],
) -> Result<Vec<Tensor>> {
    if executor.supports_session() {
        return execute_prepared_extension_on(engine, executor, input_reads);
    }
    let identity = engine.target().context_identity();
    let mut caches = engine.take_extension_caches();
    let result = {
        let mut erased =
            tenferro_runtime::ErasedExecutionContext::from_dyn(identity, engine.as_any_mut());
        executor.execute(&mut erased, &mut caches, input_reads)
    };
    engine.restore_extension_caches(caches);
    result
}

fn try_prepared_eager_extension(
    session: &mut EagerSession<'_>,
    op: &StdTensorOp,
    input_reads: &[TensorRead<'_>],
) -> Result<Option<Vec<Tensor>>> {
    let ctx = Arc::clone(session.runtime());
    // An engine kind without an eager extension target falls back to the
    // compiled one-op program.
    let Ok(target) = ctx.eager_extension_target(session.engine()) else {
        return Ok(None);
    };
    let Some(executor) = prepared_eager_extension_executor(&ctx, &target, op, input_reads)? else {
        return Ok(None);
    };
    Ok(Some(execute_prepared_executor(
        session.engine(),
        executor.as_ref(),
        input_reads,
    )?))
}

fn prepared_eager_extension_executor(
    ctx: &EagerRuntime,
    target: &EagerExtensionTarget,
    op: &StdTensorOp,
    input_reads: &[TensorRead<'_>],
) -> Result<Option<PreparedOperationExecutorHandle>> {
    let StdTensorOp::Extension(ext) = op else {
        return Ok(None);
    };
    let signature = InputSignature::from_reads(input_reads).map_err(|source| {
        Error::runtime_state_source("extension::apply_eager", ErrorPhase::Execution, source)
    })?;
    let PrepareCapability::Prepared(plan) =
        ctx.runtime()
            .prepare_extension_immediate(&target.engine_id, ext.as_ref(), &signature)?
    else {
        return Ok(None);
    };
    Ok(plan.executor().cloned())
}

/// Install `module` and apply `op` on `engine`.
///
/// # Errors
///
/// As [`apply_eager`], plus the runtime's error when the module cannot be
/// installed.
#[doc(hidden)]
pub fn apply_eager_with_extension_session(
    engine: &mut dyn ExecutionEngine,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module: Arc<dyn ExtensionModule>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ctx.install_extension_module(module)?;
    apply_eager(engine, op, inputs)
}

/// Install the module `module_factory` builds for `engine`'s target and apply
/// `op` on `engine`.
///
/// # Errors
///
/// As [`apply_eager`], plus input-ingress and module-installation errors.
#[doc(hidden)]
pub fn apply_eager_with_targeted_extension_session(
    engine: &mut dyn ExecutionEngine,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let target = ctx.eager_extension_target(engine)?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    validate_eager_extension_input_signature(&ctx, &target, &input_reads)?;
    let module = module_factory(target.clone())?;
    ctx.ensure_extension_module_for_engine(module, op.family_id(), &target.engine_id)?;
    apply_eager(engine, op, inputs)
}

/// [`apply_eager_with_targeted_extension_session`] inside an eager session.
///
/// # Errors
///
/// As [`apply_eager_with_targeted_extension_session`], plus a context mismatch
/// when the inputs belong to another runtime.
#[doc(hidden)]
pub fn apply_eager_with_targeted_extension_in_session(
    session: &mut EagerSession<'_>,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ensure_session_runtime(session, &ctx)?;
    let target = ctx.eager_extension_target(session.engine())?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    validate_eager_extension_input_signature(&ctx, &target, &input_reads)?;
    let module = module_factory(target.clone())?;
    ctx.ensure_extension_module_for_engine(module, op.family_id(), &target.engine_id)?;
    apply_eager_in_session(session, op, inputs)
}

pub(crate) fn validate_eager_extension_target(
    runtime: &Runtime,
    target: &EagerExtensionTarget,
) -> Result<()> {
    let snapshot = runtime.snapshot().map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    if snapshot.engine(&target.engine_id).is_none() {
        return Err(Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingEngine {
                engine_id: target.engine_id.clone(),
            },
        ));
    }
    Ok(())
}

fn validate_eager_extension_input_signature(
    ctx: &EagerRuntime,
    target: &EagerExtensionTarget,
    input_reads: &[TensorRead<'_>],
) -> Result<()> {
    let signature = InputSignature::from_reads(input_reads).map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    let snapshot = ctx.runtime().snapshot().map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    let engine = snapshot.engine(&target.engine_id).ok_or_else(|| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingEngine {
                engine_id: target.engine_id.clone(),
            },
        )
    })?;
    for (input_index, entry) in signature.entries().iter().enumerate() {
        if !engine.accepts_input_signature(entry) {
            return Err(Error::runtime_state_source(
                "extension::apply_eager_with_extension_session",
                ErrorPhase::Execution,
                PrepareError::NoInputIngress {
                    input_index,
                    placement: entry.placement().clone(),
                },
            ));
        }
    }
    Ok(())
}

fn validate_eager_extension_inputs(
    op: &dyn ExtensionOp,
    inputs: &[&EagerTensor],
) -> Result<Arc<EagerRuntime>> {
    let Some(first) = inputs.first() else {
        return Err(Error::invalid_argument(
            "extension::apply_eager",
            ErrorPhase::Execution,
            "inputs",
            "at least one input tensor is required",
        ));
    };
    if inputs.len() != op.input_count() {
        return Err(Error::invalid_argument(
            "extension::apply_eager",
            ErrorPhase::Execution,
            "inputs",
            format!(
                "op family {:?} expects {} inputs, got {}",
                op.family_id(),
                op.input_count(),
                inputs.len()
            ),
        ));
    }

    let ctx = Arc::clone(&first.ctx);
    for tensor in inputs.iter().skip(1) {
        if !first.same_context(tensor) {
            return Err(Error::ContextMismatch {
                lhs: first.ctx_id(),
                rhs: tensor.ctx_id(),
            });
        }
    }
    Ok(ctx)
}

fn finish_eager_extension_outputs(
    ctx: Arc<EagerRuntime>,
    op: StdTensorOp,
    inputs: &[&EagerTensor],
    outputs: Vec<Tensor>,
    session: &mut EagerSession<'_>,
) -> Result<Vec<EagerTensor>> {
    if outputs.len() != op.output_count() {
        return Err(Error::Internal(format!(
            "expected {} eager outputs for {:?}, got {}",
            op.output_count(),
            op,
            outputs.len()
        )));
    }

    if !eager_grad_recording_enabled()
        || (!eager_capture_active() && !inputs.iter().any(|input| input.requires_grad))
    {
        return outputs
            .into_iter()
            .map(|output| EagerTensor::new_untracked_result(Arc::clone(&ctx), output))
            .collect();
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let recorded = session.record_outputs(&op, &output_refs, inputs)?;
    if recorded.traces.len() != outputs.len() {
        return Err(Error::Internal(format!(
            "expected {} eager traces for {:?}, got {}",
            outputs.len(),
            op,
            recorded.traces.len()
        )));
    }
    let results = recorded
        .traces
        .into_iter()
        .zip(recorded.semantic_traces)
        .zip(outputs)
        .map(|((trace, semantic_trace), output)| {
            if trace.requires_grad {
                EagerTensor::new_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                )
            } else {
                EagerTensor::new_unregistered_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                )
            }
        })
        .collect::<Result<Vec<_>>>()?;
    crate::eager::finish_residuals(&op, inputs, &results.iter().collect::<Vec<_>>())?;
    Ok(results)
}
