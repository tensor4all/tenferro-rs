//! Eager AD support for out-of-tree extension primitives.

use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ad_support::push_metadata_scope;
use tenferro_runtime::{
    Error, ErrorPhase, ExtensionModule, InputSignature, PrepareCapability, PrepareError, Result,
    Runtime, RuntimeConfigError,
};
use tenferro_tensor::{Tensor, TensorRead, TensorValue};

use crate::eager::{
    eager_capture_active, eager_grad_recording_enabled, record_eager_outputs, EagerRuntime,
    EagerTensor,
};

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

/// Adopt an untracked eager tensor value produced by this runtime's backend.
///
/// This is a low-level extension contract for eager composite operations that
/// execute through a lifetime-bound backend session and receive a lazy
/// [`TensorValue`] from the backend. The value must have been produced for the
/// same eager runtime; this helper intentionally does not register gradient
/// metadata and must not be used for tracked outputs.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::adopt_untracked_eager_value;
/// use tenferro_ad::EagerRuntime;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{Tensor, TensorValue};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let value = TensorValue::from_tensor(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
/// );
/// let eager = adopt_untracked_eager_value(ctx, value)?;
/// assert_eq!(eager.shape(), &[1]);
/// assert!(!eager.tracks_grad());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
/// # Errors
///
/// Returns [`Error::RuntimeState`] when the value cannot be registered in the
/// supplied runtime, including an invalid or incompatible retained descriptor.
#[must_use = "the adopted eager tensor carries the runtime value"]
pub fn adopt_untracked_eager_value(
    ctx: Arc<EagerRuntime>,
    value: TensorValue,
) -> Result<EagerTensor> {
    EagerTensor::new_untracked_value_result(ctx, value)
}

/// Apply an extension op to eager AD tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::apply_eager;
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::Tensor;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
///     ctx,
/// ).unwrap();
/// let _ = &x;
/// let _apply = apply_eager;
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
/// # Errors
///
/// Returns `Error::Validation` with `InvalidArgument` when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// `Error::ContextMismatch` when tensors belong to different eager runtimes;
/// backend, extension, and runtime-state failures retain their typed sources.
pub fn apply_eager(op: Arc<dyn ExtensionOp>, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let std_op = StdTensorOp::Extension(op);
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    // Native immediate path: resolve the extension engine from the runtime
    // snapshot, prepare the op, and execute through the prepared plan's
    // scheduler-session executor when it is session-capable. This skips the
    // SemanticProgram build/compile + run_compiled cost on every call.
    if let Some(outputs) = try_prepared_eager_extension(&ctx, &std_op, &input_reads)? {
        return finish_eager_extension_outputs(ctx, std_op, inputs, outputs);
    }
    let outputs = ctx.exec_outputs_read(&std_op, &input_reads)?;
    finish_eager_extension_outputs(ctx, std_op, inputs, outputs)
}

/// Run one extension op through the snapshot-resolved native prepared path.
///
/// Returns `None` (so the caller falls back to the compiled-program path for
/// the exact op and signature) when the eager runtime has no exact extension
/// engine, the engine has no slot for the op's family and cannot prepare it,
/// or the prepared plan has no scheduler-session executor. AD recording is
/// never touched here; the caller owns `finish_eager_extension_outputs`.
fn try_prepared_eager_extension(
    ctx: &EagerRuntime,
    op: &StdTensorOp,
    input_reads: &[TensorRead<'_>],
) -> Result<Option<Vec<Tensor>>> {
    let StdTensorOp::Extension(ext) = op else {
        return Ok(None);
    };
    // The eager runtime owns its exact extension engine; provider selection
    // must not wander to a different engine that happens to be first in slot
    // order. If the target is unavailable (e.g. the recording test backend),
    // fall back to the compiled path.
    let Ok(target) = ctx.eager_extension_target() else {
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
    let Some(executor) = plan.executor() else {
        return Ok(None);
    };
    let executor = Arc::clone(executor);
    if executor.supports_session() {
        // native-session: scheduler-owned session executor.
        let outputs = ctx.with_extension_execution_context(|extension_ctx| {
            let (session, caches) = extension_ctx.parts_mut();
            executor.execute_in_session(session, caches, input_reads)
        })??;
        Ok(Some(outputs))
    } else {
        // native-context: mandatory `execute` bridge over the erased backend
        // context (for out-of-tree prepared ops without a session executor).
        let outputs = ctx.with_extension_erased_context(|erased, caches| {
            executor.execute(erased, caches, input_reads)
        })??;
        Ok(Some(outputs))
    }
}

/// Ensure an eager extension module is installed, then apply the op through
/// the single [`apply_eager`] entry.
///
/// This thin wrapper is retained for module-owner eager call sites (linalg,
/// einsum). Forward execution always routes through [`apply_eager`]'s native
/// prepared path; this wrapper only owns the install-ensure step.
///
/// # Errors
///
/// Returns `Error::Validation` with `InvalidArgument` when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// `Error::ContextMismatch` when tensors belong to different eager runtimes;
/// backend, extension, and runtime-state failures retain their typed sources.
#[doc(hidden)]
pub fn apply_eager_with_extension_session(
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module: Arc<dyn ExtensionModule>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ctx.install_extension_module(module)?;
    apply_eager(op, inputs)
}

/// Apply an eager extension through the owner-selected engine and backend kind.
///
/// This narrow sibling-crate wrapper is used by FFT, whose module factory must
/// follow the eager runtime's exact backend selection. Input, target, and
/// ingress validation always run before `module_factory`; errors returned by
/// the factory are propagated unchanged. The returned module is then passed to
/// the owner-scoped ensure operation. Forward execution always routes through
/// [`apply_eager`]'s native prepared path.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::Validation`] with
/// [`tenferro_tensor::ValidationError::InvalidArgument`] when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// [`tenferro_runtime::Error::ContextMismatch`] when tensors belong to
/// different eager runtimes.
///
/// Returns [`tenferro_runtime::Error::RuntimeStateSource`] when the selected
/// engine is missing through
/// [`tenferro_runtime::RuntimeConfigError::MissingEngine`], an input has no
/// ingress through [`tenferro_runtime::PrepareError::NoInputIngress`], or the
/// cold/missing-registration ensure path rejects the module. Errors returned by
/// `module_factory` are propagated unchanged. Session, cache, and output
/// registration failures retain their typed runtime sources.
#[doc(hidden)]
pub fn apply_eager_with_targeted_extension_session(
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let target = ctx.eager_extension_target()?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    validate_eager_extension_input_signature(&ctx, &target, &input_reads)?;
    let module = module_factory(target.clone())?;
    ctx.ensure_extension_module_for_engine(module, op.family_id(), &target.engine_id)?;
    apply_eager(op, inputs)
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
    let recorded = record_eager_outputs(&op, &output_refs, inputs)?;
    if recorded.traces.len() != outputs.len() {
        return Err(Error::Internal(format!(
            "expected {} eager traces for {:?}, got {}",
            outputs.len(),
            op,
            recorded.traces.len()
        )));
    }
    let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
    for input in inputs {
        for scope in &input.metadata_scopes {
            push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
        }
    }

    recorded
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
                    metadata_scopes.clone(),
                )
            } else {
                EagerTensor::new_unregistered_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                    metadata_scopes.clone(),
                )
            }
        })
        .collect()
}

/// Apply one standard tensor op eagerly and record it for AD when needed.
///
/// Extension crates use this when an extension-level eager operation expands
/// into ordinary `StdTensorOp` nodes instead of a custom extension primitive.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::TensorRuntime`] containing
/// [`tenferro_tensor::ValidationError::InvalidArgument`] if an extension
/// op is passed to this standard-op entry point. Returns
/// [`tenferro_runtime::Error::ContextMismatch`] for tensors from different
/// eager contexts and propagates typed tensor/backend/runtime-state failures
/// from the selected eager context.
pub fn apply_standard_op(op: StdTensorOp, inputs: &[&EagerTensor]) -> Result<EagerTensor> {
    if matches!(op, StdTensorOp::Extension(_)) {
        return Err(Error::invalid_argument(
            "extension::apply_standard_op",
            ErrorPhase::Execution,
            "op",
            "Extension ops must be passed to apply_eager",
        ));
    }
    EagerTensor::nary_op(inputs, op)
}
