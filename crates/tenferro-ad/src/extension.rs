//! Eager AD support for out-of-tree extension primitives.

use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ad_support::push_metadata_scope;
use tenferro_runtime::{Error, ErrorPhase, ExtensionModule, Result, Runtime, RuntimeConfigError};
use tenferro_tensor::{BackendSession, Tensor, TensorRead, TensorValue};

use crate::eager::{eager_grad_recording_enabled, record_eager_outputs, EagerRuntime, EagerTensor};

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
    let outputs = ctx.exec_outputs_read(&std_op, &input_reads)?;
    finish_eager_extension_outputs(ctx, std_op, inputs, outputs)
}

/// Apply an extension op to eager tensors through a direct prepared-operation
/// callback receiving a non-owning backend session.
///
/// The eager runtime owner resolves the exact engine and closed backend kind
/// before invoking `module_factory`. Extension crates use that target to build
/// their module; this bridge does not infer placement from tensor values or
/// construct an engine identifier on their behalf. The module is installed and
/// the callback is entered under the same borrowed backend session.
///
/// # Examples
///
/// This low-level bridge validates the installed module before entering the
/// backend session. A module that does not register the operation family for
/// the selected engine is rejected with a typed runtime source.
///
/// ```rust
/// use std::any::Any;
/// use std::hash::Hasher;
/// use std::sync::Arc;
///
/// use tenferro_ad::extension::{
///     apply_eager_with_extension_session, EagerExtensionTarget, ExtensionOp,
/// };
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ops::{ExtensionShapeContext, SymDim};
/// use tenferro_runtime::{
///     ExtensionModule, ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar,
/// };
/// use tenferro_tensor::{DType, Tensor};
///
/// #[derive(Debug)]
/// struct ExampleOp;
///
/// impl ExtensionOp for ExampleOp {
///     fn family_id(&self) -> &'static str {
///         "example.eager-bridge.v1"
///     }
///     fn payload_hash(&self, _hasher: &mut dyn Hasher) {}
///     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
///         other.as_any().downcast_ref::<Self>().is_some()
///     }
///     fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
///         Arc::new(Self)
///     }
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///     fn input_count(&self) -> usize {
///         1
///     }
///     fn output_count(&self) -> usize {
///         1
///     }
///     fn infer_output_meta(
///         &self,
///         ctx: &mut ExtensionShapeContext<'_>,
///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
///         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
///     }
/// }
///
/// #[derive(Debug)]
/// struct ExampleModule {
///     id: ExtensionModuleId,
/// }
///
/// impl ExtensionModule for ExampleModule {
///     fn module_id(&self) -> &ExtensionModuleId {
///         &self.id
///     }
///     fn configure(
///         &self,
///         _registrar: &mut ExtensionModuleRegistrar<'_>,
///     ) -> Result<(), ExtensionModuleError> {
///         Ok(())
///     }
/// }
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let input = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?,
///     runtime,
/// )?;
/// let result = apply_eager_with_extension_session(
///     Arc::new(ExampleOp),
///     &[&input],
///     |_target: EagerExtensionTarget| {
///         Ok(Arc::new(ExampleModule {
///             id: ExtensionModuleId::new("example.eager-bridge.module").unwrap(),
///         }) as Arc<dyn ExtensionModule>)
///     },
///     |_op, _inputs, _session| unreachable!("module validation runs first"),
/// );
/// assert!(result.is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` with `InvalidArgument` when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// `Error::ContextMismatch` when tensors belong to different eager runtimes.
/// The module factory's [`tenferro_runtime::Error`] is propagated unchanged.
/// A selected-engine or module-family mismatch is returned as
/// `Error::RuntimeStateSource` retaining [`RuntimeConfigError`] as its source;
/// backend and extension failures retain their typed sources.
#[doc(hidden)]
pub fn apply_eager_with_extension_session(
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
    execute: impl FnOnce(
            &dyn ExtensionOp,
            &[TensorRead<'_>],
            &mut ExtensionExecutionContext<'_, dyn BackendSession + '_>,
        ) -> tenferro_tensor::Result<Vec<Tensor>>
        + Send,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let target = ctx.eager_extension_target()?;
    let module = module_factory(target.clone())?;
    ctx.install_extension_module(module)?;
    validate_eager_extension_module(ctx.runtime(), op.family_id(), &target)?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    let outputs = ctx.with_extension_execution_context(|extension_ctx| {
        execute(op.as_ref(), &input_reads, extension_ctx)
    })??;
    finish_eager_extension_outputs(ctx, StdTensorOp::Extension(op), inputs, outputs)
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

pub(crate) fn validate_eager_extension_module(
    runtime: &Runtime,
    family_id: &'static str,
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
    if !snapshot.has_extension_engine(family_id, &target.engine_id) {
        return Err(Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingExtensionEngine {
                family_id,
                engine_id: target.engine_id.clone(),
            },
        ));
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

    if !eager_grad_recording_enabled() {
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
