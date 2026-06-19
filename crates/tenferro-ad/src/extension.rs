//! Eager AD support for out-of-tree extension primitives.

use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ad_support::push_metadata_scope;
use tenferro_runtime::{Error, Result};
use tenferro_tensor::{Tensor, TensorValue};

use crate::eager::{record_eager_outputs, EagerRuntime, EagerTensor};

pub use tenferro_ops::ext_op::{ExtensionAdRule, ExtensionRegistryError, ExtensionRuleSet};
pub use tenferro_runtime::extension::{
    apply, ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
    ExtensionExecutionContext, ExtensionExecutor, ExtensionFamilyId, ExtensionOp,
    ExtensionRegistry, ExtensionRuntime, ExtensionRuntimeRegistryError,
};

/// Adopt an untracked eager tensor value produced by this runtime's backend.
///
/// This is a low-level extension contract for eager composite operations that
/// execute through [`EagerRuntime::with_backend_mut`] and receive a lazy
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
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let value = TensorValue::from_tensor(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
/// );
/// let eager = adopt_untracked_eager_value(ctx, value);
/// assert_eq!(eager.shape(), &[1]);
/// assert!(!eager.tracks_grad());
/// ```
#[must_use]
pub fn adopt_untracked_eager_value(ctx: Arc<EagerRuntime>, value: TensorValue) -> EagerTensor {
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
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let x = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
///     ctx,
/// ).unwrap();
/// let _ = &x;
/// let _apply = apply_eager;
/// ```
pub fn apply_eager(op: Arc<dyn ExtensionOp>, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    let Some(first) = inputs.first() else {
        return Err(Error::Internal(
            "extension::apply_eager requires at least one input tensor".to_string(),
        ));
    };
    if inputs.len() != op.input_count() {
        return Err(Error::Internal(format!(
            "extension::apply_eager: op family {:?} expects {} inputs, got {}",
            op.family_id(),
            op.input_count(),
            inputs.len()
        )));
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

    let op = StdTensorOp::Extension(op);
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    let outputs = ctx.exec_outputs_read(&op, &input_reads)?;
    if outputs.len() != op.output_count() {
        return Err(Error::Internal(format!(
            "expected {} eager outputs for {:?}, got {}",
            op.output_count(),
            op,
            outputs.len()
        )));
    }

    if !inputs.iter().any(|input| input.requires_grad) {
        return outputs
            .into_iter()
            .map(|output| EagerTensor::new_untracked_result(Arc::clone(&ctx), output))
            .collect();
    }

    let outputs: Vec<Arc<Tensor>> = outputs.into_iter().map(Arc::new).collect();
    let recorded = record_eager_outputs(&op, &outputs, inputs)?;
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
        .zip(outputs)
        .map(|(trace, output)| {
            EagerTensor::new_result(
                Arc::clone(&ctx),
                trace.key,
                output.as_ref().clone(),
                trace.requires_grad,
                trace.trace,
                metadata_scopes.clone(),
            )
        })
        .collect()
}
