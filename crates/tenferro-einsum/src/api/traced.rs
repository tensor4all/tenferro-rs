//! Traced forward bodies of the registry (`TracedEinsumOps` on `TraceContext`,
//! `TracedTensordotOps` on `TracedTensor`).
//!
//! Each entry records exactly what the former `TraceContext` / `TracedTensor`
//! entries recorded; a requested conjugation is recorded as a `conj` node on
//! the selected operands first.

use tenferro_runtime::{Error as RuntimeError, ErrorPhase, Result, TraceContext, TracedTensor};

use super::options::{EinsumOptions, EinsumSpec};
use crate::{Error, TensorDotAxes, EINSUM_EXTENSION_FAMILY_ID};

/// Converts an einsum-domain error into the runtime error every traced and
/// eager entry returns, keeping the einsum error reachable through `source()`.
pub(crate) fn runtime_err(error: Error, op: &'static str) -> RuntimeError {
    match error {
        Error::Runtime(error) => error,
        Error::Tensor(error) => RuntimeError::TensorRuntime(error),
        error => {
            let kind = error.kind();
            RuntimeError::extension(
                op,
                ErrorPhase::GraphBuild,
                EINSUM_EXTENSION_FAMILY_ID,
                kind,
                error,
            )
        }
    }
}

pub(crate) fn einsum(
    trace: &TraceContext,
    inputs: &[&TracedTensor],
    spec: &(impl EinsumSpec + ?Sized),
    options: &EinsumOptions,
) -> Result<TracedTensor> {
    if options.conj.len() > inputs.len() {
        return Err(runtime_err(
            Error::invalid_argument(
                "einsum",
                "conj",
                format!(
                    "the conjugation mask has {} entries for {} inputs",
                    options.conj.len(),
                    inputs.len()
                ),
            ),
            "einsum",
        ));
    }
    let mut values = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        if !input.context().same_context(trace) {
            return Err(RuntimeError::invalid_argument(
                "einsum",
                ErrorPhase::GraphBuild,
                "inputs",
                format!("input {i} belongs to a different trace context than the receiver"),
            ));
        }
        let value = if options.conj.get(i).is_yes() {
            input.conj()?.trace_value()
        } else {
            input.trace_value()
        };
        values.push(value);
    }
    let value =
        crate::traced::trace_einsum_spec(trace, &values, spec.spec(), options.optimize.clone())
            .map_err(|e| runtime_err(e, "einsum"))?;
    trace.derive_tensor(value)
}

pub(crate) fn tensordot(
    a: &TracedTensor,
    b: &TracedTensor,
    axes: TensorDotAxes<'_>,
) -> Result<TracedTensor> {
    crate::traced::tensordot(a, b, axes).map_err(|e| runtime_err(e, "tensordot"))
}
