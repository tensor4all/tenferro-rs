//! Eager forward bodies of the registry (`EagerEinsumOps` on `EagerSession`).
//!
//! Default options run the former session entries; an explicit contraction
//! order is passed to the same route as its plan spec; a requested
//! conjugation is applied with `EagerSession::conj` on the selected operands
//! first, so it is differentiated like any other recorded operation.

use tenferro_ad::{EagerSession, EagerTensor, Result};

use super::options::{EinsumOptions, EinsumSpec};
use super::traced::runtime_err;
use crate::{Error, TensorDotAxes};

pub(crate) fn einsum(
    session: &mut EagerSession<'_>,
    inputs: &[&EagerTensor],
    spec: &(impl EinsumSpec + ?Sized),
    options: &EinsumOptions,
) -> Result<EagerTensor> {
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
    let mut conjugated = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        conjugated.push(if options.conj.get(i).is_yes() {
            Some(session.conj(input)?)
        } else {
            None
        });
    }
    let operands: Vec<&EagerTensor> = inputs
        .iter()
        .zip(&conjugated)
        .map(|(input, c)| c.as_ref().unwrap_or(*input))
        .collect();
    let optimize = match &options.optimize {
        crate::EinsumOptimize::Auto(o) if !o.anneals() => None,
        optimize => Some(optimize),
    };
    crate::eager_ad::einsum_spec_planned(session, &operands, spec.spec(), optimize)
        .map_err(|e| runtime_err(e, "einsum"))
}

/// One-shot eager sampled einsum (#2675): prepare a
/// [`crate::SampledEinsumPlan`] from the inputs' metadata and run it on the
/// session.
pub(crate) fn einsum_at(
    session: &mut EagerSession<'_>,
    inputs: &[&EagerTensor],
    spec: &(impl EinsumSpec + ?Sized),
    samples: &crate::Samples,
    options: &crate::SampleOptions,
) -> Result<EagerTensor> {
    let op = "einsum_at";
    let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
    let subscripts = match spec.spec() {
        crate::api::Spec::Text(text) => crate::subscripts::parse_einsum_notation(text)
            .and_then(|notation| crate::concrete::resolve_shapes_notation(&notation, &shapes)),
        crate::api::Spec::Notation(notation) => {
            crate::concrete::resolve_shapes_notation(notation, &shapes)
        }
        crate::api::Spec::Subscripts(subscripts) => Ok(crate::Subscripts::from(subscripts)),
    }
    .map_err(|e| runtime_err(e, op))?;
    let metadata = inputs
        .iter()
        .map(|t| (t.dtype(), t.shape().to_vec()))
        .collect();
    let plan = crate::SampledEinsumPlan::plan(metadata, &subscripts, samples, options)
        .map_err(|e| runtime_err(e, op))?;
    plan.execute_eager(session, inputs)
}

pub(crate) fn tensordot(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    b: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> Result<EagerTensor> {
    crate::eager_ad::tensordot(session, a, b, axes).map_err(|e| runtime_err(e, "tensordot"))
}
