//! Eager forward bodies of the registry (`EagerEinsumOps` on `EagerSession`).
//!
//! Default options run the former session entries; an explicit contraction
//! order is passed to the same route as its plan spec; a requested
//! conjugation is applied with `EagerSession::conj` on the selected operands
//! first, so it is differentiated like any other recorded operation.

use tenferro_ad::{EagerSession, EagerTensor, Result};

use super::options::{EinsumOptions, EinsumSpec, TensordotOptions};
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

pub(crate) fn tensordot(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    b: &EagerTensor,
    axes: TensorDotAxes<'_>,
    _options: &TensordotOptions,
) -> Result<EagerTensor> {
    crate::eager_ad::tensordot(session, a, b, axes).map_err(|e| runtime_err(e, "tensordot"))
}
