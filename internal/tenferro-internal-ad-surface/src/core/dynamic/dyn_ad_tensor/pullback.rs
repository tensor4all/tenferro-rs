use tenferro_internal_ad_core::{AdTensor, DynAdTensorRef};
use tidu::expert::Tape;

use super::{accessors::TypedTensorBorrowTyped, Tensor};
use crate::core::{DynTensor, NodeId};
use crate::ops::ad::normalize_cotangent_payload;
use crate::{Error, Result};

macro_rules! match_same_dtype_dyn_ad_tensor_ref_pair {
    ($lhs:expr, $rhs:expr, |$lhs_var:ident, $rhs_var:ident| $body:expr) => {{
        match ($lhs.as_dyn_ad_ref(), $rhs.as_dyn_ad_ref()) {
            (DynAdTensorRef::F32($lhs_var), DynAdTensorRef::F32($rhs_var)) => Some($body),
            (DynAdTensorRef::F64($lhs_var), DynAdTensorRef::F64($rhs_var)) => Some($body),
            (DynAdTensorRef::C32($lhs_var), DynAdTensorRef::C32($rhs_var)) => Some($body),
            (DynAdTensorRef::C64($lhs_var), DynAdTensorRef::C64($rhs_var)) => Some($body),
            _ => None,
        }
    }};
}

fn dyn_primal_from_snapshot(snapshot: DynTensor) -> Tensor {
    Tensor::from(snapshot)
}

fn normalize_dyn_cotangent_typed<T>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    op_name: &'static str,
) -> Result<DynTensor>
where
    T: crate::DynTensorTyped + tenferro_algebra::Scalar + Clone + 'static,
    DynTensor: From<crate::StructuredTensor<T>>,
{
    Ok(DynTensor::from(
        output
            .structured_primal()
            .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
    ))
}

fn normalize_dyn_cotangent(
    output: &Tensor,
    cotangent: &Tensor,
    op_name: &'static str,
) -> Result<DynTensor> {
    if let Some(result) =
        match_same_dtype_dyn_ad_tensor_ref_pair!(output, cotangent, |output, cotangent| {
            normalize_dyn_cotangent_typed(output, cotangent, op_name)
        })
    {
        return result;
    }

    Err(Error::InvalidAdTensor {
        message: format!(
            "{op_name} requires cotangent dtype {:?} to match output dtype {:?}",
            cotangent.scalar_type(),
            output.scalar_type()
        ),
    })
}

fn reverse_handle(value: &Tensor) -> Option<(NodeId, Tape<DynTensor>)> {
    value.reverse_handle()
}

fn typed_edge_pullback_wrt<T>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    wrt: &[&Tensor],
    wrap: fn(crate::StructuredTensor<T>) -> Tensor,
) -> Result<Option<Vec<Option<Tensor>>>>
where
    T: crate::DynTensorTyped + TypedTensorBorrowTyped + tenferro_algebra::Scalar + Clone + 'static,
{
    let Some(output_value) = output.reverse_edge_value() else {
        return Ok(None);
    };
    let output_tape = output.reverse_tape();
    let seed = output
        .structured_primal()
        .with_payload_like(normalize_cotangent_payload(
            output,
            cotangent,
            "Tensor::pullback_wrt",
        )?)?;

    let mut query_values = Vec::new();
    let mut query_indices = Vec::new();
    for (index, wrt_tensor) in wrt.iter().enumerate() {
        let maybe_value = T::reverse_edge_value_from_dyn_ad(&wrt_tensor.0);
        if wrt_tensor.requires_grad() && maybe_value.is_none() {
            if let Some(wrt_tape) = wrt_tensor.reverse_tape() {
                if let Some(output_tape) = output_tape.as_ref() {
                    if !wrt_tape.same_tape(output_tape as &Tape<DynTensor>) {
                        return Err(Error::MixedReverseTape {
                            expected: output_tape.id() as u64,
                            found: wrt_tape.id() as u64,
                        });
                    }
                } else {
                    return Err(Error::InvalidAdTensor {
                        message: "Tensor::pullback_wrt cannot mix edge-based outputs with legacy tape-only wrt tensors".to_string(),
                    });
                }
            }
            return Ok(None);
        }
        if let Some(value) = maybe_value {
            query_indices.push(index);
            query_values.push(value);
        }
    }

    let query_refs: Vec<_> = query_values.iter().map(|value| value.as_ref()).collect();
    let grads = output_value
        .grad_wrt_with_seed(seed, &query_refs)
        .map_err(Error::from)?;
    let mut out = vec![None; wrt.len()];
    for (index, grad) in query_indices.into_iter().zip(grads.into_iter()) {
        out[index] = grad.map(wrap);
    }
    Ok(Some(out))
}

fn dyn_pullback_grads(
    output: &Tensor,
    cotangent: &Tensor,
) -> Result<(Tape<DynTensor>, tidu::expert::Gradients<DynTensor>)> {
    let seed = normalize_dyn_cotangent(output, cotangent, "Tensor::pullback_wrt")?;
    let tracked = output.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
        message: "Tensor::pullback_wrt requires reverse-mode output tensor".to_string(),
    })?;
    let tape = tracked
        .tape()
        .map(Clone::clone)
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "Tensor::pullback_wrt requires reverse-mode output tensor".to_string(),
        })?;
    let grads = tape
        .pullback_with_seed(&tracked, seed)
        .map_err(Error::from)?;
    Ok((tape, grads))
}

impl Tensor {
    pub(crate) fn pullback_wrt(
        &self,
        cotangent: &Self,
        wrt: &[&Self],
    ) -> Result<Vec<Option<Self>>> {
        if let Some(grads) =
            match_same_dtype_dyn_ad_tensor_ref_pair!(self, cotangent, |output, cotangent| {
                typed_edge_pullback_wrt(output, cotangent, wrt, Tensor::from_structured)
            })
        {
            if let Some(grads) = grads? {
                return Ok(grads);
            }
        }

        let (tape, grads) = dyn_pullback_grads(self, cotangent)?;
        let mut out = Vec::with_capacity(wrt.len());

        for wrt_tensor in wrt {
            match reverse_handle(wrt_tensor) {
                Some((node, wrt_tape)) => {
                    if !wrt_tape.same_tape(&tape as &Tape<DynTensor>) {
                        return Err(Error::MixedReverseTape {
                            expected: tape.id() as u64,
                            found: wrt_tape.id() as u64,
                        });
                    }
                    out.push(grads.get(node).cloned().map(dyn_primal_from_snapshot));
                }
                None => out.push(None),
            }
        }

        Ok(out)
    }
}
