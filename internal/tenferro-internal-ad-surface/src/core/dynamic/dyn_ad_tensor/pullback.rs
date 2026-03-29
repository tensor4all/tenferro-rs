use tidu::Tape;

use super::Tensor;
use crate::core::{DynTensor, NodeId};
use crate::ops::ad::normalize_cotangent_payload;
use crate::{AdTensor, Error, Result};

fn dyn_primal_from_snapshot(snapshot: DynTensor) -> Tensor {
    match snapshot {
        DynTensor::F32(value) => Tensor::from(AdTensor::new_primal(value)),
        DynTensor::F64(value) => Tensor::from(AdTensor::new_primal(value)),
        DynTensor::C32(value) => Tensor::from(AdTensor::new_primal(value)),
        DynTensor::C64(value) => Tensor::from(AdTensor::new_primal(value)),
    }
}

fn normalize_dyn_cotangent(
    output: &Tensor,
    cotangent: &Tensor,
    op_name: &'static str,
) -> Result<DynTensor> {
    match (output, cotangent) {
        (Tensor::F32(output), Tensor::F32(cotangent)) => Ok(DynTensor::F32(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (Tensor::F64(output), Tensor::F64(cotangent)) => Ok(DynTensor::F64(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (Tensor::C32(output), Tensor::C32(cotangent)) => Ok(DynTensor::C32(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (Tensor::C64(output), Tensor::C64(cotangent)) => Ok(DynTensor::C64(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        _ => Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires cotangent dtype {:?} to match output dtype {:?}",
                cotangent.scalar_type(),
                output.scalar_type()
            ),
        }),
    }
}

fn reverse_handle(value: &Tensor) -> Option<(NodeId, Tape<DynTensor>)> {
    match value {
        Tensor::F32(value) => value.reverse_handle().map(|(node, tape)| (node, tape)),
        Tensor::F64(value) => value.reverse_handle().map(|(node, tape)| (node, tape)),
        Tensor::C32(value) => value.reverse_handle().map(|(node, tape)| (node, tape)),
        Tensor::C64(value) => value.reverse_handle().map(|(node, tape)| (node, tape)),
    }
}

fn dyn_pullback_grads(
    output: &Tensor,
    cotangent: &Tensor,
) -> Result<(Tape<DynTensor>, tidu::Gradients<DynTensor>)> {
    let seed = normalize_dyn_cotangent(output, cotangent, "Tensor::pullback_wrt")?;
    match output {
        Tensor::F32(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
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
        Tensor::F64(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
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
        Tensor::C32(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
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
        Tensor::C64(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
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
    }
}

impl Tensor {
    pub(crate) fn pullback_wrt(
        &self,
        cotangent: &Self,
        wrt: &[&Self],
    ) -> Result<Vec<Option<Self>>> {
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
