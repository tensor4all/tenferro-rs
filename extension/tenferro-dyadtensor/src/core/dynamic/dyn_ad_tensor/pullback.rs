use chainrules::Tape;

use super::DynAdTensor;
use crate::core::DynTensor;
use crate::ops::ad::normalize_cotangent_payload;
use crate::{AdTensor, Error, NodeId, Result};

fn dyn_primal_from_snapshot(snapshot: DynTensor) -> DynAdTensor {
    match snapshot {
        DynTensor::F32(value) => DynAdTensor::from(AdTensor::new_primal(value)),
        DynTensor::F64(value) => DynAdTensor::from(AdTensor::new_primal(value)),
        DynTensor::C32(value) => DynAdTensor::from(AdTensor::new_primal(value)),
        DynTensor::C64(value) => DynAdTensor::from(AdTensor::new_primal(value)),
    }
}

fn normalize_dyn_cotangent(
    output: &DynAdTensor,
    cotangent: &DynAdTensor,
    op_name: &'static str,
) -> Result<DynTensor> {
    match (output, cotangent) {
        (DynAdTensor::F32(output), DynAdTensor::F32(cotangent)) => Ok(DynTensor::F32(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (DynAdTensor::F64(output), DynAdTensor::F64(cotangent)) => Ok(DynTensor::F64(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (DynAdTensor::C32(output), DynAdTensor::C32(cotangent)) => Ok(DynTensor::C32(
            output
                .structured_primal()
                .with_payload_like(normalize_cotangent_payload(output, cotangent, op_name)?)?,
        )),
        (DynAdTensor::C64(output), DynAdTensor::C64(cotangent)) => Ok(DynTensor::C64(
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

fn reverse_handle(value: &DynAdTensor) -> Option<(NodeId, Tape<DynTensor>)> {
    match value {
        DynAdTensor::F32(value) => value
            .reverse_node_id()
            .zip(value.reverse_tape().cloned())
            .map(|(node, tape)| (node, tape)),
        DynAdTensor::F64(value) => value
            .reverse_node_id()
            .zip(value.reverse_tape().cloned())
            .map(|(node, tape)| (node, tape)),
        DynAdTensor::C32(value) => value
            .reverse_node_id()
            .zip(value.reverse_tape().cloned())
            .map(|(node, tape)| (node, tape)),
        DynAdTensor::C64(value) => value
            .reverse_node_id()
            .zip(value.reverse_tape().cloned())
            .map(|(node, tape)| (node, tape)),
    }
}

fn dyn_pullback_grads(
    output: &DynAdTensor,
    cotangent: &DynAdTensor,
) -> Result<(Tape<DynTensor>, chainrules::Gradients<DynTensor>)> {
    let seed = normalize_dyn_cotangent(output, cotangent, "DynAdTensor::pullback_wrt")?;
    match output {
        DynAdTensor::F32(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
                message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                    .to_string(),
            })?;
            let tape = tracked
                .tape()
                .cloned()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                        .to_string(),
                })?;
            let grads = tape
                .pullback_with_seed(tracked, seed)
                .map_err(Error::from)?;
            Ok((tape, grads))
        }
        DynAdTensor::F64(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
                message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                    .to_string(),
            })?;
            let tape = tracked
                .tape()
                .cloned()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                        .to_string(),
                })?;
            let grads = tape
                .pullback_with_seed(tracked, seed)
                .map_err(Error::from)?;
            Ok((tape, grads))
        }
        DynAdTensor::C32(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
                message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                    .to_string(),
            })?;
            let tape = tracked
                .tape()
                .cloned()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                        .to_string(),
                })?;
            let grads = tape
                .pullback_with_seed(tracked, seed)
                .map_err(Error::from)?;
            Ok((tape, grads))
        }
        DynAdTensor::C64(value) => {
            let tracked = value.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
                message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                    .to_string(),
            })?;
            let tape = tracked
                .tape()
                .cloned()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "DynAdTensor::pullback_wrt requires reverse-mode output tensor"
                        .to_string(),
                })?;
            let grads = tape
                .pullback_with_seed(tracked, seed)
                .map_err(Error::from)?;
            Ok((tape, grads))
        }
    }
}

impl DynAdTensor {
    /// Computes reverse-mode pullback projected to selected tensors using only
    /// the dynamic public tensor API.
    ///
    /// Returns `None` for disconnected or non-reverse tensors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynAdTensor, DynTape};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let tape = DynTape::new();
    /// let x = DynAdTensor::new_reverse_leaf(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    ///     &tape,
    /// )
    /// .unwrap();
    /// let out = x.scale(&DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// ));
    /// let cotangent = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let grads = out.unwrap().pullback_wrt(&cotangent, &[&x]).unwrap();
    /// assert_eq!(
    ///     grads[0]
    ///         .as_ref()
    ///         .unwrap()
    ///         .as_f64()
    ///         .unwrap()
    ///         .primal()
    ///         .buffer()
    ///         .as_slice()
    ///         .unwrap(),
    ///     &[3.0]
    /// );
    /// ```
    pub fn pullback_wrt(&self, cotangent: &Self, wrt: &[&Self]) -> Result<Vec<Option<Self>>> {
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
