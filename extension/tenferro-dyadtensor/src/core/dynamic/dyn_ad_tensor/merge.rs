use core::ops::Add;

use chainrules_scalarops::{self, ScalarAd};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::super::tensor_ops::{tensor_map_binary_typed, tensor_map_unary_typed};
use crate::core::{AdTensorSnapshot, DynTensor, DynTensorTyped};
use crate::structured::StructuredTensor;
use crate::{tape, AdTensor, Error, NodeId, Result};

pub(super) fn map_ad_tensor_same_type_linear_typed<T, F>(
    input: &AdTensor<T>,
    map: F,
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    F: Fn(T) -> T + Copy + 'static,
{
    let mapped = match input.snapshot() {
        AdTensorSnapshot::Primal(primal) => AdTensorSnapshot::Primal(
            primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?,
        ),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensorSnapshot::Forward {
            primal: primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?,
            tangent: tangent.with_payload_like(tensor_map_unary_typed(tangent.payload(), map)?)?,
        },
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let input_layout = primal.clone();
            let output_primal =
                primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?;
            let output_tangent = tangent
                .as_ref()
                .map(|t| t.with_payload_like(tensor_map_unary_typed(t.payload(), map)?))
                .transpose()?;
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "same-type linear output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    let grad = input_layout
                        .with_payload_like(tensor_map_unary_typed(cotangent.payload(), map)?)?;
                    Ok(vec![(input_node, grad)])
                }),
            );
            return Ok(out);
        }
    };
    AdTensor::try_from(mapped)
}

pub(super) fn map_ad_tensor_mixed_linear_typed<TIn, TOut, P, R>(
    input: &AdTensor<TIn>,
    primal_map: P,
    reverse_map: R,
) -> Result<AdTensor<TOut>>
where
    TIn: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    TOut: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    P: Fn(TIn) -> TOut + Copy,
    R: Fn(TOut) -> TIn + Copy + 'static,
{
    let mapped = match input.snapshot() {
        AdTensorSnapshot::Primal(primal) => AdTensorSnapshot::Primal(StructuredTensor::new(
            primal.logical_dims().to_vec(),
            primal.axis_classes().to_vec(),
            tensor_map_unary_typed(primal.payload(), primal_map)?,
        )?),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensorSnapshot::Forward {
            primal: StructuredTensor::new(
                primal.logical_dims().to_vec(),
                primal.axis_classes().to_vec(),
                tensor_map_unary_typed(primal.payload(), primal_map)?,
            )?,
            tangent: StructuredTensor::new(
                tangent.logical_dims().to_vec(),
                tangent.axis_classes().to_vec(),
                tensor_map_unary_typed(tangent.payload(), primal_map)?,
            )?,
        },
        AdTensorSnapshot::Reverse { .. } => {
            let AdTensorSnapshot::Reverse {
                primal,
                node: input_node,
                tape,
                tangent,
            } = input.snapshot()
            else {
                unreachable!("snapshot matched reverse above");
            };
            let input_layout = primal.clone();
            let output_primal = StructuredTensor::new(
                primal.logical_dims().to_vec(),
                primal.axis_classes().to_vec(),
                tensor_map_unary_typed(primal.payload(), primal_map)?,
            )?;
            let output_tangent = tangent
                .as_ref()
                .map(|t| {
                    StructuredTensor::new(
                        t.logical_dims().to_vec(),
                        t.axis_classes().to_vec(),
                        tensor_map_unary_typed(t.payload(), primal_map)?,
                    )
                })
                .transpose()?;
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "mixed-type linear output is missing a tape node".to_string(),
                })?;
            tape::register_mixed_rule::<TOut, TIn>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    let grad = input_layout.with_payload_like(tensor_map_unary_typed(
                        cotangent.payload(),
                        reverse_map,
                    )?)?;
                    Ok(vec![(input_node, grad)])
                }),
            );
            return Ok(out);
        }
    };
    AdTensor::try_from(mapped)
}

fn tensor_add_typed<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + Copy + Add<Output = T>,
{
    tensor_map_binary_typed(lhs, rhs, |x, y| x + y)
}

struct AdTensorBinaryState<T: Scalar> {
    primal: StructuredTensor<T>,
    tangent: Option<StructuredTensor<T>>,
    reverse: Option<(NodeId, ::chainrules::Tape<DynTensor>)>,
}

fn split_ad_tensor_state<T: Scalar>(value: AdTensorSnapshot<T>) -> AdTensorBinaryState<T> {
    match value {
        AdTensorSnapshot::Primal(primal) => AdTensorBinaryState {
            primal,
            tangent: None,
            reverse: None,
        },
        AdTensorSnapshot::Forward { primal, tangent } => AdTensorBinaryState {
            primal,
            tangent: Some(tangent),
            reverse: None,
        },
        AdTensorSnapshot::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => AdTensorBinaryState {
            primal,
            tangent,
            reverse: Some((node, tape)),
        },
    }
}

fn ensure_same_structured_layout<T: Scalar>(
    op_name: &'static str,
    lhs: &StructuredTensor<T>,
    rhs: &StructuredTensor<T>,
) -> Result<()> {
    if lhs.logical_dims() != rhs.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching logical_dims, got lhs={:?}, rhs={:?}",
                lhs.logical_dims(),
                rhs.logical_dims()
            ),
        });
    }
    if lhs.axis_classes() != rhs.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching axis_classes, got lhs={:?}, rhs={:?}",
                lhs.axis_classes(),
                rhs.axis_classes()
            ),
        });
    }
    Ok(())
}

pub(super) fn merge_add_ad_tensors<T>(
    lhs: AdTensorSnapshot<T>,
    rhs: AdTensorSnapshot<T>,
) -> Result<AdTensorSnapshot<T>>
where
    T: Scalar + Copy + Add<Output = T> + DynTensorTyped + 'static,
{
    let lhs_state = split_ad_tensor_state(lhs);
    let rhs_state = split_ad_tensor_state(rhs);
    ensure_same_structured_layout("tensor add merge", &lhs_state.primal, &rhs_state.primal)?;

    let primal = StructuredTensor::new(
        lhs_state.primal.logical_dims().to_vec(),
        lhs_state.primal.axis_classes().to_vec(),
        tensor_add_typed(lhs_state.primal.payload(), rhs_state.primal.payload())?,
    )?;
    let tangent = match (lhs_state.tangent, rhs_state.tangent) {
        (Some(a), Some(b)) => {
            ensure_same_structured_layout("tensor add merge tangent/lhs", &a, &lhs_state.primal)?;
            ensure_same_structured_layout("tensor add merge tangent/rhs", &b, &rhs_state.primal)?;
            ensure_same_structured_layout("tensor add merge tangent", &a, &b)?;
            Some(StructuredTensor::new(
                a.logical_dims().to_vec(),
                a.axis_classes().to_vec(),
                tensor_add_typed(a.payload(), b.payload())?,
            )?)
        }
        (Some(a), None) => {
            ensure_same_structured_layout("tensor add merge tangent/lhs", &a, &lhs_state.primal)?;
            Some(a)
        }
        (None, Some(b)) => {
            ensure_same_structured_layout("tensor add merge tangent/rhs", &b, &rhs_state.primal)?;
            Some(b)
        }
        (None, None) => None,
    };

    match (lhs_state.reverse, rhs_state.reverse) {
        (None, None) => match tangent {
            Some(tangent) => Ok(AdTensorSnapshot::Forward { primal, tangent }),
            None => Ok(AdTensorSnapshot::Primal(primal)),
        },
        (Some((lhs_node, lhs_tape)), rhs_reverse) => {
            if let Some((_, ref rhs_tape)) = rhs_reverse {
                if !lhs_tape.same_tape(&rhs_tape) {
                    return Err(Error::InvalidAdTensor {
                        message: format!(
                            "reverse-mode tape mismatch in tensor add (lhs={}, rhs={})",
                            lhs_tape.id(),
                            rhs_tape.id()
                        ),
                    });
                }
            }
            let rhs_node = rhs_reverse.map(|(node, _)| node);
            let out = AdTensor::new_reverse_output(primal, &lhs_tape, tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "tensor merge output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &lhs_tape,
                output_node,
                Box::new(move |cotangent| {
                    let mut input_grads = Vec::new();
                    input_grads.push((lhs_node, cotangent.clone()));
                    if let Some(node) = rhs_node {
                        input_grads.push((node, cotangent.clone()));
                    }
                    Ok(input_grads)
                }),
            );
            Ok(out.snapshot())
        }
        (None, Some((rhs_node, rhs_tape))) => {
            let out = AdTensor::new_reverse_output(primal, &rhs_tape, tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "tensor merge output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &rhs_tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(rhs_node, cotangent.clone())])),
            );
            Ok(out.snapshot())
        }
    }
}
