use chainrules::{self, ScalarAd};
use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::AdTensor;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::expert::Tape;

use super::super::super::tensor_ops::{
    tensor_element, tensor_map_binary_typed, tensor_map_unary_typed, unflatten_index_column_major,
};
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::{DynTensor, Error, Result};

fn tensor_scalar_rrule_typed<T>(
    tensor_primal: &DenseTensor<T>,
    scalar_primal: T,
    cotangent: &DenseTensor<T>,
    rrule: fn(T, T, T) -> (T, T),
) -> Result<(DenseTensor<T>, T)>
where
    T: Scalar + ScalarAd + Copy,
{
    if tensor_primal.dims() != cotangent.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in mixed reverse pullback: primal={:?}, cotangent={:?}",
                tensor_primal.dims(),
                cotangent.dims()
            ),
        });
    }

    let dims = tensor_primal.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut tensor_grad = Vec::with_capacity(total);
    let mut scalar_grad = T::from_i32(0);

    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let x = tensor_element(tensor_primal, &idx)?;
        let dy = tensor_element(cotangent, &idx)?;
        let (dx, da) = rrule(x, scalar_primal, dy);
        tensor_grad.push(dx);
        scalar_grad = scalar_grad + da;
    }

    Ok((
        DenseTensor::from_slice(&tensor_grad, &dims, MemoryOrder::ColumnMajor)
            .map_err(Error::from)?,
        scalar_grad,
    ))
}

fn tensor_binary_scalar_ad_typed<T>(
    primal: &DenseTensor<T>,
    tensor_tangent: Option<&DenseTensor<T>>,
    scalar_primal: T,
    scalar_tangent: Option<T>,
    primal_rule: fn(T, T) -> T,
    frule: fn(T, T, T, T) -> (T, T),
) -> Result<(DenseTensor<T>, Option<DenseTensor<T>>)>
where
    T: Scalar + ScalarAd + Copy,
{
    let primal_out = tensor_map_unary_typed(primal, |x| primal_rule(x, scalar_primal))?;
    let tangent_out = match (tensor_tangent, scalar_tangent) {
        (None, None) => None,
        (Some(dt), maybe_ds) => Some(tensor_map_binary_typed(primal, dt, |x, dx| {
            let (_, tangent) = frule(
                x,
                scalar_primal,
                dx,
                maybe_ds.unwrap_or_else(|| T::from_i32(0)),
            );
            tangent
        })?),
        (None, Some(ds)) => Some(tensor_map_unary_typed(primal, |x| {
            let (_, tangent) = frule(x, scalar_primal, T::from_i32(0), ds);
            tangent
        })?),
    };
    Ok((primal_out, tangent_out))
}

fn rank0_tensor<T>(value: T) -> Result<DenseTensor<T>>
where
    T: Scalar + Copy,
{
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).map_err(Error::from)
}

fn extract_rank0_scalar<T>(scalar: &AdTensor<T>, op_name: &'static str) -> Result<T>
where
    T: Scalar + Copy,
{
    if !scalar.dims().is_empty() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires a rank-0 scalar tensor, got dims={:?}",
                scalar.dims()
            ),
        });
    }
    tensor_element(scalar.primal(), &[])
}

fn extract_rank0_scalar_tangent<T>(scalar: &AdTensor<T>, op_name: &'static str) -> Result<Option<T>>
where
    T: Scalar + Copy,
{
    scalar
        .tangent()
        .map(|tangent| {
            if !tangent.dims().is_empty() {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "{op_name} requires a rank-0 scalar tangent tensor, got dims={:?}",
                        tangent.dims()
                    ),
                });
            }
            tensor_element(tangent, &[])
        })
        .transpose()
}

fn ensure_tensor_scalar_reverse_attached<T>(
    tensor: &AdTensor<T>,
    scalar: &AdTensor<T>,
) -> Result<()>
where
    T: Scalar + DynTensorTyped + 'static,
{
    let tensor_tape = tensor.reverse_tape();
    let scalar_tape = scalar.reverse_tape();
    match (tensor_tape, scalar_tape) {
        (Some(tensor_tape), Some(scalar_tape)) => {
            if !tensor_tape.same_tape(&scalar_tape as &Tape<DynTensor>) {
                return Err(Error::MixedReverseTape {
                    expected: tensor_tape.id() as u64,
                    found: scalar_tape.id() as u64,
                });
            }
            Ok(())
        }
        (Some(tape), None) => scalar.ensure_reverse_leaf_on(&tape),
        (None, Some(tape)) => tensor.ensure_reverse_leaf_on(&tape),
        (None, None) => {
            let tape = Tape::new();
            tensor.ensure_reverse_leaf_on(&tape)?;
            scalar.ensure_reverse_leaf_on(&tape)?;
            Ok(())
        }
    }
}

fn merge_tensor_scalar_output<T>(
    tensor: &AdTensor<T>,
    scalar: &AdTensor<T>,
    primal: DenseTensor<T>,
    tangent: Option<DenseTensor<T>>,
    rrule: fn(T, T, T) -> (T, T),
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    if tensor.requires_grad() || scalar.requires_grad() {
        ensure_tensor_scalar_reverse_attached(tensor, scalar)?;
    }
    let tensor_reverse = tensor.reverse_handle();
    let scalar_reverse = scalar.reverse_handle();

    let reverse = match (tensor_reverse.clone(), scalar_reverse.clone()) {
        (Some((_lhs_node, lhs_tape)), Some((_, rhs_tape)))
            if !lhs_tape.same_tape(&rhs_tape as &Tape<DynTensor>) =>
        {
            return Err(Error::MixedReverseTape {
                expected: lhs_tape.id() as u64,
                found: rhs_tape.id() as u64,
            });
        }
        (Some((node, tape)), Some(_)) => Some((node, tape)),
        (Some((node, tape)), None) => Some((node, tape)),
        (None, Some((node, tape))) => Some((node, tape)),
        (None, None) => None,
    };

    let structured_primal = tensor.structured_primal().with_payload_like(primal)?;
    let structured_tangent = tangent
        .map(|payload| tensor.structured_primal().with_payload_like(payload))
        .transpose()?;

    if let Some((_, tape)) = reverse {
        let tensor_node = tensor_reverse.map(|(node, _)| node);
        let scalar_node = scalar_reverse.map(|(node, _)| node);
        let tensor_layout = tensor.structured_primal().clone();
        let scalar_layout = scalar.structured_primal().clone();
        let tensor_primal = tensor.primal().clone();
        let scalar_primal = extract_rank0_scalar(scalar, "tensor_scalar_reverse")?;
        let out = AdTensor::from_reverse_output(structured_primal, &tape, structured_tangent)?;
        let output_node = out
            .reverse_node_id()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "tensor-scalar reverse output is missing a tape node".to_string(),
            })?;

        let input_node_ids: Vec<_> = [tensor_node, scalar_node]
            .iter()
            .filter_map(|n| *n)
            .collect();
        crate::tape::register_closure_rule::<T>(
            &tape,
            output_node,
            input_node_ids,
            Box::new(move |cotangent| {
                let mut input_grads = Vec::new();
                if let Some(node) = tensor_node {
                    let (tensor_grad, _) = tensor_scalar_rrule_typed(
                        &tensor_primal,
                        scalar_primal,
                        cotangent.payload(),
                        rrule,
                    )?;
                    input_grads.push((node, tensor_layout.with_payload_like(tensor_grad)?));
                }
                if let Some(node) = scalar_node {
                    let (_, scalar_grad) = tensor_scalar_rrule_typed(
                        &tensor_primal,
                        scalar_primal,
                        cotangent.payload(),
                        rrule,
                    )?;
                    input_grads.push((
                        node,
                        scalar_layout.with_payload_like(rank0_tensor(scalar_grad)?)?,
                    ));
                }
                Ok(input_grads)
            }),
        );

        return Ok(out);
    }
    if let Some(tangent) = structured_tangent {
        return AdTensor::try_from(AdTensorSnapshot::Forward {
            primal: structured_primal,
            tangent,
        });
    }
    AdTensor::try_from(AdTensorSnapshot::Primal(structured_primal))
}

pub(super) fn scale_ad_tensor_typed<T>(
    tensor: &AdTensor<T>,
    scalar: &AdTensor<T>,
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        extract_rank0_scalar(scalar, "scale")?,
        extract_rank0_scalar_tangent(scalar, "scale")?,
        chainrules::mul,
        chainrules::mul_frule,
    )?;
    merge_tensor_scalar_output(tensor, scalar, primal, tangent, chainrules::mul_rrule)
}

pub(super) fn div_ad_tensor_typed<T>(
    tensor: &AdTensor<T>,
    scalar: &AdTensor<T>,
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        extract_rank0_scalar(scalar, "div_scalar")?,
        extract_rank0_scalar_tangent(scalar, "div_scalar")?,
        chainrules::div,
        chainrules::div_frule,
    )?;
    merge_tensor_scalar_output(tensor, scalar, primal, tangent, chainrules::div_rrule)
}
