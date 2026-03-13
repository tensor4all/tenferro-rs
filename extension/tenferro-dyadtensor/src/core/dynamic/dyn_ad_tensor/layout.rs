use num_traits::Zero;
use std::sync::atomic::{AtomicU64, Ordering};

use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::super::tensor_ops::{tensor_element, unflatten_index_column_major};
use crate::structured::StructuredTensor;
use crate::{tape, AdTensor, AdValue, Error, NodeId, Result};

static NEXT_AD_TENSOR_NODE_ID: AtomicU64 = AtomicU64::new(1_u64 << 61);

pub(super) fn fresh_ad_tensor_node_id() -> NodeId {
    NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed))
}

fn ensure_dense_ad_tensor_layout<T>(input: &AdTensor<T>, op_name: &'static str) -> Result<()>
where
    T: Scalar,
{
    if input.is_dense() {
        return Ok(());
    }
    Err(Error::InvalidAdTensor {
        message: format!("{op_name} currently supports only dense tensors"),
    })
}

pub(super) fn reshape_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    new_dims: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + 'static,
{
    ensure_dense_ad_tensor_layout(input, "reshape")?;

    let old_dims = input.primal().dims().to_vec();
    let old_len: usize = old_dims.iter().product();
    let new_len: usize = new_dims.iter().product();
    if old_len != new_len {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "reshape requires element count to stay constant, got old_dims={old_dims:?}, new_dims={new_dims:?}"
            ),
        });
    }

    match input.as_value().clone() {
        AdValue::Primal(primal) => Ok(AdTensor::new_primal(StructuredTensor::from_dense(
            primal
                .into_payload()
                .reshape(new_dims)
                .map_err(Error::from)?,
        ))),
        AdValue::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor::from_dense(
                primal
                    .into_payload()
                    .reshape(new_dims)
                    .map_err(Error::from)?,
            ),
            StructuredTensor::from_dense(
                tangent
                    .into_payload()
                    .reshape(new_dims)
                    .map_err(Error::from)?,
            ),
        ),
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = StructuredTensor::from_dense(
                primal
                    .into_payload()
                    .reshape(new_dims)
                    .map_err(Error::from)?,
            );
            let output_tangent = tangent
                .map(|t| {
                    Result::Ok(StructuredTensor::from_dense(
                        t.into_payload().reshape(new_dims).map_err(Error::from)?,
                    ))
                })
                .transpose()?;
            let output_node = fresh_ad_tensor_node_id();
            tape::register_rule::<T>(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    let contiguous = cotangent.contiguous(MemoryOrder::ColumnMajor);
                    Ok(vec![(
                        input_node,
                        contiguous.reshape(&old_dims).map_err(Error::from)?,
                    )])
                }),
            );
            AdTensor::new_reverse(output_primal, output_node, tape, output_tangent)
        }
    }
}

fn take_prefix_payload_typed<T>(tensor: &Tensor<T>, axis: usize, len: usize) -> Result<Tensor<T>>
where
    T: Scalar + Copy + Zero,
{
    if axis >= tensor.dims().len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "take_prefix axis {} out of range for dims {:?}",
                axis,
                tensor.dims()
            ),
        });
    }
    if len > tensor.dims()[axis] {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "take_prefix length {} exceeds dimension {} on axis {}",
                len,
                tensor.dims()[axis],
                axis
            ),
        });
    }

    let mut new_dims = tensor.dims().to_vec();
    new_dims[axis] = len;
    let total: usize = new_dims.iter().product();
    let mut idx = vec![0usize; new_dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, &new_dims, &mut idx);
        out.push(tensor_element(tensor, &idx)?);
    }
    Tensor::from_slice(&out, &new_dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

fn take_prefix_pullback_typed<T>(
    cotangent: &Tensor<T>,
    axis: usize,
    original_dims: &[usize],
) -> Result<Tensor<T>>
where
    T: Scalar + Copy + Zero,
{
    if axis >= original_dims.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "take_prefix pullback axis {} out of range for dims {:?}",
                axis, original_dims
            ),
        });
    }
    if cotangent.dims().len() != original_dims.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "take_prefix pullback rank mismatch: cotangent={:?}, original={:?}",
                cotangent.dims(),
                original_dims
            ),
        });
    }

    for (i, (&got, &expected)) in cotangent
        .dims()
        .iter()
        .zip(original_dims.iter())
        .enumerate()
    {
        if i == axis {
            if got > expected {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "take_prefix pullback axis {} exceeds original size: got={}, expected<={}",
                        axis, got, expected
                    ),
                });
            }
        } else if got != expected {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "take_prefix pullback shape mismatch on axis {}: got={}, expected={}",
                    i, got, expected
                ),
            });
        }
    }

    let total: usize = original_dims.iter().product();
    let mut idx = vec![0usize; original_dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, original_dims, &mut idx);
        if idx[axis] < cotangent.dims()[axis] {
            out.push(tensor_element(cotangent, &idx)?);
        } else {
            out.push(T::zero());
        }
    }
    Tensor::from_slice(&out, original_dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

pub(super) fn take_prefix_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    axis: usize,
    len: usize,
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + Zero + 'static,
{
    ensure_dense_ad_tensor_layout(input, "take_prefix")?;

    let original_dims = input.primal().dims().to_vec();
    match input.as_value().clone() {
        AdValue::Primal(primal) => Ok(AdTensor::new_primal(StructuredTensor::from_dense(
            take_prefix_payload_typed(primal.payload(), axis, len)?,
        ))),
        AdValue::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor::from_dense(take_prefix_payload_typed(primal.payload(), axis, len)?),
            StructuredTensor::from_dense(take_prefix_payload_typed(tangent.payload(), axis, len)?),
        ),
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = StructuredTensor::from_dense(take_prefix_payload_typed(
                primal.payload(),
                axis,
                len,
            )?);
            let output_tangent = tangent
                .as_ref()
                .map(|t| {
                    Result::Ok(StructuredTensor::from_dense(take_prefix_payload_typed(
                        t.payload(),
                        axis,
                        len,
                    )?))
                })
                .transpose()?;
            let output_node = fresh_ad_tensor_node_id();
            tape::register_rule::<T>(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        take_prefix_pullback_typed(cotangent, axis, &original_dims)?,
                    )])
                }),
            );
            AdTensor::new_reverse(output_primal, output_node, tape, output_tangent)
        }
    }
}

pub(super) fn diag_embed_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    logical_rank: usize,
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + 'static,
{
    ensure_dense_ad_tensor_layout(input, "diag_embed")?;
    if input.ndim() != 1 {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "diag_embed requires a rank-1 dense tensor, got dims {:?}",
                input.dims()
            ),
        });
    }

    match input.as_value().clone() {
        AdValue::Primal(primal) => Ok(AdTensor::new_primal(
            StructuredTensor::from_diagonal_vector(primal.into_payload(), logical_rank)?,
        )),
        AdValue::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor::from_diagonal_vector(primal.into_payload(), logical_rank)?,
            StructuredTensor::from_diagonal_vector(tangent.into_payload(), logical_rank)?,
        ),
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal =
                StructuredTensor::from_diagonal_vector(primal.into_payload(), logical_rank)?;
            let output_tangent = tangent
                .map(|t| StructuredTensor::from_diagonal_vector(t.into_payload(), logical_rank))
                .transpose()?;
            let output_node = fresh_ad_tensor_node_id();
            tape::register_rule::<T>(
                tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(input_node, cotangent.clone())])),
            );
            AdTensor::new_reverse(output_primal, output_node, tape, output_tangent)
        }
    }
}

pub(super) fn contiguous_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    order: MemoryOrder,
) -> Result<AdTensor<T>>
where
    T: Scalar,
{
    let mapped = match input.as_value().clone() {
        AdValue::Primal(primal) => {
            AdValue::Primal(primal.with_payload_like(primal.payload().contiguous(order))?)
        }
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: primal.with_payload_like(primal.payload().contiguous(order))?,
            tangent: tangent.with_payload_like(tangent.payload().contiguous(order))?,
        },
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => AdValue::Reverse {
            primal: primal.with_payload_like(primal.payload().contiguous(order))?,
            node,
            tape,
            tangent: tangent
                .map(|value| value.with_payload_like(value.payload().contiguous(order)))
                .transpose()?,
        },
    };
    AdTensor::try_from(mapped)
}
