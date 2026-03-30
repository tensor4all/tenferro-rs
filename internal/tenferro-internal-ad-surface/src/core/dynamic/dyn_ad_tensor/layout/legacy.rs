use num_traits::Zero;

use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::AdTensor;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::super::super::tensor_ops::{tensor_element, unflatten_index_column_major};
use super::{
    can_use_edge_reverse, dense_structured, edge_dense_layout, edge_permute, edge_take_prefix,
    ensure_dense_ad_tensor_layout, ensure_reverse_leaf_attached, inverse_permutation,
    DenseLayoutUnaryKind,
};
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::{tape, Error, Result};

pub(crate) fn reshape_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    new_dims: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + 'static,
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

    if can_use_edge_reverse(input) {
        return edge_dense_layout(input, DenseLayoutUnaryKind::Reshape, new_dims);
    }

    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => {
            AdTensor::try_from(AdTensorSnapshot::Primal(dense_structured(
                primal
                    .into_payload()
                    .reshape(new_dims)
                    .map_err(Error::from)?,
            )))
        }
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: dense_structured(
                    primal
                        .into_payload()
                        .reshape(new_dims)
                        .map_err(Error::from)?,
                ),
                tangent: dense_structured(
                    tangent
                        .into_payload()
                        .reshape(new_dims)
                        .map_err(Error::from)?,
                ),
            })
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = dense_structured(
                primal
                    .into_payload()
                    .reshape(new_dims)
                    .map_err(Error::from)?,
            );
            let output_tangent = tangent
                .map(|t| {
                    Result::Ok(dense_structured(
                        t.into_payload().reshape(new_dims).map_err(Error::from)?,
                    ))
                })
                .transpose()?;
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "reshape reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
                Box::new(move |cotangent| {
                    let grad = dense_structured(
                        cotangent
                            .payload()
                            .reshape(&old_dims)
                            .map_err(Error::from)?,
                    );
                    Ok(vec![(input_node, grad)])
                }),
            );
            Ok(out)
        }
    }
}

pub(crate) fn view_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    new_dims: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + 'static,
{
    ensure_dense_ad_tensor_layout(input, "view")?;

    let old_dims = input.primal().dims().to_vec();
    let old_len: usize = old_dims.iter().product();
    let new_len: usize = new_dims.iter().product();
    if old_len != new_len {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "view requires element count to stay constant, got old_dims={old_dims:?}, new_dims={new_dims:?}"
            ),
        });
    }

    if can_use_edge_reverse(input) {
        return edge_dense_layout(input, DenseLayoutUnaryKind::View, new_dims);
    }

    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => AdTensor::try_from(AdTensorSnapshot::Primal(
            dense_structured(primal.into_payload().view(new_dims).map_err(Error::from)?),
        )),
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: dense_structured(
                    primal.into_payload().view(new_dims).map_err(Error::from)?,
                ),
                tangent: dense_structured(
                    tangent.into_payload().view(new_dims).map_err(Error::from)?,
                ),
            })
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal =
                dense_structured(primal.into_payload().view(new_dims).map_err(Error::from)?);
            let output_tangent = tangent
                .map(|t| {
                    Result::Ok(dense_structured(
                        t.into_payload().view(new_dims).map_err(Error::from)?,
                    ))
                })
                .transpose()?;
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "view reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
                Box::new(move |cotangent| {
                    let grad = dense_structured(
                        cotangent
                            .payload()
                            .reshape(&old_dims)
                            .map_err(Error::from)?,
                    );
                    Ok(vec![(input_node, grad)])
                }),
            );
            Ok(out)
        }
    }
}

pub(crate) fn permute_ad_tensor_typed<T>(input: &AdTensor<T>, perm: &[usize]) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    if can_use_edge_reverse(input) {
        return edge_permute(input, perm);
    }

    ensure_reverse_leaf_attached(input)?;
    let inverse = inverse_permutation(perm);

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => {
            AdTensor::try_from(AdTensorSnapshot::Primal(primal.permute_logical(perm)?))
        }
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: primal.permute_logical(perm)?,
                tangent: tangent.permute_logical(perm)?,
            })
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = primal.permute_logical(perm)?;
            let output_tangent = tangent
                .as_ref()
                .map(|value| value.permute_logical(perm))
                .transpose()?;
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "permute reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
                Box::new(move |cotangent| {
                    Ok(vec![(input_node, cotangent.permute_logical(&inverse)?)])
                }),
            );
            Ok(out)
        }
    }
}

pub(super) fn take_prefix_payload_typed<T>(
    tensor: &Tensor<T>,
    axis: usize,
    len: usize,
) -> Result<Tensor<T>>
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

pub(super) fn take_prefix_pullback_typed<T>(
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

pub(crate) fn take_prefix_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    axis: usize,
    len: usize,
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + Zero + DynTensorTyped + Send + Sync + 'static,
{
    ensure_dense_ad_tensor_layout(input, "take_prefix")?;
    let original_dims = input.primal().dims().to_vec();

    if can_use_edge_reverse(input) {
        return edge_take_prefix(input, axis, len);
    }

    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => AdTensor::try_from(AdTensorSnapshot::Primal(
            dense_structured(take_prefix_payload_typed(primal.payload(), axis, len)?),
        )),
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: dense_structured(take_prefix_payload_typed(primal.payload(), axis, len)?),
                tangent: dense_structured(take_prefix_payload_typed(tangent.payload(), axis, len)?),
            })
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal =
                dense_structured(take_prefix_payload_typed(primal.payload(), axis, len)?);
            let output_tangent = tangent
                .as_ref()
                .map(|t| {
                    Result::Ok(dense_structured(take_prefix_payload_typed(
                        t.payload(),
                        axis,
                        len,
                    )?))
                })
                .transpose()?;
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "take_prefix reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
                Box::new(move |cotangent| {
                    let grad = dense_structured(take_prefix_pullback_typed(
                        cotangent.payload(),
                        axis,
                        &original_dims,
                    )?);
                    Ok(vec![(input_node, grad)])
                }),
            );
            Ok(out)
        }
    }
}

pub(crate) fn contiguous_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    order: MemoryOrder,
) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    if can_use_edge_reverse(input) {
        let output_primal = input
            .structured_primal()
            .with_payload_like(input.structured_primal().payload().contiguous(order))?;
        return AdTensor::from_reverse_edge_handle(
            output_primal,
            input.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
                op: "edge_contiguous",
            })?,
        );
    }

    ensure_reverse_leaf_attached(input)?;
    let mapped = match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => {
            AdTensorSnapshot::Primal(primal.with_payload_like(primal.payload().contiguous(order))?)
        }
        AdTensorSnapshot::Forward { primal, tangent } => AdTensorSnapshot::Forward {
            primal: primal.with_payload_like(primal.payload().contiguous(order))?,
            tangent: tangent.with_payload_like(tangent.payload().contiguous(order))?,
        },
        AdTensorSnapshot::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => AdTensorSnapshot::Reverse {
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
