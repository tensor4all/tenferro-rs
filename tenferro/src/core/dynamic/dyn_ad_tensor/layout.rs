use num_traits::Zero;

use chainrules::Tape;
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::super::tensor_ops::{tensor_element, unflatten_index_column_major};
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::structured::{canonicalize_axis_classes, StructuredTensor};
use crate::{tape, AdTensor, Error, Result};

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

fn ensure_reverse_leaf_attached<T>(input: &AdTensor<T>) -> Result<()>
where
    T: Scalar + DynTensorTyped + 'static,
{
    if input.requires_grad() && input.reverse_tape().is_none() {
        let tape = Tape::new();
        input.ensure_reverse_leaf_on(&tape)?;
    }
    Ok(())
}

fn build_axis_class_layout_from_dense<T>(
    payload: StructuredTensor<T>,
    axis_classes: &[usize],
    op_name: &'static str,
) -> Result<StructuredTensor<T>>
where
    T: Scalar,
{
    if !payload.is_dense() {
        return Err(Error::InvalidAdTensor {
            message: format!("{op_name} requires a dense payload tensor"),
        });
    }

    let canonical = canonicalize_axis_classes(axis_classes);
    if canonical != axis_classes {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires canonical axis classes, got {:?}; expected {:?}",
                axis_classes, canonical
            ),
        });
    }

    let payload = payload.into_payload();
    let class_count = axis_classes
        .last()
        .copied()
        .map(|class_id| class_id + 1)
        .unwrap_or(0);
    if payload.dims().len() != class_count {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires payload rank {} to match the number of axis classes {}",
                payload.dims().len(),
                class_count
            ),
        });
    }

    let logical_dims = axis_classes
        .iter()
        .map(|&class_id| payload.dims()[class_id])
        .collect();
    StructuredTensor::new(logical_dims, axis_classes.to_vec(), payload)
}

pub(super) fn reshape_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    new_dims: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + 'static,
{
    ensure_dense_ad_tensor_layout(input, "reshape")?;
    ensure_reverse_leaf_attached(input)?;

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

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(StructuredTensor::from_dense(
            primal
                .into_payload()
                .reshape(new_dims)
                .map_err(Error::from)?,
        ))),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensor::new_forward(
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
        AdTensorSnapshot::Reverse {
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
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "reshape reverse output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    let contiguous = cotangent.payload().contiguous(MemoryOrder::ColumnMajor);
                    let grad = StructuredTensor::from_dense(
                        contiguous.reshape(&old_dims).map_err(Error::from)?,
                    );
                    Ok(vec![(input_node, grad)])
                }),
            );
            Ok(out)
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
    T: Scalar + Copy + Zero + DynTensorTyped + 'static,
{
    ensure_dense_ad_tensor_layout(input, "take_prefix")?;
    ensure_reverse_leaf_attached(input)?;

    let original_dims = input.primal().dims().to_vec();
    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(StructuredTensor::from_dense(
            take_prefix_payload_typed(primal.payload(), axis, len)?,
        ))),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor::from_dense(take_prefix_payload_typed(primal.payload(), axis, len)?),
            StructuredTensor::from_dense(take_prefix_payload_typed(tangent.payload(), axis, len)?),
        ),
        AdTensorSnapshot::Reverse {
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
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "take_prefix reverse output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    let grad = StructuredTensor::from_dense(take_prefix_pullback_typed(
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

pub(super) fn diag_embed_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    logical_rank: usize,
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + 'static,
{
    ensure_dense_ad_tensor_layout(input, "diag_embed")?;
    ensure_reverse_leaf_attached(input)?;
    if input.ndim() != 1 {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "diag_embed requires a rank-1 dense tensor, got dims {:?}",
                input.dims()
            ),
        });
    }

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(
            StructuredTensor::from_diagonal_vector(primal.into_payload(), logical_rank)?,
        )),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor::from_diagonal_vector(primal.into_payload(), logical_rank)?,
            StructuredTensor::from_diagonal_vector(tangent.into_payload(), logical_rank)?,
        ),
        AdTensorSnapshot::Reverse {
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
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "diag_embed reverse output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        StructuredTensor::from_dense(cotangent.payload().clone()),
                    )])
                }),
            );
            Ok(out)
        }
    }
}

pub(super) fn with_axis_classes_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    axis_classes: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + 'static,
{
    ensure_dense_ad_tensor_layout(input, "with_axis_classes")?;
    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(
            build_axis_class_layout_from_dense(primal, axis_classes, "Tensor::with_axis_classes")?,
        )),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensor::new_forward(
            build_axis_class_layout_from_dense(primal, axis_classes, "Tensor::with_axis_classes")?,
            build_axis_class_layout_from_dense(tangent, axis_classes, "Tensor::with_axis_classes")?,
        ),
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = build_axis_class_layout_from_dense(
                primal,
                axis_classes,
                "Tensor::with_axis_classes",
            )?;
            let output_tangent = tangent
                .map(|value| {
                    build_axis_class_layout_from_dense(
                        value,
                        axis_classes,
                        "Tensor::with_axis_classes",
                    )
                })
                .transpose()?;
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "with_axis_classes reverse output is missing a tape node".to_string(),
                })?;
            tape::register_rule::<T>(
                &tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        StructuredTensor::from_dense(cotangent.payload().clone()),
                    )])
                }),
            );
            Ok(out)
        }
    }
}

pub(super) fn contiguous_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    order: MemoryOrder,
) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + 'static,
{
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
