use tenferro_algebra::Scalar;

use super::{ensure_dense_ad_tensor_layout, ensure_reverse_leaf_attached};
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::structured::{canonicalize_axis_classes, StructuredTensor};
use crate::{tape, AdTensor, Error, Result};

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
    Ok(StructuredTensor(tenferro_tensor::StructuredTensor::new(
        logical_dims,
        axis_classes.to_vec(),
        payload,
    )?))
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
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_diagonal_vector(
                primal.0.into_payload(),
                logical_rank,
            )?,
        ))),
        AdTensorSnapshot::Forward { primal, tangent } => AdTensor::new_forward(
            StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                primal.0.into_payload(),
                logical_rank,
            )?),
            StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                tangent.0.into_payload(),
                logical_rank,
            )?),
        ),
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal =
                StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                    primal.0.into_payload(),
                    logical_rank,
                )?);
            let output_tangent = tangent
                .map(|t| {
                    tenferro_tensor::StructuredTensor::from_diagonal_vector(
                        t.0.into_payload(),
                        logical_rank,
                    )
                    .map(StructuredTensor)
                })
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
                        StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(
                            cotangent.payload().clone(),
                        )),
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
                        StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(
                            cotangent.payload().clone(),
                        )),
                    )])
                }),
            );
            Ok(out)
        }
    }
}
