use std::marker::PhantomData;

use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::AdTensor;
use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema};

use super::{edge_output_to_ad, ensure_dense_ad_tensor_layout, ensure_reverse_leaf_attached};
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::structured::{canonicalize_axis_classes, StructuredTensor};
use crate::{tape, Error, Result};

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

fn can_use_edge_reverse<T>(input: &AdTensor<T>) -> bool
where
    T: Scalar + DynTensorTyped + 'static,
{
    input.structured_tangent().is_none() && input.reverse_edge_value().is_some()
}

#[derive(Clone, Copy)]
struct EdgeDiagEmbedOp<T> {
    logical_rank: usize,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgeDiagEmbedOp<T>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        Ok(vec![StructuredTensor(
            tenferro_tensor::StructuredTensor::from_diagonal_vector(
                inputs[0].payload().clone(),
                self.logical_rank,
            )
            .map_err(ad_invalid_argument)?,
        )])
    }

    fn input_schema(&self, _inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(())
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(cotangent) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(cotangent.payload().clone()),
        ))])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_diagonal_vector(
                tangent.payload().clone(),
                self.logical_rank,
            )
            .map_err(ad_invalid_argument)?,
        ))])
    }
}

#[derive(Clone)]
struct EdgeWithAxisClassesOp<T> {
    axis_classes: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgeWithAxisClassesOp<T>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    type SavedBackward = ();
    type SavedJvp = ();

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        Ok(vec![build_axis_class_layout_from_dense(
            inputs[0].clone(),
            &self.axis_classes,
            "Tensor::with_axis_classes",
        )
        .map_err(ad_invalid_argument)?])
    }

    fn input_schema(&self, _inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(())
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        _saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(cotangent) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(cotangent.payload().clone()),
        ))])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(
            build_axis_class_layout_from_dense(
                tangent.clone(),
                &self.axis_classes,
                "Tensor::with_axis_classes",
            )
            .map_err(ad_invalid_argument)?,
        )])
    }
}

fn edge_diag_embed<T>(input: &AdTensor<T>, logical_rank: usize) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    let input = input.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
        op: "edge_diag_embed",
    })?;
    let output = EdgeDiagEmbedOp::<T> {
        logical_rank,
        _marker: PhantomData,
    }
    .apply_one(&[input.as_ref()])
    .map_err(Error::from)?;
    edge_output_to_ad(output)
}

fn edge_with_axis_classes<T>(input: &AdTensor<T>, axis_classes: &[usize]) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    let input = input.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
        op: "edge_with_axis_classes",
    })?;
    let output = EdgeWithAxisClassesOp::<T> {
        axis_classes: axis_classes.to_vec(),
        _marker: PhantomData,
    }
    .apply_one(&[input.as_ref()])
    .map_err(Error::from)?;
    edge_output_to_ad(output)
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
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
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

    if can_use_edge_reverse(input) {
        return edge_diag_embed(input, logical_rank);
    }

    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => AdTensor::try_from(AdTensorSnapshot::Primal(
            StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                primal.0.into_payload(),
                logical_rank,
            )?),
        )),
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                    primal.0.into_payload(),
                    logical_rank,
                )?),
                tangent: StructuredTensor(tenferro_tensor::StructuredTensor::from_diagonal_vector(
                    tangent.0.into_payload(),
                    logical_rank,
                )?),
            })
        }
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
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "diag_embed reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
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
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    ensure_dense_ad_tensor_layout(input, "with_axis_classes")?;

    if can_use_edge_reverse(input) {
        return edge_with_axis_classes(input, axis_classes);
    }

    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => AdTensor::try_from(AdTensorSnapshot::Primal(
            build_axis_class_layout_from_dense(primal, axis_classes, "Tensor::with_axis_classes")?,
        )),
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: build_axis_class_layout_from_dense(
                    primal,
                    axis_classes,
                    "Tensor::with_axis_classes",
                )?,
                tangent: build_axis_class_layout_from_dense(
                    tangent,
                    axis_classes,
                    "Tensor::with_axis_classes",
                )?,
            })
        }
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
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "with_axis_classes reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
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
