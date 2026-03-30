mod legacy;
mod structured;

use std::marker::PhantomData;

use num_traits::Zero;

use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::AdTensor;
use tenferro_tensor::Tensor;
use tidu::expert::Tape;
use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema, Value};

use crate::core::DynTensorTyped;
use crate::structured::StructuredTensor;
use crate::{Error, Result};

pub(crate) use legacy::{
    contiguous_ad_tensor_typed, permute_ad_tensor_typed, reshape_ad_tensor_typed,
    take_prefix_ad_tensor_typed, view_ad_tensor_typed,
};

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

fn can_use_edge_reverse<T>(input: &AdTensor<T>) -> bool
where
    T: Scalar + DynTensorTyped + 'static,
{
    input.structured_tangent().is_none() && input.reverse_edge_value().is_some()
}

pub(super) fn edge_output_to_ad<T>(output: Value<StructuredTensor<T>>) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + 'static,
{
    AdTensor::from_reverse_edge_value(output)
}

#[derive(Clone, Copy)]
enum DenseLayoutUnaryKind {
    Reshape,
    View,
}

#[derive(Clone)]
struct EdgeDenseLayoutSaved {
    old_dims: Vec<usize>,
}

#[derive(Clone)]
struct EdgeDenseLayoutOp<T> {
    kind: DenseLayoutUnaryKind,
    new_dims: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgeDenseLayoutOp<T>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    type SavedBackward = EdgeDenseLayoutSaved;
    type SavedJvp = EdgeDenseLayoutSaved;

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let input = inputs[0];
        if !input.is_dense() {
            return Err(AutodiffError::InvalidArgument(
                "edge dense layout op currently supports only dense tensors".to_string(),
            ));
        }
        let payload = match self.kind {
            DenseLayoutUnaryKind::Reshape => input.payload().reshape(&self.new_dims),
            DenseLayoutUnaryKind::View => input.payload().view(&self.new_dims),
        }
        .map_err(ad_invalid_argument)?;
        Ok(vec![dense_structured(payload)])
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
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeDenseLayoutSaved {
            old_dims: inputs[0].logical_dims().to_vec(),
        })
    }

    fn save_for_jvp(
        &self,
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgeDenseLayoutSaved {
            old_dims: inputs[0].logical_dims().to_vec(),
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(cotangent) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        let grad = cotangent
            .payload()
            .reshape(&saved.old_dims)
            .map(dense_structured)
            .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        if !tangent.is_dense() {
            return Err(AutodiffError::InvalidArgument(
                "edge dense layout op tangent must be dense".to_string(),
            ));
        }
        let payload = match self.kind {
            DenseLayoutUnaryKind::Reshape => tangent.payload().reshape(&self.new_dims),
            DenseLayoutUnaryKind::View => tangent.payload().view(&self.new_dims),
        }
        .map_err(ad_invalid_argument)?;
        Ok(vec![Some(dense_structured(payload))])
    }
}

#[derive(Clone)]
struct EdgePermuteSaved {
    inverse: Vec<usize>,
}

#[derive(Clone)]
struct EdgePermuteOp<T> {
    perm: Vec<usize>,
    inverse: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgePermuteOp<T>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    type SavedBackward = EdgePermuteSaved;
    type SavedJvp = EdgePermuteSaved;

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        Ok(vec![inputs[0]
            .permute_logical(&self.perm)
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
        Ok(EdgePermuteSaved {
            inverse: self.inverse.clone(),
        })
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgePermuteSaved {
            inverse: self.inverse.clone(),
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(cotangent) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        Ok(vec![Some(
            cotangent
                .permute_logical(&saved.inverse)
                .map_err(ad_invalid_argument)?,
        )])
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
            tangent
                .permute_logical(&self.perm)
                .map_err(ad_invalid_argument)?,
        )])
    }
}

#[derive(Clone)]
struct EdgeTakePrefixSaved {
    axis: usize,
    original_dims: Vec<usize>,
}

#[derive(Clone)]
struct EdgeTakePrefixOp<T> {
    axis: usize,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgeTakePrefixOp<T>
where
    T: Scalar + Copy + Zero + DynTensorTyped + Send + Sync + 'static,
{
    type SavedBackward = EdgeTakePrefixSaved;
    type SavedJvp = EdgeTakePrefixSaved;

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let input = inputs[0];
        if !input.is_dense() {
            return Err(AutodiffError::InvalidArgument(
                "edge take_prefix currently supports only dense tensors".to_string(),
            ));
        }
        let payload = legacy::take_prefix_payload_typed(input.payload(), self.axis, self.len)
            .map_err(ad_invalid_argument)?;
        Ok(vec![dense_structured(payload)])
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
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeTakePrefixSaved {
            axis: self.axis,
            original_dims: inputs[0].logical_dims().to_vec(),
        })
    }

    fn save_for_jvp(
        &self,
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgeTakePrefixSaved {
            axis: self.axis,
            original_dims: inputs[0].logical_dims().to_vec(),
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(cotangent) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        let grad = legacy::take_prefix_pullback_typed(
            cotangent.payload(),
            saved.axis,
            &saved.original_dims,
        )
        .map(dense_structured)
        .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        if !tangent.is_dense() {
            return Err(AutodiffError::InvalidArgument(
                "edge take_prefix tangent must be dense".to_string(),
            ));
        }
        let payload = legacy::take_prefix_payload_typed(tangent.payload(), self.axis, self.len)
            .map_err(ad_invalid_argument)?;
        Ok(vec![Some(dense_structured(payload))])
    }
}

fn edge_dense_layout<T>(
    input: &AdTensor<T>,
    kind: DenseLayoutUnaryKind,
    new_dims: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    let input = input.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
        op: "edge_dense_layout",
    })?;
    let output = EdgeDenseLayoutOp::<T> {
        kind,
        new_dims: new_dims.to_vec(),
        _marker: PhantomData,
    }
    .apply_one(&[input.as_ref()])
    .map_err(Error::from)?;
    edge_output_to_ad(output)
}

fn edge_permute<T>(input: &AdTensor<T>, perm: &[usize]) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped + Send + Sync + 'static,
{
    let input = input
        .reverse_edge_value()
        .ok_or(Error::UnsupportedAdOp { op: "edge_permute" })?;
    let output = EdgePermuteOp::<T> {
        perm: perm.to_vec(),
        inverse: inverse_permutation(perm),
        _marker: PhantomData,
    }
    .apply_one(&[input.as_ref()])
    .map_err(Error::from)?;
    edge_output_to_ad(output)
}

fn edge_take_prefix<T>(input: &AdTensor<T>, axis: usize, len: usize) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + Zero + DynTensorTyped + Send + Sync + 'static,
{
    let input = input.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
        op: "edge_take_prefix",
    })?;
    let output = EdgeTakePrefixOp::<T> {
        axis,
        len,
        _marker: PhantomData,
    }
    .apply_one(&[input.as_ref()])
    .map_err(Error::from)?;
    edge_output_to_ad(output)
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

fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; perm.len()];
    for (new_axis, &old_axis) in perm.iter().enumerate() {
        inverse[old_axis] = new_axis;
    }
    inverse
}

fn dense_structured<T: Scalar>(payload: Tensor<T>) -> StructuredTensor<T> {
    StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(payload))
}

pub(super) fn diag_embed_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    logical_rank: usize,
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    structured::diag_embed_ad_tensor_typed(input, logical_rank)
}

pub(super) fn with_axis_classes_ad_tensor_typed<T>(
    input: &AdTensor<T>,
    axis_classes: &[usize],
) -> Result<AdTensor<T>>
where
    T: Scalar + Copy + DynTensorTyped + Send + Sync + 'static,
{
    structured::with_axis_classes_ad_tensor_typed(input, axis_classes)
}
