use std::ops::Add;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::{
    AdResult, AutodiffError, CheckpointHint, DynValue, LinearizableOp, LinearizedOp, Schema,
    SlotSchema,
};
use tenferro_internal_frontend_core::tensor_ops::{
    tensor_element, tensor_map_binary_typed, tensor_map_unary_typed,
};
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::math::{einsum_frule, einsum_primal, einsum_rrule};
use crate::{Error, Result};

#[derive(Clone, Copy)]
pub struct AddOp;

#[derive(Clone, Copy)]
pub struct ExpOp;

#[derive(Clone, Copy)]
pub struct SumOp;

#[derive(Clone)]
pub struct EinsumOp {
    subscripts: Arc<str>,
}

#[doc(hidden)]
pub struct AddLinearized;

#[doc(hidden)]
pub struct ExpLinearized {
    output: DynTensor,
}

#[doc(hidden)]
pub struct SumLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct EinsumLinearized {
    subscripts: Arc<str>,
    inputs: Vec<DynTensor>,
}

fn differentiable_schema(slots: usize) -> Schema {
    Schema {
        slots: (0..slots)
            .map(|_| SlotSchema {
                differentiable: true,
                auxiliary: false,
            })
            .collect(),
    }
}

fn invalid_argument(message: impl Into<String>) -> Error {
    AutodiffError::InvalidArgument(message.into()).into()
}

fn into_ad_error(error: Error) -> AutodiffError {
    match error {
        Error::Autodiff(error) => error,
        other => AutodiffError::InvalidArgument(other.to_string()),
    }
}

fn structured_binary<T>(
    lhs: &StructuredTensor<T>,
    rhs: &StructuredTensor<T>,
    f: impl FnMut(T, T) -> T,
) -> Result<StructuredTensor<T>>
where
    T: Scalar + Copy,
{
    lhs.with_payload_like(tensor_map_binary_typed(lhs.payload(), rhs.payload(), f)?)
}

fn structured_unary<T, U>(
    input: &StructuredTensor<T>,
    f: impl FnMut(T) -> U,
) -> Result<StructuredTensor<U>>
where
    T: Scalar + Copy,
    U: Scalar + Copy,
{
    let payload = tensor_map_unary_typed(input.payload(), f)?;
    Ok(StructuredTensor::from(payload))
}

fn dense_host_slice<'a, T>(tensor: &'a DenseTensor<T>, context: &str) -> Result<&'a [T]> {
    tensor.buffer().as_slice().ok_or_else(|| {
        invalid_argument(format!("{context} requires host-accessible dense payload"))
    })
}

fn scalar_from_rank0<T>(value: &StructuredTensor<T>, context: &str) -> Result<T>
where
    T: Scalar + Copy,
{
    if !value.logical_dims().is_empty() {
        return Err(invalid_argument(format!(
            "{context} requires a rank-0 tensor, got {:?}",
            value.logical_dims()
        )));
    }
    tensor_element(value.payload(), &[])
}

fn structured_sum_all<T>(input: &StructuredTensor<T>) -> Result<StructuredTensor<T>>
where
    T: Scalar + Copy + Zero + Add<Output = T>,
{
    let dense = input.to_dense()?;
    let mut acc = T::zero();
    for &value in dense_host_slice(&dense, "sum")? {
        acc = acc + value;
    }
    let payload = DenseTensor::from_slice(&[acc], &[], MemoryOrder::ColumnMajor)?;
    Ok(StructuredTensor::from(payload))
}

fn structured_broadcast_scalar_like<T>(
    scalar: &StructuredTensor<T>,
    like: &StructuredTensor<T>,
) -> Result<StructuredTensor<T>>
where
    T: Scalar + Copy,
{
    let value = scalar_from_rank0(scalar, "broadcast_scalar_like")?;
    let total = like.logical_dims().iter().product();
    let payload = DenseTensor::from_slice(
        &vec![value; total],
        like.logical_dims(),
        MemoryOrder::ColumnMajor,
    )?;
    like.with_payload_like(payload)
}

fn dyn_add(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
    match (lhs, rhs) {
        (DynTensor::F32(lhs), DynTensor::F32(rhs)) => {
            Ok(DynTensor::F32(structured_binary(lhs, rhs, |x, y| x + y)?))
        }
        (DynTensor::F64(lhs), DynTensor::F64(rhs)) => {
            Ok(DynTensor::F64(structured_binary(lhs, rhs, |x, y| x + y)?))
        }
        (DynTensor::C32(lhs), DynTensor::C32(rhs)) => {
            Ok(DynTensor::C32(structured_binary(lhs, rhs, |x, y| x + y)?))
        }
        (DynTensor::C64(lhs), DynTensor::C64(rhs)) => {
            Ok(DynTensor::C64(structured_binary(lhs, rhs, |x, y| x + y)?))
        }
        _ => Err(invalid_argument(format!(
            "add requires matching dtypes, got lhs={:?}, rhs={:?}",
            lhs.scalar_type(),
            rhs.scalar_type()
        ))),
    }
}

fn dyn_mul(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
    match (lhs, rhs) {
        (DynTensor::F32(lhs), DynTensor::F32(rhs)) => {
            Ok(DynTensor::F32(structured_binary(lhs, rhs, |x, y| x * y)?))
        }
        (DynTensor::F64(lhs), DynTensor::F64(rhs)) => {
            Ok(DynTensor::F64(structured_binary(lhs, rhs, |x, y| x * y)?))
        }
        (DynTensor::C32(lhs), DynTensor::C32(rhs)) => {
            Ok(DynTensor::C32(structured_binary(lhs, rhs, |x, y| x * y)?))
        }
        (DynTensor::C64(lhs), DynTensor::C64(rhs)) => {
            Ok(DynTensor::C64(structured_binary(lhs, rhs, |x, y| x * y)?))
        }
        _ => Err(invalid_argument(format!(
            "mul requires matching dtypes, got lhs={:?}, rhs={:?}",
            lhs.scalar_type(),
            rhs.scalar_type()
        ))),
    }
}

fn dyn_exp(input: &DynTensor) -> Result<DynTensor> {
    match input {
        DynTensor::F32(value) => Ok(DynTensor::F32(structured_unary(value, |x: f32| x.exp())?)),
        DynTensor::F64(value) => Ok(DynTensor::F64(structured_unary(value, |x: f64| x.exp())?)),
        DynTensor::C32(value) => Ok(DynTensor::C32(structured_unary(value, |z: Complex32| {
            z.exp()
        })?)),
        DynTensor::C64(value) => Ok(DynTensor::C64(structured_unary(value, |z: Complex64| {
            z.exp()
        })?)),
    }
}

fn dyn_sum_all(input: &DynTensor) -> Result<DynTensor> {
    match input {
        DynTensor::F32(value) => Ok(DynTensor::F32(structured_sum_all(value)?)),
        DynTensor::F64(value) => Ok(DynTensor::F64(structured_sum_all(value)?)),
        DynTensor::C32(value) => Ok(DynTensor::C32(structured_sum_all(value)?)),
        DynTensor::C64(value) => Ok(DynTensor::C64(structured_sum_all(value)?)),
    }
}

fn dyn_broadcast_scalar_like(scalar: &DynTensor, like: &DynTensor) -> Result<DynTensor> {
    match (scalar, like) {
        (DynTensor::F32(scalar), DynTensor::F32(like)) => Ok(DynTensor::F32(
            structured_broadcast_scalar_like(scalar, like)?,
        )),
        (DynTensor::F64(scalar), DynTensor::F64(like)) => Ok(DynTensor::F64(
            structured_broadcast_scalar_like(scalar, like)?,
        )),
        (DynTensor::C32(scalar), DynTensor::C32(like)) => Ok(DynTensor::C32(
            structured_broadcast_scalar_like(scalar, like)?,
        )),
        (DynTensor::C64(scalar), DynTensor::C64(like)) => Ok(DynTensor::C64(
            structured_broadcast_scalar_like(scalar, like)?,
        )),
        _ => Err(invalid_argument(format!(
            "broadcast requires matching dtypes, got scalar={:?}, like={:?}",
            scalar.scalar_type(),
            like.scalar_type()
        ))),
    }
}

fn dense_dyn_tensor_typed<T>(value: &DynTensor, context: &str) -> Result<DenseTensor<T>>
where
    T: DynTensorTyped + Copy,
{
    let structured = T::structured_ref(value)
        .ok_or_else(|| invalid_argument(format!("{context} requires matching dtypes")))?;
    structured.to_dense()
}

fn collect_dense_dyn_tensors<T>(values: &[&DynTensor], context: &str) -> Result<Vec<DenseTensor<T>>>
where
    T: DynTensorTyped + Copy,
{
    values
        .iter()
        .map(|value| dense_dyn_tensor_typed::<T>(value, context))
        .collect()
}

fn optional_dense_dyn_tensor_typed<T>(
    value: &Option<DynTensor>,
    context: &str,
) -> Result<Option<DenseTensor<T>>>
where
    T: DynTensorTyped + Copy,
{
    value
        .as_ref()
        .map(|tensor| dense_dyn_tensor_typed::<T>(tensor, context))
        .transpose()
}

fn collect_optional_dense_dyn_tensors<T>(
    values: &[Option<DynTensor>],
    context: &str,
) -> Result<Vec<Option<DenseTensor<T>>>>
where
    T: DynTensorTyped + Copy,
{
    values
        .iter()
        .map(|value| optional_dense_dyn_tensor_typed::<T>(value, context))
        .collect()
}

fn dyn_from_dense<T>(value: DenseTensor<T>) -> DynTensor
where
    T: DynTensorTyped + Copy,
{
    T::into_dyn(StructuredTensor::from(value))
}

fn dyn_einsum_primal_t<T>(subscripts: &str, inputs: &[&DynTensor]) -> Result<DynTensor>
where
    T: crate::runtime::contracts::EinsumRuntimeValue + DynTensorTyped + Copy,
{
    let dense_inputs = collect_dense_dyn_tensors::<T>(inputs, "einsum")?;
    let input_refs: Vec<&DenseTensor<T>> = dense_inputs.iter().collect();
    let output = einsum_primal(subscripts, &input_refs)?;
    Ok(dyn_from_dense(output))
}

fn dyn_einsum_jvp_t<T>(
    subscripts: &str,
    primals: &[DynTensor],
    tangents: &[Option<DynTensor>],
) -> Result<Option<DynTensor>>
where
    T: crate::runtime::contracts::EinsumRuntimeValue + DynTensorTyped + Copy,
{
    if tangents.iter().all(Option::is_none) {
        return Ok(None);
    }
    let primal_refs: Vec<&DynTensor> = primals.iter().collect();
    let dense_primals = collect_dense_dyn_tensors::<T>(&primal_refs, "einsum_jvp")?;
    let dense_tangents = collect_optional_dense_dyn_tensors::<T>(tangents, "einsum_jvp")?;
    let primal_refs: Vec<&DenseTensor<T>> = dense_primals.iter().collect();
    let tangent_refs: Vec<Option<&DenseTensor<T>>> =
        dense_tangents.iter().map(Option::as_ref).collect();
    let tangent = einsum_frule(subscripts, &primal_refs, &tangent_refs)?;
    Ok(Some(dyn_from_dense(tangent)))
}

fn dyn_einsum_vjp_t<T>(
    subscripts: &str,
    inputs: &[DynTensor],
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: crate::runtime::contracts::EinsumRuntimeValue + DynTensorTyped + Copy,
{
    let input_refs: Vec<&DynTensor> = inputs.iter().collect();
    let dense_inputs = collect_dense_dyn_tensors::<T>(&input_refs, "einsum_vjp")?;
    let input_refs: Vec<&DenseTensor<T>> = dense_inputs.iter().collect();
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "einsum_vjp")?;
    let grads = einsum_rrule(subscripts, &input_refs, &dense_cotangent)?;
    Ok(grads
        .into_iter()
        .zip(input_grad_mask.iter().copied())
        .map(|(grad, needed)| needed.then(|| dyn_from_dense(grad)))
        .collect())
}

impl LinearizableOp<DynTensor> for AddOp {
    type Linearized = AddLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_add(inputs[0], inputs[1]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        _inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(AddLinearized)
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for AddLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match (&input_tangents[0], &input_tangents[1]) {
            (None, None) => None,
            (Some(lhs), None) => Some(lhs.clone()),
            (None, Some(rhs)) => Some(rhs.clone()),
            (Some(lhs), Some(rhs)) => Some(dyn_add(lhs, rhs).map_err(into_ad_error)?),
        };
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let grad = output_cotangents[0].clone();
        Ok(vec![
            input_grad_mask[0].then(|| grad.clone()).flatten(),
            input_grad_mask[1].then_some(grad).flatten(),
        ])
    }
}

impl LinearizableOp<DynTensor> for ExpOp {
    type Linearized = ExpLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_exp(inputs[0]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        _inputs: &[&DynTensor],
        outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(ExpLinearized {
            output: outputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for ExpLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![match &input_tangents[0] {
            Some(tangent) => Some(dyn_mul(&self.output, tangent).map_err(into_ad_error)?),
            None => None,
        }])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![if input_grad_mask[0] {
            match &output_cotangents[0] {
                Some(grad_out) => Some(dyn_mul(&self.output, grad_out).map_err(into_ad_error)?),
                None => None,
            }
        } else {
            None
        }])
    }
}

impl LinearizableOp<DynTensor> for SumOp {
    type Linearized = SumLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_sum_all(inputs[0]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(SumLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for SumLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![match &input_tangents[0] {
            Some(tangent) => Some(dyn_sum_all(tangent).map_err(into_ad_error)?),
            None => None,
        }])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![if input_grad_mask[0] {
            match &output_cotangents[0] {
                Some(grad_out) => {
                    Some(dyn_broadcast_scalar_like(grad_out, &self.input).map_err(into_ad_error)?)
                }
                None => None,
            }
        } else {
            None
        }])
    }
}

impl EinsumOp {
    pub fn new(subscripts: impl Into<String>) -> Self {
        Self {
            subscripts: Arc::<str>::from(subscripts.into()),
        }
    }
}

impl LinearizableOp<DynTensor> for EinsumOp {
    type Linearized = EinsumLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs.first() {
            Some(DynTensor::F32(_)) => dyn_einsum_primal_t::<f32>(&self.subscripts, inputs),
            Some(DynTensor::F64(_)) => dyn_einsum_primal_t::<f64>(&self.subscripts, inputs),
            Some(DynTensor::C32(_)) => dyn_einsum_primal_t::<Complex32>(&self.subscripts, inputs),
            Some(DynTensor::C64(_)) => dyn_einsum_primal_t::<Complex64>(&self.subscripts, inputs),
            None => Err(invalid_argument("einsum requires at least one input")),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(inputs.len()))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(EinsumLinearized {
            subscripts: self.subscripts.clone(),
            inputs: inputs.iter().map(|input| (*input).clone()).collect(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for EinsumLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match self.inputs.first() {
            Some(DynTensor::F32(_)) => {
                dyn_einsum_jvp_t::<f32>(&self.subscripts, &self.inputs, input_tangents)
            }
            Some(DynTensor::F64(_)) => {
                dyn_einsum_jvp_t::<f64>(&self.subscripts, &self.inputs, input_tangents)
            }
            Some(DynTensor::C32(_)) => {
                dyn_einsum_jvp_t::<Complex32>(&self.subscripts, &self.inputs, input_tangents)
            }
            Some(DynTensor::C64(_)) => {
                dyn_einsum_jvp_t::<Complex64>(&self.subscripts, &self.inputs, input_tangents)
            }
            None => Err(invalid_argument(
                "einsum linearization requires at least one input",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok((0..self.inputs.len()).map(|_| None).collect());
        };
        match self.inputs.first() {
            Some(DynTensor::F32(_)) => {
                dyn_einsum_vjp_t::<f32>(&self.subscripts, &self.inputs, cotangent, input_grad_mask)
            }
            Some(DynTensor::F64(_)) => {
                dyn_einsum_vjp_t::<f64>(&self.subscripts, &self.inputs, cotangent, input_grad_mask)
            }
            Some(DynTensor::C32(_)) => dyn_einsum_vjp_t::<Complex32>(
                &self.subscripts,
                &self.inputs,
                cotangent,
                input_grad_mask,
            ),
            Some(DynTensor::C64(_)) => dyn_einsum_vjp_t::<Complex64>(
                &self.subscripts,
                &self.inputs,
                cotangent,
                input_grad_mask,
            ),
            None => Err(invalid_argument(
                "einsum linearization requires at least one input",
            )),
        }
        .map_err(into_ad_error)
    }
}

pub fn add_dyn_values(lhs: &DynValue, rhs: &DynValue) -> AdResult<DynValue> {
    AddOp.apply_one(&[lhs, rhs])
}

pub fn exp_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    ExpOp.apply_one(&[input])
}

pub fn sum_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    SumOp.apply_one(&[input])
}

pub fn einsum_dyn_values(subscripts: &str, inputs: &[&DynValue]) -> AdResult<DynValue> {
    EinsumOp::new(subscripts).apply_one(inputs)
}
