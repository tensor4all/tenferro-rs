use std::fmt;
use std::ops::Add;

use chainrules_core::AutodiffError;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_algebra::Scalar;
use tenferro_internal_frontend_core::tensor_ops::{
    tensor_element, tensor_map_binary_typed, tensor_map_unary_typed,
};
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, ScalarType, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::{CheckpointHint, Error, LinearizableOp, LinearizedOp, Result, Schema, SlotSchema};

pub struct Tensor {
    inner: tidu::Value<DynTensor>,
}

impl Tensor {
    pub fn new(primal: DynTensor) -> Self {
        Self {
            inner: tidu::Value::new(primal),
        }
    }

    pub(crate) fn from_value(inner: tidu::Value<DynTensor>) -> Self {
        Self { inner }
    }

    pub(crate) fn value(&self) -> &tidu::Value<DynTensor> {
        &self.inner
    }

    pub(crate) fn primal(&self) -> &DynTensor {
        self.inner.primal()
    }

    pub fn from_slice<T>(data: &[T], dims: &[usize]) -> Result<Self>
    where
        T: DynTensorTyped + Copy,
    {
        let payload = DenseTensor::<T>::from_slice(data, dims, MemoryOrder::ColumnMajor)?;
        Ok(Self::from(payload))
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.primal().scalar_type()
    }

    pub fn dims(&self) -> &[usize] {
        self.primal().dims()
    }

    pub fn ndim(&self) -> usize {
        self.primal().ndim()
    }

    pub fn len(&self) -> usize {
        self.primal().len()
    }

    pub fn is_empty(&self) -> bool {
        self.primal().is_empty()
    }

    pub fn axis_classes(&self) -> &[usize] {
        self.primal().axis_classes()
    }

    pub fn is_dense(&self) -> bool {
        self.primal().is_dense()
    }

    pub fn is_diag(&self) -> bool {
        self.primal().is_diag()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn requires_grad_(self, enabled: bool) -> Self {
        Self {
            inner: self.inner.requires_grad_(enabled),
        }
    }

    pub fn detach(&self) -> Self {
        Self::new(self.primal().clone())
    }

    pub fn to_dense(&self) -> Result<Self> {
        Ok(Self::new(self.primal().to_dense()?))
    }

    pub fn grad(&self) -> Result<Option<Self>> {
        Ok(self.inner.grad()?.map(Self::new))
    }

    pub fn zero_grad(&self) -> Result<()> {
        Ok(self.inner.zero_grad()?)
    }

    pub fn backward(&self) -> Result<()> {
        Ok(self.inner.backward()?)
    }

    pub fn backward_with_seed(&self, seed: &Self) -> Result<()> {
        Ok(self.inner.backward_with_seed(seed.primal().clone())?)
    }

    pub fn shares_reverse_graph(&self, other: &Self) -> bool {
        self.inner.shares_reverse_graph(&other.inner)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_value(
            AddOp.apply_one(&[self.value(), rhs.value()])?,
        ))
    }

    pub fn exp(&self) -> Result<Self> {
        Ok(Self::from_value(ExpOp.apply_one(&[self.value()])?))
    }

    pub fn sum(&self) -> Result<Self> {
        Ok(Self::from_value(SumOp.apply_one(&[self.value()])?))
    }

    pub fn try_to_vec<T>(&self) -> Result<Vec<T>>
    where
        T: DynTensorTyped + Copy,
    {
        let structured = T::structured_ref(self.primal()).ok_or_else(|| {
            invalid_argument(format!(
                "dtype mismatch in try_to_vec: tensor={:?}",
                self.scalar_type()
            ))
        })?;
        let dense = structured.to_dense()?;
        let slice = dense
            .buffer()
            .as_slice()
            .ok_or_else(|| invalid_argument("try_to_vec requires host-accessible dense payload"))?;
        Ok(slice.to_vec())
    }

    pub fn try_get<T>(&self, index: &[usize]) -> Result<T>
    where
        T: DynTensorTyped + Copy,
    {
        let structured = T::structured_ref(self.primal()).ok_or_else(|| {
            invalid_argument(format!(
                "dtype mismatch in try_get: tensor={:?}",
                self.scalar_type()
            ))
        })?;
        tensor_element(structured.payload(), index)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("scalar_type", &self.scalar_type())
            .field("dims", &self.dims())
            .field("requires_grad", &self.requires_grad())
            .finish()
    }
}

impl<T> From<DenseTensor<T>> for Tensor
where
    T: DynTensorTyped + Copy,
{
    fn from(value: DenseTensor<T>) -> Self {
        Self::from(StructuredTensor::from(value))
    }
}

impl<T> From<StructuredTensor<T>> for Tensor
where
    T: DynTensorTyped + Copy,
{
    fn from(value: StructuredTensor<T>) -> Self {
        Self::new(T::into_dyn(value))
    }
}

impl From<DynTensor> for Tensor {
    fn from(value: DynTensor) -> Self {
        Self::new(value)
    }
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
    U: Scalar + Copy + DynTensorTyped,
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

#[derive(Clone, Copy)]
struct AddOp;

struct AddLinearized;

impl LinearizableOp<DynTensor> for AddOp {
    type Linearized = AddLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> crate::AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_add(inputs[0], inputs[1]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn output_schema(
        &self,
        _inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        _inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> crate::AdResult<Self::Linearized> {
        Ok(AddLinearized)
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for AddLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> crate::AdResult<Vec<Option<DynTensor>>> {
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
    ) -> crate::AdResult<Vec<Option<DynTensor>>> {
        let grad = output_cotangents[0].clone();
        Ok(vec![
            input_grad_mask[0].then(|| grad.clone()).flatten(),
            input_grad_mask[1].then_some(grad).flatten(),
        ])
    }
}

#[derive(Clone, Copy)]
struct ExpOp;

struct ExpLinearized {
    output: DynTensor,
}

impl LinearizableOp<DynTensor> for ExpOp {
    type Linearized = ExpLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> crate::AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_exp(inputs[0]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(
        &self,
        _inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        _inputs: &[&DynTensor],
        outputs: &[DynTensor],
    ) -> crate::AdResult<Self::Linearized> {
        Ok(ExpLinearized {
            output: outputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for ExpLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> crate::AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![match &input_tangents[0] {
            Some(tangent) => Some(dyn_mul(&self.output, tangent).map_err(into_ad_error)?),
            None => None,
        }])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> crate::AdResult<Vec<Option<DynTensor>>> {
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

#[derive(Clone, Copy)]
struct SumOp;

struct SumLinearized {
    input: DynTensor,
}

impl LinearizableOp<DynTensor> for SumOp {
    type Linearized = SumLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> crate::AdResult<Vec<DynTensor>> {
        Ok(vec![dyn_sum_all(inputs[0]).map_err(into_ad_error)?])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(
        &self,
        _inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> crate::AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> crate::AdResult<Self::Linearized> {
        Ok(SumLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizedOp<DynTensor> for SumLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> crate::AdResult<Vec<Option<DynTensor>>> {
        Ok(vec![match &input_tangents[0] {
            Some(tangent) => Some(dyn_sum_all(tangent).map_err(into_ad_error)?),
            None => None,
        }])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> crate::AdResult<Vec<Option<DynTensor>>> {
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
