use chainrules_core::AutodiffError;
use std::fmt;
use tenferro_internal_ad_linalg::{
    det_dyn_value, norm_dyn_value, qr_dyn_value, solve_dyn_values, svd_dyn_value,
};
use tenferro_internal_ad_ops::{add_dyn_values, einsum_dyn_values, exp_dyn_value, sum_dyn_value};
use tenferro_internal_frontend_core::tensor_ops::tensor_element;
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, ScalarType, StructuredTensor};
use tenferro_linalg::{NormKind, SvdOptions};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use super::{QrResult, SvdResult};
use crate::{Error, Result};

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
        Ok(Self::from_value(add_dyn_values(self.value(), rhs.value())?))
    }

    pub fn exp(&self) -> Result<Self> {
        Ok(Self::from_value(exp_dyn_value(self.value())?))
    }

    pub fn sum(&self) -> Result<Self> {
        Ok(Self::from_value(sum_dyn_value(self.value())?))
    }

    pub fn einsum(subscripts: &str, operands: &[&Self]) -> Result<Self> {
        let values = operands
            .iter()
            .map(|tensor| tensor.value())
            .collect::<Vec<_>>();
        Ok(Self::from_value(einsum_dyn_values(subscripts, &values)?))
    }

    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        Ok(Self::from_value(solve_dyn_values(
            self.value(),
            rhs.value(),
        )?))
    }

    pub fn det(&self) -> Result<Self> {
        Ok(Self::from_value(det_dyn_value(self.value())?))
    }

    pub fn norm(&self, kind: NormKind) -> Result<Self> {
        Ok(Self::from_value(norm_dyn_value(self.value(), kind)?))
    }

    pub fn qr(&self) -> Result<QrResult> {
        Ok(qr_dyn_value(self.value())?.into())
    }

    pub fn svd(&self, options: Option<SvdOptions>) -> Result<SvdResult> {
        Ok(svd_dyn_value(self.value(), options)?.into())
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

fn invalid_argument(message: impl Into<String>) -> Error {
    AutodiffError::InvalidArgument(message.into()).into()
}
