use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{ComputeDevice, LogicalMemorySpace};
use tenferro_internal_error::Result;
use tenferro_internal_frontend_core::{
    DynTensor, DynTensorTyped, ScalarType, ScalarValue, StructuredTensor,
};
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::expert::{Tape, TrackedValue};

use crate::{AdMode, AdTensor, NodeId};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Erased AD tensor carrier for cross-crate result wiring.
///
/// This is a transitional internal type that lets higher internal crates return
/// AD-aware mixed-dtype results without depending on the dynamic surface crate.
#[derive(Clone)]
pub enum DynAdTensor {
    F32(AdTensor<f32>),
    F64(AdTensor<f64>),
    C32(AdTensor<Complex32>),
    C64(AdTensor<Complex64>),
}

#[derive(Clone, Copy)]
pub enum DynAdTensorRef<'a> {
    F32(&'a AdTensor<f32>),
    F64(&'a AdTensor<f64>),
    C32(&'a AdTensor<Complex32>),
    C64(&'a AdTensor<Complex64>),
}

pub enum DynAdTensorMutRef<'a> {
    F32(&'a mut AdTensor<f32>),
    F64(&'a mut AdTensor<f64>),
    C32(&'a mut AdTensor<Complex32>),
    C64(&'a mut AdTensor<Complex64>),
}

pub trait DynAdTensorTyped: sealed::Sealed + Scalar + DynTensorTyped {
    fn into_dyn_ad(value: AdTensor<Self>) -> DynAdTensor;
}

pub trait DynAdTensorRefTyped: sealed::Sealed + Scalar + DynTensorTyped {
    fn as_dyn_ad_ref(value: &AdTensor<Self>) -> DynAdTensorRef<'_>;
}

pub trait DynAdTensorBorrowTyped: sealed::Sealed + Scalar + DynTensorTyped {
    fn primal_from_dyn_ad_ref(value: DynAdTensorRef<'_>) -> Option<&Tensor<Self>>;
    fn structured_primal_from_dyn_ad_ref(
        value: DynAdTensorRef<'_>,
    ) -> Option<&StructuredTensor<Self>>;
    fn tangent_from_dyn_ad_ref(value: DynAdTensorRef<'_>) -> Option<&Tensor<Self>>;
    fn structured_tangent_from_dyn_ad_ref(
        value: DynAdTensorRef<'_>,
    ) -> Option<&StructuredTensor<Self>>;
    fn reverse_edge_value_from_dyn_ad_ref(
        value: DynAdTensorRef<'_>,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>>;
}

impl DynAdTensor {
    pub fn new_primal<T>(tensor: impl Into<StructuredTensor<T>>) -> Self
    where
        T: DynAdTensorTyped,
    {
        AdTensor::new_primal(tensor).into()
    }

    pub fn new_forward<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: DynAdTensorTyped,
    {
        Ok(AdTensor::new_forward(primal, tangent)?.into())
    }

    pub fn new_reverse_leaf<T>(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: DynAdTensorTyped,
    {
        Ok(AdTensor::new_reverse_leaf(primal, tape)?.into())
    }

    pub fn new_reverse_leaf_with_tangent<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: DynAdTensorTyped,
    {
        Ok(AdTensor::new_reverse_leaf_with_tangent(primal, tangent, tape)?.into())
    }

    pub fn new_reverse_output<T>(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: DynAdTensorTyped,
    {
        Ok(AdTensor::new_reverse_output(primal, tape, tangent)?.into())
    }

    pub fn new_reverse_output_from_edge<T>(
        primal: impl Into<StructuredTensor<T>>,
        edge_value: std::sync::Arc<tidu::Value<StructuredTensor<T>>>,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: DynAdTensorTyped,
    {
        Ok(AdTensor::new_reverse_output_from_edge(primal, edge_value, tangent)?.into())
    }

    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(value) => value.mode(),
            Self::F64(value) => value.mode(),
            Self::C32(value) => value.mode(),
            Self::C64(value) => value.mode(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(value) => value.dims(),
            Self::F64(value) => value.dims(),
            Self::C32(value) => value.dims(),
            Self::C64(value) => value.dims(),
        }
    }

    pub fn axis_classes(&self) -> &[usize] {
        match self {
            Self::F32(value) => value.axis_classes(),
            Self::F64(value) => value.axis_classes(),
            Self::C32(value) => value.axis_classes(),
            Self::C64(value) => value.axis_classes(),
        }
    }

    pub fn is_dense(&self) -> bool {
        self.as_ref().is_dense()
    }

    pub fn is_diag(&self) -> bool {
        self.as_ref().is_diag()
    }

    pub fn memory_space(&self) -> LogicalMemorySpace {
        self.as_ref().memory_space()
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.as_ref().preferred_compute_device()
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        self.as_ref().to_memory_space_async(target)
    }

    pub fn wait(&self) {
        self.as_ref().wait();
    }

    pub fn is_ready(&self) -> bool {
        self.as_ref().is_ready()
    }

    pub fn has_tangent(&self) -> bool {
        match self {
            Self::F32(value) => value.tangent().is_some(),
            Self::F64(value) => value.tangent().is_some(),
            Self::C32(value) => value.tangent().is_some(),
            Self::C64(value) => value.tangent().is_some(),
        }
    }

    pub fn requires_grad(&self) -> bool {
        match self {
            Self::F32(value) => value.requires_grad(),
            Self::F64(value) => value.requires_grad(),
            Self::C32(value) => value.requires_grad(),
            Self::C64(value) => value.requires_grad(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            Self::F32(value) => value.is_leaf(),
            Self::F64(value) => value.is_leaf(),
            Self::C32(value) => value.is_leaf(),
            Self::C64(value) => value.is_leaf(),
        }
    }

    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::F32(value) => value.node_id(),
            Self::F64(value) => value.node_id(),
            Self::C32(value) => value.node_id(),
            Self::C64(value) => value.node_id(),
        }
    }

    pub fn tape(&self) -> Option<Tape<DynTensor>> {
        self.as_ref().reverse_tape()
    }

    pub fn reverse_handle(&self) -> Option<(NodeId, Tape<DynTensor>)> {
        self.as_ref().reverse_handle()
    }

    pub fn as_tracked(&self) -> Option<TrackedValue<DynTensor>> {
        self.as_ref().as_tracked()
    }

    pub fn primal_snapshot(&self) -> DynTensor {
        self.as_ref().primal_snapshot()
    }

    pub fn tangent_snapshot(&self) -> Option<DynTensor> {
        self.as_ref().tangent_snapshot()
    }

    pub fn try_scalar_value(&self) -> Result<ScalarValue> {
        self.as_ref().try_scalar_value()
    }

    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        self.as_mut().set_preferred_compute_device(device);
    }

    pub fn as_f32(&self) -> Option<&AdTensor<f32>> {
        match self {
            Self::F32(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&AdTensor<f64>> {
        match self {
            Self::F64(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_c32(&self) -> Option<&AdTensor<Complex32>> {
        match self {
            Self::C32(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_c64(&self) -> Option<&AdTensor<Complex64>> {
        match self {
            Self::C64(value) => Some(value),
            _ => None,
        }
    }

    pub fn with_requires_grad(&self, enabled: bool) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(value.with_requires_grad(enabled)?.into()),
            Self::F64(value) => Ok(value.with_requires_grad(enabled)?.into()),
            Self::C32(value) => Ok(value.with_requires_grad(enabled)?.into()),
            Self::C64(value) => Ok(value.with_requires_grad(enabled)?.into()),
        }
    }

    pub fn grad(&self) -> Option<Self> {
        match self {
            Self::F32(value) => value.grad().map(Self::new_primal),
            Self::F64(value) => value.grad().map(Self::new_primal),
            Self::C32(value) => value.grad().map(Self::new_primal),
            Self::C64(value) => value.grad().map(Self::new_primal),
        }
    }

    pub fn zero_grad(&self) -> Result<()> {
        match self {
            Self::F32(value) => value.zero_grad(),
            Self::F64(value) => value.zero_grad(),
            Self::C32(value) => value.zero_grad(),
            Self::C64(value) => value.zero_grad(),
        }
    }

    pub fn accumulate_input_grad_from(&self, grad: &Self) -> Result<()> {
        match (self, grad) {
            (Self::F32(value), Self::F32(grad)) => {
                value.accumulate_input_grad(grad.structured_primal().clone())
            }
            (Self::F64(value), Self::F64(grad)) => {
                value.accumulate_input_grad(grad.structured_primal().clone())
            }
            (Self::C32(value), Self::C32(grad)) => {
                value.accumulate_input_grad(grad.structured_primal().clone())
            }
            (Self::C64(value), Self::C64(grad)) => {
                value.accumulate_input_grad(grad.structured_primal().clone())
            }
            _ => Err(tenferro_internal_error::Error::InvalidAdTensor {
                message: format!(
                    "gradient dtype {:?} does not match input dtype {:?}",
                    grad.scalar_type(),
                    self.scalar_type()
                ),
            }),
        }
    }

    pub fn primal_as<T>(&self) -> Option<&Tensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        self.as_ref().primal_as::<T>()
    }

    pub fn structured_primal_as<T>(&self) -> Option<&StructuredTensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        self.as_ref().structured_primal_as::<T>()
    }

    pub fn tangent_as<T>(&self) -> Option<&Tensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        self.as_ref().tangent_as::<T>()
    }

    pub fn structured_tangent_as<T>(&self) -> Option<&StructuredTensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        self.as_ref().structured_tangent_as::<T>()
    }

    pub fn reverse_edge_value_as<T>(
        &self,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<T>>>>
    where
        T: DynAdTensorBorrowTyped,
    {
        self.as_ref().reverse_edge_value_as::<T>()
    }

    fn as_ref(&self) -> DynAdTensorRef<'_> {
        DynAdTensorRef::from(self)
    }

    fn as_mut(&mut self) -> DynAdTensorMutRef<'_> {
        DynAdTensorMutRef::from(self)
    }
}

impl<'a> DynAdTensorRef<'a> {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(value) => value.mode(),
            Self::F64(value) => value.mode(),
            Self::C32(value) => value.mode(),
            Self::C64(value) => value.mode(),
        }
    }

    pub fn dims(self) -> &'a [usize] {
        match self {
            Self::F32(value) => value.dims(),
            Self::F64(value) => value.dims(),
            Self::C32(value) => value.dims(),
            Self::C64(value) => value.dims(),
        }
    }

    pub fn primal_snapshot(&self) -> DynTensor {
        match self {
            Self::F32(value) => value.structured_primal().clone().into(),
            Self::F64(value) => value.structured_primal().clone().into(),
            Self::C32(value) => value.structured_primal().clone().into(),
            Self::C64(value) => value.structured_primal().clone().into(),
        }
    }

    pub fn tangent_snapshot(&self) -> Option<DynTensor> {
        match self {
            Self::F32(value) => value.structured_tangent().cloned().map(Into::into),
            Self::F64(value) => value.structured_tangent().cloned().map(Into::into),
            Self::C32(value) => value.structured_tangent().cloned().map(Into::into),
            Self::C64(value) => value.structured_tangent().cloned().map(Into::into),
        }
    }

    pub fn try_scalar_value(&self) -> Result<ScalarValue> {
        if !self.dims().is_empty() {
            return Err(tenferro_internal_error::Error::InvalidAdTensor {
                message: format!(
                    "try_scalar_value requires rank-0 tensor, got dims {:?}",
                    self.dims()
                ),
            });
        }

        match self {
            Self::F32(value) => Ok(ScalarValue::F32(read_rank0_scalar(
                value.primal(),
                "DynAdTensorRef::try_scalar_value",
            )?)),
            Self::F64(value) => Ok(ScalarValue::F64(read_rank0_scalar(
                value.primal(),
                "DynAdTensorRef::try_scalar_value",
            )?)),
            Self::C32(value) => Ok(ScalarValue::C32(read_rank0_scalar(
                value.primal(),
                "DynAdTensorRef::try_scalar_value",
            )?)),
            Self::C64(value) => Ok(ScalarValue::C64(read_rank0_scalar(
                value.primal(),
                "DynAdTensorRef::try_scalar_value",
            )?)),
        }
    }

    pub fn requires_grad(&self) -> bool {
        match self {
            Self::F32(value) => value.requires_grad(),
            Self::F64(value) => value.requires_grad(),
            Self::C32(value) => value.requires_grad(),
            Self::C64(value) => value.requires_grad(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            Self::F32(value) => value.is_leaf(),
            Self::F64(value) => value.is_leaf(),
            Self::C32(value) => value.is_leaf(),
            Self::C64(value) => value.is_leaf(),
        }
    }

    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::F32(value) => value.node_id(),
            Self::F64(value) => value.node_id(),
            Self::C32(value) => value.node_id(),
            Self::C64(value) => value.node_id(),
        }
    }

    pub fn reverse_tape(&self) -> Option<Tape<DynTensor>> {
        match self {
            Self::F32(value) => value.reverse_tape(),
            Self::F64(value) => value.reverse_tape(),
            Self::C32(value) => value.reverse_tape(),
            Self::C64(value) => value.reverse_tape(),
        }
    }

    pub fn reverse_handle(&self) -> Option<(NodeId, Tape<DynTensor>)> {
        match self {
            Self::F32(value) => value.reverse_handle(),
            Self::F64(value) => value.reverse_handle(),
            Self::C32(value) => value.reverse_handle(),
            Self::C64(value) => value.reverse_handle(),
        }
    }

    pub fn as_tracked(&self) -> Option<TrackedValue<DynTensor>> {
        match self {
            Self::F32(value) => value.as_tracked(),
            Self::F64(value) => value.as_tracked(),
            Self::C32(value) => value.as_tracked(),
            Self::C64(value) => value.as_tracked(),
        }
    }

    pub fn ensure_reverse_leaf_on(&self, tape: &Tape<DynTensor>) -> Result<()> {
        match self {
            Self::F32(value) => value.ensure_reverse_leaf_on(tape),
            Self::F64(value) => value.ensure_reverse_leaf_on(tape),
            Self::C32(value) => value.ensure_reverse_leaf_on(tape),
            Self::C64(value) => value.ensure_reverse_leaf_on(tape),
        }
    }

    pub fn primal_as<T>(self) -> Option<&'a Tensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        T::primal_from_dyn_ad_ref(self)
    }

    pub fn structured_primal_as<T>(self) -> Option<&'a StructuredTensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        T::structured_primal_from_dyn_ad_ref(self)
    }

    pub fn tangent_as<T>(self) -> Option<&'a Tensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        T::tangent_from_dyn_ad_ref(self)
    }

    pub fn structured_tangent_as<T>(self) -> Option<&'a StructuredTensor<T>>
    where
        T: DynAdTensorBorrowTyped,
    {
        T::structured_tangent_from_dyn_ad_ref(self)
    }

    pub fn reverse_edge_value_as<T>(
        self,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<T>>>>
    where
        T: DynAdTensorBorrowTyped,
    {
        T::reverse_edge_value_from_dyn_ad_ref(self)
    }

    pub fn as_f32(&self) -> Option<&'a AdTensor<f32>> {
        match self {
            Self::F32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&'a AdTensor<f64>> {
        match self {
            Self::F64(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_c32(&self) -> Option<&'a AdTensor<Complex32>> {
        match self {
            Self::C32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_c64(&self) -> Option<&'a AdTensor<Complex64>> {
        match self {
            Self::C64(value) => Some(*value),
            _ => None,
        }
    }

    pub fn axis_classes(self) -> &'a [usize] {
        match self {
            Self::F32(value) => value.axis_classes(),
            Self::F64(value) => value.axis_classes(),
            Self::C32(value) => value.axis_classes(),
            Self::C64(value) => value.axis_classes(),
        }
    }

    pub fn is_dense(&self) -> bool {
        match self {
            Self::F32(value) => value.is_dense(),
            Self::F64(value) => value.is_dense(),
            Self::C32(value) => value.is_dense(),
            Self::C64(value) => value.is_dense(),
        }
    }

    pub fn is_diag(&self) -> bool {
        match self {
            Self::F32(value) => value.is_diag(),
            Self::F64(value) => value.is_diag(),
            Self::C32(value) => value.is_diag(),
            Self::C64(value) => value.is_diag(),
        }
    }

    pub fn memory_space(&self) -> LogicalMemorySpace {
        match self {
            Self::F32(value) => value.memory_space(),
            Self::F64(value) => value.memory_space(),
            Self::C32(value) => value.memory_space(),
            Self::C64(value) => value.memory_space(),
        }
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        match self {
            Self::F32(value) => value.preferred_compute_device(),
            Self::F64(value) => value.preferred_compute_device(),
            Self::C32(value) => value.preferred_compute_device(),
            Self::C64(value) => value.preferred_compute_device(),
        }
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<DynAdTensor> {
        match self {
            Self::F32(value) => Ok(value.to_memory_space_async(target)?.into()),
            Self::F64(value) => Ok(value.to_memory_space_async(target)?.into()),
            Self::C32(value) => Ok(value.to_memory_space_async(target)?.into()),
            Self::C64(value) => Ok(value.to_memory_space_async(target)?.into()),
        }
    }

    pub fn wait(&self) {
        match self {
            Self::F32(value) => value.wait(),
            Self::F64(value) => value.wait(),
            Self::C32(value) => value.wait(),
            Self::C64(value) => value.wait(),
        }
    }

    pub fn is_ready(&self) -> bool {
        match self {
            Self::F32(value) => value.is_ready(),
            Self::F64(value) => value.is_ready(),
            Self::C32(value) => value.is_ready(),
            Self::C64(value) => value.is_ready(),
        }
    }
}

impl<'a> DynAdTensorMutRef<'a> {
    pub fn set_preferred_compute_device(self, device: Option<ComputeDevice>) {
        match self {
            Self::F32(value) => value.set_preferred_compute_device(device),
            Self::F64(value) => value.set_preferred_compute_device(device),
            Self::C32(value) => value.set_preferred_compute_device(device),
            Self::C64(value) => value.set_preferred_compute_device(device),
        }
    }
}

fn read_rank0_scalar<T>(tensor: &tenferro_tensor::Tensor<T>, op_name: &'static str) -> Result<T>
where
    T: tenferro_algebra::Scalar + Conjugate + Copy,
{
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let is_conjugated = contiguous.is_conjugated();
    let contiguous = if contiguous.logical_memory_space() == LogicalMemorySpace::MainMemory {
        contiguous
    } else {
        contiguous
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .map_err(tenferro_internal_error::Error::from)?
    };
    let offset = usize::try_from(contiguous.offset()).map_err(|_| {
        tenferro_internal_error::Error::InvalidAdTensor {
            message: format!("{op_name} computed negative rank-0 offset"),
        }
    })?;
    contiguous
        .buffer()
        .as_slice()
        .and_then(|values| values.get(offset))
        .copied()
        .map(|value| if is_conjugated { value.conj() } else { value })
        .ok_or_else(|| tenferro_internal_error::Error::InvalidAdTensor {
            message: format!("{op_name} could not materialize rank-0 tensor on host memory"),
        })
}

macro_rules! impl_from_ad_tensor {
    ($variant:ident, $ty:ty) => {
        impl DynAdTensorTyped for $ty {
            fn into_dyn_ad(value: AdTensor<Self>) -> DynAdTensor {
                DynAdTensor::$variant(value)
            }
        }

        impl DynAdTensorRefTyped for $ty {
            fn as_dyn_ad_ref(value: &AdTensor<Self>) -> DynAdTensorRef<'_> {
                DynAdTensorRef::$variant(value)
            }
        }

        impl DynAdTensorBorrowTyped for $ty {
            fn primal_from_dyn_ad_ref(value: DynAdTensorRef<'_>) -> Option<&Tensor<Self>> {
                match value {
                    DynAdTensorRef::$variant(value) => Some(value.primal()),
                    _ => None,
                }
            }

            fn structured_primal_from_dyn_ad_ref(
                value: DynAdTensorRef<'_>,
            ) -> Option<&StructuredTensor<Self>> {
                match value {
                    DynAdTensorRef::$variant(value) => Some(value.structured_primal()),
                    _ => None,
                }
            }

            fn tangent_from_dyn_ad_ref(value: DynAdTensorRef<'_>) -> Option<&Tensor<Self>> {
                match value {
                    DynAdTensorRef::$variant(value) => value.tangent(),
                    _ => None,
                }
            }

            fn structured_tangent_from_dyn_ad_ref(
                value: DynAdTensorRef<'_>,
            ) -> Option<&StructuredTensor<Self>> {
                match value {
                    DynAdTensorRef::$variant(value) => value.structured_tangent(),
                    _ => None,
                }
            }

            fn reverse_edge_value_from_dyn_ad_ref(
                value: DynAdTensorRef<'_>,
            ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>> {
                match value {
                    DynAdTensorRef::$variant(value) => value.reverse_edge_value(),
                    _ => None,
                }
            }
        }
    };
}

impl_from_ad_tensor!(F32, f32);
impl_from_ad_tensor!(F64, f64);
impl_from_ad_tensor!(C32, Complex32);
impl_from_ad_tensor!(C64, Complex64);

impl<T> From<AdTensor<T>> for DynAdTensor
where
    T: DynAdTensorTyped,
{
    fn from(value: AdTensor<T>) -> Self {
        T::into_dyn_ad(value)
    }
}

impl<'a, T> From<&'a AdTensor<T>> for DynAdTensorRef<'a>
where
    T: DynAdTensorRefTyped,
{
    fn from(value: &'a AdTensor<T>) -> Self {
        T::as_dyn_ad_ref(value)
    }
}

impl<'a> From<&'a mut AdTensor<f32>> for DynAdTensorMutRef<'a> {
    fn from(value: &'a mut AdTensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl<'a> From<&'a mut AdTensor<f64>> for DynAdTensorMutRef<'a> {
    fn from(value: &'a mut AdTensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl<'a> From<&'a mut AdTensor<Complex32>> for DynAdTensorMutRef<'a> {
    fn from(value: &'a mut AdTensor<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl<'a> From<&'a mut AdTensor<Complex64>> for DynAdTensorMutRef<'a> {
    fn from(value: &'a mut AdTensor<Complex64>) -> Self {
        Self::C64(value)
    }
}

impl<'a> From<&'a DynAdTensor> for DynAdTensorRef<'a> {
    fn from(value: &'a DynAdTensor) -> Self {
        match value {
            DynAdTensor::F32(inner) => Self::F32(inner),
            DynAdTensor::F64(inner) => Self::F64(inner),
            DynAdTensor::C32(inner) => Self::C32(inner),
            DynAdTensor::C64(inner) => Self::C64(inner),
        }
    }
}

impl<'a> From<&'a mut DynAdTensor> for DynAdTensorMutRef<'a> {
    fn from(value: &'a mut DynAdTensor) -> Self {
        match value {
            DynAdTensor::F32(inner) => Self::F32(inner),
            DynAdTensor::F64(inner) => Self::F64(inner),
            DynAdTensor::C32(inner) => Self::C32(inner),
            DynAdTensor::C64(inner) => Self::C64(inner),
        }
    }
}
