use std::ops::{Deref, DerefMut};

use chainrules_core::Differentiable;
use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{ComputeDevice, LogicalMemorySpace};
use tenferro_internal_error::Result;
use tenferro_tensor::Tensor;

/// AD-capable structured tensor wrapper shared by dynamic tenferro frontends.
///
/// This newtype keeps `Differentiable` and placement helpers on top of
/// [`tenferro_tensor::StructuredTensor<T>`].
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_frontend_core::StructuredTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let wrapped = StructuredTensor::from(payload);
/// assert!(wrapped.is_dense());
/// ```
#[derive(Debug, Clone)]
pub struct StructuredTensor<T: Scalar>(pub tenferro_tensor::StructuredTensor<T>);

impl<T: Scalar> StructuredTensor<T> {
    pub fn with_payload_like(&self, payload: Tensor<T>) -> Result<Self> {
        Ok(Self(self.0.with_payload_like(payload)?))
    }

    pub fn into_payload(self) -> Tensor<T> {
        self.0.into_payload()
    }

    pub fn permute_logical(&self, perm: &[usize]) -> Result<Self> {
        Ok(Self(self.0.permute_logical(perm)?))
    }

    pub fn conj(&self) -> Self
    where
        T: Conjugate,
    {
        Self(self.0.conj())
    }

    pub fn to_dense(&self) -> Result<Tensor<T>> {
        Ok(self.0.to_dense()?)
    }

    pub fn memory_space(&self) -> LogicalMemorySpace {
        self.payload().logical_memory_space()
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.payload().preferred_compute_device()
    }

    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        let mut payload = self.payload().clone();
        payload.set_preferred_compute_device(device);
        *self = Self(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            payload,
        ));
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        let payload = self.payload().to_memory_space_async(target)?;
        Ok(Self(self.0.with_payload_like(payload)?))
    }

    pub fn wait(&self) {
        self.payload().wait();
    }

    pub fn is_ready(&self) -> bool {
        self.payload().is_ready()
    }
}

impl<T> Differentiable for StructuredTensor<T>
where
    T: Scalar,
{
    type Tangent = StructuredTensor<T>;

    fn zero_tangent(&self) -> Self::Tangent {
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            self.payload().zero_tangent(),
        ))
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        assert_eq!(
            a.logical_dims(),
            b.logical_dims(),
            "StructuredTensor::accumulate_tangent requires matching logical dims"
        );
        assert_eq!(
            a.axis_classes(),
            b.axis_classes(),
            "StructuredTensor::accumulate_tangent requires matching axis classes"
        );
        let logical_dims = a.logical_dims().to_vec();
        let axis_classes = a.axis_classes().to_vec();
        let payload = Tensor::<T>::accumulate_tangent(a.0.into_payload(), b.payload());
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            logical_dims,
            axis_classes,
            payload,
        ))
    }

    fn num_elements(&self) -> usize {
        self.logical_dims().iter().product()
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        StructuredTensor(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            self.payload().seed_cotangent(),
        ))
    }
}

impl<T: Scalar> Deref for StructuredTensor<T> {
    type Target = tenferro_tensor::StructuredTensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Scalar> DerefMut for StructuredTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Scalar> From<tenferro_tensor::StructuredTensor<T>> for StructuredTensor<T> {
    fn from(inner: tenferro_tensor::StructuredTensor<T>) -> Self {
        Self(inner)
    }
}

impl<T: Scalar> From<Tensor<T>> for StructuredTensor<T> {
    fn from(tensor: Tensor<T>) -> Self {
        Self(tenferro_tensor::StructuredTensor::from_dense(tensor))
    }
}

impl<T: Scalar> AsRef<Tensor<T>> for StructuredTensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self.0.payload()
    }
}
