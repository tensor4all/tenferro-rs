use std::ops::{Deref, DerefMut};

use tenferro_algebra::{Conjugate, Scalar};
use tenferro_tensor::Tensor;

use crate::Result;

/// AD-aware structured tensor in the `tenferro` umbrella crate.
///
/// This newtype keeps `tenferro`-specific runtime and autodiff impls on top of
/// [`tenferro_tensor::StructuredTensor<T>`] without violating Rust's orphan
/// rules. Base structured-tensor operations live in `tenferro-tensor`; wrapper
/// code constructs this type as `StructuredTensor(inner)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{MemoryOrder, StructuredTensor as InnerStructuredTensor, Tensor};
///
/// let payload =
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let inner = InnerStructuredTensor::from_diagonal_vector(payload, 2).unwrap();
/// let wrapped = crate::structured::StructuredTensor(inner);
/// assert!(wrapped.is_diag());
/// ```
#[derive(Debug, Clone)]
pub struct StructuredTensor<T: Scalar>(pub tenferro_tensor::StructuredTensor<T>);

impl<T: Scalar> StructuredTensor<T> {
    pub(crate) fn with_payload_like(&self, payload: Tensor<T>) -> Result<Self> {
        Ok(Self(self.0.with_payload_like(payload)?))
    }

    pub(crate) fn into_payload(self) -> Tensor<T> {
        self.0.into_payload()
    }

    pub(crate) fn permute_logical(&self, perm: &[usize]) -> Result<Self> {
        Ok(Self(self.0.permute_logical(perm)?))
    }

    pub(crate) fn conj(&self) -> Self
    where
        T: Conjugate,
    {
        Self(self.0.conj())
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

#[cfg(test)]
mod tests;
