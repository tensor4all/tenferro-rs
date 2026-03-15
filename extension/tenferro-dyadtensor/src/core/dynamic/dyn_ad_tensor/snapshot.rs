use super::Tensor;
use crate::AdTensor;

impl Tensor {
    /// Returns a detached primal tensor while intentionally dropping AD
    /// metadata.
    ///
    /// This is the PyTorch-like public boundary for storage, FFI, or any
    /// downstream code that wants the same logical tensor object without tape
    /// connectivity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let detached = x.detach();
    /// assert!(!detached.requires_grad());
    /// assert_eq!(detached.scalar_type(), x.scalar_type());
    /// assert!(detached.is_dense());
    /// ```
    pub fn detach(&self) -> Self {
        match self {
            Self::F32(value) => Self::F32(AdTensor::new_primal(value.structured_primal().clone())),
            Self::F64(value) => Self::F64(AdTensor::new_primal(value.structured_primal().clone())),
            Self::C32(value) => Self::C32(AdTensor::new_primal(value.structured_primal().clone())),
            Self::C64(value) => Self::C64(AdTensor::new_primal(value.structured_primal().clone())),
        }
    }
}
