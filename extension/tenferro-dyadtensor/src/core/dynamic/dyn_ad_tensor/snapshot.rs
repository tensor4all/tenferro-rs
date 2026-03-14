use super::DynAdTensor;
use crate::{DynTensor, Result};

impl DynAdTensor {
    /// Returns a primal-only structured snapshot while intentionally dropping
    /// AD metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let snapshot = x.primal_snapshot().unwrap();
    /// assert_eq!(snapshot.scalar_type(), x.scalar_type());
    /// assert!(snapshot.is_dense());
    /// ```
    pub fn primal_snapshot(&self) -> Result<DynTensor> {
        Ok(match self {
            Self::F32(value) => DynTensor::F32(value.structured_primal().clone()),
            Self::F64(value) => DynTensor::F64(value.structured_primal().clone()),
            Self::C32(value) => DynTensor::C32(value.structured_primal().clone()),
            Self::C64(value) => DynTensor::C64(value.structured_primal().clone()),
        })
    }
}
