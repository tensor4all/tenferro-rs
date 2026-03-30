use super::Tensor;
use crate::{snapshot, ScalarValue};

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
    /// ```ignore
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let detached = x.detach();
    /// assert_eq!(detached.scalar_type(), x.scalar_type());
    /// assert_eq!(detached.scalar_type(), x.scalar_type());
    /// assert!(detached.is_dense());
    /// ```
    pub fn detach(&self) -> Tensor {
        Tensor::from(self.as_dyn_ad_ref().primal_snapshot())
    }

    /// Returns a primal-only snapshot suitable for export, storage, or FFI.
    ///
    /// Unlike [`Tensor::detach`], this returns a dedicated snapshot type rather
    /// than another compute tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{snapshot, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// let snapshot = x.primal_snapshot();
    /// assert!(matches!(snapshot, snapshot::DynTensor::F64(_)));
    /// ```
    pub fn primal_snapshot(&self) -> snapshot::DynTensor {
        self.as_dyn_ad_ref().primal_snapshot()
    }

    pub(crate) fn tangent_snapshot(&self) -> Option<snapshot::DynTensor> {
        self.as_dyn_ad_ref().tangent_snapshot()
    }

    /// Extracts the scalar value of a rank-0 tensor without casting.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{ScalarValue, Tensor};
    ///
    /// let x = Tensor::from_slice(&[3.0_f64], &[]).unwrap();
    /// assert_eq!(x.try_scalar_value().unwrap(), ScalarValue::F64(3.0));
    /// ```
    pub fn try_scalar_value(&self) -> crate::Result<ScalarValue> {
        self.as_dyn_ad_ref().try_scalar_value()
    }
}
