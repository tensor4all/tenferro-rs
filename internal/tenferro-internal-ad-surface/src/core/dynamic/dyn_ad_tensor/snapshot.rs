use super::Tensor;
use crate::{snapshot, ScalarValue};
use tenferro_algebra::Conjugate;
use tenferro_device::LogicalMemorySpace;
use tenferro_dynamic_compute as dynamic_compute;
use tenferro_tensor::MemoryOrder;

fn read_rank0_scalar<T>(
    tensor: &tenferro_tensor::Tensor<T>,
    op_name: &'static str,
) -> crate::Result<T>
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
            .map_err(crate::Error::from)?
    };
    let offset =
        usize::try_from(contiguous.offset()).map_err(|_| crate::Error::InvalidAdTensor {
            message: format!("{op_name} computed negative rank-0 offset"),
        })?;
    contiguous
        .buffer()
        .as_slice()
        .and_then(|values| values.get(offset))
        .copied()
        .map(|value| if is_conjugated { value.conj() } else { value })
        .ok_or_else(|| crate::Error::InvalidAdTensor {
            message: format!("{op_name} could not materialize rank-0 tensor on host memory"),
        })
}

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
    /// let detached: tenferro_dynamic_compute::Tensor = x.detach();
    /// assert_eq!(detached.scalar_type(), x.scalar_type());
    /// assert_eq!(detached.scalar_type(), x.scalar_type());
    /// assert!(detached.is_dense());
    /// ```
    pub fn detach(&self) -> dynamic_compute::Tensor {
        match self {
            Self::F32(value) => dynamic_compute::Tensor::F32(value.structured_primal().clone()),
            Self::F64(value) => dynamic_compute::Tensor::F64(value.structured_primal().clone()),
            Self::C32(value) => dynamic_compute::Tensor::C32(value.structured_primal().clone()),
            Self::C64(value) => dynamic_compute::Tensor::C64(value.structured_primal().clone()),
        }
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
        match self {
            Self::F32(value) => snapshot::DynTensor::F32(value.structured_primal().clone()),
            Self::F64(value) => snapshot::DynTensor::F64(value.structured_primal().clone()),
            Self::C32(value) => snapshot::DynTensor::C32(value.structured_primal().clone()),
            Self::C64(value) => snapshot::DynTensor::C64(value.structured_primal().clone()),
        }
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
        if self.ndim() != 0 {
            return Err(crate::Error::InvalidAdTensor {
                message: format!(
                    "try_scalar_value requires rank-0 tensor, got dims {:?}",
                    self.dims()
                ),
            });
        }

        match self {
            Self::F32(value) => Ok(ScalarValue::F32(read_rank0_scalar(
                value.primal(),
                "Tensor::try_scalar_value",
            )?)),
            Self::F64(value) => Ok(ScalarValue::F64(read_rank0_scalar(
                value.primal(),
                "Tensor::try_scalar_value",
            )?)),
            Self::C32(value) => Ok(ScalarValue::C32(read_rank0_scalar(
                value.primal(),
                "Tensor::try_scalar_value",
            )?)),
            Self::C64(value) => Ok(ScalarValue::C64(read_rank0_scalar(
                value.primal(),
                "Tensor::try_scalar_value",
            )?)),
        }
    }
}
