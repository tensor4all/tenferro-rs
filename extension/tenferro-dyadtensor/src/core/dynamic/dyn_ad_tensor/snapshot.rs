use num_complex::{Complex32, Complex64};

use super::DynAdTensor;
use crate::{Result, ScalarType, StructuredTensor};

/// Dynamic primal-only snapshot of a [`DynAdTensor`].
///
/// This boundary preserves structured primal information and intentionally drops
/// all AD metadata. It is the intended export/storage/FFI boundary for dynamic
/// AD tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdTensor, DynAdTensor, DynStructuredPrimal};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let x: DynAdTensor = AdTensor::new_primal(
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
/// )
/// .into();
///
/// match x.primal_snapshot().unwrap() {
///     DynStructuredPrimal::F64(snapshot) => assert_eq!(snapshot.logical_dims(), &[2]),
///     _ => unreachable!("dtype should stay f64"),
/// }
/// ```
#[derive(Debug, Clone)]
pub enum DynStructuredPrimal {
    F32(StructuredTensor<f32>),
    F64(StructuredTensor<f64>),
    C32(StructuredTensor<Complex32>),
    C64(StructuredTensor<Complex64>),
}

impl DynStructuredPrimal {
    /// Returns the runtime scalar type tag for the snapshot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    ///
    /// assert_eq!(x.primal_snapshot().unwrap().scalar_type(), ScalarType::F64);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }
}

impl DynAdTensor {
    /// Returns a primal-only structured snapshot while intentionally dropping
    /// AD metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    ///
    /// let snapshot = x.primal_snapshot().unwrap();
    /// assert_eq!(snapshot.scalar_type(), x.scalar_type());
    /// ```
    pub fn primal_snapshot(&self) -> Result<DynStructuredPrimal> {
        Ok(match self {
            Self::F32(value) => DynStructuredPrimal::F32(value.structured_primal().clone()),
            Self::F64(value) => DynStructuredPrimal::F64(value.structured_primal().clone()),
            Self::C32(value) => DynStructuredPrimal::C32(value.structured_primal().clone()),
            Self::C64(value) => DynStructuredPrimal::C64(value.structured_primal().clone()),
        })
    }
}
