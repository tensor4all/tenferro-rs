use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::core::{AdMode, AdValue, NodeId, TapeId};
use crate::structured::StructuredTensor;
use crate::{Error, Result};

/// Tensor newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: AdTensor<f64> = t.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Debug, Clone)]
pub struct AdTensor<T: Scalar>(AdValue<StructuredTensor<T>>);

fn ensure_same_structured_layout<T: Scalar>(
    op_name: &'static str,
    primal: &StructuredTensor<T>,
    tangent: &StructuredTensor<T>,
) -> Result<()> {
    if primal.logical_dims() != tangent.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires tangent.logical_dims == primal.logical_dims, got primal={:?}, tangent={:?}",
                primal.logical_dims(),
                tangent.logical_dims()
            ),
        });
    }
    if primal.axis_classes() != tangent.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires tangent.axis_classes == primal.axis_classes, got primal={:?}, tangent={:?}",
                primal.axis_classes(),
                tangent.axis_classes()
            ),
        });
    }
    Ok(())
}

fn validate_ad_tensor_value<T: Scalar>(
    op_name: &'static str,
    value: &AdValue<StructuredTensor<T>>,
) -> Result<()> {
    match value {
        AdValue::Primal(_) => Ok(()),
        AdValue::Forward { primal, tangent } => {
            ensure_same_structured_layout(op_name, primal, tangent)
        }
        AdValue::Reverse {
            primal,
            tangent: Some(tangent),
            ..
        } => ensure_same_structured_layout(op_name, primal, tangent),
        AdValue::Reverse { tangent: None, .. } => Ok(()),
    }
}

impl<T: Scalar> AdTensor<T> {
    /// Creates a primal tensor.
    pub fn new_primal(tensor: impl Into<StructuredTensor<T>>) -> Self {
        Self(AdValue::primal(tensor.into()))
    }

    /// Creates a forward-mode tensor.
    pub fn new_forward(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self> {
        let value = AdValue::forward(primal.into(), tangent.into());
        Self::try_from(value)
    }

    /// Creates a reverse-mode tensor.
    pub fn new_reverse(
        primal: impl Into<StructuredTensor<T>>,
        node: NodeId,
        tape: TapeId,
        tangent: Option<StructuredTensor<T>>,
    ) -> Result<Self> {
        let value = AdValue::reverse(primal.into(), node, tape, tangent);
        Self::try_from(value)
    }

    /// Returns AD mode.
    pub fn mode(&self) -> AdMode {
        self.0.mode()
    }

    /// Returns reference to underlying [`AdValue`].
    pub fn as_value(&self) -> &AdValue<StructuredTensor<T>> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    pub fn into_value(self) -> AdValue<StructuredTensor<T>> {
        self.0
    }

    pub(crate) fn from_value_unchecked(value: AdValue<StructuredTensor<T>>) -> Self {
        Self(value)
    }

    /// Returns structured primal payload reference.
    pub fn structured_primal(&self) -> &StructuredTensor<T> {
        self.0.primal_ref()
    }

    /// Returns compressed primal payload tensor reference.
    pub fn primal(&self) -> &Tensor<T> {
        self.structured_primal().payload()
    }

    /// Returns structured tangent reference when available.
    pub fn structured_tangent(&self) -> Option<&StructuredTensor<T>> {
        self.0.tangent_ref()
    }

    /// Returns compressed tangent payload tensor reference when available.
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.structured_tangent().map(StructuredTensor::payload)
    }

    /// Returns dimensions of the primal tensor.
    pub fn dims(&self) -> &[usize] {
        self.structured_primal().logical_dims()
    }

    /// Returns number of dimensions of the primal tensor.
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns axis classes of the structured primal.
    pub fn axis_classes(&self) -> &[usize] {
        self.structured_primal().axis_classes()
    }

    /// Returns `true` when the structured primal is dense.
    pub fn is_dense(&self) -> bool {
        self.structured_primal().is_dense()
    }

    /// Returns `true` when the structured primal is diagonal.
    pub fn is_diag(&self) -> bool {
        self.structured_primal().is_diag()
    }
}

impl<T: Scalar> From<Tensor<T>> for AdTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self(AdValue::Primal(StructuredTensor::from_dense(value)))
    }
}

impl<T: Scalar> From<StructuredTensor<T>> for AdTensor<T> {
    fn from(value: StructuredTensor<T>) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T: Scalar> TryFrom<AdValue<StructuredTensor<T>>> for AdTensor<T> {
    type Error = Error;

    fn try_from(value: AdValue<StructuredTensor<T>>) -> Result<Self> {
        validate_ad_tensor_value("AdTensor::try_from", &value)?;
        Ok(Self(value))
    }
}

impl<T: Scalar> From<AdValue<Tensor<T>>> for AdTensor<T> {
    fn from(value: AdValue<Tensor<T>>) -> Self {
        let mapped = match value {
            AdValue::Primal(primal) => AdValue::Primal(StructuredTensor::from_dense(primal)),
            AdValue::Forward { primal, tangent } => AdValue::Forward {
                primal: StructuredTensor::from_dense(primal),
                tangent: StructuredTensor::from_dense(tangent),
            },
            AdValue::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => AdValue::Reverse {
                primal: StructuredTensor::from_dense(primal),
                node,
                tape,
                tangent: tangent.map(StructuredTensor::from_dense),
            },
        };
        Self::from_value_unchecked(mapped)
    }
}
