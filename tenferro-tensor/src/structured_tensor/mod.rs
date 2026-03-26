//! Structured tensor metadata layered on top of dense [`Tensor`] payloads.

mod conversion;
mod validation;
mod views;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::Tensor;

pub(crate) use validation::validate_permutation;
pub use validation::{canonicalize_axis_classes, validate_layout};

/// Structured tensor payload with logical axis metadata.
///
/// This stores logical tensor metadata separately from the compressed payload.
/// Dense and diagonal tensors are representation cases of the same type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
///
/// let payload =
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
/// assert_eq!(x.logical_dims(), &[2, 2]);
/// assert!(x.is_diag());
/// ```
#[derive(Debug, Clone)]
pub struct StructuredTensor<T: Scalar> {
    payload: Tensor<T>,
    logical_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}

impl<T: Scalar> StructuredTensor<T> {
    /// Construct a structured tensor from logical metadata and compressed payload.
    ///
    /// Axis classes are canonicalized to first-appearance order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::new(vec![2, 2], vec![9, 9], payload).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn new(
        logical_dims: Vec<usize>,
        axis_classes: Vec<usize>,
        payload: Tensor<T>,
    ) -> Result<Self> {
        let axis_classes = canonicalize_axis_classes(&axis_classes);
        validate_layout(&logical_dims, &axis_classes, &payload)?;
        Ok(Self {
            payload,
            logical_dims,
            axis_classes,
        })
    }

    /// Construct a dense structured tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0],
    ///     &[2, 2],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x = StructuredTensor::from_dense(dense);
    /// assert!(x.is_dense());
    /// ```
    pub fn from_dense(payload: Tensor<T>) -> Self {
        let logical_dims = payload.dims().to_vec();
        let axis_classes = (0..logical_dims.len()).collect();
        Self {
            payload,
            logical_dims,
            axis_classes,
        }
    }

    /// Construct a diagonal structured tensor from a rank-1 payload.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert!(x.is_diag());
    /// ```
    pub fn from_diagonal_vector(payload: Tensor<T>, logical_rank: usize) -> Result<Self> {
        if payload.dims().len() != 1 {
            return Err(Error::InvalidArgument(format!(
                "from_diagonal_vector expects rank-1 payload, got rank {}",
                payload.dims().len()
            )));
        }
        if logical_rank == 0 {
            return Err(Error::InvalidArgument(
                "from_diagonal_vector requires logical_rank >= 1".to_string(),
            ));
        }
        let n = payload.dims()[0];
        Self::new(vec![n; logical_rank], vec![0; logical_rank], payload)
    }

    /// Borrow the compressed payload tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let dense =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_dense(dense);
    /// assert_eq!(x.payload().dims(), &[2]);
    /// ```
    pub fn payload(&self) -> &Tensor<T> {
        &self.payload
    }

    /// Consume the structured tensor and return the compressed payload tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let dense =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_dense(dense);
    /// let payload = x.into_payload();
    /// assert_eq!(payload.dims(), &[2]);
    /// ```
    pub fn into_payload(self) -> Tensor<T> {
        self.payload
    }

    /// Returns logical dimensions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert_eq!(x.logical_dims(), &[2, 2]);
    /// ```
    pub fn logical_dims(&self) -> &[usize] {
        &self.logical_dims
    }

    /// Returns axis classes for logical axes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        &self.axis_classes
    }

    /// Returns the number of distinct axis classes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 3).unwrap();
    /// assert_eq!(x.class_count(), 1);
    /// ```
    pub fn class_count(&self) -> usize {
        self.payload.dims().len()
    }

    /// Returns `true` when the layout is dense.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0],
    ///     &[2, 2],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x = StructuredTensor::from_dense(dense);
    /// assert!(x.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        self.axis_classes.len() == self.logical_dims.len()
            && self.logical_dims == self.payload.dims()
            && self
                .axis_classes
                .iter()
                .enumerate()
                .all(|(i, &class_id)| class_id == i)
    }

    /// Returns `true` when the layout is a pure diagonal.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert!(x.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        if self.logical_dims.is_empty() || self.axis_classes.len() != self.logical_dims.len() {
            return false;
        }
        let first_dim = self.logical_dims[0];
        self.axis_classes.iter().all(|&class_id| class_id == 0)
            && self.logical_dims.iter().all(|&dim| dim == first_dim)
            && self.payload.dims().len() == 1
            && self.payload.dims()[0] == first_dim
    }

    /// Rebuild the same structured layout with a different payload tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let layout = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// let replacement =
    ///     Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let updated = layout.with_payload_like(replacement).unwrap();
    /// assert!(updated.is_diag());
    /// ```
    pub fn with_payload_like(&self, payload: Tensor<T>) -> Result<Self> {
        Self::new(
            self.logical_dims.clone(),
            self.axis_classes.clone(),
            payload,
        )
    }

    /// Construct a structured tensor without re-validating the metadata.
    ///
    /// # Safety
    ///
    /// Callers must ensure that `logical_dims`, `axis_classes`, and `payload`
    /// already satisfy the invariants enforced by [`StructuredTensor::new`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_validated_parts(vec![2, 2], vec![0, 0], payload);
    /// assert!(x.is_diag());
    /// ```
    pub fn from_validated_parts(
        logical_dims: Vec<usize>,
        axis_classes: Vec<usize>,
        payload: Tensor<T>,
    ) -> Self {
        Self {
            payload,
            logical_dims,
            axis_classes,
        }
    }
}

impl<T: Scalar> From<Tensor<T>> for StructuredTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self::from_dense(value)
    }
}

impl<T: Scalar> AsRef<Tensor<T>> for StructuredTensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self.payload()
    }
}

#[cfg(test)]
mod tests;
