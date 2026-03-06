use std::collections::HashMap;

use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{Error, Result};

/// Structured tensor payload for root AD tensors.
///
/// This stores logical tensor metadata separately from the compressed payload.
/// Dense and diagonal tensors are representation cases of the same type.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::StructuredTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// assert!(x.is_diag());
    /// ```
    pub fn from_diagonal_vector(payload: Tensor<T>, logical_rank: usize) -> Result<Self> {
        if payload.dims().len() != 1 {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "from_diagonal_vector expects rank-1 payload, got rank {}",
                    payload.dims().len()
                ),
            });
        }
        if logical_rank == 0 {
            return Err(Error::InvalidAdTensor {
                message: "from_diagonal_vector requires logical_rank >= 1".to_string(),
            });
        }
        let n = payload.dims()[0];
        Self::new(vec![n; logical_rank], vec![0; logical_rank], payload)
    }

    /// Borrow the compressed payload tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
    /// ```rust
    /// use tenferro_dyadtensor::StructuredTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
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
}

pub(crate) fn canonicalize_axis_classes(classes: &[usize]) -> Vec<usize> {
    let mut map = HashMap::new();
    let mut next = 0usize;
    classes
        .iter()
        .map(|&class_id| {
            if let Some(&mapped) = map.get(&class_id) {
                mapped
            } else {
                let mapped = next;
                next += 1;
                map.insert(class_id, mapped);
                mapped
            }
        })
        .collect()
}

pub(crate) fn validate_layout<T: Scalar>(
    logical_dims: &[usize],
    axis_classes: &[usize],
    payload: &Tensor<T>,
) -> Result<()> {
    if logical_dims.len() != axis_classes.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "logical_dims length ({}) must match axis_classes length ({})",
                logical_dims.len(),
                axis_classes.len(),
            ),
        });
    }
    if logical_dims.is_empty() && payload.dims().is_empty() {
        return Ok(());
    }

    let class_count = axis_classes
        .iter()
        .copied()
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);
    if payload.dims().len() != class_count {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "payload rank {} must equal number of classes {}",
                payload.dims().len(),
                class_count,
            ),
        });
    }

    let mut class_dims = vec![None; class_count];
    for (&dim, &class_id) in logical_dims.iter().zip(axis_classes.iter()) {
        if let Some(existing) = class_dims[class_id] {
            if existing != dim {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "axis class {class_id} has inconsistent logical dims: {existing} vs {dim}",
                    ),
                });
            }
        } else {
            class_dims[class_id] = Some(dim);
        }
    }

    for (class_id, maybe_dim) in class_dims.iter().enumerate() {
        let expected = maybe_dim.unwrap_or(0);
        let got = payload.dims()[class_id];
        if expected != got {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "payload dim mismatch for class {class_id}: expected {expected}, got {got}",
                ),
            });
        }
    }

    Ok(())
}
