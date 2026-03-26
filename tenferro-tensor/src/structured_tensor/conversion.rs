use tenferro_algebra::Scalar;
use tenferro_device::{checked_batch_count, unflatten_col_major_index_into, Error, Result};

use super::StructuredTensor;
use crate::{MemoryOrder, Tensor};

impl<T: Scalar> StructuredTensor<T> {
    /// Materialize this structured tensor into a dense tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// let dense = x.to_dense().unwrap();
    /// assert_eq!(dense.dims(), &[2, 2]);
    /// assert_eq!(dense.get(&[0, 0]), Some(&1.0));
    /// assert_eq!(dense.get(&[0, 1]), Some(&0.0));
    /// ```
    pub fn to_dense(&self) -> Result<Tensor<T>> {
        if self.is_dense() {
            return Ok(self.payload.clone());
        }

        let mut dense = Tensor::zeros(
            &self.logical_dims,
            self.payload.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )?;

        let payload_dims = self.payload.dims();
        let payload_numel = checked_batch_count(payload_dims)?;
        let mut payload_index = vec![0usize; payload_dims.len()];
        let mut logical_index = vec![0usize; self.logical_dims.len()];

        for linear in 0..payload_numel {
            if !payload_dims.is_empty() {
                unflatten_col_major_index_into(linear, payload_dims, &mut payload_index)?;
            }

            for (logical_axis, &class_id) in self.axis_classes.iter().enumerate() {
                logical_index[logical_axis] = payload_index[class_id];
            }

            let value = self.payload.get(&payload_index).copied().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "to_dense: could not read payload at index {payload_index:?}"
                ))
            })?;
            dense.set(&logical_index, value)?;
        }

        Ok(dense)
    }

    /// Consume this structured tensor and return a dense tensor.
    ///
    /// Returns the payload directly when the layout is already dense.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    /// let dense = x.into_dense().unwrap();
    /// assert_eq!(dense.dims(), &[2, 2]);
    /// ```
    pub fn into_dense(self) -> Result<Tensor<T>> {
        if self.is_dense() {
            Ok(self.payload)
        } else {
            self.to_dense()
        }
    }
}
