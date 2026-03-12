use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use super::Tensor;
use crate::layout::{
    compute_contiguous_strides, is_contiguous_in_order, validate_layout_against_len,
};
use crate::MemoryOrder;

impl<T: Scalar> Tensor<T> {
    /// Permute (reorder) the dimensions of the tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let transposed = t.permute(&[1, 0]).unwrap();
    /// assert_eq!(transposed.dims(), &[3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        if perm.len() != self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "permutation length {} doesn't match ndim {}",
                perm.len(),
                self.ndim()
            )));
        }

        let mut seen = vec![false; self.ndim()];
        for &axis in perm {
            if axis >= self.ndim() {
                return Err(Error::InvalidArgument(format!(
                    "permutation index {axis} out of range for ndim {}",
                    self.ndim()
                )));
            }
            if seen[axis] {
                return Err(Error::InvalidArgument(format!(
                    "duplicate index {axis} in permutation"
                )));
            }
            seen[axis] = true;
        }

        let new_dims: Arc<[usize]> = perm.iter().map(|&axis| self.dims[axis]).collect();
        let new_strides: Arc<[isize]> = perm.iter().map(|&axis| self.strides[axis]).collect();
        Ok(self.shared_view_with(new_dims, new_strides, self.offset))
    }

    /// Broadcast the tensor to a larger shape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[1, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let b = t.broadcast(&[4, 3]).unwrap();
    /// assert_eq!(b.dims(), &[4, 3]);
    /// ```
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        if target_dims.len() != self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "target dims length {} doesn't match ndim {}",
                target_dims.len(),
                self.ndim()
            )));
        }

        let mut new_strides = self.strides.to_vec();
        for (axis, (&current, &target)) in self.dims.iter().zip(target_dims).enumerate() {
            if current == target {
                continue;
            }
            if current == 1 {
                new_strides[axis] = 0;
            } else {
                return Err(Error::ShapeMismatch {
                    expected: self.dims.to_vec(),
                    got: target_dims.to_vec(),
                });
            }
        }

        Ok(self.shared_view_with(Arc::from(target_dims), Arc::from(new_strides), self.offset))
    }

    /// Extract a diagonal view by merging pairs of axes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let d = t.diagonal(&[(0, 1)]).unwrap();
    /// assert_eq!(d.dims(), &[3]);
    /// ```
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>> {
        self.wait();
        let mut used = vec![false; self.ndim()];
        let mut diag_dims = Vec::new();
        let mut diag_strides = Vec::new();

        for &(i, j) in axes {
            if i >= self.ndim() || j >= self.ndim() {
                return Err(Error::InvalidArgument(format!(
                    "axis out of range: ({i}, {j}) for tensor with {} dimensions",
                    self.ndim()
                )));
            }
            if i == j {
                return Err(Error::InvalidArgument(format!(
                    "diagonal axes must be distinct, got ({i}, {j})"
                )));
            }
            if used[i] || used[j] {
                return Err(Error::InvalidArgument(format!(
                    "axis {i} or {j} used in multiple diagonal pairs"
                )));
            }
            if self.dims[i] != self.dims[j] {
                return Err(Error::ShapeMismatch {
                    expected: vec![self.dims[i]],
                    got: vec![self.dims[j]],
                });
            }
            used[i] = true;
            used[j] = true;
            diag_dims.push(self.dims[i]);
            diag_strides.push(self.strides[i] + self.strides[j]);
        }

        let mut new_dims = Vec::new();
        let mut new_strides = Vec::new();
        for (axis, was_used) in used.iter().enumerate() {
            if !was_used {
                new_dims.push(self.dims[axis]);
                new_strides.push(self.strides[axis]);
            }
        }
        new_dims.extend_from_slice(&diag_dims);
        new_strides.extend_from_slice(&diag_strides);

        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), self.offset))
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let r = t.reshape(&[6]).unwrap();
    /// assert_eq!(r.dims(), &[6]);
    /// ```
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        if self.len() != new_dims.iter().product::<usize>() {
            return Err(Error::ShapeMismatch {
                expected: self.dims.to_vec(),
                got: new_dims.to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(Error::StrideError(
                "reshape requires contiguous data".into(),
            ));
        }

        let order = if is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor) {
            MemoryOrder::ColumnMajor
        } else {
            MemoryOrder::RowMajor
        };
        let new_strides = Arc::from(compute_contiguous_strides(new_dims, order));
        Ok(self.shared_view_with(Arc::from(new_dims), new_strides, self.offset))
    }

    /// Create a zero-copy view with explicit dims and strides.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let view = t.view_as_strided(vec![3, 2], vec![2, 1]).unwrap();
    /// assert_eq!(view.dims(), &[3, 2]);
    /// ```
    pub fn view_as_strided(
        &self,
        new_dims: Vec<usize>,
        new_strides: Vec<isize>,
    ) -> Result<Tensor<T>> {
        self.wait();
        validate_layout_against_len(&new_dims, &new_strides, self.offset, self.buffer.len())?;
        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), self.offset))
    }

    /// Select a single index along a dimension, removing that dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let slice = t.select(2, 1).unwrap();
    /// assert_eq!(slice.dims(), &[2, 3]);
    /// ```
    pub fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>> {
        self.wait();
        if dim >= self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "dim {dim} out of range for tensor with {} dimensions",
                self.ndim()
            )));
        }
        if index >= self.dims[dim] {
            return Err(Error::InvalidArgument(format!(
                "index {index} out of range for dimension {dim} with size {}",
                self.dims[dim]
            )));
        }

        let offset = (index as isize)
            .checked_mul(self.strides[dim])
            .and_then(|delta| self.offset.checked_add(delta))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "select offset overflow for index {index} in dimension {dim}"
                ))
            })?;
        let mut new_dims = self.dims.to_vec();
        let mut new_strides = self.strides.to_vec();
        new_dims.remove(dim);
        new_strides.remove(dim);
        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), offset))
    }

    /// Narrow (slice) a dimension to a sub-range.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 10], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let sub = t.narrow(1, 2, 3).unwrap();
    /// assert_eq!(sub.dims(), &[2, 3]);
    /// ```
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>> {
        self.wait();
        if dim >= self.ndim() {
            return Err(Error::InvalidArgument(format!(
                "dim {dim} out of range for tensor with {} dimensions",
                self.ndim()
            )));
        }
        if start
            .checked_add(length)
            .is_none_or(|end| end > self.dims[dim])
        {
            return Err(Error::InvalidArgument(format!(
                "narrow range out of bounds for dimension {dim} with size {}",
                self.dims[dim]
            )));
        }

        let offset = self
            .offset
            .checked_add(
                (start as isize)
                    .checked_mul(self.strides[dim])
                    .ok_or_else(|| {
                        Error::InvalidArgument("overflow in narrow offset calculation".to_string())
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidArgument("overflow in narrow offset calculation".to_string())
            })?;
        let mut new_dims = self.dims.to_vec();
        new_dims[dim] = length;
        Ok(self.shared_view_with(Arc::from(new_dims), self.strides.clone(), offset))
    }
}
