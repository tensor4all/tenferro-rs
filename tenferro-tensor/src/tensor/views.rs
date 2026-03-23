use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use super::Tensor;
use crate::layout::{compute_contiguous_strides, validate_layout_against_len};

fn view_as_real_strides(strides: &[isize]) -> Result<Vec<isize>> {
    let mut output = Vec::with_capacity(strides.len() + 1);
    for (axis, &stride) in strides.iter().enumerate() {
        output.push(stride.checked_mul(2).ok_or_else(|| {
            Error::StrideError(format!(
                "view_as_real: stride overflow for axis {axis} with stride {stride}"
            ))
        })?);
    }
    output.push(1);
    Ok(output)
}

fn view_as_complex_strides(strides: &[isize]) -> Result<Vec<isize>> {
    if strides.is_empty() {
        return Err(Error::InvalidArgument(
            "view_as_complex requires at least one dimension".into(),
        ));
    }

    let mut output = Vec::with_capacity(strides.len().saturating_sub(1));
    for (axis, &stride) in strides[..strides.len() - 1].iter().enumerate() {
        if stride % 2 != 0 {
            return Err(Error::InvalidArgument(format!(
                "view_as_complex requires even strides on all leading dimensions, got stride {stride} at axis {axis}"
            )));
        }
        output.push(stride / 2);
    }
    Ok(output)
}

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
            let stride = self.strides[i]
                .checked_add(self.strides[j])
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "diagonal stride overflow for axes ({i}, {j}) with strides {} and {}",
                        self.strides[i], self.strides[j]
                    ))
                })?;
            diag_strides.push(stride);
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
    /// Reshape follows tenferro's internal column-major semantics for contiguous
    /// tensors. Import/export helpers may accept other memory orders, but the
    /// reshaped view itself is laid out as column-major metadata.
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

        let new_strides = Arc::from(compute_contiguous_strides(
            new_dims,
            crate::MemoryOrder::ColumnMajor,
        ));
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

        let offset = (start as isize)
            .checked_mul(self.strides[dim])
            .and_then(|delta| self.offset.checked_add(delta))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "narrow offset overflow for start {start} in dimension {dim}"
                ))
            })?;
        let mut new_dims = self.dims.to_vec();
        new_dims[dim] = length;
        Ok(self.shared_view_with(Arc::from(new_dims), self.strides.clone(), offset))
    }

    /// Insert a size-1 dimension at the specified position.
    ///
    /// This is a zero-copy view operation. Negative dimensions are supported
    /// and count from the end.
    ///
    /// # Arguments
    ///
    /// * `dim` - Position to insert the new dimension. Must be in range `[-ndim-1, ndim]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension is out of range.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let u = t.unsqueeze(0).unwrap();
    /// assert_eq!(u.dims(), &[1, 2, 3]);
    ///
    /// let u2 = t.unsqueeze(-1).unwrap();
    /// assert_eq!(u2.dims(), &[2, 3, 1]);
    /// ```
    pub fn unsqueeze(&self, dim: isize) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();

        let dim = if dim < 0 {
            let wrapped = dim + (ndim as isize) + 1;
            if wrapped < 0 {
                return Err(Error::InvalidArgument(format!(
                    "unsqueeze dim {dim} out of range for tensor with {ndim} dimensions (valid: [{}, {}])",
                    -(ndim as isize) - 1,
                    ndim
                )));
            }
            wrapped as usize
        } else if dim as usize > ndim {
            return Err(Error::InvalidArgument(format!(
                "unsqueeze dim {dim} out of range for tensor with {ndim} dimensions (valid: [{}, {}])",
                -(ndim as isize) - 1,
                ndim
            )));
        } else {
            dim as usize
        };

        let mut new_dims: Vec<usize> = self.dims.to_vec();
        new_dims.insert(dim, 1);

        let mut new_strides: Vec<isize> = self.strides.to_vec();
        let new_stride = if dim < ndim {
            self.strides[dim]
        } else {
            let last_stride = if ndim > 0 { self.strides[ndim - 1] } else { 1 };
            last_stride
        };
        new_strides.insert(dim, new_stride);

        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), self.offset))
    }

    /// Remove all size-1 dimensions from the tensor.
    ///
    /// This is a zero-copy view operation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[1, 2, 1, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let s = t.squeeze().unwrap();
    /// assert_eq!(s.dims(), &[2, 3]);
    /// ```
    pub fn squeeze(&self) -> Result<Tensor<T>> {
        self.wait();
        let new_dims: Vec<usize> = self.dims.iter().filter(|&&d| d != 1).copied().collect();
        let new_strides: Vec<isize> = self
            .dims
            .iter()
            .zip(self.strides.iter())
            .filter(|(&d, _)| d != 1)
            .map(|(_, &s)| s)
            .collect();

        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), self.offset))
    }

    /// Remove a specific size-1 dimension from the tensor.
    ///
    /// This is a zero-copy view operation. Negative dimensions are supported
    /// and count from the end.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to remove. Must be in range `[-ndim, ndim-1]` and have size 1.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dimension is out of range
    /// - The dimension does not have size 1
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 1, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let s = t.squeeze_dim(1).unwrap();
    /// assert_eq!(s.dims(), &[2, 3]);
    ///
    /// let s2 = t.squeeze_dim(-2).unwrap();
    /// assert_eq!(s2.dims(), &[2, 3]);
    /// ```
    pub fn squeeze_dim(&self, dim: isize) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();

        if ndim == 0 {
            return Err(Error::InvalidArgument(
                "squeeze_dim: cannot squeeze a rank-0 tensor".to_string(),
            ));
        }

        let dim = if dim < 0 {
            let wrapped = dim + (ndim as isize);
            if wrapped < 0 {
                return Err(Error::InvalidArgument(format!(
                    "squeeze_dim dim {dim} out of range for tensor with {ndim} dimensions (valid: [{}, {}])",
                    -(ndim as isize),
                    ndim - 1
                )));
            }
            wrapped as usize
        } else if dim as usize >= ndim {
            return Err(Error::InvalidArgument(format!(
                "squeeze_dim dim {dim} out of range for tensor with {ndim} dimensions (valid: [{}, {}])",
                -(ndim as isize),
                ndim - 1
            )));
        } else {
            dim as usize
        };

        if self.dims[dim] != 1 {
            return Err(Error::InvalidArgument(format!(
                "squeeze_dim: dimension {dim} has size {} (expected 1)",
                self.dims[dim]
            )));
        }

        let mut new_dims = self.dims.to_vec();
        new_dims.remove(dim);

        let mut new_strides = self.strides.to_vec();
        new_strides.remove(dim);

        Ok(self.shared_view_with(Arc::from(new_dims), Arc::from(new_strides), self.offset))
    }
}

impl Tensor<Complex32> {
    /// Return a zero-copy real view of a complex tensor.
    ///
    /// The last logical axis is expanded to length `2`, exposing the real and
    /// imaginary parts as adjacent real-valued elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let z = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 2.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let r = z.view_as_real().unwrap();
    /// assert_eq!(r.dims(), &[1, 2]);
    /// assert_eq!(r.logical_memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn view_as_real(&self) -> Result<Tensor<f32>> {
        self.wait();
        if self.is_conjugated() {
            return Err(Error::InvalidArgument(
                "view_as_real requires a resolved complex tensor".into(),
            ));
        }

        let mut dims = self.dims().to_vec();
        dims.push(2);
        let strides = view_as_real_strides(self.strides())?;
        let offset = self.offset().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: offset overflow when reinterpreting tensor".into())
        })?;
        let buffer_len = self.buffer().len().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: storage length overflow".into())
        })?;
        let buffer = self.buffer().reinterpret_as::<f32>(buffer_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<Complex64> {
    /// Return a zero-copy real view of a complex tensor.
    ///
    /// The last logical axis is expanded to length `2`, exposing the real and
    /// imaginary parts as adjacent real-valued elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let z = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 2.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let r = z.view_as_real().unwrap();
    /// assert_eq!(r.dims(), &[1, 2]);
    /// ```
    pub fn view_as_real(&self) -> Result<Tensor<f64>> {
        self.wait();
        if self.is_conjugated() {
            return Err(Error::InvalidArgument(
                "view_as_real requires a resolved complex tensor".into(),
            ));
        }

        let mut dims = self.dims().to_vec();
        dims.push(2);
        let strides = view_as_real_strides(self.strides())?;
        let offset = self.offset().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: offset overflow when reinterpreting tensor".into())
        })?;
        let buffer_len = self.buffer().len().checked_mul(2).ok_or_else(|| {
            Error::StrideError("view_as_real: storage length overflow".into())
        })?;
        let buffer = self.buffer().reinterpret_as::<f64>(buffer_len)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<f32> {
    /// Return a zero-copy complex view of a real tensor whose last dimension
    /// stores paired real and imaginary components.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let r = Tensor::<f32>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let z = r.view_as_complex().unwrap();
    /// assert_eq!(z.dims(), &[] as &[usize]);
    /// ```
    pub fn view_as_complex(&self) -> Result<Tensor<Complex32>> {
        self.wait();
        if self.ndim() == 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires at least one dimension".into(),
            ));
        }
        if self.dims().last().copied() != Some(2) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last dimension to have size 2".into(),
            ));
        }
        if self.strides().last().copied() != Some(1) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last stride to be 1".into(),
            ));
        }
        if self.offset() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even element offset".into(),
            ));
        }

        let dims = self.dims()[..self.ndim() - 1].to_vec();
        let strides = view_as_complex_strides(self.strides())?;
        let offset = self.offset() / 2;
        let source_len = self.buffer().len();
        if source_len % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even-sized real storage buffer".into(),
            ));
        }
        let buffer = self.buffer().reinterpret_as::<Complex32>(source_len / 2)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}

impl Tensor<f64> {
    /// Return a zero-copy complex view of a real tensor whose last dimension
    /// stores paired real and imaginary components.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let r = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let z = r.view_as_complex().unwrap();
    /// assert_eq!(z.dims(), &[] as &[usize]);
    /// ```
    pub fn view_as_complex(&self) -> Result<Tensor<Complex64>> {
        self.wait();
        if self.ndim() == 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires at least one dimension".into(),
            ));
        }
        if self.dims().last().copied() != Some(2) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last dimension to have size 2".into(),
            ));
        }
        if self.strides().last().copied() != Some(1) {
            return Err(Error::InvalidArgument(
                "view_as_complex requires the last stride to be 1".into(),
            ));
        }
        if self.offset() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even element offset".into(),
            ));
        }

        let dims = self.dims()[..self.ndim() - 1].to_vec();
        let strides = view_as_complex_strides(self.strides())?;
        let offset = self.offset() / 2;
        let source_len = self.buffer().len();
        if source_len % 2 != 0 {
            return Err(Error::InvalidArgument(
                "view_as_complex requires an even-sized real storage buffer".into(),
            ));
        }
        let buffer = self.buffer().reinterpret_as::<Complex64>(source_len / 2)?;
        validate_layout_against_len(&dims, &strides, offset, buffer.len())?;
        Ok(Tensor::from_parts(
            buffer,
            Arc::from(dims),
            Arc::from(strides),
            offset,
            self.logical_memory_space(),
            self.preferred_compute_device(),
            None,
            false,
            None,
        ))
    }
}
