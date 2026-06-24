use super::{try_linear_offset_for_shape, Tensor, TensorRank, TensorScalar, TypedTensor};

fn debug_assert_index_in_bounds(shape: &[usize], indices: &[usize]) {
    debug_assert_eq!(indices.len(), shape.len());
    for (&idx, &extent) in indices.iter().zip(shape) {
        debug_assert!(idx < extent, "index out of bounds");
    }
}

fn linear_offset_unchecked(shape: &[usize], indices: &[usize]) -> usize {
    debug_assert_index_in_bounds(shape, indices);
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&idx, &extent) in indices.iter().zip(shape) {
        offset = offset.wrapping_add(idx.wrapping_mul(stride));
        stride = stride.wrapping_mul(extent);
    }
    offset
}

impl<T: Clone, R: TensorRank> TypedTensor<T, R> {
    /// Iterate over the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let sum: f64 = t.iter()?.copied().sum();
    /// assert_eq!(sum, 3.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn iter(&self) -> crate::Result<std::slice::Iter<'_, T>> {
        Ok(self.host_data()?.iter())
    }

    /// Mutably iterate over the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// for value in t.iter_mut()? {
    ///     *value *= 2.0;
    /// }
    /// assert_eq!(t.as_slice()?, &[2.0, 4.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn iter_mut(&mut self) -> crate::Result<std::slice::IterMut<'_, T>> {
        Ok(self.host_data_mut()?.iter_mut())
    }

    /// Compute the linear physical-buffer offset for a rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.linear_offset2(1, 2)?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset2(&self, i: usize, j: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j], "TypedTensor::linear_offset2")
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3, 2], vec![0.0; 12]).unwrap();
    /// assert_eq!(t.linear_offset3(1, 2, 1)?, 11);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j, k], "TypedTensor::linear_offset3")
    }

    /// Borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(t.get2(1, 0)?, &2.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get2(&self, i: usize, j: usize) -> crate::Result<&T> {
        let off = self.linear_offset2(i, j)?;
        self.host_data()?
            .get(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get2",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![1, 1, 2], vec![3.0, 4.0]).unwrap();
    /// assert_eq!(t.get3(0, 0, 1)?, &4.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get3(&self, i: usize, j: usize, k: usize) -> crate::Result<&T> {
        let off = self.linear_offset3(i, j, k)?;
        self.host_data()?
            .get(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get3",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Borrow a single element by multi-index without release-mode bounds
    /// checks.
    ///
    /// Debug builds still validate the rank and bounds.
    ///
    /// # Safety
    ///
    /// `indices` must have the same rank as this tensor and every index must
    /// be in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(unsafe { *t.get_unchecked(&[1])? }, 2.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> crate::Result<&T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        Ok(unsafe { self.host_data()?.get_unchecked(off) })
    }

    /// Mutably borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// *t.get_mut2(1, 0)? = 5.0;
    /// assert_eq!(t.as_slice()?, &[1.0, 5.0, 3.0, 4.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut2(&mut self, i: usize, j: usize) -> crate::Result<&mut T> {
        let off = self.linear_offset2(i, j)?;
        self.host_data_mut()?
            .get_mut(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get_mut2",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Mutably borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1, 1, 2], vec![3.0, 4.0]).unwrap();
    /// *t.get_mut3(0, 0, 1)? = 5.0;
    /// assert_eq!(t.as_slice()?, &[3.0, 5.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut3(&mut self, i: usize, j: usize, k: usize) -> crate::Result<&mut T> {
        let off = self.linear_offset3(i, j, k)?;
        self.host_data_mut()?
            .get_mut(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get_mut3",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Mutably borrow a single element by multi-index without release-mode
    /// bounds checks.
    ///
    /// Debug builds still validate the rank and bounds.
    ///
    /// # Safety
    ///
    /// `indices` must have the same rank as this tensor and every index must
    /// be in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    /// unsafe {
    ///     *t.get_unchecked_mut(&[0])? = 2.0;
    /// }
    /// assert_eq!(t.as_slice()?, &[2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> crate::Result<&mut T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        Ok(unsafe { self.host_data_mut()?.get_unchecked_mut(off) })
    }
}

impl Tensor {
    /// Compute the linear physical-buffer offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]).unwrap();
    /// assert_eq!(t.linear_offset(&[1, 2])?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "Tensor::linear_offset")
    }

    /// Compute the linear physical-buffer offset for a rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]).unwrap();
    /// assert_eq!(t.linear_offset2(1, 2)?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset2(&self, i: usize, j: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j], "Tensor::linear_offset2")
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 3, 2], vec![0.0_f64; 12]).unwrap();
    /// assert_eq!(t.linear_offset3(1, 2, 1)?, 11);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j, k], "Tensor::linear_offset3")
    }

    /// Borrow a single typed element by multi-index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// assert_eq!(t.get::<f64>(&[1])?, &2.0);
    /// assert!(t.get::<f32>(&[1]).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get<T: TensorScalar>(&self, indices: &[usize]) -> crate::Result<&T> {
        let off = self.linear_offset(indices)?;
        self.as_slice::<T>()?
            .get(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "Tensor::get",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Mutably borrow a single typed element by multi-index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// *t.get_mut::<f64>(&[0])? = 2.0;
    /// assert_eq!(t.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut<T: TensorScalar>(&mut self, indices: &[usize]) -> crate::Result<&mut T> {
        let off = self.linear_offset(indices)?;
        self.as_slice_mut::<T>()?
            .get_mut(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "Tensor::get_mut",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Try to borrow a single typed element by multi-index without
    /// release-mode bounds checks.
    ///
    /// Debug builds still validate the rank and bounds. Dtype and backend
    /// host-access failures are still reported as errors.
    ///
    /// # Safety
    ///
    /// `indices` must have the same rank as this tensor and every index must
    /// be in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// assert_eq!(unsafe { *t.get_unchecked::<f64>(&[1])? }, 2.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub unsafe fn get_unchecked<T: TensorScalar>(&self, indices: &[usize]) -> crate::Result<&T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        let data = self.as_slice::<T>()?;
        Ok(unsafe { data.get_unchecked(off) })
    }

    /// Try to mutably borrow a single typed element by multi-index without
    /// release-mode bounds checks.
    ///
    /// Debug builds still validate the rank and bounds. Dtype and backend
    /// host-access failures are still reported as errors.
    ///
    /// # Safety
    ///
    /// `indices` must have the same rank as this tensor and every index must
    /// be in bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// unsafe {
    ///     *t.get_unchecked_mut::<f64>(&[0])? = 2.0;
    /// }
    /// assert_eq!(t.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub unsafe fn get_unchecked_mut<T: TensorScalar>(
        &mut self,
        indices: &[usize],
    ) -> crate::Result<&mut T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        let data = self.as_slice_mut::<T>()?;
        Ok(unsafe { data.get_unchecked_mut(off) })
    }

    /// Mutably borrow the host data as a typed slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// t.as_slice_mut::<f64>()?[0] = 3.0;
    /// assert_eq!(t.as_slice::<f64>()?, &[3.0, 2.0]);
    /// assert!(t.as_slice_mut::<f32>().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice_mut<T: TensorScalar>(&mut self) -> crate::Result<&mut [T]> {
        T::as_slice_mut(self)
    }

    /// Iterate over the contiguous host buffer in physical memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let sum: f64 = t.iter::<f64>()?.copied().sum();
    /// assert_eq!(sum, 3.0);
    /// assert!(t.iter::<f32>().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn iter<T: TensorScalar>(&self) -> crate::Result<std::slice::Iter<'_, T>> {
        Ok(self.as_slice::<T>()?.iter())
    }

    /// Mutably iterate over the contiguous host buffer in physical memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// for value in t.iter_mut::<f64>()? {
    ///     *value += 1.0;
    /// }
    /// assert_eq!(t.as_slice::<f64>()?, &[2.0, 3.0]);
    /// assert!(t.iter_mut::<f32>().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn iter_mut<T: TensorScalar>(&mut self) -> crate::Result<std::slice::IterMut<'_, T>> {
        Ok(self.as_slice_mut::<T>()?.iter_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::{linear_offset_unchecked, try_linear_offset_for_shape};

    #[test]
    fn linear_offset_helpers_check_overflow() {
        let shape = [usize::MAX, 3];

        assert!(try_linear_offset_for_shape(&shape, &[0, 2], "test").is_err());
        assert!(std::panic::catch_unwind(|| linear_offset_unchecked(&shape, &[0, 2])).is_ok());
    }
}
