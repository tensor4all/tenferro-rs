use super::{
    linear_offset, try_linear_offset_for_shape, Tensor, TensorRank, TensorScalar, TypedTensor,
};

fn try_linear_offset(shape: &[usize], indices: &[usize]) -> Option<usize> {
    if indices.len() != shape.len() {
        return None;
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&idx, &extent) in indices.iter().zip(shape) {
        if idx >= extent {
            return None;
        }
        offset = offset.checked_add(idx.checked_mul(stride)?)?;
        stride = stride.checked_mul(extent)?;
    }
    Some(offset)
}

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
        offset = offset
            .checked_add(
                idx.checked_mul(stride)
                    .unwrap_or_else(|| panic!("linear offset multiply overflows")),
            )
            .unwrap_or_else(|| panic!("linear offset add overflows"));
        stride = stride
            .checked_mul(extent)
            .unwrap_or_else(|| panic!("linear offset stride overflows"));
    }
    offset
}

fn linear_offset2(shape: &[usize], i: usize, j: usize) -> usize {
    // Compatibility panic wrapper; public non-panicking callers use
    // `try_linear_offset2`/`try_get2`.
    try_linear_offset2(shape, i, j).unwrap_or_else(|| panic!("rank-2 index out of bounds"))
}

fn try_linear_offset2(shape: &[usize], i: usize, j: usize) -> Option<usize> {
    if shape.len() != 2 || i >= shape[0] || j >= shape[1] {
        return None;
    }
    i.checked_add(shape[0].checked_mul(j)?)
}

fn linear_offset3(shape: &[usize], i: usize, j: usize, k: usize) -> usize {
    // Compatibility panic wrapper; public non-panicking callers use
    // `try_linear_offset3`/`try_get3`.
    try_linear_offset3(shape, i, j, k).unwrap_or_else(|| panic!("rank-3 index out of bounds"))
}

fn try_linear_offset3(shape: &[usize], i: usize, j: usize, k: usize) -> Option<usize> {
    if shape.len() != 3 || i >= shape[0] || j >= shape[1] || k >= shape[2] {
        return None;
    }
    let inner = j.checked_add(shape[1].checked_mul(k)?)?;
    i.checked_add(shape[0].checked_mul(inner)?)
}

impl<T: Clone, R: TensorRank> TypedTensor<T, R> {
    /// View the tensor data as a flat slice in physical memory order.
    ///
    /// This is an explicit alias for [`TypedTensor::as_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.as_physical_slice(), &[1.0, 2.0]);
    /// ```
    pub fn as_physical_slice(&self) -> &[T] {
        self.host_data()
    }

    /// Iterate over the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// let sum: f64 = t.iter().copied().sum();
    /// assert_eq!(sum, 3.0);
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.host_data().iter()
    }

    /// Mutably view the tensor data as a flat slice in physical memory order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// t.as_physical_slice_mut()[1] = 5.0;
    /// assert_eq!(t.as_physical_slice(), &[1.0, 5.0]);
    /// ```
    pub fn as_physical_slice_mut(&mut self) -> &mut [T] {
        self.host_data_mut()
    }

    /// Mutably iterate over the contiguous column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// for value in t.iter_mut() {
    ///     *value *= 2.0;
    /// }
    /// assert_eq!(t.as_slice(), &[2.0, 4.0]);
    /// ```
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.host_data_mut().iter_mut()
    }

    /// Compute the linear physical-buffer offset for a rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.linear_offset2(1, 2), 5);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank or
    /// bounds errors; use [`Self::try_linear_offset2`] for user input.
    pub fn linear_offset2(&self, i: usize, j: usize) -> usize {
        linear_offset2(self.shape(), i, j)
    }

    /// Try to compute the linear physical-buffer offset for a rank-2 logical index.
    pub fn try_linear_offset2(&self, i: usize, j: usize) -> Option<usize> {
        try_linear_offset2(self.shape(), i, j)
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3, 2], vec![0.0; 12]);
    /// assert_eq!(t.linear_offset3(1, 2, 1), 11);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank or
    /// bounds errors; use [`Self::try_linear_offset3`] for user input.
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> usize {
        linear_offset3(self.shape(), i, j, k)
    }

    /// Try to compute the linear physical-buffer offset for a rank-3 logical index.
    pub fn try_linear_offset3(&self, i: usize, j: usize, k: usize) -> Option<usize> {
        try_linear_offset3(self.shape(), i, j, k)
    }

    /// Try to borrow a single element by multi-index.
    ///
    /// Returns `None` when the rank or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.try_get(&[1]), Some(&2.0));
    /// assert_eq!(t.try_get(&[2]), None);
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        let off = try_linear_offset(self.shape(), indices)?;
        self.try_host_data().ok()?.get(off)
    }

    /// Borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.get2(1, 0), &2.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank,
    /// bounds, or backend-buffer host access errors; use [`Self::try_get2`]
    /// for user input.
    pub fn get2(&self, i: usize, j: usize) -> &T {
        let off = self.linear_offset2(i, j);
        &self.host_data()[off]
    }

    /// Try to borrow a single element by rank-2 logical index.
    pub fn try_get2(&self, i: usize, j: usize) -> Option<&T> {
        let off = self.try_linear_offset2(i, j)?;
        self.try_host_data().ok()?.get(off)
    }

    /// Borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![1, 1, 2], vec![3.0, 4.0]);
    /// assert_eq!(t.get3(0, 0, 1), &4.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank,
    /// bounds, or backend-buffer host access errors; use [`Self::try_get3`]
    /// for user input.
    pub fn get3(&self, i: usize, j: usize, k: usize) -> &T {
        let off = self.linear_offset3(i, j, k);
        &self.host_data()[off]
    }

    /// Try to borrow a single element by rank-3 logical index.
    pub fn try_get3(&self, i: usize, j: usize, k: usize) -> Option<&T> {
        let off = self.try_linear_offset3(i, j, k)?;
        self.try_host_data().ok()?.get(off)
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
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(unsafe { *t.get_unchecked(&[1]) }, 2.0);
    /// ```
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        let off = linear_offset_unchecked(self.shape(), indices);
        unsafe { self.host_data().get_unchecked(off) }
    }

    /// Try to mutably borrow a single element by multi-index.
    ///
    /// Returns `None` when the rank or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]);
    /// *t.try_get_mut(&[0]).unwrap() = 2.0;
    /// assert_eq!(t.as_slice(), &[2.0]);
    /// ```
    pub fn try_get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let off = try_linear_offset(self.shape(), indices)?;
        self.try_host_data_mut().ok()?.get_mut(off)
    }

    /// Mutably borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// *t.get_mut2(1, 0) = 5.0;
    /// assert_eq!(t.as_slice(), &[1.0, 5.0, 3.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank,
    /// bounds, or backend-buffer host access errors; use [`Self::try_get_mut2`]
    /// for user input.
    pub fn get_mut2(&mut self, i: usize, j: usize) -> &mut T {
        let off = self.linear_offset2(i, j);
        &mut self.host_data_mut()[off]
    }

    /// Try to mutably borrow a single element by rank-2 logical index.
    pub fn try_get_mut2(&mut self, i: usize, j: usize) -> Option<&mut T> {
        let off = self.try_linear_offset2(i, j)?;
        self.try_host_data_mut().ok()?.get_mut(off)
    }

    /// Mutably borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1, 1, 2], vec![3.0, 4.0]);
    /// *t.get_mut3(0, 0, 1) = 5.0;
    /// assert_eq!(t.as_slice(), &[3.0, 5.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank,
    /// bounds, or backend-buffer host access errors; use [`Self::try_get_mut3`]
    /// for user input.
    pub fn get_mut3(&mut self, i: usize, j: usize, k: usize) -> &mut T {
        let off = self.linear_offset3(i, j, k);
        &mut self.host_data_mut()[off]
    }

    /// Try to mutably borrow a single element by rank-3 logical index.
    pub fn try_get_mut3(&mut self, i: usize, j: usize, k: usize) -> Option<&mut T> {
        let off = self.try_linear_offset3(i, j, k)?;
        self.try_host_data_mut().ok()?.get_mut(off)
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
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]);
    /// unsafe {
    ///     *t.get_unchecked_mut(&[0]) = 2.0;
    /// }
    /// assert_eq!(t.as_slice(), &[2.0]);
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        let off = linear_offset_unchecked(self.shape(), indices);
        unsafe { self.host_data_mut().get_unchecked_mut(off) }
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
    /// let t = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]);
    /// assert_eq!(t.linear_offset(&[1, 2]), 5);
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> usize {
        linear_offset(self.shape(), indices)
    }

    /// Try to compute the linear physical-buffer offset for a logical index.
    pub fn try_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "Tensor::try_linear_offset")
    }

    /// Compute the linear physical-buffer offset for a rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]);
    /// assert_eq!(t.linear_offset2(1, 2), 5);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank or
    /// bounds errors; use [`Self::try_linear_offset2`] for user input.
    pub fn linear_offset2(&self, i: usize, j: usize) -> usize {
        linear_offset2(self.shape(), i, j)
    }

    /// Try to compute the linear physical-buffer offset for a rank-2 logical index.
    pub fn try_linear_offset2(&self, i: usize, j: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j], "Tensor::try_linear_offset2")
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 3, 2], vec![0.0_f64; 12]);
    /// assert_eq!(t.linear_offset3(1, 2, 1), 11);
    /// ```
    ///
    /// # Panics
    ///
    /// Compatibility wrapper for hot valid-index paths. Panics on rank or
    /// bounds errors; use [`Self::try_linear_offset3`] for user input.
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> usize {
        linear_offset3(self.shape(), i, j, k)
    }

    /// Try to compute the linear physical-buffer offset for a rank-3 logical index.
    pub fn try_linear_offset3(&self, i: usize, j: usize, k: usize) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), &[i, j, k], "Tensor::try_linear_offset3")
    }

    /// Try to borrow the host data as a typed physical-memory-order slice.
    ///
    /// This is an explicit alias for [`Tensor::as_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// assert_eq!(t.as_physical_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn as_physical_slice<T: TensorScalar>(&self) -> Option<&[T]> {
        self.as_slice::<T>()
    }

    /// Try to mutably borrow the host data as a typed physical-memory-order
    /// slice.
    ///
    /// This is an explicit alias for [`Tensor::as_slice_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// t.as_physical_slice_mut::<f64>().unwrap()[1] = 5.0;
    /// assert_eq!(t.as_physical_slice::<f64>().unwrap(), &[1.0, 5.0]);
    /// ```
    pub fn as_physical_slice_mut<T: TensorScalar>(&mut self) -> Option<&mut [T]> {
        self.as_slice_mut::<T>()
    }

    /// Try to borrow a single typed element by multi-index.
    ///
    /// Returns `None` when the dtype does not match `T`, the rank does not
    /// match, or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// assert_eq!(t.try_get::<f64>(&[1]), Some(&2.0));
    /// assert_eq!(t.try_get::<f32>(&[1]), None);
    /// ```
    pub fn try_get<T: TensorScalar>(&self, indices: &[usize]) -> Option<&T> {
        let off = try_linear_offset(self.shape(), indices)?;
        self.as_slice::<T>()?.get(off)
    }

    /// Try to mutably borrow a single typed element by multi-index.
    ///
    /// Returns `None` when the dtype does not match `T`, the rank does not
    /// match, or any index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]);
    /// *t.try_get_mut::<f64>(&[0]).unwrap() = 2.0;
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub fn try_get_mut<T: TensorScalar>(&mut self, indices: &[usize]) -> Option<&mut T> {
        let off = try_linear_offset(self.shape(), indices)?;
        self.as_slice_mut::<T>()?.get_mut(off)
    }

    /// Try to borrow a single typed element by multi-index without
    /// release-mode bounds checks.
    ///
    /// Returns `None` when the dtype does not match `T`. Debug builds still
    /// validate the rank and bounds.
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
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// assert_eq!(unsafe { *t.get_unchecked::<f64>(&[1]).unwrap() }, 2.0);
    /// ```
    pub unsafe fn get_unchecked<T: TensorScalar>(&self, indices: &[usize]) -> Option<&T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        let data = self.as_slice::<T>()?;
        Some(unsafe { data.get_unchecked(off) })
    }

    /// Try to mutably borrow a single typed element by multi-index without
    /// release-mode bounds checks.
    ///
    /// Returns `None` when the dtype does not match `T`. Debug builds still
    /// validate the rank and bounds.
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
    /// let mut t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]);
    /// unsafe {
    ///     *t.get_unchecked_mut::<f64>(&[0]).unwrap() = 2.0;
    /// }
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub unsafe fn get_unchecked_mut<T: TensorScalar>(
        &mut self,
        indices: &[usize],
    ) -> Option<&mut T> {
        let off = linear_offset_unchecked(self.shape(), indices);
        let data = self.as_slice_mut::<T>()?;
        Some(unsafe { data.get_unchecked_mut(off) })
    }

    /// Try to mutably borrow the host data as a typed slice.
    ///
    /// Returns `None` if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// t.as_slice_mut::<f64>().unwrap()[0] = 3.0;
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[3.0, 2.0]);
    /// assert_eq!(t.as_slice_mut::<f32>(), None);
    /// ```
    pub fn as_slice_mut<T: TensorScalar>(&mut self) -> Option<&mut [T]> {
        T::try_as_slice_mut(self)
    }

    /// Try to iterate over the contiguous host buffer in physical memory
    /// order.
    ///
    /// Returns `None` if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let sum: f64 = t.iter::<f64>().unwrap().copied().sum();
    /// assert_eq!(sum, 3.0);
    /// assert!(t.iter::<f32>().is_none());
    /// ```
    pub fn iter<T: TensorScalar>(&self) -> Option<std::slice::Iter<'_, T>> {
        self.as_slice::<T>().map(|slice| slice.iter())
    }

    /// Try to mutably iterate over the contiguous host buffer in physical
    /// memory order.
    ///
    /// Returns `None` if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// for value in t.iter_mut::<f64>().unwrap() {
    ///     *value += 1.0;
    /// }
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    /// assert!(t.iter_mut::<f32>().is_none());
    /// ```
    pub fn iter_mut<T: TensorScalar>(&mut self) -> Option<std::slice::IterMut<'_, T>> {
        self.as_slice_mut::<T>().map(|slice| slice.iter_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::{linear_offset2, linear_offset3, linear_offset_unchecked, try_linear_offset};

    #[test]
    fn linear_offset_helpers_check_overflow() {
        let shape = [usize::MAX, 3];

        assert_eq!(try_linear_offset(&shape, &[0, 2]), None);
        assert!(std::panic::catch_unwind(|| linear_offset_unchecked(&shape, &[0, 2])).is_err());
        assert!(std::panic::catch_unwind(|| linear_offset2(&shape, 0, 2)).is_err());
        assert!(std::panic::catch_unwind(|| linear_offset3(&[usize::MAX, 3, 2], 0, 2, 1)).is_err());
    }
}
