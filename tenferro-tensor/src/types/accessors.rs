use super::{linear_offset, Tensor, TensorScalar, TypedTensor};

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
        offset += idx * stride;
        stride *= extent;
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
        offset += idx * stride;
        stride *= extent;
    }
    offset
}

fn linear_offset2(shape: &[usize], i: usize, j: usize) -> usize {
    assert_eq!(shape.len(), 2);
    assert!(i < shape[0], "index out of bounds");
    assert!(j < shape[1], "index out of bounds");
    linear_offset2_unchecked(shape, i, j)
}

fn linear_offset2_unchecked(shape: &[usize], i: usize, j: usize) -> usize {
    debug_assert_eq!(shape.len(), 2);
    debug_assert!(i < shape[0], "index out of bounds");
    debug_assert!(j < shape[1], "index out of bounds");
    i + shape[0] * j
}

fn linear_offset3(shape: &[usize], i: usize, j: usize, k: usize) -> usize {
    assert_eq!(shape.len(), 3);
    assert!(i < shape[0], "index out of bounds");
    assert!(j < shape[1], "index out of bounds");
    assert!(k < shape[2], "index out of bounds");
    linear_offset3_unchecked(shape, i, j, k)
}

fn linear_offset3_unchecked(shape: &[usize], i: usize, j: usize, k: usize) -> usize {
    debug_assert_eq!(shape.len(), 3);
    debug_assert!(i < shape[0], "index out of bounds");
    debug_assert!(j < shape[1], "index out of bounds");
    debug_assert!(k < shape[2], "index out of bounds");
    i + shape[0] * (j + shape[1] * k)
}

impl<T: Clone> TypedTensor<T> {
    /// View the tensor data as a flat slice in physical memory order.
    ///
    /// This is an explicit alias for [`TypedTensor::as_slice`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
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
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
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
    /// let mut t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
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
    /// let mut t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
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
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.linear_offset2(1, 2), 5);
    /// ```
    pub fn linear_offset2(&self, i: usize, j: usize) -> usize {
        linear_offset2(&self.shape, i, j)
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3, 2], vec![0.0; 12]);
    /// assert_eq!(t.linear_offset3(1, 2, 1), 11);
    /// ```
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> usize {
        linear_offset3(&self.shape, i, j, k)
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
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.try_get(&[1]), Some(&2.0));
    /// assert_eq!(t.try_get(&[2]), None);
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        let off = try_linear_offset(&self.shape, indices)?;
        self.host_data().get(off)
    }

    /// Borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.get2(1, 0), &2.0);
    /// ```
    pub fn get2(&self, i: usize, j: usize) -> &T {
        let off = self.linear_offset2(i, j);
        &self.host_data()[off]
    }

    /// Borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![1, 1, 2], vec![3.0, 4.0]);
    /// assert_eq!(t.get3(0, 0, 1), &4.0);
    /// ```
    pub fn get3(&self, i: usize, j: usize, k: usize) -> &T {
        let off = self.linear_offset3(i, j, k);
        &self.host_data()[off]
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
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(unsafe { *t.get_unchecked(&[1]) }, 2.0);
    /// ```
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        let off = linear_offset_unchecked(&self.shape, indices);
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
    /// let mut t = TypedTensor::<f64>::from_vec(vec![1], vec![1.0]);
    /// *t.try_get_mut(&[0]).unwrap() = 2.0;
    /// assert_eq!(t.as_slice(), &[2.0]);
    /// ```
    pub fn try_get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let off = try_linear_offset(&self.shape, indices)?;
        self.host_data_mut().get_mut(off)
    }

    /// Mutably borrow a single element by rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// *t.get_mut2(1, 0) = 5.0;
    /// assert_eq!(t.as_slice(), &[1.0, 5.0, 3.0, 4.0]);
    /// ```
    pub fn get_mut2(&mut self, i: usize, j: usize) -> &mut T {
        let off = self.linear_offset2(i, j);
        &mut self.host_data_mut()[off]
    }

    /// Mutably borrow a single element by rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::from_vec(vec![1, 1, 2], vec![3.0, 4.0]);
    /// *t.get_mut3(0, 0, 1) = 5.0;
    /// assert_eq!(t.as_slice(), &[3.0, 5.0]);
    /// ```
    pub fn get_mut3(&mut self, i: usize, j: usize, k: usize) -> &mut T {
        let off = self.linear_offset3(i, j, k);
        &mut self.host_data_mut()[off]
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
    /// let mut t = TypedTensor::<f64>::from_vec(vec![1], vec![1.0]);
    /// unsafe {
    ///     *t.get_unchecked_mut(&[0]) = 2.0;
    /// }
    /// assert_eq!(t.as_slice(), &[2.0]);
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        let off = linear_offset_unchecked(&self.shape, indices);
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
    /// let t = Tensor::from_vec(vec![2, 3], vec![0.0_f64; 6]);
    /// assert_eq!(t.linear_offset(&[1, 2]), 5);
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> usize {
        linear_offset(self.shape(), indices)
    }

    /// Compute the linear physical-buffer offset for a rank-2 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3], vec![0.0_f64; 6]);
    /// assert_eq!(t.linear_offset2(1, 2), 5);
    /// ```
    pub fn linear_offset2(&self, i: usize, j: usize) -> usize {
        linear_offset2(self.shape(), i, j)
    }

    /// Compute the linear physical-buffer offset for a rank-3 logical index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3, 2], vec![0.0_f64; 12]);
    /// assert_eq!(t.linear_offset3(1, 2, 1), 11);
    /// ```
    pub fn linear_offset3(&self, i: usize, j: usize, k: usize) -> usize {
        linear_offset3(self.shape(), i, j, k)
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
    /// let t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let mut t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let mut t = Tensor::from_vec(vec![1], vec![1.0_f64]);
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
    /// let t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let mut t = Tensor::from_vec(vec![1], vec![1.0_f64]);
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
    /// let mut t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
    /// let mut t = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
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
