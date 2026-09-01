//! Validated const-generic column-major host views.
//!
//! This prototype implements the API proposed by
//! [tenferro-rs issue #1736](https://github.com/tensor4all/tenferro-rs/issues/1736).
//! Construction preserves the compile-time rank and validates host residency,
//! compact column-major layout, shape arithmetic, and the exact logical slice
//! once. Traversal then operates directly on the validated slice.

use std::fmt;

use tenferro_tensor_core::ValidationError;

use super::{checked_view_element_count, Rank, TensorScalar, TypedTensor, TypedTensorView};

#[inline(always)]
fn in_bounds<const N: usize>(shape: &[usize; N], index: [usize; N]) -> bool {
    let mut axis = 0usize;
    while axis < N {
        if index[axis] >= shape[axis] {
            return false;
        }
        axis += 1;
    }
    true
}

#[inline(always)]
fn linear_offset<const N: usize>(shape: &[usize; N], index: [usize; N]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    let mut axis = 0usize;
    while axis < N {
        offset += index[axis] * stride;
        stride *= shape[axis];
        axis += 1;
    }
    offset
}

fn validate_slice_len<const N: usize>(
    shape: &[usize; N],
    actual: usize,
    op: &'static str,
) -> crate::Result<()> {
    let expected = checked_view_element_count(shape, op)?;
    if expected == actual {
        Ok(())
    } else {
        Err(crate::Error::validation(
            op,
            ValidationError::ShapeDataLengthMismatch { expected, actual },
        ))
    }
}

/// Shared view of a validated compact column-major host tensor with rank `N`.
///
/// The first index varies fastest in memory. Construction is available through
/// [`TypedTensor::host_col_major_view`] and
/// [`TypedTensorView::host_col_major_view`].
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Rank, TypedTensor};
///
/// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major(
///     [2, 2],
///     vec![1, 2, 3, 4],
/// )?;
/// let view = tensor.host_col_major_view()?;
/// assert_eq!(view.get([1, 0]), Some(&2));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct ColMajorView<'a, T, const N: usize> {
    data: &'a [T],
    shape: [usize; N],
}

impl<T, const N: usize> fmt::Debug for ColMajorView<'_, T, N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ColMajorView")
            .field("shape", &self.shape)
            .field("len", &self.data.len())
            .finish()
    }
}

impl<'a, T, const N: usize> ColMajorView<'a, T, N> {
    fn new(data: &'a [T], shape: [usize; N], op: &'static str) -> crate::Result<Self> {
        validate_slice_len(&shape, data.len(), op)?;
        Ok(Self { data, shape })
    }

    /// Return the const-generic logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<f64, Rank<2>>::zeros([2, 3])?;
    /// assert_eq!(tensor.host_col_major_view()?.shape(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    /// Return the exact logical slice in column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// assert_eq!(tensor.host_col_major_view()?.as_slice(), &[4, 5]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Iterate in linear column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// assert_eq!(tensor.host_col_major_view()?.iter().copied().sum::<i32>(), 9);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Return an element when every index is in bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4])?;
    /// let view = tensor.host_col_major_view()?;
    /// assert_eq!(view.get([0, 1]), Some(&3));
    /// assert_eq!(view.get([2, 0]), None);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        if in_bounds(&self.shape, index) {
            // SAFETY: every axis was checked above, and construction proved
            // that the compact shape product equals the slice length.
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Return an element without checking rank, bounds, backend, or layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index[axis] < self.shape()[axis]` for every
    /// axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// let view = tensor.host_col_major_view()?;
    /// // SAFETY: index 1 is below the only axis extent, 2.
    /// assert_eq!(unsafe { view.get_unchecked([1]) }, &5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = linear_offset(&self.shape, index);
        // INVARIANT: `ColMajorView::new` calls `validate_slice_len`, proving
        // that the checked shape product equals `data.len()`.
        // SAFETY: the caller guarantees an in-bounds index. The checked
        // constructor proved the compact shape product equals `data.len()`.
        unsafe { self.data.get_unchecked(offset) }
    }

    /// Iterate over contiguous first-axis lanes in column-major order.
    ///
    /// Rank-zero tensors yield one scalar lane. Tensors with an empty axis
    /// yield no lanes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4])?;
    /// let view = tensor.host_col_major_view()?;
    /// let lanes = view.axis0_lanes().collect::<Vec<_>>();
    /// assert_eq!(lanes, vec![&[1, 2][..], &[3, 4][..]]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn axis0_lanes(&self) -> std::slice::ChunksExact<'_, T> {
        self.data.chunks_exact(self.axis0_extent())
    }

    #[inline(always)]
    fn axis0_extent(&self) -> usize {
        if N == 0 || self.shape[0] == 0 {
            1
        } else {
            self.shape[0]
        }
    }
}

/// Exclusive view of a validated compact column-major host tensor with rank `N`.
///
/// Mutable traversal yields disjoint references through Rust slice iterators.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Rank, TypedTensor};
/// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![1, 2])?;
/// let mut view = tensor.host_col_major_view_mut()?;
/// if let Some(value) = view.get_mut([1]) { *value = 7; }
/// assert_eq!(view.as_slice(), &[1, 7]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct ColMajorViewMut<'a, T, const N: usize> {
    data: &'a mut [T],
    shape: [usize; N],
}

impl<T, const N: usize> fmt::Debug for ColMajorViewMut<'_, T, N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ColMajorViewMut")
            .field("shape", &self.shape)
            .field("len", &self.data.len())
            .finish()
    }
}

impl<'a, T, const N: usize> ColMajorViewMut<'a, T, N> {
    fn new(data: &'a mut [T], shape: [usize; N], op: &'static str) -> crate::Result<Self> {
        validate_slice_len(&shape, data.len(), op)?;
        Ok(Self { data, shape })
    }

    /// Return the const-generic logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<f64, Rank<2>>::zeros([2, 3])?;
    /// assert_eq!(tensor.host_col_major_view_mut()?.shape(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    /// Return the exact logical slice in column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// assert_eq!(tensor.host_col_major_view_mut()?.as_slice(), &[4, 5]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Return the exact mutable logical slice in column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![4])?;
    /// *tensor.host_col_major_view_mut()?.as_mut_slice().first_mut().unwrap() = 9;
    /// assert_eq!(tensor.as_slice()?, &[9]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Iterate immutably in linear column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// assert_eq!(tensor.host_col_major_view_mut()?.iter().copied().sum::<i32>(), 9);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably in linear column-major order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![1, 2])?;
    /// for value in tensor.host_col_major_view_mut()?.iter_mut() { *value += 1; }
    /// assert_eq!(tensor.as_slice()?, &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Return an element when every index is in bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// assert_eq!(tensor.host_col_major_view_mut()?.get([1]), Some(&5));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        if in_bounds(&self.shape, index) {
            // SAFETY: every axis was checked above and the constructor proved
            // the compact slice length.
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Return a mutable element when every index is in bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([2], vec![4, 5])?;
    /// *tensor.host_col_major_view_mut()?.get_mut([1]).unwrap() = 8;
    /// assert_eq!(tensor.as_slice()?, &[4, 8]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, index: [usize; N]) -> Option<&mut T> {
        if in_bounds(&self.shape, index) {
            // SAFETY: every axis was checked above and `&mut self` guarantees
            // exclusive access for the returned borrow.
            Some(unsafe { self.get_unchecked_mut(index) })
        } else {
            None
        }
    }

    /// Return an element without checking rank, bounds, backend, or layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index[axis] < self.shape()[axis]` for every
    /// axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![4])?;
    /// let view = tensor.host_col_major_view_mut()?;
    /// // SAFETY: index 0 is below the only axis extent, 1.
    /// assert_eq!(unsafe { view.get_unchecked([0]) }, &4);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = linear_offset(&self.shape, index);
        // INVARIANT: `ColMajorViewMut::new` calls `validate_slice_len`, proving
        // that the checked shape product equals `data.len()`.
        // SAFETY: the caller guarantees an in-bounds index. The checked
        // constructor proved the compact shape product equals `data.len()`.
        unsafe { self.data.get_unchecked(offset) }
    }

    /// Return a mutable element without checking rank, bounds, backend, or layout.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index[axis] < self.shape()[axis]` for every
    /// axis and must not retain overlapping mutable references.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![4])?;
    /// let mut view = tensor.host_col_major_view_mut()?;
    /// // SAFETY: index 0 is in bounds and this is the only active element borrow.
    /// *unsafe { view.get_unchecked_mut([0]) } = 7;
    /// assert_eq!(view.as_slice(), &[7]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; N]) -> &mut T {
        let offset = linear_offset(&self.shape, index);
        // INVARIANT: `ColMajorViewMut::new` calls `validate_slice_len`, proving
        // that the checked shape product equals `data.len()`.
        // SAFETY: the caller guarantees an in-bounds index and exclusive
        // access to the selected element for the returned borrow.
        unsafe { self.data.get_unchecked_mut(offset) }
    }

    /// Iterate immutably over contiguous first-axis lanes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 1], vec![1, 2])?;
    /// assert_eq!(tensor.host_col_major_view_mut()?.axis0_lanes().next(), Some(&[1, 2][..]));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn axis0_lanes(&self) -> std::slice::ChunksExact<'_, T> {
        self.data.chunks_exact(self.axis0_extent())
    }

    /// Iterate mutably over disjoint contiguous first-axis lanes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4])?;
    /// for lane in tensor.host_col_major_view_mut()?.axis0_lanes_mut() { lane[0] *= 10; }
    /// assert_eq!(tensor.as_slice()?, &[10, 2, 30, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[inline(always)]
    pub fn axis0_lanes_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        let extent = self.axis0_extent();
        self.data.chunks_exact_mut(extent)
    }

    #[inline(always)]
    fn axis0_extent(&self) -> usize {
        if N == 0 || self.shape[0] == 0 {
            1
        } else {
            self.shape[0]
        }
    }
}

impl<T: TensorScalar, const N: usize> TypedTensor<T, Rank<N>> {
    /// Validate and borrow this owned tensor as a compact column-major host view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 1], vec![1, 2])?;
    /// assert_eq!(tensor.host_col_major_view()?.get([1, 0]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when storage is backend-owned,
    /// or [`crate::Error::Validation`] when compact layout, shape arithmetic,
    /// or the logical host range is invalid.
    pub fn host_col_major_view(&self) -> crate::Result<ColMajorView<'_, T, N>> {
        const OP: &str = "TypedTensor::host_col_major_view";
        self.assert_col_major_contiguous()?;
        let shape = *self.layout().shape_array();
        ColMajorView::new(self.as_slice()?, shape, OP)
    }

    /// Validate and mutably borrow this owned tensor as a compact column-major host view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    /// let mut tensor = TypedTensor::<i32, Rank<1>>::from_vec_col_major([1], vec![1])?;
    /// if let Some(value) = tensor.host_col_major_view_mut()?.get_mut([0]) { *value = 3; }
    /// assert_eq!(tensor.as_slice()?, &[3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when storage is backend-owned,
    /// or [`crate::Error::Validation`] when compact layout, shape arithmetic,
    /// or the logical host range is invalid.
    pub fn host_col_major_view_mut(&mut self) -> crate::Result<ColMajorViewMut<'_, T, N>> {
        const OP: &str = "TypedTensor::host_col_major_view_mut";
        self.assert_col_major_contiguous()?;
        let shape = *self.layout().shape_array();
        ColMajorViewMut::new(self.host_data_mut()?, shape, OP)
    }
}

impl<'a, T: 'static, const N: usize> TypedTensorView<'a, T, Rank<N>> {
    /// Validate and borrow this tensor view as a compact column-major host view.
    ///
    /// Nonzero compact offsets are represented by the returned logical slice
    /// without copying.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    /// let data = [0_i32, 1, 2, 3, 4];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 1, &data)?;
    /// assert_eq!(view.host_col_major_view()?.as_slice(), &[1, 2, 3, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when storage is backend-owned,
    /// or [`crate::Error::Validation`] when compact layout, shape arithmetic,
    /// or the logical host range is invalid.
    pub fn host_col_major_view(&self) -> crate::Result<ColMajorView<'a, T, N>> {
        const OP: &str = "TypedTensorView::host_col_major_view";
        let shape = *self.layout().shape_array();
        ColMajorView::new(self.as_slice()?, shape, OP)
    }
}

#[cfg(test)]
mod tests;
