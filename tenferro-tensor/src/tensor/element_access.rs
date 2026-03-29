use super::Tensor;

impl<T> Tensor<T> {
    /// Access a single element by multi-dimensional index.
    ///
    /// Returns `None` if the index is out of bounds or the underlying buffer
    /// is not CPU-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// // Column-major: data is laid out column by column.
    /// // from_slice with ColumnMajor and data [1,2,3,4] gives:
    /// //   column 0 = [1, 2], column 1 = [3, 4]
    /// //   matrix = [[1, 3],
    /// //             [2, 4]]
    /// let t = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// assert_eq!(t.get(&[0, 0]), Some(&1.0));
    /// assert_eq!(t.get(&[1, 0]), Some(&2.0));
    /// assert_eq!(t.get(&[0, 1]), Some(&3.0));
    /// assert_eq!(t.get(&[1, 1]), Some(&4.0));
    /// assert_eq!(t.get(&[2, 0]), None); // out of bounds
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let pos = self.linear_offset(index)?;
        self.buffer.as_slice().and_then(|s| s.get(pos))
    }

    /// Access a single element mutably by multi-dimensional index.
    ///
    /// Returns `None` if the index is out of bounds, the buffer is not
    /// CPU-accessible, or the buffer is shared (Arc refcount > 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut t = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// *t.get_mut(&[0, 1]).unwrap() = 99.0;
    /// assert_eq!(t.get(&[0, 1]), Some(&99.0));
    /// // Out of bounds returns None:
    /// assert!(t.get_mut(&[2, 0]).is_none());
    /// ```
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let pos = self.linear_offset(index)?;
        self.buffer.as_mut_slice().and_then(|s| s.get_mut(pos))
    }

    /// Write a value at the given multi-dimensional index.
    ///
    /// Returns `Ok(())` on success, or an error if the index is out of bounds,
    /// the buffer is not CPU-accessible, or the buffer is shared
    /// (Arc refcount > 1). Call [`deep_clone`](Tensor::deep_clone) first to
    /// obtain an exclusively-owned copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut t = Tensor::<f64>::from_slice(
    ///     &[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// t.set(&[1, 0], 10.0).unwrap();
    /// assert_eq!(t.get(&[1, 0]), Some(&10.0));
    ///
    /// // Shared buffers cannot be written:
    /// let shared = t.clone(); // refcount == 2
    /// // t.set(&[0, 0], 5.0) would fail here because buffer is shared
    /// ```
    pub fn set(&mut self, index: &[usize], value: T) -> tenferro_device::Result<()> {
        // Collect error context before taking &mut self.
        let dims_debug = format!("{:?}", &*self.dims);
        let unique = self.buffer.is_unique();
        let elem = self.get_mut(index).ok_or_else(|| {
            tenferro_device::Error::InvalidArgument(format!(
                "set: cannot write at index {index:?} (dims {dims_debug}, buffer {})",
                if unique { "accessible" } else { "shared" },
            ))
        })?;
        *elem = value;
        Ok(())
    }

    /// Compute the linear buffer offset for a multi-dimensional index.
    ///
    /// Returns `None` if the index is out of bounds.
    fn linear_offset(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.dims.len() {
            return None;
        }
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.dims[i] {
                return None;
            }
        }
        let pos: isize = index.iter().zip(self.strides.iter()).try_fold(
            self.offset,
            |acc, (&idx, &stride)| {
                (idx as isize)
                    .checked_mul(stride)
                    .and_then(|v| acc.checked_add(v))
            },
        )?;
        usize::try_from(pos).ok()
    }
}

#[cfg(test)]
mod tests {
    use crate::{MemoryOrder, Tensor};

    #[test]
    fn get_mut_and_set() {
        let mut t =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        *t.get_mut(&[0, 1]).unwrap() = 99.0;
        assert_eq!(t.get(&[0, 1]), Some(&99.0));

        t.set(&[1, 0], 42.0).unwrap();
        assert_eq!(t.get(&[1, 0]), Some(&42.0));
    }

    #[test]
    fn get_mut_out_of_bounds() {
        let mut t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(t.get_mut(&[2]).is_none());
    }

    #[test]
    fn get_mut_wrong_rank() {
        let mut t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(t.get_mut(&[0, 0]).is_none());
    }

    #[test]
    fn set_shared_buffer_fails() {
        let mut t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let _shared = t.clone(); // refcount == 2
        assert!(t.set(&[0], 99.0).is_err());
    }

    #[test]
    fn set_out_of_bounds_fails() {
        let mut t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(t.set(&[5], 99.0).is_err());
    }

    #[test]
    fn get_and_set_on_view() {
        let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], MemoryOrder::ColumnMajor)
            .unwrap();
        // narrow creates a view sharing the buffer
        let view = t.narrow(0, 1, 2).unwrap();
        assert_eq!(view.get(&[0]), Some(&2.0));
        assert_eq!(view.get(&[1]), Some(&3.0));

        // view is shared, so deep_clone to get exclusive ownership
        let mut owned = view.deep_clone();
        owned.set(&[0], 99.0).unwrap();
        assert_eq!(owned.get(&[0]), Some(&99.0));
        // original unchanged
        assert_eq!(t.get(&[1]), Some(&2.0));
    }

    #[test]
    fn deep_clone_is_independent() {
        let a =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let mut b = a.deep_clone();
        b.set(&[0], 99.0).unwrap();
        assert_eq!(b.get(&[0]), Some(&99.0));
        assert_eq!(a.get(&[0]), Some(&1.0));
    }

    #[test]
    fn deep_clone_empty_tensor() {
        let a = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
        let b = a.deep_clone();
        assert_eq!(b.dims(), &[0]);
        assert_eq!(b.to_vec(), Vec::<f64>::new());
    }
}
