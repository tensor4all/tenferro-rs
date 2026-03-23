use std::sync::Arc;

use num_traits::{Float, NumCast};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::{compute_contiguous_strides, validate_layout_against_len};
use crate::{DataBuffer, MemoryOrder};

impl<T: Scalar> Tensor<T> {
    fn finish_allocation_result(tensor: Self, memory_space: LogicalMemorySpace) -> Result<Self> {
        if memory_space == LogicalMemorySpace::MainMemory {
            Ok(tensor)
        } else {
            tensor.to_memory_space_async(memory_space)
        }
    }

    fn finish_allocation(tensor: Self, memory_space: LogicalMemorySpace) -> Self {
        Self::finish_allocation_result(tensor, memory_space)
            .unwrap_or_else(|err| panic!("tensor allocation for {memory_space:?} failed: {err}"))
    }

    fn main_memory_contiguous(data: Vec<T>, dims: &[usize], order: MemoryOrder) -> Self {
        Self::from_owned_contiguous_data(
            data,
            Arc::from(dims),
            order,
            LogicalMemorySpace::MainMemory,
            None,
            false,
        )
    }

    fn like_order(reference: &Self) -> MemoryOrder {
        if reference.is_row_major_contiguous() && !reference.is_col_major_contiguous() {
            MemoryOrder::RowMajor
        } else {
            MemoryOrder::ColumnMajor
        }
    }

    fn required_storage_len(dims: &[usize], strides: &[isize], offset: isize) -> Result<usize> {
        if dims.len() != strides.len() {
            return Err(Error::InvalidArgument(format!(
                "empty_strided rank mismatch: dims={} strides={}",
                dims.len(),
                strides.len()
            )));
        }

        if dims.iter().product::<usize>() == 0 {
            return Ok(0);
        }

        let mut min_pos = offset;
        let mut max_pos = offset;
        for (axis, (&dim, &stride)) in dims.iter().zip(strides).enumerate() {
            let extent = isize::try_from(dim - 1)
                .ok()
                .and_then(|d| d.checked_mul(stride))
                .ok_or_else(|| {
                    Error::StrideError(format!(
                        "empty_strided extent overflow for dimension {axis} (size={dim}, stride={stride})"
                    ))
                })?;
            if extent >= 0 {
                max_pos = max_pos.checked_add(extent).ok_or_else(|| {
                    Error::StrideError(format!(
                        "empty_strided maximum offset overflow for dimension {axis}"
                    ))
                })?;
            } else {
                min_pos = min_pos.checked_add(extent).ok_or_else(|| {
                    Error::StrideError(format!(
                        "empty_strided minimum offset overflow for dimension {axis}"
                    ))
                })?;
            }
        }

        if min_pos < 0 {
            return Err(Error::StrideError(format!(
                "empty_strided accesses negative buffer positions {}..={}",
                min_pos, max_pos
            )));
        }

        let max_pos = usize::try_from(max_pos).map_err(|_| {
            Error::StrideError(format!(
                "empty_strided maximum position {max_pos} exceeds usize range"
            ))
        })?;
        max_pos
            .checked_add(1)
            .ok_or_else(|| Error::StrideError("empty_strided storage length overflow".into()))
    }

    /// Create a tensor filled with zeros.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(
    ///     &[3, 4],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// ```
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        let n_elements: usize = dims.iter().product();
        Self::finish_allocation(
            Self::main_memory_contiguous(vec![T::zero(); n_elements], dims, order),
            memory_space,
        )
    }

    /// Create a tensor with allocated storage.
    ///
    /// The current safe implementation initializes the backing storage to
    /// zero, which keeps the constructor deterministic while preserving the
    /// requested layout and device placement.
    ///
    /// The `*_like` family preserves a row-major layout only when the source
    /// tensor is row-major contiguous and not column-major contiguous.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::empty(
    ///     &[3, 4],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// ```
    pub fn empty(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        Self::zeros(dims, memory_space, order)
    }

    /// Create a tensor filled with ones.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::ones(
    ///     &[2, 3],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// ```
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        let n_elements: usize = dims.iter().product();
        Self::finish_allocation(
            Self::main_memory_contiguous(vec![T::one(); n_elements], dims, order),
            memory_space,
        )
    }

    /// Create a tensor with explicit strides.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout would access storage outside the
    /// allocated buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::empty_strided(&[2, 2], &[1, 2], 0, LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(t.strides(), &[1, 2]);
    /// ```
    pub fn empty_strided(
        dims: &[usize],
        strides: &[isize],
        offset: isize,
        memory_space: LogicalMemorySpace,
    ) -> Result<Self> {
        let storage_len = Self::required_storage_len(dims, strides, offset)?;
        let tensor = Self::from_vec(vec![T::zero(); storage_len], dims, strides, offset)?;
        Self::finish_allocation_result(tensor, memory_space)
    }

    /// Create a tensor filled with `value`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::full(
    ///     &[2, 3],
    ///     7.5,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// ```
    pub fn full(
        dims: &[usize],
        value: T,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Self {
        let n_elements: usize = dims.iter().product();
        Self::finish_allocation(
            Self::main_memory_contiguous(vec![value; n_elements], dims, order),
            memory_space,
        )
    }

    /// Create a tensor from a data slice.
    ///
    /// `order` describes how to interpret `data` at the import boundary. View
    /// operations continue to use tenferro's internal column-major semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `dims`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let t = Tensor::<f64>::from_slice(&data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    /// ```
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self> {
        let n_elements: usize = dims.iter().product();
        if data.len() != n_elements {
            return Err(Error::InvalidArgument(format!(
                "data length {} doesn't match dims product {}",
                data.len(),
                n_elements
            )));
        }
        Ok(Self::main_memory_contiguous(data.to_vec(), dims, order))
    }

    /// Create a tensor from an owned `Vec<T>` with explicit layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout is inconsistent with the data length.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &[1, 2], 0).unwrap();
    /// ```
    pub fn from_vec(
        data: Vec<T>,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        validate_layout_against_len(dims, strides, offset, data.len())?;
        Ok(Self::from_parts(
            DataBuffer::from_vec(data),
            Arc::from(dims),
            Arc::from(strides),
            offset,
            LogicalMemorySpace::MainMemory,
            None,
            None,
            false,
            None,
        ))
    }

    /// Create a tensor from externally-owned CPU-accessible memory.
    ///
    /// # Safety
    ///
    /// - `ptr` must remain valid for at least `len` elements until `release` is called.
    /// - The layout described by `dims`, `strides`, and `offset` must stay in bounds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let ptr = data.as_ptr();
    /// let tensor = unsafe {
    ///     Tensor::from_external_parts(ptr, data.len(), &[2, 2], &[1, 2], 0, move || drop(data))
    /// }.unwrap();
    /// assert_eq!(tensor.dims(), &[2, 2]);
    /// ```
    pub unsafe fn from_external_parts(
        ptr: *const T,
        len: usize,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
        release: impl FnOnce() + Send + 'static,
    ) -> Result<Self> {
        validate_layout_against_len(dims, strides, offset, len)?;
        Ok(Self::from_parts(
            DataBuffer::from_external(ptr, len, release),
            Arc::from(dims),
            Arc::from(strides),
            offset,
            LogicalMemorySpace::MainMemory,
            None,
            None,
            false,
            None,
        ))
    }

    /// Try to extract the underlying data as `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let _data = t.try_into_data_vec();
    /// ```
    pub fn try_into_data_vec(self) -> Option<Vec<T>> {
        self.buffer.try_into_vec()
    }

    /// Create a tensor with the same shape and layout convention as another tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let like = base.empty_like();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn empty_like(&self) -> Self {
        Self::empty(
            self.dims(),
            self.logical_memory_space(),
            Self::like_order(self),
        )
    }

    /// Create a zero-filled tensor with the same shape and layout convention as another tensor.
    ///
    /// The `*_like` family preserves a row-major layout only when the source
    /// tensor is row-major contiguous and not column-major contiguous.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let like = base.zeros_like();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn zeros_like(&self) -> Self {
        Self::zeros(
            self.dims(),
            self.logical_memory_space(),
            Self::like_order(self),
        )
    }

    /// Create a one-filled tensor with the same shape and layout convention as another tensor.
    ///
    /// The `*_like` family preserves a row-major layout only when the source
    /// tensor is row-major contiguous and not column-major contiguous.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let like = base.ones_like();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn ones_like(&self) -> Self {
        Self::ones(
            self.dims(),
            self.logical_memory_space(),
            Self::like_order(self),
        )
    }

    /// Create a tensor filled with `value` and matching the shape/layout convention of another tensor.
    ///
    /// The `*_like` family preserves a row-major layout only when the source
    /// tensor is row-major contiguous and not column-major contiguous.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// let like = base.full_like(3.25);
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn full_like(&self, value: T) -> Self {
        Self::full(
            self.dims(),
            value,
            self.logical_memory_space(),
            Self::like_order(self),
        )
    }

    /// Create an identity matrix.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let id = Tensor::<f64>::eye(
    ///     3,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// assert_eq!(id.dims(), &[3, 3]);
    /// ```
    pub fn eye(n: usize, memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        let dims = [n, n];
        let strides = compute_contiguous_strides(&dims, order);
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            let pos = (i as isize)
                .checked_mul(strides[0])
                .and_then(|a| {
                    (i as isize)
                        .checked_mul(strides[1])
                        .and_then(|b| a.checked_add(b))
                })
                .and_then(|pos| usize::try_from(pos).ok())
                .unwrap_or_else(|| {
                    panic!(
                        "eye: position overflow for index {} with strides {:?}",
                        i, strides
                    )
                });
            data[pos] = T::one();
        }
        Self::finish_allocation(
            Self::main_memory_contiguous(data, &dims, order),
            memory_space,
        )
    }
}

impl<T> Tensor<T>
where
    T: Scalar + Float + NumCast,
{
    /// Create a regularly spaced 1-D tensor from `start` toward `end`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let xs = Tensor::<f64>::arange(
    ///     0.0,
    ///     5.0,
    ///     1.0,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// assert_eq!(xs.dims(), &[5]);
    /// ```
    pub fn arange(
        start: T,
        end: T,
        step: T,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
        if step.is_zero() {
            return Err(Error::InvalidArgument(
                "arange: step must be non-zero".into(),
            ));
        }

        let mut data = Vec::new();
        let zero = T::zero();
        if step > zero {
            let mut current = start;
            while current < end {
                data.push(current);
                current = current + step;
            }
        } else {
            let mut current = start;
            while current > end {
                data.push(current);
                current = current + step;
            }
        }

        let dims = [data.len()];
        let tensor = Self::main_memory_contiguous(data, &dims, order);
        Self::finish_allocation_result(tensor, memory_space)
    }

    /// Create a 1-D tensor containing `n_samples` evenly spaced values.
    ///
    /// Returns an error if `n_samples` is negative.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let xs = Tensor::<f64>::linspace(
    ///     0.0,
    ///     1.0,
    ///     5,
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// assert_eq!(xs.dims(), &[5]);
    /// ```
    pub fn linspace(
        start: T,
        end: T,
        n_samples: isize,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
        if n_samples < 0 {
            return Err(Error::InvalidArgument(format!(
                "linspace: steps must be non-negative, got {n_samples}"
            )));
        }

        let n_samples = usize::try_from(n_samples).map_err(|_| {
            Error::InvalidArgument(format!(
                "linspace: steps {n_samples} cannot be represented as usize"
            ))
        })?;
        let mut data = Vec::with_capacity(n_samples);
        match n_samples {
            0 => {}
            1 => data.push(start),
            _ => {
                let Some(denom) = <T as NumCast>::from(n_samples - 1) else {
                    return Err(Error::InvalidArgument(format!(
                        "linspace: sample count {} cannot be represented in target scalar type",
                        n_samples
                    )));
                };
                let step = (end - start) / denom;
                let mut current = start;
                for _ in 0..n_samples {
                    data.push(current);
                    current = current + step;
                }
                if let Some(last) = data.last_mut() {
                    *last = end;
                }
            }
        }

        let dims = [data.len()];
        let tensor = Self::main_memory_contiguous(data, &dims, order);
        Self::finish_allocation_result(tensor, memory_space)
    }
}
