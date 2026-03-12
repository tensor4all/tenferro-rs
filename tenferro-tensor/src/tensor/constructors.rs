use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::{compute_contiguous_strides, validate_layout_against_len};
use crate::{DataBuffer, MemoryOrder};

impl<T: Scalar> Tensor<T> {
    fn finish_allocation(tensor: Self, memory_space: LogicalMemorySpace) -> Self {
        if memory_space == LogicalMemorySpace::MainMemory {
            tensor
        } else {
            tensor
                .to_memory_space_async(memory_space)
                .unwrap_or_else(|err| {
                    panic!("tensor allocation for {memory_space:?} failed: {err}")
                })
        }
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

    /// Create a tensor from a data slice.
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
            let pos = (i as isize * strides[0] + i as isize * strides[1]) as usize;
            data[pos] = T::one();
        }
        Self::finish_allocation(
            Self::main_memory_contiguous(data, &dims, order),
            memory_space,
        )
    }
}
