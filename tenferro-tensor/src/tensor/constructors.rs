use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Generator, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::validate_layout_against_len;
use crate::{DataBuffer, MemoryOrder};

impl<T: Scalar> Tensor<T> {
    pub(crate) fn finish_allocation(
        tensor: Self,
        memory_space: LogicalMemorySpace,
    ) -> Result<Self> {
        if memory_space == LogicalMemorySpace::MainMemory {
            Ok(tensor)
        } else {
            tensor.to_memory_space_async(memory_space)
        }
    }

    pub(crate) fn main_memory_contiguous(data: Vec<T>, dims: &[usize], order: MemoryOrder) -> Self {
        Self::from_owned_contiguous_data(
            data,
            Arc::from(dims),
            order,
            LogicalMemorySpace::MainMemory,
            None,
            false,
        )
    }

    /// Layout policy for the `*_like` constructors.
    ///
    /// Row-major is preserved only when the source is row-major contiguous
    /// and not column-major contiguous. Ambiguous or non-contiguous sources
    /// fall back to column-major so the result layout remains deterministic.
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
    /// ).unwrap();
    /// ```
    pub fn zeros(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
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
    /// Ambiguous or non-contiguous inputs fall back to column-major.
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
    /// ).unwrap();
    /// ```
    pub fn empty(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
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
    /// ).unwrap();
    /// ```
    pub fn ones(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
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
        Self::finish_allocation(tensor, memory_space)
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
    /// ).unwrap();
    /// ```
    pub fn full(
        dims: &[usize],
        value: T,
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
    ) -> Result<Self> {
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
    /// ).unwrap();
    /// let like = base.empty_like().unwrap();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn empty_like(&self) -> Result<Self> {
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
    /// Ambiguous or non-contiguous inputs fall back to column-major.
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
    /// ).unwrap();
    /// let like = base.zeros_like().unwrap();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn zeros_like(&self) -> Result<Self> {
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
    /// Ambiguous or non-contiguous inputs fall back to column-major.
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
    /// ).unwrap();
    /// let like = base.ones_like().unwrap();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn ones_like(&self) -> Result<Self> {
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
    /// Ambiguous or non-contiguous inputs fall back to column-major.
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
    /// ).unwrap();
    /// let like = base.full_like(3.25).unwrap();
    /// assert_eq!(like.dims(), base.dims());
    /// ```
    pub fn full_like(&self, value: T) -> Result<Self> {
        Self::full(
            self.dims(),
            value,
            self.logical_memory_space(),
            Self::like_order(self),
        )
    }
}

fn require_generator<'a>(generator: Option<&'a mut Generator>) -> Result<&'a mut Generator> {
    generator.ok_or_else(|| {
        Error::InvalidArgument("random constructors require an explicit Generator".into())
    })
}

fn finish_generated_allocation<T: Scalar>(
    data: Vec<T>,
    dims: &[usize],
    memory_space: LogicalMemorySpace,
    order: MemoryOrder,
) -> Result<Tensor<T>> {
    Tensor::finish_allocation(
        Tensor::main_memory_contiguous(data, dims, order),
        memory_space,
    )
}

impl Tensor<f64> {
    /// Create a tensor filled with uniform samples on `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut generator = Generator::cpu(1234);
    /// let t = Tensor::<f64>::rand(
    ///     &[2, 2],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     Some(&mut generator),
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[2, 2]);
    /// ```
    pub fn rand(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        let generator = require_generator(generator)?;
        let n_elements: usize = dims.iter().product();
        let mut data = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            data.push(generator.sample_uniform_f64());
        }
        finish_generated_allocation(data, dims, memory_space, order)
    }

    /// Create a tensor filled with standard-normal samples.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut generator = Generator::cpu(1234);
    /// let t = Tensor::<f64>::randn(
    ///     &[4],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     Some(&mut generator),
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[4]);
    /// ```
    pub fn randn(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        let generator = require_generator(generator)?;
        let n_elements: usize = dims.iter().product();
        let mut data = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            data.push(generator.sample_standard_normal_f64());
        }
        finish_generated_allocation(data, dims, memory_space, order)
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// uniform samples on `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::RowMajor,
    /// ).unwrap();
    /// let mut generator = Generator::cpu(1234);
    /// let t = base.rand_like(Some(&mut generator)).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn rand_like(reference: &Self, generator: Option<&mut Generator>) -> Result<Self> {
        Self::rand(
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// standard-normal samples.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<f64>::zeros(
    ///     &[2, 3],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::RowMajor,
    /// ).unwrap();
    /// let mut generator = Generator::cpu(1234);
    /// let t = base.randn_like(Some(&mut generator)).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn randn_like(reference: &Self, generator: Option<&mut Generator>) -> Result<Self> {
        Self::randn(
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }
}

impl Tensor<i32> {
    /// Create a tensor filled with integer samples in `[low, high)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let mut generator = Generator::cpu(1234);
    /// let t = Tensor::<i32>::randint(
    ///     -2,
    ///     5,
    ///     &[2, 2],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     Some(&mut generator),
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[2, 2]);
    /// ```
    pub fn randint(
        low: i32,
        high: i32,
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        let generator = require_generator(generator)?;
        let n_elements: usize = dims.iter().product();
        let mut data = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            data.push(generator.sample_integer_i32(low, high)?);
        }
        finish_generated_allocation(data, dims, memory_space, order)
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// integer samples in `[low, high)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::Generator;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<i32>::zeros(
    ///     &[2, 3],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// ).unwrap();
    /// let mut generator = Generator::cpu(1234);
    /// let t = base.randint_like(-2, 5, Some(&mut generator)).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn randint_like(
        reference: &Self,
        low: i32,
        high: i32,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        Self::randint(
            low,
            high,
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }
}
