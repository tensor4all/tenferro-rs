//! Dense tensor type with CPU/GPU support.
//!
//! This crate provides [`Tensor<T>`], a multi-dimensional array type composed of
//! shape, strides, and a device-aware [`DataBuffer`]. It supports:
//!
//! - **Zero-copy view operations**: [`Tensor::permute`], [`Tensor::broadcast`],
//!   [`Tensor::diagonal`] modify only metadata (dims/strides)
//! - **Data operations**: [`Tensor::contiguous`] / [`Tensor::into_contiguous`] copy
//!   data into a contiguous layout (the consuming variant avoids allocation when
//!   the tensor is already contiguous)
//! - **strided-rs interop**: [`Tensor::view`] / [`Tensor::view_mut`] produce
//!   [`StridedView`](strided_view::StridedView) /
//!   [`StridedViewMut`](strided_view::StridedViewMut) for use with
//!   [`TensorPrims`](tenferro_prims::TensorPrims) backends
//!
//! # Memory layout
//!
//! [`Tensor`] stores explicit strides and is not tied to any particular memory
//! order. [`MemoryOrder`] is only used as a parameter when allocating new memory
//! (e.g., [`Tensor::zeros`], [`Tensor::contiguous`]).
//!
//! # Examples
//!
//! ## Creating tensors
//!
//! ```ignore
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::Device;
//!
//! // Zeros / ones
//! let a = Tensor::<f64>::zeros(&[3, 4], Device::Cpu, MemoryOrder::ColumnMajor);
//! let b = Tensor::<f64>::ones(&[3, 4], Device::Cpu, MemoryOrder::RowMajor);
//!
//! // From existing data (column-major: Julia convention)
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let m = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();
//! // Logical layout:
//! //   [[1, 3, 5],
//! //    [2, 4, 6]]
//! ```
//!
//! ## Transpose and reshape
//!
//! ```ignore
//! // Transpose a matrix (zero-copy, only strides change)
//! let mt = m.permute(&[1, 0]).unwrap();
//! assert_eq!(mt.dims(), &[3, 2]);
//!
//! // Reshape (requires contiguous data)
//! let flat = m.reshape(&[6]).unwrap();
//! assert_eq!(flat.dims(), &[6]);
//! ```
//!
//! ## Broadcasting
//!
//! ```ignore
//! // Column vector [3,1] broadcast to [3,4] for element-wise ops
//! let col = Tensor::<f64>::ones(&[3, 1], Device::Cpu, MemoryOrder::ColumnMajor);
//! let expanded = col.broadcast(&[3, 4]).unwrap();
//! assert_eq!(expanded.dims(), &[3, 4]);
//! // No data is copied; stride along axis 1 is set to 0
//! ```
//!
//! ## Interop with strided-rs
//!
//! ```ignore
//! // Get a StridedView for use with TensorPrims or strided-kernel
//! let view = a.view();
//! let mut b_mut = b;
//! let view_mut = b_mut.view_mut();
//! ```

use strided_traits::ScalarBase;
use strided_view::{StridedArray, StridedView, StridedViewMut};
use tenferro_device::{Device, Result};

/// Memory ordering for new allocations.
///
/// Specifies how elements are laid out in memory when creating new tensors
/// or copying data into a contiguous buffer. This is **not** stored on the
/// tensor itself — the tensor's [`strides`](Tensor::strides) fully describe
/// the memory layout.
///
/// - [`ColumnMajor`](MemoryOrder::ColumnMajor): First dimension is contiguous
///   (Fortran/Julia convention)
/// - [`RowMajor`](MemoryOrder::RowMajor): Last dimension is contiguous
///   (C/NumPy convention)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// Column-major (Fortran/Julia order). First dimension has stride 1.
    ColumnMajor,
    /// Row-major (C/NumPy order). Last dimension has stride 1.
    RowMajor,
}

/// Owned data buffer, device-aware.
///
/// Wraps the underlying storage for a tensor's data. Currently only CPU
/// storage is supported via [`StridedArray`].
pub enum DataBuffer<T> {
    /// CPU-resident data backed by a [`StridedArray`].
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional dense tensor.
///
/// `Tensor<T>` is the primary data type in tenferro. It owns its data via
/// [`DataBuffer`] and carries shape, strides, and device information.
///
/// ## Zero-copy views
///
/// Operations like [`permute`](Tensor::permute), [`broadcast`](Tensor::broadcast),
/// and [`diagonal`](Tensor::diagonal) return new `Tensor` values that share the
/// same underlying data buffer, modifying only the dims/strides/offset metadata.
///
/// ## strided-rs interop
///
/// Use [`view`](Tensor::view) and [`view_mut`](Tensor::view_mut) to obtain
/// [`StridedView`] / [`StridedViewMut`] references for passing to
/// low-level operations.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    device: Device,
}

impl<T: ScalarBase> Tensor<T> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor (e.g., `&[3, 4]` for a 3×4 matrix)
    /// * `device` — Device on which to allocate the tensor
    /// * `order` — Memory layout for the new allocation
    pub fn zeros(dims: &[usize], device: Device, order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor
    /// * `device` — Device on which to allocate the tensor
    /// * `order` — Memory layout for the new allocation
    pub fn ones(dims: &[usize], device: Device, order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor from a data slice.
    ///
    /// The slice length must equal the product of `dims`.
    /// Data is copied into owned storage with the specified memory order.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `dims`.
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self> {
        todo!()
    }

    /// Create a tensor from an existing [`StridedArray`].
    ///
    /// Takes ownership of the array. The tensor inherits the array's
    /// dims, strides, and offset. Device is set to [`Device::Cpu`].
    pub fn from_strided_array(array: StridedArray<T>) -> Self {
        todo!()
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    /// Returns the shape (size of each dimension).
    pub fn dims(&self) -> &[usize] {
        todo!()
    }

    /// Returns the strides (in units of `T`).
    pub fn strides(&self) -> &[isize] {
        todo!()
    }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        todo!()
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Returns the device on which this tensor resides.
    pub fn device(&self) -> &Device {
        todo!()
    }

    // ========================================================================
    // View operations (zero-copy)
    // ========================================================================

    /// Returns an immutable strided view of the tensor data.
    pub fn view(&self) -> StridedView<'_, T> {
        todo!()
    }

    /// Returns a mutable strided view of the tensor data.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T> {
        todo!()
    }

    /// Permute (reorder) the dimensions of the tensor.
    ///
    /// This is a zero-copy operation that only modifies dims and strides.
    ///
    /// # Arguments
    ///
    /// * `perm` — Permutation of dimension indices (e.g., `&[1, 0]` to transpose)
    ///
    /// # Errors
    ///
    /// Returns an error if `perm` is not a valid permutation of `0..ndim()`.
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Broadcast the tensor to a larger shape.
    ///
    /// Dimensions of size 1 are expanded to the target size (zero-copy via
    /// stride 0). This is a zero-copy metadata operation.
    ///
    /// # Errors
    ///
    /// Returns an error if `target_dims` is incompatible with the current shape.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Extract a diagonal view by merging pairs of axes.
    ///
    /// For each `(axis_i, axis_j)` pair, the two dimensions are replaced
    /// by a single diagonal dimension. This is a zero-copy stride trick.
    ///
    /// # Errors
    ///
    /// Returns an error if any axis is out of range or the paired
    /// dimensions have different sizes.
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    /// Requires contiguous data; returns an error if the tensor is not contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not contiguous or the new shape
    /// has a different total element count.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    // ========================================================================
    // Data operations
    // ========================================================================

    /// Return a contiguous copy of this tensor in the given memory order.
    ///
    /// If the tensor is already contiguous in the requested order,
    /// this may avoid copying (implementation-defined).
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Consume this tensor and return a contiguous version.
    ///
    /// If the tensor is already contiguous in the requested order, returns
    /// `self` without copying or allocating. Otherwise, copies data into a
    /// new contiguous buffer.
    ///
    /// Prefer this over [`contiguous`](Tensor::contiguous) when you no
    /// longer need the original tensor, as it avoids unnecessary allocation
    /// and reference-count overhead.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::Device;
    ///
    /// let a = Tensor::<f64>::zeros(&[3, 4], Device::Cpu, MemoryOrder::ColumnMajor);
    ///
    /// // Transpose creates a non-contiguous view
    /// let at = a.permute(&[1, 0]).unwrap();
    /// assert!(!at.is_contiguous());
    ///
    /// // into_contiguous copies only when necessary
    /// let at_contig = at.into_contiguous(MemoryOrder::ColumnMajor);
    /// assert!(at_contig.is_contiguous());
    ///
    /// // Already contiguous: zero-cost passthrough
    /// let b = Tensor::<f64>::zeros(&[3, 4], Device::Cpu, MemoryOrder::RowMajor);
    /// let b2 = b.into_contiguous(MemoryOrder::RowMajor); // no copy
    /// ```
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Returns `true` if the tensor data is contiguous in memory.
    ///
    /// A tensor is contiguous if its elements occupy a dense block of
    /// memory with no gaps, in either column-major or row-major order.
    pub fn is_contiguous(&self) -> bool {
        todo!()
    }

    /// Return a tensor with complex-conjugated elements.
    ///
    /// For real types (`f32`, `f64`), returns a copy unchanged.
    /// For complex types (`Complex32`, `Complex64`), negates the imaginary part.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use num_complex::Complex64;
    ///
    /// let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)];
    /// let a = Tensor::from_slice(&data, &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let a_conj = a.conj();
    /// // a_conj contains [1.0 - 2.0i, 3.0 + 4.0i]
    /// ```
    pub fn conj(&self) -> Tensor<T> {
        todo!()
    }
}
