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
//! - **DLPack interop**: [`DataBuffer`] supports both Rust-owned (`Vec<T>`) and
//!   externally-owned memory (e.g., imported via DLPack) with automatic cleanup.
//!
//! # Memory layout
//!
//! [`Tensor`] stores explicit strides and is not tied to any particular memory
//! order. [`MemoryOrder`] is only used as a parameter when allocating new memory
//! (e.g., [`Tensor::zeros`], [`Tensor::contiguous`]).
//!
//! # No strided-rs dependency
//!
//! This crate does **not** depend on `strided-rs`. The strided-rs types
//! (`StridedView`, `StridedViewMut`) are backend implementation details
//! used only in `tenferro-prims`. To pass tensor data to prims backends,
//! use [`DataBuffer::as_slice`] combined with [`Tensor::dims`],
//! [`Tensor::strides`], and [`Tensor::offset`].
//!
//! # Examples
//!
//! ## Creating tensors
//!
//! ```ignore
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! // Zeros / ones
//! let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let b = Tensor::<f64>::ones(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
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
//! let col = Tensor::<f64>::ones(&[3, 1], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let expanded = col.broadcast(&[3, 4]).unwrap();
//! assert_eq!(expanded.dims(), &[3, 4]);
//! // No data is copied; stride along axis 1 is set to 0
//! ```
//!
//! ## TensorView — borrowed, zero-copy views
//!
//! [`TensorView`] is the borrowed counterpart to [`Tensor`], following the
//! `String` / `&str` pattern. View operations modify only metadata
//! (dims, strides, offset) and never copy data.
//!
//! ```ignore
//! // tensor_view() borrows the tensor — no data copy
//! let tv = m.tensor_view();
//! assert_eq!(tv.dims(), m.dims());
//!
//! // permute: reorder dimensions (zero-copy, strides reordered)
//! let tv_t = tv.permute(&[1, 0]).unwrap();
//! assert_eq!(tv_t.dims(), &[3, 2]);
//!
//! // broadcast: expand size-1 dims (zero-copy, stride set to 0)
//! let col = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3, 1],
//!     MemoryOrder::ColumnMajor).unwrap();
//! let col_tv = col.tensor_view();
//! let expanded = col_tv.broadcast(&[3, 4]).unwrap();
//! assert_eq!(expanded.dims(), &[3, 4]);
//!
//! // diagonal: extract diagonal view (zero-copy, strides merged)
//! let sq = Tensor::<f64>::zeros(&[4, 4],
//!     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let sq_tv = sq.tensor_view();
//! let diag = sq_tv.diagonal(&[(0, 1)]).unwrap();
//! assert_eq!(diag.dims(), &[4]);
//!
//! // to_tensor() / contiguous(): materialize a view into owned Tensor
//! let owned = tv_t.to_tensor(MemoryOrder::ColumnMajor);
//! ```

use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{ComputeDevice, LogicalMemorySpace, OpKind, Result};

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

// ============================================================================
// DataBuffer — unified owned/external storage
// ============================================================================

/// Data storage for tensor elements.
///
/// Abstracts over ownership: data may be Rust-owned ([`Vec<T>`]) or
/// externally-owned (e.g., imported via DLPack with a release callback).
/// Shape and stride metadata are NOT stored here — they live on
/// [`Tensor<T>`].
///
/// # Clone behavior
///
/// Cloning an externally-owned buffer performs a **deep copy** into a new
/// Rust-owned `Vec<T>`. The release callback cannot be cloned; the clone
/// is always Rust-owned.
pub struct DataBuffer<T> {
    inner: BufferInner<T>,
}

/// Private ownership representation.
enum BufferInner<T> {
    /// Rust-owned contiguous data.
    Owned(Vec<T>),
    /// Externally-owned data with release callback.
    External {
        ptr: *const T,
        len: usize,
        /// Called on drop to notify the external owner.
        release: Option<Box<dyn FnOnce() + Send>>,
    },
}

// Safety: External buffer pointers are treated as Send/Sync since
// the external framework guarantees the data is valid for the lifetime
// of the DataBuffer. The release callback is Send.
unsafe impl<T: Send> Send for DataBuffer<T> {}
unsafe impl<T: Sync> Sync for DataBuffer<T> {}

impl<T: Copy> Clone for DataBuffer<T> {
    fn clone(&self) -> Self {
        match &self.inner {
            BufferInner::Owned(v) => DataBuffer {
                inner: BufferInner::Owned(v.clone()),
            },
            // Deep copy: can't clone the release callback.
            BufferInner::External { ptr, len, .. } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                DataBuffer {
                    inner: BufferInner::Owned(slice.to_vec()),
                }
            }
        }
    }
}

impl<T> Drop for DataBuffer<T> {
    fn drop(&mut self) {
        if let BufferInner::External { release, .. } = &mut self.inner {
            if let Some(f) = release.take() {
                f();
            }
        }
    }
}

impl<T> DataBuffer<T> {
    /// Create a buffer from an owned `Vec<T>`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(buf.len(), 3);
    /// assert!(buf.is_owned());
    /// ```
    pub fn from_vec(v: Vec<T>) -> Self {
        DataBuffer {
            inner: BufferInner::Owned(v),
        }
    }

    /// Create a buffer from externally-owned data with a release callback.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid, properly aligned allocation of at
    ///   least `len` elements of type `T`.
    /// - The allocation must remain valid until the release callback is invoked
    ///   (which happens when this `DataBuffer` is dropped).
    /// - The release callback must correctly notify the external owner.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let ptr = data.as_ptr();
    /// let len = data.len();
    /// let buf = unsafe {
    ///     DataBuffer::from_external(ptr, len, move || drop(data))
    /// };
    /// assert!(!buf.is_owned());
    /// ```
    pub unsafe fn from_external(
        ptr: *const T,
        len: usize,
        release: impl FnOnce() + Send + 'static,
    ) -> Self {
        DataBuffer {
            inner: BufferInner::External {
                ptr,
                len,
                release: Some(Box::new(release)),
            },
        }
    }

    /// Returns the raw data as a slice.
    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            BufferInner::Owned(v) => v.as_slice(),
            BufferInner::External { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }

    /// Returns the raw data as a mutable slice, if Rust-owned.
    ///
    /// Returns `None` for externally-owned buffers (they are read-only
    /// through tenferro).
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match &mut self.inner {
            BufferInner::Owned(v) => Some(v.as_mut_slice()),
            BufferInner::External { .. } => None,
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        match &self.inner {
            BufferInner::Owned(v) => v.len(),
            BufferInner::External { len, .. } => *len,
        }
    }

    /// Returns `true` if the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the buffer is Rust-owned (backed by `Vec<T>`).
    pub fn is_owned(&self) -> bool {
        matches!(self.inner, BufferInner::Owned(_))
    }

    /// Returns a raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        match &self.inner {
            BufferInner::Owned(v) => v.as_ptr(),
            BufferInner::External { ptr, .. } => *ptr,
        }
    }
}

// ============================================================================
// Tensor<T>
// ============================================================================

/// Multi-dimensional dense tensor.
///
/// `Tensor<T>` is the primary data type in tenferro. It owns its data via
/// [`DataBuffer`] and carries shape, strides, and memory space information.
///
/// ## Zero-copy views
///
/// Operations like [`permute`](Tensor::permute), [`broadcast`](Tensor::broadcast),
/// and [`diagonal`](Tensor::diagonal) return new `Tensor` values that share the
/// same underlying data buffer, modifying only the dims/strides/offset metadata.
///
/// ## Accessing raw data
///
/// Use [`DataBuffer::as_slice`] via [`Tensor::buffer`] combined with
/// [`dims`](Tensor::dims), [`strides`](Tensor::strides), and
/// [`offset`](Tensor::offset) to construct backend-specific views
/// (e.g., `StridedView` in `tenferro-prims`).
///
/// ## GPU async support
///
/// The `event` field tracks pending GPU computation via
/// [`CompletionEvent`]. When a GPU operation produces a tensor, `event`
/// is set to `Some(...)`. Passing this tensor to another GPU operation
/// chains via stream dependencies without CPU synchronization. Methods
/// that access data from CPU call [`wait`](Tensor::wait) internally.
/// For CPU tensors, `event` is always `None` with zero overhead.
///
/// See `tenferro-einsum` crate docs for async chaining examples.
pub struct Tensor<T: Scalar> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    /// The logical memory space where this tensor's data resides.
    logical_memory_space: LogicalMemorySpace,
    /// Optional preferred compute device override.
    preferred_compute_device: Option<ComputeDevice>,
    /// Pending GPU computation event.
    event: Option<CompletionEvent>,
}

/// Borrowed tensor view, lifetime-tied to the source [`Tensor`].
///
/// `TensorView` is the borrowed counterpart to [`Tensor`], following the
/// `String`/`&str` pattern. It references the source tensor's data buffer
/// without copying.
///
/// ## Public vs. internal views
///
/// Public API methods ([`Tensor::tensor_view`], etc.) call
/// [`Tensor::wait`] before constructing a view, so the returned
/// `TensorView` always has `event = None` — data is ready to read.
///
/// The crate-internal `as_operand_view()` skips the wait and
/// propagates the pending event, allowing accelerator operations to chain
/// without CPU synchronization.
pub struct TensorView<'a, T: Scalar> {
    data: &'a DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    /// The logical memory space where the source tensor's data resides.
    logical_memory_space: LogicalMemorySpace,
    /// Optional preferred compute device override from the source tensor.
    preferred_compute_device: Option<ComputeDevice>,
    /// Pending event from the source tensor. Always `None` in public API.
    event: Option<&'a CompletionEvent>,
}

impl<'a, T: Scalar> TensorView<'a, T> {
    /// Returns the shape (size of each dimension).
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides (in units of `T`).
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the logical memory space where the source tensor's data resides.
    pub fn logical_memory_space(&self) -> LogicalMemorySpace {
        self.logical_memory_space
    }

    /// Returns the preferred compute device override, if set.
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.preferred_compute_device
    }

    /// Returns a reference to the underlying data buffer.
    pub fn buffer(&self) -> &DataBuffer<T> {
        self.data
    }

    /// Returns the element offset into the data buffer.
    pub fn offset(&self) -> isize {
        self.offset
    }

    // ========================================================================
    // View operations (zero-copy)
    // ========================================================================

    /// Permute (reorder) the dimensions of this view.
    ///
    /// Returns a new `TensorView` with reordered dims and strides (zero-copy).
    ///
    /// # Errors
    ///
    /// Returns an error if `perm` is not a valid permutation of `0..ndim()`.
    pub fn permute(&self, _perm: &[usize]) -> Result<TensorView<'a, T>> {
        todo!()
    }

    /// Broadcast this view to a larger shape.
    ///
    /// Dimensions of size 1 are expanded to the target size (zero-copy
    /// via stride 0).
    ///
    /// # Errors
    ///
    /// Returns an error if `target_dims` is incompatible with the current shape.
    pub fn broadcast(&self, _target_dims: &[usize]) -> Result<TensorView<'a, T>> {
        todo!()
    }

    /// Extract a diagonal view by merging pairs of axes.
    ///
    /// # Errors
    ///
    /// Returns an error if any axis is out of range or paired dimensions
    /// have different sizes.
    pub fn diagonal(&self, _axes: &[(usize, usize)]) -> Result<TensorView<'a, T>> {
        todo!()
    }

    /// Select a single index along a dimension, removing that dimension.
    ///
    /// Returns a view with `ndim() - 1` dimensions. Zero-copy: adjusts
    /// offset and removes the selected dimension from dims/strides.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim >= ndim()` or `index >= dims()[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(&[3, 4, 10],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let tv = a.tensor_view();
    /// // Select batch index 5 → view of shape [3, 4]
    /// let mat = tv.select(2, 5).unwrap();
    /// assert_eq!(mat.dims(), &[3, 4]);
    /// ```
    pub fn select(&self, _dim: usize, _index: usize) -> Result<TensorView<'a, T>> {
        todo!()
    }

    /// Narrow (slice) a dimension to a sub-range.
    ///
    /// Returns a view with the same number of dimensions, but
    /// `dims()[dim]` reduced to `length`. Zero-copy: only offset and
    /// dim size change.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim >= ndim()` or `start + length > dims()[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(&[3, 10],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let tv = a.tensor_view();
    /// // Take columns 2..5 → view of shape [3, 3]
    /// let sub = tv.narrow(1, 2, 3).unwrap();
    /// assert_eq!(sub.dims(), &[3, 3]);
    /// ```
    pub fn narrow(&self, _dim: usize, _start: usize, _length: usize) -> Result<TensorView<'a, T>> {
        todo!()
    }

    // ========================================================================
    // Materialize (copy data into a new owned Tensor)
    // ========================================================================

    /// Copy this view into an owned [`Tensor`].
    pub fn to_tensor(&self, _order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Return a contiguous copy of this view's data.
    pub fn contiguous(&self, _order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Return a tensor with complex-conjugated elements from this view.
    ///
    /// For real types, returns a copy unchanged.
    pub fn conj(&self) -> Tensor<T>
    where
        T: Conjugate,
    {
        todo!()
    }
}

/// Placeholder for an accelerator synchronization event.
///
/// Tracks completion of asynchronous operations on accelerator devices
/// (GPU, FPGA, etc.), enabling operation chaining without CPU
/// synchronization. Will be replaced with an actual implementation
/// (e.g., CUDA/HIP event handle) when accelerator backends are added.
#[derive(Clone)]
pub struct CompletionEvent {
    _private: (),
}

impl<T: Scalar> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            // Cloned tensor starts with no pending event — the data in the
            // cloned buffer is a snapshot taken after any pending computation
            // completes (clone reads the buffer, which requires completion).
            event: None,
        }
    }
}

impl<T: Scalar> Tensor<T> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor (e.g., `&[3, 4]` for a 3×4 matrix)
    /// * `memory_space` — Logical memory space for the allocation
    /// * `order` — Memory layout for the new allocation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(
    ///     &[3, 4],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
    /// ```
    pub fn zeros(_dims: &[usize], _memory_space: LogicalMemorySpace, _order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor
    /// * `memory_space` — Logical memory space for the allocation
    /// * `order` — Memory layout for the new allocation
    pub fn ones(_dims: &[usize], _memory_space: LogicalMemorySpace, _order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor from a data slice.
    ///
    /// The slice length must equal the product of `dims`.
    /// Data is copied into owned storage with the specified memory order.
    /// Memory space is set to [`LogicalMemorySpace::MainMemory`].
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `dims`.
    pub fn from_slice(_data: &[T], _dims: &[usize], _order: MemoryOrder) -> Result<Self> {
        todo!()
    }

    /// Create a tensor from an owned `Vec<T>` with explicit layout.
    ///
    /// Takes ownership of the data. The caller specifies the dims, strides,
    /// and offset that describe how the data is laid out.
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
    /// // 2×3 column-major: strides [1, 2], offset 0
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let t = Tensor::<f64>::from_vec(data, &[2, 3], &[1, 2], 0).unwrap();
    /// ```
    pub fn from_vec(
        _data: Vec<T>,
        _dims: &[usize],
        _strides: &[isize],
        _offset: isize,
    ) -> Result<Self> {
        todo!()
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    /// Returns the shape (size of each dimension).
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides (in units of `T`).
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the element offset into the data buffer.
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns a reference to the underlying data buffer.
    pub fn buffer(&self) -> &DataBuffer<T> {
        &self.buffer
    }

    /// Returns a mutable reference to the underlying data buffer.
    pub fn buffer_mut(&mut self) -> &mut DataBuffer<T> {
        &mut self.buffer
    }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Returns the logical memory space where this tensor's data resides.
    pub fn logical_memory_space(&self) -> LogicalMemorySpace {
        self.logical_memory_space
    }

    /// Returns the preferred compute device override, if set.
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.preferred_compute_device
    }

    /// Set the preferred compute device override.
    ///
    /// When set, this device will be used for operations on this tensor
    /// instead of the default device selected by
    /// [`preferred_compute_devices`](tenferro_device::preferred_compute_devices).
    /// Pass `None` to clear the override and revert to automatic selection.
    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        self.preferred_compute_device = device;
    }

    /// Return the effective compute devices for a given operation kind.
    ///
    /// If a preferred compute device is set, returns a single-element vector
    /// containing that device. Otherwise, delegates to
    /// [`preferred_compute_devices`](tenferro_device::preferred_compute_devices).
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible compute device is found.
    pub fn effective_compute_devices(
        &self,
        _op_kind: OpKind,
    ) -> tenferro_device::Result<Vec<ComputeDevice>> {
        todo!()
    }

    // ========================================================================
    // View operations (zero-copy, public API waits if pending)
    // ========================================================================

    /// Returns a [`TensorView`] for data inspection.
    ///
    /// Waits for any pending accelerator computation before returning.
    /// The returned view has `event = None` (data is ready to read).
    pub fn tensor_view(&self) -> TensorView<'_, T> {
        self.wait();
        TensorView {
            data: &self.buffer,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
        }
    }

    /// Returns a non-blocking [`TensorView`] that propagates the
    /// pending event (if any) from the source tensor.
    ///
    /// This is an internal API used by `einsum` and other accelerator
    /// operations to chain computations without CPU synchronization.
    pub(crate) fn as_operand_view(&self) -> TensorView<'_, T> {
        TensorView {
            data: &self.buffer,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event.as_ref(),
        }
    }

    /// Permute (reorder) the dimensions of the tensor.
    ///
    /// This is a zero-copy operation that only modifies dims and strides.
    /// Waits for any pending accelerator computation before returning.
    ///
    /// # Arguments
    ///
    /// * `perm` — Permutation of dimension indices (e.g., `&[1, 0]` to transpose)
    ///
    /// # Errors
    ///
    /// Returns an error if `perm` is not a valid permutation of `0..ndim()`.
    pub fn permute(&self, _perm: &[usize]) -> Result<Tensor<T>> {
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
    pub fn broadcast(&self, _target_dims: &[usize]) -> Result<Tensor<T>> {
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
    pub fn diagonal(&self, _axes: &[(usize, usize)]) -> Result<Tensor<T>> {
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
    pub fn reshape(&self, _new_dims: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Select a single index along a dimension, removing that dimension.
    ///
    /// Returns a tensor with `ndim() - 1` dimensions. This is a zero-copy
    /// operation that adjusts the offset and removes the selected dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim >= ndim()` or `index >= dims()[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// // Batched matrices [m, n, batch] = [3, 4, 10]
    /// let a = Tensor::<f64>::zeros(&[3, 4, 10],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// // Select batch index 5 → [3, 4]
    /// let mat = a.select(2, 5).unwrap();
    /// assert_eq!(mat.dims(), &[3, 4]);
    /// ```
    pub fn select(&self, _dim: usize, _index: usize) -> Result<Tensor<T>> {
        todo!()
    }

    /// Narrow (slice) a dimension to a sub-range.
    ///
    /// Returns a tensor with the same number of dimensions, but
    /// `dims()[dim]` reduced to `length`. Zero-copy: only offset and
    /// dim size change.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim >= ndim()` or `start + length > dims()[dim]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(&[3, 10],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// // Take columns 2..5 → [3, 3]
    /// let sub = a.narrow(1, 2, 3).unwrap();
    /// assert_eq!(sub.dims(), &[3, 3]);
    /// ```
    pub fn narrow(&self, _dim: usize, _start: usize, _length: usize) -> Result<Tensor<T>> {
        todo!()
    }

    // ========================================================================
    // Data operations
    // ========================================================================

    /// Return a contiguous copy of this tensor in the given memory order.
    ///
    /// If the tensor is already contiguous in the requested order,
    /// this may avoid copying (implementation-defined).
    pub fn contiguous(&self, _order: MemoryOrder) -> Tensor<T> {
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
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::zeros(
    ///     &[3, 4],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    /// );
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
    /// let b = Tensor::<f64>::zeros(
    ///     &[3, 4],
    ///     LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::RowMajor,
    /// );
    /// let b2 = b.into_contiguous(MemoryOrder::RowMajor); // no copy
    /// ```
    pub fn into_contiguous(self, _order: MemoryOrder) -> Tensor<T> {
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
    pub fn conj(&self) -> Tensor<T>
    where
        T: Conjugate,
    {
        // Conjugation is element-wise and position-independent,
        // so we conjugate the raw buffer directly and preserve layout.
        let conj_data: Vec<T> = self
            .buffer
            .as_slice()
            .iter()
            .copied()
            .map(T::conj)
            .collect();
        Tensor {
            buffer: DataBuffer::from_vec(conj_data),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
        }
    }

    /// Consume this tensor and return one with complex-conjugated elements.
    ///
    /// Like [`conj`](Tensor::conj) but consumes `self`, potentially
    /// reusing the buffer if no other references exist.
    pub fn into_conj(self) -> Tensor<T>
    where
        T: Conjugate,
    {
        todo!()
    }

    /// Asynchronously transfer this tensor to a different memory space.
    ///
    /// Returns a new tensor in the target memory space. If the source
    /// and destination spaces are the same, returns a zero-copy no-op.
    /// Otherwise, data is copied (potentially asynchronously for GPU).
    ///
    /// # Errors
    ///
    /// Returns an error if the transfer is not supported.
    pub fn to_memory_space_async(&self, _target: LogicalMemorySpace) -> Result<Tensor<T>> {
        todo!()
    }

    // ========================================================================
    // GPU async support
    // ========================================================================

    /// Wait for any pending GPU computation to complete.
    ///
    /// No-op for CPU tensors or when GPU computation has already completed.
    /// Methods that access tensor data from CPU call this internally, so
    /// explicit calls are only needed when the caller wants to ensure
    /// completion at a specific point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // GPU einsum returns immediately with pending event
    /// let c = einsum("ij,jk->ik", &[&a_gpu, &b_gpu]).unwrap();
    /// assert!(!c.is_ready());
    ///
    /// // Explicit wait
    /// c.wait();
    /// assert!(c.is_ready());
    ///
    /// // Chaining: implicit sync via stream dependencies, no CPU wait
    /// let d = einsum("ij,jk->ik", &[&c, &e_gpu]).unwrap();
    /// //  → detects c.event → chains on GPU → returns immediately
    /// ```
    pub fn wait(&self) {
        // Currently a no-op: only CPU tensors exist (event is always None).
        // Will synchronize on CompletionEvent when GPU backends are implemented.
    }

    /// Check if tensor data is ready without blocking.
    ///
    /// Returns `true` for CPU tensors (always ready) and for GPU tensors
    /// whose computation has completed. Returns `false` if a GPU operation
    /// is still in progress.
    pub fn is_ready(&self) -> bool {
        self.event.is_none()
    }
}

// ============================================================================
// Differentiable impl — connects Tensor<T> to the generic AD framework
// ============================================================================

impl<T: Scalar> chainrules_core::Differentiable for Tensor<T> {
    type Tangent = Tensor<T>;

    fn zero_tangent(&self) -> Tensor<T> {
        todo!()
    }

    fn accumulate_tangent(_a: Tensor<T>, _b: &Tensor<T>) -> Tensor<T> {
        todo!()
    }
}

// ============================================================================
// PhantomData usage for unused type parameter warning suppression
// ============================================================================

// DataBuffer<T> uses T directly in Vec<T> and *const T, so no PhantomData needed.
// This module-level comment documents the design decision.
