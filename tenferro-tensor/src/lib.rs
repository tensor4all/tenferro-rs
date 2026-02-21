//! Dense tensor type with CPU/GPU support.
//!
//! This crate provides [`Tensor<T>`], a multi-dimensional array type composed of
//! shape, strides, and a device-aware [`DataBuffer`]. It supports:
//!
//! - **Zero-copy view operations**: [`Tensor::permute`], [`Tensor::broadcast`],
//!   [`Tensor::diagonal`], [`Tensor::select`], [`Tensor::narrow`] modify only
//!   metadata (dims/strides)
//! - **Data operations**: [`Tensor::contiguous`] / [`Tensor::into_contiguous`] copy
//!   data into a contiguous layout (the consuming variant avoids allocation when
//!   the tensor is already contiguous); [`Tensor::tril`] / [`Tensor::triu`] extract
//!   triangular parts
//! - **Factory functions**: [`Tensor::zeros`], [`Tensor::ones`], [`Tensor::eye`]
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

use std::sync::Arc;

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
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::MemoryOrder;
///
/// let order = MemoryOrder::RowMajor;
/// ```
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
/// externally-owned (e.g., imported via DLPack with a release callback),
/// or GPU device memory. Shape and stride metadata are NOT stored here
/// — they live on [`Tensor<T>`].
///
/// # Shared ownership (Arc)
///
/// `DataBuffer` wraps the internal storage in `Arc`, enabling shallow
/// clone (reference count increment). This follows PyTorch's pattern:
/// - `clone()` on `Tensor` is shallow (shared buffer, O(1))
/// - `conj()` is lazy (shared buffer + flag flip, O(1))
/// - Deep copy (actual data duplication) uses prims `Permute(identity)`
///   or dedicated operations
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::DataBuffer;
///
/// let buf = DataBuffer::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(buf.len(), 3);
/// ```
pub struct DataBuffer<T> {
    inner: Arc<BufferInner<T>>,
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
    /// GPU device memory (CUDA or ROCm).
    ///
    /// The pointer is a device pointer — it MUST NOT be dereferenced from
    /// the CPU. It is only valid as an argument to GPU API calls (cuTENSOR,
    /// hipTENSOR, cudaMemcpy, etc.).
    ///
    Gpu {
        /// Device pointer (NOT dereferenceable from CPU).
        device_ptr: *mut T,
        /// Number of elements.
        len: usize,
        /// Memory space identifying which GPU device owns this buffer.
        space: LogicalMemorySpace,
        /// Called on drop to free GPU memory (e.g., cudaFree / hipFree).
        release: Option<Box<dyn FnOnce() + Send>>,
    },
}

// Safety: External buffer pointers are treated as Send/Sync since
// the external framework guarantees the data is valid for the lifetime
// of the DataBuffer. The release callback is Send.
unsafe impl<T: Send> Send for DataBuffer<T> {}
unsafe impl<T: Sync> Sync for DataBuffer<T> {}

impl<T> Clone for DataBuffer<T> {
    /// Shallow clone: increments the `Arc` reference count.
    ///
    /// No data is copied. Multiple `Tensor` values can share the same
    /// underlying buffer. This matches PyTorch's semantics where
    /// `tensor.clone()` is a metadata-level operation.
    fn clone(&self) -> Self {
        DataBuffer {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for BufferInner<T> {
    fn drop(&mut self) {
        match self {
            BufferInner::External { release, .. } | BufferInner::Gpu { release, .. } => {
                if let Some(f) = release.take() {
                    f();
                }
            }
            BufferInner::Owned(_) => {}
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
            inner: Arc::new(BufferInner::Owned(v)),
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
            inner: Arc::new(BufferInner::External {
                ptr,
                len,
                release: Some(Box::new(release)),
            }),
        }
    }

    /// Returns the raw data as a slice (CPU buffers only).
    ///
    /// Returns `None` for GPU buffers — device pointers are not
    /// dereferenceable from the CPU. Use [`as_device_ptr`](DataBuffer::as_device_ptr)
    /// to obtain a GPU device pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(buf.as_slice(), Some(&[1.0, 2.0, 3.0][..]));
    /// ```
    pub fn as_slice(&self) -> Option<&[T]> {
        match &*self.inner {
            BufferInner::Owned(v) => Some(v.as_slice()),
            BufferInner::External { ptr, len, .. } => {
                Some(unsafe { std::slice::from_raw_parts(*ptr, *len) })
            }
            BufferInner::Gpu { .. } => None,
        }
    }

    /// Returns the raw data as a mutable slice, if Rust-owned and uniquely held.
    ///
    /// Returns `None` if the buffer is shared (Arc refcount > 1),
    /// externally-owned, or GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let mut buf = DataBuffer::from_vec(vec![1.0, 2.0]);
    /// if let Some(slice) = buf.as_mut_slice() {
    ///     slice[0] = 42.0;
    /// }
    /// assert_eq!(buf.as_slice().unwrap()[0], 42.0);
    /// ```
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match Arc::get_mut(&mut self.inner)? {
            BufferInner::Owned(v) => Some(v.as_mut_slice()),
            BufferInner::External { .. } | BufferInner::Gpu { .. } => None,
        }
    }

    /// Returns the number of elements in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(buf.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match &*self.inner {
            BufferInner::Owned(v) => v.len(),
            BufferInner::External { len, .. } | BufferInner::Gpu { len, .. } => *len,
        }
    }

    /// Returns `true` if the buffer has no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::<f64>::from_vec(vec![]);
    /// assert!(buf.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the buffer is Rust-owned (backed by `Vec<T>`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.is_owned());
    /// ```
    pub fn is_owned(&self) -> bool {
        matches!(&*self.inner, BufferInner::Owned(_))
    }

    /// Returns `true` if the buffer resides on GPU device memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(!buf.is_gpu());
    /// ```
    pub fn is_gpu(&self) -> bool {
        matches!(&*self.inner, BufferInner::Gpu { .. })
    }

    /// Returns `true` if this is the only reference to the underlying buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.is_unique());
    /// let buf2 = buf.clone(); // Arc clone
    /// assert!(!buf.is_unique());
    /// ```
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Returns a raw CPU pointer to the data, or `None` for GPU buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.as_ptr().is_some());
    /// ```
    pub fn as_ptr(&self) -> Option<*const T> {
        match &*self.inner {
            BufferInner::Owned(v) => Some(v.as_ptr()),
            BufferInner::External { ptr, .. } => Some(*ptr),
            BufferInner::Gpu { .. } => None,
        }
    }

    /// Returns the GPU device pointer, or `None` for CPU buffers.
    ///
    /// The returned pointer is a GPU device pointer — it MUST NOT be
    /// dereferenced from the CPU. It is only valid as an argument to
    /// GPU API calls (cuTENSOR, hipTENSOR, cudaMemcpy, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.as_device_ptr().is_none()); // CPU buffer
    /// ```
    pub fn as_device_ptr(&self) -> Option<*const T> {
        match &*self.inner {
            BufferInner::Gpu { device_ptr, .. } => Some(*device_ptr as *const T),
            _ => None,
        }
    }

    /// Returns the logical memory space of a GPU buffer, or `None` for CPU buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.gpu_memory_space().is_none()); // CPU buffer
    /// ```
    pub fn gpu_memory_space(&self) -> Option<LogicalMemorySpace> {
        match &*self.inner {
            BufferInner::Gpu { space, .. } => Some(*space),
            _ => None,
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
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::Tensor;
///
/// let t = Tensor::<f64>::zeros(&[2, 3]);
/// assert_eq!(t.dims(), &[2, 3]);
/// assert_eq!(t.len(), 6);
/// ```
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
    /// Lazy conjugation flag.
    ///
    /// When `true`, the tensor's elements are logically conjugated without
    /// materializing a copy. GPU backends (cuTENSOR/hipTENSOR) support this
    /// natively via `CUTENSOR_OP_CONJ` / `HIPTENSOR_OP_CONJ` in tensor
    /// descriptors. CPU backends apply conjugation during execution.
    conjugated: bool,
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
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::Tensor;
///
/// let t = Tensor::<f64>::zeros(&[2, 3]);
/// let view = t.tensor_view();
/// assert_eq!(view.ndim(), 2);
/// ```
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
    /// Lazy conjugation flag from the source tensor.
    conjugated: bool,
}

impl<'a, T: Scalar> TensorView<'a, T> {
    /// Returns the shape (size of each dimension).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let view = t.tensor_view();
    /// assert_eq!(view.dims(), &[3, 4]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides (in units of `T`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// let strides = view.strides();
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the number of dimensions (rank).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the logical memory space where the source tensor's data resides.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let view = tensor.tensor_view();
    /// assert_eq!(view.logical_memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn logical_memory_space(&self) -> LogicalMemorySpace {
        self.logical_memory_space
    }

    /// Returns the preferred compute device override, if set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// assert!(view.preferred_compute_device().is_none());
    /// ```
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.preferred_compute_device
    }

    /// Returns a reference to the underlying data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// let buf = view.buffer();
    /// ```
    pub fn buffer(&self) -> &DataBuffer<T> {
        self.data
    }

    /// Returns the element offset into the data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// assert_eq!(view.offset(), 0);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns `true` if the source tensor is logically conjugated (lazy).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// assert!(!view.is_conjugated());
    /// ```
    pub fn is_conjugated(&self) -> bool {
        self.conjugated
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view(); // shape [3, 4]
    /// let transposed = view.permute(&[1, 0]).unwrap(); // shape [4, 3]
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view(); // shape [1, 4]
    /// let expanded = view.broadcast(&[3, 4]).unwrap(); // shape [3, 4]
    /// ```
    pub fn broadcast(&self, _target_dims: &[usize]) -> Result<TensorView<'a, T>> {
        todo!()
    }

    /// Extract a diagonal view by merging pairs of axes.
    ///
    /// # Errors
    ///
    /// Returns an error if any axis is out of range or paired dimensions
    /// have different sizes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view(); // shape [3, 3]
    /// let diag = view.diagonal(&[(0, 1)]).unwrap(); // shape [3]
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// let owned = view.to_tensor(MemoryOrder::ColumnMajor);
    /// ```
    pub fn to_tensor(&self, _order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Return a contiguous copy of this view's data.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// let contig = view.contiguous(MemoryOrder::ColumnMajor);
    /// ```
    pub fn contiguous(&self, _order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Return a view with the conjugated flag toggled (lazy, zero-cost).
    ///
    /// Does not copy data. The conjugation is applied by backends
    /// when this view is used in operations.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = tensor.tensor_view();
    /// let conjugated = view.conj();
    /// assert!(conjugated.is_conjugated());
    /// ```
    pub fn conj(&self) -> TensorView<'a, T> {
        TensorView {
            data: self.data,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event,
            conjugated: !self.conjugated,
        }
    }
}

/// Synchronization event for asynchronous accelerator operations.
///
/// Tracks completion of asynchronous operations on accelerator devices,
/// enabling operation chaining without CPU synchronization.
///
/// - `Noop`: no pending operation (used by CPU tensors).
/// - `Cuda`: wraps a CUDA event handle from `cudaEventCreate`.
/// - `Rocm`: wraps a HIP event handle from `hipEventCreate`.
///
/// GPU event handles are opaque pointers — the actual synchronization
/// (cudaEventSynchronize / hipEventSynchronize) will be implemented
/// when GPU backends are added.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::CompletionEvent;
///
/// // CompletionEvent is typically created by GPU backends.
/// // CPU tensors use event = None (no pending operation).
/// ```
#[derive(Clone)]
pub struct CompletionEvent {
    inner: CompletionEventInner,
}

#[derive(Clone)]
enum CompletionEventInner {
    /// No pending operation.
    Noop,
    /// CUDA event handle (cudaEvent_t).
    Cuda {
        /// Opaque CUDA event handle.
        _event: *mut std::ffi::c_void,
    },
    /// ROCm/HIP event handle (hipEvent_t).
    Rocm {
        /// Opaque HIP event handle.
        _event: *mut std::ffi::c_void,
    },
}

// Safety: Event handles are only used by GPU API calls and do not
// dereference the pointer from the CPU. The GPU runtime guarantees
// thread safety of event queries.
unsafe impl Send for CompletionEvent {}
unsafe impl Sync for CompletionEvent {}

impl<T: Scalar> Clone for Tensor<T> {
    /// Shallow clone: shares the underlying data buffer (Arc refcount++).
    ///
    /// No data is copied. The cloned tensor references the same buffer
    /// with the same metadata. This matches PyTorch's `Tensor` clone
    /// semantics.
    ///
    /// For a deep copy (actual data duplication), use prims operations
    /// such as `Permute(identity)` or `MakeContiguous`.
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(), // Arc refcount++, O(1)
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event.clone(),
            conjugated: self.conjugated,
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::ones(&[2, 3],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let t = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();
    /// ```
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

    /// Create an identity matrix.
    ///
    /// Returns a 2D tensor of shape `[n, n]` with ones on the diagonal
    /// and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let id = Tensor::<f64>::eye(3,
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// assert_eq!(id.dims(), &[3, 3]);
    /// ```
    pub fn eye(_n: usize, _memory_space: LogicalMemorySpace, _order: MemoryOrder) -> Self {
        todo!()
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    /// Returns the shape (size of each dimension).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert_eq!(t.dims(), &[3, 4]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides (in units of `T`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// let strides = t.strides();
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the element offset into the data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert_eq!(t.offset(), 0);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns a reference to the underlying data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// let buf = t.buffer();
    /// ```
    pub fn buffer(&self) -> &DataBuffer<T> {
        &self.buffer
    }

    /// Returns a mutable reference to the underlying data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// let buf = t.buffer_mut();
    /// ```
    pub fn buffer_mut(&mut self) -> &mut DataBuffer<T> {
        &mut self.buffer
    }

    /// Returns the number of dimensions (rank).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert_eq!(t.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert_eq!(t.len(), 12);
    /// ```
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[0, 4], mem, col);
    /// assert!(t.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Returns the logical memory space where this tensor's data resides.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, col);
    /// assert_eq!(t.logical_memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn logical_memory_space(&self) -> LogicalMemorySpace {
        self.logical_memory_space
    }

    /// Returns the preferred compute device override, if set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert!(t.preferred_compute_device().is_none());
    /// ```
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.preferred_compute_device
    }

    /// Set the preferred compute device override.
    ///
    /// When set, this device will be used for operations on this tensor
    /// instead of the default device selected by
    /// [`preferred_compute_devices`](tenferro_device::preferred_compute_devices).
    /// Pass `None` to clear the override and revert to automatic selection.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::ComputeDevice;
    ///
    /// let mut t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// t.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    /// ```
    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        self.preferred_compute_device = device;
    }

    /// Returns `true` if this tensor is logically conjugated (lazy).
    ///
    /// GPU backends (cuTENSOR/hipTENSOR) read this flag when building
    /// tensor descriptors to set `CUTENSOR_OP_CONJ` / `HIPTENSOR_OP_CONJ`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// assert!(!t.is_conjugated());
    /// ```
    pub fn is_conjugated(&self) -> bool {
        self.conjugated
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::OpKind;
    ///
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col);
    /// let devices = t.effective_compute_devices(OpKind::BatchedGemm).unwrap();
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let view = t.tensor_view();
    /// assert_eq!(view.ndim(), 2);
    /// ```
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
            conjugated: self.conjugated,
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
            conjugated: self.conjugated,
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[3, 4], mem, col); // [3, 4]
    /// let transposed = t.permute(&[1, 0]).unwrap();    // [4, 3]
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[1, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let b = t.broadcast(&[4, 3]).unwrap();
    /// assert_eq!(b.dims(), &[4, 3]);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[3, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// // Extract the main diagonal by merging axes 0 and 1
    /// let d = t.diagonal(&[(0, 1)]).unwrap();
    /// assert_eq!(d.dims(), &[3]);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let r = t.reshape(&[6]).unwrap();
    /// assert_eq!(r.dims(), &[6]);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
    /// let c = t.contiguous(MemoryOrder::RowMajor);
    /// assert!(c.is_contiguous());
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3]);
    /// assert!(t.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        todo!()
    }

    /// Return a lazily-conjugated tensor (shared buffer, flag flip).
    ///
    /// No data is copied. The returned tensor shares the same underlying
    /// buffer (via Arc) with the `conjugated` flag toggled. This matches
    /// PyTorch's `torch.conj()` semantics: always lazy, both CPU and GPU.
    ///
    /// Backends apply conjugation implicitly:
    /// - **GPU**: `CUTENSOR_OP_CONJ` / `HIPTENSOR_OP_CONJ` in tensor descriptors
    /// - **CPU**: conjugation applied during computation kernels
    ///
    /// To materialize the conjugation into a new buffer, use
    /// `Backend::resolve_conj()` from `tenferro-prims`.
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
    /// assert!(a_conj.is_conjugated());
    /// // Data is NOT copied — shared buffer with flipped flag
    /// let a_conj2 = a_conj.conj();
    /// assert!(!a_conj2.is_conjugated()); // double conj cancels out
    /// ```
    pub fn conj(&self) -> Tensor<T>
    where
        T: Conjugate,
    {
        Tensor {
            buffer: self.buffer.clone(), // Arc refcount++, O(1)
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event.clone(),
            conjugated: !self.conjugated,
        }
    }

    /// Consume this tensor and return a lazily-conjugated version.
    ///
    /// Like [`conj`](Tensor::conj) but consumes `self`, avoiding the
    /// Arc refcount increment when the original is no longer needed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3]);
    /// let tc = t.into_conj();
    /// assert!(tc.is_conjugated());
    /// ```
    pub fn into_conj(self) -> Tensor<T>
    where
        T: Conjugate,
    {
        Tensor {
            buffer: self.buffer,
            dims: self.dims,
            strides: self.strides,
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: self.event,
            conjugated: !self.conjugated,
        }
    }

    /// Extract the lower triangular part of a matrix.
    ///
    /// Returns a new tensor with elements above the `diagonal`-th diagonal
    /// set to zero. For batched tensors `(m, n, *)`, applies independently
    /// to each batch element.
    ///
    /// - `diagonal = 0`: main diagonal (default)
    /// - `diagonal > 0`: above main diagonal
    /// - `diagonal < 0`: below main diagonal
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::ones(&[3, 3],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let lower = a.tril(0);
    /// // [[1, 0, 0],
    /// //  [1, 1, 0],
    /// //  [1, 1, 1]]
    /// ```
    pub fn tril(&self, _diagonal: isize) -> Tensor<T> {
        todo!()
    }

    /// Extract the upper triangular part of a matrix.
    ///
    /// Returns a new tensor with elements below the `diagonal`-th diagonal
    /// set to zero. For batched tensors `(m, n, *)`, applies independently
    /// to each batch element.
    ///
    /// - `diagonal = 0`: main diagonal (default)
    /// - `diagonal > 0`: above main diagonal
    /// - `diagonal < 0`: below main diagonal
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let a = Tensor::<f64>::ones(&[3, 3],
    ///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let upper = a.triu(0);
    /// // [[1, 1, 1],
    /// //  [0, 1, 1],
    /// //  [0, 0, 1]]
    /// ```
    pub fn triu(&self, _diagonal: isize) -> Tensor<T> {
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3]);
    /// let t2 = t.to_memory_space_async(LogicalMemorySpace::MainMemory).unwrap();
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3]);
    /// // CPU tensors are always ready.
    /// assert!(t.is_ready());
    /// ```
    pub fn is_ready(&self) -> bool {
        self.event.is_none()
    }
}

// ============================================================================
// Differentiable impl — connects Tensor<T> to the generic AD framework
// ============================================================================

impl<T: Scalar> chainrules_core::Differentiable for Tensor<T> {
    type Tangent = Tensor<T>;

    /// Returns a zero tangent tensor with the same shape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    /// use chainrules_core::Differentiable;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3]);
    /// let zt = t.zero_tangent();
    /// ```
    fn zero_tangent(&self) -> Tensor<T> {
        todo!()
    }

    /// Accumulates a tangent into this tensor (in-place addition).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::Tensor;
    /// use chainrules_core::Differentiable;
    ///
    /// let mut t = Tensor::<f64>::zeros(&[2, 3]);
    /// let tangent = Tensor::<f64>::zeros(&[2, 3]);
    /// t.accumulate_tangent(&tangent);
    /// ```
    fn accumulate_tangent(_a: Tensor<T>, _b: &Tensor<T>) -> Tensor<T> {
        todo!()
    }
}

// ============================================================================
// PhantomData usage for unused type parameter warning suppression
// ============================================================================

// DataBuffer<T> uses T directly in Vec<T> and *const T, so no PhantomData needed.
// This module-level comment documents the design decision.
