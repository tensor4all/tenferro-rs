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
//! ## Zero-Copy View Operations
//!
//! View operations on [`Tensor`] share the underlying data buffer via `Arc`
//! and only modify metadata (dims, strides, offset). No data is copied.
//!
//! ```ignore
//! // permute: reorder dimensions (zero-copy, strides reordered)
//! let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let transposed = t.permute(&[1, 0]).unwrap();
//! assert_eq!(transposed.dims(), &[3, 2]);
//!
//! // broadcast: expand size-1 dims (zero-copy, stride set to 0)
//! let col = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3, 1],
//!     MemoryOrder::ColumnMajor).unwrap();
//! let expanded = col.broadcast(&[3, 4]).unwrap();
//! assert_eq!(expanded.dims(), &[3, 4]);
//!
//! // diagonal: extract diagonal (zero-copy, strides merged)
//! let sq = Tensor::<f64>::zeros(&[4, 4],
//!     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
//! let diag = sq.diagonal(&[(0, 1)]).unwrap();
//! assert_eq!(diag.dims(), &[4]);
//!
//! // contiguous(): materialize into a contiguous Tensor
//! let owned = transposed.contiguous(MemoryOrder::ColumnMajor);
//! ```

use std::sync::Arc;

use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{
    preferred_compute_devices, ComputeDevice, Error, LogicalMemorySpace, OpKind, Result,
};

// ============================================================================
// Private helpers
// ============================================================================

/// Compute contiguous strides for given dimensions and memory order.
fn compute_contiguous_strides(dims: &[usize], order: MemoryOrder) -> Vec<isize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0isize; ndim];
    match order {
        MemoryOrder::ColumnMajor => {
            strides[0] = 1;
            for i in 1..ndim {
                strides[i] = strides[i - 1] * dims[i - 1] as isize;
            }
        }
        MemoryOrder::RowMajor => {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1] as isize;
            }
        }
    }
    strides
}

/// Check if strides match a contiguous layout for a specific memory order.
fn is_contiguous_in_order(dims: &[usize], strides: &[isize], order: MemoryOrder) -> bool {
    let ndim = dims.len();
    if ndim == 0 {
        return true;
    }
    if dims.contains(&0) {
        return true;
    }
    let expected = compute_contiguous_strides(dims, order);
    for i in 0..ndim {
        if dims[i] > 1 && strides[i] != expected[i] {
            return false;
        }
    }
    true
}

/// Copy elements from strided source to a destination with different strides.
fn copy_strided<T: Copy>(
    src: &[T],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    dst: &mut [T],
    dst_strides: &[isize],
) {
    let ndim = dims.len();
    let n_elements: usize = dims.iter().product();
    if n_elements == 0 {
        return;
    }
    if ndim == 0 {
        dst[0] = src[src_offset as usize];
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..n_elements {
        let src_pos = src_offset
            + index
                .iter()
                .zip(src_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let dst_pos: isize = index
            .iter()
            .zip(dst_strides)
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
        dst[dst_pos as usize] = src[src_pos as usize];

        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

struct StridedInput<'a, T> {
    data: &'a [T],
    strides: &'a [isize],
    offset: isize,
}

/// Element-wise addition of two strided tensors into a contiguous destination.
fn add_strided<T: Copy + std::ops::Add<Output = T>>(
    dims: &[usize],
    a: StridedInput<'_, T>,
    b: StridedInput<'_, T>,
    dst: &mut [T],
    dst_strides: &[isize],
) {
    let ndim = dims.len();
    let n_elements: usize = dims.iter().product();
    if n_elements == 0 {
        return;
    }
    if ndim == 0 {
        dst[0] = a.data[a.offset as usize] + b.data[b.offset as usize];
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..n_elements {
        let a_pos = a.offset
            + index
                .iter()
                .zip(a.strides.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let b_pos = b.offset
            + index
                .iter()
                .zip(b.strides.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let dst_pos: isize = index
            .iter()
            .zip(dst_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
        dst[dst_pos as usize] = a.data[a_pos as usize] + b.data[b_pos as usize];

        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

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
    #[allow(dead_code)]
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

    /// Extract the inner `Vec<T>` if this is the sole Arc owner of a CPU-owned buffer.
    ///
    /// Returns `None` if the Arc is shared, the buffer is externally-owned, or GPU.
    pub fn try_into_vec(self) -> Option<Vec<T>> {
        let inner = Arc::try_unwrap(self.inner).ok()?;
        // Wrap in ManuallyDrop to control when Drop runs.
        // For Owned, the custom Drop impl is a no-op, so skipping it is safe.
        // For External/Gpu we manually invoke drop to fire the release callback.
        let mut md = std::mem::ManuallyDrop::new(inner);
        match &mut *md {
            BufferInner::Owned(v) => {
                // Safety: We hold sole Arc ownership. ptr::read moves the Vec out.
                // ManuallyDrop prevents BufferInner::drop from running (no-op for Owned).
                Some(unsafe { std::ptr::read(v as *const Vec<T>) })
            }
            _ => {
                // Non-CPU-owned: run drop to invoke the release callback.
                // Safety: md is not accessed again after this call.
                unsafe { std::mem::ManuallyDrop::drop(&mut md) };
                None
            }
        }
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
    dims: Arc<[usize]>,
    strides: Arc<[isize]>,
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
    /// Forward-mode tangent (libtorch-aligned).
    ///
    /// When `Some`, this tensor carries a forward-mode gradient used for
    /// JVP (Jacobian-vector product) propagation. Operations that support
    /// forward-mode AD detect this field and propagate tangents automatically.
    /// Single-level only (no nested forward AD).
    fw_grad: Option<Box<Tensor<T>>>,
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
    #[allow(dead_code)]
    inner: CompletionEventInner,
}

#[derive(Clone)]
#[allow(dead_code)]
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
            fw_grad: self.fw_grad.clone(),
        }
    }
}

impl<T: Scalar> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_pending_event = self.event.is_some();
        let has_fw_grad = self.fw_grad.is_some();
        f.debug_struct("Tensor")
            .field("dtype", &std::any::type_name::<T>())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("len", &self.len())
            .field("logical_memory_space", &self.logical_memory_space)
            .field("preferred_compute_device", &self.preferred_compute_device)
            .field("is_contiguous", &self.is_contiguous())
            .field("conjugated", &self.conjugated)
            .field("has_pending_event", &has_pending_event)
            .field("has_fw_grad", &has_fw_grad)
            .finish()
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
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        assert!(
            memory_space == LogicalMemorySpace::MainMemory,
            "GPU memory allocation not yet implemented"
        );
        let n_elements: usize = dims.iter().product();
        let strides = compute_contiguous_strides(dims, order);
        Tensor {
            buffer: DataBuffer::from_vec(vec![T::zero(); n_elements]),
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
            logical_memory_space: memory_space,
            preferred_compute_device: None,
            event: None,
            conjugated: false,
            fw_grad: None,
        }
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
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        assert!(
            memory_space == LogicalMemorySpace::MainMemory,
            "GPU memory allocation not yet implemented"
        );
        let n_elements: usize = dims.iter().product();
        let strides = compute_contiguous_strides(dims, order);
        Tensor {
            buffer: DataBuffer::from_vec(vec![T::one(); n_elements]),
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
            logical_memory_space: memory_space,
            preferred_compute_device: None,
            event: None,
            conjugated: false,
            fw_grad: None,
        }
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
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self> {
        let n_elements: usize = dims.iter().product();
        if data.len() != n_elements {
            return Err(Error::InvalidArgument(format!(
                "data length {} doesn't match dims product {}",
                data.len(),
                n_elements
            )));
        }
        let strides = compute_contiguous_strides(dims, order);
        Ok(Tensor {
            buffer: DataBuffer::from_vec(data.to_vec()),
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
            logical_memory_space: LogicalMemorySpace::MainMemory,
            preferred_compute_device: None,
            event: None,
            conjugated: false,
            fw_grad: None,
        })
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
        data: Vec<T>,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        let ndim = dims.len();
        if strides.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "strides length {} doesn't match dims length {}",
                strides.len(),
                ndim
            )));
        }
        let n_elements: usize = dims.iter().product();
        if n_elements > 0 {
            let mut min_pos = offset;
            let mut max_pos = offset;
            for k in 0..ndim {
                if dims[k] == 0 {
                    continue;
                }
                let extent = (dims[k] - 1) as isize * strides[k];
                if extent >= 0 {
                    max_pos += extent;
                } else {
                    min_pos += extent;
                }
            }
            if min_pos < 0 || max_pos >= data.len() as isize {
                return Err(Error::StrideError(format!(
                    "layout accesses buffer positions {}..={} but buffer length is {}",
                    min_pos,
                    max_pos,
                    data.len()
                )));
            }
        }
        Ok(Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
            logical_memory_space: LogicalMemorySpace::MainMemory,
            preferred_compute_device: None,
            event: None,
            conjugated: false,
            fw_grad: None,
        })
    }

    /// Try to extract the underlying data as `Vec<T>`.
    ///
    /// Returns `Some` only if this is the sole owner of a CPU-owned buffer.
    /// Returns `None` if the buffer is shared, externally-owned, or GPU.
    pub fn try_into_data_vec(self) -> Option<Vec<T>> {
        self.buffer.try_into_vec()
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
    pub fn eye(n: usize, memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
        assert!(
            memory_space == LogicalMemorySpace::MainMemory,
            "GPU memory allocation not yet implemented"
        );
        let dims = [n, n];
        let strides = compute_contiguous_strides(&dims, order);
        let n_elements = n * n;
        let mut data = vec![T::zero(); n_elements];
        for i in 0..n {
            let pos = (i as isize * strides[0] + i as isize * strides[1]) as usize;
            data[pos] = T::one();
        }
        Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: Arc::from(dims.as_slice()),
            strides: Arc::from(strides),
            offset: 0,
            logical_memory_space: memory_space,
            preferred_compute_device: None,
            event: None,
            conjugated: false,
            fw_grad: None,
        }
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
        self.dims.iter().product()
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
        self.len() == 0
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

    /// Returns a reference to the forward-mode tangent, if set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// assert!(t.fw_grad().is_none());
    /// ```
    pub fn fw_grad(&self) -> Option<&Tensor<T>> {
        self.fw_grad.as_deref()
    }

    /// Returns `true` if this tensor carries a forward-mode tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// assert!(!t.has_fw_grad());
    /// ```
    pub fn has_fw_grad(&self) -> bool {
        self.fw_grad.is_some()
    }

    /// Attach a forward-mode tangent to this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let grad = Tensor::<f64>::ones(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// t.set_fw_grad(grad);
    /// assert!(t.has_fw_grad());
    /// ```
    pub fn set_fw_grad(&mut self, grad: Tensor<T>) {
        self.fw_grad = Some(Box::new(grad));
    }

    /// Detach and return the forward-mode tangent, leaving `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, MemoryOrder};
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// t.set_fw_grad(Tensor::<f64>::ones(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor));
    /// let grad = t.detach_fw_grad().unwrap();
    /// assert!(!t.has_fw_grad());
    /// ```
    pub fn detach_fw_grad(&mut self) -> Option<Tensor<T>> {
        self.fw_grad.take().map(|b| *b)
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
        op_kind: OpKind,
    ) -> tenferro_device::Result<Vec<ComputeDevice>> {
        if let Some(device) = self.preferred_compute_device {
            Ok(vec![device])
        } else {
            preferred_compute_devices(self.logical_memory_space, op_kind)
        }
    }

    // ========================================================================
    // View operations (zero-copy, public API waits if pending)
    // ========================================================================

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
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();
        if perm.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "permutation length {} doesn't match ndim {}",
                perm.len(),
                ndim
            )));
        }
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim {
                return Err(Error::InvalidArgument(format!(
                    "permutation index {p} out of range for ndim {ndim}"
                )));
            }
            if seen[p] {
                return Err(Error::InvalidArgument(format!(
                    "duplicate index {p} in permutation"
                )));
            }
            seen[p] = true;
        }
        let new_dims: Arc<[usize]> = perm.iter().map(|&p| self.dims[p]).collect();
        let new_strides: Arc<[isize]> = perm.iter().map(|&p| self.strides[p]).collect();
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: new_dims,
            strides: new_strides,
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();
        if target_dims.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "target dims length {} doesn't match ndim {}",
                target_dims.len(),
                ndim
            )));
        }
        let mut new_strides = self.strides.to_vec();
        for i in 0..ndim {
            if self.dims[i] == target_dims[i] {
                // keep stride
            } else if self.dims[i] == 1 {
                new_strides[i] = 0;
            } else {
                return Err(Error::ShapeMismatch {
                    expected: self.dims.to_vec(),
                    got: target_dims.to_vec(),
                });
            }
        }
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(target_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();
        let mut used = vec![false; ndim];
        let mut diag_dims = Vec::new();
        let mut diag_strides = Vec::new();

        for &(i, j) in axes {
            if i >= ndim || j >= ndim {
                return Err(Error::InvalidArgument(format!(
                    "axis out of range: ({i}, {j}) for tensor with {ndim} dimensions"
                )));
            }
            if i == j {
                return Err(Error::InvalidArgument(format!(
                    "diagonal axes must be distinct, got ({i}, {j})"
                )));
            }
            if used[i] || used[j] {
                return Err(Error::InvalidArgument(format!(
                    "axis {i} or {j} used in multiple diagonal pairs"
                )));
            }
            if self.dims[i] != self.dims[j] {
                return Err(Error::ShapeMismatch {
                    expected: vec![self.dims[i]],
                    got: vec![self.dims[j]],
                });
            }
            used[i] = true;
            used[j] = true;
            diag_dims.push(self.dims[i]);
            diag_strides.push(self.strides[i] + self.strides[j]);
        }

        let mut new_dims = Vec::new();
        let mut new_strides = Vec::new();
        for (k, was_used) in used.iter().enumerate().take(ndim) {
            if !*was_used {
                new_dims.push(self.dims[k]);
                new_strides.push(self.strides[k]);
            }
        }
        new_dims.extend_from_slice(&diag_dims);
        new_strides.extend_from_slice(&diag_strides);

        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>> {
        self.wait();
        let old_len = self.len();
        let new_len: usize = new_dims.iter().product();
        if old_len != new_len {
            return Err(Error::ShapeMismatch {
                expected: self.dims.to_vec(),
                got: new_dims.to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(Error::StrideError(
                "reshape requires contiguous data".into(),
            ));
        }
        let order = if is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor) {
            MemoryOrder::ColumnMajor
        } else {
            MemoryOrder::RowMajor
        };
        let new_strides = compute_contiguous_strides(new_dims, order);
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
    }

    /// Create a zero-copy view with explicit dims and strides.
    ///
    /// Shares the underlying buffer. The caller is responsible for ensuring
    /// that the new layout is valid (all accessed positions are within bounds).
    /// This is verified at construction time.
    ///
    /// # Errors
    ///
    /// Returns an error if the new layout accesses positions outside the buffer.
    pub fn view_as_strided(
        &self,
        new_dims: Vec<usize>,
        new_strides: Vec<isize>,
    ) -> Result<Tensor<T>> {
        self.wait();
        let ndim = new_dims.len();
        if new_strides.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "strides length {} doesn't match dims length {}",
                new_strides.len(),
                ndim
            )));
        }
        // Bounds check
        let n_elements: usize = new_dims.iter().product();
        if n_elements > 0 {
            let buf_len = self.buffer.len();
            let mut min_pos = self.offset;
            let mut max_pos = self.offset;
            for k in 0..ndim {
                if new_dims[k] == 0 {
                    continue;
                }
                let extent = (new_dims[k] - 1) as isize * new_strides[k];
                if extent >= 0 {
                    max_pos += extent;
                } else {
                    min_pos += extent;
                }
            }
            if min_pos < 0 || max_pos >= buf_len as isize {
                return Err(Error::StrideError(format!(
                    "view_as_strided accesses positions {}..={} but buffer length is {}",
                    min_pos, max_pos, buf_len
                )));
            }
        }
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidArgument(format!(
                "dim {dim} out of range for tensor with {ndim} dimensions"
            )));
        }
        if index >= self.dims[dim] {
            return Err(Error::InvalidArgument(format!(
                "index {index} out of range for dimension {dim} with size {}",
                self.dims[dim]
            )));
        }
        let new_offset = self.offset + index as isize * self.strides[dim];
        let mut new_dims = self.dims.to_vec();
        let mut new_strides = self.strides.to_vec();
        new_dims.remove(dim);
        new_strides.remove(dim);
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: new_offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>> {
        self.wait();
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidArgument(format!(
                "dim {dim} out of range for tensor with {ndim} dimensions"
            )));
        }
        if start + length > self.dims[dim] {
            return Err(Error::InvalidArgument(format!(
                "narrow range {}..{} out of bounds for dimension {dim} with size {}",
                start,
                start + length,
                self.dims[dim]
            )));
        }
        let new_offset = self.offset + start as isize * self.strides[dim];
        let mut new_dims = self.dims.to_vec();
        new_dims[dim] = length;
        Ok(Tensor {
            buffer: self.buffer.clone(),
            dims: Arc::from(new_dims),
            strides: self.strides.clone(),
            offset: new_offset,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        })
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
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        self.wait();
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return self.clone();
        }
        let n_elements: usize = self.dims.iter().product();
        let dst_strides = compute_contiguous_strides(&self.dims, order);
        let mut data = vec![T::zero(); n_elements];
        if n_elements > 0 {
            let src = self.buffer.as_slice().expect("CPU-only: contiguous");
            copy_strided(
                src,
                &self.dims,
                &self.strides,
                self.offset,
                &mut data,
                &dst_strides,
            );
        }
        Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: self.dims.clone(),
            strides: Arc::from(dst_strides),
            offset: 0,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        }
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
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T> {
        if is_contiguous_in_order(&self.dims, &self.strides, order) && self.offset == 0 {
            return self;
        }
        self.contiguous(order)
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
        is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor)
            || is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::RowMajor)
    }

    /// Check if the tensor has column-major contiguous layout.
    ///
    /// Returns `true` when data elements are stored in Fortran order:
    /// stride\[0\] = 1, stride\[i\] = stride\[i-1\] * dims\[i-1\].
    pub fn is_col_major_contiguous(&self) -> bool {
        is_contiguous_in_order(&self.dims, &self.strides, MemoryOrder::ColumnMajor)
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
            fw_grad: None,
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
            fw_grad: None,
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
    pub fn tril(&self, diagonal: isize) -> Tensor<T> {
        self.wait();
        let ndim = self.ndim();
        assert!(ndim >= 2, "tril requires at least 2 dimensions");
        let m = self.dims[0];
        let n = self.dims[1];
        let order = MemoryOrder::ColumnMajor;
        let out_strides = compute_contiguous_strides(&self.dims, order);
        let total = self.len();
        let mut data = vec![T::zero(); total];
        let src = self.buffer.as_slice().expect("CPU-only: tril");
        let batch_dims = &self.dims[2..];
        let n_batch: usize = batch_dims.iter().product();
        let n_batch = if n_batch == 0 { 1 } else { n_batch };
        let mut batch_index = vec![0usize; batch_dims.len()];
        for _ in 0..n_batch {
            let src_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(k, &idx)| idx as isize * self.strides[k + 2])
                .sum();
            let dst_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(k, &idx)| idx as isize * out_strides[k + 2])
                .sum();
            for j in 0..n {
                for i in 0..m {
                    if (j as isize - i as isize) <= diagonal {
                        let src_pos = (self.offset
                            + src_batch_off
                            + i as isize * self.strides[0]
                            + j as isize * self.strides[1])
                            as usize;
                        let dst_pos = (dst_batch_off
                            + i as isize * out_strides[0]
                            + j as isize * out_strides[1])
                            as usize;
                        data[dst_pos] = src[src_pos];
                    }
                }
            }
            for d in 0..batch_dims.len() {
                batch_index[d] += 1;
                if batch_index[d] < batch_dims[d] {
                    break;
                }
                batch_index[d] = 0;
            }
        }
        Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: self.dims.clone(),
            strides: Arc::from(out_strides),
            offset: 0,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        }
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
    pub fn triu(&self, diagonal: isize) -> Tensor<T> {
        self.wait();
        let ndim = self.ndim();
        assert!(ndim >= 2, "triu requires at least 2 dimensions");
        let m = self.dims[0];
        let n = self.dims[1];
        let order = MemoryOrder::ColumnMajor;
        let out_strides = compute_contiguous_strides(&self.dims, order);
        let total = self.len();
        let mut data = vec![T::zero(); total];
        let src = self.buffer.as_slice().expect("CPU-only: triu");
        let batch_dims = &self.dims[2..];
        let n_batch: usize = batch_dims.iter().product();
        let n_batch = if n_batch == 0 { 1 } else { n_batch };
        let mut batch_index = vec![0usize; batch_dims.len()];
        for _ in 0..n_batch {
            let src_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(k, &idx)| idx as isize * self.strides[k + 2])
                .sum();
            let dst_batch_off: isize = batch_index
                .iter()
                .enumerate()
                .map(|(k, &idx)| idx as isize * out_strides[k + 2])
                .sum();
            for j in 0..n {
                for i in 0..m {
                    if (j as isize - i as isize) >= diagonal {
                        let src_pos = (self.offset
                            + src_batch_off
                            + i as isize * self.strides[0]
                            + j as isize * self.strides[1])
                            as usize;
                        let dst_pos = (dst_batch_off
                            + i as isize * out_strides[0]
                            + j as isize * out_strides[1])
                            as usize;
                        data[dst_pos] = src[src_pos];
                    }
                }
            }
            for d in 0..batch_dims.len() {
                batch_index[d] += 1;
                if batch_index[d] < batch_dims[d] {
                    break;
                }
                batch_index[d] = 0;
            }
        }
        Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: self.dims.clone(),
            strides: Arc::from(out_strides),
            offset: 0,
            logical_memory_space: self.logical_memory_space,
            preferred_compute_device: self.preferred_compute_device,
            event: None,
            conjugated: self.conjugated,
            fw_grad: None,
        }
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
    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Tensor<T>> {
        if target == self.logical_memory_space {
            return Ok(self.clone());
        }
        Err(Error::DeviceError(
            "GPU memory transfer not yet implemented".into(),
        ))
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
        Tensor::zeros(
            &self.dims,
            self.logical_memory_space,
            MemoryOrder::ColumnMajor,
        )
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
    fn num_elements(&self) -> usize {
        self.len()
    }

    fn seed_cotangent(&self) -> Tensor<T> {
        Tensor::ones(
            &self.dims,
            self.logical_memory_space,
            MemoryOrder::ColumnMajor,
        )
    }

    fn accumulate_tangent(a: Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            a.dims, b.dims,
            "tangent shape mismatch in accumulate_tangent"
        );

        // Capture fw_grad before consuming a's fields
        let a_fw = a.fw_grad().cloned();
        let b_fw = b.fw_grad().cloned();

        let n_elements = a.len();
        let order = MemoryOrder::ColumnMajor;
        let dst_strides = compute_contiguous_strides(&a.dims, order);
        let mut data = vec![T::zero(); n_elements];
        if n_elements > 0 {
            let a_src = a.buffer.as_slice().expect("CPU-only: accumulate_tangent");
            let b_src = b.buffer.as_slice().expect("CPU-only: accumulate_tangent");
            add_strided(
                &a.dims,
                StridedInput {
                    data: a_src,
                    strides: &a.strides,
                    offset: a.offset,
                },
                StridedInput {
                    data: b_src,
                    strides: &b.strides,
                    offset: b.offset,
                },
                &mut data,
                &dst_strides,
            );
        }

        // Propagate fw_grad
        let fw = match (a_fw, b_fw) {
            (Some(fa), Some(fb)) => Some(Self::accumulate_tangent(fa, &fb)),
            (Some(fa), None) => Some(fa),
            (None, Some(fb)) => Some(fb.clone()),
            (None, None) => None,
        };

        let mut result = Tensor {
            buffer: DataBuffer::from_vec(data),
            dims: a.dims.clone(),
            strides: Arc::from(dst_strides),
            offset: 0,
            logical_memory_space: a.logical_memory_space,
            preferred_compute_device: a.preferred_compute_device,
            event: None,
            conjugated: false,
            fw_grad: None,
        };
        if let Some(fg) = fw {
            result.set_fw_grad(fg);
        }
        result
    }
}

// ============================================================================
// PhantomData usage for unused type parameter warning suppression
// ============================================================================

// DataBuffer<T> uses T directly in Vec<T> and *const T, so no PhantomData needed.
// This module-level comment documents the design decision.

#[cfg(test)]
mod tests;
