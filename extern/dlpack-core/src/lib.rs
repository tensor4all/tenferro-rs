//! Lightweight DLPack-compatible tensor and matrix containers.
//!
//! This crate provides minimal, GPU-aware data containers for N-dimensional
//! tensors and 2D matrices. The design follows the [DLPack v1.0](https://github.com/dmlc/dlpack)
//! device model, enabling zero-copy interop across frameworks and devices
//! (CPU, CUDA, ROCm, etc.).
//!
//! # Key types
//!
//! - [`DeviceType`] / [`DLDevice`] — DLPack-compatible device identification
//! - [`Matrix`] / [`MatrixView`] / [`MatrixViewMut`] — Owned / borrowed 2D containers
//! - [`Tensor`] / [`TensorView`] / [`TensorViewMut`] — Owned / borrowed N-dimensional containers
//! - [`Alloc`] — Device-aware memory allocator trait
//!
//! # Design
//!
//! Owned types ([`Matrix`], [`Tensor`]) carry a deleter callback that frees
//! memory on drop, following the `DLManagedTensor` pattern. This allows a
//! single type to own CPU heap memory, CUDA device memory, or any other
//! device-specific allocation.
//!
//! View types ([`MatrixView`], [`TensorView`]) are borrowed references with
//! no ownership — they carry a pointer, shape, strides, and device info.
//!
//! # Examples
//!
//! ## CPU matrix from a Vec
//!
//! ```
//! use dlpack_core::{Matrix, DLDevice, DeviceType};
//!
//! // 3×4 column-major matrix on CPU
//! let data = vec![0.0f64; 12];
//! let mat = Matrix::from_vec(data, 3, 4, [1, 3]);
//! assert_eq!(mat.nrows(), 3);
//! assert_eq!(mat.ncols(), 4);
//! assert_eq!(mat.device().device_type, DeviceType::Cpu);
//! ```
//!
//! ## Borrowing a view
//!
//! ```
//! use dlpack_core::{Matrix, MatrixView};
//!
//! let mat = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, [1, 2]);
//! let view = mat.as_view();
//! assert_eq!(view.nrows(), 2);
//! assert_eq!(view.ncols(), 3);
//! ```

use std::marker::PhantomData;

// ============================================================================
// Device types (DLPack v1.0 compatible)
// ============================================================================

/// DLPack-compatible device type identifier.
///
/// Values match the DLPack v1.0 specification (`DLDeviceType` enum).
///
/// # Examples
///
/// ```
/// use dlpack_core::DeviceType;
///
/// assert_eq!(DeviceType::Cpu as u32, 1);
/// assert_eq!(DeviceType::Cuda as u32, 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DeviceType {
    /// CPU (host) memory.
    Cpu = 1,
    /// NVIDIA CUDA device memory.
    Cuda = 2,
    /// CUDA pinned (page-locked host) memory.
    CudaHost = 3,
    /// AMD ROCm device memory.
    Rocm = 10,
    /// ROCm pinned host memory.
    RocmHost = 11,
    /// CUDA managed (unified) memory.
    CudaManaged = 13,
}

/// DLPack-compatible device descriptor.
///
/// Identifies a specific device by type and ordinal index.
///
/// # Examples
///
/// ```
/// use dlpack_core::{DLDevice, DeviceType};
///
/// let cpu = DLDevice::cpu();
/// assert_eq!(cpu.device_type, DeviceType::Cpu);
/// assert_eq!(cpu.device_id, 0);
///
/// let gpu = DLDevice::new(DeviceType::Cuda, 0);
/// assert_eq!(gpu.device_type, DeviceType::Cuda);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLDevice {
    /// Device type (CPU, CUDA, ROCm, etc.).
    pub device_type: DeviceType,
    /// Device ordinal (0 for single-device systems).
    pub device_id: i32,
}

impl DLDevice {
    /// Creates a new device descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{DLDevice, DeviceType};
    ///
    /// let dev = DLDevice::new(DeviceType::Cuda, 1);
    /// assert_eq!(dev.device_id, 1);
    /// ```
    pub fn new(device_type: DeviceType, device_id: i32) -> Self {
        Self {
            device_type,
            device_id,
        }
    }

    /// Returns the CPU device (id=0).
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{DLDevice, DeviceType};
    ///
    /// let cpu = DLDevice::cpu();
    /// assert_eq!(cpu.device_type, DeviceType::Cpu);
    /// ```
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
        }
    }
}

// ============================================================================
// Allocator trait
// ============================================================================

/// Device-aware memory allocator.
///
/// Provides allocation and deallocation for a specific device. Implementations
/// exist for CPU (`Vec`-based) and GPU backends (cuMalloc, hipMalloc, etc.).
///
/// The allocator returns a raw pointer and a deleter callback. The deleter
/// is stored in the owned container ([`Matrix`] or [`Tensor`]) and called
/// on drop.
///
/// # Examples
///
/// ```ignore
/// use dlpack_core::{Alloc, DLDevice};
///
/// struct CpuAlloc;
/// impl Alloc for CpuAlloc {
///     fn alloc(&self, device: DLDevice, size_bytes: usize, align: usize)
///         -> (*mut u8, Box<dyn FnOnce(*mut u8)>)
///     {
///         // CPU allocation using Vec
///         todo!()
///     }
/// }
/// ```
pub trait Alloc {
    /// Allocates `size_bytes` of memory on the given device.
    ///
    /// Returns `(ptr, deleter)` where `deleter` will be called with `ptr`
    /// when the owning container is dropped.
    fn alloc(
        &self,
        device: DLDevice,
        size_bytes: usize,
        align: usize,
    ) -> (*mut u8, Box<dyn FnOnce(*mut u8)>);
}

// ============================================================================
// Matrix — Owned 2D container
// ============================================================================

/// Owned 2D matrix with device-aware memory management.
///
/// Memory is freed on drop via the stored deleter callback, following the
/// `DLManagedTensor` ownership pattern. This allows a single type to hold
/// CPU heap memory, CUDA device memory, or any other allocation.
///
/// # Layout
///
/// Element `(i, j)` is at byte offset `(i * strides[0] + j * strides[1]) * size_of::<T>()`
/// from `ptr`. Strides are in units of elements (not bytes).
///
/// - Column-major: `strides = [1, nrows]`
/// - Row-major: `strides = [ncols, 1]`
///
/// # Examples
///
/// ```
/// use dlpack_core::Matrix;
///
/// // 3×4 column-major matrix from Vec
/// let mat = Matrix::from_vec(vec![0.0f64; 12], 3, 4, [1, 3]);
/// assert_eq!(mat.nrows(), 3);
/// assert_eq!(mat.ncols(), 4);
/// ```
pub struct Matrix<T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    strides: [isize; 2],
    device: DLDevice,
    deleter: Option<Box<dyn FnOnce(*mut u8)>>,
    _phantom: PhantomData<T>,
}

impl<T> Matrix<T> {
    /// Creates a CPU matrix from a `Vec<T>`, taking ownership of the data.
    ///
    /// The `Vec` is consumed and its memory is managed by the `Matrix`.
    /// The deleter reconstructs and drops the `Vec` on drop.
    ///
    /// # Arguments
    ///
    /// * `data` — Data vector (length must be >= nrows * ncols for the given strides)
    /// * `nrows` — Number of rows
    /// * `ncols` — Number of columns
    /// * `strides` — Element strides `[row_stride, col_stride]`
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Matrix;
    ///
    /// let mat = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2, [1, 2]);
    /// assert_eq!(mat.nrows(), 2);
    /// ```
    pub fn from_vec(data: Vec<T>, nrows: usize, ncols: usize, strides: [isize; 2]) -> Self {
        let cap = data.capacity();
        let len = data.len();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut T;
        let deleter: Box<dyn FnOnce(*mut u8)> = Box::new(move |p: *mut u8| unsafe {
            let _ = Vec::from_raw_parts(p as *mut T, len, cap);
        });
        Self {
            ptr,
            nrows,
            ncols,
            strides,
            device: DLDevice::cpu(),
            deleter: Some(deleter),
            _phantom: PhantomData,
        }
    }

    /// Creates a matrix from a raw pointer with a custom deleter.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for reads/writes of `nrows * ncols` elements
    ///   (according to the stride layout) on the specified device.
    /// - `deleter` must correctly free the memory when called with `ptr`.
    /// - The caller must ensure the pointer remains valid until the `Matrix`
    ///   is dropped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use dlpack_core::{Matrix, DLDevice, DeviceType};
    ///
    /// // Wrapping externally-allocated GPU memory
    /// let gpu_ptr = cuda_malloc(1024);
    /// let mat = unsafe {
    ///     Matrix::<f64>::from_raw(
    ///         gpu_ptr, 16, 16, [1, 16],
    ///         DLDevice::new(DeviceType::Cuda, 0),
    ///         Box::new(|p| cuda_free(p)),
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        strides: [isize; 2],
        device: DLDevice,
        deleter: Box<dyn FnOnce(*mut u8)>,
    ) -> Self {
        Self {
            ptr,
            nrows,
            ncols,
            strides,
            device,
            deleter: Some(deleter),
            _phantom: PhantomData,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Element strides `[row_stride, col_stride]`.
    pub fn strides(&self) -> [isize; 2] {
        self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Creates an immutable view of this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Matrix;
    ///
    /// let mat = Matrix::from_vec(vec![1.0; 6], 2, 3, [1, 2]);
    /// let view = mat.as_view();
    /// assert_eq!(view.nrows(), 2);
    /// ```
    pub fn as_view(&self) -> MatrixView<'_, T> {
        MatrixView {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            strides: self.strides,
            device: self.device,
            _phantom: PhantomData,
        }
    }

    /// Creates a mutable view of this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Matrix;
    ///
    /// let mut mat = Matrix::from_vec(vec![1.0; 6], 2, 3, [1, 2]);
    /// let view = mat.as_view_mut();
    /// assert_eq!(view.nrows(), 2);
    /// ```
    pub fn as_view_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            ptr: self.ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            strides: self.strides,
            device: self.device,
            _phantom: PhantomData,
        }
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        if let Some(del) = self.deleter.take() {
            del(self.ptr as *mut u8);
        }
    }
}

// ============================================================================
// MatrixView — Borrowed immutable 2D view
// ============================================================================

/// Borrowed immutable 2D matrix view.
///
/// Carries a pointer, shape, strides, and device info without ownership.
/// The data must outlive the view (enforced by lifetime `'a`).
///
/// # Examples
///
/// ```
/// use dlpack_core::{MatrixView, DLDevice};
///
/// let data = vec![1.0f64; 6];
/// let view = MatrixView::new(data.as_ptr(), 2, 3, [1, 2], DLDevice::cpu());
/// assert_eq!(view.nrows(), 2);
/// assert_eq!(view.ncols(), 3);
/// ```
pub struct MatrixView<'a, T> {
    ptr: *const T,
    nrows: usize,
    ncols: usize,
    strides: [isize; 2],
    device: DLDevice,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> MatrixView<'a, T> {
    /// Creates a new immutable matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{MatrixView, DLDevice};
    ///
    /// let data = vec![0.0f64; 12];
    /// let view = MatrixView::new(data.as_ptr(), 3, 4, [1, 3], DLDevice::cpu());
    /// assert_eq!(view.nrows(), 3);
    /// ```
    pub fn new(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        strides: [isize; 2],
        device: DLDevice,
    ) -> Self {
        Self {
            ptr,
            nrows,
            ncols,
            strides,
            device,
            _phantom: PhantomData,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Element strides `[row_stride, col_stride]`.
    pub fn strides(&self) -> [isize; 2] {
        self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

// ============================================================================
// MatrixViewMut — Borrowed mutable 2D view
// ============================================================================

/// Borrowed mutable 2D matrix view.
///
/// Same as [`MatrixView`] but allows mutation of the underlying data.
///
/// # Examples
///
/// ```
/// use dlpack_core::{MatrixViewMut, DLDevice};
///
/// let mut data = vec![0.0f64; 6];
/// let view = MatrixViewMut::new(data.as_mut_ptr(), 2, 3, [1, 2], DLDevice::cpu());
/// assert_eq!(view.nrows(), 2);
/// ```
pub struct MatrixViewMut<'a, T> {
    ptr: *mut T,
    nrows: usize,
    ncols: usize,
    strides: [isize; 2],
    device: DLDevice,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// Creates a new mutable matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{MatrixViewMut, DLDevice};
    ///
    /// let mut data = vec![0.0f64; 12];
    /// let view = MatrixViewMut::new(data.as_mut_ptr(), 3, 4, [1, 3], DLDevice::cpu());
    /// assert_eq!(view.nrows(), 3);
    /// ```
    pub fn new(
        ptr: *mut T,
        nrows: usize,
        ncols: usize,
        strides: [isize; 2],
        device: DLDevice,
    ) -> Self {
        Self {
            ptr,
            nrows,
            ncols,
            strides,
            device,
            _phantom: PhantomData,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Element strides `[row_stride, col_stride]`.
    pub fn strides(&self) -> [isize; 2] {
        self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

// ============================================================================
// Tensor — Owned N-dimensional container
// ============================================================================

/// Owned N-dimensional tensor with device-aware memory management.
///
/// The N-dimensional generalization of [`Matrix`]. Memory is freed on drop
/// via the stored deleter callback.
///
/// # Examples
///
/// ```
/// use dlpack_core::Tensor;
///
/// // 2×3×4 tensor from Vec
/// let data = vec![0.0f64; 24];
/// let t = Tensor::from_vec(data, vec![2, 3, 4], vec![1, 2, 6]);
/// assert_eq!(t.ndim(), 3);
/// assert_eq!(t.shape(), &[2, 3, 4]);
/// ```
pub struct Tensor<T> {
    ptr: *mut T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    device: DLDevice,
    deleter: Option<Box<dyn FnOnce(*mut u8)>>,
    _phantom: PhantomData<T>,
}

impl<T> Tensor<T> {
    /// Creates a CPU tensor from a `Vec<T>`, taking ownership.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![0.0f64; 6], vec![2, 3], vec![1, 2]);
    /// assert_eq!(t.ndim(), 2);
    /// ```
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>, strides: Vec<isize>) -> Self {
        let cap = data.capacity();
        let len = data.len();
        let ptr = Box::into_raw(data.into_boxed_slice()) as *mut T;
        let deleter: Box<dyn FnOnce(*mut u8)> = Box::new(move |p: *mut u8| unsafe {
            let _ = Vec::from_raw_parts(p as *mut T, len, cap);
        });
        Self {
            ptr,
            shape,
            strides,
            device: DLDevice::cpu(),
            deleter: Some(deleter),
            _phantom: PhantomData,
        }
    }

    /// Creates a tensor from a raw pointer with a custom deleter.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Matrix::from_raw`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use dlpack_core::{Tensor, DLDevice, DeviceType};
    ///
    /// let gpu_ptr = cuda_malloc(1024);
    /// let t = unsafe {
    ///     Tensor::<f64>::from_raw(
    ///         gpu_ptr, vec![16, 16], vec![1, 16],
    ///         DLDevice::new(DeviceType::Cuda, 0),
    ///         Box::new(|p| cuda_free(p)),
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw(
        ptr: *mut T,
        shape: Vec<usize>,
        strides: Vec<isize>,
        device: DLDevice,
        deleter: Box<dyn FnOnce(*mut u8)>,
    ) -> Self {
        Self {
            ptr,
            shape,
            strides,
            device,
            deleter: Some(deleter),
            _phantom: PhantomData,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape (dimension sizes).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Creates an immutable view of this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![0.0f64; 6], vec![2, 3], vec![1, 2]);
    /// let view = t.as_view();
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            ptr: self.ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: self.device,
            _phantom: PhantomData,
        }
    }

    /// Creates a mutable view of this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Tensor;
    ///
    /// let mut t = Tensor::from_vec(vec![0.0f64; 6], vec![2, 3], vec![1, 2]);
    /// let view = t.as_view_mut();
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn as_view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            ptr: self.ptr,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: self.device,
            _phantom: PhantomData,
        }
    }
}

impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        if let Some(del) = self.deleter.take() {
            del(self.ptr as *mut u8);
        }
    }
}

// ============================================================================
// TensorView — Borrowed immutable N-dimensional view
// ============================================================================

/// Borrowed immutable N-dimensional tensor view.
///
/// # Examples
///
/// ```
/// use dlpack_core::{TensorView, DLDevice};
///
/// let data = vec![0.0f64; 24];
/// let view = TensorView::new(
///     data.as_ptr(), vec![2, 3, 4], vec![1, 2, 6], DLDevice::cpu(),
/// );
/// assert_eq!(view.ndim(), 3);
/// assert_eq!(view.shape(), &[2, 3, 4]);
/// ```
pub struct TensorView<'a, T> {
    ptr: *const T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    device: DLDevice,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> TensorView<'a, T> {
    /// Creates a new immutable tensor view.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{TensorView, DLDevice};
    ///
    /// let data = vec![0.0f64; 6];
    /// let view = TensorView::new(data.as_ptr(), vec![2, 3], vec![1, 2], DLDevice::cpu());
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn new(ptr: *const T, shape: Vec<usize>, strides: Vec<isize>, device: DLDevice) -> Self {
        Self {
            ptr,
            shape,
            strides,
            device,
            _phantom: PhantomData,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape (dimension sizes).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

// ============================================================================
// TensorViewMut — Borrowed mutable N-dimensional view
// ============================================================================

/// Borrowed mutable N-dimensional tensor view.
///
/// # Examples
///
/// ```
/// use dlpack_core::{TensorViewMut, DLDevice};
///
/// let mut data = vec![0.0f64; 6];
/// let view = TensorViewMut::new(
///     data.as_mut_ptr(), vec![2, 3], vec![1, 2], DLDevice::cpu(),
/// );
/// assert_eq!(view.ndim(), 2);
/// ```
pub struct TensorViewMut<'a, T> {
    ptr: *mut T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    device: DLDevice,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T> TensorViewMut<'a, T> {
    /// Creates a new mutable tensor view.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{TensorViewMut, DLDevice};
    ///
    /// let mut data = vec![0.0f64; 6];
    /// let view = TensorViewMut::new(
    ///     data.as_mut_ptr(), vec![2, 3], vec![1, 2], DLDevice::cpu(),
    /// );
    /// assert_eq!(view.ndim(), 2);
    /// ```
    pub fn new(ptr: *mut T, shape: Vec<usize>, strides: Vec<isize>, device: DLDevice) -> Self {
        Self {
            ptr,
            shape,
            strides,
            device,
            _phantom: PhantomData,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape (dimension sizes).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Device where the data resides.
    pub fn device(&self) -> DLDevice {
        self.device
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

// ============================================================================
// Conversions: Matrix <-> Tensor (ndim == 2)
// ============================================================================

impl<T> Matrix<T> {
    /// Converts this matrix into a 2D tensor, transferring ownership.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::Matrix;
    ///
    /// let mat = Matrix::from_vec(vec![0.0f64; 6], 2, 3, [1, 2]);
    /// let t = mat.into_tensor();
    /// assert_eq!(t.ndim(), 2);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn into_tensor(mut self) -> Tensor<T> {
        let deleter = self.deleter.take();
        let tensor = Tensor {
            ptr: self.ptr,
            shape: vec![self.nrows, self.ncols],
            strides: vec![self.strides[0], self.strides[1]],
            device: self.device,
            deleter,
            _phantom: PhantomData,
        };
        // Prevent double-free: self.deleter is now None, so Drop is a no-op.
        std::mem::forget(self);
        tensor
    }
}

impl<'a, T> MatrixView<'a, T> {
    /// Creates a 2D tensor view from this matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use dlpack_core::{MatrixView, DLDevice};
    ///
    /// let data = vec![0.0f64; 6];
    /// let mv = MatrixView::new(data.as_ptr(), 2, 3, [1, 2], DLDevice::cpu());
    /// let tv = mv.as_tensor_view();
    /// assert_eq!(tv.ndim(), 2);
    /// ```
    pub fn as_tensor_view(&self) -> TensorView<'a, T> {
        TensorView {
            ptr: self.ptr,
            shape: vec![self.nrows, self.ncols],
            strides: vec![self.strides[0], self.strides[1]],
            device: self.device,
            _phantom: PhantomData,
        }
    }
}
