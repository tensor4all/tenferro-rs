use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

/// Data storage for tensor elements.
///
/// Abstracts over ownership: data may be Rust-owned (`Vec<T>`),
/// externally-owned (e.g., imported via DLPack with a release callback),
/// or GPU device memory. Shape and stride metadata are not stored here.
///
/// # Shared ownership
///
/// `DataBuffer` uses `Arc`, so cloning a tensor or buffer is shallow and
/// shares storage. Deep copies happen only when a tensor is materialized into
/// a new contiguous allocation.
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

enum BufferInner<T> {
    Owned(Vec<T>),
    External {
        ptr: *const T,
        len: usize,
        release: Option<Box<dyn FnOnce() + Send>>,
    },
    #[allow(dead_code)]
    Gpu {
        device_ptr: *mut T,
        len: usize,
        space: LogicalMemorySpace,
        release: Option<Box<dyn FnOnce() + Send>>,
    },
}

unsafe impl<T: Send> Send for DataBuffer<T> {}
unsafe impl<T: Sync> Sync for DataBuffer<T> {}

impl<T> Clone for DataBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for BufferInner<T> {
    fn drop(&mut self) {
        match self {
            BufferInner::External { release, .. } | BufferInner::Gpu { release, .. } => {
                if let Some(callback) = release.take() {
                    callback();
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
    /// assert!(buf.is_owned());
    /// ```
    pub fn from_vec(v: Vec<T>) -> Self {
        Self {
            inner: Arc::new(BufferInner::Owned(v)),
        }
    }

    /// Create a buffer from externally-owned data with a release callback.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid, properly aligned allocation of at least
    ///   `len` elements of `T`.
    /// - The allocation must remain valid until the release callback fires.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let ptr = data.as_ptr();
    /// let len = data.len();
    /// let buf = unsafe { DataBuffer::from_external(ptr, len, move || drop(data)) };
    /// assert!(!buf.is_owned());
    /// ```
    pub unsafe fn from_external(
        ptr: *const T,
        len: usize,
        release: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(BufferInner::External {
                ptr,
                len,
                release: Some(Box::new(release)),
            }),
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn from_gpu_parts(
        device_ptr: *mut T,
        len: usize,
        space: LogicalMemorySpace,
        release: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(BufferInner::Gpu {
                device_ptr,
                len,
                space,
                release: Some(Box::new(release)),
            }),
        }
    }

    /// Returns the raw data as a slice for CPU-accessible buffers.
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

    /// Returns the raw data as a mutable slice, if uniquely owned.
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

    /// Returns `true` if the buffer is Rust-owned.
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
    /// let buf2 = buf.clone();
    /// assert!(!buf.is_unique());
    /// ```
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Extract the inner `Vec<T>` if this is the sole owner of a CPU-owned buffer.
    pub fn try_into_vec(self) -> Option<Vec<T>> {
        let inner = Arc::try_unwrap(self.inner).ok()?;
        let mut inner = std::mem::ManuallyDrop::new(inner);
        match &mut *inner {
            BufferInner::Owned(v) => Some(unsafe { std::ptr::read(v as *const Vec<T>) }),
            _ => {
                unsafe { std::mem::ManuallyDrop::drop(&mut inner) };
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
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DataBuffer;
    ///
    /// let buf = DataBuffer::from_vec(vec![1.0f64]);
    /// assert!(buf.as_device_ptr().is_none());
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
    /// assert!(buf.gpu_memory_space().is_none());
    /// ```
    pub fn gpu_memory_space(&self) -> Option<LogicalMemorySpace> {
        match &*self.inner {
            BufferInner::Gpu { space, .. } => Some(*space),
            _ => None,
        }
    }

    pub(crate) fn reinterpret_as<U>(&self, new_len: usize) -> Result<DataBuffer<U>>
    where
        T: Send + Sync + 'static,
        U: Send + Sync + 'static,
    {
        let src_bytes = self
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArgument("buffer byte-size overflow".into()))?;
        let dst_bytes = new_len
            .checked_mul(std::mem::size_of::<U>())
            .ok_or_else(|| {
                Error::InvalidArgument("reinterpreted buffer byte-size overflow".into())
            })?;
        if dst_bytes > src_bytes {
            return Err(Error::InvalidArgument(format!(
                "buffer reinterpretation exceeds source byte size: src_bytes={src_bytes} dst_bytes={dst_bytes}"
            )));
        }

        let align = std::mem::align_of::<U>();
        match &*self.inner {
            BufferInner::Owned(v) => {
                let ptr = v.as_ptr() as *const U;
                if new_len != 0 && (ptr as usize) % align != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "buffer reinterpretation would violate alignment {}",
                        align
                    )));
                }
                let owner = self.clone();
                Ok(unsafe { DataBuffer::from_external(ptr, new_len, move || drop(owner)) })
            }
            BufferInner::External { ptr, .. } => {
                let ptr = *ptr as *const U;
                if new_len != 0 && (ptr as usize) % align != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "external buffer reinterpretation would violate alignment {}",
                        align
                    )));
                }
                let owner = self.clone();
                Ok(unsafe { DataBuffer::from_external(ptr, new_len, move || drop(owner)) })
            }
            BufferInner::Gpu {
                device_ptr, space, ..
            } => {
                let ptr = *device_ptr as *mut U;
                if new_len != 0 && (ptr as usize) % align != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "gpu buffer reinterpretation would violate alignment {}",
                        align
                    )));
                }
                let owner = self.clone();
                Ok(
                    unsafe {
                        DataBuffer::from_gpu_parts(ptr, new_len, *space, move || drop(owner))
                    },
                )
            }
        }
    }
}

impl<T: Scalar> DataBuffer<T> {
    /// Allocate a zero-filled buffer on the specified device.
    ///
    /// For CPU (`MainMemory`) this creates an owned `Vec<T>` filled with
    /// `T::zero()`. For CUDA GPU memory (when the `cuda` feature is enabled)
    /// this allocates directly on the device and zero-fills via host-to-device
    /// copy.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported memory spaces or if device allocation
    /// fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::DataBuffer;
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let buf = DataBuffer::<f64>::zeros_on_device(4, LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(buf.as_slice().unwrap(), &[0.0; 4]);
    /// ```
    pub fn zeros_on_device(n_elements: usize, memory_space: LogicalMemorySpace) -> Result<Self> {
        match memory_space {
            LogicalMemorySpace::MainMemory => Ok(Self::from_vec(vec![T::zero(); n_elements])),
            #[cfg(feature = "cuda")]
            LogicalMemorySpace::GpuMemory { .. } => {
                crate::cuda_runtime::alloc_zeros_gpu(n_elements, memory_space)
            }
            _ => Err(Error::DeviceError(format!(
                "zeros_on_device: unsupported memory space {memory_space:?}"
            ))),
        }
    }

    /// Allocate an uninitialized buffer on the specified device.
    ///
    /// For CPU (`MainMemory`) this creates an owned `Vec<T>` with the
    /// requested capacity. The contents are **unspecified** (currently
    /// zero-filled for safety, but callers must not rely on that).
    ///
    /// For CUDA GPU memory (when the `cuda` feature is enabled) this
    /// allocates device memory without initialization.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported memory spaces or if device allocation
    /// fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::DataBuffer;
    /// use tenferro_device::LogicalMemorySpace;
    ///
    /// let buf = DataBuffer::<f64>::allocate_uninit_on_device(4, LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(buf.len(), 4);
    /// ```
    pub fn allocate_uninit_on_device(
        n_elements: usize,
        memory_space: LogicalMemorySpace,
    ) -> Result<Self> {
        match memory_space {
            LogicalMemorySpace::MainMemory => {
                // Zero-fill for safety on CPU; the "uninit" contract means
                // callers will overwrite, but we avoid UB by initializing.
                Ok(Self::from_vec(vec![T::zero(); n_elements]))
            }
            #[cfg(feature = "cuda")]
            LogicalMemorySpace::GpuMemory { .. } => {
                // True uninit allocation on GPU — caller is responsible for
                // writing before reading.
                crate::cuda_runtime::alloc_gpu_uninit(n_elements, memory_space)
            }
            _ => Err(Error::DeviceError(format!(
                "allocate_uninit_on_device: unsupported memory space {memory_space:?}"
            ))),
        }
    }
}
