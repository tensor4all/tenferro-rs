use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::runtime::result as cuda_result;

use super::kernels::{cuda_error, runtime_cache};
use crate::{Error, Result};

/// Shared CUDA runtime handle for one device ordinal.
///
/// The handle retains the CUDA primary context and exposes low-level memory
/// allocation and copy primitives that higher-level crates can reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
#[derive(Debug)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
}

impl CudaRuntime {
    fn new(device_id: usize) -> Result<Arc<Self>> {
        let context =
            CudaContext::new(device_id).map_err(|err| cuda_error("CUDA device init", err))?;
        Ok(Arc::new(Self { context }))
    }

    pub(super) fn bind_context(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|err| cuda_error("CUDA context bind", err))
    }

    /// Returns the CUDA device ordinal this runtime is bound to.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// assert_eq!(runtime.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }

    /// Returns a clone of the shared CUDA context handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ctx = runtime.context();
    /// assert_eq!(ctx.ordinal(), 0);
    /// ```
    pub fn context(&self) -> Arc<CudaContext> {
        Arc::clone(&self.context)
    }
}

impl CudaRuntime {
    pub(super) fn ensure_same_device(&self, device_id: usize) -> Result<()> {
        if self.device_id() == device_id {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "buffer belongs to device {device_id}, runtime is bound to device {}",
                self.device_id()
            )))
        }
    }
}

/// Owning CUDA device buffer allocated by [`CudaRuntime`].
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// let buffer = runtime.alloc::<f32>(4).unwrap();
/// assert_eq!(buffer.len(), 4);
/// ```
#[derive(Debug)]
pub struct CudaBuffer<T> {
    context: Arc<CudaContext>,
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    pub(super) fn new(context: Arc<CudaContext>, ptr: *mut c_void, len: usize) -> Self {
        Self {
            context,
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements in this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the raw device pointer for this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert!(!buffer.device_ptr().is_null());
    /// ```
    pub fn device_ptr(&self) -> *mut T {
        self.ptr.cast::<T>()
    }

    /// Returns the device ordinal that owns this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if !self.ptr.is_null() {
            let _ = unsafe { cuda_result::free_sync(self.ptr) };
        }
    }
}

/// Returns the shared runtime handle for one CUDA device ordinal.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
pub fn get_or_init(device_id: usize) -> Result<Arc<CudaRuntime>> {
    let mut cache = runtime_cache()
        .lock()
        .map_err(|_| Error::DeviceError("CUDA runtime cache mutex poisoned".into()))?;
    if let Some(runtime) = cache.get(&device_id) {
        return Ok(Arc::clone(runtime));
    }

    let runtime = CudaRuntime::new(device_id)?;
    cache.insert(device_id, Arc::clone(&runtime));
    Ok(runtime)
}
