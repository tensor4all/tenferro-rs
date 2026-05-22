use std::{ffi::c_void, mem::MaybeUninit};

use cudarc::runtime::result as cuda_result;

use super::{kernels::*, state::CudaRuntime};
use crate::{Error, Result};

impl CudaRuntime {
    /// Allocates a raw device pointer for `len` elements of `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer must eventually be passed to [`CudaRuntime::free_raw`]
    /// on the same runtime.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub fn alloc_raw<T>(&self, len: usize) -> Result<*mut T> {
        self.bind_context()?;
        if len == 0 {
            return Ok(std::ptr::null_mut());
        }

        let ptr = unsafe { cuda_result::malloc_sync(checked_num_bytes::<T>(len)?) }
            .map_err(|err| cuda_error("cudaMalloc", err))?;
        Ok(ptr.cast::<T>())
    }

    /// Frees a raw device pointer previously allocated by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    ///
    /// `ptr` must either be null or a live allocation returned by this runtime.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn free_raw<T>(&self, ptr: *mut T) -> Result<()> {
        self.bind_context()?;
        if ptr.is_null() {
            return Ok(());
        }

        unsafe { cuda_result::free_sync(ptr.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaFree", err))
    }

    /// Copies a host slice into a raw device allocation.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live device allocation holding at least `dst_len` elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_htod_raw(&[1.0_f32, 2.0, 3.0, 4.0], ptr, 4).unwrap();
    ///     runtime.free_raw(ptr).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_htod_raw<T>(&self, src: &[T], dst: *mut T, dst_len: usize) -> Result<()> {
        if src.len() != dst_len {
            return Err(Error::InvalidArgument(format!(
                "host/device length mismatch: src={} dst={dst_len}",
                src.len()
            )));
        }

        self.bind_context()?;
        if src.is_empty() {
            return Ok(());
        }

        unsafe { cuda_result::memcpy_htod_sync(dst.cast::<c_void>(), as_byte_slice(src)) }
            .map_err(|err| cuda_error("cudaMemcpyHtoD", err))
    }

    /// Copies a raw device allocation into a host vector.
    ///
    /// # Safety
    ///
    /// `src` must point to a live device allocation holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// let host = unsafe { runtime.copy_dtoh_raw(ptr, 4).unwrap() };
    /// assert_eq!(host.len(), 4);
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn copy_dtoh_raw<T>(&self, src: *const T, len: usize) -> Result<Vec<T>> {
        self.bind_context()?;
        if len == 0 {
            return Ok(Vec::new());
        }

        let num_bytes = checked_num_bytes::<T>(len)?;
        let mut host = Vec::<MaybeUninit<T>>::with_capacity(len);
        unsafe { host.set_len(len) };
        let host_bytes =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), num_bytes) };

        unsafe { cuda_result::memcpy_dtoh_sync(host_bytes, src.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaMemcpyDtoH", err))?;

        let ptr = host.as_mut_ptr().cast::<T>();
        let len = host.len();
        let cap = host.capacity();
        std::mem::forget(host);
        Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    }

    /// Copies one raw device allocation into another.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_dtod_raw(src, dst, 4).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_dtod_raw<T>(&self, src: *const T, dst: *mut T, len: usize) -> Result<()> {
        self.bind_context()?;
        if len == 0 {
            return Ok(());
        }

        unsafe {
            cuda_result::memcpy_dtod_sync(
                dst.cast::<c_void>(),
                src.cast::<c_void>(),
                checked_num_bytes::<T>(len)?,
            )
        }
        .map_err(|err| cuda_error("cudaMemcpyDtoD", err))
    }
}
