//! Type-safe raw CUDA extension session (issue #1597).
//!
//! [`Session`] is the typed context capability exposed by
//! [`CudaExecSession::with_raw`](super::exec_session::CudaExecSession::with_raw):
//! while a raw session is alive the tenferro primary context is activated on
//! the current thread with a single captured execution stream bound.
//!
//! Safety model
//! ------------
//!
//! - `Session`, [`StreamRef`], [`TensorRef`], and [`DeviceBytes`] are all
//!   scoped to the request lifetime and are `!Send + !Sync`, so the
//!   thread-local context/stream capability cannot migrate to another thread.
//! - `StreamRef` is not a `u64` and does not expose its native handle in safe
//!   code; vendor FFI receives the handle only through an explicit unsafe
//!   escape that must not retain it past the session scope.
//! - `TensorRef` carries a validated device span and ownership identity; it
//!   provides no `Deref` to the device buffer.
//! - A mutable device access can only be produced through [`Session::tensor_mut`]
//!   which requires an exclusive `&mut` borrow of a compatibly-typed tensor, or
//!   through a freshly allocated [`Session::alloc_output`] tensor.

use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use tenferro_tensor::{TensorRank, TypedTensor};

use super::runtime::CudaRuntime;
use super::{CudaExtensionCache, CudaExtensionCacheGuard, CudaRuntimeIdentity};

/// Raw CUDA extension session.
///
/// Not constructible by users. Represents "tenferro primary context active on
/// this thread with one captured execution stream bound". `!Send + !Sync` by
/// construction (`Rc`).
pub struct Session<'s> {
    runtime: CudaRuntime,
    cache: &'s CudaExtensionCache,
    stream: u64,
    _not_send_sync: PhantomData<Rc<()>>,
    _scope: PhantomData<&'s ()>,
}

impl<'s> Session<'s> {
    /// Create a raw session bound to `runtime`, `cache`, and the current
    /// CubeCL stream.
    ///
    /// # Safety
    ///
    /// Caller must have activated the tenferro primary context for `runtime`
    /// and hold it stronger than `'s`; `stream` must be the captured CubeCL
    /// stream bound to the current thread.
    pub(crate) unsafe fn new(
        runtime: CudaRuntime,
        cache: &'s CudaExtensionCache,
        stream: u64,
    ) -> Self {
        Self {
            runtime,
            cache,
            stream,
            _not_send_sync: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Return the identity of the runtime backing this session.
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity {
        self.runtime.runtime_identity()
    }

    /// Borrow the captured execution stream.
    pub fn stream(&self) -> StreamRef<'s> {
        StreamRef {
            raw: self.stream,
            _scope: PhantomData,
        }
    }

    /// Build a checked read-only device reference for a GPU-backed tensor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident on
    /// this session's runtime/device.
    pub fn tensor<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
    ) -> crate::Result<TensorRef<'s, T>>
    where
        T: 'static,
    {
        let prepared = super::dispatch::cubecl_buffer(tensor, "raw.tensor")?;
        let byte_len = prepared.byte_len;
        let resource = self
            .runtime
            .client()
            .get_resource(prepared.handle().clone())
            .map_err(|err| crate::Error::backend_source("raw.tensor", err))?;
        let base =
            super::interop::cuda_device_ptr_from_addr(resource.resource().ptr, "raw.tensor")?;
        Ok(TensorRef {
            base,
            byte_len,
            _scope: PhantomData,
            _dtype: PhantomData,
        })
    }

    /// Build a checked mutable device reference for a GPU-backed tensor.
    ///
    /// Requires an exclusive borrow of the tensor, so aliasing mutable device
    /// access cannot be produced from a shared `&TypedTensor<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident on
    /// this session's runtime/device.
    pub fn tensor_mut<T>(
        &self,
        tensor: &mut TypedTensor<T, impl TensorRank>,
    ) -> crate::Result<TensorMut<'s, T>>
    where
        T: 'static,
    {
        let prepared = super::dispatch::cubecl_buffer(tensor, "raw.tensor_mut")?;
        let byte_len = prepared.byte_len;
        let resource = self
            .runtime
            .client()
            .get_resource(prepared.handle().clone())
            .map_err(|err| crate::Error::backend_source("raw.tensor_mut", err))?;
        let base =
            super::interop::cuda_device_ptr_from_addr(resource.resource().ptr, "raw.tensor_mut")?;
        Ok(TensorMut {
            base,
            byte_len,
            _scope: PhantomData,
            _dtype: PhantomData,
        })
    }

    /// Block the host until work enqueued on the session's stream completes.
    ///
    /// This is the only host barrier on the raw-session success path; regular
    /// successes only enqueue.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the CUDA synchronize call
    /// fails.
    pub fn synchronize(&self) -> crate::Result<()> {
        self.runtime.synchronize()
    }

    /// Allocate a dense GPU tensor of `T` on the session's device.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] on shape overflow or
    /// [`crate::Error::BackendSource`] on allocation failure.
    pub fn alloc_output<T>(&self, shape: &[usize]) -> crate::Result<TypedTensor<T>>
    where
        T: cubecl::prelude::CubeElement
            + tenferro_tensor::TensorScalar
            + Clone
            + Send
            + Sync
            + 'static,
    {
        super::dispatch::alloc_output(&self.runtime, shape)
    }

    /// Allocate a CubeCL-owned byte workspace returned as [`DeviceBytes`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot allocate or
    /// inspect the workspace resource.
    pub fn alloc_bytes(&self, nbytes: usize, op: &'static str) -> crate::Result<DeviceBytes<'s>> {
        let inner = super::interop::alloc_device_bytes(&self.runtime, nbytes, op)?;
        Ok(DeviceBytes {
            inner,
            _scope: PhantomData,
        })
    }

    /// Get or lazily initialize a runtime-scoped, type-keyed extension resource.
    ///
    /// This is the narrow public view over the existing bounded per-runtime
    /// [`CudaExtensionCache`](super::CudaExtensionCache). The guard holds the
    /// cache lock and serializes access; cache keys remain per exact runtime
    /// instance (never per-device).
    ///
    /// # Errors
    ///
    /// Propagates the initializer's typed error or a cache poison/runtime-state
    /// error.
    pub fn resource<T>(
        &self,
        init: impl FnOnce() -> crate::Result<T>,
    ) -> crate::Result<CudaResourceGuard<'_, T>>
    where
        T: Send + 'static,
    {
        let guard = self.cache.get_or_try_init(init)?;
        Ok(CudaResourceGuard { inner: guard })
    }
}

/// Borrowed, opaque CUDA execution stream.
///
/// Non-copy and non-constructible. The native handle is only exposed through an
/// explicit unsafe FFI escape; the value must not be retained past the session
/// scope.
pub struct StreamRef<'s> {
    raw: u64,
    _scope: PhantomData<&'s ()>,
}

impl<'s> StreamRef<'s> {
    /// Extract the native stream handle for an FFI call.
    ///
    /// # Safety
    ///
    /// The returned handle is valid only for the lifetime `'s` and only while
    /// the session's context is current on this thread. It must not be retained
    /// or destroyed by the caller.
    pub unsafe fn raw_handle(&self) -> u64 {
        self.raw
    }
}

impl fmt::Debug for StreamRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamRef").finish_non_exhaustive()
    }
}

/// Checked read-only device span of a GPU-backed tensor.
pub struct TensorRef<'s, T> {
    base: *mut std::ffi::c_void,
    byte_len: usize,
    _scope: PhantomData<&'s ()>,
    _dtype: PhantomData<T>,
}

impl<'s, T> TensorRef<'s, T> {
    /// Return the validated device span in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Extract the device base pointer for an FFI call.
    ///
    /// # Safety
    ///
    /// The pointer is a read-only view unless the caller's kernel contract
    /// proves otherwise, and is valid only for lifetime `'s` for the span
    /// `[base, base + byte_len)`. It must not be retained past the session
    /// scope.
    pub unsafe fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.base
    }
}

impl<T> fmt::Debug for TensorRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorRef")
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

/// Checked mutable device span of a GPU-backed tensor.
///
/// Only obtainable from an exclusive `&mut` tensor borrow or a fresh output.
pub struct TensorMut<'s, T> {
    base: *mut std::ffi::c_void,
    byte_len: usize,
    _scope: PhantomData<&'s ()>,
    _dtype: PhantomData<T>,
}

impl<'s, T> TensorMut<'s, T> {
    /// Return the validated device span in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Extract the mutable device base pointer for an FFI call.
    ///
    /// # Safety
    ///
    /// The pointer is valid only for lifetime `'s` for the span
    /// `[base, base + byte_len)`. The caller is responsible for bounds,
    /// aliasing, and initialization under its own kernel contract.
    pub unsafe fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.base
    }
}

/// CubeCL-owned device byte workspace kept alive for CUDA library calls.
pub struct DeviceBytes<'s> {
    inner: super::interop::DeviceByteBuffer,
    _scope: PhantomData<&'s ()>,
}

impl<'s> DeviceBytes<'s> {
    /// Borrow the workspace device pointer for an FFI call.
    pub fn with_ptr(&self, f: impl FnOnce(*mut std::ffi::c_void)) {
        self.inner.with_ptr(f)
    }

    /// Return whether this workspace holds a live allocation.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl fmt::Debug for DeviceBytes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBytes")
            .field("is_empty", &self.is_empty())
            .finish_non_exhaustive()
    }
}

/// Runtime-scoped, type-keyed resource guard.
///
/// A narrow borrowed view over the bounded per-runtime extension cache.
pub struct CudaResourceGuard<'cache, T> {
    inner: CudaExtensionCacheGuard<'cache, T>,
}

impl<T: 'static> std::ops::Deref for CudaResourceGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: 'static> fmt::Debug for CudaResourceGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaResourceGuard")
            .field("value_type", &std::any::type_name::<T>())
            .finish_non_exhaustive()
    }
}
