//! The owner-scoped raw session an extension crate runs its vendor calls in.
//!
//! The session exposes exactly what a vendor library needs — the runtime's stream, device
//! addresses, and scratch — and nothing that would re-enter the runtime's admission: transfers run
//! on the runtime before the scope opens, because their admission is the lock the scope holds.
//!
//! A caller's tensor is safe to borrow for as long as the vendor work runs: dropping a buffer
//! releases it with `cuMemFreeAsync` on the same stream, so the free is ordered after any call the
//! session enqueued. `retain_tensor` therefore only proves the tensor belongs to this runtime.

use tenferro_tensor::{StorageBuffer, TensorScalar, TypedTensor};

use super::engine::CudaEngine;
use super::storage::CudaStorage;

/// A handle to the runtime's own stream, valid for the session's life.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct StreamRef {
    handle: u64,
}

impl StreamRef {
    /// The raw `CUstream`, as the value a vendor binding takes.
    #[must_use]
    #[doc(hidden)]
    pub fn raw_handle(self) -> u64 {
        self.handle
    }
}

/// A device address the session handed out, valid until the scope's work completes.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct DeviceBytes<'s> {
    pointer: u64,
    bytes: usize,
    _session: std::marker::PhantomData<&'s ()>,
}

impl DeviceBytes<'_> {
    /// Run `f` with the device address.
    #[doc(hidden)]
    pub fn with_ptr<R>(&self, f: impl FnOnce(*mut std::ffi::c_void) -> R) -> R {
        f(self.pointer as usize as *mut std::ffi::c_void)
    }

    /// Number of bytes at the address.
    #[must_use]
    #[doc(hidden)]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

/// A read-only device view of a caller's tensor.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct TensorRef<'a, T> {
    pointer: u64,
    bytes: usize,
    _tensor: std::marker::PhantomData<&'a T>,
}

impl<T> TensorRef<'_, T> {
    /// The tensor's device address, as a vendor binding takes it.
    #[doc(hidden)]
    pub fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.pointer as usize as *mut std::ffi::c_void
    }

    /// Run `f` with the tensor's device address.
    #[doc(hidden)]
    pub fn with_ptr<R>(&self, f: impl FnOnce(*mut std::ffi::c_void) -> R) -> R {
        f(self.pointer as usize as *mut std::ffi::c_void)
    }

    /// Number of bytes the tensor spans.
    #[must_use]
    #[doc(hidden)]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

/// A writable device view of a caller's tensor.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct TensorMut<'a, T> {
    pointer: u64,
    bytes: usize,
    _tensor: std::marker::PhantomData<&'a mut T>,
}

impl<T> TensorMut<'_, T> {
    /// The tensor's device address, as a vendor binding takes it.
    #[doc(hidden)]
    pub fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.pointer as usize as *mut std::ffi::c_void
    }

    /// Run `f` with the tensor's device address.
    #[doc(hidden)]
    pub fn with_ptr<R>(&self, f: impl FnOnce(*mut std::ffi::c_void) -> R) -> R {
        f(self.pointer as usize as *mut std::ffi::c_void)
    }

    /// Number of bytes the tensor spans.
    #[must_use]
    #[doc(hidden)]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

/// One raw scope, handed to an extension's closure.
#[doc(hidden)]
pub struct Session<'s> {
    runtime: super::CudaRuntime,
    scope: t4a_cuda_runtime::RawScope<'s>,
    /// The runtime's extension caches, lent by the engine: a vendor handle is created once per
    /// engine and reused by every later session. The cell is per session and the session is
    /// thread-bound, so a shared borrow during a vendor call costs nothing and keeps `resource`
    /// callable while the session is read.
    caches: &'s std::cell::RefCell<tenferro_runtime::ExtensionCacheStore>,
}

impl<'s> Session<'s> {
    /// The runtime's stream, for a vendor library to bind its plan to.
    #[must_use]
    #[doc(hidden)]
    pub fn stream(&self) -> StreamRef {
        StreamRef {
            handle: self.scope.stream(),
        }
    }

    /// Zero-filled scratch in this runtime's domain.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeStateSource`] when the allocation or its zero fill fails.
    #[doc(hidden)]
    pub fn alloc_bytes(&self, nbytes: usize, op: &'static str) -> crate::Result<DeviceBytes<'s>> {
        let buffer = self
            .scope
            .alloc_zeroed(nbytes, 256)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        let pointer = self
            .scope
            .register(&buffer)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        // The allocation lives as long as this handle: it moves into the scope's keepalives so the
        // device address stays valid until the scope's completion, whatever the caller does.
        self.scope
            .retain(Box::new(buffer), nbytes)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(DeviceBytes {
            pointer,
            bytes: nbytes,
            _session: std::marker::PhantomData,
        })
    }

    /// Copy `bytes` between two device addresses on this session's stream.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the driver refuses the copy.
    #[doc(hidden)]
    /// # Safety
    ///
    /// `destination` and `source` must be device addresses this session's engine owns, `bytes`
    /// must not exceed either allocation, and the ranges must not overlap unless
    /// `cuMemcpyDtoDAsync` allows it. The copy is enqueued on this session's stream, so the
    /// allocations must outlive the point where that work retires.
    pub unsafe fn copy_bytes(
        &self,
        destination: *mut std::ffi::c_void,
        source: *const std::ffi::c_void,
        bytes: usize,
        op: &'static str,
    ) -> crate::Result<()> {
        let status = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                destination as usize as cudarc::driver::sys::CUdeviceptr,
                source as usize as cudarc::driver::sys::CUdeviceptr,
                bytes,
                self.scope.stream() as usize as *mut cudarc::driver::sys::CUstream_st,
            )
        };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::Error::runtime_state(
                op,
                format!("cuMemcpyDtoDAsync failed with {status:?}"),
            ));
        }
        Ok(())
    }

    /// Copy host `bytes` into fresh scratch of this runtime, on this session's stream.
    ///
    /// The host source is pageable and the copy is asynchronous, so the bytes are moved into the
    /// scope's keepalives: the driver may still be reading them when the caller returns.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the allocation or the copy is refused.
    #[doc(hidden)]
    pub fn upload_bytes(&self, bytes: &[u8], op: &'static str) -> crate::Result<DeviceBytes<'s>> {
        let device = self.alloc_bytes(bytes.len(), op)?;
        let owned = bytes.to_vec();
        let source = owned.as_ptr() as usize;
        device.with_ptr(|destination| {
            let status = unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    destination as usize as cudarc::driver::sys::CUdeviceptr,
                    source as *const std::ffi::c_void,
                    owned.len(),
                    self.scope.stream() as usize as *mut cudarc::driver::sys::CUstream_st,
                )
            };
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::Error::runtime_state(
                    op,
                    format!("cuMemcpyHtoDAsync failed with {status:?}"),
                ));
            }
            Ok(())
        })?;
        // The driver may still be reading the host source, so it outlives this call through the
        // scope's completion.
        self.scope
            .retain(Box::new(owned), 0)
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok(device)
    }

    /// Copy a device tensor into a host tensor of the same shape.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the tensor is not this runtime's allocation, and the
    /// runtime's own failure when the copy or its barrier fails.
    #[doc(hidden)]
    pub fn download_tensor<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<T>> {
        let (pointer, bytes) = self.address(tensor, op)?;
        let staging = PinnedStaging::new(bytes, op)?;
        let status = unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                staging.pointer,
                pointer as cudarc::driver::sys::CUdeviceptr,
                bytes,
                self.scope.stream() as usize as *mut cudarc::driver::sys::CUstream_st,
            )
        };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::Error::runtime_state(
                op,
                format!("cuMemcpyDtoHAsync failed with {status:?}"),
            ));
        }
        self.synchronize()?;
        let values: Vec<T> = staging
            .as_slice::<T>(bytes / std::mem::size_of::<T>().max(1))
            .to_vec();
        TypedTensor::<T>::from_vec_col_major(tensor.shape().to_vec(), values)
            .map_err(|error| crate::Error::runtime_state(op, error.to_string()))
    }

    /// Zero-filled device storage of `shape` for `T`, as a tensor of this runtime.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the allocation fails or the shape does not describe a
    /// column-major tensor of `T`.
    #[doc(hidden)]
    pub fn alloc_output<T: TensorScalar>(&self, shape: &[usize]) -> crate::Result<TypedTensor<T>> {
        const OP: &str = "cuda raw alloc_output";
        let count = shape.iter().try_fold(1usize, |product, &extent| {
            product
                .checked_mul(extent)
                .ok_or_else(|| crate::Error::runtime_state(OP, "the element count overflows usize"))
        })?;
        let bytes = count
            .checked_mul(std::mem::size_of::<T>().max(1))
            .ok_or_else(|| crate::Error::runtime_state(OP, "the byte length overflows usize"))?;
        let buffer = self
            .scope
            .alloc_zeroed(bytes, 256)
            .map_err(|error| crate::Error::runtime_state_source(OP, error))?;
        // The returned tensor owns the allocation, so its own drop frees it — in this runtime's
        // stream order, after any vendor call the session enqueued.
        let storage = super::storage::fresh_storage(&self.runtime, buffer);
        TypedTensor::<T>::from_buffer_col_major(
            shape.to_vec(),
            tenferro_tensor::StorageBuffer::Backend(Box::new(storage)),
            super::transfer::placement(&self.runtime),
        )
        .map_err(|error| crate::Error::runtime_state(OP, error.to_string()))
    }

    /// A read-only device view of `tensor`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the tensor is not this runtime's allocation.
    #[doc(hidden)]
    pub fn tensor<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
    ) -> crate::Result<TensorRef<'_, T>> {
        let (pointer, bytes) = self.address(tensor, "cuda raw tensor")?;
        Ok(TensorRef {
            pointer,
            bytes,
            _tensor: std::marker::PhantomData,
        })
    }

    /// A writable device view of `tensor`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the tensor is not this runtime's allocation.
    #[doc(hidden)]
    pub fn tensor_mut<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
    ) -> crate::Result<TensorMut<'_, T>> {
        let (pointer, bytes) = self.address(tensor, "cuda raw tensor mut")?;
        Ok(TensorMut {
            pointer,
            bytes,
            _tensor: std::marker::PhantomData,
        })
    }

    /// Prove `tensor` is one of this runtime's allocations, for a vendor call that reads it.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeState`] when the tensor is not this runtime's allocation.
    #[doc(hidden)]
    /// # Safety
    ///
    /// The returned address stays valid only while `tensor`'s allocation is resident on this
    /// engine; the caller must not outlive it.
    pub unsafe fn retain_tensor<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
        op: &'static str,
    ) -> crate::Result<DeviceBytes<'static>> {
        let (pointer, bytes) = self.address(tensor, op)?;
        Ok(DeviceBytes {
            pointer,
            bytes,
            _session: std::marker::PhantomData,
        })
    }

    /// This session's vendor resource of type `T`, created by `init` on first use.
    ///
    /// One slot per type: the cache key carries the type name, so two resources of different types
    /// in one engine cannot collide, and the same handle is reused by every later session.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine};
    ///
    /// # #[cfg(feature = "cuda")]
    /// if gpu_available() {
    ///     let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
    ///     // The guard borrows the session, so it stays inside the closure.
    ///     let total = engine.with_raw("resource", |session| {
    ///         let counter = session.resource(|| Ok::<u64, tenferro_tensor::Error>(7))?;
    ///         Ok(*counter + 1)
    ///     })?;
    ///     assert_eq!(total, 8);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Only if storing the initialised resource succeeded and it is then missing from the cache,
    /// which is an internal invariant of the store above.
    ///
    /// # Errors
    ///
    /// The initialiser's failure, and the engine's when the cache is poisoned.
    pub fn resource<T: Send + Sync + 'static>(
        &self,
        init: impl FnOnce() -> crate::Result<T>,
    ) -> crate::Result<CudaResourceGuard<'_, T>> {
        let key = tenferro_runtime::ExtensionCacheKey::new(
            "tenferro.cuda.session",
            std::any::type_name::<T>(),
            0,
        );
        let mut store = self.caches.borrow_mut();
        if store.get::<T>(&key).is_none() {
            let value = init()?;
            store.put(key, value, 0);
        }
        let entry = std::cell::RefMut::map(store, |store| {
            store
                .get_mut::<T>(&key)
                .expect("the resource was stored above")
        });
        Ok(CudaResourceGuard { entry })
    }

    /// Wait for the work enqueued on this session's stream.
    ///
    /// # Errors
    ///
    /// [`crate::Error::RuntimeStateSource`] when the stream cannot be synchronised.
    #[doc(hidden)]
    pub fn synchronize(&self) -> crate::Result<()> {
        let stream = self.scope.stream() as usize as *mut cudarc::driver::sys::CUstream_st;
        // SAFETY: the scope hands out this runtime's own stream; a barrier on it neither reads nor
        // writes an operand.
        let status = unsafe { cudarc::driver::sys::cuStreamSynchronize(stream) };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::Error::runtime_state(
                "cuda raw synchronize",
                format!("cuStreamSynchronize failed with {status:?}"),
            ));
        }
        Ok(())
    }

    fn address<T: TensorScalar>(
        &self,
        tensor: &TypedTensor<T>,
        op: &'static str,
    ) -> crate::Result<(u64, usize)> {
        let storage = match tensor.buffer() {
            StorageBuffer::Backend(buffer) => buffer
                .as_ref()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| {
                    crate::Error::runtime_state(op, "expected a CUDA allocation of this backend")
                })?,
            StorageBuffer::Host(_) => {
                return Err(crate::Error::runtime_state(
                    op,
                    "expected a device tensor, not host storage",
                ))
            }
        };
        if storage.domain() != self.runtime.allocation_domain_id() {
            return Err(crate::Error::runtime_state(
                op,
                "the tensor belongs to another runtime",
            ));
        }
        let pointer = self
            .scope
            .register(storage.allocation())
            .map_err(|error| crate::Error::runtime_state_source(op, error))?;
        Ok((pointer, storage.bytes()))
    }
}

/// Marks a [`NativeSessionRef`](tenferro_tensor::NativeSessionRef) as this engine's, so an
/// extension can reach the concrete engine through `&mut dyn Engine`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaEngineMarker;

/// Visit this engine through the tensor crate's engine trait, as an extension does.
///
/// Returns `None` when `engine` is not this backend's engine.
#[doc(hidden)]
pub fn with_cuda_engine<B, R>(engine: &mut B, f: impl FnOnce(&mut CudaEngine) -> R) -> Option<R>
where
    B: tenferro_tensor::Engine + ?Sized,
{
    let pointer = engine
        .native_engine()?
        .into_marked_ptr::<CudaEngineMarker>()?;
    // SAFETY: only this crate's `native_engine` creates a token carrying this marker, and it points
    // that token at a live `CudaEngine`; the `&mut B` keeps the exclusive borrow for the visit.
    Some(f(unsafe { pointer.cast::<CudaEngine>().as_mut() }))
}

impl CudaEngine {
    /// Run `f` inside this engine's raw scope.
    ///
    /// Every vendor call the closure makes must be enqueued on [`Session::stream`]; a buffer the
    /// caller drops afterwards is freed in that stream's order, so the call still sees it.
    ///
    /// # Errors
    ///
    /// The closure's own failure, and the runtime's when the scope cannot be opened.
    #[doc(hidden)]
    pub fn with_raw<R>(
        &mut self,
        _op: &'static str,
        f: impl FnOnce(&mut Session<'_>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        // The runtime handle is cloned first so the engine's caches can be borrowed mutably for the
        // same call.
        let runtime = self.runtime().clone();
        let caches = &self.extension_caches;
        // SAFETY: the closure only reads the addresses the session registers and enqueues its work
        // on this scope's stream.
        unsafe {
            runtime
                .with_raw_locked(Vec::new(), |scope| {
                    let mut session = Session {
                        runtime: runtime.clone(),
                        scope,
                        caches,
                    };
                    f(&mut session)
                })
                .and_then(|result| result)
        }
    }
}

/// A pinned host buffer used as the staging area of one download.
struct PinnedStaging {
    pointer: *mut std::ffi::c_void,
    bytes: usize,
}

impl PinnedStaging {
    fn new(bytes: usize, op: &'static str) -> crate::Result<PinnedStaging> {
        let mut pointer: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe {
            cudarc::driver::sys::cuMemHostAlloc(
                &mut pointer,
                bytes.max(1),
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
        };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::Error::runtime_state(
                op,
                format!("cuMemHostAlloc failed with {status:?}"),
            ));
        }
        Ok(PinnedStaging { pointer, bytes })
    }

    fn as_slice<T>(&self, count: usize) -> &[T] {
        // SAFETY: the allocation holds `bytes` pinned bytes, the copy above filled them after a
        // stream barrier, and `T` is plain data whose count fits the allocation.
        unsafe { std::slice::from_raw_parts(self.pointer as *const T, count) }
    }
}

impl Drop for PinnedStaging {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cuMemHostAlloc` and is freed once here.
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.pointer);
        }
        self.bytes = 0;
    }
}

/// A vendor resource lent from the engine's extension caches for the session's life.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::raw::CudaResourceGuard;
///
/// fn handle_lifetime(guard: &CudaResourceGuard<'_, u64>) -> u64 {
///     // The guard dereferences to the cached resource and releases it when dropped.
///     **guard + 1
/// }
///
/// # fn main() {}
/// ```
pub struct CudaResourceGuard<'s, T> {
    entry: std::cell::RefMut<'s, T>,
}

impl<T> std::ops::Deref for CudaResourceGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.entry
    }
}

impl<T> std::ops::DerefMut for CudaResourceGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.entry
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for CudaResourceGuard<'_, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.entry.fmt(formatter)
    }
}
