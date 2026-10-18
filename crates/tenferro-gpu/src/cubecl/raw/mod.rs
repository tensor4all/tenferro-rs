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
//! - `Session`, [`StreamRef`], [`TensorRef`], [`DeviceBytes`], [`Module`], and
//!   [`Function`] are all scoped to the request lifetime and are `!Send +
//!   !Sync`, so the thread-local context/stream capability cannot migrate to
//!   another thread.
//! - `StreamRef` is not a `u64` and does not expose its native handle in safe
//!   code; vendor FFI receives the handle only through an explicit unsafe
//!   escape that must not retain it past the session scope.
//! - `TensorRef` carries a validated device span and ownership identity; it
//!   provides no `Deref` to the device buffer.
//! - View storage (an eager constant, or any metadata-only view) is bound in
//!   place by [`Session::tensor_read`], which resolves the base allocation,
//!   folds in the view's logical offset, and validates the reachable span. It
//!   accepts compact column-major views only: `TensorRef` has no per-axis
//!   stride metadata, so a strided view is rejected rather than materialized.
//! - A mutable device access can only be produced through [`Session::tensor_mut`]
//!   which requires an exclusive `&mut` borrow of a compatibly-typed tensor, or
//!   through a freshly allocated [`Session::alloc_output`] tensor.
//! - A raw kernel launch is an `unsafe fn` (see [`Session::launch`]) whose
//!   ABI, argument order, alias, and liveness invariants are the caller's
//!   contract.

mod module;
mod nvrtc;

// Crate-internal view of the launch-time opt-in decision, unit-tested without
// a device in `crate::cubecl::tests::raw_launch_tests`.
#[cfg(test)]
pub(crate) use module::needs_shared_memory_opt_in;

use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use tenferro_tensor::{TensorRank, TensorRead, TypedTensor, TypedTensorView};

use cudarc::cublas::sys as cublas_sys;

use super::runtime::CudaRuntime;
use super::{CudaExtensionCache, CudaExtensionCacheGuard, CudaRuntimeIdentity};

pub use nvrtc::NvrtcOptions;

/// A loaded CUDA module (PTX or CUBIN) on the tenferro primary context.
///
/// Retains the tenferro `CudaRuntime` so the context/device stays alive as
/// long as the module exists. Every [`Function`] derived from it retains the
/// same inner handle (`Rc`), so the image is never unloaded while a function
/// handle is still in use. Dropping the last strong reference unloads the
/// module.
///
/// `Module` is `!Send + !Sync` (like the rest of the raw seam), matching the
/// session-scoped execution authority of issue #1597.
#[allow(dead_code)]
// `runtime` is never dereferenced: holding the `CudaRuntime` Arc keeps the
// tenferro primary context alive for the module's whole lifetime, which is
// the load-bearing effect this field provides.
pub struct Module {
    inner: Rc<ModuleInner>,
    runtime: CudaRuntime,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Shared inner handle shared between a [`Module`] and its [`Function`]s.
///
/// Retains the tenferro primary context identity so unload re-activates the
/// correct context on the (same, session-scoped) thread before the driver
/// frees the image. `Module`/`Function` are `!Send + !Sync`, so this drop
/// never races a foreign current-context.
pub(crate) struct ModuleInner {
    handle: cudarc::driver::sys::CUmodule,
    context: cudarc::driver::sys::CUcontext,
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `cuModuleLoadData` under the tenferro
        // primary context and is not yet unloaded; every `Function` that
        // could still reference it retains an `Rc<ModuleInner>`. The module
        // is `!Send + !Sync`, so this drop runs on the session thread.
        unsafe {
            let was_current = cudarc::driver::result::ctx::get_current().ok();
            let _ = cudarc::driver::result::ctx::set_current(self.context);
            let _ = cudarc::driver::result::module::unload(self.handle);
            if let Some(previous) = was_current.flatten() {
                if previous != self.context {
                    let _ = cudarc::driver::result::ctx::set_current(previous);
                }
            }
        }
    }
}

impl Module {
    /// Wrap a freshly loaded driver module handle, retaining `runtime` so the
    /// primary context stays alive for the module's lifetime.
    pub(crate) fn from_handle(
        op: &'static str,
        handle: cudarc::driver::sys::CUmodule,
        runtime: CudaRuntime,
    ) -> crate::Result<Self> {
        if handle.is_null() {
            return Err(crate::Error::backend_source(
                op,
                std::io::Error::other("driver returned a null module handle"),
            ));
        }
        let context = runtime.primary_context();
        Ok(Self {
            inner: Rc::new(ModuleInner { handle, context }),
            runtime,
            _not_send_sync: PhantomData,
        })
    }

    /// Look up a kernel entry point by name.
    ///
    /// The returned [`Function`] retains this module, so the module cannot be
    /// unloaded while the function is alive.
    ///
    /// # Errors
    ///
    /// Returns the driver's typed error via [`crate::Error::BackendSource`]
    /// when the symbol does not exist in the module.
    pub fn function(&self, name: &str) -> crate::Result<Function> {
        module::module_function(&self.inner, name, "module.function")
    }

    /// Return the CUDA module handle (for internal bridge and test use).
    #[allow(dead_code)]
    pub(crate) fn handle(&self) -> cudarc::driver::sys::CUmodule {
        self.inner.handle
    }
}

/// A handle to a kernel within a loaded [`Module`].
///
/// Retains the owning module inner (`Rc`), so the underlying function handle
/// is valid for as long as this handle lives and the module is not unloaded
/// mid-flight.
#[allow(dead_code)]
// `module` is never dereferenced: the retained `Rc<ModuleInner>` prevents
// module unload while a queued kernel may still reference the function handle,
// which is the load-bearing effect this field provides.
pub struct Function {
    handle: cudarc::driver::sys::CUfunction,
    module: Rc<ModuleInner>,
    /// Largest dynamic shared-memory budget this function was opted into with
    /// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`). `0` means "the
    /// driver's default", i.e. no attribute was set yet.
    ///
    /// The attribute only ever grows: a launch that requests no more than this
    /// value reuses it instead of paying another driver call (#2085).
    shared_mem_opt_in: std::cell::Cell<u32>,
}

impl Function {
    /// Return the raw function handle (for internal launch use).
    pub(crate) fn handle(&self) -> cudarc::driver::sys::CUfunction {
        self.handle
    }

    /// Largest dynamic shared-memory budget already opted into on this handle.
    pub(crate) fn shared_mem_opt_in(&self) -> u32 {
        self.shared_mem_opt_in.get()
    }

    /// Record a successful dynamic shared-memory opt-in on this handle.
    pub(crate) fn set_shared_mem_opt_in(&self, bytes: u32) {
        self.shared_mem_opt_in.set(bytes);
    }
}

/// Launch geometry for one raw CUDA kernel.
///
/// `grid` is the thread-block grid `[x, y, z]`; `block` is the per-block
/// thread dimensions `[x, y, z]`; `shared_mem_bytes` is the dynamic
/// shared-memory budget per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Grid dimensions `[x, y, z]`.
    pub grid: [u32; 3],
    /// Block dimensions `[x, y, z]`.
    pub block: [u32; 3],
    /// Dynamic shared-memory bytes per block.
    ///
    /// Up to 48 KiB (49152 bytes) needs no opt-in. A larger request makes the
    /// following [`Session::launch`] raise
    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on the launched
    /// [`Function`] once; see [`Session::launch`] for the device limit.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// A flat one-dimensional launch of `threads` threads in blocks of `block`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] carrying
    /// `ValidationError::InvalidArgument` when `block == 0`.
    pub fn flat(threads: u32, block: u32, shared_mem_bytes: u32) -> crate::Result<Self> {
        if block == 0 {
            return Err(crate::Error::invalid_argument(
                "launch_config.flat",
                "block",
                "block size must be non-zero",
            ));
        }
        let grids = threads.div_ceil(block);
        Ok(Self {
            grid: [grids.max(1), 1, 1],
            block: [block, 1, 1],
            shared_mem_bytes,
        })
    }
}

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

    /// Dispatch every CubeCL launch queued so far onto this session's stream.
    ///
    /// A raw session does **not** order a `#[cube]` launch issued inside it
    /// before later raw work: CubeCL hands launches to its device thread while
    /// raw kernels and vendor calls are enqueued from this thread, so a raw
    /// kernel can reach the shared stream first, and the raw reader then sees
    /// the value the launch was about to overwrite (tenferro-rs #2092). Call
    /// this after the launch and before the raw work that depends on it.
    ///
    /// Entering a raw session already dispatches the launches queued *before*
    /// the scope, and leaving a CubeCL session dispatches those issued inside
    /// it; only a launch issued after this scope was entered needs the fence.
    ///
    /// This is a dispatch fence, not a result barrier: it hands the queued
    /// launches to CubeCL's device thread, which can also retire the drop queue
    /// behind a stream fence, and returns without reading any result back.
    /// Use [`Session::synchronize`] when the host must wait for device
    /// completion, and a download when it needs the values. A launch that fails
    /// is reported by the buffer it left unwritten, not here.
    ///
    /// # Examples
    ///
    /// Hardware-gated: a machine without a CUDA device returns early, and a
    /// CUDA host runs the fence inside a real raw session.
    ///
    /// ```
    /// use tenferro_gpu::cuda::{
    ///     cuda_devices, gpu_available, with_cuda_exec_session, CudaBackend, CudaExecSession,
    /// };
    /// use tenferro_tensor::backend::BackendSessionHost;
    ///
    /// fn flush(session: &mut CudaExecSession<'_>) -> tenferro_tensor::Result<()> {
    ///     session.with_raw("docs.flush_cubecl", |raw| raw.flush_cubecl())
    /// }
    ///
    /// // `gpu_available` never panics without a CUDA driver.
    /// if !gpu_available() {
    ///     return Ok(());
    /// }
    /// let device = cuda_devices()?
    ///     .first()
    ///     .map(|device| device.id())
    ///     .ok_or_else(|| {
    ///         tenferro_tensor::Error::runtime_state("docs.flush_cubecl", "no CUDA device")
    ///     })?;
    /// let mut backend = CudaBackend::new(device)?;
    /// backend
    ///     .with_backend_session(|session| with_cuda_exec_session(session, flush))?
    ///     .ok_or_else(|| {
    ///         tenferro_tensor::Error::runtime_state("docs.flush_cubecl", "not a CUDA backend")
    ///     })??;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot hand the
    /// queued launches to the device thread. A failure of the launched work
    /// itself is not reported here; it surfaces on the buffer the work left
    /// unwritten.
    pub fn flush_cubecl(&self) -> crate::Result<()> {
        self.runtime.flush_cubecl("raw::Session::flush_cubecl")
    }

    /// Borrow the captured execution stream.
    pub fn stream(&self) -> StreamRef<'s> {
        StreamRef {
            raw: self.stream,
            _scope: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Run one cuBLAS enqueue with the runtime's stream-bound handle.
    ///
    /// The handle belongs to the runtime and is cached per CubeCL stream slot,
    /// already bound to this session's stream, and re-bound by tenferro when the
    /// slot changes. A raw session therefore does not have to create and destroy
    /// its own handle: on an RTX 3060 that pattern costs about 0.45 ms and 10 MiB
    /// per session, against the 7-10 ms of an eight-token forward that uses it
    /// (tenferro-rs #2083). The per-stream slot lock is held across `execute`, so
    /// concurrent sessions of one runtime cannot race the handle's mutable
    /// state, and the handle is usable only inside the closure.
    ///
    /// `pointer_mode` selects how cuBLAS reads scalar arguments: pass
    /// [`cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST`] for host scalars such as
    /// `&alpha`, and `..._DEVICE` for device-resident scalars.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Io`] when the cuBLAS library is not loadable,
    /// [`crate::Error::RuntimeState`] when the handle cache cannot be locked, or
    /// [`crate::Error::BackendSource`] when cuBLAS rejects the configuration or
    /// the enqueue.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cudarc::cublas::sys as cublas_sys;
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible.
    /// fn check(session: &mut CudaExecSession<'_>) -> tenferro_tensor::Result<()> {
    ///     session.with_raw("test.raw_cublas", |raw| {
    ///         raw.with_cublas_handle(
    ///             "test.raw_cublas",
    ///             cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
    ///             |_handle| Ok(()),
    ///         )
    ///     })?;
    ///     Ok(())
    /// }
    /// let _ = check;
    /// ```
    pub fn with_cublas_handle<R>(
        &self,
        op: &'static str,
        pointer_mode: cublas_sys::cublasPointerMode_t,
        execute: impl FnOnce(cublas_sys::cublasHandle_t) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.runtime
            .with_cublas_handle(op, pointer_mode, Vec::new(), execute)
    }

    /// Build a checked read-only device reference for a GPU-backed tensor.
    ///
    /// The returned reference is tied to both this session borrow and the
    /// tensor borrow, so the span cannot outlive the tensor's allocation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident on
    /// this session's runtime/device.
    pub fn tensor<'a, T>(
        &'a self,
        tensor: &'a TypedTensor<T, impl TensorRank>,
    ) -> crate::Result<TensorRef<'a, T>>
    where
        T: 'static,
    {
        super::dispatch::ensure_resident_on_runtime(&self.runtime, tensor, "raw.tensor")?;
        let prepared = super::dispatch::cubecl_buffer(tensor, "raw.tensor")?;
        let byte_len = prepared.byte_len;
        let base = self.device_ptr(prepared.handle().clone(), "raw.tensor")?;
        Ok(TensorRef {
            base,
            byte_len,
            _scope: PhantomData,
            _dtype: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Build a checked mutable device reference for a GPU-backed tensor.
    ///
    /// Requires an exclusive borrow of the tensor, so aliasing mutable device
    /// access cannot be produced from a shared `&TypedTensor<T>`. The returned
    /// reference is tied to both this session borrow and the tensor borrow, so
    /// the span cannot outlive the tensor's allocation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident on
    /// this session's runtime/device.
    pub fn tensor_mut<'a, T>(
        &'a self,
        tensor: &'a mut TypedTensor<T, impl TensorRank>,
    ) -> crate::Result<TensorMut<'a, T>>
    where
        T: 'static,
    {
        super::dispatch::ensure_resident_on_runtime(&self.runtime, tensor, "raw.tensor_mut")?;
        let prepared = super::dispatch::cubecl_buffer(tensor, "raw.tensor_mut")?;
        let byte_len = prepared.byte_len;
        let base = self.device_ptr(prepared.handle().clone(), "raw.tensor_mut")?;
        Ok(TensorMut {
            base,
            byte_len,
            _scope: PhantomData,
            _dtype: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Build a checked read-only device reference for any device-resident read
    /// target, including view storage such as an eager constant.
    ///
    /// [`Session::tensor`] takes an owned [`TypedTensor`], but a value
    /// registered by `constant_from_host` is held as a view over its device
    /// allocation: [`TensorRead::as_tensor`] is `None` for it, so there is no
    /// owned tensor to pass. This method binds that storage in place. It
    /// resolves the view's base CubeCL allocation, applies the view's logical
    /// element offset, and validates the reachable byte span against that
    /// allocation. No device copy or re-allocation is made, so a model weight
    /// does not have to be duplicated into an owned tensor to reach a kernel.
    ///
    /// The binding is **compact column-major only**: a [`TensorRef`] carries
    /// one base pointer and one byte length, and the raw seam does not carry
    /// per-axis stride metadata, so a strided view cannot be described safely
    /// by it. A noncompact or transposed view is rejected with
    /// [`crate::Error::Validation`] instead of being materialized behind the
    /// caller's back; canonicalize such a source through the backend first.
    /// The returned span starts at the view's first element and covers exactly
    /// its `n_elements` elements, never the whole backing allocation.
    ///
    /// Owned storage keeps the exact [`Session::tensor`] binding, so the
    /// existing owned kernel path is unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::raw::{Session, TensorRef};
    /// use tenferro_tensor::TensorValue;
    ///
    /// fn check<'s, 't>(
    ///     raw: &'s Session<'s>,
    ///     value: &'t TensorValue,
    /// ) -> tenferro_tensor::Result<TensorRef<'s, f64>>
    /// where
    ///     't: 's,
    /// {
    ///     let read = value.tensor_read();
    ///     raw.tensor_read::<f64>(&read)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with
    /// [`tenferro_tensor::ValidationError::DTypeMismatch`] when the read's
    /// dtype is not `T`, or with `InvalidArgument` when the read is not a
    /// compact column-major view whose span fits its allocation, and
    /// [`crate::Error::RuntimeState`] when the read is not resident on this
    /// session's runtime/device.
    pub fn tensor_read<'a, 'b, T>(
        &'a self,
        read: &TensorRead<'b>,
    ) -> crate::Result<TensorRef<'a, T>>
    where
        T: tenferro_tensor::TensorScalar + 'static,
        'b: 'a,
    {
        const OP: &str = "raw.tensor_read";
        if read.dtype() != T::dtype() {
            return Err(crate::Error::dtype_mismatch(OP, T::dtype(), read.dtype()));
        }
        if let Some(tensor) = read.as_tensor() {
            // Owned storage keeps the exact `Session::tensor` binding: the read
            // already names an owned tensor with the requested dtype.
            let typed = tensor
                .as_typed::<T>()
                .ok_or_else(|| crate::Error::dtype_mismatch(OP, T::dtype(), tensor.dtype()))?;
            return self.tensor(typed);
        }
        let (base, byte_len) = self.compact_view_span_from_read(read, OP)?;
        Ok(TensorRef {
            base,
            byte_len,
            _scope: PhantomData,
            _dtype: PhantomData,
            _not_send_sync: PhantomData,
        })
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

    /// Retain a clone of a resident tensor's CubeCL allocation handle.
    ///
    /// The returned [`DeviceBytes`] holds a reference-counted clone of the
    /// tensor's allocation handle, so the device memory stays alive until the
    /// guard is dropped. Use this when a vendor library enqueues asynchronous
    /// work against the tensor's address and, on a failed synchronization
    /// barrier, the guard must be intentionally forgotten so allocation
    /// reclamation cannot race an in-flight kernel.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when `tensor` is host-backed,
    /// belongs to a non-CubeCL backend family, belongs to a different CUDA
    /// runtime domain, or is not resident on this session's device, or
    /// [`crate::Error::BackendSource`] when CubeCL cannot inspect the retained
    /// resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::raw::{DeviceBytes, Session};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// fn check<'s>(
    ///     raw: &Session<'s>,
    ///     tensor: &TypedTensor<f32>,
    /// ) -> tenferro_tensor::Result<DeviceBytes<'s>> {
    ///     raw.retain_tensor(tensor, "test.retain_tensor")
    /// }
    /// ```
    pub fn retain_tensor<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<DeviceBytes<'s>>
    where
        T: 'static,
    {
        let inner = super::interop::retain_tensor_bytes(&self.runtime, tensor, op)?;
        Ok(DeviceBytes {
            inner,
            _scope: PhantomData,
            _not_send_sync: PhantomData,
        })
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
            _not_send_sync: PhantomData,
        })
    }

    /// Copy bytes from one device span to another on the session stream.
    ///
    /// Both spans must be produced by this session (via [`Session::tensor`],
    /// [`Session::tensor_mut`], or [`Session::alloc_bytes`]) and must not
    /// overlap; the copy is stream-ordered with other enqueued work. This is
    /// the raw equivalent of a same-device `memcpyDeviceToDeviceAsync`.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be aligned device spans of at least `nbytes`
    /// bytes, backed by allocations that outlive the (unsynchronized) copy;
    /// `dst` must be uniquely owned for writing and `src` must not alias it.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the driver rejects the
    /// device-to-device copy.
    pub unsafe fn copy_bytes(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        nbytes: usize,
        op: &'static str,
    ) -> crate::Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        let stream = self.stream as *mut cudarc::runtime::sys::CUstream_st;
        cudarc::runtime::result::memcpy_dtod_async(dst, src, nbytes, stream)
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    /// Upload host bytes into a CubeCL-owned workspace returned as
    /// [`DeviceBytes`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot upload or
    /// inspect the workspace resource, or [`crate::Error::Validation`] when
    /// the pointer address cannot be represented as `usize`.
    pub fn upload_bytes(&self, bytes: &[u8], op: &'static str) -> crate::Result<DeviceBytes<'s>> {
        let inner = super::interop::upload_device_bytes(&self.runtime, bytes, op)?;
        Ok(DeviceBytes {
            inner,
            _scope: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Read a resident tensor's bytes back to host memory.
    ///
    /// Synchronizes the session stream, then returns the tensor data as a
    /// host-resident typed tensor (same shape). This is the only host barrier
    /// introduced by the raw-session path; call it only where a value must be
    /// inspected or returned to the host.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this session's runtime/device, or [`crate::Error::BackendSource`]
    /// when synchronization or readback fails.
    pub fn download_tensor<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: cubecl::prelude::CubeElement
            + tenferro_tensor::TensorScalar
            + cubecl::frontend::Scalar
            + Clone
            + Send
            + Sync
            + 'static,
    {
        super::interop::download_typed_tensor(&self.runtime, tensor, op)
    }

    /// Load a CUDA module from PTX source text.
    ///
    /// The PTX is compiled for the current device's architecture by the
    /// driver; on older toolchains pass PTX compiled with a compatible
    /// `compute_XX` target. The raw session context is current, so no explicit
    /// context switch is needed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the driver rejects the
    /// image (unsupported format, architecture mismatch).
    pub fn load_ptx(&self, ptx: &std::ffi::CStr) -> crate::Result<Module> {
        module::load_module_data(ptx.as_ptr().cast(), self.runtime.clone(), "raw.load_ptx")
    }

    /// Load a CUDA module from CUBIN binary data.
    ///
    /// CUBIN is architecture-specific; it must have been compiled for the
    /// current device.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the driver rejects the
    /// image (mismatched architecture, corrupt data).
    pub fn load_cubin(&self, cubin: &[u8]) -> crate::Result<Module> {
        module::load_module_data(
            cubin.as_ptr().cast(),
            self.runtime.clone(),
            "raw.load_cubin",
        )
    }

    /// Compile CUDA source to PTX on the host using NVRTC, then load it.
    ///
    /// The resulting module is loaded on the session's primary context.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when NVRTC compilation or the
    /// driver load fails, or [`crate::Error::Validation`] when the source
    /// contains a NUL byte.
    pub fn compile_nvrtc(&self, src: &str, opts: &NvrtcOptions) -> crate::Result<Module> {
        let ptx = nvrtc::compile_nvrtc(src, opts)?;
        let ptx_src = ptx.to_src();
        // SAFETY: the freshly compiled PTX has no interior NULs (NVRTC output
        // is a well-formed CUDA image string).
        let cstr = std::ffi::CString::new(ptx_src).map_err(|_| {
            crate::Error::invalid_argument("raw.compile_nvrtc", "ptx", "PTX contains NUL")
        })?;
        self.load_ptx(&cstr)
    }

    /// Launch a raw CUDA kernel on the session's captured stream.
    ///
    /// The kernel is enqueued, not synchronized; call [`Session::synchronize`]
    /// for a host barrier.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of the following for the duration of the
    /// launch and until the work is synchronized:
    ///
    /// - **ABI/arguments**: every [`KernelArg`] matches the kernel's formal
    ///   parameter list in order (scalars of the exact width/type, device
    ///   pointers exactly where the kernel expects them). `TensorRef` args
    ///   that the kernel writes must be passed as the mutable form.
    /// - **Read/write ranges**: the kernel only reads/writes within the span
    ///   validated by the tensor's [`TensorRef`]/[`TensorMut`] and only where
    ///   the launch geometry covers; out-of-bounds access is UB.
    /// - **Aliasing**: no two args alias the same device memory unless the
    ///   kernel contract permits it.
    /// - **Liveness**: every referenced device allocation and the [`Module`]
    ///   (via its [`Function`]) outlive the asynchronous work; the caller must
    ///   not drop the tensors or module until after a subsequent
    ///   [`Session::synchronize`].
    ///
    /// Dynamic shared memory above 48 KiB needs a per-function opt-in that the
    /// driver does not grant by default (since Volta). `launch` performs it
    /// implicitly: before the first launch whose
    /// [`LaunchConfig::shared_mem_bytes`] exceeds 48 KiB, it raises
    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on `function` to that
    /// byte count, and it calls the driver again only when a later launch asks
    /// for more. A request at or below a value already set, or at or below the
    /// 48 KiB default, is launched without a driver call.
    ///
    /// The device cap is
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, readable as
    /// `CubeCL`'s `client().properties().hardware.max_shared_memory_size`
    /// through
    /// [`CudaExecSession::with_cubecl`](super::exec_session::CudaExecSession::with_cubecl).
    /// A request above it is rejected by the driver and returned as
    /// [`crate::Error::BackendSource`]; the value is never clamped.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] carrying
    /// `ValidationError::InvalidArgument` when the launch geometry or limits
    /// are invalid, or [`crate::Error::BackendSource`] when the driver rejects
    /// the dynamic shared-memory opt-in or the launch.
    pub unsafe fn launch(
        &self,
        function: &Function,
        config: LaunchConfig,
        args: &[KernelArg<'_>],
    ) -> crate::Result<()> {
        module::validate_launch_config(&config)?;
        module::ensure_dynamic_shared_memory(function, config.shared_mem_bytes)?;

        // Build stable storage for every argument: scalar values live in heap
        // boxes (the param slot points directly at the value bytes), device
        // pointers live in heap boxes (the param slot points at the location
        // holding the device address, i.e. pointer-to-pointer, as CUDA kernel
        // params require). `cuLaunchKernel` copies the parameter list
        // synchronously at enqueue, so the storage only needs to stay alive
        // for this call.
        let mut scalar_storage: Vec<Box<[u8]>> = Vec::with_capacity(args.len());
        let mut ptr_storage: Vec<Box<*mut std::ffi::c_void>> = Vec::with_capacity(args.len());
        let mut kernel_params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                KernelArg::Scalar(bytes) => {
                    scalar_storage.push(bytes.clone().into_boxed_slice());
                    // The slot was just pushed, so `len - 1` is always valid;
                    // index instead of unwrapping to keep `launch` panic-free.
                    let slot = &scalar_storage[scalar_storage.len() - 1];
                    let last_ptr = slot.as_ptr() as *const std::ffi::c_void;
                    kernel_params.push(last_ptr as *mut std::ffi::c_void);
                }
                KernelArg::DevicePtr(ptr, _) => {
                    // The Box's heap allocation is stable across the move into
                    // `ptr_storage`; the parameter slot points at the location
                    // holding the device address (pointer-to-pointer, as CUDA
                    // kernel params require), and is pushed as a byte address.
                    let mut slot_box = Box::new(*ptr);
                    let slot = (&mut *slot_box) as *mut *mut std::ffi::c_void;
                    kernel_params.push(slot as *mut std::ffi::c_void);
                    ptr_storage.push(slot_box);
                }
            }
        }

        // SAFETY: the caller upholds the ABI/liveness contract above; the
        // function handle comes from a module retained by `function`, and the
        // parameter storage is alive for the (synchronous) enqueue call.
        let stream = self.stream as *mut cudarc::driver::sys::CUstream_st;
        cudarc::driver::result::launch_kernel(
            function.handle(),
            (config.grid[0], config.grid[1], config.grid[2]),
            (config.block[0], config.block[1], config.block[2]),
            config.shared_mem_bytes,
            stream,
            &mut kernel_params,
        )
        .map_err(|err| crate::Error::backend_source("raw.launch", err))
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
    /// Returns the initializer's typed error, or
    /// [`crate::Error::RuntimeState`] when the extension cache is poisoned.
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

impl Session<'_> {
    /// Resolve a CubeCL allocation handle to its base CUDA device address.
    ///
    /// A retained handle keeps its storage offset and is never deallocated
    /// while one of its slices is live (see the `CubeclBuffer::device_addr`
    /// invariant), so the address is stable for the borrowing tensor or view.
    /// Resolving through `get_resource` is also what aligns this access with
    /// CubeCL's stream (issue #1868).
    fn device_ptr(
        &self,
        handle: cubecl_runtime::server::Handle,
        op: &'static str,
    ) -> crate::Result<*mut std::ffi::c_void> {
        let resource = self
            .runtime
            .client()
            .get_resource::<cubecl_cuda::ffi_interop::CudaServer>(handle)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        super::interop::cuda_device_ptr_from_addr(resource.resource().ptr, op)
    }

    /// Resolve a view read's compact device span through its concrete scalar.
    ///
    /// [`TensorRead`] is dtype-erased while the raw binding returns a typed
    /// [`TensorRef`], so the view has to be recovered as a typed view before a
    /// span can be described. Dispatching on the read's dtype keeps every
    /// preset scalar on the same checked path instead of guessing a
    /// representation.
    fn compact_view_span_from_read(
        &self,
        read: &TensorRead<'_>,
        op: &'static str,
    ) -> crate::Result<(*mut std::ffi::c_void, usize)> {
        tenferro_tensor::with_scalar_read!(
            read.clone(),
            all,
            backend = tenferro_tensor::BackendId::Cuda,
            op = op,
            |view| -> crate::Result<(*mut std::ffi::c_void, usize)> {
                self.compact_view_span(&view, op)
            }
        )
    }

    /// Resolve and validate the compact device span of a resident view.
    ///
    /// The provider preparation covers the whole backing allocation, so the
    /// view's element offset is folded into the returned base pointer here and
    /// the reachable span is checked against the allocation before any pointer
    /// is handed out.
    fn compact_view_span<T>(
        &self,
        view: &TypedTensorView<'_, T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<(*mut std::ffi::c_void, usize)>
    where
        T: tenferro_tensor::TensorScalar + 'static,
    {
        super::dispatch::ensure_view_resident_on_runtime(&self.runtime, view, op)?;
        if !view.is_col_major_contiguous()? {
            return Err(crate::Error::invalid_argument(
                op,
                "read",
                format!(
                    "raw binding requires a compact column-major view; got {} \
                     (canonicalize a strided view through the backend first)",
                    view.layout_summary()
                ),
            ));
        }
        let prepared = super::dispatch::prepared_view_access(view, op)?;
        let allocation_byte_len = prepared.byte_len(op)?;
        let element_size = std::mem::size_of::<T>();
        let layout_error =
            |message: &'static str| crate::Error::invalid_argument(op, "read", message);
        let byte_offset = usize::try_from(view.offset())
            .ok()
            .and_then(|offset| offset.checked_mul(element_size))
            .ok_or_else(|| layout_error("view byte offset is negative or overflows"))?;
        let byte_len = view
            .n_elements()
            .checked_mul(element_size)
            .ok_or_else(|| layout_error("view byte length overflows"))?;
        let span_end = byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| layout_error("view byte span overflows"))?;
        if span_end > allocation_byte_len {
            return Err(crate::Error::invalid_argument(
                op,
                "read",
                format!(
                    "compact view span [offset {byte_offset}, end {span_end}) exceeds the \
                     {allocation_byte_len}-byte backing allocation"
                ),
            ));
        }
        let base = self.device_ptr(prepared.into_handle(), op)?;
        // The offset and length were just validated to end inside the resolved
        // allocation, so the exposed address stays within the view's own span.
        Ok((base.cast::<u8>().wrapping_add(byte_offset).cast(), byte_len))
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
    _not_send_sync: PhantomData<Rc<()>>,
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

/// Checked read-only device span of a GPU-backed tensor or view.
///
/// The span is the one [`Session::tensor`] or [`Session::tensor_read`] validated
/// for its source. For a view it starts at the view's first element and covers
/// only the view's own elements, not the whole backing allocation.
pub struct TensorRef<'s, T> {
    base: *mut std::ffi::c_void,
    byte_len: usize,
    _scope: PhantomData<&'s ()>,
    _dtype: PhantomData<T>,
    _not_send_sync: PhantomData<Rc<()>>,
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
    _not_send_sync: PhantomData<Rc<()>>,
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

/// A typed argument for a raw CUDA kernel launch.
///
/// Either a value scalar (copied into a stable buffer) or a device pointer
/// obtained from a validated [`TensorRef`]/[`TensorMut`] or [`DeviceBytes`].
/// Constructing an argument is safe; the ABI contract is enforced by the
/// caller at the `unsafe` launch site.
#[derive(Clone)]
pub enum KernelArg<'a> {
    /// A scalar value encoded as an exact-size byte sequence.
    Scalar(Vec<u8>),
    /// A device memory address argument (in/out pointer).
    DevicePtr(*mut std::ffi::c_void, PhantomData<&'a ()>),
}

impl<'a> KernelArg<'a> {
    /// Build a scalar argument from an exact-size value encoding.
    pub fn scalar(bytes: &[u8]) -> Self {
        Self::Scalar(bytes.to_vec())
    }

    /// Build a u32 scalar parameter.
    pub fn u32(value: u32) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build an i32 scalar parameter.
    pub fn i32(value: i32) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build a f32 scalar parameter.
    pub fn f32(value: f32) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build a u64 scalar parameter.
    pub fn u64(value: u64) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build an i64 scalar parameter.
    pub fn i64(value: i64) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build a f64 scalar parameter.
    pub fn f64(value: f64) -> Self {
        Self::scalar(&value.to_ne_bytes())
    }

    /// Build a device-pointer parameter from a read-only tensor reference.
    ///
    /// The kernel may only read the tensor unless the launch safety contract
    /// states otherwise; for a write target use [`KernelArg::tensor_mut`] or
    /// [`KernelArg::output`].
    pub fn tensor<T>(reference: &TensorRef<'a, T>) -> Self {
        // SAFETY: `raw_ptr` exposes the validated device span; the pointer is
        // only used here to form the launch argument while the session scope
        // `'a` is alive, which the borrow enforces.
        Self::DevicePtr(unsafe { reference.raw_ptr() }, PhantomData)
    }

    /// Build a device-pointer parameter from a mutable tensor reference.
    pub fn tensor_mut<T>(reference: &TensorMut<'a, T>) -> Self {
        // SAFETY: exclusive `&mut` borrow guarantees no aliasing for `'a`.
        Self::DevicePtr(unsafe { reference.raw_ptr() }, PhantomData)
    }

    /// Build a device-pointer parameter from a freshly allocated output tensor.
    pub fn output<T>(reference: &TensorMut<'a, T>) -> Self {
        Self::tensor_mut(reference)
    }

    /// Build a device-pointer parameter from a workspace allocation.
    pub fn workspace(bytes: &DeviceBytes<'a>) -> Self {
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        bytes.with_ptr(|p| ptr = p);
        Self::DevicePtr(ptr, PhantomData)
    }
}

/// CubeCL-owned device byte workspace kept alive for CUDA library calls.
pub struct DeviceBytes<'s> {
    inner: super::interop::DeviceByteBuffer,
    _scope: PhantomData<&'s ()>,
    _not_send_sync: PhantomData<Rc<()>>,
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
