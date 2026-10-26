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
    pub fn retain_tensor<T: TensorScalar>(
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
        let runtime = self.runtime().clone();
        // SAFETY: the closure only reads the addresses the session registers and enqueues its work
        // on this scope's stream.
        unsafe {
            runtime
                .with_raw_locked(Vec::new(), |scope| {
                    let mut session = Session {
                        runtime: runtime.clone(),
                        scope,
                    };
                    f(&mut session)
                })
                .and_then(|result| result)
        }
    }
}
