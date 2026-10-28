//! Pinned host buffers owned by one CUDA runtime.

use crate::cuda::CudaDeviceError;

use super::runtime::CudaRuntime;

/// A page-locked host allocation of one runtime.
///
/// The bytes are host-visible for their whole life; the allocation belongs to the
/// runtime that created it and is released when the last handle drops.
#[doc(hidden)]
pub struct PinnedHostBuffer {
    buffer: t4a_cuda_runtime::PinnedHostBuffer,
}

impl PinnedHostBuffer {
    /// Allocate `bytes` of pinned host memory in `runtime`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() {
    /// use tenferro_gpu::cuda::{PinnedHostBuffer, CudaDeviceId, CudaRuntime};
    ///
    /// # #[cfg(feature = "cuda")]
    /// if let Ok(runtime) = CudaRuntime::new(CudaDeviceId::from_ordinal(0)) {
    ///     let buffer = PinnedHostBuffer::new(&runtime, 16).unwrap();
    ///     assert_eq!(buffer.len(), 16);
    /// }
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the runtime cannot allocate pinned
    /// memory.
    #[doc(hidden)]
    pub fn new(runtime: &CudaRuntime, bytes: usize) -> Result<PinnedHostBuffer, CudaDeviceError> {
        let buffer = runtime.alloc_pinned(bytes)?;
        Ok(PinnedHostBuffer { buffer })
    }

    /// The pinned bytes, read-only.
    #[doc(hidden)]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// The pinned bytes, writable.
    #[doc(hidden)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut_slice()
    }

    /// Length in bytes.
    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("bytes", &self.len())
            .finish()
    }
}
