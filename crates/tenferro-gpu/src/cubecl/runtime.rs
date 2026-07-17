//! CubeCL CUDA runtime initialization and synchronization.

use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime as CubeclCudaRuntime};
use cudarc::driver::sys::{CUcontext, CUdevice};
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};

/// Returns `true` if a CUDA device is available for CubeCL.
///
/// Use this in test helpers to skip GPU tests on machines without hardware.
pub fn gpu_available() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = CudaDevice::new(0);
        let _ = CubeclCudaRuntime::client(&device);
    }))
    .is_ok()
}

/// CubeCL CUDA runtime wrapper.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::CudaRuntime;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<CudaRuntime> = CudaRuntime::new;
/// let _sync: fn(&CudaRuntime) -> tenferro_tensor::Result<()> =
///     CudaRuntime::synchronize;
/// ```
pub struct CudaRuntime {
    client: ComputeClient<CubeclCudaRuntime>,
    device_ordinal: usize,
    primary_context: CudaPrimaryContext,
}

impl fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

// SAFETY: CUDA primary contexts and CubeCL clients are owned handles. Backend
// methods set the context current before raw CUDA-library calls, and
// higher-level eager execution serializes backend access through a mutex.
unsafe impl Send for CudaRuntime {}

struct CudaPrimaryContext {
    cuda_device: CUdevice,
    cuda_context: CUcontext,
}

impl CudaPrimaryContext {
    fn retain(cuda_device: CUdevice) -> crate::Result<Self> {
        let cuda_context = unsafe { cudarc::driver::result::primary_ctx::retain(cuda_device) }
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        Ok(Self {
            cuda_device,
            cuda_context,
        })
    }

    fn context(&self) -> CUcontext {
        self.cuda_context
    }
}

impl Drop for CudaPrimaryContext {
    fn drop(&mut self) {
        if let Err(err) = unsafe { cudarc::driver::result::primary_ctx::release(self.cuda_device) }
        {
            report_cuda_primary_context_release_error(&err);
        }
    }
}

#[cold]
fn report_cuda_primary_context_release_error(err: &impl fmt::Debug) {
    eprintln!("tenferro-gpu: failed to release CUDA primary context during Drop: {err:?}");
}

#[cold]
fn report_cuda_runtime_drop_error(err: &crate::Error) {
    eprintln!("tenferro-gpu: failed to synchronize CUDA runtime during Drop: {err}");
}

impl CudaRuntime {
    /// Initialize the CubeCL CUDA runtime on the given device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaRuntime;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<CudaRuntime> = CudaRuntime::new;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CUDA device selection,
    /// driver initialization, primary-context retention, or CubeCL client
    /// creation fails.
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        // INVARIANT: CUDA ordinals are device identifiers, not tensor extents;
        // an unrepresentable ordinal becomes a CUDA invalid-device error.
        cudarc::runtime::result::device::set(device_ordinal as i32)
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        cudarc::driver::result::init()
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        let cuda_device = cudarc::driver::result::device::get(device_ordinal as i32)
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        let primary_context = CudaPrimaryContext::retain(cuda_device)?;
        unsafe { cudarc::driver::result::ctx::set_current(primary_context.context()) }
            .map_err(|err| crate::Error::backend_source("cubecl_runtime_init", err))?;
        let device = CudaDevice::new(device_ordinal);
        let client = CubeclCudaRuntime::client(&device);
        Ok(Self {
            client,
            device_ordinal,
            primary_context,
        })
    }

    pub(crate) fn client(&self) -> &ComputeClient<CubeclCudaRuntime> {
        &self.client
    }

    /// Return the CUDA device ordinal that this runtime targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaRuntime;
    ///
    /// let _device_ordinal: fn(&CudaRuntime) -> usize = CudaRuntime::device_ordinal;
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    #[doc(hidden)]
    pub fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
        // INVARIANT: CUDA ordinals are device identifiers; bad ordinals are
        // reported by CUDA instead of indexing memory in tenferro.
        cudarc::runtime::result::device::set(self.device_ordinal as i32)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        unsafe { cudarc::driver::result::ctx::set_current(self.primary_context.context()) }
            .map_err(|err| crate::Error::backend_source(op, err))
    }

    pub(crate) fn raw_cuda_stream(&self) -> crate::Result<u64> {
        self.client
            .with_server(|server| {
                server
                    .raw_stream(StreamId::current())
                    .map(|stream| stream as u64)
                    .map_err(|err| crate::Error::backend_source("raw_cuda_stream", err))
            })
            .ok_or_else(|| {
                crate::Error::runtime_state("raw_cuda_stream", "CubeCL server is unavailable")
            })?
    }

    /// Block the current thread until work submitted to the current CUDA stream completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaRuntime;
    ///
    /// let _sync: fn(&CudaRuntime) -> tenferro_tensor::Result<()> =
    ///     CudaRuntime::synchronize;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when CubeCL cannot expose the
    /// current stream, or [`crate::Error::BackendSource`] when CUDA context or
    /// stream synchronization fails.
    pub fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "cubecl_runtime_synchronize";
        self.set_current_cuda_context(OP)?;
        let stream = self.raw_cuda_stream()? as usize as cudaStream_t;
        unsafe { cuda_result::stream::synchronize(stream) }
            .map_err(|err| crate::Error::backend_source(OP, err))
    }
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        // Drop cannot surface errors, but the runtime must not release the
        // primary context while queued kernels may still reference it.
        if let Err(err) = self.synchronize() {
            report_cuda_runtime_drop_error(&err);
        }
    }
}
