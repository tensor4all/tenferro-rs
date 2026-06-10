//! CubeCL CUDA runtime initialization and synchronization.

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::sys::{CUcontext, CUdevice};
use cudarc::runtime::{result as cuda_result, sys::cudaStream_t};

/// Returns `true` if a CUDA device is available for CubeCL.
///
/// Use this in test helpers to skip GPU tests on machines without hardware.
pub fn gpu_available() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = CudaDevice::new(0);
        let _ = CudaRuntime::client(&device);
    }))
    .is_ok()
}

/// CubeCL CUDA runtime wrapper.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::CubeclRuntime;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclRuntime> = CubeclRuntime::new;
/// let _sync: fn(&CubeclRuntime) -> tenferro_tensor::Result<()> =
///     CubeclRuntime::synchronize;
/// ```
pub struct CubeclRuntime {
    client: ComputeClient<CudaRuntime>,
    device_ordinal: usize,
    cuda_device: CUdevice,
    cuda_context: CUcontext,
}

// CUDA primary contexts and CubeCL clients are owned handles. Backend methods
// set the context current before raw CUDA-library calls, and higher-level eager
// execution serializes backend access through a mutex.
unsafe impl Send for CubeclRuntime {}

impl CubeclRuntime {
    /// Initialize the CubeCL CUDA runtime on the given device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CubeclRuntime;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclRuntime> = CubeclRuntime::new;
    /// ```
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        let device = CudaDevice::new(device_ordinal);
        let client = CudaRuntime::client(&device);
        cudarc::runtime::result::device::set(device_ordinal as i32).map_err(|err| {
            crate::Error::backend_failure(
                "cubecl_runtime_init",
                format!("failed to set CUDA runtime device: {err:?}"),
            )
        })?;
        cudarc::driver::result::init().map_err(|err| {
            crate::Error::backend_failure(
                "cubecl_runtime_init",
                format!("failed to initialize CUDA driver: {err:?}"),
            )
        })?;
        let cuda_device =
            cudarc::driver::result::device::get(device_ordinal as i32).map_err(|err| {
                crate::Error::backend_failure(
                    "cubecl_runtime_init",
                    format!("failed to obtain CUDA device {device_ordinal}: {err:?}"),
                )
            })?;
        let cuda_context = unsafe { cudarc::driver::result::primary_ctx::retain(cuda_device) }
            .map_err(|err| {
                crate::Error::backend_failure(
                    "cubecl_runtime_init",
                    format!("failed to retain CUDA primary context: {err:?}"),
                )
            })?;
        unsafe { cudarc::driver::result::ctx::set_current(cuda_context) }.map_err(|err| {
            crate::Error::backend_failure(
                "cubecl_runtime_init",
                format!("failed to set CUDA primary context current: {err:?}"),
            )
        })?;
        Ok(Self {
            client,
            device_ordinal,
            cuda_device,
            cuda_context,
        })
    }

    pub(crate) fn client(&self) -> &ComputeClient<CudaRuntime> {
        &self.client
    }

    /// Return the CUDA device ordinal that this runtime targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CubeclRuntime;
    ///
    /// let _device_ordinal: fn(&CubeclRuntime) -> usize = CubeclRuntime::device_ordinal;
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    #[doc(hidden)]
    pub fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
        cudarc::runtime::result::device::set(self.device_ordinal as i32).map_err(|err| {
            crate::Error::backend_failure(op, format!("failed to set CUDA runtime device: {err:?}"))
        })?;
        unsafe { cudarc::driver::result::ctx::set_current(self.cuda_context) }.map_err(|err| {
            crate::Error::backend_failure(
                op,
                format!("failed to activate CUDA primary context: {err:?}"),
            )
        })
    }

    pub(crate) fn raw_cuda_stream(&self) -> crate::Result<u64> {
        self.client
            .with_server(|server| {
                server
                    .raw_stream(StreamId::current())
                    .map(|stream| stream as u64)
                    .map_err(|err| {
                        crate::Error::backend_failure("raw_cuda_stream", format!("{err:?}"))
                    })
            })
            .ok_or_else(|| {
                crate::Error::backend_failure("raw_cuda_stream", "with_server returned None")
            })?
    }

    /// Block the current thread until work submitted to the current CUDA stream completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CubeclRuntime;
    ///
    /// let _sync: fn(&CubeclRuntime) -> tenferro_tensor::Result<()> =
    ///     CubeclRuntime::synchronize;
    /// ```
    pub fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "cubecl_runtime_synchronize";
        self.set_current_cuda_context(OP)?;
        let stream = self.raw_cuda_stream()? as usize as cudaStream_t;
        unsafe { cuda_result::stream::synchronize(stream) }.map_err(|err| {
            crate::Error::backend_failure(OP, format!("CUDA stream synchronize failed: {err:?}"))
        })
    }
}

impl Drop for CubeclRuntime {
    fn drop(&mut self) {
        let _ = unsafe { cudarc::driver::result::primary_ctx::release(self.cuda_device) };
    }
}
