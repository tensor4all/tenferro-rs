//! CubeCL CUDA runtime initialization and stream access.

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::sys::{CUcontext, CUdevice};

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
/// use tenferro_tensor::cubecl::CubeclRuntime;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclRuntime> = CubeclRuntime::new;
/// let _stream: fn(&CubeclRuntime) -> tenferro_tensor::Result<u64> =
///     CubeclRuntime::raw_cuda_stream;
/// ```
pub struct CubeclRuntime {
    client: ComputeClient<CudaRuntime>,
    device_ordinal: usize,
    cuda_device: CUdevice,
    cuda_context: CUcontext,
}

impl CubeclRuntime {
    /// Initialize the CubeCL CUDA runtime on the given device ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cubecl::CubeclRuntime;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<CubeclRuntime> = CubeclRuntime::new;
    /// ```
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        let device = CudaDevice::new(device_ordinal);
        let client = CudaRuntime::client(&device);
        cudarc::runtime::result::device::set(device_ordinal as i32).map_err(|err| {
            crate::Error::BackendFailure {
                op: "cubecl_runtime_init",
                message: format!("failed to set CUDA runtime device: {err:?}"),
            }
        })?;
        cudarc::driver::result::init().map_err(|err| crate::Error::BackendFailure {
            op: "cubecl_runtime_init",
            message: format!("failed to initialize CUDA driver: {err:?}"),
        })?;
        let cuda_device =
            cudarc::driver::result::device::get(device_ordinal as i32).map_err(|err| {
                crate::Error::BackendFailure {
                    op: "cubecl_runtime_init",
                    message: format!("failed to obtain CUDA device {device_ordinal}: {err:?}"),
                }
            })?;
        let cuda_context = unsafe { cudarc::driver::result::primary_ctx::retain(cuda_device) }
            .map_err(|err| crate::Error::BackendFailure {
                op: "cubecl_runtime_init",
                message: format!("failed to retain CUDA primary context: {err:?}"),
            })?;
        unsafe { cudarc::driver::result::ctx::set_current(cuda_context) }.map_err(|err| {
            crate::Error::BackendFailure {
                op: "cubecl_runtime_init",
                message: format!("failed to set CUDA primary context current: {err:?}"),
            }
        })?;
        Ok(Self {
            client,
            device_ordinal,
            cuda_device,
            cuda_context,
        })
    }

    /// Borrow the underlying CubeCL compute client.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::client::ComputeClient;
    /// use cubecl_cuda::CudaRuntime;
    /// use tenferro_tensor::cubecl::CubeclRuntime;
    ///
    /// let _client_getter: fn(&CubeclRuntime) -> &ComputeClient<CudaRuntime> = CubeclRuntime::client;
    /// ```
    pub fn client(&self) -> &ComputeClient<CudaRuntime> {
        &self.client
    }

    /// Return the CUDA device ordinal that this runtime targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cubecl::CubeclRuntime;
    ///
    /// let _device_ordinal: fn(&CubeclRuntime) -> usize = CubeclRuntime::device_ordinal;
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn set_current_cuda_context(&self, op: &'static str) -> crate::Result<()> {
        cudarc::runtime::result::device::set(self.device_ordinal as i32).map_err(|err| {
            crate::Error::BackendFailure {
                op,
                message: format!("failed to set CUDA runtime device: {err:?}"),
            }
        })?;
        unsafe { cudarc::driver::result::ctx::set_current(self.cuda_context) }.map_err(|err| {
            crate::Error::BackendFailure {
                op,
                message: format!("failed to activate CUDA primary context: {err:?}"),
            }
        })
    }

    /// Extract the raw CUDA stream pointer used by the current CubeCL stream.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cubecl::CubeclRuntime;
    ///
    /// let _stream: fn(&CubeclRuntime) -> tenferro_tensor::Result<u64> =
    ///     CubeclRuntime::raw_cuda_stream;
    /// ```
    pub fn raw_cuda_stream(&self) -> crate::Result<u64> {
        self.client
            .with_server(|server| {
                server
                    .raw_stream(StreamId::current())
                    .map(|stream| stream as u64)
                    .map_err(|err| crate::Error::BackendFailure {
                        op: "raw_cuda_stream",
                        message: format!("{err:?}"),
                    })
            })
            .ok_or_else(|| crate::Error::BackendFailure {
                op: "raw_cuda_stream",
                message: "with_server returned None".into(),
            })?
    }
}

impl Drop for CubeclRuntime {
    fn drop(&mut self) {
        let _ = unsafe { cudarc::driver::result::primary_ctx::release(self.cuda_device) };
    }
}
