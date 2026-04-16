//! CubeCL CUDA runtime initialization and stream access.

use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::Runtime;
use cubecl_cuda::{CudaDevice, CudaRuntime};

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
        Ok(Self {
            client,
            device_ordinal,
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
