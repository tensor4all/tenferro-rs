use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::Runtime;
use cubecl_common::future;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use super::apple::AppleDomainState;

/// Returns `true` if a WebGPU adapter is available for CubeCL.
///
/// Use this in test helpers to skip WebGPU tests on machines without an
/// adapter.
pub fn webgpu_available() -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = WgpuDevice::DefaultDevice;
        let _ = WgpuRuntime::client(&device);
    }))
    .is_ok()
}

/// CubeCL WebGPU runtime wrapper.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::WebGpuRuntime;
///
/// let _ctor: fn(usize) -> tenferro_tensor::Result<WebGpuRuntime> = WebGpuRuntime::new;
/// let _sync: fn(&WebGpuRuntime) -> tenferro_tensor::Result<()> =
///     WebGpuRuntime::synchronize;
/// ```
#[derive(Clone)]
pub struct WebGpuRuntime {
    client: ComputeClient<WgpuRuntime>,
    device_ordinal: usize,
    pub(super) apple_domain: Option<std::sync::Arc<AppleDomainState>>,
}

impl fmt::Debug for WebGpuRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebGpuRuntime")
            .field("device_ordinal", &self.device_ordinal)
            .finish_non_exhaustive()
    }
}

impl WebGpuRuntime {
    /// Initialize the CubeCL WebGPU runtime for a discrete GPU ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuRuntime;
    ///
    /// let _ctor: fn(usize) -> tenferro_tensor::Result<WebGpuRuntime> = WebGpuRuntime::new;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the requested adapter is
    /// unavailable or CubeCL initialization panics while selecting it.
    pub fn new(device_ordinal: usize) -> crate::Result<Self> {
        Self::from_device(WgpuDevice::DiscreteGpu(device_ordinal), device_ordinal)
    }

    /// Initialize the CubeCL WebGPU runtime using CubeCL's default adapter selection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuRuntime;
    ///
    /// let _ctor: fn() -> tenferro_tensor::Result<WebGpuRuntime> = WebGpuRuntime::new_default;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when default adapter selection is
    /// unavailable or CubeCL initialization panics.
    pub fn new_default() -> crate::Result<Self> {
        Self::from_device(WgpuDevice::DefaultDevice, 0)
    }

    fn from_device(device: WgpuDevice, device_ordinal: usize) -> crate::Result<Self> {
        let client = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device)
        }))
        .map_err(|payload| {
            crate::Error::runtime_state(
                "webgpu_runtime_init",
                format!("failed to initialize CubeCL WebGPU runtime: {payload:?}"),
            )
        })?;
        Ok(Self {
            client,
            device_ordinal,
            apple_domain: None,
        })
    }

    pub(super) fn from_apple_client(
        client: ComputeClient<WgpuRuntime>,
        device_ordinal: usize,
        domain: std::sync::Arc<AppleDomainState>,
    ) -> Self {
        Self {
            client,
            device_ordinal,
            apple_domain: Some(domain),
        }
    }

    pub(super) fn allocation_domain(&self) -> Option<&std::sync::Arc<AppleDomainState>> {
        self.apple_domain.as_ref()
    }

    pub(super) fn record_upload(&self, bytes: usize) {
        if let Some(domain) = &self.apple_domain {
            domain.record_upload(bytes);
        }
    }

    pub(super) fn record_download(&self, bytes: usize) {
        if let Some(domain) = &self.apple_domain {
            domain.record_download(bytes);
        }
    }

    pub(crate) fn client(&self) -> &ComputeClient<WgpuRuntime> {
        &self.client
    }

    /// Return the WebGPU device ordinal requested at construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuRuntime;
    ///
    /// let _device_ordinal: fn(&WebGpuRuntime) -> usize = WebGpuRuntime::device_ordinal;
    /// ```
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Block the current thread until work submitted to the WebGPU queue completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuRuntime;
    ///
    /// let _sync: fn(&WebGpuRuntime) -> tenferro_tensor::Result<()> =
    ///     WebGpuRuntime::synchronize;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when queue flush or sync fails.
    pub fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "webgpu_runtime_synchronize";
        self.client
            .flush()
            .map_err(|err| crate::Error::backend_source(OP, err))?;
        future::block_on(self.client.sync()).map_err(|err| crate::Error::backend_source(OP, err))
    }
}
