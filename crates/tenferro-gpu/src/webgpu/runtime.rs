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

/// Opaque identity of one exact WebGPU runtime instance.
///
/// The identity follows the executable queue/client resources, including
/// Apple-backed host-visible Metal runtimes. Cloning preserves identity;
/// independently initialized runtimes are distinct even for the same device
/// ordinal. The token intentionally carries no provider or device identifier.
#[derive(Clone, Debug)]
pub struct WebGpuRuntimeIdentity {
    marker: std::sync::Arc<()>,
}

impl WebGpuRuntimeIdentity {
    fn fresh() -> Self {
        Self {
            marker: std::sync::Arc::new(()),
        }
    }
}

impl PartialEq for WebGpuRuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.marker, &other.marker)
    }
}

impl Eq for WebGpuRuntimeIdentity {}

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
    identity: WebGpuRuntimeIdentity,
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
            identity: WebGpuRuntimeIdentity::fresh(),
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
            identity: WebGpuRuntimeIdentity::fresh(),
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

    /// Return the opaque identity of this exact executable runtime instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::WebGpuRuntime;
    ///
    /// let _identity: fn(&WebGpuRuntime) -> tenferro_gpu::WebGpuRuntimeIdentity =
    ///     WebGpuRuntime::runtime_identity;
    /// ```
    pub fn runtime_identity(&self) -> WebGpuRuntimeIdentity {
        self.identity.clone()
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
    /// Returns [`crate::Error::BackendSource`] when CubeCL queue synchronization fails.
    pub fn synchronize(&self) -> crate::Result<()> {
        const OP: &str = "webgpu_runtime_synchronize";
        future::block_on(self.client.sync()).map_err(|err| crate::Error::backend_source(OP, err))
    }
}

#[cfg(test)]
mod identity_tests {
    use super::{webgpu_available, WebGpuRuntimeIdentity};
    use crate::WebGpuBackend;

    #[test]
    fn webgpu_runtime_identity_is_clone_stable_and_instance_scoped() {
        let first = WebGpuRuntimeIdentity::fresh();
        let clone = first.clone();
        let independent = WebGpuRuntimeIdentity::fresh();

        assert_eq!(first, clone);
        assert_ne!(first, independent);
    }

    #[test]
    fn webgpu_backend_identity_tracks_the_exact_runtime_when_hardware_is_available() {
        if !webgpu_available() {
            return;
        }

        let first = WebGpuBackend::new(0).expect("WebGPU backend should initialize");
        let clone = first.clone();
        let independent = WebGpuBackend::new(0).expect("second WebGPU backend should initialize");

        assert_eq!(first.runtime_identity(), clone.runtime_identity());
        assert_ne!(first.runtime_identity(), independent.runtime_identity());
        assert_eq!(
            first.runtime_identity(),
            WebGpuBackend::from_runtime(first.runtime().clone()).runtime_identity()
        );
    }
}
