use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cubecl::client::ComputeClient;
use cubecl::Runtime;
use cubecl_wgpu::{
    init_device_for_graphics_api, MemoryConfiguration, Metal, PrimaryMemoryMode, RuntimeOptions,
    WgpuDevice, WgpuRuntime as CubeWgpuRuntime,
};

use super::{alloc_tensor_in_runtime, WebGpuBackend, WebGpuRuntime};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{AllocationDomainId, DType, SharedTensorAllocationDomain, Tensor};

use crate::{download_webgpu_tensor, upload_webgpu_tensor};

/// Explicit host/device transfer counters for one [`AppleContext`].
///
/// Guarded host mappings do not change these counters.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::AppleTransferStats;
///
/// assert_eq!(AppleTransferStats::default().uploaded_bytes, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppleTransferStats {
    /// Bytes copied by explicit host-to-managed creation.
    pub uploaded_bytes: usize,
    /// Bytes copied by explicit managed-to-host download.
    pub downloaded_bytes: usize,
}

pub(super) struct AppleDomainState {
    pub(super) id: AllocationDomainId,
    client: ComputeClient<CubeWgpuRuntime>,
    device_ordinal: usize,
    uploaded_bytes: AtomicUsize,
    downloaded_bytes: AtomicUsize,
}

impl fmt::Debug for AppleDomainState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppleDomainState")
            .field("id", &self.id)
            .field("device_ordinal", &self.device_ordinal)
            .field("transfers", &self.snapshot())
            .finish_non_exhaustive()
    }
}

impl AppleDomainState {
    pub(super) fn record_upload(&self, bytes: usize) {
        self.uploaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(super) fn record_download(&self, bytes: usize) {
        self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn snapshot(&self) -> AppleTransferStats {
        AppleTransferStats {
            uploaded_bytes: self.uploaded_bytes.load(Ordering::Relaxed),
            downloaded_bytes: self.downloaded_bytes.load(Ordering::Relaxed),
        }
    }

    fn runtime(self: &Arc<Self>) -> WebGpuRuntime {
        WebGpuRuntime::from_apple_client(self.client.clone(), self.device_ordinal, Arc::clone(self))
    }
}

#[derive(Debug)]
struct AppleAllocationDomain {
    state: Arc<AppleDomainState>,
}

impl SharedTensorAllocationDomain for AppleAllocationDomain {
    fn id(&self) -> AllocationDomainId {
        self.state.id
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> crate::Result<Tensor> {
        alloc_tensor_in_runtime(&self.state.runtime(), dtype, shape)
    }
}

/// Explicit Apple shared-allocation context for CPU and Metal backends.
///
/// Every context owns a fresh host-visible Metal client and allocation domain.
/// Tensors created through [`Self::upload_tensor`] can be mapped by the paired
/// CPU backend and launched by the paired Metal backend without implicit
/// transfers. The initial mapped CPU operations are RustFFT and rank-2
/// Cholesky; this context does not turn other CPU operations into automatic
/// fallbacks. Clone [`Self::cpu_backend`] or [`Self::metal_backend`] to obtain
/// the mutable handle required by an explicitly selected operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_gpu::AppleContext;
///
/// let _constructor: fn() -> tenferro_tensor::Result<AppleContext> = AppleContext::new;
/// ```
#[derive(Clone)]
pub struct AppleContext {
    state: Arc<AppleDomainState>,
    cpu: CpuBackend,
    metal: WebGpuBackend,
}

impl fmt::Debug for AppleContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppleContext")
            .field("domain_id", &self.domain_id())
            .field("transfers", &self.transfer_stats())
            .finish_non_exhaustive()
    }
}

impl AppleContext {
    /// Create an independent host-visible Metal context and paired CPU backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    ///
    /// let _constructor: fn() -> tenferro_tensor::Result<AppleContext> = AppleContext::new;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed runtime or backend error when Metal or host-visible
    /// primary buffers are unavailable.
    pub fn new() -> crate::Result<Self> {
        static NEXT_ORDINAL: AtomicUsize = AtomicUsize::new(1_000_000);
        let options = RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
            primary_memory: PrimaryMemoryMode::HostVisible,
        };
        let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            init_device_for_graphics_api::<Metal>(&WgpuDevice::DefaultDevice, options)
        }))
        .map_err(|payload| {
            crate::Error::runtime_state(
                "AppleContext::new",
                format!("failed to initialize host-visible Metal runtime: {payload:?}"),
            )
        })?;
        let client = CubeWgpuRuntime::client(&device);
        let device_ordinal = NEXT_ORDINAL.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(AppleDomainState {
            id: AllocationDomainId::fresh(),
            client,
            device_ordinal,
            uploaded_bytes: AtomicUsize::new(0),
            downloaded_bytes: AtomicUsize::new(0),
        });
        let domain: Arc<dyn SharedTensorAllocationDomain> = Arc::new(AppleAllocationDomain {
            state: Arc::clone(&state),
        });
        let cpu = CpuBackend::new().with_allocation_domain(domain);
        let metal = WebGpuBackend::from_runtime(state.runtime());
        Ok(Self { state, cpu, metal })
    }

    /// Return this context's allocation-domain identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// use tenferro_tensor::AllocationDomainId;
    /// let _method: fn(&AppleContext) -> AllocationDomainId = AppleContext::domain_id;
    /// ```
    pub fn domain_id(&self) -> AllocationDomainId {
        self.state.id
    }

    /// Return the paired CPU backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// let _method = AppleContext::cpu_backend;
    /// ```
    pub fn cpu_backend(&self) -> &CpuBackend {
        &self.cpu
    }

    /// Return the paired Metal WebGPU backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// let _method = AppleContext::metal_backend;
    /// ```
    pub fn metal_backend(&self) -> &WebGpuBackend {
        &self.metal
    }

    /// Return an atomic snapshot of explicit transfer counters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// let _method = AppleContext::transfer_stats;
    /// ```
    pub fn transfer_stats(&self) -> AppleTransferStats {
        self.state.snapshot()
    }

    /// Create a managed tensor by explicitly copying a host tensor once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// let _method = AppleContext::upload_tensor;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the input is not host
    /// resident, [`crate::Error::Validation`] for invalid tensor metadata, or
    /// [`crate::Error::BackendSource`] when managed allocation or queue
    /// synchronization fails.
    pub fn upload_tensor(&self, tensor: &Tensor) -> crate::Result<Tensor> {
        let output = upload_webgpu_tensor(self.metal.runtime(), tensor)?;
        self.metal.synchronize()?;
        Ok(output)
    }

    /// Explicitly copy a matching managed tensor back to host memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_gpu::AppleContext;
    /// let _method = AppleContext::download_tensor;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::HostAccess`] for a foreign allocation domain or
    /// a typed backend error when readback fails.
    pub fn download_tensor(&self, tensor: &Tensor) -> crate::Result<Tensor> {
        download_webgpu_tensor(self.metal.runtime(), tensor)
    }
}
