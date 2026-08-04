//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! #[cfg(feature = "cuda")]
//! use tenferro_gpu::{cuda_devices, CudaBackend, CudaDeviceError};
//!
//! #[cfg(feature = "cuda")]
//! fn first_cuda_backend() -> Result<Option<CudaBackend>, CudaDeviceError> {
//!     let devices = cuda_devices()?;
//!     let Some(device) = devices.first() else {
//!         return Ok(None);
//!     };
//!     Ok(Some(CudaBackend::new(device.id())?))
//! }
//!
//! // This ordinary doctest checks the discovery-based selection API without
//! // requiring CUDA hardware at test time.
//! #[cfg(feature = "cuda")]
//! let _example: fn() -> Result<Option<CudaBackend>, CudaDeviceError> = first_cuda_backend;
//! ```

#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
mod cubecl;
#[cfg(any(feature = "cuda", feature = "webgpu"))]
mod event_domain_admission;
#[cfg(any(feature = "cuda", feature = "webgpu"))]
mod event_retirement;
#[cfg(any(feature = "cuda", feature = "webgpu"))]
mod kernels;
#[cfg(any(feature = "cuda", feature = "webgpu"))]
mod native_permutation;
#[cfg(feature = "webgpu")]
mod webgpu;

#[cfg(feature = "cuda")]
pub use cubecl::{
    cuda_capabilities, cuda_devices, cuda_runtime_engine_registration, cuda_runtime_hardware_class,
    device_ptr, download_tensor, gpu_available, upload_tensor, with_cuda_exec_session, CudaBackend,
    CudaDeviceError, CudaDeviceId, CudaDeviceInfo, CudaExecSession, CudaRuntime,
    CudaRuntimeIdentity,
};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cubecl::{CudaExtensionCache, CudaExtensionCacheGuard};
#[cfg(feature = "webgpu")]
pub use webgpu::{
    download_webgpu_tensor, upload_webgpu_tensor, webgpu_available, webgpu_runtime_engine_id,
    webgpu_runtime_engine_registration, webgpu_runtime_engine_registration_with_id,
    webgpu_runtime_hardware_class, with_webgpu_exec_session, AppleContext, AppleTransferStats,
    WebGpuBackend, WebGpuExecSession, WebGpuRuntime, WebGpuRuntimeIdentity,
};

/// Narrow owner-scoped WebGPU handle interop for extension crates.
#[cfg(feature = "webgpu")]
#[doc(hidden)]
pub mod webgpu_interop {
    pub use crate::webgpu::interop::*;
}

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod cuda_interop {
    pub use crate::cubecl::interop::*;
    pub use crate::cubecl::{CudaExtensionCache, CudaExtensionCacheGuard};
}

#[cfg(any(feature = "cuda", feature = "webgpu"))]
use tenferro_tensor::*;

#[cfg(feature = "cuda")]
pub(crate) mod backend {
    pub use tenferro_tensor::backend::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod config {
    pub use tenferro_tensor::config::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod types {
    pub(crate) use crate::CubeclBuffer;
    pub use tenferro_tensor::types::*;
}

/// CubeCL-managed GPU buffer stored behind tensor backend-buffer trait objects.
#[cfg(feature = "cuda")]
pub(crate) struct CubeclBuffer<T> {
    handle: cubecl_runtime::server::Handle,
    len: usize,
    device_ordinal: usize,
    allocation_domain: Option<AllocationDomainId>,
    allocation_id: AllocationId,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "cuda")]
static NEXT_CUDA_ALLOCATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CubeclBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclBuffer")
            .field("len", &self.len)
            .field("device_ordinal", &self.device_ordinal)
            .field("allocation_domain", &self.allocation_domain)
            .field("allocation_id", &self.allocation_id)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl<T> CubeclBuffer<T> {
    pub(crate) fn new(
        handle: cubecl_runtime::server::Handle,
        len: usize,
        device_ordinal: usize,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Self {
        Self {
            handle,
            len,
            device_ordinal,
            allocation_domain,
            allocation_id: AllocationId::from_backend_id(
                NEXT_CUDA_ALLOCATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ),
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    pub(crate) fn element_len(&self) -> usize {
        self.len
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain
    }
}

#[cfg(feature = "cuda")]
impl<T: Send + Sync + 'static> BackendStorage<T> for CubeclBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "cubecl"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
