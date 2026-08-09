//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! #[cfg(feature = "cuda")]
//! use tenferro_gpu::{cuda::cuda_devices, cuda::CudaBackend, cuda::CudaDeviceError};
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
pub mod webgpu;

/// CUDA provider namespace.
#[cfg(feature = "cuda")]
pub mod cuda {
    pub use super::cubecl::{
        cuda_capabilities, cuda_devices, cuda_runtime_engine_registration,
        cuda_runtime_hardware_class, download_tensor, gpu_available, upload_tensor,
        with_cuda_exec_session, CudaBackend, CudaComputeCapability, CudaDeviceError, CudaDeviceId,
        CudaDeviceInfo, CudaDeviceUuid, CudaExecSession, CudaRuntime, CudaRuntimeIdentity,
        GpuExtensionCapability,
    };

    /// Public tenferro-wide CubeCL session (issue #1597).
    ///
    /// Exposes a narrow prelude of the CubeCL types needed to write and launch
    /// `#[cube]` kernels against tenferro's GPU runtime. This module does not
    /// re-export the whole of `cubecl`; downstream crates declare the framework
    /// `t4a-cubecl` package explicitly.
    pub mod cubecl {
        pub use super::super::cubecl::session_cubecl::Session;
        // Narrow prelude: only the types needed to describe a CubeCL launch.
        pub use ::cubecl::prelude::{ArrayArg, CubeCount, CubeDim, TensorBinding};
    }

    /// Type-safe raw CUDA extension session (issue #1597).
    pub mod raw {
        pub use super::super::cubecl::raw::{
            CudaResourceGuard, DeviceBytes, Function, KernelArg, LaunchConfig, Module,
            NvrtcOptions, Session, StreamRef, TensorMut, TensorRef,
        };
    }

    /// Provider-specific CUDA interop scoped to an active execution session.
    #[doc(hidden)]
    pub mod interop {
        pub use super::super::cubecl::interop::*;
        pub use super::super::cubecl::{CudaExtensionCache, CudaExtensionCacheGuard};
    }
}

/// Apple shared-allocation provider namespace.
#[cfg(feature = "webgpu")]
pub mod apple {
    pub use super::webgpu::{AppleContext, AppleTransferStats};
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

/// Scalar-independent CubeCL allocation stored behind tensor backend-buffer
/// trait objects; dtype is carried by the borrowed tensor descriptor.
#[cfg(feature = "cuda")]
pub(crate) struct CubeclBuffer {
    handle: cubecl_runtime::server::Handle,
    byte_len: usize,
    device_ordinal: usize,
    allocation_domain: AllocationDomainId,
    allocation_id: AllocationId,
}

#[cfg(feature = "cuda")]
static NEXT_CUDA_ALLOCATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CubeclBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclBuffer")
            .field("byte_len", &self.byte_len)
            .field("device_ordinal", &self.device_ordinal)
            .field("allocation_domain", &self.allocation_domain)
            .field("allocation_id", &self.allocation_id)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl CubeclBuffer {
    pub(crate) fn new(
        handle: cubecl_runtime::server::Handle,
        byte_len: usize,
        device_ordinal: usize,
        allocation_domain: AllocationDomainId,
    ) -> Self {
        Self {
            handle,
            byte_len,
            device_ordinal,
            allocation_domain,
            allocation_id: AllocationId::from_backend_id(
                NEXT_CUDA_ALLOCATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }

    pub(crate) fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    pub(crate) fn element_len<T: 'static>(&self) -> usize {
        let element_size = std::mem::size_of::<T>();
        debug_assert!(element_size != 0 && self.byte_len.is_multiple_of(element_size));
        self.byte_len / element_size
    }

    pub(crate) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn allocation_domain(&self) -> AllocationDomainId {
        self.allocation_domain
    }
}

#[cfg(feature = "cuda")]
impl<T: Send + Sync + 'static> BackendStorage<T> for CubeclBuffer {
    fn backend_family(&self) -> &'static str {
        "cubecl"
    }

    fn len(&self) -> usize {
        self.element_len::<T>()
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.allocation_domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation_id)
    }

    fn prepare_device_access(
        &self,
        request: DeviceAccessRequest<'_>,
    ) -> std::result::Result<Box<dyn PreparedDeviceAccess>, DeviceAccessError> {
        Ok(Box::new(crate::cubecl::dispatch::prepare_cubecl_access(
            self, request,
        )?))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
