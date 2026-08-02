//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! #[cfg(feature = "cuda")]
//! {
//!     use tenferro_gpu::{
//!         download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceId,
//!     };
//!     use tenferro_tensor::{Tensor, TensorElementwise};
//!
//!     if gpu_available() {
//!         let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))?;
//!         let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
//!         let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
//!         let gpu_a = upload_tensor(backend.runtime(), &a).unwrap();
//!         let gpu_b = upload_tensor(backend.runtime(), &b).unwrap();
//!         let gpu_sum = backend.add(&gpu_a, &gpu_b).unwrap();
//!         let sum = download_tensor(backend.runtime(), &gpu_sum).unwrap();
//!         assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
mod cubecl;
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
#[derive(Clone)]
pub(crate) struct CubeclBuffer<T> {
    handle: cubecl_runtime::server::Handle,
    len: usize,
    device_ordinal: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CubeclBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclBuffer")
            .field("len", &self.len)
            .field("device_ordinal", &self.device_ordinal)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl<T> CubeclBuffer<T> {
    pub(crate) fn new(
        handle: cubecl_runtime::server::Handle,
        len: usize,
        device_ordinal: usize,
    ) -> Self {
        Self {
            handle,
            len,
            device_ordinal,
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
}

#[cfg(feature = "cuda")]
impl<T: Send + Sync + 'static> BackendBuffer<T> for CubeclBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "cubecl"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
