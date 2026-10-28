//! The CUDA backend: one engine over `t4a-cuda-runtime` and the `t4a-cuda-kernels`
//! catalogue (issue #2208).
//!
//! One [`CudaRuntime`] per engine owns the stream, the allocation domain, transfers and
//! pending handles; every operation reaches the kernel catalogue through the engine's typed
//! dispatchers, and the vendor libraries through [`raw::Session`].
//!
//! # Examples
//!
//! ```no_run
//! # fn main() -> Result<(), tenferro_gpu::cuda::CudaEngineError> {
//! use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine};
//!
//! # #[cfg(feature = "cuda")]
//! if gpu_available() {
//!     let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
//!     let _ = engine.runtime();
//! }
//! # Ok(())
//! # }
//! ```

pub(crate) mod device;
pub(crate) mod identity;

/// The `cudarc` crate tenferro-gpu is built against (issue #1940).
///
/// Downstream code that calls a CUDA vendor library (cuBLAS, cuSOLVER, ...) on tenferro buffers
/// through [`raw::Session`] should take its bindings from here, so they always match tenferro's
/// `cudarc` version and CUDA version selection. The re-export is not a stable API of its own: it
/// moves with tenferro's `cudarc` requirement.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::cudarc::cublas::sys::cublasOperation_t;
///
/// let no_transpose = cublasOperation_t::CUBLAS_OP_N;
/// assert_ne!(no_transpose, cublasOperation_t::CUBLAS_OP_T);
/// ```
pub use ::cudarc;
pub use device::{cuda_devices, CudaDeviceError, CudaDeviceId, CudaDeviceInfo};

pub use identity::{CudaComputeCapability, CudaDeviceUuid, GpuExtensionCapability};

/// The CUDA backend's seam identity.
///
/// Its allocation is the `t4a-cuda-runtime` buffer the runtime hands out; a
/// [`gpu::Buffer<CudaBackend>`] owns it, so no consumer reaches into tensor storage for a
/// device address.
#[doc(hidden)]
pub struct CudaBackend;

impl crate::gpu::Backend for CudaBackend {
    type Allocation = t4a_cuda_runtime::Buffer;
}

/// Whether any CUDA device is usable, probed through the CUDA runtime.
#[doc(hidden)]
pub fn gpu_available() -> bool {
    let available = t4a_cuda_runtime::device_count()
        .map(|count| count > 0)
        .unwrap_or(false);
    // The GPU lane sets `TENFERRO_REQUIRE_GPU=1`: there a missing device must fail the test
    // rather than skip it, so a hardware lane cannot report success without hardware.
    assert!(
        available || std::env::var_os("TENFERRO_REQUIRE_GPU").is_none(),
        "TENFERRO_REQUIRE_GPU is set but no CUDA device is usable"
    );
    available
}
/// The seam this adapter implements.
pub use crate::gpu;

mod cutensor;
mod dispatch;
mod engine;
mod event_domain;
mod fusion;
mod launch;
mod pending;
mod pinned;
mod publication;
pub mod raw;
mod runtime;
mod runtime_adapter;

pub use runtime_adapter::{cuda_runtime_engine_registration, cuda_runtime_hardware_class};
mod storage;
mod structural;
mod transfer;
pub(crate) mod vendor;

mod capability;
pub use capability::cuda_capabilities;
pub use cutensor::{Contraction, Cutensor, Operand};
pub use engine::{CudaEngine, CudaEngineError};
pub use fusion::PreparedFusion;
pub use launch::{device_buffer, layout_arg, KernelModules};
pub use pending::{PendingDownload, PendingUpload, RejectedTransfer, TransferSubmitError};
pub use pinned::PinnedHostBuffer;
pub use publication::{
    PublicationAdmissionError, PublicationDownload, PublicationError, RejectedPublication,
    SourcePublication,
};
pub use runtime::{CudaRuntime, CudaRuntimeIdentity};
pub use transfer::{download_pending, download_tensor, upload_pending, upload_tensor};
pub use vendor::{Cublas, GemmRequest};
