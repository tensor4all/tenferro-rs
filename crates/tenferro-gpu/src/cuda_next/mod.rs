//! The CUDA adapter over `t4a-cuda-runtime` (issue #2208 unit U1).
//!
//! This module is the new backend path: one [`CudaRuntime`] per engine, storage with
//! an allocation-domain identity, transfers and pending handles. It is private until
//! the operation surface moves over (U2-U6); the public `cuda::` names keep the
//! CubeCL implementation until then.
//!
//! # Examples
//!
//! ```no_run
//! # fn main() -> Result<(), tenferro_gpu::cuda::CudaDeviceError> {
//! use tenferro_gpu::cuda_next::{gpu_available, CudaDeviceId, CudaEngine};
//!
//! # #[cfg(feature = "cuda-next")]
//! if gpu_available() {
//!     let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
//!     let _ = engine.runtime();
//! }
//! # Ok(())
//! # }
//! ```

pub use crate::cuda::{cuda_devices, CudaDeviceError, CudaDeviceId, CudaDeviceInfo};

/// Whether any CUDA device is usable, probed through the new runtime rather than the
/// CubeCL path, so a new-path test cannot be skipped because the old backend failed.
#[doc(hidden)]
pub fn gpu_available() -> bool {
    t4a_cuda_runtime::device_count()
        .map(|count| count > 0)
        .unwrap_or(false)
}
/// The seam this adapter implements.
pub use crate::gpu;

mod dispatch;
mod engine;
mod fusion;
mod launch;
mod pending;
mod pinned;
mod publication;
mod runtime;
mod storage;
mod transfer;

pub use dispatch::Dispatcher;
pub use engine::{CudaEngine, CudaEngineError};
pub use fusion::{PrepareCounters, PreparedFusion};
pub use launch::{device_buffer, layout_arg, KernelModules};
pub use pending::{PendingDownload, PendingUpload, RejectedTransfer, TransferSubmitError};
pub use pinned::PinnedHostBuffer;
pub use publication::{
    PublicationAdmissionError, PublicationDownload, PublicationError, RejectedPublication,
    SourcePublication,
};
pub use runtime::{CudaRuntime, CudaRuntimeIdentity};
pub use storage::CudaStorage;
pub use transfer::{download_pending, download_tensor, upload_pending, upload_tensor};
