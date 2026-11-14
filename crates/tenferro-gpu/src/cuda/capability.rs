//! The public CUDA capability-table surface.
//!
//! The table itself lives in [`crate::capability`] so it compiles without the `cuda` feature
//! and its guide table can be checked in a default-features build (issue #2477). This module
//! is the engine-facing spelling: the `TensorBackendCapability` impl for [`CudaEngine`] and
//! the public re-export.

use tenferro_tensor::{BackendId, OperationCapability, TensorBackendCapability};

use super::CudaEngine;

/// Return the CUDA backend operation capability descriptor table.
///
/// The table states what this backend serves natively; it is written down rather than derived
/// from the kernel catalogue's `accepts` list, and `tests/seam_capability.rs` holds each entry to
/// the engine's observed behaviour. `docs/guides/devices-and-gpu.md` renders the same table and
/// a default-features test checks the two agree.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_gpu::cuda::cuda_capabilities;
/// use tenferro_tensor::{DType, SupportLevel};
///
/// let add = cuda_capabilities()
///     .iter()
///     .find(|entry| entry.op == PrimitiveOpKind::Add && entry.dtype == DType::I32)
///     .unwrap();
/// assert_eq!(add.result, SupportLevel::Native);
/// ```
#[must_use]
pub const fn cuda_capabilities() -> &'static [OperationCapability] {
    crate::capability::cuda_capability_table()
}

impl TensorBackendCapability for CudaEngine {
    fn backend_id(&self) -> BackendId {
        BackendId::Cuda
    }

    fn capabilities(&self) -> &'static [OperationCapability] {
        crate::capability::cuda_capability_table()
    }
}
