//! CUDA storage: one `t4a-cuda-runtime` allocation, carried as the seam's buffer.

use tenferro_tensor::{AllocationDomainId, AllocationId, BackendStorage};

use super::runtime::CudaRuntime;
use super::CudaBackend;

/// Next allocation identity, process-wide; every buffer of every runtime takes one.
static NEXT_ALLOCATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One device allocation owned by a CUDA runtime.
///
/// The seam buffer carries the runtime's domain identity, a stable allocation id and the
/// runtime's own allocation handle. `t4a-cuda-runtime` addresses are stable for the
/// allocation's life, so no relocation pin is needed.
pub(crate) struct CudaStorage {
    pub(crate) buffer: crate::gpu::Buffer<CudaBackend>,
}

impl CudaStorage {
    /// Number of bytes in the allocation.
    pub(crate) fn bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Device ordinal of the allocation.
    pub(crate) fn device_ordinal(&self) -> usize {
        self.buffer.identity().device
    }

    /// The runtime's allocation domain.
    pub(crate) fn domain(&self) -> AllocationDomainId {
        self.buffer.identity().domain
    }

    /// The runtime allocation handle, for a launch or vendor call.
    pub(crate) fn allocation(&self) -> &t4a_cuda_runtime::Buffer {
        self.buffer.allocation()
    }
}

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("bytes", &self.bytes())
            .field("device", &self.device_ordinal())
            .finish()
    }
}

impl<T: Send + Sync + 'static> BackendStorage<T> for CudaStorage {
    fn backend_family(&self) -> &'static str {
        "cuda"
    }

    fn len(&self) -> usize {
        self.bytes() / std::mem::size_of::<T>().max(1)
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain())
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.buffer.identity().allocation)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Storage for a fresh allocation of `runtime`: the identity comes from `runtime`, so the
/// handle and its identity cannot disagree.
pub(crate) fn fresh_storage(
    runtime: &CudaRuntime,
    allocation: t4a_cuda_runtime::Buffer,
) -> CudaStorage {
    let id = NEXT_ALLOCATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let bytes = allocation.len();
    CudaStorage {
        buffer: crate::gpu::Buffer::new(
            crate::gpu::StorageIdentity {
                domain: runtime.allocation_domain_id(),
                allocation: AllocationId::from_backend_id(id),
                device: runtime.device_ordinal(),
            },
            bytes,
            allocation,
        ),
    }
}
