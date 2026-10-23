//! CUDA storage: one `t4a-cuda-runtime` allocation behind the tensor storage trait.

use tenferro_tensor::{AllocationDomainId, AllocationId, BackendStorage};

use super::runtime::CudaRuntime;

/// Next allocation identity, process-wide; every buffer of every runtime takes one.
static NEXT_ALLOCATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// One device allocation owned by a CUDA runtime.
///
/// The allocation carries its runtime's domain identity and a stable allocation id; a
/// tensor view keeps both. `t4a-cuda-runtime` addresses are stable for the allocation's
/// life, so no relocation pin is needed.
#[doc(hidden)]
pub struct CudaStorage {
    pub(crate) buffer: t4a_cuda_runtime::Buffer,
    pub(crate) bytes: usize,
    pub(crate) device: usize,
    pub(crate) domain: AllocationDomainId,
    pub(crate) id: AllocationId,
}

impl CudaStorage {
    /// Number of bytes in the allocation.
    #[doc(hidden)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Device ordinal of the allocation.
    #[doc(hidden)]
    pub fn device_ordinal(&self) -> usize {
        self.device
    }

    /// The runtime's allocation domain.
    #[doc(hidden)]
    pub fn domain(&self) -> AllocationDomainId {
        self.domain
    }
}

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("bytes", &self.bytes)
            .field("device", &self.device)
            .finish()
    }
}

impl<T: Send + Sync + 'static> BackendStorage<T> for CudaStorage {
    fn backend_family(&self) -> &'static str {
        "cuda"
    }

    fn len(&self) -> usize {
        self.bytes / std::mem::size_of::<T>().max(1)
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Identity of a fresh storage allocation inside `runtime`.
pub(crate) fn fresh_identity(runtime: &CudaRuntime) -> (AllocationDomainId, AllocationId) {
    let id = NEXT_ALLOCATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (
        runtime.allocation_domain_id(),
        AllocationId::from_backend_id(id),
    )
}
