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
// A misaligned pointer handed to a CUDA library is undefined behaviour and
// fails only on some library versions: `cuDoubleComplex` is `double2`, which
// CUDA declares `__align__(16)`, while `num_complex::Complex64` is 8-aligned,
// so casting `&Complex64` to `*const cuDoubleComplex` produced a pointer
// cuBLAS >= 12.9 faults on (issue #1870). Deny the whole cast class at this
// FFI boundary rather than re-auditing it by hand.
#![deny(clippy::cast_ptr_alignment)]

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
        cuda_runtime_hardware_class, download_pending, download_tensor, gpu_available,
        upload_pending, upload_tensor, with_cuda_exec_session, CompletionUnproven, CudaBackend,
        CudaComputeCapability, CudaDeviceError, CudaDeviceId, CudaDeviceInfo, CudaDeviceUuid,
        CudaExecSession, CudaExtensionCache, CudaExtensionCacheGuard, CudaHeldSession, CudaRuntime,
        CudaRuntimeIdentity, CudaSessionStats, CutensorWorkspaceStats, GpuExtensionCapability,
        PendingDownload, PendingRead, PendingUpload, PinnedHostBuffer, PublicationAdmissionError,
        PublicationError, PublicationRejected, SourcePublication, TransferRejected,
        TransferSubmitError, WorkspaceRetirementStats,
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
        pub use ::cubecl::prelude::{BufferArg, CubeCount, CubeDim, TensorBinding};
    }

    /// The `cudarc` crate tenferro-gpu is built against (issue #1940).
    ///
    /// Downstream code that calls a CUDA vendor library (cuBLAS, cuSOLVER, ...)
    /// on tenferro buffers through [`raw::Session`] should take its bindings
    /// from here, so they always match tenferro's `cudarc` version and CUDA
    /// version selection. The enabled features are `driver`, `runtime`,
    /// `nvrtc`, `cublas`, `dynamic-loading` and `cuda-12080`; a downstream crate
    /// that needs another `cudarc` module (for example `cusolver`) adds its own
    /// `cudarc` dependency of the same `0.19` line with that feature, and Cargo
    /// unifies the two into one crate. The re-export is not a stable API of its
    /// own: it moves with tenferro's `cudarc` requirement.
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

    /// Type-safe raw CUDA extension session (issue #1597).
    pub mod raw {
        pub use super::super::cubecl::raw::{
            CudaResourceGuard, DeviceBytes, Function, KernelArg, LaunchConfig, Module,
            NvrtcOptions, Session, StreamRef, TensorMut, TensorRef,
        };
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
    // Memoized device address resolved by the first raw-FFI access.
    // Zero means "not memoized": no CUDA allocation lives at the null address.
    //
    // INVARIANT: in pinned CubeCL rev a2adda17, a retained handle's memory
    // slice keeps its storage offset (pool coalescing merges only free
    // slices) and its backing storage is never deallocated while any of its
    // slices is live, so the resolved address is stable for this buffer's
    // lifetime. Raw-FFI callers must still route cross-stream accesses
    // through `get_resource` for CubeCL's stream alignment.
    //
    // INVARIANT (issue #1868): the memoized address is cleared whenever a
    // CubeCL kernel is queued to write this buffer. Resolving the address
    // through `get_resource` is a blocking server round trip, which is also
    // what pushes queued kernels onto the CUstream; a raw vendor call reached
    // through a memoized address skips that round trip and could otherwise be
    // issued ahead of a kernel that precedes it in program order. Clearing on
    // write keeps the fast path for read-only reuse and forces exactly one
    // round trip after each write.
    device_addr: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "cuda")]
static NEXT_CUDA_ALLOCATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Size-keyed free list of device allocations, one per device ordinal.
///
/// An eager op's output does not have to ask the device thread for a buffer on
/// every call: when a buffer's owning tensor is dropped its handle comes back
/// here, and [`take_output_handle`] hands it to the next allocation of the same
/// byte length. Reuse is write-after-write on one stream, which CUDA orders —
/// the same argument the eager retention pool relies on — and the list is
/// bounded, so a length that stops recurring does not retain memory.
#[cfg(feature = "cuda")]
static OUTPUT_POOLS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, OutputPool>>,
> = std::sync::OnceLock::new();

/// Retained bytes per device. Beyond this the returned handle is simply dropped.
#[cfg(feature = "cuda")]
const OUTPUT_POOL_MAX_RETAINED_BYTES: usize = 64 << 20;

#[cfg(feature = "cuda")]
struct OutputPool {
    free: std::collections::HashMap<usize, Vec<cubecl_runtime::server::Handle>>,
    retained_bytes: usize,
    max_retained_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}

#[cfg(feature = "cuda")]
impl Default for OutputPool {
    fn default() -> Self {
        Self {
            free: std::collections::HashMap::new(),
            retained_bytes: 0,
            max_retained_bytes: OUTPUT_POOL_MAX_RETAINED_BYTES,
            hits: 0,
            misses: 0,
            evictions: 0,
            clears: 0,
        }
    }
}

#[cfg(feature = "cuda")]
fn output_pools() -> &'static std::sync::Mutex<std::collections::HashMap<usize, OutputPool>> {
    OUTPUT_POOLS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Take a retained handle of exactly `byte_len` for this device, if one is free.
#[cfg(feature = "cuda")]
pub(crate) fn take_output_handle(
    device_ordinal: usize,
    byte_len: usize,
) -> Option<cubecl_runtime::server::Handle> {
    let mut pools = output_pools().lock().ok()?;
    let pool = pools.get_mut(&device_ordinal)?;
    let handle = pool.free.get_mut(&byte_len)?.pop()?;
    pool.retained_bytes = pool.retained_bytes.saturating_sub(byte_len);
    pool.hits += 1;
    Some(handle)
}

/// Retain a dropped handle for the next allocation of the same length.
#[cfg(feature = "cuda")]
pub(crate) fn return_output_handle(
    device_ordinal: usize,
    byte_len: usize,
    handle: cubecl_runtime::server::Handle,
) {
    if byte_len == 0 {
        return;
    }
    let Ok(mut pools) = output_pools().lock() else {
        return;
    };
    let pool = pools.entry(device_ordinal).or_default();
    if pool.retained_bytes + byte_len > pool.max_retained_bytes {
        // INVARIANT: a dropped handle is released rather than retained once the
        // device's retained-byte bound is reached, so a length that stops
        // recurring does not pin memory.
        pool.evictions += 1;
        return;
    }
    pool.retained_bytes += byte_len;
    pool.free.entry(byte_len).or_default().push(handle);
}

/// Record that no retained handle of this length was available, so the caller
/// asked the device thread for a fresh allocation.
#[cfg(feature = "cuda")]
pub(crate) fn record_output_pool_miss(device_ordinal: usize) {
    let Ok(mut pools) = output_pools().lock() else {
        return;
    };
    pools.entry(device_ordinal).or_default().misses += 1;
}

/// Snapshot the retained entries, retained bytes and lookup counters of a device.
#[cfg(feature = "cuda")]
pub(crate) fn output_pool_stats(device_ordinal: usize) -> tenferro_tensor::CacheStats {
    let Ok(pools) = output_pools().lock() else {
        return empty_output_pool_stats();
    };
    match pools.get(&device_ordinal) {
        Some(pool) => tenferro_tensor::CacheStats {
            entries: pool.free.values().map(Vec::len).sum(),
            retained_bytes: pool.retained_bytes,
            hits: pool.hits,
            misses: pool.misses,
            evictions: pool.evictions,
            clears: pool.clears,
        },
        None => empty_output_pool_stats(),
    }
}

#[cfg(feature = "cuda")]
fn empty_output_pool_stats() -> tenferro_tensor::CacheStats {
    tenferro_tensor::CacheStats {
        entries: 0,
        retained_bytes: 0,
        hits: 0,
        misses: 0,
        evictions: 0,
        clears: 0,
    }
}

/// Return the retained-byte bound of a device.
#[cfg(feature = "cuda")]
pub(crate) fn output_pool_max_retained_bytes(device_ordinal: usize) -> usize {
    let Ok(mut pools) = output_pools().lock() else {
        return OUTPUT_POOL_MAX_RETAINED_BYTES;
    };
    pools.entry(device_ordinal).or_default().max_retained_bytes
}

/// Configure the retained-byte bound of a device and evict the excess.
#[cfg(feature = "cuda")]
pub(crate) fn set_output_pool_max_retained_bytes(device_ordinal: usize, max_retained_bytes: usize) {
    let Ok(mut pools) = output_pools().lock() else {
        return;
    };
    let pool = pools.entry(device_ordinal).or_default();
    pool.max_retained_bytes = max_retained_bytes;
    while pool.retained_bytes > max_retained_bytes {
        let Some(byte_len) = pool
            .free
            .iter()
            .filter(|(_, handles)| !handles.is_empty())
            .map(|(&byte_len, _)| byte_len)
            .max()
        else {
            break;
        };
        if let Some(handles) = pool.free.get_mut(&byte_len) {
            handles.pop();
        }
        pool.retained_bytes = pool.retained_bytes.saturating_sub(byte_len);
        pool.evictions += 1;
    }
    pool.free.retain(|_, handles| !handles.is_empty());
}

/// Release every handle retained for a device.
#[cfg(feature = "cuda")]
pub(crate) fn clear_output_pool(device_ordinal: usize) {
    let Ok(mut pools) = output_pools().lock() else {
        return;
    };
    if let Some(pool) = pools.get_mut(&device_ordinal) {
        pool.free.clear();
        pool.retained_bytes = 0;
        pool.clears += 1;
    }
}

/// Return a dropped buffer's allocation to the pool of its device.
#[cfg(feature = "cuda")]
impl Drop for CubeclBuffer {
    fn drop(&mut self) {
        return_output_handle(self.device_ordinal, self.byte_len, self.handle.clone());
    }
}

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
            device_addr: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub(crate) fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    /// Return the memoized device address, if one is currently valid.
    pub(crate) fn cached_device_addr(&self) -> Option<u64> {
        match self.device_addr.load(std::sync::atomic::Ordering::Acquire) {
            0 => None,
            addr => Some(addr),
        }
    }

    /// Memoize the device address resolved through `get_resource` for this
    /// buffer's handle.
    pub(crate) fn memoize_device_addr(&self, addr: u64) {
        self.device_addr
            .store(addr, std::sync::atomic::Ordering::Release);
    }

    /// Drop the memoized address because a CubeCL kernel was queued to write
    /// this buffer.
    ///
    /// The next raw-FFI access then resolves through `get_resource`, whose
    /// blocking server round trip also pushes the queued kernel onto the
    /// CUstream, so the vendor call cannot overtake it. See the
    /// `device_addr` invariant and issue #1868.
    pub(crate) fn invalidate_device_addr(&self) {
        self.device_addr
            .store(0, std::sync::atomic::Ordering::Release);
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
