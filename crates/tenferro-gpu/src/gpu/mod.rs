//! Backend-neutral GPU seam: storage identity, the allocation a buffer carries, and the
//! failure vocabulary every GPU backend reports through.
//!
//! Nothing here names a CUDA (or any other vendor) type. A backend implements [`Backend`]
//! to name its own allocation handle; the shared layer stores and passes allocations as
//! [`Buffer<B>`] and never reads a vendor type. The seam exists so a second backend
//! (ROCm/HIP, then Metal) is added without touching the dispatch, transfer or publication
//! code.
//!
//! # Examples
//!
//! ```
//! use tenferro_gpu::gpu::{GpuError, SeamError};
//!
//! let error = GpuError::unsupported("gemm", "the vendor adapter is not installed");
//! assert_eq!(error.kind(), SeamError::Unsupported);
//! ```

use tenferro_tensor::{AllocationDomainId, AllocationId};

/// A failure a GPU backend reports through the seam.
///
/// The provider-specific error types keep their own variants and ownership packages
/// (a rejected submission hands the submitted value back, for example); this type is
/// the seam's *cause* vocabulary only, and every backend maps into it one way.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub enum GpuError {
    /// No device or runtime is available for the request.
    Unavailable {
        /// What was probed and why it failed.
        detail: String,
    },
    /// The operation is not served by this backend or this build.
    Unsupported {
        /// Operation name.
        op: &'static str,
        /// Why it is not served.
        detail: String,
    },
    /// A buffer or device belongs to another backend or allocation domain.
    ForeignDomain,
    /// The backend cannot admit the work within its pending budget.
    Backpressure {
        /// Bytes the admission needs.
        bytes: usize,
    },
    /// The runtime or engine has been closed.
    Closed,
    /// The transfer failed, with the route that failed.
    Transfer {
        /// Whether the failed route staged through host memory.
        staged: bool,
        /// What the backend reported.
        detail: String,
    },
    /// An earlier failure left the backend unusable.
    Broken {
        /// The original failure.
        detail: String,
    },
}

/// The category of a [`GpuError`], for callers that only branch on the class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum SeamError {
    /// No device or runtime.
    Unavailable,
    /// Not served by this backend or build.
    Unsupported,
    /// Another backend or domain.
    ForeignDomain,
    /// Out of pending budget.
    Backpressure,
    /// Closed.
    Closed,
    /// A transfer failed.
    Transfer,
    /// Unusable after an earlier failure.
    Broken,
}

impl GpuError {
    /// The category of this failure.
    #[doc(hidden)]
    pub fn kind(&self) -> SeamError {
        match self {
            GpuError::Unavailable { .. } => SeamError::Unavailable,
            GpuError::Unsupported { .. } => SeamError::Unsupported,
            GpuError::ForeignDomain => SeamError::ForeignDomain,
            GpuError::Backpressure { .. } => SeamError::Backpressure,
            GpuError::Closed => SeamError::Closed,
            GpuError::Transfer { .. } => SeamError::Transfer,
            GpuError::Broken { .. } => SeamError::Broken,
        }
    }

    /// The cause of an unsupported operation.
    #[doc(hidden)]
    pub fn unsupported(op: &'static str, detail: impl Into<String>) -> GpuError {
        GpuError::Unsupported {
            op,
            detail: detail.into(),
        }
    }

    /// The cause of a transfer failure.
    #[doc(hidden)]
    pub fn transfer(staged: bool, detail: impl Into<String>) -> GpuError {
        GpuError::Transfer {
            staged,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Unavailable { detail } => write!(f, "GPU backend unavailable: {detail}"),
            GpuError::Unsupported { op, detail } => write!(f, "{op} is not served: {detail}"),
            GpuError::ForeignDomain => f.write_str("the operand belongs to another GPU domain"),
            GpuError::Backpressure { bytes } => {
                write!(f, "the GPU pending budget cannot admit {bytes} bytes")
            }
            GpuError::Closed => f.write_str("the GPU runtime has been closed"),
            GpuError::Transfer { staged, detail } => {
                let route = if *staged { "staged" } else { "direct" };
                write!(f, "{route} transfer failed: {detail}")
            }
            GpuError::Broken { detail } => {
                write!(
                    f,
                    "the GPU backend is unusable after an earlier failure: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// Identity of one storage allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct StorageIdentity {
    /// Shared-allocation domain of the runtime that owns the allocation.
    pub domain: AllocationDomainId,
    /// Stable identity of the allocation inside that domain.
    pub allocation: AllocationId,
    /// Device ordinal that owns the allocation.
    pub device: usize,
}

/// One GPU backend: the concrete types the seam carries without naming them.
///
/// The shared layer stores and passes allocations through [`Buffer`] and never reads a
/// backend type. A second backend supplies its own allocation handle here; the execution
/// vocabulary that lets it reuse the dispatch, transfer and publication code lands with
/// that vocabulary's first implementor rather than as an unused declaration.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::gpu::{Backend, Buffer, StorageIdentity};
/// use tenferro_tensor::AllocationDomainId;
///
/// struct ExampleBackend;
/// impl Backend for ExampleBackend {
///     type Allocation = u64;
/// }
///
/// let identity = StorageIdentity {
///     domain: AllocationDomainId::fresh(),
///     allocation: tenferro_tensor::AllocationId::from_backend_id(2),
///     device: 0,
/// };
/// let buffer = Buffer::<ExampleBackend>::new(identity, 64, 0xdead_beef);
/// assert_eq!(buffer.len(), 64);
/// assert_eq!(*buffer.allocation(), 0xdead_beef);
/// ```
pub trait Backend: Send + 'static {
    /// One device allocation. The [`Buffer`] that carries it owns it.
    type Allocation: Send + 'static;
}

/// A backend allocation the seam carries by identity, owning the handle itself.
///
/// The handle is the backend's; the seam carries the identity every consumer needs
/// (domain, allocation, device), the byte length, and the allocation the adapter serves
/// vendor and kernel calls with. A constructor is the backend allocator's, so identity
/// and handle cannot disagree.
#[doc(hidden)]
pub struct Buffer<B: Backend> {
    identity: StorageIdentity,
    bytes: usize,
    allocation: B::Allocation,
}

impl<B: Backend> Buffer<B> {
    /// Build a seam buffer from a backend's identity, byte length and allocation.
    #[doc(hidden)]
    pub fn new(identity: StorageIdentity, bytes: usize, allocation: B::Allocation) -> Buffer<B> {
        Buffer {
            identity,
            bytes,
            allocation,
        }
    }

    /// The allocation's identity.
    #[doc(hidden)]
    pub fn identity(&self) -> StorageIdentity {
        self.identity
    }

    /// Allocation length in bytes.
    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Whether the allocation is empty.
    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// The backend's own allocation handle.
    #[doc(hidden)]
    pub fn allocation(&self) -> &B::Allocation {
        &self.allocation
    }
}

impl<B: Backend> std::fmt::Debug for Buffer<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("identity", &self.identity)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
