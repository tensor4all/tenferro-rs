//! Backend-neutral GPU seam: storage identity, the transfer protocol and the two
//! provider traits every GPU backend implements.
//!
//! Nothing here names a CUDA (or any other vendor) type. A backend supplies the
//! concrete storage handle behind [`Buffer`]'s private enum, one [`KernelSource`]
//! for its kernel catalogue and one [`VendorOps`] for its vendor libraries. The
//! seam exists so a second backend (ROCm/HIP, then Metal) is added without touching
//! the dispatch, transfer or publication code.
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

/// A backend allocation handle the seam knows only by identity.
///
/// A backend keeps its concrete handle in its own module; the seam carries the
/// identity every consumer needs (domain, allocation, device) and the byte length.
#[derive(Debug)]
#[doc(hidden)]
pub struct Buffer {
    identity: StorageIdentity,
    bytes: usize,
}

impl Buffer {
    /// Build a seam buffer from a backend's identity and byte length.
    #[doc(hidden)]
    pub fn new(identity: StorageIdentity, bytes: usize) -> Buffer {
        Buffer { identity, bytes }
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
}

/// A plan for one operation, produced by a [`KernelSource`].
///
/// The concrete representation is the backend's: a bound module function plus the
/// prepared artifact key for CUDA. It stays opaque here so no backend type crosses
/// the seam.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct Program {
    /// Symbol the program launches.
    pub entry: &'static str,
    /// Identity of the prepared artifact, part of any cache key.
    pub artifact: u64,
}

/// One runtime's source of launchable programs.
///
/// `Send` is required because an engine moves between threads; the trait is not
/// `Sync`, because planning borrows the backend's mutable caches.
#[doc(hidden)]
pub trait KernelSource: Send {
    /// Plan one operation into a launchable program.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] for an operation this backend does not serve and
    /// [`GpuError::Closed`] after teardown.
    #[doc(hidden)]
    fn plan(&mut self, op: &OpDescriptor) -> Result<Program, GpuError>;
}

/// Description of one operation to plan.
///
/// The concrete tags live with the dispatch layer; the seam only needs an identity
/// to key a plan by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct OpDescriptor {
    /// Family name, for diagnostics.
    pub family: &'static str,
    /// Stable tag of the operation inside its family.
    pub tag: u32,
}

/// The vendor libraries one backend serves.
///
/// The concrete request types (GEMM, contraction, linalg, transform) land with their
/// adapters in U4; U1 fixes only the trait and its borrow direction: a request borrows
/// the engine's buffers and scalars, and the adapter resolves the runtime's stream and
/// plans through the runtime it was built with. `Send` for the same reason as
/// [`KernelSource`].
#[doc(hidden)]
pub trait VendorOps: Send {
    /// Serve one vendor operation described by `op` over `buffers`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] when the adapter does not serve the operation, plus the
    /// backend's typed vendor failures.
    #[doc(hidden)]
    fn serve(&mut self, op: &OpDescriptor, buffers: &[&Buffer]) -> Result<(), GpuError>;
}
