//! Provider-neutral GPU extension vocabulary (issue #1597).
//!
//! These types describe capabilities and stable device identity without
//! forcing CUDA-, CubeCL-, or WebGPU-specific concepts into a shared surface.
//! Backend-specific operations live in their provider namespaces
//! (`cuda`, `webgpu`); the types here are shared vocabulary only.

use std::fmt;

/// Stable physical or MIG device identity as a 16-byte UUID.
///
/// This is the durable comparison identity for diagnostics and topology. It is
/// distinct from the process-visible ordinal [`CudaDeviceId`](crate::cuda::CudaDeviceId);
/// the two must never be conflated in equality semantics.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaDeviceUuid;
///
/// let uuid = CudaDeviceUuid::from_bytes([7; 16]);
/// assert_eq!(uuid.to_bytes(), [7; 16]);
/// assert_eq!(uuid, CudaDeviceUuid::from_bytes([7; 16]));
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CudaDeviceUuid([u8; 16]);

impl CudaDeviceUuid {
    /// Construct a device UUID from its 16 raw bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceUuid;
    ///
    /// const UUID: CudaDeviceUuid = CudaDeviceUuid::from_bytes([1; 16]);
    /// assert_eq!(UUID.as_bytes(), &[1; 16]);
    /// ```
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Borrow the 16 raw UUID bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceUuid;
    ///
    /// let uuid = CudaDeviceUuid::from_bytes([2; 16]);
    /// assert_eq!(uuid.as_bytes(), &[2; 16]);
    /// ```
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Return the 16 raw UUID bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceUuid;
    ///
    /// let uuid = CudaDeviceUuid::from_bytes([3; 16]);
    /// assert_eq!(uuid.to_bytes(), [3; 16]);
    /// ```
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }
}

impl fmt::Debug for CudaDeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CudaDeviceUuid").field(&self.0).finish()
    }
}

/// CUDA compute capability `major.minor` (e.g. 9.0 for Hopper-class parts).
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaComputeCapability;
///
/// let cc = CudaComputeCapability { major: 9, minor: 0 };
/// assert_eq!(cc.major, 9);
/// assert_eq!(cc.minor, 0);
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CudaComputeCapability {
    /// Major component of the compute capability.
    pub major: u32,
    /// Minor component of the compute capability.
    pub minor: u32,
}

/// Capability a GPU extension may query before using a provider-specific path.
///
/// [`CudaExecSession::supports`](crate::cuda::CudaExecSession::supports) reports
/// whether the capability is plausibly available on the current session. This is
/// orthogonal to the primitive [`OperationCapability`](tenferro_tensor::OperationCapability)
/// matrix: extension capabilities describe what the extension seam itself can
/// do, not which tensor operations the backend implements.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::GpuExtensionCapability;
///
/// let capability = GpuExtensionCapability::RawStream;
/// assert!(matches!(capability, GpuExtensionCapability::RawStream));
/// ```
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum GpuExtensionCapability {
    /// Launch external kernels through the CubeCL client.
    CubeClKernel,
    /// Load native (PTX/CUBIN/SPIR-V) module artifacts.
    NativeModule,
    /// Compile kernel source at runtime (NVRTC / shader compiler).
    RuntimeCompilation,
    /// Borrow a raw provider stream for library interop.
    RawStream,
    /// Asynchronous same-device copy across runtime/domain boundaries.
    SameDeviceAsyncCopy,
    /// Directional peer-to-peer copy between devices.
    PeerCopy,
}
