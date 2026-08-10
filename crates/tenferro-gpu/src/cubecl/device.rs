use std::fmt;

use tenferro_tensor::BoxError;

use super::identity::{CudaComputeCapability, CudaDeviceUuid};

#[derive(Debug, thiserror::Error)]
enum CudaDriverDiscoveryError {
    #[error("CUDA driver call {function} failed: {source}")]
    DriverCall {
        function: &'static str,
        #[source]
        source: cudarc::driver::result::DriverError,
    },
    #[error("CUDA returned an invalid device count {count}")]
    InvalidDeviceCount { count: i32 },
    #[error("CUDA device ordinal {device:?} is out of range")]
    DeviceOrdinalOutOfRange { device: CudaDeviceId },
}

fn boxed_discovery_error(error: CudaDriverDiscoveryError) -> BoxError {
    Box::new(error)
}

struct CudaDriverApi;

/// Provider-qualified identity of a CUDA device ordinal.
///
/// A device ID is an opaque CUDA provider value. Use [`Self::ordinal`] only
/// when passing the selected ordinal to CUDA APIs. The ordinal is
/// process-visible and may change when `CUDA_VISIBLE_DEVICES` changes.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaDeviceId;
///
/// let device = CudaDeviceId::from_ordinal(2);
/// assert_eq!(device.ordinal(), 2);
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CudaDeviceId(u32);

impl CudaDeviceId {
    /// Construct a device ID from its CUDA ordinal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceId;
    ///
    /// const DEVICE: CudaDeviceId = CudaDeviceId::from_ordinal(0);
    /// assert_eq!(DEVICE.ordinal(), 0);
    /// ```
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// Return the CUDA ordinal represented by this device ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceId;
    ///
    /// let device = CudaDeviceId::from_ordinal(3);
    /// assert_eq!(device.ordinal(), 3);
    /// ```
    pub const fn ordinal(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for CudaDeviceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CudaDeviceId")
            .field(&self.0)
            .finish()
    }
}

/// Immutable metadata describing one CUDA device.
///
/// Equality and debug output are deterministic only while the process-visible
/// CUDA topology remains unchanged; they do not provide stable identity across
/// topology changes.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::CudaDeviceInfo;
///
/// let _type_name = std::any::type_name::<CudaDeviceInfo>();
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    id: CudaDeviceId,
    name: String,
    uuid: CudaDeviceUuid,
    compute_capability: CudaComputeCapability,
    total_memory_bytes: u64,
}

impl CudaDeviceInfo {
    /// Construct device metadata for a CUDA device.
    pub(crate) fn new(
        id: CudaDeviceId,
        name: impl Into<String>,
        uuid: CudaDeviceUuid,
        compute_capability: CudaComputeCapability,
        total_memory_bytes: u64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            uuid,
            compute_capability,
            total_memory_bytes,
        }
    }

    /// Return this device's provider-qualified ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceInfo;
    ///
    /// let _id = CudaDeviceInfo::id;
    /// ```
    pub fn id(&self) -> CudaDeviceId {
        self.id
    }

    /// Borrow this device's display name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaDeviceInfo;
    ///
    /// let _name = CudaDeviceInfo::name;
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the stable physical/MIG identity of this device.
    ///
    /// The UUID is a durable comparison identity independent of the
    /// process-visible ordinal. It is used for diagnostics, topology, and
    /// explicit cross-runtime/cross-device placement decisions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cuda_devices;
    ///
    /// // `cudarc` panics when the CUDA driver library is absent, so the call
    /// // is guarded (the same pattern `gpu_available` uses).
    /// let devices = std::panic::catch_unwind(cuda_devices)
    ///     .unwrap_or_else(|_| Ok(Vec::new()))
    ///     .unwrap_or_default();
    /// let uuids: Vec<_> = devices.iter().map(|info| info.uuid()).collect();
    /// let _ = uuids;
    /// ```
    pub fn uuid(&self) -> CudaDeviceUuid {
        self.uuid
    }

    /// Return the compute capability of this device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cuda_devices;
    ///
    /// // `cudarc` panics when the CUDA driver library is absent, so the call
    /// // is guarded (the same pattern `gpu_available` uses).
    /// let devices = std::panic::catch_unwind(cuda_devices)
    ///     .unwrap_or_else(|_| Ok(Vec::new()))
    ///     .unwrap_or_default();
    /// let capabilities: Vec<_> = devices
    ///     .iter()
    ///     .map(|info| info.compute_capability())
    ///     .collect();
    /// let _ = capabilities;
    /// ```
    pub fn compute_capability(&self) -> CudaComputeCapability {
        self.compute_capability
    }

    /// Return the total device memory in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cuda_devices;
    ///
    /// // `cudarc` panics when the CUDA driver library is absent, so the call
    /// // is guarded (the same pattern `gpu_available` uses).
    /// let devices = std::panic::catch_unwind(cuda_devices)
    ///     .unwrap_or_else(|_| Ok(Vec::new()))
    ///     .unwrap_or_default();
    /// let memory: Vec<u64> = devices
    ///     .iter()
    ///     .map(|info| info.total_memory_bytes())
    ///     .collect();
    /// let _ = memory;
    /// ```
    pub fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }
}

pub(crate) trait DiscoveryDriver {
    fn initialize(&self) -> Result<(), BoxError>;

    fn device_count(&self) -> Result<u32, BoxError>;

    fn device_name(&self, device: CudaDeviceId) -> Result<String, BoxError>;

    fn device_uuid(&self, device: CudaDeviceId) -> Result<CudaDeviceUuid, BoxError>;

    fn compute_capability(&self, device: CudaDeviceId) -> Result<CudaComputeCapability, BoxError>;

    fn total_memory_bytes(&self, device: CudaDeviceId) -> Result<u64, BoxError>;
}

pub(crate) fn discover_with(
    driver: &impl DiscoveryDriver,
) -> Result<Vec<CudaDeviceInfo>, CudaDeviceError> {
    driver
        .initialize()
        .map_err(|source| CudaDeviceError::Discovery {
            operation: "initialize_driver",
            source,
        })?;
    let device_count = driver
        .device_count()
        .map_err(|source| CudaDeviceError::Discovery {
            operation: "enumerate_devices",
            source,
        })?;

    let mut devices = Vec::new();
    for ordinal in 0..device_count {
        let id = CudaDeviceId::from_ordinal(ordinal);
        let name = driver
            .device_name(id)
            .map_err(|source| CudaDeviceError::Discovery {
                operation: "get_device_name",
                source,
            })?;
        let uuid = driver
            .device_uuid(id)
            .map_err(|source| CudaDeviceError::Discovery {
                operation: "get_device_uuid",
                source,
            })?;
        let compute_capability =
            driver
                .compute_capability(id)
                .map_err(|source| CudaDeviceError::Discovery {
                    operation: "get_compute_capability",
                    source,
                })?;
        let total_memory_bytes =
            driver
                .total_memory_bytes(id)
                .map_err(|source| CudaDeviceError::Discovery {
                    operation: "get_total_memory",
                    source,
                })?;
        devices.push(CudaDeviceInfo::new(
            id,
            name,
            uuid,
            compute_capability,
            total_memory_bytes,
        ));
    }
    Ok(devices)
}

impl DiscoveryDriver for CudaDriverApi {
    fn initialize(&self) -> Result<(), BoxError> {
        cudarc::driver::result::init().map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuInit",
                source,
            })
        })
    }

    fn device_count(&self) -> Result<u32, BoxError> {
        let count = cudarc::driver::result::device::get_count().map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGetCount",
                source,
            })
        })?;
        u32::try_from(count).map_err(|_| {
            boxed_discovery_error(CudaDriverDiscoveryError::InvalidDeviceCount { count })
        })
    }

    fn device_name(&self, device: CudaDeviceId) -> Result<String, BoxError> {
        let cuda_device = self.cuda_device(device)?;
        let name = cudarc::driver::result::device::get_name(cuda_device).map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGetName",
                source,
            })
        })?;
        Ok(name)
    }

    fn device_uuid(&self, device: CudaDeviceId) -> Result<CudaDeviceUuid, BoxError> {
        let cuda_device = self.cuda_device(device)?;
        let uuid = cudarc::driver::result::device::get_uuid(cuda_device).map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGetUuid",
                source,
            })
        })?;
        let mut bytes = [0u8; 16];
        #[allow(clippy::needless_range_loop)]
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = uuid.bytes[index] as u8;
        }
        Ok(CudaDeviceUuid::from_bytes(bytes))
    }

    fn compute_capability(&self, device: CudaDeviceId) -> Result<CudaComputeCapability, BoxError> {
        let cuda_device = self.cuda_device(device)?;
        use cudarc::driver::sys::CUdevice_attribute_enum as Attr;
        let (major_name, minor_name) = unsafe {
            (
                cudarc::driver::result::device::get_attribute(
                    cuda_device,
                    Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                ),
                cudarc::driver::result::device::get_attribute(
                    cuda_device,
                    Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                ),
            )
        };
        let major = major_name.map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGetAttribute(CC_MAJOR)",
                source,
            })
        })?;
        let minor = minor_name.map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGetAttribute(CC_MINOR)",
                source,
            })
        })?;
        Ok(CudaComputeCapability {
            major: u32::try_from(major).map_err(|_| {
                boxed_discovery_error(CudaDriverDiscoveryError::InvalidDeviceCount { count: major })
            })?,
            minor: u32::try_from(minor).map_err(|_| {
                boxed_discovery_error(CudaDriverDiscoveryError::InvalidDeviceCount { count: minor })
            })?,
        })
    }

    fn total_memory_bytes(&self, device: CudaDeviceId) -> Result<u64, BoxError> {
        let cuda_device = self.cuda_device(device)?;
        // SAFETY: `cuda_device` was returned by `cuDeviceGet` for this session.
        let bytes = unsafe { cudarc::driver::result::device::total_mem(cuda_device) }.map_err(
            |source| {
                boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                    function: "cuDeviceTotalMem",
                    source,
                })
            },
        )?;
        u64::try_from(bytes).map_err(|_| {
            boxed_discovery_error(CudaDriverDiscoveryError::InvalidDeviceCount { count: i32::MAX })
        })
    }
}

impl CudaDriverApi {
    fn cuda_device(&self, device: CudaDeviceId) -> Result<cudarc::driver::sys::CUdevice, BoxError> {
        let ordinal = i32::try_from(device.ordinal()).map_err(|_| {
            boxed_discovery_error(CudaDriverDiscoveryError::DeviceOrdinalOutOfRange { device })
        })?;
        cudarc::driver::result::device::get(ordinal).map_err(|source| {
            boxed_discovery_error(CudaDriverDiscoveryError::DriverCall {
                function: "cuDeviceGet",
                source,
            })
        })
    }
}

/// Discover CUDA devices visible to the current process.
///
/// Discovery initializes the CUDA driver and queries device ordinals directly.
/// It does not create a CUDA context, CUDA runtime, CubeCL runtime, backend, or
/// client. Device ordinals are process-visible and may change when
/// `CUDA_VISIBLE_DEVICES` changes. The returned values have deterministic
/// `Eq` and `Debug` behavior only while that process-visible topology remains
/// unchanged.
///
/// # Errors
///
/// Returns [`CudaDeviceError::Discovery`] for driver initialization, device
/// enumeration, or device-name lookup failures.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{cuda::cuda_devices, cuda::CudaDeviceError, cuda::CudaDeviceInfo};
///
/// let _discover: fn() -> Result<Vec<CudaDeviceInfo>, CudaDeviceError> = cuda_devices;
/// ```
pub fn cuda_devices() -> Result<Vec<CudaDeviceInfo>, CudaDeviceError> {
    discover_with(&CudaDriverApi)
}

pub(crate) fn unavailable_device_error(
    requested: CudaDeviceId,
    discovered: Vec<CudaDeviceInfo>,
) -> CudaDeviceError {
    CudaDeviceError::Unavailable {
        requested,
        discovered: discovered.into_boxed_slice(),
    }
}

/// Structured failures from CUDA device discovery and initialization.
///
/// The error retains only provider-neutral device identities and metadata. It
/// does not expose the concrete CUDA runtime error type behind a source.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId};
///
/// let error = CudaDeviceError::Unavailable {
///     requested: CudaDeviceId::from_ordinal(1),
///     discovered: Vec::new().into_boxed_slice(),
/// };
/// assert_eq!(error.requested(), Some(CudaDeviceId::from_ordinal(1)));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaDeviceError {
    /// Device discovery failed while performing the named operation.
    #[error("CUDA device discovery failed during {operation}: {source}")]
    Discovery {
        /// The provider-neutral discovery operation being performed.
        operation: &'static str,
        /// The underlying discovery failure.
        #[source]
        source: tenferro_tensor::BoxError,
    },
    /// The requested device was not present in the discovered device list.
    #[error(
        "requested CUDA device {requested:?} is unavailable; discovered devices: {discovered:?}"
    )]
    Unavailable {
        /// The device selected by the caller.
        requested: CudaDeviceId,
        /// Devices found during discovery, in discovery order.
        discovered: Box<[CudaDeviceInfo]>,
    },
    /// Device initialization failed while performing the named operation.
    #[error("CUDA device {device:?} initialization failed during {operation}: {source}")]
    Initialization {
        /// The device whose runtime could not be initialized.
        device: CudaDeviceId,
        /// The provider-neutral initialization operation being performed.
        operation: &'static str,
        /// The underlying initialization failure.
        #[source]
        source: tenferro_tensor::BoxError,
    },
}

impl CudaDeviceError {
    /// Return the operation associated with discovery or initialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId};
    ///
    /// let error = CudaDeviceError::Unavailable {
    ///     requested: CudaDeviceId::from_ordinal(1),
    ///     discovered: Vec::new().into_boxed_slice(),
    /// };
    /// assert_eq!(error.operation(), None);
    /// ```
    pub fn operation(&self) -> Option<&'static str> {
        match self {
            Self::Discovery { operation, .. } | Self::Initialization { operation, .. } => {
                Some(operation)
            }
            Self::Unavailable { .. } => None,
        }
    }

    /// Return the requested device for an unavailable-device error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId};
    ///
    /// let requested = CudaDeviceId::from_ordinal(1);
    /// let error = CudaDeviceError::Unavailable {
    ///     requested,
    ///     discovered: Vec::new().into_boxed_slice(),
    /// };
    /// assert_eq!(error.requested(), Some(requested));
    /// ```
    pub fn requested(&self) -> Option<CudaDeviceId> {
        match self {
            Self::Unavailable { requested, .. } => Some(*requested),
            Self::Discovery { .. } | Self::Initialization { .. } => None,
        }
    }

    /// Borrow the devices discovered for an unavailable-device error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId};
    ///
    /// let error = CudaDeviceError::Unavailable {
    ///     requested: CudaDeviceId::from_ordinal(1),
    ///     discovered: Vec::new().into_boxed_slice(),
    /// };
    /// assert!(error.discovered().is_some_and(<[_]>::is_empty));
    /// ```
    pub fn discovered(&self) -> Option<&[CudaDeviceInfo]> {
        match self {
            Self::Unavailable { discovered, .. } => Some(discovered),
            Self::Discovery { .. } | Self::Initialization { .. } => None,
        }
    }

    /// Return the device whose initialization failed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{cuda::CudaDeviceError, cuda::CudaDeviceId};
    ///
    /// let device = CudaDeviceId::from_ordinal(1);
    /// let error = CudaDeviceError::Initialization {
    ///     device,
    ///     operation: "create_client",
    ///     source: Box::new(std::io::Error::other("context failed")),
    /// };
    /// assert_eq!(error.device(), Some(device));
    /// ```
    pub fn device(&self) -> Option<CudaDeviceId> {
        match self {
            Self::Initialization { device, .. } => Some(*device),
            Self::Discovery { .. } | Self::Unavailable { .. } => None,
        }
    }
}
