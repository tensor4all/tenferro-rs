use std::fmt;

use tenferro_tensor::BoxError;

/// Provider-qualified identity of a CUDA device ordinal.
///
/// A device ID is an opaque CUDA provider value. Use [`Self::ordinal`] only
/// when passing the selected ordinal to CUDA APIs.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::CudaDeviceId;
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
    /// use tenferro_gpu::CudaDeviceId;
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
    /// use tenferro_gpu::CudaDeviceId;
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
/// # Examples
///
/// ```
/// use tenferro_gpu::CudaDeviceInfo;
///
/// let _type_name = std::any::type_name::<CudaDeviceInfo>();
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    id: CudaDeviceId,
    name: String,
}

impl CudaDeviceInfo {
    /// Construct device metadata for a CUDA device.
    pub(crate) fn new(id: CudaDeviceId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }

    /// Return this device's provider-qualified ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaDeviceInfo;
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
    /// use tenferro_gpu::CudaDeviceInfo;
    ///
    /// let _name = CudaDeviceInfo::name;
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }
}

trait DiscoveryDriver {
    fn initialize(&self) -> Result<(), BoxError>;

    fn device_count(&self) -> Result<u32, BoxError>;

    fn device_name(&self, device: CudaDeviceId) -> Result<String, BoxError>;
}

fn discover_with(driver: &impl DiscoveryDriver) -> Result<Vec<CudaDeviceInfo>, CudaDeviceError> {
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
        devices.push(CudaDeviceInfo::new(id, name));
    }
    Ok(devices)
}

/// Structured failures from CUDA device discovery and initialization.
///
/// The error retains only provider-neutral device identities and metadata. It
/// does not expose the concrete CUDA runtime error type behind a source.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::{CudaDeviceError, CudaDeviceId};
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
    /// use tenferro_gpu::{CudaDeviceError, CudaDeviceId};
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
    /// use tenferro_gpu::{CudaDeviceError, CudaDeviceId};
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
    /// use tenferro_gpu::{CudaDeviceError, CudaDeviceId};
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
    /// use tenferro_gpu::{CudaDeviceError, CudaDeviceId};
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

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::error::Error as _;
    use std::hash::{Hash, Hasher};

    use tenferro_tensor::BoxError;

    use super::{discover_with, CudaDeviceError, CudaDeviceId, CudaDeviceInfo, DiscoveryDriver};

    struct FakeDriver {
        names: Vec<String>,
    }

    impl DiscoveryDriver for FakeDriver {
        fn initialize(&self) -> Result<(), BoxError> {
            Ok(())
        }

        fn device_count(&self) -> Result<u32, BoxError> {
            Ok(self.names.len() as u32)
        }

        fn device_name(&self, device: CudaDeviceId) -> Result<String, BoxError> {
            Ok(self.names[device.ordinal() as usize].clone())
        }
    }

    fn assert_cuda_device_id_traits<T>()
    where
        T: Copy + Clone + Eq + PartialEq + Ord + PartialOrd + Hash,
    {
    }

    #[test]
    fn cuda_device_id_has_value_semantics_and_deterministic_debug() {
        const ID: CudaDeviceId = CudaDeviceId::from_ordinal(7);

        assert_cuda_device_id_traits::<CudaDeviceId>();
        assert_eq!(ID.ordinal(), 7);
        assert!(ID < CudaDeviceId::from_ordinal(8));
        assert_eq!(format!("{ID:?}"), "CudaDeviceId(7)");

        let mut first_hasher = DefaultHasher::new();
        ID.hash(&mut first_hasher);
        let mut second_hasher = DefaultHasher::new();
        CudaDeviceId::from_ordinal(7).hash(&mut second_hasher);
        assert_eq!(first_hasher.finish(), second_hasher.finish());
    }

    #[test]
    fn cuda_device_info_exposes_id_and_name() {
        let id = CudaDeviceId::from_ordinal(2);
        let info = CudaDeviceInfo::new(id, "NVIDIA H100");

        assert_eq!(info.id(), id);
        assert_eq!(info.name(), "NVIDIA H100");
        assert_eq!(info, info.clone());
    }

    #[test]
    fn discovery_of_zero_devices_returns_empty() {
        let driver = FakeDriver { names: Vec::new() };

        assert!(discover_with(&driver).unwrap().is_empty());
    }

    #[test]
    fn discovery_preserves_ordinal_order_and_is_deterministic() {
        let driver = FakeDriver {
            names: vec!["NVIDIA A100".into(), "NVIDIA H100".into()],
        };
        let expected = vec![
            CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA A100"),
            CudaDeviceInfo::new(CudaDeviceId::from_ordinal(1), "NVIDIA H100"),
        ];

        let first = discover_with(&driver).unwrap();
        let second = discover_with(&driver).unwrap();

        assert_eq!(first, expected);
        assert_eq!(first, second);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }

    #[test]
    fn cuda_device_error_discovery_preserves_fields_and_source() {
        let error = CudaDeviceError::Discovery {
            operation: "enumerate_devices",
            source: Box::new(std::io::Error::other("driver query failed")),
        };

        assert!(matches!(
            &error,
            CudaDeviceError::Discovery {
                operation: "enumerate_devices",
                source,
            } if source.to_string() == "driver query failed"
        ));
        assert_eq!(
            error.to_string(),
            "CUDA device discovery failed during enumerate_devices: driver query failed"
        );
        assert_eq!(error.operation(), Some("enumerate_devices"));
        assert_eq!(error.requested(), None);
        assert_eq!(error.discovered(), None);
        assert_eq!(error.device(), None);
        assert_eq!(
            error.source().map(ToString::to_string).as_deref(),
            Some("driver query failed")
        );
    }

    #[test]
    fn cuda_device_error_unavailable_preserves_fields_without_source() {
        let requested = CudaDeviceId::from_ordinal(2);
        let discovered = vec![
            CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
            CudaDeviceInfo::new(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
        ]
        .into_boxed_slice();
        let error = CudaDeviceError::Unavailable {
            requested,
            discovered,
        };

        assert!(matches!(
            &error,
            CudaDeviceError::Unavailable {
                requested: actual_requested,
                discovered: actual_discovered,
            } if *actual_requested == requested
                && actual_discovered.as_ref() == [
                    CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
                    CudaDeviceInfo::new(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
                ]
        ));
        assert_eq!(
            error.to_string(),
            "requested CUDA device CudaDeviceId(2) is unavailable; discovered devices: [CudaDeviceInfo { id: CudaDeviceId(0), name: \"NVIDIA H100\" }, CudaDeviceInfo { id: CudaDeviceId(1), name: \"NVIDIA A100\" }]"
        );
        assert_eq!(error.operation(), None);
        assert_eq!(error.requested(), Some(requested));
        assert_eq!(
            error.discovered(),
            Some(
                [
                    CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA H100"),
                    CudaDeviceInfo::new(CudaDeviceId::from_ordinal(1), "NVIDIA A100"),
                ]
                .as_slice()
            )
        );
        assert_eq!(error.device(), None);
        assert!(error.source().is_none());
    }

    #[test]
    fn cuda_device_error_initialization_preserves_fields_and_source() {
        let device = CudaDeviceId::from_ordinal(1);
        let error = CudaDeviceError::Initialization {
            device,
            operation: "create_client",
            source: Box::new(std::io::Error::other("CUDA context failed")),
        };

        assert!(matches!(
            &error,
            CudaDeviceError::Initialization {
                device: actual_device,
                operation: "create_client",
                source,
            } if *actual_device == device && source.to_string() == "CUDA context failed"
        ));
        assert_eq!(
            error.to_string(),
            "CUDA device CudaDeviceId(1) initialization failed during create_client: CUDA context failed"
        );
        assert_eq!(error.operation(), Some("create_client"));
        assert_eq!(error.requested(), None);
        assert_eq!(error.discovered(), None);
        assert_eq!(error.device(), Some(device));
        assert_eq!(
            error.source().map(ToString::to_string).as_deref(),
            Some("CUDA context failed")
        );
    }
}
