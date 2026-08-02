use std::fmt;

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
/// use tenferro_gpu::{CudaDeviceId, CudaDeviceInfo};
///
/// let info = CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA GPU");
/// assert_eq!(info.name(), "NVIDIA GPU");
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    id: CudaDeviceId,
    name: String,
}

impl CudaDeviceInfo {
    /// Construct device metadata for a CUDA device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{CudaDeviceId, CudaDeviceInfo};
    ///
    /// let info = CudaDeviceInfo::new(CudaDeviceId::from_ordinal(1), "NVIDIA GPU");
    /// assert_eq!(info.id().ordinal(), 1);
    /// ```
    pub fn new(id: CudaDeviceId, name: impl Into<String>) -> Self {
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
    /// use tenferro_gpu::{CudaDeviceId, CudaDeviceInfo};
    ///
    /// let id = CudaDeviceId::from_ordinal(4);
    /// let info = CudaDeviceInfo::new(id, "NVIDIA GPU");
    /// assert_eq!(info.id(), id);
    /// ```
    pub fn id(&self) -> CudaDeviceId {
        self.id
    }

    /// Borrow this device's display name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::{CudaDeviceId, CudaDeviceInfo};
    ///
    /// let info = CudaDeviceInfo::new(CudaDeviceId::from_ordinal(0), "NVIDIA GPU");
    /// assert_eq!(info.name(), "NVIDIA GPU");
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::{CudaDeviceId, CudaDeviceInfo};

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
}
