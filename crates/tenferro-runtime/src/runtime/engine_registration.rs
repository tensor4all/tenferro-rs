use std::fmt;
use std::sync::Arc;

use super::{
    execution, CoreCapabilityBundle, EngineId, ExecutionContextIdentity, HardwareClassId,
    RuntimeCacheOwner, RuntimeConfigError, StorageClass,
};
use tenferro_tensor::TensorBackend;

#[derive(Debug)]
struct CandidateRegistrationToken;

/// Immutable direct engine registration candidate.
///
/// `EngineRegistration` values can be cloned and registered repeatedly. Before
/// publication, candidate identity is the pair of engine ID and an internal
/// pointer token, so reusing the same cloned candidate is idempotent while a
/// distinct value for the same engine ID is a conflict unless explicitly
/// replaced.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::sync::Arc;
/// use tenferro_runtime::{
///     CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
///     HardwareClassId, StorageClass,
/// };
///
/// let storage = StorageClass::new("tenferro.storage.host")?;
/// let registration = EngineRegistration::new(
///     EngineId::new("tenferro.cpu")?,
///     ExecutionContextIdentity::of::<()>(),
///     HardwareClassId::new("tenferro.cpu.host")?,
///     Arc::from(vec![storage.clone()]),
///     storage,
///     CoreCapabilityBundle::builder().build(),
/// )?;
/// assert_eq!(registration.engine_id().as_str(), "tenferro.cpu");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct EngineRegistration {
    engine_id: EngineId,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    capabilities: CoreCapabilityBundle,
    pub(super) cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    pub(super) execution_engine: Option<Arc<dyn execution::ErasedTensorBackendExecutor>>,
    candidate_token: Arc<CandidateRegistrationToken>,
}

impl EngineRegistration {
    /// Build an immutable engine registration candidate.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::EmptyStorageClasses`] for an empty storage
    /// list, [`RuntimeConfigError::DuplicateStorageClass`] for duplicate storage
    /// classes, or [`RuntimeConfigError::DefaultStorageClassNotListed`] when the
    /// default storage class is not supported by this engine.
    pub fn new(
        engine_id: EngineId,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        capabilities: CoreCapabilityBundle,
    ) -> Result<Self, RuntimeConfigError> {
        validate_storage_classes(&engine_id, &storage_classes, &default_storage_class)?;
        Ok(Self {
            engine_id,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            capabilities,
            cache_owner: None,
            execution_engine: None,
            candidate_token: Arc::new(CandidateRegistrationToken),
        })
    }

    /// Return the engine identifier.
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// Return the execution-context type identity accepted by the engine.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    /// Return the hardware class exposed by this engine.
    pub fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }

    /// Return the supported storage classes in registration order.
    pub fn storage_classes(&self) -> &[StorageClass] {
        &self.storage_classes
    }

    /// Return the default storage class.
    pub fn default_storage_class(&self) -> &StorageClass {
        &self.default_storage_class
    }

    /// Return direct core capability slots.
    pub fn capabilities(&self) -> &CoreCapabilityBundle {
        &self.capabilities
    }

    /// Attach a runtime cache owner to this registration.
    pub fn with_cache_owner(mut self, owner: Arc<dyn RuntimeCacheOwner>) -> Self {
        self.cache_owner = Some(owner);
        self
    }

    /// Attach a runtime-owned tensor backend execution bridge to this
    /// registration.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
    ///     HardwareClassId, StorageClass,
    /// };
    ///
    /// let storage = StorageClass::new("tenferro.storage.host")?;
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("tenferro.cpu")?,
    ///     ExecutionContextIdentity::of::<CpuBackend>(),
    ///     HardwareClassId::new("tenferro.cpu.host")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::builder().build(),
    /// )?
    /// .with_tensor_backend_executor(CpuBackend::new());
    /// assert!(registration.has_execution_engine());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tensor_backend_executor<B>(mut self, backend: B) -> Self
    where
        B: TensorBackend + Send + Sync + 'static,
    {
        self.execution_engine = Some(execution::erased_tensor_backend_executor(backend));
        self
    }

    /// Return whether this registration carries a runtime-owned execution
    /// bridge.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_runtime::{
    ///     CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
    ///     HardwareClassId, StorageClass,
    /// };
    ///
    /// let storage = StorageClass::new("tenferro.storage.host")?;
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("tenferro.cpu")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("tenferro.cpu.host")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::builder().build(),
    /// )?;
    /// assert!(!registration.has_execution_engine());
    /// # Ok(())
    /// # }
    /// ```
    pub fn has_execution_engine(&self) -> bool {
        self.execution_engine.is_some()
    }

    pub(super) fn candidate_identical(&self, other: &Self) -> bool {
        self.engine_id == other.engine_id
            && Arc::ptr_eq(&self.candidate_token, &other.candidate_token)
    }
}

impl fmt::Debug for EngineRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineRegistration")
            .field("engine_id", &self.engine_id)
            .field("context_identity", &self.context_identity)
            .field("hardware_class", &self.hardware_class)
            .field("storage_class_count", &self.storage_classes.len())
            .field("default_storage_class", &self.default_storage_class)
            .field("capabilities", &self.capabilities)
            .field("cache_owner", &self.cache_owner.is_some())
            .field("execution_engine", &self.has_execution_engine())
            .finish_non_exhaustive()
    }
}

fn validate_storage_classes(
    engine_id: &EngineId,
    storage_classes: &[StorageClass],
    default_storage_class: &StorageClass,
) -> Result<(), RuntimeConfigError> {
    if storage_classes.is_empty() {
        return Err(RuntimeConfigError::EmptyStorageClasses {
            engine_id: engine_id.clone(),
        });
    }
    for duplicate_index in 0..storage_classes.len() {
        if let Some(first_index) = (0..duplicate_index)
            .find(|&first| storage_classes[first] == storage_classes[duplicate_index])
        {
            return Err(RuntimeConfigError::DuplicateStorageClass {
                engine_id: engine_id.clone(),
                storage_class: storage_classes[duplicate_index].clone(),
                first_index,
                duplicate_index,
            });
        }
    }
    if !storage_classes
        .iter()
        .any(|storage_class| storage_class == default_storage_class)
    {
        return Err(RuntimeConfigError::DefaultStorageClassNotListed {
            engine_id: engine_id.clone(),
            default_storage_class: default_storage_class.clone(),
        });
    }
    Ok(())
}
