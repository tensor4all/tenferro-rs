use std::fmt;
use std::sync::Arc;

use super::{
    execution, CoreCapabilityBundle, EngineId, EventDomainDriver, ExecutionContextIdentity,
    HardwareClassId, InputSignatureEntry, RuntimeCacheOwner, RuntimeConfigError, StorageClass,
};
use tenferro_tensor::{AllocationDomainId, Placement, TensorBackend, TensorRead};

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
    event_domain_driver: Option<Arc<dyn EventDomainDriver>>,
    pub(super) cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    pub(super) execution_engine: Option<Arc<dyn execution::ErasedTensorBackendExecutor>>,
    input_placement_validator: Option<Arc<InputPlacementValidator>>,
    input_signature_validator: Option<Arc<InputSignatureValidator>>,
    runtime_input_validator: Option<Arc<InputTensorValidator>>,
    resident_tensor_validator: Option<Arc<InputTensorValidator>>,
    candidate_token: Arc<CandidateRegistrationToken>,
}

type InputPlacementValidator = dyn Fn(&Placement, &StorageClass) -> bool + Send + Sync + 'static;
type InputSignatureValidator = dyn Fn(&Placement, Option<&'static str>, Option<AllocationDomainId>, &StorageClass) -> bool
    + Send
    + Sync
    + 'static;
type InputTensorValidator =
    dyn for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static;

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
            event_domain_driver: None,
            cache_owner: None,
            execution_engine: None,
            input_placement_validator: None,
            input_signature_validator: None,
            runtime_input_validator: None,
            resident_tensor_validator: None,
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

    /// Attach the driver that owns this engine's per-run event-domain state.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_runtime::runtime::ImmediateEventDomainDriver;
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
    ///     Arc::from([storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::default(),
    /// )?
    /// .with_event_domain_driver(Arc::new(ImmediateEventDomainDriver::new()));
    /// assert_eq!(registration.engine_id().as_str(), "tenferro.cpu");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_event_domain_driver(mut self, driver: Arc<dyn EventDomainDriver>) -> Self {
        self.event_domain_driver = Some(driver);
        self
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

    /// Declare which tensor placements may enter this engine at each storage class.
    ///
    /// This placement-only hook supports preparation-only registrations.
    /// Registrations with an execution bridge must instead use
    /// [`Self::with_input_ingress_validator`] to declare the complete ingress
    /// contract.
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
    /// use tenferro_tensor::MemoryKind;
    ///
    /// let storage = StorageClass::new("example.host")?;
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("example.cpu")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.cpu")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage.clone(),
    ///     CoreCapabilityBundle::builder().build(),
    /// )?
    /// .with_input_placement_validator(move |placement, candidate| {
    ///     placement.memory_kind == MemoryKind::UnpinnedHost && candidate == &storage
    /// });
    /// assert!(registration.has_input_placement_validator());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_input_placement_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&Placement, &StorageClass) -> bool + Send + Sync + 'static,
    {
        self.input_placement_validator = Some(Arc::new(validator));
        self
    }

    /// Declare which physical input signatures may enter this engine.
    ///
    /// The runtime uses this value-free validator while selecting a prepared
    /// input ingress and keeps the concrete tensor validator as the execution
    /// boundary check.
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
    /// let storage = StorageClass::new("example.host")?;
    /// let expected = storage.clone();
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("example.cpu")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.cpu")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::default(),
    /// )?
    /// .with_input_signature_validator(move |_, family, domain, candidate| {
    ///     candidate == &expected && family.is_none() && domain.is_none()
    /// });
    /// assert_eq!(registration.engine_id().as_str(), "example.cpu");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_input_signature_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&Placement, Option<&'static str>, Option<AllocationDomainId>, &StorageClass) -> bool
            + Send
            + Sync
            + 'static,
    {
        self.input_signature_validator = Some(Arc::new(validator));
        self
    }

    /// Declare placement, runtime-input, and destination-residency ingress rules.
    ///
    /// Execution-bridge registrations must also call
    /// [`Self::with_input_signature_validator`] so preparation can select
    /// ingress from value-free physical input metadata.
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
    /// use tenferro_tensor::{MemoryKind, TensorRead};
    ///
    /// let storage = StorageClass::new("example.host")?;
    /// let candidate = storage.clone();
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("example.cpu")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.cpu")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::default(),
    /// )?
    /// .with_input_signature_validator(|_, family, domain, _| {
    ///     family.is_none() && domain.is_none()
    /// })
    /// .with_input_ingress_validator(
    ///     move |placement, storage| {
    ///         placement.memory_kind == MemoryKind::UnpinnedHost && storage == &candidate
    ///     },
    ///     |input: &TensorRead<'_>, _| input.backend_family().is_none(),
    ///     |input: &TensorRead<'_>, _| input.backend_family().is_none(),
    /// );
    /// assert!(registration.has_input_placement_validator());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_input_ingress_validator<P, I, R>(
        mut self,
        placement_validator: P,
        runtime_input_validator: I,
        resident_tensor_validator: R,
    ) -> Self
    where
        P: Fn(&Placement, &StorageClass) -> bool + Send + Sync + 'static,
        I: for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static,
        R: for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static,
    {
        self.input_placement_validator = Some(Arc::new(placement_validator));
        self.runtime_input_validator = Some(Arc::new(runtime_input_validator));
        self.resident_tensor_validator = Some(Arc::new(resident_tensor_validator));
        self
    }

    /// Return whether this registration declares an input-placement validator.
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
    /// let storage = StorageClass::new("example.host")?;
    /// let registration = EngineRegistration::new(
    ///     EngineId::new("example.cpu")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.cpu")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::builder().build(),
    /// )?;
    /// assert!(!registration.has_input_placement_validator());
    /// # Ok(())
    /// # }
    /// ```
    pub fn has_input_placement_validator(&self) -> bool {
        self.input_placement_validator.is_some()
    }

    pub(super) fn has_input_ingress_validator(&self) -> bool {
        self.input_placement_validator.is_some()
            && self.input_signature_validator.is_some()
            && self.runtime_input_validator.is_some()
            && self.resident_tensor_validator.is_some()
    }

    pub(super) fn accepts_input_signature(
        &self,
        input: &InputSignatureEntry,
        storage_class: &StorageClass,
    ) -> bool {
        self.storage_classes.contains(storage_class)
            && self
                .input_signature_validator
                .as_ref()
                .is_some_and(|validator| {
                    validator(
                        input.placement(),
                        input.backend_family(),
                        input.allocation_domain(),
                        storage_class,
                    )
                })
    }

    pub(super) fn accepts_input_placement(
        &self,
        placement: &Placement,
        storage_class: &StorageClass,
    ) -> bool {
        self.storage_classes.contains(storage_class)
            && self
                .input_placement_validator
                .as_ref()
                .is_some_and(|validator| validator(placement, storage_class))
    }

    pub(super) fn accepts_runtime_input(
        &self,
        input: &TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.storage_classes.contains(storage_class)
            && self
                .runtime_input_validator
                .as_ref()
                .is_some_and(|validator| validator(input, storage_class))
    }

    pub(super) fn owns_resident_tensor(
        &self,
        input: &TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.storage_classes.contains(storage_class)
            && self
                .resident_tensor_validator
                .as_ref()
                .is_some_and(|validator| validator(input, storage_class))
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

    #[cfg(test)]
    pub(crate) fn event_domain_driver(&self) -> Option<&Arc<dyn EventDomainDriver>> {
        self.event_domain_driver.as_ref()
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
            .field("event_domain_driver", &self.event_domain_driver.is_some())
            .field("cache_owner", &self.cache_owner.is_some())
            .field("execution_engine", &self.has_execution_engine())
            .field(
                "input_placement_validator",
                &self.has_input_placement_validator(),
            )
            .field(
                "input_ingress_validator",
                &self.has_input_ingress_validator(),
            )
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
