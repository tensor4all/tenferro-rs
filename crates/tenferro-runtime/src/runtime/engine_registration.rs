use std::fmt;
use std::sync::Arc;

use super::{
    execution, CoreCapabilityBundle, EngineId, EventDomainDriver, ExecutionContextIdentity,
    HardwareClassId, InputSignatureEntry, ProviderDeviceIdentity, RuntimeCacheOwner,
    RuntimeConfigError, StorageClass,
};
use tenferro_tensor::{AllocationDomainId, Placement, TensorBackend, TensorRead};

#[derive(Debug)]
pub(super) struct CandidateRegistrationToken;

type InputPlacementPredicate = dyn Fn(&Placement, &StorageClass) -> bool + Send + Sync + 'static;
type InputSignaturePredicate = dyn Fn(&Placement, Option<&'static str>, Option<AllocationDomainId>, &StorageClass) -> bool
    + Send
    + Sync
    + 'static;
type InputTensorPredicate =
    dyn for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static;

/// Named placement admission contract for one provider ingress.
#[derive(Clone)]
pub struct InputPlacementContract(Arc<InputPlacementPredicate>);

impl InputPlacementContract {
    /// Construct a named placement admission contract.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputPlacementContract, StorageClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let contract = InputPlacementContract::new(move |placement: &Placement, candidate| {
    ///     placement == &Placement::default() && candidate == &storage
    /// });
    /// let _ = contract;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        predicate: impl Fn(&Placement, &StorageClass) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(predicate))
    }

    fn accepts(&self, placement: &Placement, storage_class: &StorageClass) -> bool {
        (self.0)(placement, storage_class)
    }
}

impl fmt::Debug for InputPlacementContract {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("InputPlacementContract(..)")
    }
}

/// Named value-free physical input signature contract.
#[derive(Clone)]
pub struct InputSignatureContract(Arc<InputSignaturePredicate>);

impl InputSignatureContract {
    /// Construct a named value-free physical input signature contract.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignatureContract, StorageClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let contract = InputSignatureContract::new(move |
    ///     placement: &Placement, family, domain, candidate| {
    ///         placement == &Placement::default()
    ///             && family.is_none()
    ///             && domain.is_none()
    ///             && candidate == &storage
    ///     });
    /// let _ = contract;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        predicate: impl Fn(&Placement, Option<&'static str>, Option<AllocationDomainId>, &StorageClass) -> bool
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self(Arc::new(predicate))
    }

    fn accepts(&self, input: &InputSignatureEntry, storage_class: &StorageClass) -> bool {
        (self.0)(
            input.placement(),
            input.backend_family(),
            input.allocation_domain(),
            storage_class,
        )
    }
}

impl fmt::Debug for InputSignatureContract {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("InputSignatureContract(..)")
    }
}

/// Named runtime-input residency contract.
#[derive(Clone)]
pub struct RuntimeInputContract(Arc<InputTensorPredicate>);

impl RuntimeInputContract {
    /// Construct a named runtime-input residency contract.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::RuntimeInputContract;
    ///
    /// let contract = RuntimeInputContract::new(|_, _| true);
    /// let _ = contract;
    /// ```
    pub fn new(
        predicate: impl for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(predicate))
    }

    fn accepts(&self, input: &TensorRead<'_>, storage_class: &StorageClass) -> bool {
        (self.0)(input, storage_class)
    }
}

impl fmt::Debug for RuntimeInputContract {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RuntimeInputContract(..)")
    }
}

/// Named resident-output ownership contract.
#[derive(Clone)]
pub struct ResidentOutputContract(Arc<InputTensorPredicate>);

impl ResidentOutputContract {
    /// Construct a named resident-output ownership contract.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ResidentOutputContract;
    ///
    /// let contract = ResidentOutputContract::new(|_, _| true);
    /// let _ = contract;
    /// ```
    pub fn new(
        predicate: impl for<'a> Fn(&TensorRead<'a>, &StorageClass) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(predicate))
    }

    fn accepts(&self, input: &TensorRead<'_>, storage_class: &StorageClass) -> bool {
        (self.0)(input, storage_class)
    }
}

impl fmt::Debug for ResidentOutputContract {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ResidentOutputContract(..)")
    }
}

/// Complete named ingress contract for an executable provider binding.
#[derive(Clone, Debug)]
pub struct InputIngressContract {
    placement: InputPlacementContract,
    signature: InputSignatureContract,
    runtime_input: RuntimeInputContract,
    resident_output: ResidentOutputContract,
}

impl InputIngressContract {
    /// Assemble the four distinct ingress contracts atomically.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{
    ///     InputIngressContract, InputPlacementContract, InputSignatureContract,
    ///     ResidentOutputContract, RuntimeInputContract,
    /// };
    ///
    /// let ingress = InputIngressContract::new(
    ///     InputPlacementContract::new(|_, _| true),
    ///     InputSignatureContract::new(|_, _, _, _| true),
    ///     RuntimeInputContract::new(|_, _| true),
    ///     ResidentOutputContract::new(|_, _| true),
    /// );
    /// let _ = ingress;
    /// ```
    pub fn new(
        placement: InputPlacementContract,
        signature: InputSignatureContract,
        runtime_input: RuntimeInputContract,
        resident_output: ResidentOutputContract,
    ) -> Self {
        Self {
            placement,
            signature,
            runtime_input,
            resident_output,
        }
    }

    fn accepts_placement(&self, placement: &Placement, storage_class: &StorageClass) -> bool {
        self.placement.accepts(placement, storage_class)
    }

    fn accepts_signature(&self, input: &InputSignatureEntry, storage_class: &StorageClass) -> bool {
        self.signature.accepts(input, storage_class)
    }

    fn accepts_runtime_input(&self, input: &TensorRead<'_>, storage_class: &StorageClass) -> bool {
        self.runtime_input.accepts(input, storage_class)
    }

    fn owns_resident_output(&self, input: &TensorRead<'_>, storage_class: &StorageClass) -> bool {
        self.resident_output.accepts(input, storage_class)
    }
}

/// The complete execution witness stored by an executable provider binding.
#[derive(Clone)]
pub struct ExecutableEngineContract {
    provider_device_identity: ProviderDeviceIdentity,
    context_identity: ExecutionContextIdentity,
    capabilities: CoreCapabilityBundle,
    executor: Arc<dyn execution::ErasedTensorBackendExecutor>,
    event_domain_driver: Arc<dyn EventDomainDriver>,
    ingress: InputIngressContract,
    cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
}

impl ExecutableEngineContract {
    /// Assemble an executable provider witness atomically.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     CoreCapabilityBundle, ExecutableEngineContract, ImmediateEventDomainDriver,
    ///     InputIngressContract, InputPlacementContract, InputSignatureContract,
    ///     ProviderDeviceIdentity, ProviderId, ResidentOutputContract,
    ///     RuntimeInputContract,
    /// };
    ///
    /// let ingress = InputIngressContract::new(
    ///     InputPlacementContract::new(|_, _| true),
    ///     InputSignatureContract::new(|_, _, _, _| true),
    ///     RuntimeInputContract::new(|_, _| true),
    ///     ResidentOutputContract::new(|_, _| true),
    /// );
    /// let contract = ExecutableEngineContract::new(
    ///     ProviderDeviceIdentity::new(ProviderId::new("example.provider")?, "device:0")?,
    ///     CoreCapabilityBundle::default(),
    ///     CpuBackend::new(),
    ///     Arc::new(ImmediateEventDomainDriver::new()),
    ///     ingress,
    ///     None,
    /// );
    /// let _ = contract;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new<B>(
        provider_device_identity: ProviderDeviceIdentity,
        capabilities: CoreCapabilityBundle,
        backend: B,
        event_domain_driver: Arc<dyn EventDomainDriver>,
        ingress: InputIngressContract,
        cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    ) -> Self
    where
        B: TensorBackend + Send + Sync + 'static,
    {
        Self {
            provider_device_identity,
            context_identity: ExecutionContextIdentity::of::<B>(),
            capabilities,
            executor: execution::erased_tensor_backend_executor(backend),
            event_domain_driver,
            ingress,
            cache_owner,
        }
    }

    #[cfg(test)]
    pub(super) fn from_erased_for_test(
        provider_device_identity: ProviderDeviceIdentity,
        context_identity: ExecutionContextIdentity,
        capabilities: CoreCapabilityBundle,
        executor: Arc<dyn execution::ErasedTensorBackendExecutor>,
        event_domain_driver: Arc<dyn EventDomainDriver>,
        ingress: InputIngressContract,
        cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    ) -> Self {
        Self {
            provider_device_identity,
            context_identity,
            capabilities,
            executor,
            event_domain_driver,
            ingress,
            cache_owner,
        }
    }

    pub(super) fn capabilities(&self) -> &CoreCapabilityBundle {
        &self.capabilities
    }

    pub(super) fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        &self.provider_device_identity
    }

    pub(super) fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    pub(super) fn executor(&self) -> &Arc<dyn execution::ErasedTensorBackendExecutor> {
        &self.executor
    }

    pub(super) fn event_domain_driver(&self) -> &Arc<dyn EventDomainDriver> {
        &self.event_domain_driver
    }

    pub(super) fn cache_owner(&self) -> Option<&Arc<dyn RuntimeCacheOwner>> {
        self.cache_owner.as_ref()
    }

    pub(super) fn accepts_input_placement(
        &self,
        placement: &Placement,
        storage_class: &StorageClass,
    ) -> bool {
        self.ingress.accepts_placement(placement, storage_class)
    }

    pub(super) fn accepts_input_signature(
        &self,
        input: &InputSignatureEntry,
        storage_class: &StorageClass,
    ) -> bool {
        self.ingress.accepts_signature(input, storage_class)
    }

    pub(super) fn accepts_runtime_input(
        &self,
        input: &TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.ingress.accepts_runtime_input(input, storage_class)
    }

    pub(super) fn owns_resident_tensor(
        &self,
        input: &TensorRead<'_>,
        storage_class: &StorageClass,
    ) -> bool {
        self.ingress.owns_resident_output(input, storage_class)
    }
}

impl fmt::Debug for ExecutableEngineContract {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutableEngineContract")
            .field("provider_device_identity", &self.provider_device_identity)
            .field("context_identity", &self.context_identity)
            .field("capabilities", &self.capabilities)
            .field("executor", &self.executor.backend_type_name())
            .field("event_domain_driver", &self.event_domain_driver)
            .field("ingress", &self.ingress)
            .field("cache_owner", &self.cache_owner.is_some())
            .finish_non_exhaustive()
    }
}

/// Provider-owned metadata plus one complete executable witness.
pub struct ProviderExecutableBinding {
    engine_id: EngineId,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    contract: ExecutableEngineContract,
}

impl ProviderExecutableBinding {
    /// Construct a provider-owned executable binding.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeConfigError::EmptyStorageClasses`],
    /// [`RuntimeConfigError::DuplicateStorageClass`], or
    /// [`RuntimeConfigError::DefaultStorageClassNotListed`] when the provider
    /// metadata is inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{
    ///     CoreCapabilityBundle, EngineId, ExecutableEngineContract,
    ///     HardwareClassId, ImmediateEventDomainDriver,
    ///     InputIngressContract, InputPlacementContract, InputSignatureContract,
    ///     ProviderDeviceIdentity, ProviderExecutableBinding, ProviderId,
    ///     ResidentOutputContract, RuntimeInputContract, StorageClass,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let ingress = InputIngressContract::new(
    ///     InputPlacementContract::new(|_, _| true),
    ///     InputSignatureContract::new(|_, _, _, _| true),
    ///     RuntimeInputContract::new(|_, _| true),
    ///     ResidentOutputContract::new(|_, _| true),
    /// );
    /// let contract = ExecutableEngineContract::new(
    ///     ProviderDeviceIdentity::new(ProviderId::new("example.provider")?, "device:0")?,
    ///     CoreCapabilityBundle::default(),
    ///     CpuBackend::new(),
    ///     Arc::new(ImmediateEventDomainDriver::new()),
    ///     ingress,
    ///     None,
    /// );
    /// let binding = ProviderExecutableBinding::new(
    ///     EngineId::new("example.engine")?,
    ///     HardwareClassId::new("example.hardware")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     contract,
    /// )?;
    /// let _ = binding;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        engine_id: EngineId,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        contract: ExecutableEngineContract,
    ) -> Result<Self, RuntimeConfigError> {
        validate_storage_classes(&engine_id, &storage_classes, &default_storage_class)?;
        Ok(Self {
            engine_id,
            hardware_class,
            storage_classes,
            default_storage_class,
            contract,
        })
    }
}

impl fmt::Debug for ProviderExecutableBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderExecutableBinding")
            .field("engine_id", &self.engine_id)
            .field("hardware_class", &self.hardware_class)
            .field("storage_class_count", &self.storage_classes.len())
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

impl Clone for ProviderExecutableBinding {
    fn clone(&self) -> Self {
        Self {
            engine_id: self.engine_id.clone(),
            hardware_class: self.hardware_class.clone(),
            storage_classes: Arc::clone(&self.storage_classes),
            default_storage_class: self.default_storage_class.clone(),
            contract: self.contract.clone(),
        }
    }
}

/// Provider-owned metadata plus preparation capabilities without execution.
pub struct ProviderPreparationBinding {
    engine_id: EngineId,
    provider_device_identity: ProviderDeviceIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    capabilities: CoreCapabilityBundle,
}

impl ProviderPreparationBinding {
    /// Construct a provider-owned preparation-only binding.
    ///
    /// # Errors
    ///
    /// Returns a [`RuntimeConfigError`] when the storage metadata is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::sync::Arc;
    /// use tenferro_runtime::{
    ///     CoreCapabilityBundle, EngineId, ExecutionContextIdentity, HardwareClassId,
    ///     ProviderDeviceIdentity, ProviderId, ProviderPreparationBinding, StorageClass,
    /// };
    ///
    /// let storage = StorageClass::new("example.storage.host")?;
    /// let binding = ProviderPreparationBinding::new(
    ///     EngineId::new("example.engine")?,
    ///     ProviderDeviceIdentity::new(ProviderId::new("example.provider")?, "device:0")?,
    ///     ExecutionContextIdentity::of::<()>(),
    ///     HardwareClassId::new("example.hardware")?,
    ///     Arc::from(vec![storage.clone()]),
    ///     storage,
    ///     CoreCapabilityBundle::default(),
    /// )?;
    /// let _ = binding;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        engine_id: EngineId,
        provider_device_identity: ProviderDeviceIdentity,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        capabilities: CoreCapabilityBundle,
    ) -> Result<Self, RuntimeConfigError> {
        validate_storage_classes(&engine_id, &storage_classes, &default_storage_class)?;
        Ok(Self {
            engine_id,
            provider_device_identity,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            capabilities,
        })
    }
}

impl Clone for ProviderPreparationBinding {
    fn clone(&self) -> Self {
        Self {
            engine_id: self.engine_id.clone(),
            provider_device_identity: self.provider_device_identity.clone(),
            context_identity: self.context_identity,
            hardware_class: self.hardware_class.clone(),
            storage_classes: Arc::clone(&self.storage_classes),
            default_storage_class: self.default_storage_class.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
}

impl fmt::Debug for ProviderPreparationBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderPreparationBinding")
            .field("engine_id", &self.engine_id)
            .field("provider_device_identity", &self.provider_device_identity)
            .field("context_identity", &self.context_identity)
            .field("hardware_class", &self.hardware_class)
            .field("storage_class_count", &self.storage_classes.len())
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// Mutually exclusive preparation-only or executable registration state.
#[derive(Clone, Debug)]
pub(crate) enum EngineRegistrationState {
    /// Capabilities are available to planning, but this registration cannot
    /// admit runtime inputs or execute a schedule.
    PreparationOnly { capabilities: CoreCapabilityBundle },
    /// A complete provider execution witness.
    Executable(ExecutableEngineContract),
}

impl EngineRegistrationState {
    pub(super) fn capabilities(&self) -> &CoreCapabilityBundle {
        match self {
            Self::PreparationOnly { capabilities } => capabilities,
            Self::Executable(contract) => contract.capabilities(),
        }
    }
}

/// Immutable direct engine registration candidate.
#[derive(Clone)]
pub struct EngineRegistration {
    engine_id: EngineId,
    provider_device_identity: ProviderDeviceIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    state: EngineRegistrationState,
    candidate_token: Arc<CandidateRegistrationToken>,
}

impl EngineRegistration {
    /// Consume one provider-owned preparation binding.
    pub fn preparation_only(binding: ProviderPreparationBinding) -> Self {
        let ProviderPreparationBinding {
            engine_id,
            provider_device_identity,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            capabilities,
        } = binding;
        Self {
            engine_id,
            provider_device_identity,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            state: EngineRegistrationState::PreparationOnly { capabilities },
            candidate_token: Arc::new(CandidateRegistrationToken),
        }
    }

    /// Consume one provider-owned complete executable binding.
    pub fn executable(binding: ProviderExecutableBinding) -> Self {
        let ProviderExecutableBinding {
            engine_id,
            hardware_class,
            storage_classes,
            default_storage_class,
            contract,
        } = binding;
        let provider_device_identity = contract.provider_device_identity.clone();
        let context_identity = contract.context_identity;
        Self {
            engine_id,
            provider_device_identity,
            context_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            state: EngineRegistrationState::Executable(contract),
            candidate_token: Arc::new(CandidateRegistrationToken),
        }
    }

    /// Return the immutable registration state witness.
    pub(crate) fn execution_state(&self) -> &EngineRegistrationState {
        &self.state
    }

    /// Return the engine identifier.
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// Return the immutable provider/device binding for this engine.
    pub fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        &self.provider_device_identity
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

    pub(super) fn storage_classes_arc(&self) -> Arc<[StorageClass]> {
        Arc::clone(&self.storage_classes)
    }

    pub(super) fn candidate_token(&self) -> Arc<CandidateRegistrationToken> {
        Arc::clone(&self.candidate_token)
    }

    pub(super) fn with_candidate_token(
        mut self,
        candidate_token: Arc<CandidateRegistrationToken>,
    ) -> Self {
        self.candidate_token = candidate_token;
        self
    }

    /// Return the default storage class.
    pub fn default_storage_class(&self) -> &StorageClass {
        &self.default_storage_class
    }

    /// Return direct core capability slots.
    pub fn capabilities(&self) -> &CoreCapabilityBundle {
        self.state.capabilities()
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
            .field("provider_device_identity", &self.provider_device_identity)
            .field("context_identity", &self.context_identity)
            .field("hardware_class", &self.hardware_class)
            .field("storage_class_count", &self.storage_classes.len())
            .field("default_storage_class", &self.default_storage_class)
            .field("state", &self.state)
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
