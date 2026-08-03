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

/// Shared provider metadata for one runtime engine registration.
///
/// The caller supplies the engine and provider/device identities explicitly.
/// Storage metadata is validated when the metadata is assembled into either an
/// executable or preparation-only registration.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use tenferro_runtime::{
///     CoreCapabilityBundle, EngineId, EngineRegistrationMetadata, HardwareClassId,
///     ProviderDeviceIdentity, ProviderId, StorageClass,
/// };
///
/// let engine_id = EngineId::new("example.engine.v1")?;
/// let provider = ProviderDeviceIdentity::new(
///     ProviderId::new("example.provider")?,
///     "device:0",
/// )?;
/// let hardware = HardwareClassId::new("example.hardware.v1")?;
/// let storage = StorageClass::new("example.storage.v1")?;
/// let metadata = EngineRegistrationMetadata::new(
///     engine_id,
///     provider,
///     hardware,
///     Arc::from([storage.clone()]),
///     storage,
///     CoreCapabilityBundle::default(),
/// );
/// let _ = metadata;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct EngineRegistrationMetadata {
    engine_id: EngineId,
    provider_device_identity: ProviderDeviceIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    capabilities: CoreCapabilityBundle,
}

impl EngineRegistrationMetadata {
    /// Construct the metadata shared by executable and preparation-only
    /// registration descriptors.
    pub fn new(
        engine_id: EngineId,
        provider_device_identity: ProviderDeviceIdentity,
        hardware_class: HardwareClassId,
        storage_classes: Arc<[StorageClass]>,
        default_storage_class: StorageClass,
        capabilities: CoreCapabilityBundle,
    ) -> Self {
        Self {
            engine_id,
            provider_device_identity,
            hardware_class,
            storage_classes,
            default_storage_class,
            capabilities,
        }
    }
}

/// Typed configuration for assembling an executable engine registration.
///
/// The backend, event domain, ingress contract, and optional cache owner are
/// kept together with the shared provider metadata until the runtime creates
/// the complete execution witness.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use tenferro_runtime::{
///     CoreCapabilityBundle, EngineId, EngineRegistrationMetadata,
///     ExecutableEngineRegistrationConfig, EventDomainDriver, HardwareClassId,
///     ImmediateEventDomainDriver, InputIngressContract, InputPlacementContract,
///     InputSignatureContract, ProviderDeviceIdentity, ProviderId, ResidentOutputContract,
///     RuntimeInputContract, StorageClass,
/// };
///
/// fn make_config<B>(backend: B) -> ExecutableEngineRegistrationConfig<B> {
///     let engine_id = EngineId::new("example.engine.v1").unwrap();
///     let provider = ProviderDeviceIdentity::new(
///         ProviderId::new("example.provider").unwrap(),
///         "device:0",
///     )
///     .unwrap();
///     let hardware = HardwareClassId::new("example.hardware.v1").unwrap();
///     let storage = StorageClass::new("example.storage.v1").unwrap();
///     let metadata = EngineRegistrationMetadata::new(
///         engine_id,
///         provider,
///         hardware,
///         Arc::from([storage.clone()]),
///         storage,
///         CoreCapabilityBundle::default(),
///     );
///     let ingress = InputIngressContract::new(
///         InputPlacementContract::new(|_, _| true),
///         InputSignatureContract::new(|_, _, _, _| true),
///         RuntimeInputContract::new(|_, _| true),
///         ResidentOutputContract::new(|_, _| true),
///     );
///     let event_domain_driver: Arc<dyn EventDomainDriver> =
///         Arc::new(ImmediateEventDomainDriver::new());
///     ExecutableEngineRegistrationConfig::new(
///         metadata,
///         backend,
///         event_domain_driver,
///         ingress,
///         None,
///     )
/// }
///
/// let _ = make_config(());
/// ```
pub struct ExecutableEngineRegistrationConfig<B> {
    metadata: EngineRegistrationMetadata,
    backend: B,
    event_domain_driver: Arc<dyn EventDomainDriver>,
    ingress: InputIngressContract,
    cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
}

impl<B> ExecutableEngineRegistrationConfig<B> {
    /// Construct an executable registration descriptor.
    pub fn new(
        metadata: EngineRegistrationMetadata,
        backend: B,
        event_domain_driver: Arc<dyn EventDomainDriver>,
        ingress: InputIngressContract,
        cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    ) -> Self {
        Self {
            metadata,
            backend,
            event_domain_driver,
            ingress,
            cache_owner,
        }
    }
}

/// Typed configuration for assembling a preparation-only engine registration.
///
/// Preparation-only registrations retain the provider's execution-context
/// identity but intentionally contain no executable backend or event driver.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use tenferro_runtime::{
///     CoreCapabilityBundle, EngineId, EngineRegistrationMetadata,
///     ExecutionContextIdentity, HardwareClassId, PreparationOnlyEngineRegistrationConfig,
///     ProviderDeviceIdentity, ProviderId, StorageClass,
/// };
///
/// let engine_id = EngineId::new("example.engine.v1")?;
/// let provider = ProviderDeviceIdentity::new(
///     ProviderId::new("example.provider")?,
///     "device:0",
/// )?;
/// let hardware = HardwareClassId::new("example.hardware.v1")?;
/// let storage = StorageClass::new("example.storage.v1")?;
/// let metadata = EngineRegistrationMetadata::new(
///     engine_id,
///     provider,
///     hardware,
///     Arc::from([storage.clone()]),
///     storage,
///     CoreCapabilityBundle::default(),
/// );
/// let config = PreparationOnlyEngineRegistrationConfig::new(
///     metadata,
///     ExecutionContextIdentity::of::<()>(),
/// );
/// let _ = config;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PreparationOnlyEngineRegistrationConfig {
    metadata: EngineRegistrationMetadata,
    context_identity: ExecutionContextIdentity,
}

impl PreparationOnlyEngineRegistrationConfig {
    /// Construct a preparation-only registration descriptor.
    pub fn new(
        metadata: EngineRegistrationMetadata,
        context_identity: ExecutionContextIdentity,
    ) -> Self {
        Self {
            metadata,
            context_identity,
        }
    }
}

/// The complete execution witness stored by an executable provider binding.
#[derive(Clone)]
pub(crate) struct ExecutableEngineContract {
    provider_device_identity: ProviderDeviceIdentity,
    context_identity: ExecutionContextIdentity,
    capabilities: CoreCapabilityBundle,
    executor: Arc<dyn execution::ErasedTensorBackendExecutor>,
    event_domain_driver: Arc<dyn EventDomainDriver>,
    ingress: InputIngressContract,
    cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
}

impl ExecutableEngineContract {
    // This constructor is intentionally limited to the runtime assembly
    // boundary. Providers use assemble_executable_engine_registration instead
    // of manufacturing a partial contract and binding independently.
    pub(super) fn new<B>(
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
pub(crate) struct ProviderExecutableBinding {
    engine_id: EngineId,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    contract: ExecutableEngineContract,
}

impl ProviderExecutableBinding {
    // Storage metadata is validated as part of the runtime-owned executable
    // assembly; this partial constructor is not a public provider API.
    pub(super) fn new(
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

    pub(super) fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    pub(super) fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }

    pub(super) fn storage_classes(&self) -> &[StorageClass] {
        &self.storage_classes
    }

    pub(super) fn default_storage_class(&self) -> &StorageClass {
        &self.default_storage_class
    }

    pub(super) fn contract(&self) -> &ExecutableEngineContract {
        &self.contract
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
pub(crate) struct ProviderPreparationBinding {
    engine_id: EngineId,
    provider_device_identity: ProviderDeviceIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    storage_classes: Arc<[StorageClass]>,
    default_storage_class: StorageClass,
    capabilities: CoreCapabilityBundle,
}

impl ProviderPreparationBinding {
    // Preparation-only metadata follows the same runtime-owned assembly
    // boundary and cannot be promoted to an executable binding.
    pub(super) fn new(
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

    pub(super) fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    pub(super) fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        &self.provider_device_identity
    }

    pub(super) fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    pub(super) fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }

    pub(super) fn storage_classes(&self) -> &[StorageClass] {
        &self.storage_classes
    }

    pub(super) fn default_storage_class(&self) -> &StorageClass {
        &self.default_storage_class
    }

    pub(super) fn capabilities(&self) -> &CoreCapabilityBundle {
        &self.capabilities
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
    PreparationOnly { binding: ProviderPreparationBinding },
    /// A complete provider execution witness.
    Executable(ProviderExecutableBinding),
}

impl EngineRegistrationState {
    pub(super) fn capabilities(&self) -> &CoreCapabilityBundle {
        match self {
            Self::PreparationOnly { binding } => binding.capabilities(),
            Self::Executable(binding) => binding.contract().capabilities(),
        }
    }

    pub(super) fn engine_id(&self) -> &EngineId {
        match self {
            Self::PreparationOnly { binding } => binding.engine_id(),
            Self::Executable(binding) => binding.engine_id(),
        }
    }

    pub(super) fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        match self {
            Self::PreparationOnly { binding } => binding.provider_device_identity(),
            Self::Executable(binding) => binding.contract().provider_device_identity(),
        }
    }

    pub(super) fn context_identity(&self) -> ExecutionContextIdentity {
        match self {
            Self::PreparationOnly { binding } => binding.context_identity(),
            Self::Executable(binding) => binding.contract().context_identity(),
        }
    }

    pub(super) fn hardware_class(&self) -> &HardwareClassId {
        match self {
            Self::PreparationOnly { binding } => binding.hardware_class(),
            Self::Executable(binding) => binding.hardware_class(),
        }
    }

    pub(super) fn storage_classes(&self) -> &[StorageClass] {
        match self {
            Self::PreparationOnly { binding } => binding.storage_classes(),
            Self::Executable(binding) => binding.storage_classes(),
        }
    }

    pub(super) fn default_storage_class(&self) -> &StorageClass {
        match self {
            Self::PreparationOnly { binding } => binding.default_storage_class(),
            Self::Executable(binding) => binding.default_storage_class(),
        }
    }
}

/// Immutable direct engine registration candidate.
#[derive(Clone)]
pub struct EngineRegistration {
    state: EngineRegistrationState,
    candidate_token: Arc<CandidateRegistrationToken>,
}

impl EngineRegistration {
    /// Consume one provider-owned preparation binding.
    pub(super) fn preparation_only(binding: ProviderPreparationBinding) -> Self {
        Self {
            state: EngineRegistrationState::PreparationOnly { binding },
            candidate_token: Arc::new(CandidateRegistrationToken),
        }
    }

    /// Consume one provider-owned complete executable binding.
    pub(super) fn executable(binding: ProviderExecutableBinding) -> Self {
        Self {
            state: EngineRegistrationState::Executable(binding),
            candidate_token: Arc::new(CandidateRegistrationToken),
        }
    }

    pub(super) fn from_state(state: EngineRegistrationState) -> Self {
        Self {
            state,
            candidate_token: Arc::new(CandidateRegistrationToken),
        }
    }

    /// Return the immutable registration state witness.
    #[cfg(test)]
    pub(crate) fn execution_state(&self) -> &EngineRegistrationState {
        &self.state
    }

    /// Return the engine identifier.
    pub fn engine_id(&self) -> &EngineId {
        self.state.engine_id()
    }

    /// Return the immutable provider/device binding for this engine.
    pub fn provider_device_identity(&self) -> &ProviderDeviceIdentity {
        self.state.provider_device_identity()
    }

    /// Return the execution-context type identity accepted by the engine.
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.state.context_identity()
    }

    /// Return the hardware class exposed by this engine.
    pub fn hardware_class(&self) -> &HardwareClassId {
        self.state.hardware_class()
    }

    /// Return the supported storage classes in registration order.
    pub fn storage_classes(&self) -> &[StorageClass] {
        self.state.storage_classes()
    }

    pub(super) fn with_candidate_token(
        mut self,
        candidate_token: Arc<CandidateRegistrationToken>,
    ) -> Self {
        self.candidate_token = candidate_token;
        self
    }

    pub(super) fn into_state_and_token(
        self,
    ) -> (EngineRegistrationState, Arc<CandidateRegistrationToken>) {
        (self.state, self.candidate_token)
    }

    /// Return the default storage class.
    pub fn default_storage_class(&self) -> &StorageClass {
        self.state.default_storage_class()
    }

    /// Return direct core capability slots.
    pub fn capabilities(&self) -> &CoreCapabilityBundle {
        self.state.capabilities()
    }

    pub(super) fn candidate_identical(&self, other: &Self) -> bool {
        self.engine_id() == other.engine_id()
            && Arc::ptr_eq(&self.candidate_token, &other.candidate_token)
    }
}

/// Assemble one complete executable provider registration.
///
/// Provider adapters supply only provider-owned identity, capability, ingress,
/// event-driver, and cache-owner values.  The runtime owns the assembly of the
/// executable witness and its storage metadata so every executable engine
/// enters the runtime through the same invariant-preserving path.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the descriptor contains empty or duplicate
/// storage classes, or if its default storage class is not listed in the
/// supported storage classes.
pub fn assemble_executable_engine_registration<B>(
    config: ExecutableEngineRegistrationConfig<B>,
) -> Result<EngineRegistration, RuntimeConfigError>
where
    B: TensorBackend + Send + Sync + 'static,
{
    let ExecutableEngineRegistrationConfig {
        metadata:
            EngineRegistrationMetadata {
                engine_id,
                provider_device_identity,
                hardware_class,
                storage_classes,
                default_storage_class,
                capabilities,
            },
        backend,
        event_domain_driver,
        ingress,
        cache_owner,
    } = config;
    let contract = ExecutableEngineContract::new(
        provider_device_identity,
        capabilities,
        backend,
        event_domain_driver,
        ingress,
        cache_owner,
    );
    let binding = ProviderExecutableBinding::new(
        engine_id,
        hardware_class,
        storage_classes,
        default_storage_class,
        contract,
    )?;
    Ok(EngineRegistration::executable(binding))
}

/// Assemble one preparation-only provider registration.
///
/// This is the preparation counterpart of
/// [`assemble_executable_engine_registration`].  It deliberately cannot
/// manufacture an execution bridge or a scheduled witness.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the descriptor contains empty or duplicate
/// storage classes, or if its default storage class is not listed in the
/// supported storage classes.
pub fn assemble_preparation_only_engine_registration(
    config: PreparationOnlyEngineRegistrationConfig,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let PreparationOnlyEngineRegistrationConfig {
        metadata:
            EngineRegistrationMetadata {
                engine_id,
                provider_device_identity,
                hardware_class,
                storage_classes,
                default_storage_class,
                capabilities,
            },
        context_identity,
    } = config;
    let binding = ProviderPreparationBinding::new(
        engine_id,
        provider_device_identity,
        context_identity,
        hardware_class,
        storage_classes,
        default_storage_class,
        capabilities,
    )?;
    Ok(EngineRegistration::preparation_only(binding))
}

impl fmt::Debug for EngineRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineRegistration")
            .field("engine_id", self.engine_id())
            .field("provider_device_identity", self.provider_device_identity())
            .field("context_identity", &self.context_identity())
            .field("hardware_class", self.hardware_class())
            .field("storage_class_count", &self.storage_classes().len())
            .field("default_storage_class", self.default_storage_class())
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
