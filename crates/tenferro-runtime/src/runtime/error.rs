use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::capability::{
    CoreCapabilityKind, PreparationKeySummary, PreparedOperationBinding, UnsupportedReason,
};
use super::identity::{EngineId, RuntimeEpoch, RuntimeId};
use super::policy::{ProgramPlacementConstraint, StorageClass};
use super::specialization::SpecializationProjection;
use super::{CacheOwnerId, ExtensionModuleId};
use tenferro_ops::ShapeGuardError;
use tenferro_tensor::Error as TensorError;

/// Classifies a malformed runtime identifier without retaining its input.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::EngineId;
///
/// let error = EngineId::new("not namespaced").unwrap_err();
/// assert_eq!(error.kind(), tenferro_runtime::IdentityKind::Engine);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("malformed {kind:?} identifier")]
pub struct IdentityError {
    kind: IdentityKind,
}

impl IdentityError {
    /// Return the kind of identifier that failed validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, IdentityKind};
    ///
    /// assert_eq!(EngineId::new("engine").unwrap_err().kind(), IdentityKind::Engine);
    /// ```
    pub fn kind(&self) -> IdentityKind {
        self.kind
    }

    pub(super) const fn malformed(kind: IdentityKind) -> Self {
        Self { kind }
    }
}

/// Identifies the runtime namespace validated by an opaque identifier.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::IdentityKind;
///
/// assert_eq!(IdentityKind::CacheOwner, IdentityKind::CacheOwner);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum IdentityKind {
    /// An execution engine identifier.
    Engine,
    /// A hardware-class identifier.
    HardwareClass,
    /// A storage-class identifier.
    StorageClass,
    /// A layout-class identifier.
    LayoutClass,
    /// A runtime cache-owner identifier.
    CacheOwner,
    /// An extension module identifier.
    ExtensionModule,
}

/// Reports invalid placement constraints.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EngineId, PlacementConstraintError, ProgramPlacementConstraint};
///
/// let engine = EngineId::new("tenferro.cpu").unwrap();
/// let error = ProgramPlacementConstraint::new(vec![engine.clone(), engine], None).unwrap_err();
/// assert!(matches!(error, PlacementConstraintError::DuplicateEngine { .. }));
/// ```
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PlacementConstraintError {
    /// The same engine appears more than once in a preference list.
    #[error(
        "engine {engine_id:?} is duplicated at positions \
         {first_index} and {duplicate_index}"
    )]
    DuplicateEngine {
        /// The duplicated engine identifier.
        engine_id: EngineId,
        /// The first position where the engine appeared.
        first_index: usize,
        /// The duplicate position that failed validation.
        duplicate_index: usize,
    },
}

/// Identifies a runtime registration slot involved in a configuration conflict.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{EngineId, RegistrationKey};
///
/// let key = RegistrationKey::Engine(EngineId::new("tenferro.cpu")?);
/// assert!(format!("{key:?}").contains("tenferro.cpu"));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RegistrationKey {
    /// A direct engine registration.
    Engine(EngineId),
    /// An extension engine registration.
    ExtensionEngine {
        /// Extension family.
        family: &'static str,
        /// Runtime engine.
        engine: EngineId,
    },
    /// An extension planning config registration.
    ExtensionPlanning {
        /// Extension module.
        module: ExtensionModuleId,
        /// Runtime engine.
        engine: EngineId,
    },
    /// An extension cache-owner registration.
    ExtensionCacheOwner {
        /// Extension module.
        module: ExtensionModuleId,
        /// Local cache owner.
        owner: CacheOwnerId,
    },
    /// A transfer provider keyed by source and destination storage classes.
    TransferProvider {
        /// Source storage class.
        source: StorageClass,
        /// Destination storage class.
        destination: StorageClass,
    },
}

/// Reports invalid execution-policy configuration.
///
/// Phase 4 currently has no invalid public execution policy state, but the
/// error type is part of the accepted runtime configuration contract.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::ExecutionPolicyError;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<ExecutionPolicyError>();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionPolicyError {}

impl fmt::Display for ExecutionPolicyError {
    fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {}
    }
}

impl std::error::Error for ExecutionPolicyError {}

/// Reports invalid runtime configuration.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EngineId, RuntimeConfigError};
///
/// let error = RuntimeConfigError::DuplicateEngine {
///     engine_id: EngineId::new("tenferro.cpu").unwrap(),
/// };
/// assert!(error.to_string().contains("already registered"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeConfigError {
    /// Runtime identity or registration identity allocation exhausted.
    #[error("runtime identity space exhausted")]
    IdentityExhausted,
    /// A namespaced runtime identifier was malformed.
    #[error("malformed {kind:?} identifier")]
    MalformedIdentity {
        /// Identifier class that failed validation.
        kind: IdentityKind,
    },
    /// An engine registration tried to insert a different value for an
    /// already-registered engine ID.
    #[error("engine {engine_id:?} is already registered")]
    DuplicateEngine {
        /// Duplicated engine identifier.
        engine_id: EngineId,
    },
    /// A direct capability slot was registered twice where replacement is not
    /// accepted.
    #[error("duplicate {capability:?} capability")]
    DuplicateCapability {
        /// Duplicated capability kind.
        capability: CoreCapabilityKind,
    },
    /// A registration key conflicted across transactional registration input.
    #[error("conflicting registration {key:?}")]
    ConflictingRegistration {
        /// Conflicting registration key.
        key: RegistrationKey,
    },
    /// A requested engine replacement or removal had no existing record.
    #[error("engine {engine_id:?} is not registered")]
    MissingEngine {
        /// Missing engine identifier.
        engine_id: EngineId,
    },
    /// Engine registration provided no supported storage classes.
    #[error("engine {engine_id:?} has no storage classes")]
    EmptyStorageClasses {
        /// Engine being registered.
        engine_id: EngineId,
    },
    /// Engine registration repeated a storage class.
    #[error(
        "engine {engine_id:?} storage class {storage_class:?} is duplicated at \
         positions {first_index} and {duplicate_index}"
    )]
    DuplicateStorageClass {
        /// Engine being registered.
        engine_id: EngineId,
        /// Duplicated storage class.
        storage_class: StorageClass,
        /// First position where the storage class appeared.
        first_index: usize,
        /// Duplicate position that failed validation.
        duplicate_index: usize,
    },
    /// Engine registration selected a default storage class that is not in its
    /// supported list.
    #[error("engine {engine_id:?} default storage class {default_storage_class:?} is not listed")]
    DefaultStorageClassNotListed {
        /// Engine being registered.
        engine_id: EngineId,
        /// Missing default storage class.
        default_storage_class: StorageClass,
    },
    /// A replacement attempted to reuse an engine ID with a different execution
    /// context identity in a context where that is invalid.
    #[error("engine {engine_id:?} context identity mismatch")]
    ContextIdentityMismatch {
        /// Engine with the mismatched context identity.
        engine_id: EngineId,
    },
    /// Execution policy failed validation.
    #[error("invalid execution policy: {reason}")]
    InvalidExecutionPolicy {
        /// Typed policy validation reason.
        reason: ExecutionPolicyError,
    },
    /// Extension module transaction failed.
    #[error("extension module registration failed")]
    ExtensionModule {
        /// Typed extension module source.
        #[source]
        source: ExtensionModuleError,
    },
}

impl From<IdentityError> for RuntimeConfigError {
    fn from(source: IdentityError) -> Self {
        Self::MalformedIdentity {
            kind: source.kind(),
        }
    }
}

/// Reports failure to publish a runtime reconfiguration.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{RuntimeConfigError, RuntimeReconfigureError};
///
/// let error = RuntimeReconfigureError::Edit {
///     source: RuntimeConfigError::IdentityExhausted,
/// };
/// assert!(std::error::Error::source(&error).is_some());
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeReconfigureError {
    /// Runtime state could not be read or published.
    #[error("runtime state failed")]
    State {
        /// Typed state source.
        #[source]
        source: super::RuntimeStateError,
    },
    /// The edit closure or publication validation rejected the candidate.
    #[error("runtime reconfiguration edit failed")]
    Edit {
        /// Typed configuration source.
        #[source]
        source: RuntimeConfigError,
    },
    /// Another writer published over the snapshot this edit was based on.
    #[error("runtime was concurrently reconfigured from {base:?} to {current:?}")]
    ConcurrentReconfiguration {
        /// Epoch captured before the edit closure ran.
        base: RuntimeEpoch,
        /// Epoch observed at publication time.
        current: RuntimeEpoch,
    },
    /// Runtime epoch cannot advance without wrapping.
    #[error("runtime epoch exhausted at {current:?}")]
    EpochExhausted {
        /// Current maximum epoch.
        current: RuntimeEpoch,
    },
}

/// Reports invalid extension module registration transactions.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{EngineId, ExtensionModuleError, ExtensionModuleId};
///
/// let error = ExtensionModuleError::MissingPlanningConfig {
///     module_id: ExtensionModuleId::new("tenferro.module.test")?,
///     engine_id: EngineId::new("tenferro.engine.test")?,
/// };
/// assert!(error.to_string().contains("planning config"));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExtensionModuleError {
    /// A distinct module is already installed at the same module ID.
    #[error("extension module {module_id:?} is already installed")]
    ConflictingModule {
        /// Conflicting module ID.
        module_id: ExtensionModuleId,
    },
    /// A distinct extension engine is already registered for the same
    /// `(family, engine)` key.
    #[error("extension module {module_id:?} has conflicting engine {family_id:?}/{engine_id:?}")]
    ConflictingEngine {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Extension family ID.
        family_id: &'static str,
        /// Runtime engine ID.
        engine_id: EngineId,
    },
    /// A planning config was registered without exactly one matching engine.
    #[error(
        "extension module {module_id:?} registered planning config without engine {engine_id:?}"
    )]
    PlanningConfigWithoutEngine {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Runtime engine ID.
        engine_id: EngineId,
    },
    /// Planning config family did not match the registered engine family.
    #[error(
        "extension module {module_id:?} config for engine {engine_id:?} has family \
         {actual:?}, expected {expected:?}"
    )]
    PlanningConfigFamilyMismatch {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Runtime engine ID.
        engine_id: EngineId,
        /// Expected extension family ID.
        expected: &'static str,
        /// Actual extension family ID.
        actual: &'static str,
    },
    /// An engine was registered without its required planning config.
    #[error("extension module {module_id:?} engine {engine_id:?} has no planning config")]
    MissingPlanningConfig {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Runtime engine ID.
        engine_id: EngineId,
    },
    /// A distinct planning config is already registered for the same engine.
    #[error("extension module {module_id:?} has conflicting planning config for {engine_id:?}")]
    ConflictingPlanningConfig {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Runtime engine ID.
        engine_id: EngineId,
    },
    /// A distinct cache owner is already registered for the same local owner ID.
    #[error("extension module {module_id:?} has conflicting cache owner {owner:?}")]
    ConflictingCacheOwner {
        /// Extension module ID.
        module_id: ExtensionModuleId,
        /// Local cache owner ID.
        owner: CacheOwnerId,
    },
}

/// Reports failures while building value-free input signatures.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
/// use tenferro_tensor::Placement;
///
/// let error = InputSignatureEntry::new(
///     DType::F64,
///     [2_usize].into_iter().collect(),
///     Placement::default(),
///     LayoutClass::new("tenferro.layout.strided").unwrap(),
///     [1_isize, 2].into_iter().collect(),
///     None,
/// )
/// .unwrap_err();
/// assert!(matches!(error, tenferro_runtime::InputSignatureError::ShapeStrideRankMismatch { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum InputSignatureError {
    /// Shape rank and stride rank disagree.
    #[error("shape rank {rank} does not match stride count {stride_count}")]
    ShapeStrideRankMismatch {
        /// Number of shape axes.
        rank: usize,
        /// Number of stride axes.
        stride_count: usize,
    },
    /// The alignment class is outside the finite `usize` alignment lattice.
    #[error("alignment class {alignment_log2} is outside the usize alignment lattice")]
    InvalidAlignmentClass {
        /// Base-2 logarithm of the requested alignment.
        alignment_log2: u8,
    },
    /// Tensor metadata could not be read for an input.
    #[error("input {input} metadata is unavailable")]
    TensorMetadata {
        /// Input position in the prepare request.
        input: usize,
        /// Original typed tensor error.
        source: TensorError,
    },
}

/// Explains why rank specialization is required by a requested field.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::RankRequirement;
///
/// assert_eq!(RankRequirement::ExactStrides.to_string(), "exact strides");
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RankRequirement {
    /// A concrete axis dimension was requested.
    ConcreteAxis {
        /// Axis that requires rank specialization.
        axis: u32,
    },
    /// Exact strides were requested.
    ExactStrides,
}

impl std::fmt::Display for RankRequirement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConcreteAxis { axis } => write!(formatter, "concrete axis {axis}"),
            Self::ExactStrides => formatter.write_str("exact strides"),
        }
    }
}

/// Reports invalid input-specialization requirements.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{
///     InputSpecializationRequirements, InputSpecializationRequirementsError, RankRequirement,
/// };
///
/// let mut builder = InputSpecializationRequirements::builder();
/// builder.rank(false).concrete_dimensions(vec![2]);
/// assert_eq!(
///     builder.build().unwrap_err(),
///     InputSpecializationRequirementsError::RankRequired {
///         reason: RankRequirement::ConcreteAxis { axis: 2 },
///     }
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum InputSpecializationRequirementsError {
    /// The same concrete axis appears more than once.
    #[error("axis {axis} is duplicated at positions {first_index} and {duplicate_index}")]
    DuplicateAxis {
        /// Duplicated concrete axis.
        axis: u32,
        /// First position carrying the axis.
        first_index: usize,
        /// Duplicate position that failed validation.
        duplicate_index: usize,
    },
    /// A requested specialization field requires rank specialization.
    #[error("{reason} requires rank specialization")]
    RankRequired {
        /// Requirement that needs rank specialization.
        reason: RankRequirement,
    },
    /// The alignment class is outside the finite `usize` alignment lattice.
    #[error("alignment class {alignment_log2} is outside the usize alignment lattice")]
    InvalidAlignmentClass {
        /// Base-2 logarithm of the requested alignment.
        alignment_log2: u8,
    },
}

/// Reports failures while projecting signatures through specialization requirements.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{InputSignature, SpecializationRequirements};
///
/// let error = SpecializationRequirements::polymorphic(1)
///     .project(&InputSignature::new(Vec::new()))
///     .unwrap_err();
/// assert!(matches!(error, tenferro_runtime::PrepareError::Specialization { .. }));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SpecializationError {
    /// The signature does not have the required input arity.
    #[error("expected {expected} inputs but got {actual}")]
    WrongInputCount {
        /// Required input count.
        expected: usize,
        /// Actual input count.
        actual: usize,
    },
    /// A concrete-axis request names an axis outside the actual rank.
    #[error("input {input} axis {axis} is outside rank {rank}")]
    AxisOutOfRange {
        /// Input position in the signature.
        input: usize,
        /// Requested axis.
        axis: u32,
        /// Actual rank.
        rank: usize,
    },
    /// A required host-pointer alignment class is unavailable.
    #[error("input {input} has unknown alignment; required class {required_alignment_log2}")]
    AlignmentUnavailable {
        /// Input position in the signature.
        input: usize,
        /// Required base-2 logarithm alignment class.
        required_alignment_log2: u8,
    },
    /// A specialization projection could not encode every requested axis in
    /// the finite public axis type.
    #[error("input {input} rank {rank} is too large to project into u32 axes")]
    ProjectionOverflow {
        /// Input position in the signature.
        input: usize,
        /// Rank that could not be encoded.
        rank: usize,
    },
}

/// A typed execution-context mismatch from erased runtime dispatch.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{ExecutionContextIdentity, ExecutionContextMismatch};
///
/// let error = ExecutionContextMismatch {
///     expected: ExecutionContextIdentity::of::<u64>(),
///     actual: ExecutionContextIdentity::of::<u32>(),
/// };
/// assert_eq!(error.expected, ExecutionContextIdentity::of::<u64>());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("execution context mismatch: expected {expected:?}, actual {actual:?}")]
pub struct ExecutionContextMismatch {
    /// Expected execution-context type identity.
    pub expected: super::ExecutionContextIdentity,
    /// Actual stored execution-context type identity.
    pub actual: super::ExecutionContextIdentity,
}

/// Reports provider violations of runtime preparation contracts.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{CoreCapabilityKind, ProviderContractError};
///
/// let error = ProviderContractError::WrongOperationFamily {
///     expected: CoreCapabilityKind::Elementwise,
///     operation: "dot_general",
/// };
/// assert!(error.to_string().contains("dot_general"));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ProviderContractError {
    /// A provider accepted a request from the wrong core operation family.
    #[error("operation {operation:?} is not in {expected:?}")]
    WrongOperationFamily {
        /// Expected core capability family.
        expected: CoreCapabilityKind,
        /// Operation diagnostic name.
        operation: &'static str,
    },
    /// A prepared operation returned a different binding than requested.
    #[error("prepared binding mismatch")]
    BindingMismatch {
        /// Runtime-created binding requested from the provider.
        expected: Box<PreparedOperationBinding>,
        /// Binding returned by the provider.
        actual: Box<PreparedOperationBinding>,
    },
    /// A prepared operation returned a different specialization projection.
    #[error("prepared specialization projection mismatch")]
    ProjectionMismatch {
        /// Runtime-created projection requested from the provider.
        expected: Box<SpecializationProjection>,
        /// Projection returned by the provider.
        actual: Box<SpecializationProjection>,
    },
    /// A provider returned an invalid specialization.
    #[error("invalid provider specialization: {source}")]
    InvalidSpecialization {
        /// Original specialization failure.
        #[source]
        source: SpecializationError,
    },
    /// A provider requested specialization that did not strictly widen the
    /// current requirements.
    #[error("specialization did not strictly widen the current requirements")]
    NonWideningSpecialization {
        /// Previous specialization requirements.
        previous: Box<super::SpecializationRequirements>,
        /// Provider-requested specialization requirements.
        next: Box<super::SpecializationRequirements>,
    },
    /// A provider exceeded the finite specialization retry bound.
    #[error("specialization retry limit exceeded after {attempts} attempts; limit was {limit}")]
    SpecializationRetryLimitExceeded {
        /// Number of redirect attempts observed.
        attempts: usize,
        /// Computed finite retry bound.
        limit: usize,
    },
}

/// Reports failures while preparing runtime execution artifacts.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{InputSignature, PrepareError, SpecializationRequirements};
///
/// let error = SpecializationRequirements::polymorphic(1)
///     .project(&InputSignature::new(Vec::new()))
///     .unwrap_err();
/// assert!(matches!(error, PrepareError::Specialization { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    /// Prepared artifact belongs to a different runtime.
    #[error("runtime mismatch: expected {expected:?}, actual {actual:?}")]
    RuntimeMismatch {
        /// Runtime expected by the prepared artifact.
        expected: RuntimeId,
        /// Runtime currently executing.
        actual: RuntimeId,
    },
    /// Prepared artifact was created for a stale runtime epoch.
    #[error("stale runtime epoch: prepared {prepared:?}, current {current:?}")]
    StaleEpoch {
        /// Epoch captured by the prepared artifact.
        prepared: RuntimeEpoch,
        /// Runtime's current epoch.
        current: RuntimeEpoch,
    },
    /// Input signature construction failed.
    #[error("input signature failed")]
    InputSignature {
        /// Typed source error.
        source: InputSignatureError,
    },
    /// Semantic shape guards failed.
    #[error("shape guard failed")]
    ShapeGuard {
        /// Typed source error.
        #[source]
        source: ShapeGuardError,
    },
    /// Signature projection through specialization requirements failed.
    #[error("specialization failed")]
    Specialization {
        /// Typed source error.
        source: SpecializationError,
    },
    /// No eligible engine matched the placement and capability constraints.
    #[error("no eligible engine for {constraint:?}")]
    NoEligibleEngine {
        /// Placement constraint that could not be satisfied.
        constraint: ProgramPlacementConstraint,
    },
    /// No eligible engine declared an ingress compatible with an input placement.
    #[error("no eligible input ingress for input {input_index} at placement {placement:?}")]
    NoInputIngress {
        /// Input position in the compiled-program signature.
        input_index: usize,
        /// Placement rejected by every eligible engine ingress.
        placement: tenferro_tensor::Placement,
    },
    /// A resolved placement references an engine absent from its preparation snapshot.
    #[error("resolved engine {engine_id:?} is unavailable in the preparation snapshot")]
    ResolvedEngineUnavailable {
        /// Engine referenced by the resolved placement.
        engine_id: EngineId,
    },
    /// No provider supports this operation under the requested constraints.
    #[error("operation is unsupported: {reason}")]
    Unsupported {
        /// Deterministic unsupported reason.
        reason: UnsupportedReason,
    },
    /// Deterministic execution was requested from an engine that cannot provide it.
    #[error("determinism unsupported by engine {engine_id:?}")]
    DeterminismUnsupported {
        /// Engine that cannot satisfy deterministic preparation.
        engine_id: EngineId,
    },
    /// Provider returned a value that violates the runtime preparation contract.
    #[error("provider contract violation: {source}")]
    ProviderContract {
        /// Typed provider-contract source.
        #[source]
        source: ProviderContractError,
    },
    /// Preparation recursion reached an unsupported cycle.
    #[error("preparation cycle at {key:?}")]
    PreparationCycle {
        /// Bounded summary of the recursive key.
        key: PreparationKeySummary,
    },
    /// Preparation recursively requested a different key.
    #[error("nested preparation unsupported: parent {parent:?}, requested {requested:?}")]
    NestedPreparationUnsupported {
        /// Parent key currently preparing on this thread.
        parent: PreparationKeySummary,
        /// Nested key requested by the producer.
        requested: PreparationKeySummary,
    },
    /// The preparation cache is full of active/queued distinct keys.
    #[error(
        "preparation cache is at capacity: {in_flight} in flight and \
         {queued_distinct_keys} queued distinct keys"
    )]
    CacheInFlightCapacityExceeded {
        /// Active distinct preparations.
        in_flight: usize,
        /// Queued distinct keys.
        queued_distinct_keys: usize,
    },
    /// Runtime cache state could not be accessed.
    #[error("preparation cache state unavailable: {source}")]
    CacheState {
        /// Runtime state source.
        #[source]
        source: super::RuntimeStateError,
    },
    /// Engine or internal preparation source failed.
    #[error("engine preparation failed: {source}")]
    Engine {
        /// Shared typed source.
        #[source]
        source: Arc<dyn Error + Send + Sync>,
    },
}
