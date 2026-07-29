use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::Arc;

use tenferro_ops::dim_expr::{DimExpr, DimExprEvalError};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

use crate::compiler::{semantic_staging::stage_semantic_program, CompilerOptions};
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::graph::CompiledGraph;
use crate::program::{
    CoreSemanticOp, FrozenProgram, ProgramShapeRelation, SemanticFingerprint, SemanticOpRef,
    SemanticOperationView, SemanticProgram,
};

use super::cache::{
    CacheLookup, CacheProduced, PreparedCacheKey, PreparedValue, RuntimeCacheSet, SharedRetention,
};
use super::extension::ExtensionFamilyId;
use super::schedule::{
    ExecutionLocation, ScheduleBuildError, ScheduledGraph, TransferReachability,
};
use super::{
    CoreCapabilityKind, CorePrepareContext, DotGeneralPreparation, DotGeneralPrepareRequest,
    ElementwisePrepareRequest, ElementwiseRuntime, EngineId, ExecutionContextIdentity,
    ExtensionEngine, ExtensionModuleId, ExtensionPlanningConfig, ExtensionPrepareRequest,
    HardwareClassId, IndexingPrepareRequest, IndexingRuntime, InputSignature,
    InputSpecializationRequirements, LayoutPrepareRequest, LayoutProjection, LayoutRuntime,
    PlacementSpecialization, PrepareError, PrepareOptions, PrepareOptionsKey,
    PreparedOperationBinding, PreparedOperationPlan, ProgramPlacementConstraint,
    ProviderContractError, ReductionPrepareRequest, ReductionRuntime, RegistrationIdentity,
    ResolvedPlanningConfig, ResolvedPlanningKey, ResolvedProgramPlacement, Runtime,
    RuntimeStateError, SpecializationProjection, SpecializationRequirements, StorageClass,
    UnsupportedReason,
};

pub(crate) type PreparedProgramResult<T> = Result<T, Arc<PrepareError>>;

#[derive(Clone)]
struct ExtensionPlanningIdentity {
    module_id: ExtensionModuleId,
    family_id: ExtensionFamilyId,
    engine_id: EngineId,
    payload_fingerprint: u64,
    config: Arc<dyn ExtensionPlanningConfig>,
}

impl fmt::Debug for ExtensionPlanningIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionPlanningIdentity")
            .field("module_id", &self.module_id)
            .field("family_id", &self.family_id)
            .field("engine_id", &self.engine_id)
            .field("payload_fingerprint", &self.payload_fingerprint)
            .finish_non_exhaustive()
    }
}

struct PreparedRootIdentity {
    semantic_fingerprint: SemanticFingerprint,
    semantic: Arc<SemanticProgram>,
    runtime_id: super::RuntimeId,
    epoch: super::RuntimeEpoch,
    resolved_placement: ResolvedProgramPlacement,
    engine_id: EngineId,
    registration_identity: RegistrationIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
    compiler_options: CompilerOptions,
    resolved_planning: ResolvedPlanningKey,
    prepare_options: PrepareOptionsKey,
    operation_bindings: Box<[PreparedOperationBinding]>,
    operation_placements: Box<[ResolvedProgramPlacement]>,
    input_locations: Box<[ExecutionLocation]>,
    extension_planning: Box<[ExtensionPlanningIdentity]>,
}

impl fmt::Debug for PreparedRootIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedRootIdentity")
            .field("semantic_fingerprint", &self.semantic_fingerprint)
            .field("runtime_id", &self.runtime_id)
            .field("epoch", &self.epoch)
            .field("resolved_placement", &self.resolved_placement)
            .field("engine_id", &self.engine_id)
            .field("registration_identity", &self.registration_identity)
            .field("context_identity", &self.context_identity)
            .field("hardware_class", &self.hardware_class)
            .field("compiler_options", &self.compiler_options)
            .field("resolved_planning", &self.resolved_planning)
            .field("prepare_options", &self.prepare_options)
            .field("operation_bindings", &self.operation_bindings.len())
            .field("operation_placements", &self.operation_placements.len())
            .field("input_locations", &self.input_locations.len())
            .field("extension_planning", &self.extension_planning.len())
            .finish()
    }
}

#[derive(Clone, Debug)]
enum PreparedRootKey {
    Identity(Arc<PreparedRootIdentity>),
    Prepared(Arc<PreparedProgramRoot>),
}

impl PreparedRootKey {
    fn identity(&self) -> &PreparedRootIdentity {
        match self {
            Self::Identity(identity) => identity,
            Self::Prepared(root) => &root.identity,
        }
    }

    fn prepared_root(&self) -> Option<Arc<PreparedProgramRoot>> {
        match self {
            Self::Identity(_) => None,
            Self::Prepared(root) => Some(Arc::clone(root)),
        }
    }

    fn retained_bytes(&self) -> Option<usize> {
        match self {
            Self::Identity(identity) => prepared_root_identity_key_retained_bytes(identity),
            Self::Prepared(_) => Some(0),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedEntryKey {
    root: PreparedRootKey,
    requirements: SpecializationRequirements,
    specialization: SpecializationProjection,
}

impl PreparedEntryKey {
    fn new(
        root: PreparedRootKey,
        requirements: SpecializationRequirements,
        specialization: SpecializationProjection,
    ) -> Self {
        Self {
            root,
            requirements,
            specialization,
        }
    }
}

impl PreparedCacheKey for PreparedEntryKey {
    type Shared = PreparedProgramRoot;

    fn compact_digest(&self) -> u128 {
        let first = compact_digest_half(0x5034_4b31, self);
        let second = compact_digest_half(0x5034_4b32, self);
        (u128::from(first) << 64) | u128::from(second)
    }

    fn exact_eq(&self, other: &Self) -> bool {
        root_identity_exact_eq(self.root.identity(), other.root.identity())
            && self.requirements == other.requirements
            && self.specialization == other.specialization
    }

    fn retained_bytes(&self) -> Option<usize> {
        checked_sum([
            size_of::<PreparedEntryKey>(),
            self.root.retained_bytes()?,
            specialization_requirements_retained_bytes(&self.requirements)?,
            specialization_projection_retained_bytes(&self.specialization)?,
        ])
    }

    fn summary(&self) -> super::PreparationKeySummary {
        super::PreparationKeySummary::new(self.compact_digest())
    }

    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>> {
        self.root
            .prepared_root()
            .map(|root| root.shared_retention())
    }
}

pub(crate) struct PreparedProgramRoot {
    identity: Arc<PreparedRootIdentity>,
    semantic: Arc<SemanticProgram>,
    staging: Arc<ExecProgram>,
    schedule: Arc<ScheduledGraph>,
    extension_planning: Arc<[Arc<dyn ExtensionPlanningConfig>]>,
    logical_retained_bytes: Option<usize>,
}

impl PreparedProgramRoot {
    fn new(
        identity: Arc<PreparedRootIdentity>,
        staging: Arc<ExecProgram>,
        extension_planning: Arc<[Arc<dyn ExtensionPlanningConfig>]>,
        root_location: ExecutionLocation,
        operation_locations: &[ExecutionLocation],
        transfer_reachability: &TransferReachability,
    ) -> Result<Self, ScheduleBuildError> {
        let semantic = Arc::clone(&identity.semantic);
        let schedule = Arc::new(ScheduledGraph::from_exec_program(
            &staging,
            root_location,
            &identity.input_locations,
            operation_locations,
            transfer_reachability,
        )?);
        let logical_retained_bytes = prepared_program_root_retained_bytes(
            &identity,
            &semantic,
            &staging,
            &schedule,
            &extension_planning,
        );
        Ok(Self {
            identity,
            semantic,
            staging,
            schedule,
            extension_planning,
            logical_retained_bytes,
        })
    }

    fn shared_retention(self: &Arc<Self>) -> SharedRetention<Self> {
        SharedRetention {
            value: Arc::clone(self),
            retained_bytes: self.logical_retained_bytes,
        }
    }

    pub(crate) fn staging(&self) -> &ExecProgram {
        &self.staging
    }

    pub(crate) fn schedule(&self) -> &ScheduledGraph {
        &self.schedule
    }

    pub(crate) fn engine_id(&self) -> &EngineId {
        &self.identity.engine_id
    }

    pub(crate) fn resolved_placement(&self) -> &ResolvedProgramPlacement {
        &self.identity.resolved_placement
    }

    pub(crate) fn operation_placements(&self) -> &[ResolvedProgramPlacement] {
        &self.identity.operation_placements
    }

    pub(crate) fn input_locations(&self) -> &[ExecutionLocation] {
        &self.identity.input_locations
    }

    pub(crate) fn epoch(&self) -> super::RuntimeEpoch {
        self.identity.epoch
    }

    #[cfg(test)]
    pub(crate) fn semantic_for_test(&self) -> &Arc<SemanticProgram> {
        &self.semantic
    }

    #[cfg(test)]
    pub(crate) fn staging_for_test(&self) -> &ExecProgram {
        &self.staging
    }

    #[cfg(test)]
    pub(crate) fn schedule_for_test(&self) -> &ScheduledGraph {
        &self.schedule
    }

    #[cfg(test)]
    pub(crate) fn resolved_placement_for_test(&self) -> &ResolvedProgramPlacement {
        &self.identity.resolved_placement
    }

    #[cfg(test)]
    #[allow(
        dead_code,
        reason = "available to integration tests of root sharing/accounting"
    )]
    pub(crate) fn logical_retained_bytes_for_test(&self) -> Option<usize> {
        self.logical_retained_bytes
    }
}

impl fmt::Debug for PreparedProgramRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedProgramRoot")
            .field("identity", &self.identity)
            .field("semantic", &self.semantic)
            .field(
                "staging_instruction_count",
                &self.staging.instructions.len(),
            )
            .field("scheduled_node_count", &self.schedule.nodes().len())
            .field("extension_planning", &self.extension_planning.len())
            .field("logical_retained_bytes", &self.logical_retained_bytes)
            .finish()
    }
}

#[derive(Debug)]
pub(crate) struct PreparedProgram {
    #[allow(
        dead_code,
        reason = "Phase 5 execution validates runtime and epoch through the shared root"
    )]
    root: Arc<PreparedProgramRoot>,
    specialization: SpecializationProjection,
    operations: Box<[PreparedOperationPlan]>,
}

impl PreparedProgram {
    fn new(
        root: Arc<PreparedProgramRoot>,
        specialization: SpecializationProjection,
        operations: Box<[PreparedOperationPlan]>,
    ) -> Self {
        Self {
            root,
            specialization,
            operations,
        }
    }

    pub(crate) fn root(&self) -> &Arc<PreparedProgramRoot> {
        &self.root
    }

    pub(crate) fn operations(&self) -> &[PreparedOperationPlan] {
        &self.operations
    }

    #[cfg(test)]
    pub(crate) fn root_for_test(&self) -> &Arc<PreparedProgramRoot> {
        &self.root
    }

    #[cfg(test)]
    pub(crate) fn specialization_for_test(&self) -> &SpecializationProjection {
        &self.specialization
    }

    #[cfg(test)]
    pub(crate) fn operations_for_test(&self) -> &[PreparedOperationPlan] {
        &self.operations
    }
}

impl PreparedValue for PreparedProgram {
    fn retained_bytes(&self) -> Option<usize> {
        checked_sum([
            size_of::<PreparedProgram>(),
            specialization_projection_retained_bytes(&self.specialization)?,
            self.operations
                .len()
                .checked_mul(size_of::<PreparedOperationPlan>())?,
            checked_sum_options(
                self.operations
                    .iter()
                    .map(|operation| Some(operation.retained_bytes())),
            )?,
        ])
    }
}

#[derive(Clone)]
struct PreparationContext {
    root_identity: Arc<PreparedRootIdentity>,
    operation_dispatch: Arc<[OperationDispatch]>,
    staging: Arc<ExecProgram>,
    root_location: ExecutionLocation,
    operation_locations: Arc<[ExecutionLocation]>,
    transfer_reachability: Arc<TransferReachability>,
    extension_planning: Arc<[Arc<dyn ExtensionPlanningConfig>]>,
}

#[derive(Clone)]
struct OperationDispatch {
    binding: PreparedOperationBinding,
    resolved_placement: ResolvedProgramPlacement,
    planning: ResolvedPlanningConfig,
    prepare_options_key: PrepareOptionsKey,
    provider: OperationProvider,
}

#[derive(Clone)]
struct SelectedOperationDispatch {
    dispatch: OperationDispatch,
    extension_identity: Option<ExtensionPlanningIdentity>,
    extension_config: Option<Arc<dyn ExtensionPlanningConfig>>,
}

#[derive(Clone)]
enum OperationProvider {
    Elementwise(Arc<dyn ElementwiseRuntime>),
    Reduction(Arc<dyn ReductionRuntime>),
    Indexing(Arc<dyn IndexingRuntime>),
    DotGeneral(Arc<dyn DotGeneralPreparation>),
    Layout(Arc<dyn LayoutRuntime>),
    Extension {
        engine: Arc<dyn ExtensionEngine>,
        config: Arc<dyn ExtensionPlanningConfig>,
    },
}

impl fmt::Debug for OperationProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elementwise(_) => formatter.write_str("OperationProvider::Elementwise"),
            Self::Reduction(_) => formatter.write_str("OperationProvider::Reduction"),
            Self::Indexing(_) => formatter.write_str("OperationProvider::Indexing"),
            Self::DotGeneral(_) => formatter.write_str("OperationProvider::DotGeneral"),
            Self::Layout(_) => formatter.write_str("OperationProvider::Layout"),
            Self::Extension { engine, config } => formatter
                .debug_struct("OperationProvider::Extension")
                .field("family_id", &engine.family_id())
                .field("engine_id", engine.engine_id())
                .field("config_family_id", &config.family_id())
                .finish(),
        }
    }
}

#[allow(
    dead_code,
    reason = "Phase 5 public execution paths consume prepared programs"
)]
pub(crate) fn prepare_for(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
    frozen: &FrozenProgram,
    signature: &InputSignature,
    options: &PrepareOptions,
) -> PreparedProgramResult<Arc<PreparedProgram>> {
    prepare_for_with_compiler_options(
        runtime,
        caches,
        frozen,
        CompilerOptions::default(),
        signature,
        options,
    )
}

pub(crate) fn prepare_compiled_for(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
    program: &CompiledGraph,
    signature: &InputSignature,
    options: &PrepareOptions,
) -> PreparedProgramResult<Arc<PreparedProgram>> {
    prepare_for_with_compiler_options(
        runtime,
        caches,
        program.frozen(),
        program.compiler_options(),
        signature,
        options,
    )
}

fn prepare_for_with_compiler_options(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
    frozen: &FrozenProgram,
    compiler_options: CompilerOptions,
    signature: &InputSignature,
    options: &PrepareOptions,
) -> PreparedProgramResult<Arc<PreparedProgram>> {
    validate_signature_arity(frozen, signature)?;
    validate_semantic_shape_guards(&frozen.program, signature)?;
    let signature_bytes = signature
        .logical_retained_bytes()
        .ok_or_else(accounting_prepare_error)?;
    let snapshot = runtime
        .snapshot()
        .map_err(|source| Arc::new(PrepareError::CacheState { source }))?;
    let context = Arc::new(resolve_preparation_context(
        runtime,
        &snapshot,
        frozen,
        compiler_options,
        signature,
        options,
    )?);
    let mut requirements = SpecializationRequirements::polymorphic(signature.entries().len());
    let retry_limit = specialization_retry_bound(&requirements, signature)?;
    let mut specialization = project_requirements(&requirements, signature)?;
    let mut root = PreparedRootKey::Identity(Arc::clone(&context.root_identity));
    let mut redirects = 0usize;

    loop {
        let key = PreparedEntryKey::new(root.clone(), requirements.clone(), specialization.clone());
        let producer_key = key.clone();
        let producer_context = Arc::clone(&context);
        let lookup = caches.prepared().get_or_prepare(
            key,
            options.cache_in_flight(),
            signature_bytes,
            move || produce_prepared_entry(&producer_context, &producer_key, signature),
        )?;
        match lookup {
            CacheLookup::Ready(value) => return Ok(value),
            CacheLookup::FailedDeterministic(error) | CacheLookup::FailedTransient(error) => {
                return Err(error);
            }
            CacheLookup::Redirect {
                requirements: next,
                shared,
            } => {
                validate_widening(&requirements, &next)?;
                redirects = redirects
                    .checked_add(1)
                    .ok_or_else(accounting_prepare_error)?;
                if redirects > retry_limit {
                    return Err(Arc::new(PrepareError::ProviderContract {
                        source: ProviderContractError::SpecializationRetryLimitExceeded {
                            attempts: redirects,
                            limit: retry_limit,
                        },
                    }));
                }
                let shared = shared.ok_or_else(missing_shared_root_error)?;
                specialization = project_requirements(&next, signature)?;
                requirements = next;
                root = PreparedRootKey::Prepared(shared);
            }
        }
    }
}

fn resolve_preparation_context(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    frozen: &FrozenProgram,
    compiler_options: CompilerOptions,
    signature: &InputSignature,
    options: &PrepareOptions,
) -> PreparedProgramResult<PreparationContext> {
    let constraint = options
        .placement()
        .cloned()
        .unwrap_or_else(ProgramPlacementConstraint::any);
    let explicit = constraint.allowed_engines();
    let mut candidates = Vec::new();
    if explicit.is_empty() {
        candidates.extend(snapshot.engine_views_for_preparation());
    } else {
        candidates.extend(explicit.iter().filter_map(|engine| snapshot.engine(engine)));
    }
    let staging = Arc::new(
        stage_semantic_program(&frozen.program, compiler_options).map_err(|source| {
            Arc::new(PrepareError::Engine {
                source: Arc::new(source),
            })
        })?,
    );

    let mut missing_extension_family = None;
    let mut last_route_error = None;
    let storage_candidates = candidate_storage_classes(&candidates, &constraint);
    for storage_class in storage_candidates.iter().cloned() {
        match build_operation_dispatch(
            runtime,
            snapshot,
            frozen,
            compiler_options,
            &staging,
            &candidates,
            storage_class,
            signature,
            options,
            &mut missing_extension_family,
        ) {
            Ok(Some(context)) => return Ok(context),
            Ok(None) => {}
            Err(error) if is_route_specific_prepare_error(error.as_ref()) => {
                last_route_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    if constraint.storage_class().is_none() {
        match build_cross_storage_operation_dispatch(
            runtime,
            snapshot,
            frozen,
            compiler_options,
            &staging,
            &candidates,
            signature,
            options,
            &mut missing_extension_family,
        ) {
            Ok(Some(context)) => return Ok(context),
            Ok(None) => {}
            Err(error) if is_route_specific_prepare_error(error.as_ref()) => {
                last_route_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }

    if let Some(error) = last_route_error {
        return Err(error);
    }
    if let Some(operation) = missing_extension_family {
        return Err(Arc::new(PrepareError::Unsupported {
            reason: UnsupportedReason::Operation { operation },
        }));
    }

    Err(Arc::new(PrepareError::NoEligibleEngine { constraint }))
}

fn resolve_input_locations(
    candidates: &[super::EngineSnapshotView<'_>],
    signature: &InputSignature,
    program: &ExecProgram,
    root_location: &ExecutionLocation,
    operation_locations: &[ExecutionLocation],
    transfer_reachability: &TransferReachability,
) -> PreparedProgramResult<Arc<[ExecutionLocation]>> {
    let consumers = physical_input_consumers(program, root_location, operation_locations).map_err(
        |source| {
            Arc::new(PrepareError::Engine {
                source: Arc::new(source),
            })
        },
    )?;
    let mut selected = Vec::with_capacity(signature.entries().len());
    for (input_index, entry) in signature.entries().iter().enumerate() {
        let input_consumers = consumers.get(input_index).ok_or_else(|| {
            Arc::new(PrepareError::NoInputIngress {
                input_index,
                placement: entry.placement().clone(),
            })
        })?;
        let source = candidates
            .iter()
            .flat_map(|engine| {
                engine
                    .storage_classes()
                    .iter()
                    .filter(move |storage| engine.accepts_input_signature(entry, storage))
                    .map(move |storage| {
                        ExecutionLocation::new(
                            engine.engine_id().clone(),
                            engine.event_domain_id(),
                            storage.clone(),
                        )
                    })
            })
            .find(|source| {
                source_reaches_all_consumers(source, input_consumers, transfer_reachability)
            })
            .ok_or_else(|| {
                Arc::new(PrepareError::NoInputIngress {
                    input_index,
                    placement: entry.placement().clone(),
                })
            })?;
        selected.push(source);
    }
    Ok(Arc::from(selected))
}

fn physical_input_consumers(
    program: &ExecProgram,
    root_location: &ExecutionLocation,
    operation_locations: &[ExecutionLocation],
) -> Result<Vec<Vec<ExecutionLocation>>, ScheduleBuildError> {
    let mut input_by_slot = vec![None; program.n_slots];
    for (input_index, &slot) in program.input_slots.iter().enumerate() {
        let input =
            input_by_slot
                .get_mut(slot)
                .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                    slot,
                    value_count: program.n_slots,
                })?;
        *input = Some(input_index);
    }
    let mut consumers = vec![Vec::new(); program.input_slots.len()];
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        let location = match instruction.semantic_operation_index {
            Some(operation_index) => operation_locations.get(operation_index).cloned().ok_or(
                ScheduleBuildError::MissingOperationLocation {
                    instruction_index,
                    operation_index,
                },
            )?,
            None => root_location.clone(),
        };
        for &slot in &instruction.input_slots {
            let input_index =
                input_by_slot
                    .get(slot)
                    .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                        slot,
                        value_count: program.n_slots,
                    })?;
            if let Some(input_index) = input_index {
                consumers[*input_index].push(location.clone());
            }
        }
    }
    Ok(consumers)
}

fn source_reaches_all_consumers(
    source: &ExecutionLocation,
    consumers: &[ExecutionLocation],
    transfer_reachability: &TransferReachability,
) -> bool {
    let mut available = Vec::with_capacity(consumers.len().saturating_add(1));
    available.push(source.clone());
    for destination in consumers {
        if available.iter().any(|location| location == destination) {
            continue;
        }
        if !available.iter().any(|location| {
            transfer_reachability.contains(&(
                location.storage_class().clone(),
                destination.storage_class().clone(),
            ))
        }) {
            return false;
        }
        available.push(destination.clone());
    }
    true
}

fn candidate_storage_classes(
    candidates: &[super::EngineSnapshotView<'_>],
    constraint: &ProgramPlacementConstraint,
) -> Vec<StorageClass> {
    match constraint.storage_class() {
        Some(requested) => {
            if candidates
                .iter()
                .any(|engine| engine.storage_classes().contains(requested))
            {
                vec![requested.clone()]
            } else {
                Vec::new()
            }
        }
        None => {
            let mut classes = Vec::new();
            for engine in candidates {
                let storage = engine.default_storage_class();
                if !classes.iter().any(|existing| existing == storage) {
                    classes.push(storage.clone());
                }
            }
            for engine in candidates {
                for storage in engine.storage_classes() {
                    if !classes.iter().any(|existing| existing == storage) {
                        classes.push(storage.clone());
                    }
                }
            }
            classes
        }
    }
}

pub(crate) fn execution_location(
    snapshot: &super::RuntimeConfigSnapshot,
    placement: &ResolvedProgramPlacement,
) -> PreparedProgramResult<ExecutionLocation> {
    let engine = snapshot.engine(placement.engine_id()).ok_or_else(|| {
        Arc::new(PrepareError::ResolvedEngineUnavailable {
            engine_id: placement.engine_id().clone(),
        })
    })?;
    Ok(ExecutionLocation::new(
        placement.engine_id().clone(),
        engine.event_domain_id(),
        placement.storage_class().clone(),
    ))
}

// INVARIANT: this private preparation boundary carries the already-separated
// runtime, snapshot, compiler, placement, and policy inputs without bundling
// them into a second mutable configuration object.
#[allow(clippy::too_many_arguments)]
fn build_operation_dispatch(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    frozen: &FrozenProgram,
    compiler_options: CompilerOptions,
    staging: &Arc<ExecProgram>,
    candidates: &[super::EngineSnapshotView<'_>],
    storage_class: StorageClass,
    signature: &InputSignature,
    options: &PrepareOptions,
    missing_extension_family: &mut Option<ExtensionFamilyId>,
) -> PreparedProgramResult<Option<PreparationContext>> {
    let mut dispatch_candidates = Vec::with_capacity(frozen.program.operations().len());
    for operation in frozen.program.operations() {
        let selected = operation_dispatch_candidates(
            runtime,
            snapshot,
            candidates,
            operation,
            &storage_class,
            options,
            missing_extension_family,
        )?;
        if selected.is_empty() {
            return Ok(None);
        }
        dispatch_candidates.push(selected);
    }

    if dispatch_candidates.is_empty() {
        let Some(engine) = candidates
            .iter()
            .find(|engine| engine.storage_classes().contains(&storage_class))
        else {
            return Ok(None);
        };
        return build_preparation_context(
            runtime,
            snapshot,
            frozen,
            compiler_options,
            staging,
            candidates,
            signature,
            options,
            Vec::new(),
            Some((engine, storage_class)),
        )
        .map(Some);
    }

    search_dispatch_preferences(&dispatch_candidates, candidates, |selected| {
        build_preparation_context(
            runtime,
            snapshot,
            frozen,
            compiler_options,
            staging,
            candidates,
            signature,
            options,
            selected,
            None,
        )
    })
    .map(Some)
}

// INVARIANT: context construction consumes one validated dispatch selection
// plus the immutable preparation inputs; grouping them would duplicate the
// existing snapshot/options ownership boundaries.
#[allow(clippy::too_many_arguments)]
fn build_preparation_context(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    frozen: &FrozenProgram,
    compiler_options: CompilerOptions,
    staging: &Arc<ExecProgram>,
    candidates: &[super::EngineSnapshotView<'_>],
    signature: &InputSignature,
    options: &PrepareOptions,
    selected: Vec<SelectedOperationDispatch>,
    empty_root: Option<(&super::EngineSnapshotView<'_>, StorageClass)>,
) -> PreparedProgramResult<PreparationContext> {
    let mut dispatch = Vec::with_capacity(selected.len());
    let mut bindings = Vec::with_capacity(selected.len());
    let mut placements = Vec::with_capacity(selected.len());
    let mut extension_identities = Vec::new();
    let mut extension_planning = Vec::new();
    for selected in selected {
        bindings.push(selected.dispatch.binding.clone());
        placements.push(selected.dispatch.resolved_placement.clone());
        if let Some(identity) = selected.extension_identity {
            extension_identities.push(identity);
        }
        if let Some(config) = selected.extension_config {
            extension_planning.push(config);
        }
        dispatch.push(selected.dispatch);
    }

    let (primary_binding, primary_resolved_placement, primary_planning, primary_options_key) =
        if let Some(primary) = dispatch.first() {
            (
                primary.binding.clone(),
                primary.resolved_placement.clone(),
                primary.planning.clone(),
                primary.prepare_options_key.clone(),
            )
        } else {
            let (engine, storage_class) = empty_root.ok_or_else(|| {
                Arc::new(PrepareError::NoEligibleEngine {
                    constraint: options
                        .placement()
                        .cloned()
                        .unwrap_or_else(ProgramPlacementConstraint::any),
                })
            })?;
            let resolved_placement =
                ResolvedProgramPlacement::new(engine.engine_id().clone(), storage_class);
            let planning = ResolvedPlanningConfig::resolve(
                snapshot.execution_policy(),
                options,
                engine.hardware_class().clone(),
            );
            let prepare_options_key = PrepareOptionsKey::from_resolved(
                resolved_placement.clone(),
                planning.hard_workspace_limit_bytes(),
                planning.planning_seed(),
            );
            (
                PreparedOperationBinding::new(
                    runtime.id(),
                    snapshot.epoch(),
                    engine.engine_id().clone(),
                    engine.registration_identity(),
                    engine.context_identity(),
                    engine.hardware_class().clone(),
                ),
                resolved_placement,
                planning,
                prepare_options_key,
            )
        };
    let root_location = execution_location(snapshot, &primary_resolved_placement)?;
    let operation_locations: Arc<[_]> = placements
        .iter()
        .map(|placement| execution_location(snapshot, placement))
        .collect::<PreparedProgramResult<Vec<_>>>()?
        .into();
    let transfer_reachability = snapshot.transfer_reachability_for_preparation();
    let input_locations = resolve_input_locations(
        candidates,
        signature,
        staging,
        &root_location,
        &operation_locations,
        &transfer_reachability,
    )?;
    ScheduledGraph::from_exec_program(
        staging,
        root_location.clone(),
        &input_locations,
        &operation_locations,
        &transfer_reachability,
    )
    .map_err(schedule_prepare_error)?;
    let planning_key = ResolvedPlanningKey::from_config(&primary_planning);
    let root_identity = Arc::new(PreparedRootIdentity {
        semantic_fingerprint: frozen.program.semantic_fingerprint(),
        semantic: Arc::clone(&frozen.program),
        runtime_id: runtime.id(),
        epoch: snapshot.epoch(),
        resolved_placement: primary_resolved_placement,
        engine_id: primary_binding.engine_id().clone(),
        registration_identity: primary_binding.registration_identity(),
        context_identity: primary_binding.context_identity(),
        hardware_class: primary_binding.hardware_class().clone(),
        compiler_options,
        resolved_planning: planning_key,
        prepare_options: primary_options_key,
        operation_bindings: bindings.into_boxed_slice(),
        operation_placements: placements.into_boxed_slice(),
        input_locations: input_locations.to_vec().into_boxed_slice(),
        extension_planning: extension_identities.into_boxed_slice(),
    });

    Ok(PreparationContext {
        root_identity,
        operation_dispatch: dispatch.into(),
        staging: Arc::clone(staging),
        root_location,
        operation_locations,
        transfer_reachability: Arc::new(transfer_reachability),
        extension_planning: extension_planning.into(),
    })
}

fn is_route_specific_prepare_error(error: &PrepareError) -> bool {
    matches!(
        error,
        PrepareError::NoInputIngress { .. } | PrepareError::MissingTransferProvider { .. }
    )
}

fn schedule_prepare_error(source: ScheduleBuildError) -> Arc<PrepareError> {
    match source {
        ScheduleBuildError::MissingTransferProvider {
            instruction_index,
            slot,
            destination_storage_class,
            available_storage_classes,
        } => Arc::new(PrepareError::MissingTransferProvider {
            instruction_index,
            value_slot: slot,
            destination_storage_class,
            available_storage_classes,
        }),
        source => Arc::new(PrepareError::Engine {
            source: Arc::new(source),
        }),
    }
}

// INVARIANT: cross-storage dispatch uses the same immutable preparation inputs
// as same-storage dispatch and differs only in candidate generation.
#[allow(clippy::too_many_arguments)]
fn build_cross_storage_operation_dispatch(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    frozen: &FrozenProgram,
    compiler_options: CompilerOptions,
    staging: &Arc<ExecProgram>,
    candidates: &[super::EngineSnapshotView<'_>],
    signature: &InputSignature,
    options: &PrepareOptions,
    missing_extension_family: &mut Option<ExtensionFamilyId>,
) -> PreparedProgramResult<Option<PreparationContext>> {
    let mut dispatch_candidates = Vec::with_capacity(frozen.program.operations().len());
    for operation in frozen.program.operations() {
        let selected = operation_dispatch_candidates_any_storage(
            runtime,
            snapshot,
            candidates,
            operation,
            options,
            missing_extension_family,
        )?;
        if selected.is_empty() {
            return Ok(None);
        }
        dispatch_candidates.push(selected);
    }

    if dispatch_candidates.is_empty() {
        return Ok(None);
    }

    search_dispatch_preferences(&dispatch_candidates, candidates, |selected| {
        build_preparation_context(
            runtime,
            snapshot,
            frozen,
            compiler_options,
            staging,
            candidates,
            signature,
            options,
            selected,
            None,
        )
    })
    .map(Some)
}

fn search_dispatch_preferences(
    dispatch_candidates: &[Vec<SelectedOperationDispatch>],
    engine_preferences: &[super::EngineSnapshotView<'_>],
    mut build: impl FnMut(Vec<SelectedOperationDispatch>) -> PreparedProgramResult<PreparationContext>,
) -> PreparedProgramResult<PreparationContext> {
    const MAX_DISPATCH_SEARCH_ATTEMPTS: usize = 4_096;

    let mut attempted = HashSet::<Vec<(EngineId, StorageClass)>>::new();
    let mut budget = DispatchSearchBudget::new(MAX_DISPATCH_SEARCH_ATTEMPTS);
    let mut last_route_error = None;
    for preference in engine_preferences {
        let selected = dispatch_candidates
            .iter()
            .map(|candidates| {
                candidates
                    .iter()
                    .find(|candidate| {
                        candidate.dispatch.binding.engine_id() == preference.engine_id()
                    })
                    .unwrap_or(&candidates[0])
                    .clone()
            })
            .collect::<Vec<_>>();
        let key = selected
            .iter()
            .map(|candidate| {
                (
                    candidate.dispatch.binding.engine_id().clone(),
                    candidate
                        .dispatch
                        .resolved_placement
                        .storage_class()
                        .clone(),
                )
            })
            .collect::<Vec<_>>();
        if !attempted.insert(key) {
            continue;
        }
        if !budget.try_attempt() {
            return Err(dispatch_search_budget_error(&budget));
        }

        match build(selected) {
            Ok(context) => return Ok(context),
            Err(error) if is_route_specific_prepare_error(error.as_ref()) => {
                last_route_error = Some(error);
            }
            Err(error) => return Err(error),
        }
    }

    // INVARIANT: arbitrary transfer graphs form a general constraint-satisfaction
    // problem. The hard attempt budget keeps an unsatisfiable graph from turning
    // complete Cartesian fallback into unbounded preparation work.
    let mut indices = vec![0_usize; dispatch_candidates.len()];
    loop {
        let selected = dispatch_candidates
            .iter()
            .zip(&indices)
            .map(|(candidates, &index)| candidates[index].clone())
            .collect::<Vec<_>>();
        let key = selected
            .iter()
            .map(|candidate| {
                (
                    candidate.dispatch.binding.engine_id().clone(),
                    candidate
                        .dispatch
                        .resolved_placement
                        .storage_class()
                        .clone(),
                )
            })
            .collect::<Vec<_>>();
        if attempted.insert(key) {
            if !budget.try_attempt() {
                return Err(dispatch_search_budget_error(&budget));
            }
            match build(selected) {
                Ok(context) => return Ok(context),
                Err(error) if is_route_specific_prepare_error(error.as_ref()) => {
                    last_route_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }

        let mut advanced = false;
        for operation_index in (0..indices.len()).rev() {
            indices[operation_index] += 1;
            if indices[operation_index] < dispatch_candidates[operation_index].len() {
                advanced = true;
                break;
            }
            indices[operation_index] = 0;
        }
        if !advanced {
            break;
        }
    }

    Err(last_route_error.unwrap_or_else(|| {
        Arc::new(PrepareError::NoEligibleEngine {
            constraint: ProgramPlacementConstraint::any(),
        })
    }))
}

#[derive(Debug)]
struct DispatchSearchBudget {
    attempts: usize,
    limit: usize,
}

impl DispatchSearchBudget {
    fn new(limit: usize) -> Self {
        Self { attempts: 0, limit }
    }

    fn try_attempt(&mut self) -> bool {
        if self.attempts >= self.limit {
            return false;
        }
        self.attempts += 1;
        true
    }

    fn attempts(&self) -> usize {
        self.attempts
    }

    fn limit(&self) -> usize {
        self.limit
    }
}

fn dispatch_search_budget_error(budget: &DispatchSearchBudget) -> Arc<PrepareError> {
    Arc::new(PrepareError::DispatchSearchBudgetExceeded {
        attempts: budget.attempts(),
        limit: budget.limit(),
    })
}

fn operation_dispatch_candidates(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    candidates: &[super::EngineSnapshotView<'_>],
    operation: SemanticOperationView<'_>,
    storage_class: &StorageClass,
    options: &PrepareOptions,
    missing_extension_family: &mut Option<ExtensionFamilyId>,
) -> PreparedProgramResult<Vec<SelectedOperationDispatch>> {
    let mut selected_dispatches = Vec::new();
    for engine in candidates {
        if !engine.storage_classes().contains(storage_class) {
            continue;
        }
        let Some(provider) = provider_for_operation(snapshot, engine, operation)? else {
            if let SemanticOpRef::Extension(extension) = operation.op() {
                if !snapshot.has_extension_family(extension.family_id()) {
                    missing_extension_family.get_or_insert(extension.family_id());
                }
            }
            continue;
        };
        let resolved_placement =
            ResolvedProgramPlacement::new(engine.engine_id().clone(), storage_class.clone());
        let planning = ResolvedPlanningConfig::resolve(
            snapshot.execution_policy(),
            options,
            engine.hardware_class().clone(),
        );
        let prepare_options_key = PrepareOptionsKey::from_resolved(
            resolved_placement.clone(),
            planning.hard_workspace_limit_bytes(),
            planning.planning_seed(),
        );
        let (binding, extension_identity, extension_config) = match operation.op() {
            SemanticOpRef::Core(_) => (
                PreparedOperationBinding::new(
                    runtime.id(),
                    snapshot.epoch(),
                    engine.engine_id().clone(),
                    engine.registration_identity(),
                    engine.context_identity(),
                    engine.hardware_class().clone(),
                ),
                None,
                None,
            ),
            SemanticOpRef::Extension(extension) => {
                let Some(slot) = snapshot
                    .extension_slot_for_preparation(extension.family_id(), engine.engine_id())
                else {
                    continue;
                };
                let config = Arc::clone(slot.config());
                (
                    PreparedOperationBinding::new(
                        runtime.id(),
                        snapshot.epoch(),
                        slot.engine_id().clone(),
                        slot.registration_identity(),
                        slot.context_identity(),
                        engine.hardware_class().clone(),
                    ),
                    Some(ExtensionPlanningIdentity {
                        module_id: slot.module_id().clone(),
                        family_id: slot.family_id(),
                        engine_id: slot.engine_id().clone(),
                        payload_fingerprint: extension_config_payload_fingerprint(config.as_ref()),
                        config: Arc::clone(&config),
                    }),
                    Some(config),
                )
            }
        };
        selected_dispatches.push(SelectedOperationDispatch {
            dispatch: OperationDispatch {
                binding,
                resolved_placement,
                planning,
                prepare_options_key,
                provider,
            },
            extension_identity,
            extension_config,
        });
    }
    Ok(selected_dispatches)
}

fn operation_dispatch_candidates_any_storage(
    runtime: &Runtime,
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    candidates: &[super::EngineSnapshotView<'_>],
    operation: SemanticOperationView<'_>,
    options: &PrepareOptions,
    missing_extension_family: &mut Option<ExtensionFamilyId>,
) -> PreparedProgramResult<Vec<SelectedOperationDispatch>> {
    let mut selected = Vec::new();
    for engine in candidates {
        let storage_class = engine.default_storage_class().clone();
        selected.extend(operation_dispatch_candidates(
            runtime,
            snapshot,
            std::slice::from_ref(engine),
            operation,
            &storage_class,
            options,
            missing_extension_family,
        )?);
    }
    for engine in candidates {
        for storage_class in engine.storage_classes() {
            if storage_class == engine.default_storage_class() {
                continue;
            }
            selected.extend(operation_dispatch_candidates(
                runtime,
                snapshot,
                std::slice::from_ref(engine),
                operation,
                storage_class,
                options,
                missing_extension_family,
            )?);
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod dispatch_search_budget_tests {
    use super::DispatchSearchBudget;

    #[test]
    fn dispatch_search_budget_stops_before_unbounded_cartesian_enumeration() {
        let mut budget = DispatchSearchBudget::new(3);

        assert!(budget.try_attempt());
        assert!(budget.try_attempt());
        assert!(budget.try_attempt());
        assert!(!budget.try_attempt());
        assert_eq!(budget.attempts(), 3);
        assert_eq!(budget.limit(), 3);
    }
}

fn provider_for_operation(
    snapshot: &Arc<super::RuntimeConfigSnapshot>,
    engine: &super::EngineSnapshotView<'_>,
    operation: SemanticOperationView<'_>,
) -> PreparedProgramResult<Option<OperationProvider>> {
    Ok(match operation.op() {
        SemanticOpRef::Core(op) => match required_core_capability(op) {
            CoreCapabilityKind::Elementwise => engine
                .capabilities()
                .elementwise()
                .map(|provider| OperationProvider::Elementwise(Arc::clone(provider))),
            CoreCapabilityKind::Reduction => engine
                .capabilities()
                .reduction()
                .map(|provider| OperationProvider::Reduction(Arc::clone(provider))),
            CoreCapabilityKind::Indexing => engine
                .capabilities()
                .indexing()
                .map(|provider| OperationProvider::Indexing(Arc::clone(provider))),
            CoreCapabilityKind::DotGeneral => engine
                .capabilities()
                .dot_general()
                .map(|provider| OperationProvider::DotGeneral(Arc::clone(provider))),
            CoreCapabilityKind::Layout => engine
                .capabilities()
                .layout()
                .map(|provider| OperationProvider::Layout(Arc::clone(provider))),
        },
        SemanticOpRef::Extension(extension) => snapshot
            .extension_slot_for_preparation(extension.family_id(), engine.engine_id())
            .map(|slot| OperationProvider::Extension {
                engine: Arc::clone(slot.engine()),
                config: Arc::clone(slot.config()),
            }),
    })
}

fn required_core_capability(op: &CoreSemanticOp) -> CoreCapabilityKind {
    match op {
        CoreSemanticOp::DotGeneral { .. } => CoreCapabilityKind::DotGeneral,
        CoreSemanticOp::Transpose { .. }
        | CoreSemanticOp::Reshape { .. }
        | CoreSemanticOp::BroadcastInDim { .. }
        | CoreSemanticOp::Convert { .. }
        | CoreSemanticOp::Constant { .. }
        | CoreSemanticOp::ExtractDiag { .. }
        | CoreSemanticOp::EmbedDiag { .. }
        | CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. } => CoreCapabilityKind::Layout,
        CoreSemanticOp::ReduceSum { .. }
        | CoreSemanticOp::ReduceSumSquares { .. }
        | CoreSemanticOp::ReduceProd { .. }
        | CoreSemanticOp::ReduceMax { .. }
        | CoreSemanticOp::ReduceMin { .. } => CoreCapabilityKind::Reduction,
        CoreSemanticOp::Gather(_)
        | CoreSemanticOp::GatherDynamicSliceSizes { .. }
        | CoreSemanticOp::Scatter(_)
        | CoreSemanticOp::Slice(_)
        | CoreSemanticOp::DynamicSlice { .. }
        | CoreSemanticOp::DynamicUpdateSlice
        | CoreSemanticOp::Pad(_)
        | CoreSemanticOp::Concatenate { .. }
        | CoreSemanticOp::Reverse { .. }
        | CoreSemanticOp::ShapeOf { .. }
        | CoreSemanticOp::DynamicTruncate { .. }
        | CoreSemanticOp::PadToMatch { .. } => CoreCapabilityKind::Indexing,
        CoreSemanticOp::Add
        | CoreSemanticOp::Sub
        | CoreSemanticOp::Mul
        | CoreSemanticOp::Neg
        | CoreSemanticOp::Conj
        | CoreSemanticOp::Div
        | CoreSemanticOp::Rem
        | CoreSemanticOp::Abs
        | CoreSemanticOp::Sign
        | CoreSemanticOp::Maximum
        | CoreSemanticOp::Minimum
        | CoreSemanticOp::Compare(_)
        | CoreSemanticOp::Select
        | CoreSemanticOp::Clamp
        | CoreSemanticOp::Exp
        | CoreSemanticOp::Log
        | CoreSemanticOp::Sin
        | CoreSemanticOp::Cos
        | CoreSemanticOp::Tanh
        | CoreSemanticOp::Sqrt
        | CoreSemanticOp::Rsqrt
        | CoreSemanticOp::Pow
        | CoreSemanticOp::Expm1
        | CoreSemanticOp::Log1p => CoreCapabilityKind::Elementwise,
    }
}

fn produce_prepared_entry(
    context: &PreparationContext,
    key: &PreparedEntryKey,
    signature: &InputSignature,
) -> CacheProduced<PreparedEntryKey, PreparedProgram> {
    match prepare_entry(context, key, signature) {
        Ok(ProducedEntry::Ready { value, root }) => CacheProduced::Ready {
            value,
            shared: Some(root.shared_retention()),
        },
        Ok(ProducedEntry::Redirect { requirements, root }) => CacheProduced::Redirect {
            requirements,
            shared: Some(root.shared_retention()),
        },
        Err(error) if is_deterministic_prepare_error(error.as_ref()) => {
            let shared = key.root.prepared_root().map(|root| root.shared_retention());
            CacheProduced::FailedDeterministic { error, shared }
        }
        Err(error) => CacheProduced::FailedTransient(error),
    }
}

enum ProducedEntry {
    Ready {
        value: Arc<PreparedProgram>,
        root: Arc<PreparedProgramRoot>,
    },
    Redirect {
        requirements: SpecializationRequirements,
        root: Arc<PreparedProgramRoot>,
    },
}

fn prepare_entry(
    context: &PreparationContext,
    key: &PreparedEntryKey,
    signature: &InputSignature,
) -> PreparedProgramResult<ProducedEntry> {
    let root = root_for_key(context, &key.root)?;
    let mut operations = Vec::with_capacity(context.operation_dispatch.len());
    for (operation, dispatch) in root
        .semantic
        .operations()
        .zip(context.operation_dispatch.iter())
    {
        match prepare_operation(operation, dispatch, signature, &key.specialization)? {
            super::PrepareCapability::Prepared(operation) => {
                validate_prepared_operation(
                    operation.operation().as_ref(),
                    &dispatch.binding,
                    &key.specialization,
                )?;
                operations.push(operation);
            }
            super::PrepareCapability::NeedsSpecialization(next) => {
                validate_widening(&key.requirements, &next)?;
                project_requirements(&next, signature)?;
                return Ok(ProducedEntry::Redirect {
                    requirements: next,
                    root: Arc::clone(&root),
                });
            }
            super::PrepareCapability::Unsupported(reason) => {
                return Err(Arc::new(PrepareError::Unsupported { reason }));
            }
        }
    }
    Ok(ProducedEntry::Ready {
        value: Arc::new(PreparedProgram::new(
            Arc::clone(&root),
            key.specialization.clone(),
            operations.into_boxed_slice(),
        )),
        root,
    })
}

fn root_for_key(
    context: &PreparationContext,
    root_key: &PreparedRootKey,
) -> PreparedProgramResult<Arc<PreparedProgramRoot>> {
    if let Some(root) = root_key.prepared_root() {
        return Ok(root);
    }
    let identity = match root_key {
        PreparedRootKey::Identity(identity) => Arc::clone(identity),
        PreparedRootKey::Prepared(_) => unreachable!(),
    };
    PreparedProgramRoot::new(
        identity,
        Arc::clone(&context.staging),
        Arc::clone(&context.extension_planning),
        context.root_location.clone(),
        &context.operation_locations,
        &context.transfer_reachability,
    )
    .map(Arc::new)
    .map_err(|source| match source {
        ScheduleBuildError::MissingTransferProvider {
            instruction_index,
            slot,
            destination_storage_class,
            available_storage_classes,
        } => Arc::new(PrepareError::MissingTransferProvider {
            instruction_index,
            value_slot: slot,
            destination_storage_class,
            available_storage_classes,
        }),
        source => Arc::new(PrepareError::Engine {
            source: Arc::new(source),
        }),
    })
}

fn prepare_operation(
    operation: SemanticOperationView<'_>,
    dispatch: &OperationDispatch,
    signature: &InputSignature,
    specialization: &SpecializationProjection,
) -> Result<super::PrepareCapability, PrepareError> {
    let core_context = CorePrepareContext::new(
        &dispatch.binding,
        signature,
        &dispatch.resolved_placement,
        &dispatch.planning,
        &dispatch.prepare_options_key,
        specialization,
    );
    match (&dispatch.provider, operation.op()) {
        (OperationProvider::Elementwise(provider), SemanticOpRef::Core(_)) => {
            provider.prepare(ElementwisePrepareRequest::new(operation, &core_context))
        }
        (OperationProvider::Reduction(provider), SemanticOpRef::Core(_)) => {
            provider.prepare(ReductionPrepareRequest::new(operation, &core_context))
        }
        (OperationProvider::Indexing(provider), SemanticOpRef::Core(_)) => {
            provider.prepare(IndexingPrepareRequest::new(operation, &core_context))
        }
        (OperationProvider::DotGeneral(provider), SemanticOpRef::Core(_)) => {
            provider.prepare(DotGeneralPrepareRequest::new(operation, &core_context))
        }
        (OperationProvider::Layout(provider), SemanticOpRef::Core(_)) => {
            provider.prepare(LayoutPrepareRequest::new(operation, &core_context))
        }
        (OperationProvider::Extension { engine, config }, SemanticOpRef::Extension(extension)) => {
            engine.prepare(ExtensionPrepareRequest::new(
                extension,
                &dispatch.binding,
                &dispatch.resolved_placement,
                dispatch.binding.hardware_class(),
                &dispatch.planning,
                config.as_ref(),
                signature,
                &dispatch.prepare_options_key,
                specialization,
            ))
        }
        (provider, SemanticOpRef::Core(op)) => Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: required_core_capability(op),
                operation: provider_family_name(provider),
            },
        }),
        (_, SemanticOpRef::Extension(extension)) => Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::Elementwise,
                operation: extension.family_id(),
            },
        }),
    }
}

fn provider_family_name(provider: &OperationProvider) -> &'static str {
    match provider {
        OperationProvider::Elementwise(_) => "elementwise",
        OperationProvider::Reduction(_) => "reduction",
        OperationProvider::Indexing(_) => "indexing",
        OperationProvider::DotGeneral(_) => "dot_general",
        OperationProvider::Layout(_) => "layout",
        OperationProvider::Extension { .. } => "extension",
    }
}

fn validate_prepared_operation(
    operation: &dyn super::PreparedOperation,
    expected_binding: &PreparedOperationBinding,
    expected_projection: &SpecializationProjection,
) -> PreparedProgramResult<()> {
    if operation.binding() != expected_binding {
        return Err(Arc::new(PrepareError::ProviderContract {
            source: ProviderContractError::BindingMismatch {
                expected: Box::new(expected_binding.clone()),
                actual: Box::new(operation.binding().clone()),
            },
        }));
    }
    if operation.specialization() != expected_projection {
        return Err(Arc::new(PrepareError::ProviderContract {
            source: ProviderContractError::ProjectionMismatch {
                expected: Box::new(expected_projection.clone()),
                actual: Box::new(operation.specialization().clone()),
            },
        }));
    }
    Ok(())
}

fn validate_widening(
    previous: &SpecializationRequirements,
    next: &SpecializationRequirements,
) -> PreparedProgramResult<()> {
    if previous.strictly_widens(next) {
        Ok(())
    } else {
        Err(Arc::new(PrepareError::ProviderContract {
            source: ProviderContractError::NonWideningSpecialization {
                previous: Box::new(previous.clone()),
                next: Box::new(next.clone()),
            },
        }))
    }
}

fn project_requirements(
    requirements: &SpecializationRequirements,
    signature: &InputSignature,
) -> PreparedProgramResult<SpecializationProjection> {
    requirements
        .project(signature)
        .map_err(|error| match error {
            PrepareError::Specialization { source } => Arc::new(PrepareError::ProviderContract {
                source: ProviderContractError::InvalidSpecialization { source },
            }),
            other => Arc::new(other),
        })
}

fn validate_signature_arity(
    frozen: &FrozenProgram,
    signature: &InputSignature,
) -> PreparedProgramResult<()> {
    let expected = frozen.program.inputs().len();
    let actual = signature.entries().len();
    if expected == actual {
        Ok(())
    } else {
        Err(Arc::new(PrepareError::Specialization {
            source: super::SpecializationError::WrongInputCount { expected, actual },
        }))
    }
}

fn validate_semantic_shape_guards(
    program: &SemanticProgram,
    signature: &InputSignature,
) -> PreparedProgramResult<()> {
    let input_shapes: Vec<_> = signature
        .entries()
        .iter()
        .map(|entry| entry.shape())
        .collect();
    for guard in program.shape_guards() {
        let lhs = evaluate_dim_expr(guard.lhs(), &input_shapes)?;
        let rhs = evaluate_dim_expr(guard.rhs(), &input_shapes)?;
        let satisfied = match guard.relation() {
            ProgramShapeRelation::Equal => lhs == rhs,
            ProgramShapeRelation::LessEqual => lhs <= rhs,
            ProgramShapeRelation::GreaterEqual => lhs >= rhs,
        };
        if !satisfied {
            return Err(Arc::new(PrepareError::Engine {
                source: Arc::new(SemanticShapeGuardViolation {
                    relation: guard.relation(),
                    lhs,
                    rhs,
                }),
            }));
        }
    }
    Ok(())
}

fn evaluate_dim_expr(
    expression: &DimExpr,
    input_shapes: &[&[usize]],
) -> PreparedProgramResult<usize> {
    expression.eval(input_shapes).map_err(|source| {
        Arc::new(PrepareError::Engine {
            source: Arc::new(SemanticShapeGuardEvaluation { source }),
        })
    })
}

#[derive(Debug)]
struct SemanticShapeGuardEvaluation {
    source: DimExprEvalError,
}

impl fmt::Display for SemanticShapeGuardEvaluation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "semantic shape guard evaluation failed: {}",
            self.source
        )
    }
}

impl StdError for SemanticShapeGuardEvaluation {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug)]
struct SemanticShapeGuardViolation {
    relation: ProgramShapeRelation,
    lhs: usize,
    rhs: usize,
}

impl fmt::Display for SemanticShapeGuardViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "semantic shape guard {:?} failed: {} vs {}",
            self.relation, self.lhs, self.rhs
        )
    }
}

impl StdError for SemanticShapeGuardViolation {}

fn is_deterministic_prepare_error(error: &PrepareError) -> bool {
    matches!(
        error,
        PrepareError::Unsupported { .. }
            | PrepareError::DeterminismUnsupported { .. }
            | PrepareError::ProviderContract { .. }
    )
}

fn missing_shared_root_error() -> Arc<PrepareError> {
    Arc::new(PrepareError::Engine {
        source: Arc::new(MissingSharedRoot),
    })
}

#[derive(Debug)]
struct MissingSharedRoot;

impl fmt::Display for MissingSharedRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("prepared redirect did not retain its prepared root")
    }
}

impl StdError for MissingSharedRoot {}

fn compact_digest_half(prefix: u64, key: &PreparedEntryKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(prefix);
    hash_root_identity_for_digest(key.root.identity(), &mut hasher);
    key.requirements.hash(&mut hasher);
    key.specialization.hash(&mut hasher);
    hasher.finish()
}

fn hash_root_identity_for_digest(identity: &PreparedRootIdentity, state: &mut impl Hasher) {
    identity.semantic_fingerprint.hash(state);
    identity.runtime_id.hash(state);
    identity.epoch.hash(state);
    identity.resolved_placement.hash(state);
    identity.engine_id.hash(state);
    identity.registration_identity.hash(state);
    identity.context_identity.hash(state);
    identity.hardware_class.hash(state);
    identity.compiler_options.hash(state);
    identity.resolved_planning.hash(state);
    identity.prepare_options.hash(state);
    identity.operation_bindings.hash(state);
    identity.operation_placements.hash(state);
    identity.input_locations.hash(state);
    for extension in &identity.extension_planning {
        extension.module_id.hash(state);
        extension.family_id.hash(state);
        extension.engine_id.hash(state);
        extension.payload_fingerprint.hash(state);
    }
}

fn root_identity_exact_eq(left: &PreparedRootIdentity, right: &PreparedRootIdentity) -> bool {
    left.semantic_fingerprint == right.semantic_fingerprint
        && left.runtime_id == right.runtime_id
        && left.epoch == right.epoch
        && left.resolved_placement == right.resolved_placement
        && left.engine_id == right.engine_id
        && left.registration_identity == right.registration_identity
        && left.context_identity == right.context_identity
        && left.hardware_class == right.hardware_class
        && left.compiler_options == right.compiler_options
        && left.resolved_planning == right.resolved_planning
        && left.prepare_options == right.prepare_options
        && left.operation_bindings == right.operation_bindings
        && left.operation_placements == right.operation_placements
        && left.input_locations == right.input_locations
        && left.semantic.semantic_eq(&right.semantic)
        && extension_planning_exact_eq(&left.extension_planning, &right.extension_planning)
}

fn extension_planning_exact_eq(
    left: &[ExtensionPlanningIdentity],
    right: &[ExtensionPlanningIdentity],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.module_id == right.module_id
                && left.family_id == right.family_id
                && left.engine_id == right.engine_id
                && left.payload_fingerprint == right.payload_fingerprint
                && left.config.payload_eq(right.config.as_ref())
        })
}

fn extension_config_payload_fingerprint(config: &dyn ExtensionPlanningConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.payload_hash(&mut DynHasherProxy::new(&mut hasher));
    hasher.finish()
}

struct DynHasherProxy<'a, H: Hasher + ?Sized> {
    inner: &'a mut H,
}

impl<'a, H: Hasher + ?Sized> DynHasherProxy<'a, H> {
    fn new(inner: &'a mut H) -> Self {
        Self { inner }
    }
}

impl<H: Hasher + ?Sized> Hasher for DynHasherProxy<'_, H> {
    fn finish(&self) -> u64 {
        self.inner.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.inner.write(bytes);
    }
}

fn prepared_root_identity_key_retained_bytes(identity: &PreparedRootIdentity) -> Option<usize> {
    checked_sum([
        size_of::<PreparedRootIdentity>(),
        identity
            .operation_bindings
            .len()
            .checked_mul(size_of::<PreparedOperationBinding>())?,
        identity
            .operation_placements
            .len()
            .checked_mul(size_of::<ResolvedProgramPlacement>())?,
        identity
            .input_locations
            .len()
            .checked_mul(size_of::<ExecutionLocation>())?,
        identity
            .extension_planning
            .len()
            .checked_mul(size_of::<ExtensionPlanningIdentity>())?,
    ])
}

fn prepared_program_root_retained_bytes(
    identity: &PreparedRootIdentity,
    semantic: &SemanticProgram,
    staging: &ExecProgram,
    schedule: &ScheduledGraph,
    extension_planning: &[Arc<dyn ExtensionPlanningConfig>],
) -> Option<usize> {
    checked_sum([
        size_of::<PreparedProgramRoot>(),
        prepared_root_identity_key_retained_bytes(identity)?,
        semantic.logical_retained_bytes()?,
        exec_program_retained_bytes(staging)?,
        schedule.retained_bytes()?,
        extension_planning
            .len()
            .checked_mul(size_of::<Arc<dyn ExtensionPlanningConfig>>())?,
        checked_sum_options(
            extension_planning
                .iter()
                .map(|config| Some(config.retained_bytes())),
        )?,
    ])
}

fn exec_program_retained_bytes(program: &ExecProgram) -> Option<usize> {
    checked_sum([
        size_of::<ExecProgram>(),
        program
            .instructions
            .len()
            .checked_mul(size_of::<ExecInstruction>())?,
        checked_sum_options(
            program
                .instructions
                .iter()
                .map(exec_instruction_retained_bytes),
        )?,
        program.input_slots.len().checked_mul(size_of::<usize>())?,
        program.output_slots.len().checked_mul(size_of::<usize>())?,
        program
            .shape_guards
            .len()
            .checked_mul(size_of::<crate::ShapeGuard>())?,
        checked_sum_options(
            program
                .shape_guards
                .iter()
                .map(crate::ShapeGuard::logical_retained_bytes),
        )?,
    ])
}

fn exec_instruction_retained_bytes(instruction: &ExecInstruction) -> Option<usize> {
    checked_sum([
        exec_op_retained_bytes(&instruction.op)?,
        instruction
            .input_slots
            .len()
            .checked_mul(size_of::<usize>())?,
        instruction
            .output_slots
            .len()
            .checked_mul(size_of::<usize>())?,
        smallvec_outer_bytes::<Vec<DimExpr>>(
            instruction.output_shapes.spilled(),
            instruction.output_shapes.len(),
        )?,
        checked_sum_options(
            instruction
                .output_shapes
                .iter()
                .map(|shape| dim_expr_vec_bytes(shape)),
        )?,
        smallvec_outer_bytes::<Vec<ShapeExtent<DimExpr>>>(
            instruction.output_extents.spilled(),
            instruction.output_extents.len(),
        )?,
        checked_sum_options(
            instruction
                .output_extents
                .iter()
                .map(|shape| shape_extent_vec_bytes(shape)),
        )?,
        instruction.last_use.len().checked_mul(size_of::<bool>())?,
    ])
}

fn exec_op_retained_bytes(op: &ExecOp) -> Option<usize> {
    match op {
        ExecOp::Transpose { perm } => vec_bytes::<usize>(perm.len()),
        ExecOp::Reshape { shape } => dim_expr_vec_bytes(shape),
        ExecOp::BroadcastInDim { shape, dims } => {
            checked_sum([dim_expr_vec_bytes(shape)?, vec_bytes::<usize>(dims.len())?])
        }
        ExecOp::Constant { bytes, .. } => vec_bytes::<u8>(bytes.len()),
        ExecOp::DotGeneral(config) | ExecOp::DotGeneralWithConj { config, .. } => {
            dot_general_config_retained_bytes(config)
        }
        ExecOp::ReduceSum { axes }
        | ExecOp::ReduceSumSquares { axes }
        | ExecOp::Reverse { axes }
        | ExecOp::ReduceProd { axes }
        | ExecOp::ReduceMax { axes }
        | ExecOp::ReduceMin { axes } => vec_bytes::<usize>(axes.len()),
        ExecOp::Gather(config) => gather_config_retained_bytes(config),
        ExecOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            ..
        } => checked_sum([
            vec_bytes::<usize>(offset_dims.len())?,
            vec_bytes::<usize>(collapsed_slice_dims.len())?,
            vec_bytes::<usize>(start_index_map.len())?,
            dim_expr_vec_bytes(slice_sizes)?,
        ]),
        ExecOp::Scatter(config) => scatter_config_retained_bytes(config),
        ExecOp::Slice(config) => slice_config_retained_bytes(config),
        ExecOp::DynamicSlice { slice_sizes } => vec_bytes::<usize>(slice_sizes.len()),
        ExecOp::Pad(config) => pad_config_retained_bytes(config),
        ExecOp::Convert { .. }
        | ExecOp::ExtractDiag { .. }
        | ExecOp::EmbedDiag { .. }
        | ExecOp::Tril { .. }
        | ExecOp::Triu { .. }
        | ExecOp::Add
        | ExecOp::Subtract
        | ExecOp::Multiply
        | ExecOp::Negate
        | ExecOp::Conj
        | ExecOp::Divide
        | ExecOp::Remainder
        | ExecOp::Abs
        | ExecOp::Sign
        | ExecOp::Maximum
        | ExecOp::Minimum
        | ExecOp::Compare(_)
        | ExecOp::Select
        | ExecOp::Clamp
        | ExecOp::Exp
        | ExecOp::Log
        | ExecOp::Sin
        | ExecOp::Cos
        | ExecOp::Tanh
        | ExecOp::Sqrt
        | ExecOp::Rsqrt
        | ExecOp::Pow
        | ExecOp::Expm1
        | ExecOp::Log1p
        | ExecOp::DynamicUpdateSlice
        | ExecOp::Concatenate { .. }
        | ExecOp::ShapeOf { .. }
        | ExecOp::DynamicTruncate { .. }
        | ExecOp::PadToMatch { .. }
        | ExecOp::Extension(_) => Some(0),
    }
}

fn dot_general_config_retained_bytes(config: &DotGeneralConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.lhs_contracting_dims.len())?,
        vec_bytes::<usize>(config.rhs_contracting_dims.len())?,
        vec_bytes::<usize>(config.lhs_batch_dims.len())?,
        vec_bytes::<usize>(config.rhs_batch_dims.len())?,
    ])
}

fn gather_config_retained_bytes(config: &GatherConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.offset_dims.len())?,
        vec_bytes::<usize>(config.collapsed_slice_dims.len())?,
        vec_bytes::<usize>(config.start_index_map.len())?,
        vec_bytes::<usize>(config.slice_sizes.len())?,
    ])
}

fn scatter_config_retained_bytes(config: &ScatterConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.update_window_dims.len())?,
        vec_bytes::<usize>(config.inserted_window_dims.len())?,
        vec_bytes::<usize>(config.scatter_dims_to_operand_dims.len())?,
    ])
}

fn slice_config_retained_bytes(config: &SliceConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<usize>(config.starts.len())?,
        vec_bytes::<usize>(config.limits.len())?,
        vec_bytes::<usize>(config.strides.len())?,
    ])
}

fn pad_config_retained_bytes(config: &PadConfig) -> Option<usize> {
    checked_sum([
        vec_bytes::<i64>(config.edge_padding_low.len())?,
        vec_bytes::<i64>(config.edge_padding_high.len())?,
        vec_bytes::<i64>(config.interior_padding.len())?,
    ])
}

fn dim_expr_vec_bytes(values: &[DimExpr]) -> Option<usize> {
    checked_sum([
        vec_bytes::<DimExpr>(values.len())?,
        checked_sum_options(values.iter().map(dim_expr_retained_bytes))?,
    ])
}

fn shape_extent_vec_bytes(values: &[ShapeExtent<DimExpr>]) -> Option<usize> {
    checked_sum([
        vec_bytes::<ShapeExtent<DimExpr>>(values.len())?,
        checked_sum_options(values.iter().map(shape_extent_retained_bytes))?,
    ])
}

fn shape_extent_retained_bytes(extent: &ShapeExtent<DimExpr>) -> Option<usize> {
    match extent {
        ShapeExtent::Exact(expression) | ShapeExtent::UpperBound(expression) => {
            dim_expr_retained_bytes(expression)
        }
        ShapeExtent::Unknown => Some(0),
    }
}

fn dim_expr_retained_bytes(expression: &DimExpr) -> Option<usize> {
    match expression {
        DimExpr::Const(_) | DimExpr::InputDim { .. } => Some(0),
        DimExpr::Add(left, right)
        | DimExpr::Sub(left, right)
        | DimExpr::Mul(left, right)
        | DimExpr::FloorDiv(left, right)
        | DimExpr::Min(left, right)
        | DimExpr::Max(left, right) => checked_sum([
            2usize.checked_mul(size_of::<DimExpr>())?,
            dim_expr_retained_bytes(left)?,
            dim_expr_retained_bytes(right)?,
        ]),
    }
}

fn specialization_requirements_retained_bytes(
    requirements: &SpecializationRequirements,
) -> Option<usize> {
    checked_sum([
        size_of::<SpecializationRequirements>(),
        requirements
            .inputs()
            .len()
            .checked_mul(size_of::<InputSpecializationRequirements>())?,
        checked_sum_options(
            requirements
                .inputs()
                .iter()
                .map(input_requirements_retained_bytes),
        )?,
    ])
}

fn input_requirements_retained_bytes(
    requirements: &InputSpecializationRequirements,
) -> Option<usize> {
    vec_bytes::<u32>(requirements.concrete_dimensions().len())
}

fn specialization_projection_retained_bytes(
    projection: &SpecializationProjection,
) -> Option<usize> {
    checked_sum([
        size_of::<SpecializationProjection>(),
        projection
            .inputs()
            .len()
            .checked_mul(size_of::<super::InputSpecializationProjection>())?,
        checked_sum_options(
            projection
                .inputs()
                .iter()
                .map(input_projection_retained_bytes),
        )?,
    ])
}

fn input_projection_retained_bytes(
    projection: &super::InputSpecializationProjection,
) -> Option<usize> {
    checked_sum([
        projection
            .concrete_dimensions()
            .len()
            .checked_mul(size_of::<(u32, usize)>())?,
        match projection.layout() {
            Some(LayoutProjection::ExactStrides(strides)) if strides.spilled() => {
                strides.len().checked_mul(size_of::<isize>())?
            }
            _ => 0,
        },
    ])
}

fn specialization_retry_bound(
    requirements: &SpecializationRequirements,
    signature: &InputSignature,
) -> PreparedProgramResult<usize> {
    requirements
        .inputs()
        .iter()
        .zip(signature.entries())
        .try_fold(0usize, |total, (requirements, entry)| {
            let input =
                input_retry_bound(requirements, entry.shape().len(), entry.alignment_log2())?;
            total.checked_add(input)
        })
        .ok_or_else(accounting_prepare_error)
}

fn input_retry_bound(
    requirements: &InputSpecializationRequirements,
    rank: usize,
    alignment_log2: Option<u8>,
) -> Option<usize> {
    checked_sum([
        usize::from(!requirements.specializes_dtype()),
        usize::from(!requirements.specializes_rank()),
        rank.checked_sub(requirements.concrete_dimensions().len())?,
        placement_retry_edges(requirements.placement()),
        layout_retry_edges(requirements.layout()),
        alignment_retry_edges(requirements.alignment_log2(), alignment_log2)?,
    ])
}

fn placement_retry_edges(value: PlacementSpecialization) -> usize {
    match value {
        PlacementSpecialization::None => 2,
        PlacementSpecialization::StorageClass => 1,
        PlacementSpecialization::Device => 0,
    }
}

fn layout_retry_edges(value: super::LayoutSpecialization) -> usize {
    match value {
        super::LayoutSpecialization::None => 2,
        super::LayoutSpecialization::Class => 1,
        super::LayoutSpecialization::ExactStrides => 0,
    }
}

fn alignment_retry_edges(required: Option<u8>, actual: Option<u8>) -> Option<usize> {
    match (required, actual) {
        (None, None) => Some(0),
        (None, Some(_)) => Some(usize::BITS as usize),
        (Some(required), Some(actual)) => {
            let actual = usize::from(actual);
            let required = usize::from(required.min(actual as u8));
            (usize::BITS as usize - 1).checked_sub(required)
        }
        (Some(_), None) => Some(0),
    }
}

fn smallvec_outer_bytes<T>(spilled: bool, len: usize) -> Option<usize> {
    if spilled {
        len.checked_mul(size_of::<T>())
    } else {
        Some(0)
    }
}

fn vec_bytes<T>(len: usize) -> Option<usize> {
    len.checked_mul(size_of::<T>())
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

fn checked_sum_options(values: impl IntoIterator<Item = Option<usize>>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value?))
}

fn accounting_prepare_error() -> Arc<PrepareError> {
    Arc::new(PrepareError::Engine {
        source: Arc::new(PreparationAccountingOverflow),
    })
}

#[derive(Debug)]
struct PreparationAccountingOverflow;

impl fmt::Display for PreparationAccountingOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("prepared-program retained-byte accounting overflow")
    }
}

impl StdError for PreparationAccountingOverflow {}

fn runtime_cache_error_from_snapshot(source: RuntimeStateError) -> super::RuntimeCacheError {
    super::RuntimeCacheError::Aggregate {
        runtime: Some(source),
        owners: Box::new([]),
    }
}

pub(crate) fn cache_stats(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
) -> Result<super::RuntimeCacheStats, super::RuntimeCacheError> {
    let snapshot = runtime
        .snapshot()
        .map_err(runtime_cache_error_from_snapshot)?;
    caches.cache_stats(snapshot.cache_owners_for_runtime())
}

pub(crate) fn clear_caches(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
) -> Result<(), super::RuntimeCacheError> {
    let snapshot = runtime
        .snapshot()
        .map_err(runtime_cache_error_from_snapshot)?;
    caches.clear_caches(snapshot.cache_owners_for_runtime())
}
