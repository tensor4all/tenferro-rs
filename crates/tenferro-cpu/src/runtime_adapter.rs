use std::fmt;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use tenferro_runtime::program::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
use tenferro_runtime::{
    assemble_executable_engine_registration, CoreCapabilityBundle, CoreCapabilityKind,
    CorePrepareContext, DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest,
    ElementwiseRuntime, EngineId, EngineRegistration, EngineRegistrationMetadata,
    ExecutableEngineRegistrationConfig, ExecutionContextIdentity, HardwareClassId,
    IndexingPrepareRequest, IndexingRuntime, InputIngressContract, InputPlacementContract,
    InputSignature, InputSignatureContract, InputSpecializationProjection,
    InputSpecializationRequirements, LayoutPrepareRequest, LayoutProjection, LayoutRuntime,
    LayoutSpecialization, MemoryKind, PrepareCapability, PrepareError, PreparedOperation,
    PreparedOperationBinding, PreparedOperationPlan, ProviderContractError, ProviderDeviceIdentity,
    ProviderId, ReductionPrepareRequest, ReductionRuntime, ResidentOutputContract,
    RuntimeConfigError, RuntimeInputContract, SpecializationError, SpecializationProjection,
    SpecializationRequirements, StorageClass, TensorRead, UnsupportedReason,
};

use crate::CpuEngine;

/// Stateless CPU preparation witness: CPU preparation reads no engine state.
#[derive(Debug, Clone, Copy, Default)]
struct CpuPreparer;

const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const UNKNOWN_CORE_OPERATION: &str = "unknown-core-operation";

/// Return the canonical CPU runtime engine identifier.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in CPU engine identifier violates
/// runtime identifier validation.
pub fn runtime_engine_id() -> Result<EngineId, RuntimeConfigError> {
    EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)
}

/// The built-in CPU engine id, which is a valid identifier by construction.
pub(crate) fn default_runtime_engine_id() -> EngineId {
    match EngineId::new(CPU_ENGINE_ID) {
        Ok(id) => id,
        Err(_) => unreachable!("the built-in CPU engine id is a valid identifier"),
    }
}

/// Return the canonical CPU runtime hardware class.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in CPU hardware class violates
/// runtime identifier validation.
pub fn runtime_hardware_class() -> Result<HardwareClassId, RuntimeConfigError> {
    HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)
}

/// Build a runtime engine registration for a [`CpuEngine`].
///
/// The registration exposes the stateless CPU core preparation capabilities
/// and the engine's input contract.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if one of the built-in CPU runtime identifiers
/// violates runtime validation or if the registration is internally invalid.
pub fn runtime_engine_registration(
    engine: &CpuEngine,
) -> Result<EngineRegistration, RuntimeConfigError> {
    runtime_engine_registration_with_id(engine, runtime_engine_id()?)
}

/// Build a runtime engine registration for a [`CpuEngine`] under `engine_id`.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if `engine_id` or a built-in CPU runtime
/// identifier violates runtime validation, or if the registration is invalid.
pub fn runtime_engine_registration_with_id(
    engine: &CpuEngine,
    engine_id: EngineId,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let preparer = Arc::new(CpuPreparer);
    let elementwise: Arc<dyn ElementwiseRuntime> = preparer.clone();
    let reduction: Arc<dyn ReductionRuntime> = preparer.clone();
    let indexing: Arc<dyn IndexingRuntime> = preparer.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = preparer.clone();
    let layout: Arc<dyn LayoutRuntime> = preparer;

    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities
        .elementwise(elementwise)
        .reduction(reduction)
        .indexing(indexing)
        .dot_general(dot_general)
        .layout(layout);

    let storage = runtime_storage_class()?;
    let default_storage = storage.clone();
    let placement_storage = storage.clone();
    let signature_storage = storage.clone();
    let runtime_storage = storage.clone();
    let resident_storage = storage.clone();
    let allocation_domain = engine.allocation_domain();
    let provider_id = crate::cpu_provider_id();
    let provider_device_identity =
        ProviderDeviceIdentity::new(ProviderId::new(provider_id)?, "host".to_owned())?;
    let ingress = InputIngressContract::new(
        InputPlacementContract::new(move |placement, candidate| {
            candidate == &placement_storage
                && (cpu_input_placement(placement)
                    || (placement.memory_kind == MemoryKind::Managed
                        && allocation_domain.is_some()))
        }),
        InputSignatureContract::new(move |placement, family, domain, candidate| {
            candidate == &signature_storage
                && cpu_input_signature(placement, family, domain, allocation_domain)
        }),
        RuntimeInputContract::new(move |input: &TensorRead<'_>, candidate| {
            candidate == &runtime_storage && cpu_runtime_input(input, allocation_domain)
        }),
        ResidentOutputContract::new(move |input: &TensorRead<'_>, candidate| {
            candidate == &resident_storage && cpu_runtime_input(input, allocation_domain)
        }),
    );
    let metadata = EngineRegistrationMetadata::new(
        engine_id,
        provider_device_identity,
        runtime_hardware_class()?,
        Arc::from(vec![storage]),
        default_storage,
        capabilities.build(),
    );
    assemble_executable_engine_registration(ExecutableEngineRegistrationConfig::new(
        metadata,
        ExecutionContextIdentity::of::<CpuEngine>(),
        ingress,
    ))
}

fn cpu_input_signature(
    placement: &tenferro_tensor::Placement,
    backend_family: Option<&'static str>,
    input_domain: Option<tenferro_tensor::AllocationDomainId>,
    allocation_domain: Option<tenferro_tensor::AllocationDomainId>,
) -> bool {
    if placement.memory_kind == MemoryKind::Managed {
        return backend_family.is_some()
            && allocation_domain.is_some()
            && input_domain == allocation_domain;
    }
    cpu_input_placement(placement)
        && match backend_family {
            None => input_domain.is_none(),
            Some(_) => allocation_domain.is_some() && input_domain == allocation_domain,
        }
}

fn cpu_input_placement(placement: &tenferro_tensor::Placement) -> bool {
    matches!(
        placement.memory_kind,
        MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
    )
}

fn cpu_runtime_input(
    input: &TensorRead<'_>,
    allocation_domain: Option<tenferro_tensor::AllocationDomainId>,
) -> bool {
    cpu_input_signature(
        input.placement(),
        input.backend_family(),
        input.allocation_domain(),
        allocation_domain,
    )
}

fn runtime_storage_class() -> Result<StorageClass, RuntimeConfigError> {
    StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuPreparedKind {
    Elementwise,
    Reduction,
    Indexing,
    DotGeneral,
    Layout,
}

#[derive(Debug)]
struct CpuPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    #[allow(dead_code, reason = "bounded Debug records the selected CPU family")]
    kind: CpuPreparedKind,
}

impl PreparedOperation for CpuPreparedOperation {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        checked_specialization_heap_retained_bytes(&self.specialization).unwrap_or(usize::MAX)
    }
}

impl ElementwiseRuntime for CpuPreparer {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cpu(
            request.operation(),
            request.context(),
            CpuPreparedKind::Elementwise,
        )
    }

    fn max_fused_region_inputs(&self) -> Option<usize> {
        Some(tenferro_cpu_fused::ERASED_FUSION_MAX_INPUTS)
    }
}

impl ReductionRuntime for CpuPreparer {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cpu(
            request.operation(),
            request.context(),
            CpuPreparedKind::Reduction,
        )
    }
}

impl IndexingRuntime for CpuPreparer {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cpu(
            request.operation(),
            request.context(),
            CpuPreparedKind::Indexing,
        )
    }
}

impl DotGeneralPreparation for CpuPreparer {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cpu(
            request.operation(),
            request.context(),
            CpuPreparedKind::DotGeneral,
        )
    }
}

impl LayoutRuntime for CpuPreparer {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cpu(
            request.operation(),
            request.context(),
            CpuPreparedKind::Layout,
        )
    }
}

impl tenferro_runtime::ExecutionEngine for CpuEngine {
    fn target(&self) -> tenferro_runtime::EngineTarget {
        tenferro_runtime::EngineTarget::new(
            self.runtime_engine_id.clone(),
            ExecutionContextIdentity::of::<CpuEngine>(),
            self.allocation_domain(),
        )
    }

    fn take_extension_caches(&mut self) -> tenferro_runtime::ExtensionCacheStore {
        std::mem::take(&mut self.extension_caches)
    }

    fn restore_extension_caches(&mut self, caches: tenferro_runtime::ExtensionCacheStore) {
        self.extension_caches = caches;
    }

    fn mark_broken(&mut self) {
        self.broken = true;
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn prepare_cpu(
    operation: SemanticOperationView<'_>,
    context: &CorePrepareContext<'_>,
    expected_kind: CpuPreparedKind,
) -> Result<PrepareCapability, PrepareError> {
    validate_cpu_runtime_context(context)?;
    let SemanticOpRef::Core(op) = operation.op() else {
        return Err(wrong_family_error(expected_kind, "extension"));
    };
    let Some(actual_kind) = cpu_operation_kind(op) else {
        return Ok(PrepareCapability::Unsupported(
            UnsupportedReason::Operation {
                operation: UNKNOWN_CORE_OPERATION,
            },
        ));
    };
    if actual_kind != expected_kind {
        return Err(wrong_family_error(expected_kind, core_operation_name(op)));
    }

    let minimum = minimum_specialization_requirements(actual_kind, context.inputs())?;
    let merged =
        merge_specialization_requirements(context.specialization().requirements(), &minimum);
    if &merged != context.specialization().requirements() {
        return Ok(PrepareCapability::NeedsSpecialization(merged));
    }

    Ok(PrepareCapability::Prepared(
        PreparedOperationPlan::metadata(Arc::new(CpuPreparedOperation {
            binding: context.binding().clone(),
            specialization: context.specialization().clone(),
            kind: actual_kind,
        })),
    ))
}

fn validate_cpu_runtime_context(context: &CorePrepareContext<'_>) -> Result<(), PrepareError> {
    let expected_context = ExecutionContextIdentity::of::<CpuEngine>();
    if context.binding().context_identity() != expected_context {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::Elementwise,
                operation: "cpu-context-mismatch",
            },
        });
    }
    if context.binding().hardware_class().as_str() != CPU_HARDWARE_CLASS_ID {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::Elementwise,
                operation: "cpu-hardware-mismatch",
            },
        });
    }
    if context.resolved_placement().storage_class().as_str() != CPU_STORAGE_CLASS_ID {
        return Err(PrepareError::Unsupported {
            reason: UnsupportedReason::StorageClass {
                storage_class: context.resolved_placement().storage_class().clone(),
            },
        });
    }
    Ok(())
}

fn cpu_operation_kind(op: &CoreSemanticOp) -> Option<CpuPreparedKind> {
    Some(match op {
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
        | CoreSemanticOp::Log1p
        | CoreSemanticOp::Erf => CpuPreparedKind::Elementwise,
        CoreSemanticOp::ReduceSum { .. }
        | CoreSemanticOp::ReduceSumSquares { .. }
        | CoreSemanticOp::ReduceProd { .. }
        | CoreSemanticOp::ReduceMax { .. }
        | CoreSemanticOp::ReduceMin { .. } => CpuPreparedKind::Reduction,
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
        | CoreSemanticOp::PadToMatch { .. } => CpuPreparedKind::Indexing,
        CoreSemanticOp::DotGeneral { .. } => CpuPreparedKind::DotGeneral,
        CoreSemanticOp::Transpose { .. }
        | CoreSemanticOp::Reshape { .. }
        | CoreSemanticOp::BroadcastInDim { .. }
        | CoreSemanticOp::Convert { .. }
        | CoreSemanticOp::Constant { .. }
        | CoreSemanticOp::ExtractDiag { .. }
        | CoreSemanticOp::EmbedDiag { .. }
        | CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. } => CpuPreparedKind::Layout,
        _ => return None,
    })
}

fn minimum_specialization_requirements(
    kind: CpuPreparedKind,
    inputs: &InputSignature,
) -> Result<SpecializationRequirements, PrepareError> {
    let mut requirements = Vec::with_capacity(inputs.entries().len());
    for (input, entry) in inputs.entries().iter().enumerate() {
        let mut builder = InputSpecializationRequirements::builder();
        builder.dtype(true).rank(true);
        match kind {
            CpuPreparedKind::Indexing => {
                builder.concrete_dimensions(concrete_axes_for_rank(input, entry.shape().len())?);
            }
            CpuPreparedKind::DotGeneral => {
                builder
                    .concrete_dimensions(concrete_axes_for_rank(input, entry.shape().len())?)
                    .layout(LayoutSpecialization::Class);
            }
            CpuPreparedKind::Elementwise | CpuPreparedKind::Reduction | CpuPreparedKind::Layout => {
            }
        }
        requirements.push(
            builder
                .build()
                .expect("CPU minimum specialization requirements are internally valid"),
        );
    }
    Ok(SpecializationRequirements::new(requirements))
}

fn concrete_axes_for_rank(input: usize, rank: usize) -> Result<Vec<u32>, PrepareError> {
    if u32::try_from(rank).is_err() {
        return Err(PrepareError::Specialization {
            source: SpecializationError::ProjectionOverflow { input, rank },
        });
    }
    Ok((0..rank)
        .map(|axis| u32::try_from(axis).expect("rank precheck keeps axes encodable"))
        .collect())
}

fn merge_specialization_requirements(
    current: &SpecializationRequirements,
    minimum: &SpecializationRequirements,
) -> SpecializationRequirements {
    debug_assert_eq!(current.inputs().len(), minimum.inputs().len());
    let inputs = current
        .inputs()
        .iter()
        .zip(minimum.inputs())
        .map(|(current, minimum)| merge_input_requirements(current, minimum))
        .collect::<Vec<_>>();
    SpecializationRequirements::new(inputs)
}

fn merge_input_requirements(
    current: &InputSpecializationRequirements,
    minimum: &InputSpecializationRequirements,
) -> InputSpecializationRequirements {
    let mut axes = current.concrete_dimensions().to_vec();
    for axis in minimum.concrete_dimensions() {
        if !axes.contains(axis) {
            axes.push(*axis);
        }
    }
    let layout = current.layout().max(minimum.layout());
    let rank = current.specializes_rank()
        || minimum.specializes_rank()
        || !axes.is_empty()
        || layout == LayoutSpecialization::ExactStrides;
    let alignment = match (current.alignment_log2(), minimum.alignment_log2()) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    };
    let mut builder = InputSpecializationRequirements::builder();
    builder
        .dtype(current.specializes_dtype() || minimum.specializes_dtype())
        .rank(rank)
        .concrete_dimensions(axes)
        .placement(current.placement().max(minimum.placement()))
        .layout(layout)
        .alignment_log2(alignment);
    builder
        .build()
        .expect("merged CPU specialization requirements preserve builder invariants")
}

fn wrong_family_error(expected_kind: CpuPreparedKind, operation: &'static str) -> PrepareError {
    PrepareError::ProviderContract {
        source: ProviderContractError::WrongOperationFamily {
            expected: expected_kind.core_capability(),
            operation,
        },
    }
}

impl CpuPreparedKind {
    fn core_capability(self) -> CoreCapabilityKind {
        match self {
            Self::Elementwise => CoreCapabilityKind::Elementwise,
            Self::Reduction => CoreCapabilityKind::Reduction,
            Self::Indexing => CoreCapabilityKind::Indexing,
            Self::DotGeneral => CoreCapabilityKind::DotGeneral,
            Self::Layout => CoreCapabilityKind::Layout,
        }
    }
}

fn core_operation_name(op: &CoreSemanticOp) -> &'static str {
    match op {
        CoreSemanticOp::Add => "add",
        CoreSemanticOp::Sub => "sub",
        CoreSemanticOp::Mul => "mul",
        CoreSemanticOp::Neg => "neg",
        CoreSemanticOp::Conj => "conj",
        CoreSemanticOp::DotGeneral { .. } => "dot_general",
        CoreSemanticOp::Transpose { .. } => "transpose",
        CoreSemanticOp::Reshape { .. } => "reshape",
        CoreSemanticOp::BroadcastInDim { .. } => "broadcast_in_dim",
        CoreSemanticOp::Convert { .. } => "convert",
        CoreSemanticOp::Constant { .. } => "constant",
        CoreSemanticOp::ReduceSum { .. } => "reduce_sum",
        CoreSemanticOp::ReduceSumSquares { .. } => "reduce_sum_squares",
        CoreSemanticOp::Div => "div",
        CoreSemanticOp::Rem => "rem",
        CoreSemanticOp::Abs => "abs",
        CoreSemanticOp::Sign => "sign",
        CoreSemanticOp::Maximum => "maximum",
        CoreSemanticOp::Minimum => "minimum",
        CoreSemanticOp::Compare(_) => "compare",
        CoreSemanticOp::Select => "select",
        CoreSemanticOp::Clamp => "clamp",
        CoreSemanticOp::Exp => "exp",
        CoreSemanticOp::Log => "log",
        CoreSemanticOp::Sin => "sin",
        CoreSemanticOp::Cos => "cos",
        CoreSemanticOp::Tanh => "tanh",
        CoreSemanticOp::Sqrt => "sqrt",
        CoreSemanticOp::Rsqrt => "rsqrt",
        CoreSemanticOp::Pow => "pow",
        CoreSemanticOp::Expm1 => "expm1",
        CoreSemanticOp::Log1p => "log1p",
        CoreSemanticOp::Erf => "erf",
        CoreSemanticOp::ExtractDiag { .. } => "extract_diag",
        CoreSemanticOp::EmbedDiag { .. } => "embed_diag",
        CoreSemanticOp::Tril { .. } => "tril",
        CoreSemanticOp::Triu { .. } => "triu",
        CoreSemanticOp::Gather(_) => "gather",
        CoreSemanticOp::GatherDynamicSliceSizes { .. } => "gather_dynamic_slice_sizes",
        CoreSemanticOp::Scatter(_) => "scatter",
        CoreSemanticOp::Slice(_) => "slice",
        CoreSemanticOp::DynamicSlice { .. } => "dynamic_slice",
        CoreSemanticOp::DynamicUpdateSlice => "dynamic_update_slice",
        CoreSemanticOp::Pad(_) => "pad",
        CoreSemanticOp::Concatenate { .. } => "concatenate",
        CoreSemanticOp::Reverse { .. } => "reverse",
        CoreSemanticOp::ShapeOf { .. } => "shape_of",
        CoreSemanticOp::DynamicTruncate { .. } => "dynamic_truncate",
        CoreSemanticOp::PadToMatch { .. } => "pad_to_match",
        CoreSemanticOp::ReduceProd { .. } => "reduce_prod",
        CoreSemanticOp::ReduceMax { .. } => "reduce_max",
        CoreSemanticOp::ReduceMin { .. } => "reduce_min",
        _ => UNKNOWN_CORE_OPERATION,
    }
}

fn checked_specialization_heap_retained_bytes(
    specialization: &SpecializationProjection,
) -> Option<usize> {
    let requirements = specialization.requirements();
    checked_sum([
        requirements
            .inputs()
            .len()
            .checked_mul(size_of::<InputSpecializationRequirements>())?,
        checked_sum(
            requirements
                .inputs()
                .iter()
                .map(|input| size_of_val(input.concrete_dimensions())),
        )?,
        specialization
            .inputs()
            .len()
            .checked_mul(size_of::<InputSpecializationProjection>())?,
        checked_sum_options(
            specialization
                .inputs()
                .iter()
                .map(input_projection_retained_bytes),
        )?,
    ])
}

fn input_projection_retained_bytes(projection: &InputSpecializationProjection) -> Option<usize> {
    size_of_val(projection.concrete_dimensions()).checked_add(match projection.layout() {
        Some(LayoutProjection::ExactStrides(strides)) if strides.is_heap() => {
            size_of_val(strides.as_slice())
        }
        _ => 0,
    })
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

impl fmt::Display for CpuPreparedKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Elementwise => "elementwise",
            Self::Reduction => "reduction",
            Self::Indexing => "indexing",
            Self::DotGeneral => "dot_general",
            Self::Layout => "layout",
        })
    }
}

#[cfg(test)]
mod tests;
