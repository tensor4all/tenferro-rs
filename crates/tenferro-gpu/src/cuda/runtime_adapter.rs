//! The new engine's runtime registration: the preparation witness and the ingress contract.
//!
//! The CUDA preparation logic lives here: the operation kinds, the specialization requirements an
//! operation needs and the witness that reports them are the backend's, and this module also
//! supplies this engine's context identity and an ingress that recognises this path's storage.

use std::fmt;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use tenferro_runtime::program::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
use tenferro_runtime::{
    assemble_executable_engine_registration, CoreCapabilityBundle, CoreCapabilityKind,
    CorePrepareContext, DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest,
    ElementwiseRuntime, EngineId, EngineRegistration, ExecutableEngineRegistrationConfig,
    ExecutionContextIdentity, HardwareClassId, IndexingPrepareRequest, IndexingRuntime,
    InputIngressContract, InputPlacementContract, InputSignature, InputSignatureContract,
    InputSpecializationProjection, InputSpecializationRequirements, LayoutPrepareRequest,
    LayoutProjection, LayoutRuntime, LayoutSpecialization, PrepareCapability, PrepareError,
    PreparedOperation, PreparedOperationBinding, PreparedOperationPlan, ProviderContractError,
    ProviderDeviceIdentity, ProviderId, ReductionPrepareRequest, ReductionRuntime,
    ResidentOutputContract, RuntimeConfigError, RuntimeInputContract, SpecializationError,
    SpecializationProjection, SpecializationRequirements, StorageClass, UnsupportedReason,
};
use tenferro_tensor::{DeviceKind, GpuBackendKind, MemoryKind, Placement, TensorRead};

use super::engine::CudaEngine;

/// Stateless preparation witness for this path: preparation reads no engine state.
#[derive(Debug, Clone, Copy, Default)]
struct NextCudaPreparer;

impl ElementwiseRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Elementwise,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl ReductionRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Reduction,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl IndexingRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Indexing,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl DotGeneralPreparation for NextCudaPreparer {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::DotGeneral,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl LayoutRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Layout,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

/// Whether a read names an allocation of this engine: a device placement on the runtime's device
/// and one of this path's own allocations in the runtime's domain.
fn input_tensor(
    input: &TensorRead<'_>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    cuda_input_signature(
        input.placement(),
        input.backend_family(),
        input.allocation_domain(),
        device_ordinal,
        expected_domain,
    ) && match input {
        TensorRead::Tensor(tensor) => super::pending::cuda_storage(tensor).is_some_and(|storage| {
            storage.domain() == expected_domain && storage.device_ordinal() == device_ordinal
        }),
        // A borrowed view is a region of the same allocation, and the read path materializes it,
        // so it stays resident here exactly when its backing allocation is this engine's.
        TensorRead::View(view) => super::structural::region_of(view, "cuda runtime input")
            .is_ok_and(|region| {
                region.storage.domain() == expected_domain
                    && region.storage.device_ordinal() == device_ordinal
            }),
    }
}

/// Build the runtime registration for `engine`.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::cuda_runtime_engine_registration;
/// use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
/// use tenferro_runtime::Runtime;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
/// let mut builder = Runtime::builder();
/// builder.register_engine(cuda_runtime_engine_registration(&engine)?)?;
/// let runtime = builder.build()?;
/// let _snapshot = runtime.snapshot()?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// [`RuntimeConfigError`] when the provider identity, hardware or storage class, or the
/// registration itself cannot be built.
pub fn cuda_runtime_engine_registration(
    engine: &CudaEngine,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let prepared =
        prepare_cuda_registration_identity(engine.runtime_engine_id().clone(), engine.device_id())?;
    let preparer = Arc::new(NextCudaPreparer);
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

    let storage = cuda_runtime_storage_class()?;
    let default_storage = storage.clone();
    let placement_storage = storage.clone();
    let signature_storage = storage.clone();
    let runtime_storage = storage.clone();
    let resident_storage = storage.clone();
    let device_ordinal = engine.device_id().ordinal() as usize;
    let allocation_domain = engine.runtime().allocation_domain_id();
    let ingress = InputIngressContract::new(
        InputPlacementContract::new(move |placement: &Placement, candidate| {
            candidate == &placement_storage && cuda_input_placement(placement, device_ordinal)
        }),
        InputSignatureContract::new(move |placement, family, domain, candidate| {
            candidate == &signature_storage
                && cuda_input_signature(
                    placement,
                    family,
                    domain,
                    device_ordinal,
                    allocation_domain,
                )
        }),
        RuntimeInputContract::new(move |input: &TensorRead<'_>, candidate| {
            candidate == &runtime_storage && input_tensor(input, device_ordinal, allocation_domain)
        }),
        ResidentOutputContract::new(move |input: &TensorRead<'_>, candidate| {
            candidate == &resident_storage && input_tensor(input, device_ordinal, allocation_domain)
        }),
    );
    let metadata = tenferro_runtime::EngineRegistrationMetadata::new(
        prepared.engine_id,
        prepared.provider_device_identity,
        cuda_runtime_hardware_class()?,
        Arc::from(vec![storage]),
        default_storage,
        capabilities.build(),
    );
    assemble_executable_engine_registration(ExecutableEngineRegistrationConfig::new(
        metadata,
        ExecutionContextIdentity::of::<CudaEngine>(),
        ingress,
    ))
}

const CUDA_HARDWARE_CLASS_ID: &str = "tenferro-cuda.device.v1";
const CUDA_STORAGE_CLASS_ID: &str = "tenferro.storage.device.v1";
const UNKNOWN_CORE_OPERATION: &str = "unknown-core-operation";

#[derive(Debug)]
pub(crate) struct PreparedCudaRegistrationIdentity {
    pub(crate) engine_id: EngineId,
    pub(crate) provider_device_identity: ProviderDeviceIdentity,
}

pub(crate) fn prepare_cuda_registration_identity(
    engine_id: EngineId,
    device_id: super::CudaDeviceId,
) -> Result<PreparedCudaRegistrationIdentity, RuntimeConfigError> {
    Ok(PreparedCudaRegistrationIdentity {
        engine_id,
        provider_device_identity: cuda_provider_device_identity(device_id)?,
    })
}

/// Return the canonical CUDA runtime hardware class.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::cuda_runtime_hardware_class;
///
/// assert_eq!(
///     cuda_runtime_hardware_class().unwrap().as_str(),
///     "tenferro-cuda.device.v1"
/// );
/// ```
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in CUDA hardware class violates
/// runtime identifier validation.
pub fn cuda_runtime_hardware_class() -> Result<HardwareClassId, RuntimeConfigError> {
    HardwareClassId::new(CUDA_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)
}

pub(crate) fn cuda_provider_device_identity(
    device_id: super::CudaDeviceId,
) -> Result<ProviderDeviceIdentity, RuntimeConfigError> {
    Ok(ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.cuda")?,
        format!("device:{}", device_id.ordinal()),
    )?)
}

pub(crate) fn cuda_input_signature(
    placement: &Placement,
    backend_family: Option<&'static str>,
    allocation_domain: Option<tenferro_tensor::AllocationDomainId>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    cuda_input_placement(placement, device_ordinal)
        && backend_family == Some("cuda")
        && allocation_domain == Some(expected_domain)
}

pub(crate) fn cuda_input_placement(placement: &Placement, device_ordinal: usize) -> bool {
    placement.memory_kind == MemoryKind::Device
        && matches!(
            &placement.device,
            Some(device)
                if device.kind == DeviceKind::Gpu(GpuBackendKind::Cuda)
                    && device.ordinal == device_ordinal
        )
}

pub(crate) fn cuda_runtime_storage_class() -> Result<StorageClass, RuntimeConfigError> {
    StorageClass::new(CUDA_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaPreparedKind {
    Elementwise,
    Reduction,
    Indexing,
    DotGeneral,
    Layout,
}

#[derive(Debug)]
pub(crate) struct CudaPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    #[allow(dead_code, reason = "bounded Debug records the selected CUDA family")]
    kind: CudaPreparedKind,
}

impl PreparedOperation for CudaPreparedOperation {
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

pub(crate) fn prepare_cuda(
    operation: SemanticOperationView<'_>,
    context: &CorePrepareContext<'_>,
    expected_kind: CudaPreparedKind,
    expected_context: ExecutionContextIdentity,
) -> Result<PrepareCapability, PrepareError> {
    validate_cuda_runtime_context(context, expected_context)?;
    let SemanticOpRef::Core(op) = operation.op() else {
        return Err(wrong_family_error(expected_kind, "extension"));
    };
    let Some(actual_kind) = cuda_operation_kind(op) else {
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
        PreparedOperationPlan::metadata(Arc::new(CudaPreparedOperation {
            binding: context.binding().clone(),
            specialization: context.specialization().clone(),
            kind: actual_kind,
        })),
    ))
}

fn validate_cuda_runtime_context(
    context: &CorePrepareContext<'_>,
    expected_context: ExecutionContextIdentity,
) -> Result<(), PrepareError> {
    if context.binding().context_identity() != expected_context {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::Elementwise,
                operation: "cuda-context-mismatch",
            },
        });
    }
    if context.binding().hardware_class().as_str() != CUDA_HARDWARE_CLASS_ID {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::Elementwise,
                operation: "cuda-hardware-mismatch",
            },
        });
    }
    if context.resolved_placement().storage_class().as_str() != CUDA_STORAGE_CLASS_ID {
        return Err(PrepareError::Unsupported {
            reason: UnsupportedReason::StorageClass {
                storage_class: context.resolved_placement().storage_class().clone(),
            },
        });
    }
    Ok(())
}

pub(crate) fn cuda_operation_kind(op: &CoreSemanticOp) -> Option<CudaPreparedKind> {
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
        | CoreSemanticOp::Erf => CudaPreparedKind::Elementwise,
        CoreSemanticOp::ReduceSum { .. }
        | CoreSemanticOp::ReduceSumSquares { .. }
        | CoreSemanticOp::ReduceProd { .. }
        | CoreSemanticOp::ReduceMax { .. }
        | CoreSemanticOp::ReduceMin { .. } => CudaPreparedKind::Reduction,
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
        | CoreSemanticOp::PadToMatch { .. } => CudaPreparedKind::Indexing,
        CoreSemanticOp::DotGeneral { .. } => CudaPreparedKind::DotGeneral,
        CoreSemanticOp::Transpose { .. }
        | CoreSemanticOp::Reshape { .. }
        | CoreSemanticOp::BroadcastInDim { .. }
        | CoreSemanticOp::Convert { .. }
        | CoreSemanticOp::Constant { .. }
        | CoreSemanticOp::ExtractDiag { .. }
        | CoreSemanticOp::EmbedDiag { .. }
        | CoreSemanticOp::Tril { .. }
        | CoreSemanticOp::Triu { .. } => CudaPreparedKind::Layout,
        _ => return None,
    })
}

fn minimum_specialization_requirements(
    kind: CudaPreparedKind,
    inputs: &InputSignature,
) -> Result<SpecializationRequirements, PrepareError> {
    let mut requirements = Vec::with_capacity(inputs.entries().len());
    for (input, entry) in inputs.entries().iter().enumerate() {
        let mut builder = InputSpecializationRequirements::builder();
        builder.dtype(true).rank(true);
        match kind {
            CudaPreparedKind::Indexing => {
                builder.concrete_dimensions(concrete_axes_for_rank(input, entry.shape().len())?);
            }
            CudaPreparedKind::DotGeneral => {
                builder
                    .concrete_dimensions(concrete_axes_for_rank(input, entry.shape().len())?)
                    .layout(LayoutSpecialization::Class);
            }
            CudaPreparedKind::Elementwise
            | CudaPreparedKind::Reduction
            | CudaPreparedKind::Layout => {}
        }
        requirements.push(
            builder
                .build()
                .expect("CUDA minimum specialization requirements are internally valid"),
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
        .expect("merged CUDA specialization requirements preserve builder invariants")
}

fn wrong_family_error(expected_kind: CudaPreparedKind, operation: &'static str) -> PrepareError {
    PrepareError::ProviderContract {
        source: ProviderContractError::WrongOperationFamily {
            expected: expected_kind.core_capability(),
            operation,
        },
    }
}

impl CudaPreparedKind {
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

pub(crate) fn core_operation_name(op: &CoreSemanticOp) -> &'static str {
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

impl fmt::Display for CudaPreparedKind {
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
