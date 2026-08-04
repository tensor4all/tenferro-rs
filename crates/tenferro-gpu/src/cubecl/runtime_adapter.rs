use std::fmt;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use tenferro_runtime::program::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
use tenferro_runtime::{
    assemble_executable_engine_registration, CacheOwnerError, CoreCapabilityBundle,
    CoreCapabilityKind, CorePrepareContext, DotGeneralPreparation, DotGeneralPrepareRequest,
    ElementwisePrepareRequest, ElementwiseRuntime, EngineId, EngineRegistration,
    EngineRegistrationMetadata, ExecutableEngineRegistrationConfig, ExecutionContextIdentity,
    HardwareClassId, IndexingPrepareRequest, IndexingRuntime, InputIngressContract,
    InputPlacementContract, InputSignature, InputSignatureContract, InputSpecializationProjection,
    InputSpecializationRequirements, LayoutPrepareRequest, LayoutProjection, LayoutRuntime,
    LayoutSpecialization, PrepareCapability, PrepareError, PreparedOperation,
    PreparedOperationBinding, PreparedOperationPlan, ProviderContractError, ProviderDeviceIdentity,
    ProviderId, ReductionPrepareRequest, ReductionRuntime, ResidentOutputContract,
    RuntimeCacheOwner, RuntimeConfigError, RuntimeInputContract, SpecializationError,
    SpecializationProjection, SpecializationRequirements, StorageClass, UnsupportedReason,
};
use tenferro_tensor::{DeviceKind, GpuBackendKind, MemoryKind, Placement, TensorRead, TensorView};

use super::event_domain::CudaEventDomainDriver;
use super::CudaBackend;

const CUDA_HARDWARE_CLASS_ID: &str = "tenferro-cuda.device.v1";
const CUDA_STORAGE_CLASS_ID: &str = "tenferro.storage.device.v1";
const UNKNOWN_CORE_OPERATION: &str = "unknown-core-operation";

#[derive(Debug)]
pub(crate) struct PreparedCudaRegistrationIdentity {
    engine_id: EngineId,
    provider_device_identity: ProviderDeviceIdentity,
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
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in CUDA hardware class violates
/// runtime identifier validation.
pub fn cuda_runtime_hardware_class() -> Result<HardwareClassId, RuntimeConfigError> {
    HardwareClassId::new(CUDA_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)
}

/// Build a runtime engine registration for a [`CudaBackend`] and caller-selected
/// [`EngineId`].
///
/// The registration exposes CUDA direct core preparation capabilities, CUDA
/// extension-cache ownership hooks, and the runtime-owned tensor backend
/// execution bridge. The caller owns the engine namespace and identity; CUDA
/// does not provide a process-global engine ID.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the CUDA hardware/storage identifiers or
/// provider/device identity fail validation, or if the registration is
/// internally invalid.
pub fn cuda_runtime_engine_registration(
    backend: &CudaBackend,
    engine_id: EngineId,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let prepared_identity = prepare_cuda_registration_identity(engine_id, backend.device_id())?;
    let backend = Arc::new(backend.clone());
    let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
    let reduction: Arc<dyn ReductionRuntime> = backend.clone();
    let indexing: Arc<dyn IndexingRuntime> = backend.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let layout: Arc<dyn LayoutRuntime> = backend.clone();
    let cache_owner: Arc<dyn RuntimeCacheOwner> = backend.clone();
    let execution_backend = backend.as_ref().clone();

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
    let device_ordinal = backend.device_id().ordinal() as usize;
    let allocation_domain = backend.runtime().allocation_domain_id();
    let ingress = InputIngressContract::new(
        InputPlacementContract::new(move |placement, candidate| {
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
            candidate == &runtime_storage
                && cuda_input_tensor(input, device_ordinal, allocation_domain)
        }),
        ResidentOutputContract::new(move |input: &TensorRead<'_>, candidate| {
            candidate == &resident_storage
                && cuda_input_tensor(input, device_ordinal, allocation_domain)
        }),
    );
    let metadata = EngineRegistrationMetadata::new(
        prepared_identity.engine_id,
        prepared_identity.provider_device_identity,
        cuda_runtime_hardware_class()?,
        Arc::from(vec![storage]),
        default_storage,
        capabilities.build(),
    );
    assemble_executable_engine_registration(ExecutableEngineRegistrationConfig::new(
        metadata,
        execution_backend,
        Arc::new(CudaEventDomainDriver::new(backend.runtime().clone())),
        ingress,
        Some(cache_owner),
    ))
}

fn cuda_provider_device_identity(
    device_id: super::CudaDeviceId,
) -> Result<ProviderDeviceIdentity, RuntimeConfigError> {
    Ok(ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.cuda")?,
        format!("device:{}", device_id.ordinal()),
    )?)
}

fn cuda_input_signature(
    placement: &Placement,
    backend_family: Option<&'static str>,
    allocation_domain: Option<tenferro_tensor::AllocationDomainId>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    cuda_input_placement(placement, device_ordinal)
        && backend_family == Some("cubecl")
        && allocation_domain == Some(expected_domain)
}

fn cuda_input_placement(placement: &Placement, device_ordinal: usize) -> bool {
    placement.memory_kind == MemoryKind::Device
        && matches!(
            &placement.device,
            Some(device)
                if device.kind == DeviceKind::Gpu(GpuBackendKind::Cuda)
                    && device.ordinal == device_ordinal
        )
}

fn cuda_input_tensor(
    input: &TensorRead<'_>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    cuda_input_placement(input.placement(), device_ordinal)
        && input.backend_family() == Some("cubecl")
        && input.allocation_domain() == Some(expected_domain)
        && cuda_input_has_owned_buffer(input, device_ordinal, expected_domain)
}

fn cuda_input_has_owned_buffer(
    input: &TensorRead<'_>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    match input.clone().tensor_view() {
        TensorView::F32(view) => {
            cubecl_view_has_owner::<f32>(&view, device_ordinal, expected_domain)
        }
        TensorView::F64(view) => {
            cubecl_view_has_owner::<f64>(&view, device_ordinal, expected_domain)
        }
        TensorView::I32(view) => {
            cubecl_view_has_owner::<i32>(&view, device_ordinal, expected_domain)
        }
        TensorView::I64(view) => {
            cubecl_view_has_owner::<i64>(&view, device_ordinal, expected_domain)
        }
        TensorView::Bool(view) => {
            cubecl_view_has_owner::<bool>(&view, device_ordinal, expected_domain)
        }
        TensorView::C32(view) => view.backend_buffer().is_some_and(|buffer| {
            buffer
                .as_any()
                .downcast_ref::<crate::CubeclBuffer>()
                .is_some_and(|buffer| {
                    buffer.device_ordinal() == device_ordinal
                        && buffer.allocation_domain() == Some(expected_domain)
                })
        }),
        TensorView::C64(view) => view.backend_buffer().is_some_and(|buffer| {
            buffer
                .as_any()
                .downcast_ref::<crate::CubeclBuffer>()
                .is_some_and(|buffer| {
                    buffer.device_ordinal() == device_ordinal
                        && buffer.allocation_domain() == Some(expected_domain)
                })
        }),
    }
}

fn cubecl_view_has_owner<T: 'static>(
    view: &tenferro_tensor::TypedTensorView<'_, T>,
    device_ordinal: usize,
    expected_domain: tenferro_tensor::AllocationDomainId,
) -> bool {
    view.backend_buffer().is_some_and(|buffer| {
        buffer
            .as_any()
            .downcast_ref::<crate::CubeclBuffer>()
            .is_some_and(|buffer| {
                buffer.device_ordinal() == device_ordinal
                    && buffer.allocation_domain() == Some(expected_domain)
            })
    })
}

#[cfg(test)]
#[path = "tests/runtime_adapter.rs"]
mod tests;

fn cuda_runtime_storage_class() -> Result<StorageClass, RuntimeConfigError> {
    StorageClass::new(CUDA_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaPreparedKind {
    Elementwise,
    Reduction,
    Indexing,
    DotGeneral,
    Layout,
}

#[derive(Debug)]
struct CudaPreparedOperation {
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

impl ElementwiseRuntime for CudaBackend {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Elementwise,
        )
    }
}

impl ReductionRuntime for CudaBackend {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Reduction,
        )
    }
}

impl IndexingRuntime for CudaBackend {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Indexing,
        )
    }
}

impl DotGeneralPreparation for CudaBackend {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::DotGeneral,
        )
    }
}

impl LayoutRuntime for CudaBackend {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_cuda(
            request.operation(),
            request.context(),
            CudaPreparedKind::Layout,
        )
    }
}

impl RuntimeCacheOwner for CudaBackend {
    fn cache_stats(&self) -> Result<tenferro_runtime::runtime::CacheStats, CacheOwnerError> {
        let stats = self
            .cuda_extension_cache_stats()
            .map_err(cache_owner_error)?;
        Ok(tenferro_runtime::runtime::CacheStats {
            entries: stats.entries,
            retained_bytes: stats.retained_bytes,
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            clears: stats.clears,
        })
    }

    fn clear_caches(&self) -> Result<(), CacheOwnerError> {
        self.clear_cuda_extension_cache().map_err(cache_owner_error)
    }
}

fn prepare_cuda(
    operation: SemanticOperationView<'_>,
    context: &CorePrepareContext<'_>,
    expected_kind: CudaPreparedKind,
) -> Result<PrepareCapability, PrepareError> {
    validate_cuda_runtime_context(context)?;
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

fn validate_cuda_runtime_context(context: &CorePrepareContext<'_>) -> Result<(), PrepareError> {
    let expected_context = ExecutionContextIdentity::of::<CudaBackend>();
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

fn cuda_operation_kind(op: &CoreSemanticOp) -> Option<CudaPreparedKind> {
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
        | CoreSemanticOp::Log1p => CudaPreparedKind::Elementwise,
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
        Some(LayoutProjection::ExactStrides(strides)) if strides.spilled() => {
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

fn cache_owner_error(source: crate::Error) -> CacheOwnerError {
    CacheOwnerError::new(Arc::new(source))
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
