use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use tenferro_runtime::program::{CoreSemanticOp, SemanticOpRef, SemanticOperationView};
use tenferro_runtime::{
    CoreCapabilityBundle, CoreCapabilityKind, CorePrepareContext, DotGeneralPreparation,
    DotGeneralPrepareRequest, EngineId, EngineRegistration, ExecutionContextIdentity,
    HardwareClassId, InputSignature, InputSpecializationProjection,
    InputSpecializationRequirements, LayoutProjection, LayoutSpecialization, PrepareCapability,
    PrepareError, PreparedOperation, PreparedOperationBinding, PreparedOperationPlan,
    ProviderContractError, RuntimeConfigError, SpecializationError, SpecializationProjection,
    SpecializationRequirements, StorageClass, UnsupportedReason,
};
use tenferro_tensor::{
    AllocationDomainId, DeviceKind, GpuBackendKind, MemoryKind, Placement, TensorRead,
};

use super::WebGpuBackend;

const WEBGPU_ENGINE_ID: &str = "tenferro-webgpu.default.v1";
const WEBGPU_HARDWARE_CLASS_ID: &str = "tenferro-webgpu.device.v1";
const WEBGPU_STORAGE_CLASS_ID: &str = "tenferro.storage.device.v1";

/// Return the canonical WebGPU runtime engine identifier.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "webgpu")]
/// # {
/// let engine = tenferro_gpu::webgpu_runtime_engine_id()?;
/// assert_eq!(engine.as_str(), "tenferro-webgpu.default.v1");
/// # }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in WebGPU engine identifier
/// violates runtime identifier validation.
pub fn webgpu_runtime_engine_id() -> Result<EngineId, RuntimeConfigError> {
    EngineId::new(WEBGPU_ENGINE_ID).map_err(RuntimeConfigError::from)
}

/// Return the canonical WebGPU runtime hardware class.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "webgpu")]
/// # {
/// let hardware = tenferro_gpu::webgpu_runtime_hardware_class()?;
/// assert_eq!(hardware.as_str(), "tenferro-webgpu.device.v1");
/// # }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if the built-in WebGPU hardware class
/// violates runtime identifier validation.
pub fn webgpu_runtime_hardware_class() -> Result<HardwareClassId, RuntimeConfigError> {
    HardwareClassId::new(WEBGPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)
}

/// Build a runtime engine registration for a [`WebGpuBackend`].
///
/// The registration exposes WebGPU direct preparation for `dot_general` and the
/// runtime-owned tensor backend execution bridge. Other core operation families
/// remain unregistered until their WebGPU runtime preparation contracts are
/// implemented.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::WebGpuBackend;
///
/// let _register: fn(
///     &WebGpuBackend,
/// ) -> Result<tenferro_runtime::EngineRegistration, tenferro_runtime::RuntimeConfigError> =
///     tenferro_gpu::webgpu_runtime_engine_registration;
/// ```
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] if one of the built-in WebGPU runtime
/// identifiers violates runtime validation or if the registration is internally
/// invalid.
pub fn webgpu_runtime_engine_registration(
    backend: &WebGpuBackend,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let execution_backend = backend.as_ref().clone();

    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities.dot_general(dot_general);

    let storage = webgpu_runtime_storage_class()?;
    let placement_storage = storage.clone();
    let signature_storage = storage.clone();
    let runtime_storage = storage.clone();
    let resident_storage = storage.clone();
    let runtime = backend.runtime();
    let device_ordinal = runtime.device_ordinal();
    let allocation_domain = runtime.allocation_domain().map(|domain| domain.id);
    EngineRegistration::new(
        webgpu_runtime_engine_id()?,
        ExecutionContextIdentity::of::<WebGpuBackend>(),
        webgpu_runtime_hardware_class()?,
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .map(|registration| {
        registration
            .with_tensor_backend_executor(execution_backend)
            .with_input_signature_validator(move |placement, family, domain, candidate| {
                candidate == &signature_storage
                    && webgpu_input_signature(
                        placement,
                        family,
                        domain,
                        device_ordinal,
                        allocation_domain,
                    )
            })
            .with_input_ingress_validator(
                move |placement, candidate| {
                    candidate == &placement_storage
                        && webgpu_input_placement(placement, device_ordinal, allocation_domain)
                },
                move |input: &TensorRead<'_>, candidate| {
                    candidate == &runtime_storage
                        && webgpu_input_tensor(input, device_ordinal, allocation_domain)
                },
                move |input: &TensorRead<'_>, candidate| {
                    candidate == &resident_storage
                        && webgpu_input_tensor(input, device_ordinal, allocation_domain)
                },
            )
    })
}

fn webgpu_input_signature(
    placement: &Placement,
    backend_family: Option<&'static str>,
    input_domain: Option<AllocationDomainId>,
    device_ordinal: usize,
    allocation_domain: Option<AllocationDomainId>,
) -> bool {
    webgpu_input_placement(placement, device_ordinal, allocation_domain)
        && backend_family == Some("cubecl-webgpu")
        && input_domain == allocation_domain
}

fn webgpu_input_placement(
    placement: &Placement,
    device_ordinal: usize,
    allocation_domain: Option<AllocationDomainId>,
) -> bool {
    placement.memory_kind
        == if allocation_domain.is_some() {
            MemoryKind::Managed
        } else {
            MemoryKind::Device
        }
        && matches!(
            &placement.device,
            Some(device)
                if device.kind == DeviceKind::Gpu(GpuBackendKind::WebGpu)
                    && device.ordinal == device_ordinal
        )
}

fn webgpu_input_tensor(
    input: &TensorRead<'_>,
    device_ordinal: usize,
    allocation_domain: Option<AllocationDomainId>,
) -> bool {
    webgpu_input_placement(input.placement(), device_ordinal, allocation_domain)
        && input.backend_family() == Some("cubecl-webgpu")
        && input.allocation_domain() == allocation_domain
}

#[cfg(test)]
#[path = "tests/runtime_adapter.rs"]
mod ingress_tests;

fn webgpu_runtime_storage_class() -> Result<StorageClass, RuntimeConfigError> {
    StorageClass::new(WEBGPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)
}

#[derive(Debug)]
struct WebGpuPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
}

impl PreparedOperation for WebGpuPreparedOperation {
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

impl DotGeneralPreparation for WebGpuBackend {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        prepare_webgpu_dot_general(request.operation(), request.context())
    }
}

fn prepare_webgpu_dot_general(
    operation: SemanticOperationView<'_>,
    context: &CorePrepareContext<'_>,
) -> Result<PrepareCapability, PrepareError> {
    validate_webgpu_runtime_context(context)?;
    let SemanticOpRef::Core(op) = operation.op() else {
        return Err(wrong_family_error("extension"));
    };
    if !matches!(op, CoreSemanticOp::DotGeneral { .. }) {
        return Err(wrong_family_error(core_operation_name(op)));
    }

    let minimum = dot_general_specialization_requirements(context.inputs())?;
    let merged =
        merge_specialization_requirements(context.specialization().requirements(), &minimum)?;
    if &merged != context.specialization().requirements() {
        return Ok(PrepareCapability::NeedsSpecialization(merged));
    }

    Ok(PrepareCapability::Prepared(
        PreparedOperationPlan::metadata(Arc::new(WebGpuPreparedOperation {
            binding: context.binding().clone(),
            specialization: context.specialization().clone(),
        })),
    ))
}

fn validate_webgpu_runtime_context(context: &CorePrepareContext<'_>) -> Result<(), PrepareError> {
    let expected_context = ExecutionContextIdentity::of::<WebGpuBackend>();
    if context.binding().context_identity() != expected_context {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::DotGeneral,
                operation: "webgpu-context-mismatch",
            },
        });
    }
    if context.binding().hardware_class().as_str() != WEBGPU_HARDWARE_CLASS_ID {
        return Err(PrepareError::ProviderContract {
            source: ProviderContractError::WrongOperationFamily {
                expected: CoreCapabilityKind::DotGeneral,
                operation: "webgpu-hardware-mismatch",
            },
        });
    }
    if context.resolved_placement().storage_class().as_str() != WEBGPU_STORAGE_CLASS_ID {
        return Err(PrepareError::Unsupported {
            reason: UnsupportedReason::StorageClass {
                storage_class: context.resolved_placement().storage_class().clone(),
            },
        });
    }
    Ok(())
}

fn dot_general_specialization_requirements(
    inputs: &InputSignature,
) -> Result<SpecializationRequirements, PrepareError> {
    let mut requirements = Vec::with_capacity(inputs.entries().len());
    for (input, entry) in inputs.entries().iter().enumerate() {
        let mut builder = InputSpecializationRequirements::builder();
        builder
            .dtype(true)
            .rank(true)
            .concrete_dimensions(concrete_axes_for_rank(input, entry.shape().len())?)
            .layout(LayoutSpecialization::Class);
        requirements.push(builder.build().map_err(specialization_requirements_error)?);
    }
    Ok(SpecializationRequirements::new(requirements))
}

fn concrete_axes_for_rank(input: usize, rank: usize) -> Result<Vec<u32>, PrepareError> {
    if u32::try_from(rank).is_err() {
        return Err(PrepareError::Specialization {
            source: SpecializationError::ProjectionOverflow { input, rank },
        });
    }
    let mut axes = Vec::with_capacity(rank);
    for axis in 0..rank {
        axes.push(
            u32::try_from(axis).map_err(|_| PrepareError::Specialization {
                source: SpecializationError::ProjectionOverflow { input, rank },
            })?,
        );
    }
    Ok(axes)
}

fn merge_specialization_requirements(
    current: &SpecializationRequirements,
    minimum: &SpecializationRequirements,
) -> Result<SpecializationRequirements, PrepareError> {
    debug_assert_eq!(current.inputs().len(), minimum.inputs().len());
    let inputs = current
        .inputs()
        .iter()
        .zip(minimum.inputs())
        .map(|(current, minimum)| merge_input_requirements(current, minimum))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SpecializationRequirements::new(inputs))
}

fn merge_input_requirements(
    current: &InputSpecializationRequirements,
    minimum: &InputSpecializationRequirements,
) -> Result<InputSpecializationRequirements, PrepareError> {
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
    builder.build().map_err(specialization_requirements_error)
}

fn specialization_requirements_error(
    source: tenferro_runtime::InputSpecializationRequirementsError,
) -> PrepareError {
    PrepareError::Engine {
        source: Arc::new(source),
    }
}

fn wrong_family_error(operation: &'static str) -> PrepareError {
    PrepareError::ProviderContract {
        source: ProviderContractError::WrongOperationFamily {
            expected: CoreCapabilityKind::DotGeneral,
            operation,
        },
    }
}

fn core_operation_name(op: &CoreSemanticOp) -> &'static str {
    match op {
        CoreSemanticOp::DotGeneral { .. } => "dot_general",
        _ => "non-dot-general-core-operation",
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
