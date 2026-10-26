//! The new engine's runtime registration: the preparation witness and the ingress contract.
//!
//! The preparation logic itself is the CUDA backend's, not the CubeCL adapter's: it lives in
//! `crate::cubecl::runtime_adapter` today and moves here at the cutover, so this module only
//! supplies this engine's context identity and an ingress that recognises this path's storage.

use std::sync::Arc;

use tenferro_runtime::{
    assemble_executable_engine_registration, CoreCapabilityBundle, DotGeneralPreparation,
    DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime, EngineRegistration,
    ExecutableEngineRegistrationConfig, ExecutionContextIdentity, IndexingPrepareRequest,
    IndexingRuntime, InputIngressContract, InputPlacementContract, InputSignatureContract,
    LayoutPrepareRequest, LayoutRuntime, PrepareCapability, PrepareError, ReductionPrepareRequest,
    ReductionRuntime, ResidentOutputContract, RuntimeConfigError, RuntimeInputContract,
};
use tenferro_tensor::{Placement, TensorRead};

use super::engine::CudaEngine;

/// Stateless preparation witness for this path: preparation reads no engine state.
#[derive(Debug, Clone, Copy, Default)]
struct NextCudaPreparer;

impl ElementwiseRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        crate::cubecl::runtime_adapter::prepare_cuda(
            request.operation(),
            request.context(),
            crate::cubecl::runtime_adapter::CudaPreparedKind::Elementwise,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl ReductionRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        crate::cubecl::runtime_adapter::prepare_cuda(
            request.operation(),
            request.context(),
            crate::cubecl::runtime_adapter::CudaPreparedKind::Reduction,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl IndexingRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        crate::cubecl::runtime_adapter::prepare_cuda(
            request.operation(),
            request.context(),
            crate::cubecl::runtime_adapter::CudaPreparedKind::Indexing,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl DotGeneralPreparation for NextCudaPreparer {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        crate::cubecl::runtime_adapter::prepare_cuda(
            request.operation(),
            request.context(),
            crate::cubecl::runtime_adapter::CudaPreparedKind::DotGeneral,
            ExecutionContextIdentity::of::<CudaEngine>(),
        )
    }
}

impl LayoutRuntime for NextCudaPreparer {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        crate::cubecl::runtime_adapter::prepare_cuda(
            request.operation(),
            request.context(),
            crate::cubecl::runtime_adapter::CudaPreparedKind::Layout,
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
    crate::cubecl::runtime_adapter::cuda_input_signature(
        input.placement(),
        input.backend_family(),
        input.allocation_domain(),
        device_ordinal,
        expected_domain,
    ) && match input {
        TensorRead::Tensor(tensor) => super::pending::cuda_storage(tensor).is_some_and(|storage| {
            storage.domain() == expected_domain && storage.device_ordinal() == device_ordinal
        }),
        // This path serves no borrowed view, so a view is never an accepted input.
        TensorRead::View(_) => false,
    }
}

/// Build the runtime registration for `engine`.
///
/// # Errors
///
/// [`RuntimeConfigError`] when the provider identity, hardware or storage class, or the
/// registration itself cannot be built.
pub fn runtime_engine_registration(
    engine: &CudaEngine,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let prepared = crate::cubecl::runtime_adapter::prepare_cuda_registration_identity(
        engine.runtime_engine_id().clone(),
        engine.device_id(),
    )?;
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

    let storage = crate::cubecl::runtime_adapter::cuda_runtime_storage_class()?;
    let default_storage = storage.clone();
    let placement_storage = storage.clone();
    let signature_storage = storage.clone();
    let runtime_storage = storage.clone();
    let resident_storage = storage.clone();
    let device_ordinal = engine.device_id().ordinal() as usize;
    let allocation_domain = engine.runtime().allocation_domain_id();
    let ingress = InputIngressContract::new(
        InputPlacementContract::new(move |placement: &Placement, candidate| {
            candidate == &placement_storage
                && crate::cubecl::runtime_adapter::cuda_input_placement(placement, device_ordinal)
        }),
        InputSignatureContract::new(move |placement, family, domain, candidate| {
            candidate == &signature_storage
                && crate::cubecl::runtime_adapter::cuda_input_signature(
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
        crate::cubecl::runtime_adapter::cuda_runtime_hardware_class()?,
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
