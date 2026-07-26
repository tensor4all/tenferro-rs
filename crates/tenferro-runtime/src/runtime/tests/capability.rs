use std::num::NonZeroU64;
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;

use crate::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
use crate::runtime::{
    CoreCapabilityBundle, CoreCapabilityKind, CorePrepareContext, Determinism,
    DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime,
    ErasedExecutionContext, ExecutionContextIdentity, ExecutionPolicy, HardwareClassId,
    IndexingPrepareRequest, IndexingRuntime, InputSignature, LayoutPrepareRequest, LayoutRuntime,
    PrepareCapability, PrepareError, PrepareOptions, PrepareOptionsKey, PreparedOperation,
    PreparedOperationBinding, PreparedOperationHandle, ReductionPrepareRequest, ReductionRuntime,
    RegistrationIdentity, ResolvedPlanningConfig, ResolvedProgramPlacement, RuntimeEpoch,
    RuntimeId, SpecializationProjection, SpecializationRequirements, StorageClass,
    UnsupportedReason,
};
use tenferro_tensor::DType;

#[derive(Debug, Eq, PartialEq)]
struct ContextA(u32);

#[derive(Debug, Eq, PartialEq)]
struct ContextB;

#[derive(Debug)]
struct MetadataPlan {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    retained_bytes: usize,
}

impl PreparedOperation for MetadataPlan {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

#[derive(Debug)]
struct UnsupportedProvider;

impl UnsupportedProvider {
    fn unsupported() -> PrepareCapability {
        PrepareCapability::Unsupported(UnsupportedReason::MissingCapability {
            capability: CoreCapabilityKind::Elementwise,
        })
    }
}

impl ElementwiseRuntime for UnsupportedProvider {
    fn prepare(
        &self,
        _request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(Self::unsupported())
    }
}

impl ReductionRuntime for UnsupportedProvider {
    fn prepare(
        &self,
        _request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(Self::unsupported())
    }
}

impl IndexingRuntime for UnsupportedProvider {
    fn prepare(
        &self,
        _request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(Self::unsupported())
    }
}

impl DotGeneralPreparation for UnsupportedProvider {
    fn prepare(
        &self,
        _request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(Self::unsupported())
    }
}

impl LayoutRuntime for UnsupportedProvider {
    fn prepare(
        &self,
        _request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(Self::unsupported())
    }
}

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap_or(NonZeroU64::MIN)
}

fn engine_id(value: &str) -> crate::runtime::EngineId {
    crate::runtime::EngineId::new(value).unwrap_or_else(|error| panic!("{error}"))
}

fn hardware_id(value: &str) -> HardwareClassId {
    HardwareClassId::new(value).unwrap_or_else(|error| panic!("{error}"))
}

fn storage_class(value: &str) -> StorageClass {
    StorageClass::new(value).unwrap_or_else(|error| panic!("{error}"))
}

fn binding() -> PreparedOperationBinding {
    PreparedOperationBinding::new(
        RuntimeId::from_nonzero(nz(1)),
        RuntimeEpoch::from_nonzero(nz(2)),
        engine_id("tenferro.cpu"),
        RegistrationIdentity::new(nz(3), nz(4)),
        ExecutionContextIdentity::of::<ContextA>(),
        hardware_id("tenferro.cpu.host"),
    )
}

fn projection() -> SpecializationProjection {
    match SpecializationRequirements::polymorphic(0).project(&InputSignature::new(Vec::new())) {
        Ok(projection) => projection,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn prepared_operation_is_object_safe_and_metadata_only() {
    fn takes_object(_: &dyn PreparedOperation) {}

    let binding = binding();
    let specialization = projection();
    let plan = Arc::new(MetadataPlan {
        binding: binding.clone(),
        specialization: specialization.clone(),
        retained_bytes: 4096,
    });
    let handle: PreparedOperationHandle = plan;

    takes_object(handle.as_ref());
    assert_eq!(handle.binding(), &binding);
    assert_eq!(handle.specialization(), &specialization);
    assert_eq!(handle.retained_bytes(), 4096);
}

#[test]
fn all_five_core_runtime_traits_are_object_safe() {
    let provider = Arc::new(UnsupportedProvider);

    let _: Arc<dyn ElementwiseRuntime> = provider.clone();
    let _: Arc<dyn ReductionRuntime> = provider.clone();
    let _: Arc<dyn IndexingRuntime> = provider.clone();
    let _: Arc<dyn DotGeneralPreparation> = provider.clone();
    let _: Arc<dyn LayoutRuntime> = provider;
}

#[test]
fn erased_context_accepts_matching_identity() {
    let mut context = ContextA(7);
    let mut erased = ErasedExecutionContext::new(&mut context);

    let typed = erased
        .downcast_mut::<ContextA>(ExecutionContextIdentity::of::<ContextA>())
        .unwrap_or_else(|error| panic!("{error}"));
    typed.0 = 8;

    assert_eq!(
        erased.identity(),
        ExecutionContextIdentity::of::<ContextA>()
    );
    assert_eq!(context, ContextA(8));
}

#[test]
fn erased_context_rejects_context_b_with_execution_context_mismatch() {
    let mut context = ContextA(7);
    let mut erased = ErasedExecutionContext::new(&mut context);

    let error = erased
        .downcast_mut::<ContextB>(ExecutionContextIdentity::of::<ContextB>())
        .unwrap_err();

    assert_eq!(error.expected, ExecutionContextIdentity::of::<ContextB>());
    assert_eq!(error.actual, ExecutionContextIdentity::of::<ContextA>());
}

#[test]
fn prepared_binding_accessors_preserve_all_six_identity_components() {
    let binding = binding();

    assert_eq!(binding.runtime_id(), RuntimeId::from_nonzero(nz(1)));
    assert_eq!(binding.epoch(), RuntimeEpoch::from_nonzero(nz(2)));
    assert_eq!(binding.engine_id(), &engine_id("tenferro.cpu"));
    assert_eq!(
        binding.registration_identity(),
        RegistrationIdentity::new(nz(3), nz(4))
    );
    assert_eq!(
        binding.context_identity(),
        ExecutionContextIdentity::of::<ContextA>()
    );
    assert_eq!(binding.hardware_class(), &hardware_id("tenferro.cpu.host"));
}

#[test]
fn core_capability_bundle_builder_replaces_slots_infallibly() {
    let first: Arc<dyn ElementwiseRuntime> = Arc::new(UnsupportedProvider);
    let second: Arc<dyn ElementwiseRuntime> = Arc::new(UnsupportedProvider);
    let mut builder = CoreCapabilityBundle::builder();

    builder
        .elementwise(first)
        .elementwise(second.clone())
        .reduction(Arc::new(UnsupportedProvider))
        .indexing(Arc::new(UnsupportedProvider))
        .dot_general(Arc::new(UnsupportedProvider))
        .layout(Arc::new(UnsupportedProvider));
    let bundle = builder.build();

    assert!(Arc::ptr_eq(
        bundle.elementwise().expect("elementwise slot"),
        &second
    ));
    assert!(bundle.reduction().is_some());
    assert!(bundle.indexing().is_some());
    assert!(bundle.dot_general().is_some());
    assert!(bundle.layout().is_some());
}

#[test]
fn core_prepare_requests_preserve_operation_and_context_borrows() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap_or_else(|error| panic!("{error}"));
    let output = builder
        .add_op(CoreSemanticOp::Neg, &[input])
        .unwrap_or_else(|error| panic!("{error}"))[0];
    let frozen = builder
        .finish(&[output])
        .unwrap_or_else(|error| panic!("{error}"));
    let operation = frozen
        .program
        .operations()
        .next()
        .expect("one operation was added");

    let resolved_placement = ResolvedProgramPlacement::new(
        engine_id("tenferro.cpu"),
        storage_class("tenferro.storage.host"),
    );
    let policy = ExecutionPolicy::new(Determinism::Fast, Some(1024), 9);
    let options = PrepareOptions::new();
    let planning =
        ResolvedPlanningConfig::resolve(&policy, &options, hardware_id("tenferro.cpu.host"));
    let prepare_options_key =
        PrepareOptionsKey::from_resolved(resolved_placement.clone(), Some(1024), 9);
    let signature = InputSignature::new(Vec::new());
    let specialization = projection();
    let binding = binding();
    let context = CorePrepareContext::new(
        &binding,
        &signature,
        &resolved_placement,
        &planning,
        &prepare_options_key,
        &specialization,
    );
    let request = ElementwisePrepareRequest::new(operation, &context);

    assert_eq!(request.operation().outputs().len(), 1);
    assert!(std::ptr::eq(request.context(), &context));
    assert_eq!(context.binding(), &binding);
    assert_eq!(context.inputs(), &signature);
    assert_eq!(context.resolved_placement(), &resolved_placement);
    assert_eq!(context.planning(), &planning);
    assert_eq!(context.prepare_options_key(), &prepare_options_key);
    assert_eq!(context.specialization(), &specialization);
}

#[test]
fn prepared_operation_source_contract_has_bounded_execution_surface() {
    let source = include_str!("../capability.rs");
    let trait_body = source
        .split_once("pub trait PreparedOperation")
        .and_then(|(_, rest)| rest.split_once("pub type PreparedOperationHandle"))
        .map(|(body, _)| body)
        .expect("PreparedOperation trait should precede handle alias");

    for required in [
        "fn binding(&self) -> &PreparedOperationBinding",
        "fn specialization(&self) -> &SpecializationProjection",
        "fn retained_bytes(&self) -> usize",
        "fn execute(",
        "&mut ErasedExecutionContext<'_>",
        "&mut ExtensionCacheStore",
        "&[TensorRead<'_>]",
    ] {
        assert!(
            trait_body.contains(required),
            "PreparedOperation missing required method/signature fragment {required}"
        );
    }

    for forbidden in ["Runtime", "lease", "event", "schedule", "buffer", "scratch"] {
        assert!(
            !trait_body.contains(forbidden),
            "PreparedOperation must not expose {forbidden}"
        );
    }
}
