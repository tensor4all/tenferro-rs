use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::{DType, Placement, ShapeVec, StrideVec, Tensor};

use crate::program::{CoreSemanticOp, FrozenProgram, ProgramInputSpec, SemanticProgramBuilder};
use crate::runtime::{
    CacheInFlightBehavior, CoreCapabilityBundle, ElementwisePrepareRequest, ElementwiseRuntime,
    EngineId, EngineRegistration, ExecutionContextIdentity, HardwareClassId, InputSignature,
    InputSignatureEntry, InputSpecializationRequirements, LayoutClass, PrepareCapability,
    PrepareError, PrepareOptions, PreparedOperation, PreparedOperationBinding,
    ProgramPlacementConstraint, ProviderContractError, ResolvedProgramPlacement, Runtime,
    RuntimeConfigBuilder, SpecializationProjection, SpecializationRequirements, StorageClass,
};

#[derive(Clone, Debug)]
enum ProviderAction {
    Prepared { retained_bytes: usize },
    NeedsDType,
    NeedsSameRequirements,
}

#[derive(Debug)]
struct RecordingElementwise {
    calls: AtomicUsize,
    actions: Mutex<VecDeque<ProviderAction>>,
}

impl RecordingElementwise {
    fn new(actions: impl Into<VecDeque<ProviderAction>>) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            actions: Mutex::new(actions.into()),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Default for RecordingElementwise {
    fn default() -> Self {
        Self::new(VecDeque::from([ProviderAction::Prepared {
            retained_bytes: 17,
        }]))
    }
}

impl ElementwiseRuntime for RecordingElementwise {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let action = self
            .actions
            .lock()
            .expect("test action lock")
            .pop_front()
            .unwrap_or(ProviderAction::Prepared { retained_bytes: 17 });
        match action {
            ProviderAction::Prepared { retained_bytes } => Ok(PrepareCapability::Prepared(
                Arc::new(RecordingPreparedOperation {
                    binding: request.context().binding().clone(),
                    specialization: request.context().specialization().clone(),
                    retained_bytes,
                }),
            )),
            ProviderAction::NeedsDType => {
                let inputs: Vec<_> = request
                    .context()
                    .inputs()
                    .entries()
                    .iter()
                    .map(|_| {
                        let mut builder = InputSpecializationRequirements::builder();
                        builder.dtype(true);
                        builder.build().expect("valid dtype specialization")
                    })
                    .collect();
                Ok(PrepareCapability::NeedsSpecialization(
                    SpecializationRequirements::new(inputs),
                ))
            }
            ProviderAction::NeedsSameRequirements => Ok(PrepareCapability::NeedsSpecialization(
                request.context().specialization().requirements().clone(),
            )),
        }
    }
}

#[derive(Debug)]
struct RecordingPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    retained_bytes: usize,
}

impl PreparedOperation for RecordingPreparedOperation {
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

fn repo_file(path: &str) -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    std::fs::read_to_string(root).expect("source file must be readable")
}

fn engine_id(value: &str) -> EngineId {
    EngineId::new(value).expect("valid engine id")
}

fn hardware_id(value: &str) -> HardwareClassId {
    HardwareClassId::new(value).expect("valid hardware id")
}

fn storage_class(value: &str) -> StorageClass {
    StorageClass::new(value).expect("valid storage id")
}

fn layout_class(value: &str) -> LayoutClass {
    LayoutClass::new(value).expect("valid layout id")
}

fn signature_entry(dtype: DType, shape: impl Into<ShapeVec>) -> InputSignatureEntry {
    let shape = shape.into();
    let strides: StrideVec = (0..shape.len())
        .scan(1_isize, |stride, axis| {
            let current = *stride;
            *stride *= shape[axis] as isize;
            Some(current)
        })
        .collect();
    InputSignatureEntry::new(
        dtype,
        shape,
        Placement::default(),
        layout_class("tenferro.layout.compact-col-major.v1"),
        strides,
        Some(3),
    )
    .expect("valid signature entry")
}

fn two_input_add_program() -> FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let left = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1024)]))
        .expect("left input");
    let right = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1024)]))
        .expect("right input");
    let bound = Tensor::from_vec_col_major(vec![1024], vec![1.0_f64; 1024]).expect("bound tensor");
    builder
        .bind_input(left, Arc::new(bound))
        .expect("input binding");
    let sum = builder
        .add_op(CoreSemanticOp::Add, &[left, right])
        .expect("add op")[0];
    builder.finish(&[sum]).expect("frozen add program")
}

fn two_input_signature(dtype: DType) -> InputSignature {
    InputSignature::new(vec![
        signature_entry(dtype, ShapeVec::from_vec(vec![1024])),
        signature_entry(dtype, ShapeVec::from_vec(vec![1024])),
    ])
}

fn engine_registration(
    id: &str,
    storage: StorageClass,
    provider: Arc<RecordingElementwise>,
) -> EngineRegistration {
    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities.elementwise(provider);
    EngineRegistration::new(
        engine_id(id),
        ExecutionContextIdentity::of::<RecordingElementwise>(),
        hardware_id("tenferro.cpu.host"),
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .expect("engine registration")
}

fn runtime_with_engines(registrations: Vec<EngineRegistration>) -> Runtime {
    let mut builder = RuntimeConfigBuilder::new();
    for registration in registrations {
        builder
            .register_engine(registration)
            .expect("register engine");
    }
    builder.build().expect("runtime")
}

#[test]
fn prepared_program_is_binding_free_and_shares_staged_root() {
    let frozen = two_input_add_program();
    let provider = Arc::new(RecordingElementwise::default());
    let runtime = runtime_with_engines(vec![engine_registration(
        "tenferro.engine.a",
        storage_class("tenferro.storage.host"),
        provider.clone(),
    )]);

    let prepared = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new(),
        )
        .expect("prepared program");

    assert_eq!(provider.calls(), 1);
    assert!(Arc::ptr_eq(
        prepared.root_for_test().semantic_for_test(),
        &frozen.program
    ));
    assert_eq!(
        prepared
            .root_for_test()
            .staging_for_test()
            .input_slots
            .len(),
        frozen.program.inputs().len()
    );
    assert_eq!(
        prepared.operations_for_test().len(),
        frozen.program.operations().len()
    );
    assert_eq!(frozen.bindings.len(), 1);

    let source = repo_file("crates/tenferro-runtime/src/runtime/preparation.rs");
    let aggregate_region = source
        .split_once("pub(crate) struct PreparedProgramRoot")
        .and_then(|(_, rest)| rest.split_once("impl PreparedProgramRoot"))
        .map(|(body, _)| body)
        .expect("prepared aggregate region");
    for forbidden in [
        "ProgramBindings",
        "Tensor",
        concat!("Traced", "Tensor"),
        concat!("Eager", "Tensor"),
        "Runtime(",
        "schedule",
        "buffer",
        "event",
        "transfer",
        "collective",
        "execute",
    ] {
        assert!(
            !aggregate_region.contains(forbidden),
            "prepared aggregate must not retain or expose {forbidden}"
        );
    }
}

#[test]
fn placement_selection_uses_explicit_order_snapshot_order_and_default_storage() {
    let frozen = two_input_add_program();
    let provider_a = Arc::new(RecordingElementwise::default());
    let provider_b = Arc::new(RecordingElementwise::default());
    let storage_a = storage_class("tenferro.storage.a");
    let storage_b = storage_class("tenferro.storage.b");
    let runtime = runtime_with_engines(vec![
        engine_registration("tenferro.engine.b", storage_b.clone(), provider_b.clone()),
        engine_registration("tenferro.engine.a", storage_a.clone(), provider_a.clone()),
    ]);

    let explicit = PrepareOptions::new().with_placement(
        ProgramPlacementConstraint::new(
            vec![
                engine_id("tenferro.engine.b"),
                engine_id("tenferro.engine.a"),
            ],
            None,
        )
        .expect("explicit placement"),
    );
    let prepared_b = runtime
        .prepare_for(&frozen, &two_input_signature(DType::F64), &explicit)
        .expect("prepared explicit");
    assert_eq!(
        prepared_b.operations_for_test()[0].binding().engine_id(),
        &engine_id("tenferro.engine.b")
    );

    let prepared_a = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new(),
        )
        .expect("prepared default");
    assert_eq!(
        prepared_a.operations_for_test()[0].binding().engine_id(),
        &engine_id("tenferro.engine.a")
    );
    assert_eq!(
        prepared_a.root_for_test().resolved_placement_for_test(),
        &ResolvedProgramPlacement::new(engine_id("tenferro.engine.a"), storage_a)
    );
}

#[test]
fn ineligible_engine_returns_before_prepared_cache_miss() {
    let frozen = two_input_add_program();
    let runtime = runtime_with_engines(vec![EngineRegistration::new(
        engine_id("tenferro.engine.empty"),
        ExecutionContextIdentity::of::<()>(),
        hardware_id("tenferro.cpu.host"),
        Arc::from(vec![storage_class("tenferro.storage.host")]),
        storage_class("tenferro.storage.host"),
        CoreCapabilityBundle::builder().build(),
    )
    .expect("empty engine")]);

    let error = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new().with_cache_in_flight(CacheInFlightBehavior::Refuse),
        )
        .expect_err("missing elementwise capability");

    assert!(matches!(
        error.as_ref(),
        PrepareError::NoEligibleEngine { .. }
    ));
    assert_eq!(
        runtime
            .cache_stats()
            .expect("cache stats")
            .prepared_plans
            .misses,
        0
    );
}

#[test]
fn specialization_redirects_share_root_and_reproject_each_signature() {
    let frozen = two_input_add_program();
    let provider = Arc::new(RecordingElementwise::new(VecDeque::from([
        ProviderAction::NeedsDType,
        ProviderAction::Prepared { retained_bytes: 21 },
        ProviderAction::Prepared { retained_bytes: 22 },
    ])));
    let runtime = runtime_with_engines(vec![engine_registration(
        "tenferro.engine.a",
        storage_class("tenferro.storage.host"),
        provider.clone(),
    )]);

    let f64_plan = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new(),
        )
        .expect("f64 plan");
    let i32_plan = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::I32),
            &PrepareOptions::new(),
        )
        .expect("i32 plan");

    assert_eq!(provider.calls(), 3);
    assert!(Arc::ptr_eq(
        f64_plan.root_for_test(),
        i32_plan.root_for_test()
    ));
    assert_eq!(
        f64_plan
            .specialization_for_test()
            .inputs()
            .iter()
            .map(|input| input.dtype())
            .collect::<Vec<_>>(),
        vec![Some(DType::F64), Some(DType::F64)]
    );
    assert_eq!(
        i32_plan
            .specialization_for_test()
            .inputs()
            .iter()
            .map(|input| input.dtype())
            .collect::<Vec<_>>(),
        vec![Some(DType::I32), Some(DType::I32)]
    );
}

#[test]
fn nonmonotonic_specialization_response_is_provider_contract_error() {
    let frozen = two_input_add_program();
    let provider = Arc::new(RecordingElementwise::new(VecDeque::from([
        ProviderAction::NeedsSameRequirements,
    ])));
    let runtime = runtime_with_engines(vec![engine_registration(
        "tenferro.engine.a",
        storage_class("tenferro.storage.host"),
        provider,
    )]);

    let error = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new(),
        )
        .expect_err("non-widening specialization");

    assert!(matches!(
        error.as_ref(),
        PrepareError::ProviderContract {
            source: ProviderContractError::NonWideningSpecialization { .. }
        }
    ));
}

#[test]
fn runtime_prepared_cache_controls_are_public_and_affect_runtime_owned_cache() {
    let runtime = Runtime::builder().build().expect("runtime");
    let limits = runtime
        .prepared_cache_limits()
        .expect("prepared limits are readable");
    runtime
        .set_prepared_cache_limits(limits)
        .expect("prepared limits are writable");
    runtime
        .clear_prepared_cache()
        .expect("prepared cache can be cleared");
    runtime
        .clear_caches()
        .expect("aggregate caches can be cleared");
    assert_eq!(
        runtime
            .cache_stats()
            .expect("aggregate stats")
            .prepared_plans,
        Default::default()
    );
}
