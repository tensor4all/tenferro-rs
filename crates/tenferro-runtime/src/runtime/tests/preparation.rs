use std::collections::VecDeque;
use std::hash::Hasher;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_tensor::{DType, Placement, ShapeVec, StrideVec, Tensor};

use crate::program::{CoreSemanticOp, FrozenProgram, ProgramInputSpec, SemanticProgramBuilder};
use crate::runtime::{
    CacheInFlightBehavior, CoreCapabilityBundle, ElementwisePrepareRequest, ElementwiseRuntime,
    EngineId, EngineRegistration, ExecutionContextIdentity, ExtensionEngine, ExtensionModule,
    ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig,
    ExtensionPrepareRequest, HardwareClassId, InputSignature, InputSignatureEntry,
    InputSpecializationRequirements, LayoutClass, PrepareCapability, PrepareError, PrepareOptions,
    PreparedOperation, PreparedOperationBinding, PreparedOperationPlan, ProgramPlacementConstraint,
    ProviderContractError, ResolvedProgramPlacement, Runtime, RuntimeConfigBuilder,
    SpecializationProjection, SpecializationRequirements, StorageClass,
};

const TEST_EXTENSION_FAMILY: &str = "tenferro.test.identity-extension.v1";

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
                PreparedOperationPlan::metadata(Arc::new(RecordingPreparedOperation {
                    binding: request.context().binding().clone(),
                    specialization: request.context().specialization().clone(),
                    retained_bytes,
                })),
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

#[derive(Clone, Debug)]
struct IdentityExtensionOp;

impl ExtensionOp for IdentityExtensionOp {
    fn family_id(&self) -> &'static str {
        TEST_EXTENSION_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(Self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<tenferro_ops::SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
}

#[derive(Debug)]
struct IdentityExtensionConfig;

impl ExtensionPlanningConfig for IdentityExtensionConfig {
    fn family_id(&self) -> &'static str {
        TEST_EXTENSION_FAMILY
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct RecordingExtensionEngine {
    engine: EngineId,
    calls: AtomicUsize,
}

impl RecordingExtensionEngine {
    fn new(engine: EngineId) -> Self {
        Self {
            engine,
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ExtensionEngine for RecordingExtensionEngine {
    fn family_id(&self) -> &'static str {
        TEST_EXTENSION_FAMILY
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<RecordingExtensionEngine>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.operation().family_id(), TEST_EXTENSION_FAMILY);
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::metadata(Arc::new(RecordingPreparedOperation {
                binding: request.binding().clone(),
                specialization: request.specialization().clone(),
                retained_bytes: 23,
            })),
        ))
    }
}

#[derive(Debug)]
struct IdentityExtensionModule {
    id: ExtensionModuleId,
    engine: Arc<RecordingExtensionEngine>,
}

impl ExtensionModule for IdentityExtensionModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        let engine: Arc<dyn ExtensionEngine> = self.engine.clone();
        registrar.register_engine(engine)?;
        registrar.register_planning_config(
            self.engine.engine_id().clone(),
            Arc::new(IdentityExtensionConfig),
        )
    }
}

fn repo_path(path: &str) -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("../..");
    root.push(path);
    root
}

fn repo_file(path: &str) -> String {
    std::fs::read_to_string(repo_path(path)).expect("source file must be readable")
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

fn core_then_extension_program() -> FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let left = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(4)]))
        .expect("left input");
    let right = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(4)]))
        .expect("right input");
    let sum = builder
        .add_op(CoreSemanticOp::Add, &[left, right])
        .expect("add op")[0];
    let output = builder
        .add_extension(Arc::new(IdentityExtensionOp), &[sum])
        .expect("extension op")[0];
    builder.finish(&[output]).expect("frozen mixed program")
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

fn runtime_with_engines_and_module(
    registrations: Vec<EngineRegistration>,
    module: Arc<dyn ExtensionModule>,
) -> Runtime {
    let mut builder = RuntimeConfigBuilder::new();
    for registration in registrations {
        builder
            .register_engine(registration)
            .expect("register engine");
    }
    builder.install_extension_module(module).expect("module");
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
    assert_eq!(
        prepared
            .root_for_test()
            .schedule_for_test()
            .nodes_for_test()
            .len(),
        prepared
            .root_for_test()
            .staging_for_test()
            .instructions
            .len()
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
fn per_operation_placement_can_mix_same_storage_core_and_extension_engines() {
    let frozen = core_then_extension_program();
    let core_provider = Arc::new(RecordingElementwise::default());
    let extension_engine_id = engine_id("tenferro.engine.extension");
    let extension_engine = Arc::new(RecordingExtensionEngine::new(extension_engine_id.clone()));
    let storage = storage_class("tenferro.storage.host");
    let runtime = runtime_with_engines_and_module(
        vec![
            engine_registration(
                "tenferro.engine.core",
                storage.clone(),
                core_provider.clone(),
            ),
            EngineRegistration::new(
                extension_engine_id.clone(),
                ExecutionContextIdentity::of::<RecordingExtensionEngine>(),
                hardware_id("tenferro.cpu.host"),
                Arc::from(vec![storage.clone()]),
                storage,
                CoreCapabilityBundle::builder().build(),
            )
            .expect("extension engine registration"),
        ],
        Arc::new(IdentityExtensionModule {
            id: ExtensionModuleId::new("tenferro.module.identity-extension").unwrap(),
            engine: extension_engine.clone(),
        }),
    );

    let prepared = runtime
        .prepare_for(
            &frozen,
            &two_input_signature(DType::F64),
            &PrepareOptions::new(),
        )
        .expect("same-storage mixed placement prepares");

    assert_eq!(core_provider.calls(), 1);
    assert_eq!(extension_engine.calls(), 1);
    assert_eq!(
        prepared.operations_for_test()[0].binding().engine_id(),
        &engine_id("tenferro.engine.core")
    );
    assert_eq!(
        prepared.operations_for_test()[1].binding().engine_id(),
        &extension_engine_id
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

#[test]
fn phase5_deletes_phase4_only_staging_adapter_name() {
    let source = repo_file("crates/tenferro-runtime/src/compiler/semantic_staging.rs");

    assert!(!source.contains("lower_semantic_to_exec_staging"));
    assert_eq!(
        source
            .matches("pub(crate) fn stage_semantic_program")
            .count(),
        1
    );
}

#[test]
fn phase5_runtime_execution_module_is_the_only_new_execution_owner() {
    let runtime_mod = repo_file("crates/tenferro-runtime/src/runtime/mod.rs");
    let graph_executor = repo_path("crates/tenferro-runtime/src/graph/executor.rs");
    let graph_mod = repo_file("crates/tenferro-runtime/src/graph/mod.rs");

    assert!(runtime_mod.contains("mod execution;"));
    assert!(runtime_mod.contains("mod schedule;"));
    assert!(
        !graph_executor.exists(),
        "retired graph executor facade file must not remain"
    );
    assert!(
        !graph_mod.contains("mod executor")
            && !graph_mod.contains("pub use executor")
            && !graph_mod.contains("GraphExecutor")
            && !graph_mod.contains("legacy"),
        "retired graph executor facade and legacy execution path must not remain in graph modules"
    );
}
