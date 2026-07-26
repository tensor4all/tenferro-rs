use std::any::Any;
use std::error::Error as StdError;
use std::hash::Hasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_cpu::CpuBackend;
use tenferro_ops::{
    ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
    ExtensionShapeContext, SymDim,
};
use tenferro_runtime::{
    CoreCapabilityBundle, DType, DotGeneralPreparation, ElementwiseRuntime, EngineId,
    EngineRegistration, ErasedExecutionContext, Error, ErrorPhase, ExecutionContextIdentity,
    ExtensionCacheStore, ExtensionEngine, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
    ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest, GraphCompiler,
    HardwareClassId, IndexingRuntime, LayoutRuntime, PrepareCapability, PrepareError,
    PreparedOperation, PreparedOperationBinding, ReductionRuntime, Runtime, RuntimeCacheOwner,
    RuntimeConfigError, SpecializationProjection, StorageClass, TracedTensor, TransferProvider,
    TransferRequest,
};
use tenferro_tensor::{AllocationDomainId, SharedTensorAllocationDomain};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};

const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const COUNTING_EXTENSION_FAMILY: &str = "tenferro-test.counting-extension.v1";

fn cpu_registration(
    backend: &CpuBackend,
    include_execution_bridge: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
    let reduction: Arc<dyn ReductionRuntime> = backend.clone();
    let indexing: Arc<dyn IndexingRuntime> = backend.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let layout: Arc<dyn LayoutRuntime> = backend.clone();
    let cache_owner: Arc<dyn RuntimeCacheOwner> = backend.clone();

    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities
        .elementwise(elementwise)
        .reduction(reduction)
        .indexing(indexing)
        .dot_general(dot_general)
        .layout(layout);

    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    EngineRegistration::new(
        EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .map(|registration| {
        let registration = registration.with_cache_owner(cache_owner);
        if include_execution_bridge {
            registration.with_tensor_backend_executor(backend.as_ref().clone())
        } else {
            registration
        }
    })
}

fn runtime_with_cpu(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(backend, true)?)?;
    builder.build()
}

#[derive(Debug, Default)]
struct ExtensionCounters {
    prepare: AtomicUsize,
    execute: AtomicUsize,
    last_execute_domain: Mutex<Option<AllocationDomainId>>,
}

#[derive(Debug)]
struct TestAllocationDomain(AllocationDomainId);

impl SharedTensorAllocationDomain for TestAllocationDomain {
    fn id(&self) -> AllocationDomainId {
        self.0
    }

    fn allocate(&self, _dtype: DType, _shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "test-allocation-domain",
            "allocation not implemented for runtime execution tests",
        ))
    }
}

#[derive(Debug)]
struct RecordingTransferProvider {
    source: StorageClass,
    destination: StorageClass,
    calls: AtomicUsize,
}

impl RecordingTransferProvider {
    fn new(source: StorageClass, destination: StorageClass) -> Self {
        Self {
            source,
            destination,
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl TransferProvider for RecordingTransferProvider {
    fn transfer(&self, request: TransferRequest<'_>) -> tenferro_runtime::Result<Tensor> {
        assert_eq!(request.source_storage_class(), &self.source);
        assert_eq!(request.destination_storage_class(), &self.destination);
        self.calls.fetch_add(1, Ordering::SeqCst);
        request
            .input()
            .as_tensor()
            .cloned()
            .ok_or_else(|| Error::Internal("test transfer expected an owned tensor".into()))
    }
}

#[derive(Clone, Debug)]
struct CountingExtensionOp;

impl ExtensionOp for CountingExtensionOp {
    fn family_id(&self) -> &'static str {
        COUNTING_EXTENSION_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[derive(Debug)]
struct CountingExtensionConfig;

impl ExtensionPlanningConfig for CountingExtensionConfig {
    fn family_id(&self) -> &'static str {
        COUNTING_EXTENSION_FAMILY
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write_u8(0);
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct CountingExtensionEngine {
    engine_id: EngineId,
    counters: Arc<ExtensionCounters>,
}

impl ExtensionEngine for CountingExtensionEngine {
    fn family_id(&self) -> &'static str {
        COUNTING_EXTENSION_FAMILY
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<CpuBackend>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        assert_eq!(request.operation().family_id(), COUNTING_EXTENSION_FAMILY);
        assert_eq!(request.binding().engine_id(), &self.engine_id);
        self.counters.prepare.fetch_add(1, Ordering::SeqCst);
        Ok(PrepareCapability::Prepared(Arc::new(
            CountingPreparedOperation {
                binding: request.binding().clone(),
                specialization: request.specialization().clone(),
                counters: Arc::clone(&self.counters),
            },
        )))
    }
}

#[derive(Debug)]
struct CountingPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    counters: Arc<ExtensionCounters>,
}

impl PreparedOperation for CountingPreparedOperation {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }

    fn execute(
        &self,
        context: &mut ErasedExecutionContext<'_>,
        _extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        self.counters.execute.fetch_add(1, Ordering::SeqCst);
        let backend = context
            .downcast_mut::<CpuBackend>(self.binding.context_identity())
            .map_err(|source| {
                Error::runtime_state_source("counting_extension", ErrorPhase::Execution, source)
            })?;
        *self
            .counters
            .last_execute_domain
            .lock()
            .expect("domain lock") = backend.allocation_domain();
        Ok(vec![backend.with_backend_session(|session| {
            session.to_contiguous_read(inputs[0].clone())
        })?])
    }
}

#[derive(Debug)]
struct CountingExtensionModule {
    module_id: ExtensionModuleId,
    engine_id: EngineId,
    counters: Arc<ExtensionCounters>,
}

impl ExtensionModule for CountingExtensionModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        registrar.register_engine(Arc::new(CountingExtensionEngine {
            engine_id: self.engine_id.clone(),
            counters: Arc::clone(&self.counters),
        }))?;
        registrar
            .register_planning_config(self.engine_id.clone(), Arc::new(CountingExtensionConfig))
    }
}

fn runtime_with_counting_extension(
    backend: &CpuBackend,
    counters: Arc<ExtensionCounters>,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(backend, true)?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.module")
            .map_err(RuntimeConfigError::from)?,
        engine_id: EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        counters,
    }))?;
    builder.build()
}

fn cpu_registration_with_id(
    backend: &CpuBackend,
    engine_id: &str,
    include_core_capabilities: bool,
    include_execution_bridge: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let mut capabilities = CoreCapabilityBundle::builder();
    if include_core_capabilities {
        let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
        let reduction: Arc<dyn ReductionRuntime> = backend.clone();
        let indexing: Arc<dyn IndexingRuntime> = backend.clone();
        let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
        let layout: Arc<dyn LayoutRuntime> = backend.clone();
        capabilities
            .elementwise(elementwise)
            .reduction(reduction)
            .indexing(indexing)
            .dot_general(dot_general)
            .layout(layout);
    }

    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    EngineRegistration::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .map(|registration| {
        if include_execution_bridge {
            registration.with_tensor_backend_executor(backend.as_ref().clone())
        } else {
            registration
        }
    })
}

fn cpu_registration_with_storage_id(
    backend: &CpuBackend,
    engine_id: &str,
    storage_id: &str,
    include_core_capabilities: bool,
    include_execution_bridge: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let mut capabilities = CoreCapabilityBundle::builder();
    if include_core_capabilities {
        let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
        let reduction: Arc<dyn ReductionRuntime> = backend.clone();
        let indexing: Arc<dyn IndexingRuntime> = backend.clone();
        let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
        let layout: Arc<dyn LayoutRuntime> = backend.clone();
        capabilities
            .elementwise(elementwise)
            .reduction(reduction)
            .indexing(indexing)
            .dot_general(dot_general)
            .layout(layout);
    }

    let storage = StorageClass::new(storage_id).map_err(RuntimeConfigError::from)?;
    EngineRegistration::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .map(|registration| {
        if include_execution_bridge {
            registration.with_tensor_backend_executor(backend.as_ref().clone())
        } else {
            registration
        }
    })
}

#[test]
fn runtime_run_compiled_executes_extension_prepared_operation() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let counters = Arc::new(ExtensionCounters::default());
    let runtime = runtime_with_counting_extension(&backend, Arc::clone(&counters))?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(Arc::new(CountingExtensionOp), &[&x])?
        .pop()
        .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let first = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(first[0].as_slice::<f64>()?, &[3.0, 5.0]);
    assert_eq!(counters.prepare.load(Ordering::SeqCst), 1);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[3.0, 5.0]);
    assert_eq!(
        counters.prepare.load(Ordering::SeqCst),
        1,
        "second run should reuse the prepared program instead of preparing again"
    );
    assert_eq!(counters.execute.load(Ordering::SeqCst), 2);

    Ok(())
}

#[test]
fn runtime_run_compiled_dispatches_same_storage_extension_on_selected_engine(
) -> Result<(), Box<dyn StdError>> {
    let core_domain = AllocationDomainId::fresh();
    let extension_domain = AllocationDomainId::fresh();
    let core_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)));
    let extension_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)));
    let counters = Arc::new(ExtensionCounters::default());
    let extension_engine_id = "tenferro-test.extension-engine.v1";

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &core_backend,
        "tenferro-test.core-engine.v1",
        true,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &extension_backend,
        extension_engine_id,
        false,
        true,
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.per-op-module")
            .map_err(RuntimeConfigError::from)?,
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(Arc::new(CountingExtensionOp), &[&sum])?
        .pop()
        .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = runtime.run_compiled(&program, &[&input])?;

    assert_eq!(output[0].as_slice::<f64>()?, &[6.0, 10.0]);
    assert_eq!(counters.prepare.load(Ordering::SeqCst), 1);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(
        *counters.last_execute_domain.lock().expect("domain lock"),
        Some(extension_domain),
        "extension prepared operation must execute on the selected extension engine"
    );

    Ok(())
}

#[test]
fn runtime_run_compiled_transfers_between_storage_classes_on_scheduled_path(
) -> Result<(), Box<dyn StdError>> {
    let core_domain = AllocationDomainId::fresh();
    let extension_domain = AllocationDomainId::fresh();
    let core_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)));
    let extension_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)));
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-transfer-source.v1";
    let extension_engine_id = "tenferro-test.extension-transfer-destination.v1";
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.destination.v1")?;
    let transfer = Arc::new(RecordingTransferProvider::new(
        source_storage.clone(),
        destination_storage.clone(),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        core_engine_id,
        source_storage.as_str(),
        true,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_transfer_provider(
        source_storage.clone(),
        destination_storage.clone(),
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.transfer-module")
            .map_err(RuntimeConfigError::from)?,
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let source_event_domain = snapshot
        .engine(&EngineId::new(core_engine_id)?)
        .expect("source engine")
        .event_domain_id();
    let destination_event_domain = snapshot
        .engine(&EngineId::new(extension_engine_id)?)
        .expect("destination engine")
        .event_domain_id();
    assert_ne!(source_event_domain, destination_event_domain);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(Arc::new(CountingExtensionOp), &[&sum])?
        .pop()
        .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = runtime.run_compiled(&program, &[&input])?;

    assert_eq!(output[0].as_slice::<f64>()?, &[6.0, 10.0]);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(
        *counters.last_execute_domain.lock().expect("domain lock"),
        Some(extension_domain)
    );

    Ok(())
}

#[test]
fn runtime_run_compiled_reports_missing_transfer_provider_for_cross_storage(
) -> Result<(), Box<dyn StdError>> {
    let core_backend = CpuBackend::new();
    let extension_backend = CpuBackend::new();
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-missing-transfer-source.v1";
    let extension_engine_id = "tenferro-test.extension-missing-transfer-destination.v1";
    let destination_storage = StorageClass::new("tenferro-test.storage.missing-destination.v1")?;

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        core_engine_id,
        CPU_STORAGE_CLASS_ID,
        true,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        false,
        true,
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.no-transfer-module")
            .map_err(RuntimeConfigError::from)?,
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
        counters,
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(Arc::new(CountingExtensionOp), &[&sum])?
        .pop()
        .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

    assert!(error.to_string().contains("no transfer provider"));

    Ok(())
}

#[test]
fn runtime_submit_wait_uses_prepared_execution_path() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let handle = runtime.submit(&program, &[&input])?;
    let output = handle.wait()?;

    assert_eq!(output[0].as_slice::<f64>()?, &[2.0, 4.0]);
    assert!(runtime.cache_stats()?.prepared_plans.entries > 0);

    Ok(())
}

#[test]
fn runtime_run_compiled_uses_prepared_cache_on_second_call() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_first.misses >= 1);
    assert_eq!(after_second.misses, after_first.misses);
    assert!(after_second.hits > after_first.hits);
    assert_eq!(after_second.entries, after_first.entries);

    Ok(())
}

#[test]
fn runtime_run_compiled_reports_missing_execution_bridge() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(&backend, false)?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

    assert!(error.to_string().contains("execution bridge"));
    Ok(())
}

#[test]
fn runtime_run_compiled_reprepares_after_engine_reconfiguration() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(first[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    runtime.reconfigure(|edit| {
        edit.replace_engine(cpu_registration(&CpuBackend::new(), true)?)?;
        Ok(())
    })?;

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_second.misses > after_first.misses);
    Ok(())
}
