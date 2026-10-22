use std::any::Any;
use std::collections::{BTreeSet, HashSet};
use std::error::Error as StdError;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_cpu::CpuEngine;
use tenferro_ops::{
    ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
    ExtensionShapeContext, SymDim,
};
use tenferro_runtime::{
    assemble_executable_engine_registration, assemble_preparation_only_engine_registration,
    CoreCapabilityBundle, DType, DotGeneralPreparation, ElementwiseRuntime,
    EngineExecutionContractError, EngineId, EngineRegistration, EngineRegistrationMetadata,
    ErasedExecutionContext, Error, ErrorPhase, EventDomainId, ExecutableEngineRegistrationConfig,
    ExecutionContextIdentity, ExecutionEngine, ExtensionCacheStore, ExtensionEngine,
    ExtensionModule, ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar,
    ExtensionPlanningConfig, ExtensionPrepareRequest, GraphCompiler, HardwareClassId,
    IndexingRuntime, InputIngressContract, InputIngressContractError, InputPlacementContract,
    InputSignatureContract, LayoutRuntime, PreparationOnlyEngineRegistrationConfig,
    PrepareCapability, PrepareError, PreparedOperation, PreparedOperationBinding,
    PreparedOperationExecutor, PreparedOperationPlan, ProviderDeviceIdentity, ProviderId,
    ReductionRuntime, RegistrationKey, ResidentOutputContract, Runtime, RuntimeConfigError,
    RuntimeFailureReasonRef, RuntimeInputContract, RuntimeReconfigureError,
    SpecializationProjection, StorageClass, TracedTensor, TransferEndpoint, TransferError,
    TransferProvider, TransferProviderContractError, TransferRequest,
};
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendStorage, HostAccessError, HostReadGuard,
    HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain, StorageBuffer, Tensor,
    TensorRead, TensorView, TypedTensor,
};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const COUNTING_EXTENSION_FAMILY: &str = "tenferro-test.counting-extension.v1";
const DOWNSTREAM_COUNTING_EXTENSION_FAMILY: &str = "tenferro-test.downstream-counting-extension.v1";
const ROUTE_HOST_CUDA0_FAMILY: &str = "tenferro-test.route-host-cuda0.v1";
const ROUTE_HOST_CUDA1_FAMILY: &str = "tenferro-test.route-host-cuda1.v1";
const ROUTE_CUDA0_HOST_FAMILY: &str = "tenferro-test.route-cuda0-host.v1";
const ROUTE_CUDA1_HOST_FAMILY: &str = "tenferro-test.route-cuda1-host.v1";

static NEXT_TEST_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

fn next_test_allocation_id() -> AllocationId {
    AllocationId::from_backend_id(NEXT_TEST_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed))
}

fn test_provider_device_identity(
    engine_id: &str,
) -> Result<ProviderDeviceIdentity, RuntimeConfigError> {
    Ok(ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.cpu").map_err(RuntimeConfigError::from)?,
        format!("test-engine:{engine_id}"),
    )?)
}

fn test_domain(domain: AllocationDomainId) -> Arc<dyn SharedTensorAllocationDomain> {
    Arc::new(TestAllocationDomain(domain))
}

fn inputs_of<'a>(inputs: &[&'a Tensor]) -> Vec<TensorRead<'a>> {
    inputs
        .iter()
        .map(|input| TensorRead::from_tensor(input))
        .collect()
}

fn run_compiled(
    engines: &mut [&mut CpuEngine],
    runtime: &Runtime,
    program: &tenferro_runtime::CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads = inputs_of(inputs);
    let targets: Vec<tenferro_runtime::EngineTarget> =
        engines.iter().map(|engine| engine.target()).collect();
    let prepared = runtime.prepare(program, &reads, &targets)?;
    run_prepared(engines, runtime, &prepared, inputs)
}

fn prepare(
    engines: &[&CpuEngine],
    runtime: &Runtime,
    program: &tenferro_runtime::CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<tenferro_runtime::Prepared> {
    let reads = inputs_of(inputs);
    let targets: Vec<tenferro_runtime::EngineTarget> =
        engines.iter().map(|engine| engine.target()).collect();
    runtime.prepare(program, &reads, &targets)
}

fn run_prepared(
    engines: &mut [&mut CpuEngine],
    runtime: &Runtime,
    prepared: &tenferro_runtime::Prepared,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads = inputs_of(inputs);
    let outputs = {
        let mut erased: Vec<&mut dyn tenferro_runtime::ExecutionEngine> = engines
            .iter_mut()
            .map(|engine| &mut **engine as &mut dyn tenferro_runtime::ExecutionEngine)
            .collect();
        runtime.run(&mut erased, prepared, &reads)?
    };
    let mut owned = Vec::with_capacity(outputs.len());
    for output in outputs {
        owned.push(output.into_tensor(&mut *engines[0])?);
    }
    Ok(owned)
}

fn cpu_ingress_contract(engine: &CpuEngine, storage: &StorageClass) -> InputIngressContract {
    let allocation_domain = engine.allocation_domain();
    InputIngressContract::new(
        InputPlacementContract::new({
            let storage = storage.clone();
            move |placement, candidate| test_cpu_placement(placement) && candidate == &storage
        }),
        InputSignatureContract::new({
            let storage = storage.clone();
            move |placement, family, domain, candidate| {
                candidate == &storage
                    && test_cpu_input_signature(placement, family, domain, allocation_domain)
            }
        }),
        RuntimeInputContract::new({
            let storage = storage.clone();
            move |input: &TensorRead<'_>, candidate| {
                candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
            }
        }),
        ResidentOutputContract::new({
            let storage = storage.clone();
            move |input: &TensorRead<'_>, candidate| {
                candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
            }
        }),
    )
}

#[derive(Debug)]
enum CpuRegistrationState {
    PreparationOnly,
    Executable,
}

impl CpuRegistrationState {
    fn executable() -> Self {
        Self::Executable
    }
}

/// Metadata-only preparation capabilities for the test registrations.
///
/// The prepared plans carry no executable work of their own: every core
/// operation dispatches on the engine the caller passes to `Runtime::run`, so
/// the registration only has to answer the preparation questions the runtime
/// asks (binding, specialization and ingress).
#[derive(Debug)]
struct MetadataOnlyProvider;

#[derive(Debug)]
struct MetadataOnlyOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
}

impl MetadataOnlyOperation {
    fn new(context: &tenferro_runtime::CorePrepareContext<'_>) -> Self {
        Self {
            binding: context.binding().clone(),
            specialization: context.specialization().clone(),
        }
    }
}

impl PreparedOperation for MetadataOnlyOperation {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

fn metadata_only_plan(context: &tenferro_runtime::CorePrepareContext<'_>) -> PrepareCapability {
    PrepareCapability::Prepared(PreparedOperationPlan::metadata(Arc::new(
        MetadataOnlyOperation::new(context),
    )))
}

impl ElementwiseRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: tenferro_runtime::ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl ReductionRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: tenferro_runtime::ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl IndexingRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: tenferro_runtime::IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl DotGeneralPreparation for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: tenferro_runtime::DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl LayoutRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: tenferro_runtime::LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

fn cpu_core_capabilities(_engine: &CpuEngine) -> CoreCapabilityBundle {
    let provider = Arc::new(MetadataOnlyProvider);
    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities
        .elementwise(provider.clone())
        .reduction(provider.clone())
        .indexing(provider.clone())
        .dot_general(provider.clone())
        .layout(provider);
    capabilities.build()
}

fn assemble_cpu_registration(
    engine: &CpuEngine,
    metadata: EngineRegistrationMetadata,
    ingress_storage: StorageClass,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    match state {
        CpuRegistrationState::PreparationOnly => assemble_preparation_only_engine_registration(
            PreparationOnlyEngineRegistrationConfig::new(
                metadata,
                ExecutionContextIdentity::of::<CpuEngine>(),
            ),
        ),
        CpuRegistrationState::Executable => {
            assemble_executable_engine_registration(ExecutableEngineRegistrationConfig::new(
                metadata,
                ExecutionContextIdentity::of::<CpuEngine>(),
                cpu_ingress_contract(engine, &ingress_storage),
            ))
        }
    }
}

fn cpu_registration(
    engine: &CpuEngine,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(CPU_ENGINE_ID)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        capabilities,
    );
    assemble_cpu_registration(engine, metadata, storage, state)
}

fn runtime_with_cpu(backend: &CpuEngine) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(
        backend,
        cpu_core_capabilities(backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.build()
}

fn test_cpu_placement(placement: &Placement) -> bool {
    matches!(
        placement.memory_kind,
        MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
    )
}

fn test_cpu_runtime_input(
    input: &TensorRead<'_>,
    allocation_domain: Option<AllocationDomainId>,
) -> bool {
    test_cpu_placement(input.placement())
        && match input.backend_family() {
            None => true,
            Some(_) => {
                allocation_domain.is_some() && input.allocation_domain() == allocation_domain
            }
        }
}

fn test_cpu_input_signature(
    placement: &Placement,
    backend_family: Option<&'static str>,
    input_domain: Option<AllocationDomainId>,
    allocation_domain: Option<AllocationDomainId>,
) -> bool {
    test_cpu_placement(placement)
        && match backend_family {
            None => input_domain.is_none(),
            Some(_) => allocation_domain.is_some() && input_domain == allocation_domain,
        }
}

fn tensor_f64_values(tensor: &Tensor) -> tenferro_tensor::Result<Vec<f64>> {
    let Some(tensor) = tensor.as_typed::<f64>() else {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "tensor_f64_values",
            DType::F64,
            tensor.dtype(),
        ));
    };
    match tensor.buffer() {
        StorageBuffer::Host(values) => Ok(values.clone()),
        StorageBuffer::Backend(buffer) => buffer
            .map_read()
            .map(|values| values.to_vec())
            .map_err(|source| tenferro_tensor::Error::host_access("tensor_f64_values", source)),
    }
}

#[derive(Debug, Default)]
struct ExtensionCounters {
    prepare: AtomicUsize,
    execute: AtomicUsize,
    panic_execute: AtomicBool,
    last_execute_domain: Mutex<Option<AllocationDomainId>>,
    tracked_output_drops: Mutex<Option<Arc<AtomicUsize>>>,
    tracked_output_drop_events: Mutex<Option<Arc<Mutex<Vec<String>>>>>,
    foreign_output_domain: Mutex<Option<AllocationDomainId>>,
    /// When set, the prepared operation advertises the caller-borrowed session
    /// route, so `run_prepared_in` exercises the session-capable extension path.
    session_capable: bool,
}

#[derive(Debug)]
struct TestAllocationDomain(AllocationDomainId);

impl SharedTensorAllocationDomain for TestAllocationDomain {
    fn id(&self) -> AllocationDomainId {
        self.0
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
        let len = tenferro_tensor::validate::checked_shape_product(
            "test-allocation-domain",
            "shape",
            shape,
        )?;
        macro_rules! allocate {
            ($scalar:ty, $variant:ident) => {{
                let buffer = TestDomainBuffer::<$scalar> {
                    values: Arc::new(Mutex::new(vec![<$scalar>::default(); len])),
                    domain: self.0,
                    allocation: next_test_allocation_id(),
                };
                TypedTensor::from_buffer_col_major(
                    shape.to_vec(),
                    StorageBuffer::Backend(Box::new(buffer)),
                    Placement::default(),
                )
                .map(Tensor::from_typed::<preset_scalar!($variant)>)
            }};
        }
        match dtype {
            DType::F32 => allocate!(f32, F32),
            DType::F64 => allocate!(f64, F64),
            DType::I32 => allocate!(i32, I32),
            DType::I64 => allocate!(i64, I64),
            DType::Bool => allocate!(bool, Bool),
            DType::C32 => allocate!(num_complex::Complex32, C32),
            DType::C64 => allocate!(num_complex::Complex64, C64),
            // Test fixtures cover the preset dtypes; an externally defined scalar has no
            // fixture and would change what this test asserts.
            DType::External(_) => unreachable!("test fixtures cover the preset dtypes"),
        }
    }
}

#[derive(Debug)]
struct TestDomainBuffer<T> {
    values: Arc<Mutex<Vec<T>>>,
    domain: AllocationDomainId,
    allocation: AllocationId,
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> BackendStorage<T> for TestDomainBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.allocation-domain"
    }

    fn len(&self) -> usize {
        self.values.lock().expect("domain buffer lock").len()
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        Ok(HostReadGuard::new(
            self.values.lock().expect("domain buffer lock"),
        ))
    }

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        let values = Arc::clone(&self.values);
        let len = self.len();
        Ok(HostWriteGuard::new(len, move |source| {
            values
                .lock()
                .expect("domain buffer lock")
                .clone_from_slice(source);
            Ok(())
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct RecordingTransferProvider {
    source: StorageClass,
    destination: StorageClass,
    calls: AtomicUsize,
    requests: Mutex<Vec<RecordedTransferRequest>>,
    destination_domain: Option<Arc<dyn SharedTensorAllocationDomain>>,
    materialized_domains: Mutex<Vec<(Option<AllocationDomainId>, Option<AllocationDomainId>)>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordedTransferRequest {
    source_engine_id: EngineId,
    source_provider_device_identity: ProviderDeviceIdentity,
    source_event_domain_id: EventDomainId,
    source_storage_class: StorageClass,
    destination_engine_id: EngineId,
    destination_provider_device_identity: ProviderDeviceIdentity,
    destination_event_domain_id: EventDomainId,
    destination_storage_class: StorageClass,
}

fn read_f64_values(input: TensorRead<'_>) -> tenferro_runtime::Result<Vec<f64>> {
    match input.tensor_view() {
        TensorView::F64(view) => match view.backend_buffer() {
            Some(buffer) => buffer
                .map_read()
                .map_err(|source| tenferro_tensor::Error::host_access("test-transfer-read", source))
                .map(|values| values.to_vec())
                .map_err(Into::into),
            None => view
                .as_slice()
                .map(|values| values.to_vec())
                .map_err(Into::into),
        },
        _ => Err(Error::Internal(
            "test transfer currently expects f64 tensors".into(),
        )),
    }
}

fn duplicate_f64_read(input: TensorRead<'_>) -> tenferro_runtime::Result<Tensor> {
    let shape = input.shape().to_vec();
    let values = read_f64_values(input)?;
    Ok(Tensor::from_vec_col_major(shape, values)?)
}

impl RecordingTransferProvider {
    fn new(source: StorageClass, destination: StorageClass) -> Self {
        Self {
            source,
            destination,
            calls: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            destination_domain: None,
            materialized_domains: Mutex::new(Vec::new()),
        }
    }

    fn materializing(
        source: StorageClass,
        destination: StorageClass,
        destination_domain: Arc<dyn SharedTensorAllocationDomain>,
    ) -> Self {
        Self {
            destination_domain: Some(destination_domain),
            ..Self::new(source, destination)
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<RecordedTransferRequest> {
        self.requests.lock().expect("request lock").clone()
    }

    fn materialized_domains(
        &self,
    ) -> Vec<(Option<AllocationDomainId>, Option<AllocationDomainId>)> {
        self.materialized_domains
            .lock()
            .expect("materialized domain lock")
            .clone()
    }
}

impl TransferProvider for RecordingTransferProvider {
    fn transfer_blocking(&self, request: TransferRequest<'_>) -> tenferro_runtime::Result<Tensor> {
        assert_eq!(request.source_storage_class(), &self.source);
        assert_eq!(request.destination_storage_class(), &self.destination);
        self.requests
            .lock()
            .expect("request lock")
            .push(RecordedTransferRequest {
                source_engine_id: request.source_engine_id().clone(),
                source_provider_device_identity: request.source_provider_device_identity().clone(),
                source_event_domain_id: request.source_event_domain_id(),
                source_storage_class: request.source_storage_class().clone(),
                destination_engine_id: request.destination_engine_id().clone(),
                destination_provider_device_identity: request
                    .destination_provider_device_identity()
                    .clone(),
                destination_event_domain_id: request.destination_event_domain_id(),
                destination_storage_class: request.destination_storage_class().clone(),
            });
        self.calls.fetch_add(1, Ordering::SeqCst);
        let source_domain = request.input().allocation_domain();
        let output = match &self.destination_domain {
            Some(domain) => {
                let mut output =
                    domain.allocate(request.input().dtype(), request.input().shape())?;
                let source_values = read_f64_values(request.input().clone())?;
                let Some(destination) = output.as_typed_mut::<f64>() else {
                    return Err(Error::Internal(
                        "materializing test transfer currently expects f64 tensors".into(),
                    ));
                };
                let values = source_values;
                match destination.buffer() {
                    StorageBuffer::Host(_) => {
                        return Err(Error::Internal(
                            "test allocation domain returned host storage".into(),
                        ));
                    }
                    StorageBuffer::Backend(_buffer) => {
                        destination
                            .backend_buffer_mut()
                            .ok_or_else(|| {
                                Error::Internal(
                                    "test allocation domain returned host storage".into(),
                                )
                            })?
                            .map_write()
                            .map_err(|source| {
                                tenferro_tensor::Error::host_access("test-transfer-write", source)
                            })?
                            .copy_from_slice(&values)
                            .map_err(|source| {
                                tenferro_tensor::Error::host_access("test-transfer-write", source)
                            })?;
                    }
                }
                output
            }
            None => duplicate_f64_read(request.input().clone())?,
        };
        self.materialized_domains
            .lock()
            .expect("materialized domain lock")
            .push((
                source_domain,
                TensorRead::from_tensor(&output).allocation_domain(),
            ));
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug)]
enum FaultyTransferOutput {
    DType,
    Shape,
    Placement,
    BufferLength,
    Residency,
}

#[derive(Debug)]
struct FaultyTransferProvider {
    fault: FaultyTransferOutput,
}

impl TransferProvider for FaultyTransferProvider {
    fn transfer_blocking(&self, request: TransferRequest<'_>) -> tenferro_runtime::Result<Tensor> {
        match self.fault {
            FaultyTransferOutput::DType => Ok(Tensor::from_vec_col_major(vec![2], vec![1_i32, 2])?),
            FaultyTransferOutput::Shape => Ok(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?),
            FaultyTransferOutput::Placement => {
                let mut tensor = duplicate_f64_read(request.input().clone())?;
                let Some(tensor) = tensor.as_typed_mut::<f64>() else {
                    return Err(Error::Internal(
                        "test transfer expected an f64 tensor".into(),
                    ));
                };
                tensor.set_placement(Placement {
                    memory_kind: MemoryKind::Device,
                    device: None,
                    cpu_affinity: None,
                });
                Ok(tensor.duplicate()?.into())
            }
            FaultyTransferOutput::BufferLength => {
                let len = Arc::new(AtomicUsize::new(2));
                let buffer = StorageBuffer::Backend(Box::new(MutableLengthBuffer {
                    len: Arc::clone(&len),
                    domain: AllocationDomainId::fresh(),
                    allocation: next_test_allocation_id(),
                }));
                let tensor = TypedTensor::<f64>::from_buffer_col_major(
                    vec![2],
                    buffer,
                    Placement::default(),
                )?;
                len.store(1, Ordering::SeqCst);
                Ok(tensor.into())
            }
            FaultyTransferOutput::Residency => {
                let domain = AllocationDomainId::fresh();
                let buffer = StorageBuffer::Backend(Box::new(TestDomainBuffer::<f64> {
                    values: Arc::new(Mutex::new(vec![0.0; 2])),
                    domain,
                    allocation: next_test_allocation_id(),
                }));
                Ok(TypedTensor::<f64>::from_buffer_col_major(
                    vec![2],
                    buffer,
                    Placement::default(),
                )?
                .into())
            }
        }
    }
}

#[derive(Debug)]
struct MutableLengthBuffer {
    len: Arc<AtomicUsize>,
    domain: AllocationDomainId,
    allocation: AllocationId,
}

impl BackendStorage<f64> for MutableLengthBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.mutable-length"
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct DropTrackedBuffer {
    len: usize,
    drops: Arc<AtomicUsize>,
    domain: AllocationDomainId,
    allocation: AllocationId,
    events: Option<Arc<Mutex<Vec<String>>>>,
}

impl Drop for DropTrackedBuffer {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        if let Some(events) = &self.events {
            events
                .lock()
                .expect("drop event log lock")
                .push("tensor:drop".to_owned());
        }
    }
}

impl BackendStorage<f64> for DropTrackedBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.drop-tracked"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone, Debug)]
struct CountingExtensionOp {
    family_id: &'static str,
}

impl ExtensionOp for CountingExtensionOp {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family_id == other.family_id)
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
struct CountingExtensionConfig {
    family_id: &'static str,
}

impl ExtensionPlanningConfig for CountingExtensionConfig {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family_id == other.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct CountingExtensionEngine {
    family_id: &'static str,
    engine_id: EngineId,
    counters: Arc<ExtensionCounters>,
    session_capable: bool,
}

impl ExtensionEngine for CountingExtensionEngine {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<CpuEngine>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        assert_eq!(request.operation().family_id(), self.family_id);
        assert_eq!(request.binding().engine_id(), &self.engine_id);
        self.counters.prepare.fetch_add(1, Ordering::SeqCst);
        let prepared = Arc::new(CountingPreparedOperation {
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
            counters: Arc::clone(&self.counters),
            session_capable: self.session_capable,
        });
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::executable(prepared.clone(), prepared),
        ))
    }
}

#[derive(Debug)]
struct CountingPreparedOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    counters: Arc<ExtensionCounters>,
    session_capable: bool,
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
}

impl PreparedOperationExecutor for CountingPreparedOperation {
    fn execute(
        &self,
        context: &mut ErasedExecutionContext<'_>,
        _extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        self.counters.execute.fetch_add(1, Ordering::SeqCst);
        assert!(
            !self.counters.panic_execute.load(Ordering::SeqCst),
            "intentional prepared-operation panic"
        );
        let backend = context
            .downcast_mut::<CpuEngine>(self.binding.context_identity())
            .map_err(|source| {
                Error::runtime_state_source("counting_extension", ErrorPhase::Execution, source)
            })?;
        *self
            .counters
            .last_execute_domain
            .lock()
            .expect("domain lock") = backend.allocation_domain();
        if let Some(domain) = *self
            .counters
            .foreign_output_domain
            .lock()
            .expect("foreign output domain lock")
        {
            let shape = inputs[0].shape().to_vec();
            let len = tenferro_tensor::validate::checked_shape_product(
                "foreign-output-test",
                "shape",
                &shape,
            )?;
            let buffer = StorageBuffer::Backend(Box::new(TestDomainBuffer::<f64> {
                values: Arc::new(Mutex::new(vec![0.0; len])),
                domain,
                allocation: next_test_allocation_id(),
            }));
            return Ok(vec![TypedTensor::<f64>::from_buffer_col_major(
                shape,
                buffer,
                Placement::default(),
            )?
            .into()]);
        }
        if let Some(drops) = self
            .counters
            .tracked_output_drops
            .lock()
            .expect("tracked output lock")
            .clone()
        {
            let shape = inputs[0].shape().to_vec();
            let len = shape.iter().product();
            let domain = backend.allocation_domain().ok_or_else(|| {
                Error::Internal("drop-tracked output requires an allocation domain".into())
            })?;
            let events = self
                .counters
                .tracked_output_drop_events
                .lock()
                .expect("tracked output drop event lock")
                .clone();
            let buffer = StorageBuffer::Backend(Box::new(DropTrackedBuffer {
                len,
                drops,
                domain,
                allocation: next_test_allocation_id(),
                events,
            }));
            return Ok(vec![TypedTensor::<f64>::from_buffer_col_major(
                shape,
                buffer,
                Placement::default(),
            )?
            .into()]);
        }
        let same_domain = match (inputs[0].allocation_domain(), backend.allocation_domain()) {
            (Some(input_domain), Some(backend_domain)) => input_domain == backend_domain,
            _ => false,
        };
        if same_domain {
            let shape = inputs[0].shape().to_vec();
            let source_values = match inputs[0].clone().tensor_view() {
                TensorView::F64(view) => match view.backend_buffer() {
                    Some(buffer) => buffer
                        .map_read()
                        .map_err(|source| {
                            tenferro_tensor::Error::host_access(
                                "allocation-domain-test-read",
                                source,
                            )
                        })?
                        .to_vec(),
                    None => view.as_slice()?.to_vec(),
                },
                _ => {
                    return Err(Error::Internal(
                        "allocation-domain test executor currently expects f64 tensors".into(),
                    ));
                }
            };
            let domain = backend.shared_allocation_domain().ok_or_else(|| {
                Error::Internal("allocation-domain test executor requires a shared domain".into())
            })?;
            let mut output = domain.allocate(DType::F64, &shape)?;
            let Some(destination) = output.as_typed_mut::<f64>() else {
                return Err(Error::Internal(
                    "allocation-domain test executor produced a non-f64 tensor".into(),
                ));
            };
            let destination_buffer = destination.backend_buffer_mut().ok_or_else(|| {
                Error::Internal("allocation-domain test executor returned host storage".into())
            })?;
            destination_buffer
                .map_write()
                .map_err(|source| {
                    tenferro_tensor::Error::host_access("allocation-domain-test-write", source)
                })?
                .copy_from_slice(&source_values)
                .map_err(|source| {
                    tenferro_tensor::Error::host_access("allocation-domain-test-write", source)
                })?;
            return Ok(vec![output]);
        }
        Ok(vec![tenferro_tensor::TensorStructural::to_contiguous_read(
            backend,
            inputs[0].clone(),
        )?])
    }

    fn supports_session(&self) -> bool {
        self.session_capable
    }

    fn execute_in_session(
        &self,
        session: &mut dyn tenferro_tensor::Engine,
        _extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        self.counters.execute.fetch_add(1, Ordering::SeqCst);
        Ok(vec![session.to_contiguous_read(inputs[0].clone())?])
    }
}

#[derive(Debug)]
struct CountingExtensionModule {
    module_id: ExtensionModuleId,
    family_id: &'static str,
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
            family_id: self.family_id,
            engine_id: self.engine_id.clone(),
            counters: Arc::clone(&self.counters),
            session_capable: self.counters.session_capable,
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(CountingExtensionConfig {
                family_id: self.family_id,
            }),
        )
    }
}

fn runtime_with_counting_extension(
    engine: &CpuEngine,
    counters: Arc<ExtensionCounters>,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(
        engine,
        cpu_core_capabilities(engine),
        CpuRegistrationState::executable(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        counters,
    }))?;
    builder.build()
}

fn cpu_registration_with_id(
    engine: &CpuEngine,
    engine_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(engine_id)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        capabilities,
    );
    assemble_cpu_registration(engine, metadata, storage, state)
}

fn cpu_registration_with_storage_id(
    engine: &CpuEngine,
    engine_id: &str,
    storage_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let target = format!("test-engine:{engine_id}");
    cpu_registration_with_storage_id_for_target(
        engine,
        engine_id,
        storage_id,
        capabilities,
        state,
        &target,
    )
}

fn cpu_registration_with_storage_id_for_target(
    engine: &CpuEngine,
    engine_id: &str,
    storage_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
    target: &str,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let storage = StorageClass::new(storage_id).map_err(RuntimeConfigError::from)?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.cpu").map_err(RuntimeConfigError::from)?,
            target,
        )?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        capabilities,
    );
    assemble_cpu_registration(engine, metadata, storage, state)
}

fn cpu_registration_with_storage_classes(
    engine: &CpuEngine,
    engine_id: &str,
    storage_classes: Vec<StorageClass>,
    default_storage: StorageClass,
    ingress_storage: StorageClass,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(engine_id)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(storage_classes),
        default_storage,
        CoreCapabilityBundle::builder().build(),
    );
    assemble_cpu_registration(
        engine,
        metadata,
        ingress_storage,
        CpuRegistrationState::executable(),
    )
}

fn transfer_endpoint(
    engine_id: &str,
    storage_class: StorageClass,
) -> Result<TransferEndpoint, RuntimeConfigError> {
    Ok(TransferEndpoint::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        storage_class,
    ))
}

struct PublishedRouteFixture {
    runtime: Runtime,
    source_domain: AllocationDomainId,
    source_backend: CpuEngine,
    destination_backend: CpuEngine,
    source_endpoint: TransferEndpoint,
    destination_endpoint: TransferEndpoint,
    provider: Arc<RecordingTransferProvider>,
    counters: Arc<ExtensionCounters>,
}

fn published_route_fixture(
    affected_engine_id: EngineId,
    route_storage: StorageClass,
) -> Result<PublishedRouteFixture, RuntimeConfigError> {
    let source_engine_id = EngineId::new("tenferro-test.engine.reconfigure-source.v1")
        .map_err(RuntimeConfigError::from)?;
    let source_endpoint = TransferEndpoint::new(source_engine_id.clone(), route_storage.clone());
    let destination_endpoint =
        TransferEndpoint::new(affected_engine_id.clone(), route_storage.clone());
    let source_domain = AllocationDomainId::fresh();
    let destination_domain = AllocationDomainId::fresh();
    let source_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.engine.reconfigure-source.v1").expect("test engine id"),
        );
    let destination_domain = Arc::new(TestAllocationDomain(destination_domain));
    let destination_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::clone(&destination_domain) as Arc<_>)
        .with_runtime_engine_id(
            EngineId::new(affected_engine_id.as_str()).expect("test engine id"),
        );
    let provider = Arc::new(RecordingTransferProvider::materializing(
        route_storage.clone(),
        route_storage.clone(),
        Arc::clone(&destination_domain) as Arc<dyn SharedTensorAllocationDomain>,
    ));
    let counters = Arc::new(ExtensionCounters::default());

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &source_backend,
        source_engine_id.as_str(),
        route_storage.as_str(),
        cpu_core_capabilities(&source_backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &destination_backend,
        affected_engine_id.as_str(),
        route_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        source_endpoint.clone(),
        destination_endpoint.clone(),
        Arc::clone(&provider) as Arc<dyn TransferProvider>,
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.reconfigure-published-route-module.v1")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: affected_engine_id,
        counters: Arc::clone(&counters),
    }))?;

    Ok(PublishedRouteFixture {
        runtime: builder.build()?,
        source_domain,
        source_backend,
        destination_backend,
        source_endpoint,
        destination_endpoint,
        provider,
        counters,
    })
}

fn execute_published_route(fixture: &mut PublishedRouteFixture) -> Result<(), Box<dyn StdError>> {
    let source_identity =
        test_provider_device_identity(fixture.source_endpoint.engine_id().as_str())?;
    let destination_identity =
        test_provider_device_identity(fixture.destination_endpoint.engine_id().as_str())?;
    execute_published_route_with_targets(fixture, &source_identity, &destination_identity)
}

fn execute_published_route_with_targets(
    fixture: &mut PublishedRouteFixture,
    expected_source_identity: &ProviderDeviceIdentity,
    expected_destination_identity: &ProviderDeviceIdentity,
) -> Result<(), Box<dyn StdError>> {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("published route extension has one output");
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let mut input = TestAllocationDomain(fixture.source_domain).allocate(DType::F64, &[2])?;
    if let Some(input) = input.as_typed_mut::<f64>()
        && input.backend_buffer_mut().is_some()
    {
        input
            .backend_buffer_mut()
            .expect("backend input buffer")
            .map_write()
            .map_err(|source| tenferro_tensor::Error::host_access("published-route-input", source))?
            .copy_from_slice(&[3.0, 5.0])
            .map_err(|source| {
                tenferro_tensor::Error::host_access("published-route-input", source)
            })?;
    }
    let output = run_compiled(
        &mut [
            &mut fixture.source_backend,
            &mut fixture.destination_backend,
        ],
        &fixture.runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [3.0, 5.0]);
    assert_eq!(fixture.counters.prepare.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.provider.calls(), 1);
    let requests = fixture.provider.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        &requests[0].source_engine_id,
        fixture.source_endpoint.engine_id()
    );
    assert_eq!(
        &requests[0].source_provider_device_identity,
        expected_source_identity
    );
    assert_eq!(
        &requests[0].destination_engine_id,
        fixture.destination_endpoint.engine_id()
    );
    assert_eq!(
        &requests[0].destination_provider_device_identity,
        expected_destination_identity
    );
    assert_eq!(
        &requests[0].source_storage_class,
        fixture.source_endpoint.storage_class()
    );
    assert_eq!(
        &requests[0].destination_storage_class,
        fixture.destination_endpoint.storage_class()
    );
    Ok(())
}

#[test]
fn transfer_provider_registration_is_idempotent_and_rejects_conflicts(
) -> Result<(), Box<dyn StdError>> {
    let source = StorageClass::new("tenferro-test.storage.registry-source")?;
    let destination = StorageClass::new("tenferro-test.storage.registry-destination")?;
    let source_endpoint = TransferEndpoint::new(
        EngineId::new("tenferro-test.engine.registry-source.v1")?,
        source.clone(),
    );
    let destination_endpoint = TransferEndpoint::new(
        EngineId::new("tenferro-test.engine.registry-destination.v1")?,
        destination.clone(),
    );
    let provider: Arc<dyn TransferProvider> = Arc::new(RecordingTransferProvider::new(
        source.clone(),
        destination.clone(),
    ));
    let conflicting: Arc<dyn TransferProvider> = Arc::new(RecordingTransferProvider::new(
        source.clone(),
        destination.clone(),
    ));
    let mut builder = Runtime::builder();

    builder.register_transfer_provider(
        source_endpoint.clone(),
        destination_endpoint.clone(),
        Arc::clone(&provider),
    )?;
    builder.register_transfer_provider(
        source_endpoint.clone(),
        destination_endpoint.clone(),
        Arc::clone(&provider),
    )?;
    let error = builder
        .register_transfer_provider(
            source_endpoint.clone(),
            destination_endpoint.clone(),
            conflicting,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeConfigError::ConflictingRegistration {
            key: RegistrationKey::TransferProvider {
                source: actual_source,
                destination: actual_destination,
            },
        } if actual_source == source_endpoint && actual_destination == destination_endpoint
    ));
    Ok(())
}

#[test]
fn transfer_endpoint_is_immutable_ordered_and_hashable() -> Result<(), Box<dyn StdError>> {
    fn assert_endpoint_traits<T: Clone + std::fmt::Debug + Eq + Hash + Ord>() {}

    assert_endpoint_traits::<TransferEndpoint>();

    let storage = StorageClass::new("tenferro-test.storage.endpoint.v1")?;
    let first = TransferEndpoint::new(
        EngineId::new("tenferro-test.engine.endpoint-alpha.v1")?,
        storage.clone(),
    );
    let second = TransferEndpoint::new(
        EngineId::new("tenferro-test.engine.endpoint-beta.v1")?,
        storage,
    );

    assert_eq!(
        first.engine_id().as_str(),
        "tenferro-test.engine.endpoint-alpha.v1"
    );
    assert_eq!(
        first.storage_class().as_str(),
        "tenferro-test.storage.endpoint.v1"
    );

    let mut ordered = BTreeSet::new();
    ordered.insert(second.clone());
    ordered.insert(first.clone());
    assert_eq!(
        ordered.into_iter().collect::<Vec<_>>(),
        vec![first.clone(), second.clone()]
    );

    let mut hashed = HashSet::new();
    assert!(hashed.insert(first.clone()));
    assert!(!hashed.insert(first));
    assert!(hashed.insert(second.clone()));
    assert!(hashed.contains(&second));

    Ok(())
}

#[test]
fn endpoint_pair_routes_distinguish_two_engines_sharing_a_storage_class(
) -> Result<(), Box<dyn StdError>> {
    let host_engine_id = "tenferro-test.engine.host-route.v1";
    let cuda0_engine_id = "tenferro-test.engine.cuda0-route.v1";
    let cuda1_engine_id = "tenferro-test.engine.cuda1-route.v1";
    let host_domain = AllocationDomainId::fresh();
    let cuda0_domain = AllocationDomainId::fresh();
    let cuda1_domain = AllocationDomainId::fresh();
    let mut host_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(host_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.engine.host-route.v1").expect("test engine id"),
        );
    let mut cuda0_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(cuda0_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.engine.cuda0-route.v1").expect("test engine id"),
        );
    let mut cuda1_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(cuda1_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.engine.cuda1-route.v1").expect("test engine id"),
        );
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let host_endpoint = transfer_endpoint(host_engine_id, storage.clone())?;
    let cuda0_endpoint = transfer_endpoint(cuda0_engine_id, storage.clone())?;
    let cuda1_endpoint = transfer_endpoint(cuda1_engine_id, storage.clone())?;
    let host_to_cuda0 = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(cuda0_domain),
    ));
    let host_to_cuda1 = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(cuda1_domain),
    ));
    let cuda0_to_host = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(host_domain),
    ));
    let cuda1_to_host = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(host_domain),
    ));
    let host_to_cuda0_counters = Arc::new(ExtensionCounters::default());
    let host_to_cuda1_counters = Arc::new(ExtensionCounters::default());
    let cuda0_to_host_counters = Arc::new(ExtensionCounters::default());
    let cuda1_to_host_counters = Arc::new(ExtensionCounters::default());

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &host_backend,
        host_engine_id,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &cuda0_backend,
        cuda0_engine_id,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &cuda1_backend,
        cuda1_engine_id,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;

    let routes = [
        (
            host_endpoint.clone(),
            cuda0_endpoint.clone(),
            host_to_cuda0.clone(),
        ),
        (
            host_endpoint.clone(),
            cuda1_endpoint.clone(),
            host_to_cuda1.clone(),
        ),
        (
            cuda0_endpoint.clone(),
            host_endpoint.clone(),
            cuda0_to_host.clone(),
        ),
        (
            cuda1_endpoint.clone(),
            host_endpoint.clone(),
            cuda1_to_host.clone(),
        ),
    ];

    for (source, destination, provider) in routes {
        builder.register_transfer_provider(source, destination, provider)?;
    }
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-host-cuda0-module.v1")?,
        family_id: ROUTE_HOST_CUDA0_FAMILY,
        engine_id: EngineId::new("tenferro-test.engine.cuda0-route.v1")?,
        counters: Arc::clone(&host_to_cuda0_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-host-cuda1-module.v1")?,
        family_id: ROUTE_HOST_CUDA1_FAMILY,
        engine_id: EngineId::new("tenferro-test.engine.cuda1-route.v1")?,
        counters: Arc::clone(&host_to_cuda1_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-cuda0-host-module.v1")?,
        family_id: ROUTE_CUDA0_HOST_FAMILY,
        engine_id: EngineId::new("tenferro-test.engine.host-route.v1")?,
        counters: Arc::clone(&cuda0_to_host_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-cuda1-host-module.v1")?,
        family_id: ROUTE_CUDA1_HOST_FAMILY,
        engine_id: EngineId::new("tenferro-test.engine.host-route.v1")?,
        counters: Arc::clone(&cuda1_to_host_counters),
    }))?;

    let runtime = builder.build()?;
    assert_eq!(runtime.snapshot()?.transfer_provider_count(), 4);

    let execution_routes = [
        (
            &host_endpoint,
            &cuda0_endpoint,
            ROUTE_HOST_CUDA0_FAMILY,
            host_domain,
            &host_to_cuda0,
            &host_to_cuda0_counters,
        ),
        (
            &host_endpoint,
            &cuda1_endpoint,
            ROUTE_HOST_CUDA1_FAMILY,
            host_domain,
            &host_to_cuda1,
            &host_to_cuda1_counters,
        ),
        (
            &cuda0_endpoint,
            &host_endpoint,
            ROUTE_CUDA0_HOST_FAMILY,
            cuda0_domain,
            &cuda0_to_host,
            &cuda0_to_host_counters,
        ),
        (
            &cuda1_endpoint,
            &host_endpoint,
            ROUTE_CUDA1_HOST_FAMILY,
            cuda1_domain,
            &cuda1_to_host,
            &cuda1_to_host_counters,
        ),
    ];
    for (source_endpoint, destination_endpoint, family_id, source_domain, provider, counters) in
        execution_routes
    {
        let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
        let y =
            tenferro_runtime::extension::apply(Arc::new(CountingExtensionOp { family_id }), &[&x])?
                .pop()
                .expect("route extension has one output");
        let program =
            GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
        let input = TestAllocationDomain(source_domain).allocate(DType::F64, &[2])?;
        let output = run_compiled(
            &mut [&mut host_backend, &mut cuda0_backend, &mut cuda1_backend],
            &runtime,
            &program,
            &[&input],
        )?;

        assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
        assert_eq!(counters.prepare.load(Ordering::SeqCst), 1);
        assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
        assert_eq!(provider.calls(), 1);
        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(&request.source_engine_id, source_endpoint.engine_id());
        assert_eq!(
            &request.destination_engine_id,
            destination_endpoint.engine_id()
        );
    }
    Ok(())
}

#[test]
fn unknown_engine_transfer_endpoint_fails_build_without_publishing_a_runtime(
) -> Result<(), Box<dyn StdError>> {
    let backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.engine.known-route.v1").expect("test engine id"),
    );
    let known_engine = EngineId::new("tenferro-test.engine.known-route.v1")?;
    let unknown_engine = EngineId::new("tenferro-test.engine.unknown-route.v1")?;
    let storage = StorageClass::new("tenferro-test.storage.unknown-route.v1")?;
    let known_endpoint = TransferEndpoint::new(known_engine.clone(), storage.clone());
    let unknown_endpoint = TransferEndpoint::new(unknown_engine.clone(), storage.clone());
    let provider: Arc<dyn TransferProvider> = Arc::new(RecordingTransferProvider::new(
        storage.clone(),
        storage.clone(),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &backend,
        known_engine.as_str(),
        storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::PreparationOnly,
    )?)?;
    builder.register_transfer_provider(known_endpoint, unknown_endpoint.clone(), provider)?;

    let error = builder
        .build()
        .expect_err("an endpoint for an unknown engine must prevent publication");
    assert!(matches!(
        &error,
        RuntimeConfigError::UnknownTransferEndpointEngine { endpoint }
            if endpoint == &unknown_endpoint
    ));
    let rendered = error.to_string();
    assert!(rendered.contains(unknown_engine.as_str()), "{rendered}");
    assert!(rendered.contains(storage.as_str()), "{rendered}");
    Ok(())
}

#[test]
fn unsupported_transfer_endpoint_storage_fails_build() -> Result<(), Box<dyn StdError>> {
    let backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.engine.unsupported-route.v1").expect("test engine id"),
    );
    let engine = EngineId::new("tenferro-test.engine.unsupported-route.v1")?;
    let supported_storage = StorageClass::new("tenferro-test.storage.supported-route.v1")?;
    let unsupported_storage = StorageClass::new("tenferro-test.storage.unsupported-route.v1")?;
    let source = TransferEndpoint::new(engine.clone(), supported_storage.clone());
    let destination = TransferEndpoint::new(engine.clone(), unsupported_storage.clone());
    let provider: Arc<dyn TransferProvider> = Arc::new(RecordingTransferProvider::new(
        supported_storage,
        unsupported_storage.clone(),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &backend,
        engine.as_str(),
        source.storage_class().as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::PreparationOnly,
    )?)?;
    builder.register_transfer_provider(source.clone(), destination, provider)?;

    let error = builder
        .build()
        .expect_err("an endpoint with unsupported storage must fail validation");
    assert!(matches!(
        &error,
        RuntimeConfigError::UnsupportedTransferEndpointStorage { endpoint }
            if endpoint.engine_id() == source.engine_id()
                && endpoint.storage_class() == &unsupported_storage
    ));
    let rendered = error.to_string();
    assert!(rendered.contains(engine.as_str()), "{rendered}");
    assert!(
        rendered.contains(unsupported_storage.as_str()),
        "{rendered}"
    );
    Ok(())
}

#[test]
fn reconfiguration_rejects_invalid_transfer_endpoint_without_publishing(
) -> Result<(), Box<dyn StdError>> {
    let backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.engine.reconfigure-known.v1").expect("test engine id"),
    );
    let known_engine_id = "tenferro-test.engine.reconfigure-known.v1";
    let unknown_engine_id = EngineId::new("tenferro-test.engine.reconfigure-unknown.v1")?;
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let source_endpoint = transfer_endpoint(known_engine_id, storage.clone())?;
    let destination_endpoint = TransferEndpoint::new(unknown_engine_id.clone(), storage.clone());
    let provider: Arc<dyn TransferProvider> =
        Arc::new(RecordingTransferProvider::new(storage.clone(), storage));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &backend,
        known_engine_id,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::PreparationOnly,
    )?)?;
    let runtime = builder.build()?;
    let before = runtime.snapshot()?;

    let error = runtime
        .reconfigure(|edit| {
            edit.register_transfer_provider(
                source_endpoint.clone(),
                destination_endpoint.clone(),
                provider,
            )?;
            Ok(())
        })
        .expect_err("invalid endpoint must reject the candidate");

    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::UnknownTransferEndpointEngine { endpoint },
        } if endpoint == destination_endpoint
    ));
    let after = runtime.snapshot()?;
    assert_eq!(after.epoch(), before.epoch());
    assert_eq!(after.transfer_provider_count(), 0);
    Ok(())
}

#[test]
fn reconfiguration_remove_engine_rejects_dangling_route_atomically_and_keeps_same_route_executable(
) -> Result<(), Box<dyn StdError>> {
    let removed_engine_id = EngineId::new("tenferro-test.engine.reconfigure-removed.v1")?;
    let route_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let mut fixture = published_route_fixture(removed_engine_id.clone(), route_storage)?;
    let before = fixture.runtime.snapshot()?;
    let before_epoch = before.epoch();
    let before_removed_registration = before
        .engine(&removed_engine_id)
        .expect("removed engine is initially published")
        .registration_identity();
    let before_route_count = before.transfer_provider_count();
    assert_eq!(before_route_count, 1);

    let error = fixture
        .runtime
        .reconfigure(|edit| {
            edit.remove_engine(&removed_engine_id)?;
            Ok(())
        })
        .expect_err("removing an engine referenced by a route must be rejected");
    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::UnknownTransferEndpointEngine { endpoint },
        } if endpoint == fixture.destination_endpoint
    ));

    let after = fixture.runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(fixture.runtime.epoch()?, before_epoch);
    assert_eq!(after.engine_count(), before.engine_count());
    assert_eq!(after.transfer_provider_count(), before_route_count);
    assert_eq!(
        after
            .engine(&removed_engine_id)
            .expect("pre-edit snapshot remains readable")
            .registration_identity(),
        before_removed_registration
    );
    execute_published_route(&mut fixture)?;
    Ok(())
}

#[test]
fn reconfiguration_replace_engine_rejects_dropped_route_storage_atomically_and_keeps_same_route_executable(
) -> Result<(), Box<dyn StdError>> {
    let dropped_storage = StorageClass::new("tenferro-test.storage.reconfigure-dropped.v1")?;
    let replaced_engine_id = EngineId::new("tenferro-test.engine.reconfigure-replaced.v1")?;
    let mut fixture = published_route_fixture(replaced_engine_id.clone(), dropped_storage)?;
    let before = fixture.runtime.snapshot()?;
    let before_epoch = before.epoch();
    let before_replaced_registration = before
        .engine(&replaced_engine_id)
        .expect("replaced engine is initially published")
        .registration_identity();
    let before_route_count = before.transfer_provider_count();
    assert_eq!(before_route_count, 1);
    let replacement = cpu_registration_with_storage_id(
        &CpuEngine::serial(),
        replaced_engine_id.as_str(),
        CPU_STORAGE_CLASS_ID,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?;

    let error = fixture
        .runtime
        .reconfigure(|edit| {
            edit.replace_engine(replacement)?;
            Ok(())
        })
        .expect_err("dropping a route storage class must be rejected");
    assert!(matches!(
        error,
        RuntimeReconfigureError::Edit {
            source: RuntimeConfigError::UnsupportedTransferEndpointStorage { endpoint },
        } if endpoint == fixture.destination_endpoint
    ));

    let after = fixture.runtime.snapshot()?;
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(fixture.runtime.epoch()?, before_epoch);
    assert_eq!(after.engine_count(), before.engine_count());
    assert_eq!(after.transfer_provider_count(), before_route_count);
    assert_eq!(
        after
            .engine(&replaced_engine_id)
            .expect("pre-edit snapshot remains readable")
            .registration_identity(),
        before_replaced_registration
    );
    execute_published_route(&mut fixture)?;
    Ok(())
}

#[test]
fn explicit_target_rebind_updates_frozen_lookup_and_provider_request(
) -> Result<(), Box<dyn StdError>> {
    let affected_engine_id = EngineId::new("tenferro-test.engine.rebind-execution.v1")?;
    let route_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let mut fixture = published_route_fixture(affected_engine_id.clone(), route_storage)?;
    let source_identity =
        test_provider_device_identity(fixture.source_endpoint.engine_id().as_str())?;
    let replacement_identity =
        ProviderDeviceIdentity::new(ProviderId::new("tenferro.test.cpu")?, "test-rebound-target")?;
    let replacement = cpu_registration_with_storage_id_for_target(
        &fixture.destination_backend,
        affected_engine_id.as_str(),
        fixture.destination_endpoint.storage_class().as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
        replacement_identity.target_identity(),
    )?;

    fixture.runtime.reconfigure(|edit| {
        edit.remove_transfer_provider(
            fixture.source_endpoint.clone(),
            fixture.destination_endpoint.clone(),
        )?;
        edit.remove_engine(&affected_engine_id)?;
        edit.register_engine(replacement)?;
        edit.register_transfer_provider(
            fixture.source_endpoint.clone(),
            fixture.destination_endpoint.clone(),
            Arc::clone(&fixture.provider) as Arc<dyn TransferProvider>,
        )?;
        Ok(())
    })?;

    execute_published_route_with_targets(&mut fixture, &source_identity, &replacement_identity)?;
    let requests = fixture.provider.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].destination_provider_device_identity,
        replacement_identity
    );
    Ok(())
}

#[test]
fn preparation_binding_cannot_be_promoted_to_partial_execution() -> Result<(), Box<dyn StdError>> {
    let storage = StorageClass::new("tenferro-test.storage.missing-ingress.v1")?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new("tenferro-test.engine.missing-ingress.v1")?,
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.cpu")?,
            "test-engine:tenferro-test.engine.missing-ingress.v1",
        )?,
        HardwareClassId::new("tenferro-test.hardware.missing-ingress.v1")?,
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    );
    let registration = assemble_preparation_only_engine_registration(
        PreparationOnlyEngineRegistrationConfig::new(
            metadata,
            ExecutionContextIdentity::of::<CpuEngine>(),
        ),
    )?;
    let mut builder = Runtime::builder();

    builder.register_engine(registration)?;
    let runtime = builder.build()?;
    assert!(runtime
        .snapshot()?
        .engine(&EngineId::new("tenferro-test.engine.missing-ingress.v1")?)
        .is_some());
    Ok(())
}

#[test]
fn runtime_run_compiled_executes_extension_prepared_operation() -> Result<(), Box<dyn StdError>> {
    let mut backend = CpuEngine::serial()
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let counters = Arc::new(ExtensionCounters::default());
    let runtime = runtime_with_counting_extension(&backend, Arc::clone(&counters))?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let first = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
    assert_eq!(first[0].as_slice::<f64>()?, &[3.0, 5.0]);
    assert_eq!(counters.prepare.load(Ordering::SeqCst), 1);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);

    let second = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
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
fn runtime_rejects_operation_output_outside_scheduled_residency() -> Result<(), Box<dyn StdError>> {
    let expected_domain = AllocationDomainId::fresh();
    let foreign_domain = AllocationDomainId::fresh();
    let mut backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(expected_domain)))
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let counters = Arc::new(ExtensionCounters::default());
    *counters
        .foreign_output_domain
        .lock()
        .expect("foreign output domain lock") = Some(foreign_domain);
    let runtime = runtime_with_counting_extension(&backend, Arc::clone(&counters))?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let error = run_compiled(&mut [&mut backend], &runtime, &program, &[&input]).unwrap_err();

    let contract = error
        .source()
        .and_then(|source| source.downcast_ref::<EngineExecutionContractError>())
        .expect("typed engine execution contract source");
    assert!(matches!(
        contract,
        EngineExecutionContractError::OutputResidencyMismatch {
            output_slot: 1,
            allocation_domain: Some(actual),
            ..
        } if *actual == foreign_domain
    ));
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn runtime_run_prepared_reports_rejected_physical_input_residency() -> Result<(), Box<dyn StdError>>
{
    let mut backend = CpuEngine::serial()
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let host_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let prepared = prepare(&[&backend], &runtime, &program, &[&host_input])?;
    let foreign_domain = AllocationDomainId::fresh();
    let foreign_input = TestAllocationDomain(foreign_domain).allocate(DType::F64, &[2])?;

    let error =
        run_prepared(&mut [&mut backend], &runtime, &prepared, &[&foreign_input]).unwrap_err();

    let contract = error
        .source()
        .and_then(|source| source.downcast_ref::<InputIngressContractError>())
        .expect("typed input ingress contract source");
    assert!(matches!(
        contract,
        InputIngressContractError::ResidencyMismatch {
            input_slot: 0,
            backend_family: Some("tenferro-test.allocation-domain"),
            allocation_domain: Some(actual),
            ..
        } if *actual == foreign_domain
    ));
    Ok(())
}

#[test]
fn runtime_input_ingress_tracks_allocation_domain_across_prepared_cache_reuse(
) -> Result<(), Box<dyn StdError>> {
    let first_domain = AllocationDomainId::fresh();
    let second_domain = AllocationDomainId::fresh();
    let mut first_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-domain-engine.v1").expect("test engine id"),
        );
    let mut second_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(second_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.b-domain-engine.v1").expect("test engine id"),
        );
    let first_engine = "tenferro-test.a-domain-engine.v1";
    let second_engine = "tenferro-test.b-domain-engine.v1";
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let counters = Arc::new(ExtensionCounters::default());
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(first_domain),
    ));
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &first_backend,
        first_engine,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &second_backend,
        second_engine,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint(second_engine, storage.clone())?,
        transfer_endpoint(first_engine, storage.clone())?,
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.domain-cache")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.a-domain-engine.v1")?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let first_input = TestAllocationDomain(first_domain).allocate(DType::F64, &[2])?;
    let second_input = TestAllocationDomain(second_domain).allocate(DType::F64, &[2])?;

    for input in [&first_input, &second_input, &first_input, &second_input] {
        let output = run_compiled(
            &mut [&mut first_backend, &mut second_backend],
            &runtime,
            &program,
            &[input],
        )?;
        assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    }

    assert_eq!(
        transfer.calls(),
        2,
        "only the second allocation domain requires transfer to the selected engine"
    );
    assert_eq!(
        counters.prepare.load(Ordering::SeqCst),
        2,
        "each physical ingress schedule is prepared once and then reused"
    );
    Ok(())
}

#[test]
fn runtime_input_ingress_prefers_candidate_with_route_to_first_consumer(
) -> Result<(), Box<dyn StdError>> {
    let mut dead_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.a-dead-ingress.v1").expect("test engine id"),
    );
    let mut routed_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.b-routed-ingress.v1").expect("test engine id"),
    );
    let mut consumer_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.z-routed-consumer.v1").expect("test engine id"),
    );
    let dead_storage = StorageClass::new("tenferro-test.storage.dead-ingress.v1")?;
    let routed_storage = StorageClass::new("tenferro-test.storage.routed-ingress.v1")?;
    let consumer_storage = StorageClass::new("tenferro-test.storage.routed-consumer.v1")?;
    let transfer = Arc::new(RecordingTransferProvider::new(
        routed_storage.clone(),
        consumer_storage.clone(),
    ));
    let counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &dead_backend,
        "tenferro-test.a-dead-ingress.v1",
        dead_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.b-routed-ingress.v1",
        routed_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &consumer_backend,
        "tenferro-test.z-routed-consumer.v1",
        consumer_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.b-routed-ingress.v1", routed_storage.clone())?,
        transfer_endpoint("tenferro-test.z-routed-consumer.v1", consumer_storage)?,
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.routed-ingress")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.z-routed-consumer.v1")?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = run_compiled(
        &mut [
            &mut dead_backend,
            &mut routed_backend,
            &mut consumer_backend,
        ],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [3.0, 5.0]);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(transfer.requests()[0].source_storage_class, routed_storage);
    Ok(())
}

#[test]
fn runtime_operation_placement_retries_after_route_specific_ingress_failure(
) -> Result<(), Box<dyn StdError>> {
    let first_domain = AllocationDomainId::fresh();
    let second_domain = AllocationDomainId::fresh();
    let mut first_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-first-capable.v1").expect("test engine id"),
        );
    let second_allocation_domain = Arc::new(TestAllocationDomain(second_domain));
    let mut second_backend = CpuEngine::serial()
        .with_allocation_domain(second_allocation_domain.clone())
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.b-second-reachable.v1").expect("test engine id"),
        );
    let shared_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let first_counters = Arc::new(ExtensionCounters::default());
    let second_counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &first_backend,
        "tenferro-test.a-first-capable.v1",
        shared_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_backend,
        "tenferro-test.b-second-reachable.v1",
        shared_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-retry.first-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.a-first-capable.v1")?,
        counters: Arc::clone(&first_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-retry.second-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.b-second-reachable.v1")?,
        counters: Arc::clone(&second_counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = second_allocation_domain.allocate(DType::F64, &[2])?;

    let output = run_compiled(
        &mut [&mut first_backend, &mut second_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    assert_eq!(
        TensorRead::from_tensor(&output[0]).allocation_domain(),
        Some(second_domain)
    );
    assert_eq!(first_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(second_counters.execute.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn runtime_operation_placement_retries_reachable_later_storage() -> Result<(), Box<dyn StdError>> {
    let first_domain = AllocationDomainId::fresh();
    let second_domain = AllocationDomainId::fresh();
    let mut first_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-first-storage-capable.v1").expect("test engine id"),
        );
    let second_allocation_domain = Arc::new(TestAllocationDomain(second_domain));
    let mut second_backend = CpuEngine::serial()
        .with_allocation_domain(second_allocation_domain.clone())
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.b-second-storage-reachable.v1").expect("test engine id"),
        );
    let first_counters = Arc::new(ExtensionCounters::default());
    let second_counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &first_backend,
        "tenferro-test.a-first-storage-capable.v1",
        "tenferro-test.storage.first-unreachable.v1",
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_backend,
        "tenferro-test.b-second-storage-reachable.v1",
        CPU_STORAGE_CLASS_ID,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.storage-retry.first-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.a-first-storage-capable.v1")?,
        counters: Arc::clone(&first_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.storage-retry.second-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.b-second-storage-reachable.v1")?,
        counters: Arc::clone(&second_counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = second_allocation_domain.allocate(DType::F64, &[2])?;

    let output = run_compiled(
        &mut [&mut first_backend, &mut second_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    assert_eq!(first_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(second_counters.execute.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn runtime_operation_placement_explores_complete_multi_op_combinations(
) -> Result<(), Box<dyn StdError>> {
    let mut a_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.complete-a.v1").expect("test engine id"),
        );
    let b_allocation_domain = Arc::new(TestAllocationDomain(AllocationDomainId::fresh()));
    let mut b_backend = CpuEngine::serial()
        .with_allocation_domain(b_allocation_domain.clone())
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.complete-b.v1").expect("test engine id"),
        );
    let mut c_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.complete-c.v1").expect("test engine id"),
        );
    let d_allocation_domain = Arc::new(TestAllocationDomain(AllocationDomainId::fresh()));
    let mut d_backend = CpuEngine::serial()
        .with_allocation_domain(d_allocation_domain.clone())
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.complete-d.v1").expect("test engine id"),
        );
    let a_storage = StorageClass::new("tenferro-test.storage.complete-a.v1")?;
    let b_storage = StorageClass::new("tenferro-test.storage.complete-b.v1")?;
    let c_storage = StorageClass::new("tenferro-test.storage.complete-c.v1")?;
    let d_storage = StorageClass::new("tenferro-test.storage.complete-d.v1")?;
    let a_counters = Arc::new(ExtensionCounters::default());
    let b_counters = Arc::new(ExtensionCounters::default());
    let c_counters = Arc::new(ExtensionCounters::default());
    let d_counters = Arc::new(ExtensionCounters::default());
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        b_storage.clone(),
        d_storage.clone(),
        Arc::clone(&d_allocation_domain) as Arc<dyn SharedTensorAllocationDomain>,
    ));

    let mut builder = Runtime::builder();
    for (backend, engine_id, storage) in [
        (&a_backend, "tenferro-test.complete-a.v1", &a_storage),
        (&b_backend, "tenferro-test.complete-b.v1", &b_storage),
        (&c_backend, "tenferro-test.complete-c.v1", &c_storage),
        (&d_backend, "tenferro-test.complete-d.v1", &d_storage),
    ] {
        builder.register_engine(cpu_registration_with_storage_id(
            backend,
            engine_id,
            storage.as_str(),
            CoreCapabilityBundle::default(),
            CpuRegistrationState::executable(),
        )?)?;
    }
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.complete-b.v1", b_storage.clone())?,
        transfer_endpoint("tenferro-test.complete-d.v1", d_storage.clone())?,
        transfer.clone(),
    )?;
    for (module_id, family_id, engine_id, counters) in [
        (
            "tenferro-test.complete.first-a",
            COUNTING_EXTENSION_FAMILY,
            "tenferro-test.complete-a.v1",
            Arc::clone(&a_counters),
        ),
        (
            "tenferro-test.complete.first-b",
            COUNTING_EXTENSION_FAMILY,
            "tenferro-test.complete-b.v1",
            Arc::clone(&b_counters),
        ),
        (
            "tenferro-test.complete.second-c",
            DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
            "tenferro-test.complete-c.v1",
            Arc::clone(&c_counters),
        ),
        (
            "tenferro-test.complete.second-d",
            DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
            "tenferro-test.complete-d.v1",
            Arc::clone(&d_counters),
        ),
    ] {
        builder.install_extension_module(Arc::new(CountingExtensionModule {
            module_id: ExtensionModuleId::new(module_id)?,
            family_id,
            engine_id: EngineId::new(engine_id)?,
            counters,
        }))?;
    }
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let first = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("first extension has one output");
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        }),
        &[&first],
    )?
    .pop()
    .expect("second extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = b_allocation_domain.allocate(DType::F64, &[2])?;

    let output = run_compiled(
        &mut [
            &mut a_backend,
            &mut b_backend,
            &mut c_backend,
            &mut d_backend,
        ],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    assert_eq!(a_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(b_counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(c_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(d_counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(transfer.requests()[0].source_storage_class, b_storage);
    assert_eq!(transfer.requests()[0].destination_storage_class, d_storage);
    Ok(())
}

#[test]
fn runtime_operation_placement_enumerates_non_default_registered_storage(
) -> Result<(), Box<dyn StdError>> {
    let domain = Arc::new(TestAllocationDomain(AllocationDomainId::fresh()));
    let mut backend = CpuEngine::serial()
        .with_allocation_domain(domain.clone())
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.secondary-storage-engine.v1").expect("test engine id"),
        );
    let default_storage = StorageClass::new("tenferro-test.storage.default-unreachable.v1")?;
    let secondary_storage = StorageClass::new("tenferro-test.storage.secondary-reachable.v1")?;
    let counters = Arc::new(ExtensionCounters::default());
    let engine_id = "tenferro-test.secondary-storage-engine.v1";
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_classes(
        &backend,
        engine_id,
        vec![default_storage.clone(), secondary_storage.clone()],
        default_storage,
        secondary_storage.clone(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.secondary-storage-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.secondary-storage-engine.v1")?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = domain.allocate(DType::F64, &[2])?;

    let output = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;

    assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(
        TensorRead::from_tensor(&output[0]).allocation_domain(),
        backend.allocation_domain()
    );
    Ok(())
}

#[test]
fn runtime_operation_placement_reports_typed_error_after_all_ingress_routes_fail(
) -> Result<(), Box<dyn StdError>> {
    let mut first_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-unreachable-capable.v1").expect("test engine id"),
        );
    let mut second_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.b-unreachable-capable.v1").expect("test engine id"),
        );
    let first_counters = Arc::new(ExtensionCounters::default());
    let second_counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &first_backend,
        "tenferro-test.a-unreachable-capable.v1",
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &second_backend,
        "tenferro-test.b-unreachable-capable.v1",
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-failure.first-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.a-unreachable-capable.v1")?,
        counters: Arc::clone(&first_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-failure.second-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.b-unreachable-capable.v1")?,
        counters: Arc::clone(&second_counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = TestAllocationDomain(AllocationDomainId::fresh()).allocate(DType::F64, &[2])?;

    let error = run_compiled(
        &mut [&mut first_backend, &mut second_backend],
        &runtime,
        &program,
        &[&input],
    )
    .unwrap_err();
    eprintln!("all-ingress-failure error: {error:?}");

    assert!(matches!(
        error.reason(),
        RuntimeFailureReasonRef::NoInputIngress { input_index: 0, .. }
    ));
    assert_eq!(first_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(second_counters.execute.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn runtime_input_ingress_covers_all_split_consumers() -> Result<(), Box<dyn StdError>> {
    let mut dead_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.a-split-dead.v1").expect("test engine id"),
    );
    let mut routed_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.b-split-routed.v1").expect("test engine id"),
    );
    let mut first_consumer_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.y-split-first.v1").expect("test engine id"),
    );
    let mut second_consumer_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.z-split-second.v1").expect("test engine id"),
    );
    let dead_storage = StorageClass::new("tenferro-test.storage.split-dead.v1")?;
    let routed_storage = StorageClass::new("tenferro-test.storage.split-routed.v1")?;
    let first_storage = StorageClass::new("tenferro-test.storage.split-first.v1")?;
    let second_storage = StorageClass::new("tenferro-test.storage.split-second.v1")?;
    let dead_to_first = Arc::new(RecordingTransferProvider::new(
        dead_storage.clone(),
        first_storage.clone(),
    ));
    let routed_to_first = Arc::new(RecordingTransferProvider::new(
        routed_storage.clone(),
        first_storage.clone(),
    ));
    let routed_to_second = Arc::new(RecordingTransferProvider::new(
        routed_storage.clone(),
        second_storage.clone(),
    ));
    let first_counters = Arc::new(ExtensionCounters::default());
    let second_counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &dead_backend,
        "tenferro-test.a-split-dead.v1",
        dead_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.b-split-routed.v1",
        routed_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &first_consumer_backend,
        "tenferro-test.y-split-first.v1",
        first_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_consumer_backend,
        "tenferro-test.z-split-second.v1",
        second_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.a-split-dead.v1", dead_storage)?,
        transfer_endpoint("tenferro-test.y-split-first.v1", first_storage.clone())?,
        dead_to_first.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.b-split-routed.v1", routed_storage.clone())?,
        transfer_endpoint("tenferro-test.y-split-first.v1", first_storage)?,
        routed_to_first.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.b-split-routed.v1", routed_storage.clone())?,
        transfer_endpoint("tenferro-test.z-split-second.v1", second_storage)?,
        routed_to_second.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.split-first")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.y-split-first.v1")?,
        counters: Arc::clone(&first_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.split-second")?,
        family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.z-split-second.v1")?,
        counters: Arc::clone(&second_counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;
    let first = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("first extension has one output");
    let second = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("second extension has one output");
    let program = GraphCompiler::new().compile_many(&[&first, &second])?;

    let output = run_compiled(
        &mut [
            &mut dead_backend,
            &mut routed_backend,
            &mut first_consumer_backend,
            &mut second_consumer_backend,
        ],
        &runtime,
        &program,
        &[],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [3.0, 5.0]);
    assert_eq!(tensor_f64_values(&output[1])?, [3.0, 5.0]);
    assert_eq!(dead_to_first.calls(), 0);
    assert_eq!(routed_to_first.calls(), 1);
    assert_eq!(routed_to_second.calls(), 1);
    assert_eq!(
        routed_to_first.requests()[0].source_storage_class,
        routed_storage
    );
    Ok(())
}

#[test]
fn runtime_input_ingress_covers_synthesized_root_instructions() -> Result<(), Box<dyn StdError>> {
    let root_domain = AllocationDomainId::fresh();
    let source_domain = AllocationDomainId::fresh();
    let core_domain = AllocationDomainId::fresh();
    let root_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(root_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-synth-root.v1").expect("test engine id"),
        );
    let dead_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.b-synth-dead.v1").expect("test engine id"),
        );
    let routed_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.c-synth-routed.v1").expect("test engine id"),
        );
    let core_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.z-synth-core.v1").expect("test engine id"),
        );
    let root_storage = StorageClass::new("tenferro-test.storage.synth-root.v1")?;
    let dead_storage = StorageClass::new("tenferro-test.storage.synth-dead.v1")?;
    let routed_storage = StorageClass::new("tenferro-test.storage.synth-routed.v1")?;
    let core_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let dead_to_core = Arc::new(RecordingTransferProvider::materializing(
        dead_storage.clone(),
        core_storage.clone(),
        test_domain(core_domain),
    ));
    let routed_to_root = Arc::new(RecordingTransferProvider::materializing(
        routed_storage.clone(),
        root_storage.clone(),
        test_domain(root_domain),
    ));
    let routed_to_core = Arc::new(RecordingTransferProvider::materializing(
        routed_storage.clone(),
        core_storage.clone(),
        test_domain(core_domain),
    ));
    let root_to_core = Arc::new(RecordingTransferProvider::materializing(
        root_storage.clone(),
        core_storage.clone(),
        test_domain(core_domain),
    ));
    let root_counters = Arc::new(ExtensionCounters::default());
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &root_backend,
        "tenferro-test.a-synth-root.v1",
        root_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &dead_backend,
        "tenferro-test.b-synth-dead.v1",
        dead_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.c-synth-routed.v1",
        routed_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        "tenferro-test.z-synth-core.v1",
        core_storage.as_str(),
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.b-synth-dead.v1", dead_storage)?,
        transfer_endpoint("tenferro-test.z-synth-core.v1", core_storage.clone())?,
        dead_to_core.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.c-synth-routed.v1", routed_storage.clone())?,
        transfer_endpoint("tenferro-test.a-synth-root.v1", root_storage.clone())?,
        routed_to_root.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.c-synth-routed.v1", routed_storage.clone())?,
        transfer_endpoint("tenferro-test.z-synth-core.v1", core_storage.clone())?,
        routed_to_core.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint("tenferro-test.a-synth-root.v1", root_storage)?,
        transfer_endpoint("tenferro-test.z-synth-core.v1", core_storage)?,
        root_to_core,
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.synth-root")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.a-synth-root.v1")?,
        counters: Arc::clone(&root_counters),
    }))?;
    let runtime = builder.build()?;

    let root_input = TracedTensor::from_tensor_concrete_shape(
        TestAllocationDomain(root_domain).allocate(DType::F64, &[2])?,
    )?;
    let root_output = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&root_input],
    )?
    .pop()
    .expect("root extension has one output");
    let source_input = TracedTensor::from_tensor_concrete_shape(
        TestAllocationDomain(source_domain).allocate(DType::F64, &[2, 2])?,
    )?;
    let synthesized_output = source_input.transpose(&[1, 0])?.conj()?;
    let program = GraphCompiler::new().compile_many(&[&root_output, &synthesized_output])?;

    let _prepared = prepare(
        &[&root_backend, &dead_backend, &routed_backend, &core_backend],
        &runtime,
        &program,
        &[],
    )?;

    assert_eq!(dead_to_core.calls(), 0);
    assert_eq!(routed_to_root.calls(), 0);
    assert_eq!(routed_to_core.calls(), 0);
    Ok(())
}

#[test]
fn runtime_run_compiled_dispatches_same_storage_extension_on_selected_engine(
) -> Result<(), Box<dyn StdError>> {
    let core_domain = AllocationDomainId::fresh();
    let extension_domain = AllocationDomainId::fresh();
    let mut core_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.core-engine.v1").expect("test engine id"),
        );
    let mut extension_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.extension-engine.v1").expect("test engine id"),
        );
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-engine.v1";
    let extension_engine_id = "tenferro-test.extension-engine.v1";
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        test_domain(extension_domain),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &core_backend,
        core_engine_id,
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &extension_backend,
        extension_engine_id,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint(core_engine_id, storage.clone())?,
        transfer_endpoint(extension_engine_id, storage.clone())?,
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.per-op-module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.extension-engine.v1")
            .map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let source_event_domain = snapshot
        .engine(&EngineId::new("tenferro-test.core-engine.v1")?)
        .expect("source engine")
        .event_domain_id();
    let destination_event_domain = snapshot
        .engine(&EngineId::new("tenferro-test.extension-engine.v1")?)
        .expect("destination engine")
        .event_domain_id();

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&sum],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = run_compiled(
        &mut [&mut core_backend, &mut extension_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [6.0, 10.0]);
    assert_eq!(counters.prepare.load(Ordering::SeqCst), 1);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(
        transfer.materialized_domains(),
        vec![(None, Some(extension_domain))]
    );
    assert_eq!(
        transfer.requests(),
        vec![RecordedTransferRequest {
            source_engine_id: EngineId::new("tenferro-test.core-engine.v1")?,
            source_provider_device_identity: test_provider_device_identity(core_engine_id)?,
            source_event_domain_id: source_event_domain,
            source_storage_class: storage.clone(),
            destination_engine_id: EngineId::new("tenferro-test.extension-engine.v1")?,
            destination_provider_device_identity: test_provider_device_identity(
                extension_engine_id
            )?,
            destination_event_domain_id: destination_event_domain,
            destination_storage_class: storage,
        }]
    );
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
    let mut core_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.core-transfer-source.v1").expect("test engine id"),
        );
    let mut extension_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.extension-transfer-destination.v1")
                .expect("test engine id"),
        );
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-transfer-source.v1";
    let extension_engine_id = "tenferro-test.extension-transfer-destination.v1";
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.destination.v1")?;
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        source_storage.clone(),
        destination_storage.clone(),
        test_domain(extension_domain),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &core_backend,
        core_engine_id,
        source_storage.as_str(),
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable(),
        &format!("test-engine:{core_engine_id}"),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
        &format!("test-engine:{extension_engine_id}"),
    )?)?;
    builder.register_transfer_provider(
        TransferEndpoint::new(
            EngineId::new("tenferro-test.core-transfer-source.v1")?,
            source_storage.clone(),
        ),
        TransferEndpoint::new(
            EngineId::new("tenferro-test.extension-transfer-destination.v1")?,
            destination_storage.clone(),
        ),
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.transfer-module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.extension-transfer-destination.v1")
            .map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let source_event_domain = snapshot
        .engine(&EngineId::new("tenferro-test.core-transfer-source.v1")?)
        .expect("source engine")
        .event_domain_id();
    let destination_event_domain = snapshot
        .engine(&EngineId::new(
            "tenferro-test.extension-transfer-destination.v1",
        )?)
        .expect("destination engine")
        .event_domain_id();
    assert_ne!(source_event_domain, destination_event_domain);

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&sum],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = run_compiled(
        &mut [&mut core_backend, &mut extension_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [6.0, 10.0]);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(
        transfer.materialized_domains(),
        vec![(None, Some(extension_domain))]
    );
    assert_eq!(
        transfer.requests(),
        vec![RecordedTransferRequest {
            source_engine_id: EngineId::new("tenferro-test.core-transfer-source.v1")?,
            source_provider_device_identity: test_provider_device_identity(core_engine_id)?,
            source_event_domain_id: source_event_domain,
            source_storage_class: source_storage,
            destination_engine_id: EngineId::new(
                "tenferro-test.extension-transfer-destination.v1"
            )?,
            destination_provider_device_identity: test_provider_device_identity(
                extension_engine_id
            )?,
            destination_event_domain_id: destination_event_domain,
            destination_storage_class: destination_storage,
        }]
    );
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(
        *counters.last_execute_domain.lock().expect("domain lock"),
        Some(extension_domain)
    );
    Ok(())
}

#[test]
fn runtime_run_compiled_transfers_input_from_validated_ingress_to_first_consumer(
) -> Result<(), Box<dyn StdError>> {
    let ingress_domain = AllocationDomainId::fresh();
    let consumer_domain = AllocationDomainId::fresh();
    let mut ingress_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(ingress_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.a-input-ingress.v1").expect("test engine id"),
        );
    let mut consumer_backend = CpuEngine::serial()
        .with_allocation_domain(Arc::new(TestAllocationDomain(consumer_domain)))
        .with_runtime_engine_id(
            EngineId::new("tenferro-test.z-input-consumer.v1").expect("test engine id"),
        );
    let counters = Arc::new(ExtensionCounters::default());
    let ingress_engine_id = "tenferro-test.a-input-ingress.v1";
    let consumer_engine_id = "tenferro-test.z-input-consumer.v1";
    let ingress_storage = StorageClass::new("tenferro-test.storage.input-ingress.v1")?;
    let consumer_storage = StorageClass::new("tenferro-test.storage.input-consumer.v1")?;
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        ingress_storage.clone(),
        consumer_storage.clone(),
        test_domain(consumer_domain),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &ingress_backend,
        ingress_engine_id,
        ingress_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &consumer_backend,
        consumer_engine_id,
        consumer_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint(ingress_engine_id, ingress_storage.clone())?,
        transfer_endpoint(consumer_engine_id, consumer_storage.clone())?,
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.input-consumer")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.z-input-consumer.v1")
            .map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let ingress_event_domain = snapshot
        .engine(&EngineId::new("tenferro-test.a-input-ingress.v1")?)
        .expect("ingress engine")
        .event_domain_id();
    let consumer_event_domain = snapshot
        .engine(&EngineId::new("tenferro-test.z-input-consumer.v1")?)
        .expect("consumer engine")
        .event_domain_id();

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = run_compiled(
        &mut [&mut ingress_backend, &mut consumer_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(tensor_f64_values(&output[0])?, [3.0, 5.0]);
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(
        transfer.materialized_domains(),
        vec![(None, Some(consumer_domain))]
    );
    assert_ne!(
        transfer.materialized_domains()[0].0,
        transfer.materialized_domains()[0].1,
        "the provider must materialize into the destination allocation domain"
    );
    assert_eq!(
        transfer.requests(),
        vec![RecordedTransferRequest {
            source_engine_id: EngineId::new("tenferro-test.a-input-ingress.v1")?,
            source_provider_device_identity: test_provider_device_identity(ingress_engine_id)?,
            source_event_domain_id: ingress_event_domain,
            source_storage_class: ingress_storage,
            destination_engine_id: EngineId::new("tenferro-test.z-input-consumer.v1")?,
            destination_provider_device_identity: test_provider_device_identity(
                consumer_engine_id
            )?,
            destination_event_domain_id: consumer_event_domain,
            destination_storage_class: consumer_storage,
        }]
    );
    Ok(())
}

#[test]
fn runtime_rejects_faulty_transfer_provider_outputs_with_typed_contract_errors(
) -> Result<(), Box<dyn StdError>> {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    for fault in [
        FaultyTransferOutput::DType,
        FaultyTransferOutput::Shape,
        FaultyTransferOutput::Placement,
        FaultyTransferOutput::BufferLength,
        FaultyTransferOutput::Residency,
    ] {
        let mut ingress_backend = CpuEngine::serial().with_runtime_engine_id(
            EngineId::new("tenferro-test.a-faulty-ingress.v1").expect("test engine id"),
        );
        let consumer_domain = AllocationDomainId::fresh();
        let mut consumer_backend = CpuEngine::serial()
            .with_allocation_domain(Arc::new(TestAllocationDomain(consumer_domain)))
            .with_runtime_engine_id(
                EngineId::new("tenferro-test.z-faulty-consumer.v1").expect("test engine id"),
            );
        let counters = Arc::new(ExtensionCounters::default());
        let ingress_storage = StorageClass::new("tenferro-test.storage.faulty-ingress.v1")?;
        let consumer_storage = StorageClass::new("tenferro-test.storage.faulty-consumer.v1")?;
        let mut builder = Runtime::builder();
        builder.register_engine(cpu_registration_with_storage_id(
            &ingress_backend,
            "tenferro-test.a-faulty-ingress.v1",
            ingress_storage.as_str(),
            CoreCapabilityBundle::default(),
            CpuRegistrationState::executable(),
        )?)?;
        builder.register_engine(cpu_registration_with_storage_id(
            &consumer_backend,
            "tenferro-test.z-faulty-consumer.v1",
            consumer_storage.as_str(),
            CoreCapabilityBundle::default(),
            CpuRegistrationState::executable(),
        )?)?;
        builder.register_transfer_provider(
            transfer_endpoint("tenferro-test.a-faulty-ingress.v1", ingress_storage)?,
            transfer_endpoint("tenferro-test.z-faulty-consumer.v1", consumer_storage)?,
            Arc::new(FaultyTransferProvider { fault }),
        )?;
        builder.install_extension_module(Arc::new(CountingExtensionModule {
            module_id: ExtensionModuleId::new("tenferro-test.counting-extension.faulty-output")
                .map_err(RuntimeConfigError::from)?,
            family_id: COUNTING_EXTENSION_FAMILY,
            engine_id: EngineId::new("tenferro-test.z-faulty-consumer.v1")
                .map_err(RuntimeConfigError::from)?,
            counters: Arc::clone(&counters),
        }))?;
        let runtime = builder.build()?;

        let error = run_compiled(
            &mut [&mut ingress_backend, &mut consumer_backend],
            &runtime,
            &program,
            &[&input],
        )
        .unwrap_err();
        let transfer_error = error
            .source()
            .and_then(|source| source.downcast_ref::<TransferError>())
            .expect("typed transfer error source");
        let TransferError::ProviderContract { source } = transfer_error else {
            panic!("expected provider contract error for {fault:?}: {transfer_error}");
        };
        let chained_contract = transfer_error
            .source()
            .and_then(|source| source.downcast_ref::<TransferProviderContractError>())
            .expect("provider contract remains available through Error::source");
        assert_eq!(
            std::mem::discriminant(chained_contract),
            std::mem::discriminant(source)
        );
        assert!(
            matches!(
                (fault, source),
                (
                    FaultyTransferOutput::DType,
                    TransferProviderContractError::DTypeMismatch { .. }
                ) | (
                    FaultyTransferOutput::Shape,
                    TransferProviderContractError::ShapeMismatch { .. }
                ) | (
                    FaultyTransferOutput::Placement,
                    TransferProviderContractError::DestinationPlacementMismatch { .. }
                ) | (
                    FaultyTransferOutput::BufferLength,
                    TransferProviderContractError::InvalidBufferLength { .. }
                ) | (
                    FaultyTransferOutput::Residency,
                    TransferProviderContractError::DestinationResidencyMismatch { .. }
                )
            ),
            "unexpected contract error for {fault:?}: {source}"
        );
        assert_eq!(
            counters.execute.load(Ordering::SeqCst),
            0,
            "faulty transfer output must not reach the consumer"
        );
    }
    Ok(())
}

#[test]
fn runtime_run_compiled_split_use_retains_source_and_destination_values(
) -> Result<(), Box<dyn StdError>> {
    let mut core_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.core-split-source.v1").expect("test engine id"),
    );
    let mut extension_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.extension-split-destination.v1").expect("test engine id"),
    );
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-split-source.v1";
    let extension_engine_id = "tenferro-test.extension-split-destination.v1";
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.split-destination.v1")?;
    let forward = Arc::new(RecordingTransferProvider::new(
        source_storage.clone(),
        destination_storage.clone(),
    ));
    let reverse = Arc::new(RecordingTransferProvider::new(
        destination_storage.clone(),
        source_storage.clone(),
    ));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        core_engine_id,
        source_storage.as_str(),
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint(core_engine_id, source_storage.clone())?,
        transfer_endpoint(extension_engine_id, destination_storage.clone())?,
        forward.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint(extension_engine_id, destination_storage.clone())?,
        transfer_endpoint(core_engine_id, source_storage)?,
        reverse.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.split-module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.extension-split-destination.v1")
            .map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let extension = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&sum],
    )?
    .pop()
    .expect("extension has one output");
    let y = (&extension + &sum)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let output = run_compiled(
        &mut [&mut core_backend, &mut extension_backend],
        &runtime,
        &program,
        &[&input],
    )?;

    assert_eq!(output[0].as_slice::<f64>()?, &[12.0, 20.0]);
    assert_eq!(forward.calls(), 1);
    assert_eq!(
        reverse.calls(),
        1,
        "the source-side sum must remain available after its split transfer"
    );
    assert_eq!(counters.execute.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn runtime_run_compiled_reports_missing_transfer_provider_for_cross_storage(
) -> Result<(), Box<dyn StdError>> {
    let mut core_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.core-missing-transfer-source.v1").expect("test engine id"),
    );
    let mut extension_backend = CpuEngine::serial().with_runtime_engine_id(
        EngineId::new("tenferro-test.extension-missing-transfer-destination.v1")
            .expect("test engine id"),
    );
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-missing-transfer-source.v1";
    let extension_engine_id = "tenferro-test.extension-missing-transfer-destination.v1";
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.missing-destination.v1")?;
    let source_endpoint = TransferEndpoint::new(
        EngineId::new("tenferro-test.core-missing-transfer-source.v1")?,
        source_storage.clone(),
    );
    let expected_destination_endpoint = TransferEndpoint::new(
        EngineId::new("tenferro-test.extension-missing-transfer-destination.v1")?,
        destination_storage.clone(),
    );

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        core_engine_id,
        CPU_STORAGE_CLASS_ID,
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable(),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable(),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.no-transfer-module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new("tenferro-test.extension-missing-transfer-destination.v1")
            .map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let sum = (&x + &x)?;
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&sum],
    )?
    .pop()
    .expect("extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let error = run_compiled(
        &mut [&mut core_backend, &mut extension_backend],
        &runtime,
        &program,
        &[&input],
    )
    .unwrap_err();

    assert!(error.to_string().contains("no direct transfer provider"));
    assert!(
        error.to_string().contains(core_engine_id),
        "missing-route diagnostics must name the source endpoint: {error}"
    );
    assert!(
        error.to_string().contains(extension_engine_id),
        "missing-route diagnostics must name the destination endpoint: {error}"
    );
    let prepare_error = error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("typed prepare error source");
    assert!(matches!(
        prepare_error,
        PrepareError::MissingTransferProvider {
            destination_endpoint: actual_destination_endpoint,
            available_source_endpoints,
            ..
        } if available_source_endpoints == std::slice::from_ref(&source_endpoint)
            && actual_destination_endpoint == &expected_destination_endpoint
    ));
    assert_eq!(counters.execute.load(Ordering::SeqCst), 0);

    Ok(())
}

#[test]
fn runtime_run_compiled_uses_prepared_cache_on_second_call() -> Result<(), Box<dyn StdError>> {
    let mut backend = CpuEngine::serial()
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    let second = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_first.misses >= 1);
    assert_eq!(after_second.misses, after_first.misses);
    assert!(after_second.hits > after_first.hits);
    assert_eq!(after_second.entries, after_first.entries);

    Ok(())
}

#[test]
fn runtime_run_compiled_rejects_preparation_only_registration() -> Result<(), Box<dyn StdError>> {
    let mut backend = CpuEngine::serial()
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(
        &backend,
        CoreCapabilityBundle::default(),
        CpuRegistrationState::PreparationOnly,
    )?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let error = run_compiled(&mut [&mut backend], &runtime, &program, &[&input]).unwrap_err();

    assert_eq!(error.phase(), Some(ErrorPhase::Execution));
    assert!(StdError::source(&error).is_some());
    Ok(())
}

#[test]
fn runtime_run_compiled_reprepares_after_engine_reconfiguration() -> Result<(), Box<dyn StdError>> {
    let mut backend = CpuEngine::serial()
        .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
    assert_eq!(first[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    runtime.reconfigure(|edit| {
        let backend = CpuEngine::serial()
            .with_runtime_engine_id(EngineId::new(CPU_ENGINE_ID).expect("test engine id"));
        edit.replace_engine(cpu_registration(
            &backend,
            cpu_core_capabilities(&backend),
            CpuRegistrationState::executable(),
        )?)?;
        Ok(())
    })?;

    let second = run_compiled(&mut [&mut backend], &runtime, &program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_second.misses > after_first.misses);
    Ok(())
}

/// #1968: a declared placeholder the output does not depend on is rejected at
/// compile time instead of being silently dropped from the program inputs.
#[test]
fn compile_with_input_specs_rejects_unused_placeholder() {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let y = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let out = y.neg().unwrap();
    let err = GraphCompiler::new()
        .compile_with_input_specs(&out, &[(&y, DType::F64, &[3]), (&x, DType::F64, &[3])])
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("binding 1"), "{message}");
    assert!(GraphCompiler::new()
        .compile_with_input_specs(&out, &[(&y, DType::F64, &[3])])
        .is_ok());
}

#[test]
fn an_engine_without_in_place_view_reads_keeps_metadata_commands() -> Result<(), Box<dyn StdError>>
{
    // The test registration's `LayoutRuntime` keeps the default
    // `folds_metadata_views` (`false`), so the transpose stays a scheduled
    // command (#2226) and the program still computes the same values.
    let mut backend = CpuEngine::serial();
    let runtime = runtime_with_cpu(&backend)?;
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?)?;
    let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![3, 2],
        vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0],
    )?)?;
    let out = (&a.neg()?.transpose(&[1, 0])? + &b)?;
    let program = GraphCompiler::new().compile_many(&[&out])?;

    let prepared = prepare(&[&backend], &runtime, &program, &[])?;
    assert_eq!(
        prepared.execution_command_count(),
        3,
        "Negate, Transpose, Add"
    );
    let outputs = run_prepared(&mut [&mut backend], &runtime, &prepared, &[])?;
    assert_eq!(
        outputs[0].as_slice::<f64>()?,
        &[9.0, 17.0, 25.0, 38.0, 46.0, 54.0]
    );
    Ok(())
}
