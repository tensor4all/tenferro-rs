use std::any::Any;
use std::collections::{BTreeSet, HashSet};
use std::error::Error as StdError;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tenferro_cpu::CpuBackend;
use tenferro_ops::{
    ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
    ExtensionShapeContext, SymDim,
};
use tenferro_runtime::runtime::{
    EventDomainDriver, EventDomainRun, EventToken, ImmediateEventDomainDriver,
};
use tenferro_runtime::{
    assemble_executable_engine_registration, assemble_preparation_only_engine_registration,
    CoreCapabilityBundle, DType, DotGeneralPreparation, ElementwiseRuntime,
    EngineExecutionContractError, EngineId, EngineRegistration, EngineRegistrationMetadata,
    ErasedExecutionContext, Error, ErrorPhase, EventDomainId, ExecutableEngineRegistrationConfig,
    ExecutionContextIdentity, ExecutionInputs, ExtensionCacheStore, ExtensionEngine,
    ExtensionModule, ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar,
    ExtensionPlanningConfig, ExtensionPrepareRequest, GraphCompiler, HardwareClassId,
    IndexingRuntime, InputIngressContract, InputIngressContractError, InputPlacementContract,
    InputSignatureContract, LayoutRuntime, PreparationOnlyEngineRegistrationConfig,
    PrepareCapability, PrepareError, PreparedOperation, PreparedOperationBinding,
    PreparedOperationExecutor, PreparedOperationPlan, ProviderDeviceIdentity, ProviderId,
    ReductionRuntime, RegistrationKey, ResidentOutputContract, Runtime, RuntimeCacheOwner,
    RuntimeConfigError, RuntimeInputContract, RuntimeReconfigureError, SpecializationProjection,
    StorageClass, TracedTensor, TransferEndpoint, TransferError, TransferProvider,
    TransferProviderContractError, TransferRequest,
};
use tenferro_tensor::{
    AllocationDomainId, BackendBuffer, BackendSessionHost, Buffer, HostAccessError, HostReadGuard,
    HostWriteGuard, MemoryKind, Placement, SharedTensorAllocationDomain, Tensor, TensorRead,
    TypedTensor,
};

const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const COUNTING_EXTENSION_FAMILY: &str = "tenferro-test.counting-extension.v1";
const DOWNSTREAM_COUNTING_EXTENSION_FAMILY: &str = "tenferro-test.downstream-counting-extension.v1";
const ROUTE_HOST_CUDA0_FAMILY: &str = "tenferro-test.route-host-cuda0.v1";
const ROUTE_HOST_CUDA1_FAMILY: &str = "tenferro-test.route-host-cuda1.v1";
const ROUTE_CUDA0_HOST_FAMILY: &str = "tenferro-test.route-cuda0-host.v1";
const ROUTE_CUDA1_HOST_FAMILY: &str = "tenferro-test.route-cuda1-host.v1";

fn test_provider_device_identity(
    engine_id: &str,
) -> Result<ProviderDeviceIdentity, RuntimeConfigError> {
    Ok(ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.cpu").map_err(RuntimeConfigError::from)?,
        format!("test-engine:{engine_id}"),
    )?)
}

#[derive(Debug)]
struct RecordingEventToken {
    label: &'static str,
    origin: EventDomainId,
    events: Arc<Mutex<Vec<String>>>,
}

impl EventToken for RecordingEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.origin
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        self.events
            .lock()
            .expect("event log lock")
            .push(format!("{}:wait", self.label));
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum DrainBehavior {
    Success,
    ReturnError,
    Panic,
}

#[derive(Debug)]
struct RecordingEventDomainRun {
    label: &'static str,
    domain: EventDomainId,
    events: Arc<Mutex<Vec<String>>>,
    drain_behavior: DrainBehavior,
}

impl EventDomainRun for RecordingEventDomainRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> tenferro_runtime::Result<()>,
    ) -> tenferro_runtime::Result<Arc<dyn EventToken>> {
        self.events.lock().expect("event log lock").push(format!(
            "{}:enqueue:{}",
            self.label,
            dependencies.len()
        ));
        for dependency in dependencies {
            dependency.wait()?;
        }
        launch()?;
        Ok(Arc::new(RecordingEventToken {
            label: self.label,
            origin: self.domain,
            events: Arc::clone(&self.events),
        }))
    }

    fn drain(&mut self) -> tenferro_runtime::Result<()> {
        self.events
            .lock()
            .expect("event log lock")
            .push(format!("{}:drain", self.label));
        match self.drain_behavior {
            DrainBehavior::Success => Ok(()),
            DrainBehavior::ReturnError => Err(Error::Internal(format!(
                "{} event-domain drain failure",
                self.label
            ))),
            DrainBehavior::Panic => panic!("{} event-domain drain panic", self.label),
        }
    }
}

impl Drop for RecordingEventDomainRun {
    fn drop(&mut self) {
        self.events
            .lock()
            .expect("event log lock")
            .push(format!("{}:drop", self.label));
    }
}

#[derive(Debug)]
struct RecordingEventDomainDriver {
    label: &'static str,
    events: Arc<Mutex<Vec<String>>>,
    drain_behavior: DrainBehavior,
}

impl EventDomainDriver for RecordingEventDomainDriver {
    fn begin_run(
        &self,
        domain: EventDomainId,
    ) -> tenferro_runtime::Result<Box<dyn EventDomainRun>> {
        self.events
            .lock()
            .expect("event log lock")
            .push(format!("{}:begin", self.label));
        Ok(Box::new(RecordingEventDomainRun {
            label: self.label,
            domain,
            events: Arc::clone(&self.events),
            drain_behavior: self.drain_behavior,
        }))
    }
}

fn event_domain_with_drain_behavior(
    label: &'static str,
    events: &Arc<Mutex<Vec<String>>>,
    drain_behavior: DrainBehavior,
) -> Arc<dyn EventDomainDriver> {
    Arc::new(RecordingEventDomainDriver {
        label,
        events: Arc::clone(events),
        drain_behavior,
    })
}

fn recording_event_domain(
    label: &'static str,
    events: &Arc<Mutex<Vec<String>>>,
) -> Arc<dyn EventDomainDriver> {
    event_domain_with_drain_behavior(label, events, DrainBehavior::Success)
}

fn failing_drain_event_domain(
    label: &'static str,
    events: &Arc<Mutex<Vec<String>>>,
) -> Arc<dyn EventDomainDriver> {
    event_domain_with_drain_behavior(label, events, DrainBehavior::ReturnError)
}

fn cpu_ingress_contract(backend: &Arc<CpuBackend>, storage: &StorageClass) -> InputIngressContract {
    let allocation_domain = backend.allocation_domain();
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
    Executable {
        driver: Arc<dyn EventDomainDriver>,
        cache_owner: Option<Arc<dyn RuntimeCacheOwner>>,
    },
}

impl CpuRegistrationState {
    fn executable() -> Self {
        Self::Executable {
            driver: Arc::new(ImmediateEventDomainDriver::new()),
            cache_owner: None,
        }
    }

    fn executable_with_driver(driver: Arc<dyn EventDomainDriver>) -> Self {
        Self::Executable {
            driver,
            cache_owner: None,
        }
    }

    fn executable_with_cache_owner(
        driver: Arc<dyn EventDomainDriver>,
        cache_owner: Arc<dyn RuntimeCacheOwner>,
    ) -> Self {
        Self::Executable {
            driver,
            cache_owner: Some(cache_owner),
        }
    }
}

fn cpu_core_capabilities(backend: &CpuBackend) -> CoreCapabilityBundle {
    let backend = Arc::new(backend.clone());
    let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
    let reduction: Arc<dyn ReductionRuntime> = backend.clone();
    let indexing: Arc<dyn IndexingRuntime> = backend.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let layout: Arc<dyn LayoutRuntime> = backend.clone();
    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities
        .elementwise(elementwise)
        .reduction(reduction)
        .indexing(indexing)
        .dot_general(dot_general)
        .layout(layout);
    capabilities.build()
}

fn assemble_cpu_registration(
    backend: Arc<CpuBackend>,
    metadata: EngineRegistrationMetadata,
    ingress_storage: StorageClass,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    match state {
        CpuRegistrationState::PreparationOnly => assemble_preparation_only_engine_registration(
            PreparationOnlyEngineRegistrationConfig::new(
                metadata,
                ExecutionContextIdentity::of::<CpuBackend>(),
            ),
        ),
        CpuRegistrationState::Executable {
            driver,
            cache_owner,
        } => assemble_executable_engine_registration(ExecutableEngineRegistrationConfig::new(
            metadata,
            backend.as_ref().clone(),
            driver,
            cpu_ingress_contract(&backend, &ingress_storage),
            cache_owner,
        )),
    }
}

fn cpu_registration(
    backend: &CpuBackend,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(CPU_ENGINE_ID)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        capabilities,
    );
    assemble_cpu_registration(backend, metadata, storage, state)
}

fn runtime_with_cpu(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(
        backend,
        cpu_core_capabilities(backend),
        CpuRegistrationState::executable_with_cache_owner(
            Arc::new(ImmediateEventDomainDriver::new()),
            Arc::new(backend.clone()),
        ),
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
    let Tensor::F64(tensor) = tensor else {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "tensor_f64_values",
            DType::F64,
            tensor.dtype(),
        ));
    };
    match tensor.buffer() {
        Buffer::Host(values) => Ok(values.clone()),
        Buffer::Backend(buffer) => buffer
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
                };
                TypedTensor::from_buffer_col_major(
                    shape.to_vec(),
                    Buffer::Backend(Arc::new(buffer)),
                    Placement::default(),
                )
                .map(Tensor::$variant)
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
        }
    }
}

#[derive(Debug)]
struct TestDomainBuffer<T> {
    values: Arc<Mutex<Vec<T>>>,
    domain: AllocationDomainId,
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> BackendBuffer<T> for TestDomainBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.allocation-domain"
    }

    fn len(&self) -> usize {
        self.values.lock().expect("domain buffer lock").len()
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        Ok(HostReadGuard::new(
            self.values.lock().expect("domain buffer lock"),
        ))
    }

    fn map_write(&self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
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
    destination_backend: Option<CpuBackend>,
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

impl RecordingTransferProvider {
    fn new(source: StorageClass, destination: StorageClass) -> Self {
        Self {
            source,
            destination,
            calls: AtomicUsize::new(0),
            requests: Mutex::new(Vec::new()),
            destination_backend: None,
            materialized_domains: Mutex::new(Vec::new()),
        }
    }

    fn materializing(
        source: StorageClass,
        destination: StorageClass,
        destination_backend: CpuBackend,
    ) -> Self {
        Self {
            destination_backend: Some(destination_backend),
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
        let output = match &self.destination_backend {
            Some(backend) => {
                let domain = backend.shared_allocation_domain().ok_or_else(|| {
                    Error::Internal("materializing transfer requires an allocation domain".into())
                })?;
                let mut output =
                    domain.allocate(request.input().dtype(), request.input().shape())?;
                let source = request.input().as_tensor().ok_or_else(|| {
                    Error::Internal("test transfer expected an owned tensor".into())
                })?;
                let (Tensor::F64(source), Tensor::F64(destination)) = (source, &mut output) else {
                    return Err(Error::Internal(
                        "materializing test transfer currently expects f64 tensors".into(),
                    ));
                };
                let values = match source.buffer() {
                    Buffer::Host(values) => values.clone(),
                    Buffer::Backend(buffer) => buffer
                        .map_read()
                        .map_err(|source| {
                            tenferro_tensor::Error::host_access("test-transfer-read", source)
                        })?
                        .to_vec(),
                };
                match destination.buffer() {
                    Buffer::Host(_) => {
                        return Err(Error::Internal(
                            "test allocation domain returned host storage".into(),
                        ));
                    }
                    Buffer::Backend(buffer) => {
                        buffer
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
            None => request
                .input()
                .as_tensor()
                .ok_or_else(|| Error::Internal("test transfer expected an owned tensor".into()))?
                .duplicate()?,
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

#[derive(Debug)]
struct FailingTransferProvider {
    calls: AtomicUsize,
}

impl TransferProvider for FailingTransferProvider {
    fn transfer_blocking(&self, _request: TransferRequest<'_>) -> tenferro_runtime::Result<Tensor> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::runtime_state_source(
            "FailingTransferProvider::transfer_blocking",
            ErrorPhase::Execution,
            TestTransferFailure,
        ))
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
                let mut tensor = request
                    .input()
                    .as_tensor()
                    .ok_or_else(|| {
                        Error::Internal("test transfer expected an owned tensor".into())
                    })?
                    .duplicate()?;
                let Tensor::F64(tensor) = &mut tensor else {
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
                let buffer = Buffer::Backend(Arc::new(MutableLengthBuffer {
                    len: Arc::clone(&len),
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
                let buffer = Buffer::Backend(Arc::new(TestDomainBuffer::<f64> {
                    values: Arc::new(Mutex::new(vec![0.0; 2])),
                    domain,
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

#[derive(Debug, thiserror::Error)]
#[error("intentional transfer failure")]
struct TestTransferFailure;

#[derive(Debug)]
struct MutableLengthBuffer {
    len: Arc<AtomicUsize>,
}

impl BackendBuffer<f64> for MutableLengthBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.mutable-length"
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
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

impl BackendBuffer<f64> for DropTrackedBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.drop-tracked"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
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
}

impl ExtensionEngine for CountingExtensionEngine {
    fn family_id(&self) -> &'static str {
        self.family_id
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
        assert_eq!(request.operation().family_id(), self.family_id);
        assert_eq!(request.binding().engine_id(), &self.engine_id);
        self.counters.prepare.fetch_add(1, Ordering::SeqCst);
        let prepared = Arc::new(CountingPreparedOperation {
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
            counters: Arc::clone(&self.counters),
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
            .downcast_mut::<CpuBackend>(self.binding.context_identity())
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
            let buffer = Buffer::Backend(Arc::new(TestDomainBuffer::<f64> {
                values: Arc::new(Mutex::new(vec![0.0; len])),
                domain,
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
            let buffer = Buffer::Backend(Arc::new(DropTrackedBuffer {
                len,
                drops,
                domain,
                events,
            }));
            return Ok(vec![TypedTensor::<f64>::from_buffer_col_major(
                shape,
                buffer,
                Placement::default(),
            )?
            .into()]);
        }
        if inputs[0].allocation_domain() == backend.allocation_domain() {
            let tensor = inputs[0].as_tensor().ok_or_else(|| {
                Error::Internal("allocation-domain test executor expected an owned tensor".into())
            })?;
            let Tensor::F64(tensor) = tensor else {
                return Err(Error::Internal(
                    "allocation-domain test executor currently expects f64 tensors".into(),
                ));
            };
            return match tensor.buffer() {
                Buffer::Host(_) => Ok(vec![tensor.duplicate()?.into()]),
                Buffer::Backend(buffer) => Ok(vec![TypedTensor::from_buffer_col_major(
                    tensor.shape().to_vec(),
                    Buffer::Backend(Arc::clone(buffer)),
                    tensor.placement().clone(),
                )?
                .into()]),
            };
        }
        Ok(vec![backend.with_backend_session(|session| {
            session.to_contiguous_read(inputs[0].clone())
        })?])
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
    backend: &CpuBackend,
    counters: Arc<ExtensionCounters>,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(
        backend,
        cpu_core_capabilities(backend),
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
    backend: &CpuBackend,
    engine_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(engine_id)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        capabilities,
    );
    assemble_cpu_registration(backend, metadata, storage, state)
}

fn cpu_registration_with_storage_id(
    backend: &CpuBackend,
    engine_id: &str,
    storage_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let target = format!("test-engine:{engine_id}");
    cpu_registration_with_storage_id_for_target(
        backend,
        engine_id,
        storage_id,
        capabilities,
        state,
        &target,
    )
}

fn cpu_registration_with_storage_id_for_target(
    backend: &CpuBackend,
    engine_id: &str,
    storage_id: &str,
    capabilities: CoreCapabilityBundle,
    state: CpuRegistrationState,
    target: &str,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
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
    assemble_cpu_registration(backend, metadata, storage, state)
}

fn cpu_registration_with_storage_classes(
    backend: &CpuBackend,
    engine_id: &str,
    storage_classes: Vec<StorageClass>,
    default_storage: StorageClass,
    ingress_storage: StorageClass,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let metadata = EngineRegistrationMetadata::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        test_provider_device_identity(engine_id)?,
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(storage_classes),
        default_storage,
        CoreCapabilityBundle::builder().build(),
    );
    assemble_cpu_registration(
        backend,
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
    destination_backend: CpuBackend,
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
    let source_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)));
    let destination_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(destination_domain)));
    let provider = Arc::new(RecordingTransferProvider::materializing(
        route_storage.clone(),
        route_storage.clone(),
        destination_backend.clone(),
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
        destination_backend,
        source_endpoint,
        destination_endpoint,
        provider,
        counters,
    })
}

fn execute_published_route(fixture: &PublishedRouteFixture) -> Result<(), Box<dyn StdError>> {
    let source_identity =
        test_provider_device_identity(fixture.source_endpoint.engine_id().as_str())?;
    let destination_identity =
        test_provider_device_identity(fixture.destination_endpoint.engine_id().as_str())?;
    execute_published_route_with_targets(fixture, &source_identity, &destination_identity)
}

fn execute_published_route_with_targets(
    fixture: &PublishedRouteFixture,
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
    let input = TestAllocationDomain(fixture.source_domain).allocate(DType::F64, &[2])?;
    if let Tensor::F64(input) = &input {
        if let Buffer::Backend(buffer) = input.buffer() {
            buffer
                .map_write()
                .map_err(|source| {
                    tenferro_tensor::Error::host_access("published-route-input", source)
                })?
                .copy_from_slice(&[3.0, 5.0])
                .map_err(|source| {
                    tenferro_tensor::Error::host_access("published-route-input", source)
                })?;
        }
    }
    let output = fixture.runtime.run_compiled(&program, &[&input])?;

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
    let host_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(host_domain)));
    let cuda0_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(cuda0_domain)));
    let cuda1_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(cuda1_domain)));
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let host_endpoint = transfer_endpoint(host_engine_id, storage.clone())?;
    let cuda0_endpoint = transfer_endpoint(cuda0_engine_id, storage.clone())?;
    let cuda1_endpoint = transfer_endpoint(cuda1_engine_id, storage.clone())?;
    let host_to_cuda0 = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        cuda0_backend.clone(),
    ));
    let host_to_cuda1 = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        cuda1_backend.clone(),
    ));
    let cuda0_to_host = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        host_backend.clone(),
    ));
    let cuda1_to_host = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        host_backend.clone(),
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
        engine_id: EngineId::new(cuda0_engine_id)?,
        counters: Arc::clone(&host_to_cuda0_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-host-cuda1-module.v1")?,
        family_id: ROUTE_HOST_CUDA1_FAMILY,
        engine_id: EngineId::new(cuda1_engine_id)?,
        counters: Arc::clone(&host_to_cuda1_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-cuda0-host-module.v1")?,
        family_id: ROUTE_CUDA0_HOST_FAMILY,
        engine_id: EngineId::new(host_engine_id)?,
        counters: Arc::clone(&cuda0_to_host_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.route-cuda1-host-module.v1")?,
        family_id: ROUTE_CUDA1_HOST_FAMILY,
        engine_id: EngineId::new(host_engine_id)?,
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
        let output = runtime.run_compiled(&program, &[&input])?;

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
    let backend = CpuBackend::new();
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
    let backend = CpuBackend::new();
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
    let backend = CpuBackend::new();
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
    let fixture = published_route_fixture(removed_engine_id.clone(), route_storage)?;
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
    execute_published_route(&fixture)?;
    Ok(())
}

#[test]
fn reconfiguration_replace_engine_rejects_dropped_route_storage_atomically_and_keeps_same_route_executable(
) -> Result<(), Box<dyn StdError>> {
    let dropped_storage = StorageClass::new("tenferro-test.storage.reconfigure-dropped.v1")?;
    let replaced_engine_id = EngineId::new("tenferro-test.engine.reconfigure-replaced.v1")?;
    let fixture = published_route_fixture(replaced_engine_id.clone(), dropped_storage)?;
    let before = fixture.runtime.snapshot()?;
    let before_epoch = before.epoch();
    let before_replaced_registration = before
        .engine(&replaced_engine_id)
        .expect("replaced engine is initially published")
        .registration_identity();
    let before_route_count = before.transfer_provider_count();
    assert_eq!(before_route_count, 1);
    let replacement = cpu_registration_with_storage_id(
        &CpuBackend::new(),
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
    execute_published_route(&fixture)?;
    Ok(())
}

#[test]
fn explicit_target_rebind_updates_frozen_lookup_and_provider_request(
) -> Result<(), Box<dyn StdError>> {
    let affected_engine_id = EngineId::new("tenferro-test.engine.rebind-execution.v1")?;
    let route_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let fixture = published_route_fixture(affected_engine_id.clone(), route_storage)?;
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

    execute_published_route_with_targets(&fixture, &source_identity, &replacement_identity)?;
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
            ExecutionContextIdentity::of::<CpuBackend>(),
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
    let backend = CpuBackend::new();
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
fn runtime_rejects_operation_output_outside_scheduled_residency() -> Result<(), Box<dyn StdError>> {
    let expected_domain = AllocationDomainId::fresh();
    let foreign_domain = AllocationDomainId::fresh();
    let backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(expected_domain)));
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

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

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
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let host_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let prepared = runtime.prepare_compiled(&program, &[&host_input])?;
    let foreign_domain = AllocationDomainId::fresh();
    let foreign_input = TestAllocationDomain(foreign_domain).allocate(DType::F64, &[2])?;

    let error = runtime
        .run_prepared(&prepared, &[&foreign_input])
        .unwrap_err();

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
    let first_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)));
    let second_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(second_domain)));
    let first_engine = "tenferro-test.a-domain-engine.v1";
    let second_engine = "tenferro-test.b-domain-engine.v1";
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let counters = Arc::new(ExtensionCounters::default());
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        first_backend.clone(),
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
        engine_id: EngineId::new(first_engine)?,
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
        let output = runtime.run_compiled(&program, &[input])?;
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
    let dead_backend = CpuBackend::new();
    let routed_backend = CpuBackend::new();
    let consumer_backend = CpuBackend::new();
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
    let first_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)));
    let second_allocation_domain = Arc::new(TestAllocationDomain(second_domain));
    let second_backend = CpuBackend::new().with_allocation_domain(second_allocation_domain.clone());
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
    let first_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(first_domain)));
    let second_allocation_domain = Arc::new(TestAllocationDomain(second_domain));
    let second_backend = CpuBackend::new().with_allocation_domain(second_allocation_domain.clone());
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

    let output = runtime.run_compiled(&program, &[&input])?;

    assert_eq!(tensor_f64_values(&output[0])?, [0.0, 0.0]);
    assert_eq!(first_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(second_counters.execute.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn runtime_operation_placement_explores_complete_multi_op_combinations(
) -> Result<(), Box<dyn StdError>> {
    let a_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())));
    let b_allocation_domain = Arc::new(TestAllocationDomain(AllocationDomainId::fresh()));
    let b_backend = CpuBackend::new().with_allocation_domain(b_allocation_domain.clone());
    let c_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())));
    let d_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())));
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
        d_backend.clone(),
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
    let backend = CpuBackend::new().with_allocation_domain(domain.clone());
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
        engine_id: EngineId::new(engine_id)?,
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
    let first_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())));
    let second_backend = CpuBackend::new()
        .with_allocation_domain(Arc::new(TestAllocationDomain(AllocationDomainId::fresh())));
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

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

    let prepare_error = error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("typed prepare error");
    assert!(matches!(
        prepare_error,
        PrepareError::NoInputIngress { input_index: 0, .. }
    ));
    assert_eq!(first_counters.execute.load(Ordering::SeqCst), 0);
    assert_eq!(second_counters.execute.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn runtime_input_ingress_covers_all_split_consumers() -> Result<(), Box<dyn StdError>> {
    let dead_backend = CpuBackend::new();
    let routed_backend = CpuBackend::new();
    let first_consumer_backend = CpuBackend::new();
    let second_consumer_backend = CpuBackend::new();
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

    let output = runtime.run_compiled(&program, &[])?;

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
    let root_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(root_domain)));
    let dead_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)));
    let routed_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(source_domain)));
    let core_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)));
    let root_storage = StorageClass::new("tenferro-test.storage.synth-root.v1")?;
    let dead_storage = StorageClass::new("tenferro-test.storage.synth-dead.v1")?;
    let routed_storage = StorageClass::new("tenferro-test.storage.synth-routed.v1")?;
    let core_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let dead_to_core = Arc::new(RecordingTransferProvider::materializing(
        dead_storage.clone(),
        core_storage.clone(),
        core_backend.clone(),
    ));
    let routed_to_root = Arc::new(RecordingTransferProvider::materializing(
        routed_storage.clone(),
        root_storage.clone(),
        root_backend.clone(),
    ));
    let routed_to_core = Arc::new(RecordingTransferProvider::materializing(
        routed_storage.clone(),
        core_storage.clone(),
        core_backend.clone(),
    ));
    let root_to_core = Arc::new(RecordingTransferProvider::materializing(
        root_storage.clone(),
        core_storage.clone(),
        core_backend.clone(),
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

    let _prepared = runtime.prepare_compiled(&program, &[])?;

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
    let core_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(core_domain)));
    let extension_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)));
    let counters = Arc::new(ExtensionCounters::default());
    let core_engine_id = "tenferro-test.core-engine.v1";
    let extension_engine_id = "tenferro-test.extension-engine.v1";
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        storage.clone(),
        storage.clone(),
        extension_backend.clone(),
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
            source_engine_id: EngineId::new(core_engine_id)?,
            source_provider_device_identity: test_provider_device_identity(core_engine_id)?,
            source_event_domain_id: source_event_domain,
            source_storage_class: storage.clone(),
            destination_engine_id: EngineId::new(extension_engine_id)?,
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
fn runtime_run_compiled_returns_drain_failure_without_outputs() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let event_log = Arc::new(Mutex::new(Vec::new()));
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &backend,
        CPU_ENGINE_ID,
        cpu_core_capabilities(&backend),
        CpuRegistrationState::executable_with_driver(failing_drain_event_domain("cpu", &event_log)),
    )?)?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let error = runtime
        .run_compiled(&program, &[&input])
        .expect_err("drain failure must suppress outputs");

    assert!(error.to_string().contains("cpu event-domain drain failure"));
    assert_eq!(
        event_log.lock().expect("event log lock").as_slice(),
        ["cpu:begin", "cpu:enqueue:0", "cpu:drain", "cpu:drop"]
    );
    Ok(())
}

#[test]
fn runtime_run_compiled_unwind_drops_event_run_before_tensor_storage(
) -> Result<(), Box<dyn StdError>> {
    let domain = AllocationDomainId::fresh();
    let backend = CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(domain)));
    let event_log = Arc::new(Mutex::new(Vec::new()));
    let output_drops = Arc::new(AtomicUsize::new(0));
    let producer_counters = Arc::new(ExtensionCounters::default());
    *producer_counters
        .tracked_output_drops
        .lock()
        .expect("tracked output lock") = Some(Arc::clone(&output_drops));
    *producer_counters
        .tracked_output_drop_events
        .lock()
        .expect("tracked output drop event lock") = Some(Arc::clone(&event_log));
    let panic_counters = Arc::new(ExtensionCounters::default());
    panic_counters.panic_execute.store(true, Ordering::SeqCst);

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_id(
        &backend,
        CPU_ENGINE_ID,
        cpu_core_capabilities(&backend),
        CpuRegistrationState::executable_with_driver(recording_event_domain("cpu", &event_log)),
    )?)?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.panic-producer-module")?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new(CPU_ENGINE_ID)?,
        counters: producer_counters,
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.panic-consumer-module")?,
        family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new(CPU_ENGINE_ID)?,
        counters: Arc::clone(&panic_counters),
    }))?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let produced = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: COUNTING_EXTENSION_FAMILY,
        }),
        &[&x],
    )?
    .pop()
    .expect("producer has one output");
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        }),
        &[&produced],
    )?
    .pop()
    .expect("consumer has one output");
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = runtime.run_compiled(&program, &[&input]);
    }));

    assert!(panic.is_err(), "prepared-operation panic must unwind");
    assert_eq!(panic_counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(output_drops.load(Ordering::SeqCst), 1);
    let events = event_log.lock().expect("event log lock").clone();
    let run_drop = events
        .iter()
        .position(|event| event == "cpu:drop")
        .expect("event-domain run must drop during unwind");
    let tensor_drop = events
        .iter()
        .position(|event| event == "tensor:drop")
        .expect("intermediate tensor storage must drop during unwind");
    assert!(
        run_drop < tensor_drop,
        "event-domain cleanup must precede tensor storage release: {events:?}"
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
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        source_storage.clone(),
        destination_storage.clone(),
        extension_backend.clone(),
    ));
    let event_log = Arc::new(Mutex::new(Vec::new()));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &core_backend,
        core_engine_id,
        source_storage.as_str(),
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable_with_driver(recording_event_domain("source", &event_log)),
        &format!("test-engine:{core_engine_id}"),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable_with_driver(recording_event_domain(
            "destination",
            &event_log,
        )),
        &format!("test-engine:{extension_engine_id}"),
    )?)?;
    builder.register_transfer_provider(
        TransferEndpoint::new(EngineId::new(core_engine_id)?, source_storage.clone()),
        TransferEndpoint::new(
            EngineId::new(extension_engine_id)?,
            destination_storage.clone(),
        ),
        transfer.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new("tenferro-test.counting-extension.transfer-module")
            .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
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

    let output = runtime.run_compiled(&program, &[&input])?;

    assert_eq!(tensor_f64_values(&output[0])?, [6.0, 10.0]);
    assert_eq!(transfer.calls(), 1);
    assert_eq!(
        transfer.materialized_domains(),
        vec![(None, Some(extension_domain))]
    );
    assert_eq!(
        transfer.requests(),
        vec![RecordedTransferRequest {
            source_engine_id: EngineId::new(core_engine_id)?,
            source_provider_device_identity: test_provider_device_identity(core_engine_id)?,
            source_event_domain_id: source_event_domain,
            source_storage_class: source_storage,
            destination_engine_id: EngineId::new(extension_engine_id)?,
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
    let events = event_log.lock().expect("event log lock").clone();
    assert!(events.contains(&"source:begin".to_owned()), "{events:?}");
    assert!(
        events.contains(&"destination:begin".to_owned()),
        "{events:?}"
    );
    assert!(
        events.contains(&"source:enqueue:0".to_owned()),
        "{events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.as_str() == "destination:enqueue:1")
            .count(),
        1,
        "only the same-destination operation dependency reaches the destination run: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.as_str() == "destination:enqueue:0")
            .count(),
        1,
        "the transfer source dependency is host-bridged before destination enqueue: {events:?}"
    );
    assert!(events.contains(&"source:wait".to_owned()), "{events:?}");
    assert!(events.contains(&"source:drain".to_owned()), "{events:?}");
    assert!(
        events.contains(&"destination:drain".to_owned()),
        "{events:?}"
    );

    Ok(())
}

#[test]
fn runtime_run_compiled_transfers_input_from_validated_ingress_to_first_consumer(
) -> Result<(), Box<dyn StdError>> {
    let ingress_domain = AllocationDomainId::fresh();
    let consumer_domain = AllocationDomainId::fresh();
    let ingress_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(ingress_domain)));
    let consumer_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(consumer_domain)));
    let counters = Arc::new(ExtensionCounters::default());
    let ingress_engine_id = "tenferro-test.a-input-ingress.v1";
    let consumer_engine_id = "tenferro-test.z-input-consumer.v1";
    let ingress_storage = StorageClass::new("tenferro-test.storage.input-ingress.v1")?;
    let consumer_storage = StorageClass::new("tenferro-test.storage.input-consumer.v1")?;
    let transfer = Arc::new(RecordingTransferProvider::materializing(
        ingress_storage.clone(),
        consumer_storage.clone(),
        consumer_backend.clone(),
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
        engine_id: EngineId::new(consumer_engine_id).map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&counters),
    }))?;
    let runtime = builder.build()?;
    let snapshot = runtime.snapshot()?;
    let ingress_event_domain = snapshot
        .engine(&EngineId::new(ingress_engine_id)?)
        .expect("ingress engine")
        .event_domain_id();
    let consumer_event_domain = snapshot
        .engine(&EngineId::new(consumer_engine_id)?)
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

    let output = runtime.run_compiled(&program, &[&input])?;

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
            source_engine_id: EngineId::new(ingress_engine_id)?,
            source_provider_device_identity: test_provider_device_identity(ingress_engine_id)?,
            source_event_domain_id: ingress_event_domain,
            source_storage_class: ingress_storage,
            destination_engine_id: EngineId::new(consumer_engine_id)?,
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
        let ingress_backend = CpuBackend::new();
        let consumer_domain = AllocationDomainId::fresh();
        let consumer_backend = CpuBackend::new()
            .with_allocation_domain(Arc::new(TestAllocationDomain(consumer_domain)));
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

        let error = runtime.run_compiled(&program, &[&input]).unwrap_err();
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
    let core_backend = CpuBackend::new();
    let extension_backend = CpuBackend::new();
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
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
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

    let output = runtime.run_compiled(&program, &[&input])?;

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

struct TransferDrainFailureObservation {
    error: Error,
    events: Vec<String>,
    forward_calls: usize,
    failing_calls: usize,
    upstream_execute: usize,
    downstream_execute: usize,
    drops: usize,
}

fn run_transfer_and_drain_failure(
    drain_behavior: DrainBehavior,
) -> Result<TransferDrainFailureObservation, Box<dyn StdError>> {
    let core_backend = CpuBackend::new();
    let extension_domain = AllocationDomainId::fresh();
    let extension_backend =
        CpuBackend::new().with_allocation_domain(Arc::new(TestAllocationDomain(extension_domain)));
    let drops = Arc::new(AtomicUsize::new(0));
    let upstream_counters = Arc::new(ExtensionCounters::default());
    let downstream_counters = Arc::new(ExtensionCounters::default());
    *upstream_counters
        .tracked_output_drops
        .lock()
        .expect("tracked output lock") = Some(Arc::clone(&drops));
    let core_engine_id = "tenferro-test.core-transfer-failure.v1";
    let extension_engine_id = "tenferro-test.extension-transfer-failure.v1";
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.failure-destination.v1")?;
    let forward = Arc::new(RecordingTransferProvider::materializing(
        source_storage.clone(),
        destination_storage.clone(),
        extension_backend.clone(),
    ));
    let failing = Arc::new(FailingTransferProvider {
        calls: AtomicUsize::new(0),
    });
    let event_log = Arc::new(Mutex::new(Vec::new()));

    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &core_backend,
        core_engine_id,
        source_storage.as_str(),
        cpu_core_capabilities(&core_backend),
        CpuRegistrationState::executable_with_driver(event_domain_with_drain_behavior(
            "source",
            &event_log,
            drain_behavior,
        )),
        &format!("test-engine:{core_engine_id}"),
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id_for_target(
        &extension_backend,
        extension_engine_id,
        destination_storage.as_str(),
        CoreCapabilityBundle::default(),
        CpuRegistrationState::executable_with_driver(recording_event_domain(
            "destination",
            &event_log,
        )),
        &format!("test-engine:{extension_engine_id}"),
    )?)?;
    builder.register_transfer_provider(
        transfer_endpoint(core_engine_id, source_storage.clone())?,
        transfer_endpoint(extension_engine_id, destination_storage.clone())?,
        forward.clone(),
    )?;
    builder.register_transfer_provider(
        transfer_endpoint(extension_engine_id, destination_storage)?,
        transfer_endpoint(core_engine_id, source_storage)?,
        failing.clone(),
    )?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new(
            "tenferro-test.counting-extension.transfer-failure-module",
        )
        .map_err(RuntimeConfigError::from)?,
        family_id: COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&upstream_counters),
    }))?;
    builder.install_extension_module(Arc::new(CountingExtensionModule {
        module_id: ExtensionModuleId::new(
            "tenferro-test.downstream-counting-extension.transfer-failure-module",
        )
        .map_err(RuntimeConfigError::from)?,
        family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        engine_id: EngineId::new(core_engine_id).map_err(RuntimeConfigError::from)?,
        counters: Arc::clone(&downstream_counters),
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
    let y = tenferro_runtime::extension::apply(
        Arc::new(CountingExtensionOp {
            family_id: DOWNSTREAM_COUNTING_EXTENSION_FAMILY,
        }),
        &[&extension],
    )?
    .pop()
    .expect("downstream extension has one output");
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();
    eprintln!("missing bridge error: {error:?}");

    let events = event_log.lock().expect("event log lock").clone();
    Ok(TransferDrainFailureObservation {
        error,
        events,
        forward_calls: forward.calls(),
        failing_calls: failing.calls.load(Ordering::SeqCst),
        upstream_execute: upstream_counters.execute.load(Ordering::SeqCst),
        downstream_execute: downstream_counters.execute.load(Ordering::SeqCst),
        drops: drops.load(Ordering::SeqCst),
    })
}

fn assert_transfer_and_drain_failure(
    observation: &TransferDrainFailureObservation,
    cleanup_message: &str,
) {
    assert!(observation
        .error
        .to_string()
        .contains("intentional transfer failure"));
    assert!(
        observation.error.to_string().contains(cleanup_message),
        "combined execution and cleanup diagnostics must preserve both failures: {}",
        observation.error
    );
    let cleanup = observation
        .error
        .source()
        .expect("combined error must retain cleanup wrapper");
    let primary = cleanup
        .source()
        .expect("cleanup wrapper must retain the primary execution error");
    assert!(primary.to_string().contains("intentional transfer failure"));
    assert_eq!(observation.forward_calls, 1);
    assert_eq!(observation.failing_calls, 1);
    assert_eq!(observation.upstream_execute, 1);
    assert_eq!(
        observation.downstream_execute, 0,
        "the downstream operation must not execute after its input transfer fails"
    );
    assert_eq!(
        observation.drops, 1,
        "the extension output retained at its source location must be released on failure"
    );
    assert_eq!(
        observation
            .events
            .iter()
            .filter(|event| event.as_str() == "source:enqueue:0")
            .count(),
        2,
        "the reverse transfer reaches the source run without its foreign token: {:?}",
        observation.events
    );
    assert!(
        !observation
            .events
            .iter()
            .any(|event| event == "source:enqueue:1"),
        "the source run must not receive the foreign destination token: {:?}",
        observation.events
    );
    assert_eq!(
        observation
            .events
            .iter()
            .filter(|event| event.as_str() == "destination:wait")
            .count(),
        2,
        "the scheduler host bridge and downstream operation each wait on destination work: {:?}",
        observation.events
    );
    assert!(
        observation.events.contains(&"source:drain".to_owned()),
        "{:?}",
        observation.events
    );
    assert!(
        observation.events.contains(&"destination:drain".to_owned()),
        "{:?}",
        observation.events
    );
}

#[test]
fn transfer_and_drain_failure_preserve_both_errors_and_release_located_values(
) -> Result<(), Box<dyn StdError>> {
    let observation = run_transfer_and_drain_failure(DrainBehavior::ReturnError)?;
    assert_transfer_and_drain_failure(&observation, "source event-domain drain failure");
    Ok(())
}

#[test]
fn transfer_and_drain_panic_preserve_primary_and_cleanup_errors() -> Result<(), Box<dyn StdError>> {
    let observation = run_transfer_and_drain_failure(DrainBehavior::Panic)?;
    assert_transfer_and_drain_failure(&observation, "source event-domain drain panic");
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
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
    let destination_storage = StorageClass::new("tenferro-test.storage.missing-destination.v1")?;
    let source_endpoint =
        TransferEndpoint::new(EngineId::new(core_engine_id)?, source_storage.clone());
    let expected_destination_endpoint = TransferEndpoint::new(
        EngineId::new(extension_engine_id)?,
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
        engine_id: EngineId::new(extension_engine_id).map_err(RuntimeConfigError::from)?,
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

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

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

    let submit_error = runtime
        .submit(&program, ExecutionInputs::new(vec![input.duplicate()?]))
        .unwrap_err();
    let submit_prepare_error = submit_error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("submit preserves the typed preparation error synchronously");
    assert!(matches!(
        submit_prepare_error,
        PrepareError::MissingTransferProvider {
            destination_endpoint: actual_destination_endpoint,
            available_source_endpoints,
            ..
        } if available_source_endpoints == std::slice::from_ref(&source_endpoint)
            && actual_destination_endpoint == &expected_destination_endpoint
    ));

    Ok(())
}

#[test]
fn runtime_submit_reports_no_input_ingress_before_spawning() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let foreign_domain = AllocationDomainId::fresh();
    let input = TestAllocationDomain(foreign_domain).allocate(DType::F64, &[2])?;
    let input_placement = input.placement().clone();

    let error = runtime
        .submit(&program, ExecutionInputs::new(vec![input]))
        .unwrap_err();

    let prepare_error = error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("submit preserves the typed preparation error synchronously");
    assert!(matches!(
        prepare_error,
        PrepareError::NoInputIngress {
            input_index: 0,
            placement,
        } if placement == &input_placement
    ));
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

    let handle = runtime.submit(&program, ExecutionInputs::new(vec![input.duplicate()?]))?;
    let output = match handle.wait()? {
        tenferro_runtime::ExecutionOutcome::Completed(output) => output,
        other => panic!("unexpected submission outcome: {other:?}"),
    };

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
fn runtime_run_compiled_rejects_preparation_only_registration() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
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

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

    assert_eq!(error.phase(), Some(ErrorPhase::Execution));
    assert!(StdError::source(&error).is_some());
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
        let backend = CpuBackend::new();
        edit.replace_engine(cpu_registration(
            &backend,
            cpu_core_capabilities(&backend),
            CpuRegistrationState::executable_with_cache_owner(
                Arc::new(ImmediateEventDomainDriver::new()),
                Arc::new(backend.clone()),
            ),
        )?)?;
        Ok(())
    })?;

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_second.misses > after_first.misses);
    Ok(())
}
