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
    CoreCapabilityBundle, DType, DotGeneralPreparation, ElementwiseRuntime,
    EngineExecutionContractError, EngineId, EngineRegistration, ErasedExecutionContext, Error,
    ErrorPhase, EventDomainId, ExecutionContextIdentity, ExtensionCacheStore, ExtensionEngine,
    ExtensionModule, ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar,
    ExtensionPlanningConfig, ExtensionPrepareRequest, GraphCompiler, HardwareClassId,
    IndexingRuntime, InputIngressContractError, LayoutRuntime, PrepareCapability, PrepareError,
    PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor, PreparedOperationPlan,
    ReductionRuntime, RegistrationKey, Runtime, RuntimeCacheOwner, RuntimeConfigError,
    SpecializationProjection, StorageClass, TracedTensor, TransferError, TransferProvider,
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
        storage.clone(),
        capabilities.build(),
    )
    .map(|registration| {
        let registration = registration
            .with_cache_owner(cache_owner)
            .with_input_signature_validator({
                let storage = storage.clone();
                let allocation_domain = backend.allocation_domain();
                move |placement, family, domain, candidate| {
                    candidate == &storage
                        && test_cpu_input_signature(placement, family, domain, allocation_domain)
                }
            })
            .with_input_ingress_validator(
                {
                    let storage = storage.clone();
                    move |placement, candidate| {
                        test_cpu_placement(placement) && candidate == &storage
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
            );
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
    last_execute_domain: Mutex<Option<AllocationDomainId>>,
    tracked_output_drops: Mutex<Option<Arc<AtomicUsize>>>,
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
    source_event_domain_id: EventDomainId,
    source_storage_class: StorageClass,
    destination_engine_id: EngineId,
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
                source_event_domain_id: request.source_event_domain_id(),
                source_storage_class: request.source_storage_class().clone(),
                destination_engine_id: request.destination_engine_id().clone(),
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
            None => {
                request.input().as_tensor().cloned().ok_or_else(|| {
                    Error::Internal("test transfer expected an owned tensor".into())
                })?
            }
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
                let mut tensor = request.input().as_tensor().cloned().ok_or_else(|| {
                    Error::Internal("test transfer expected an owned tensor".into())
                })?;
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
                Ok(tensor.clone().into())
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
}

impl Drop for DropTrackedBuffer {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
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
            let buffer = Buffer::Backend(Arc::new(DropTrackedBuffer { len, drops, domain }));
            return Ok(vec![TypedTensor::<f64>::from_buffer_col_major(
                shape,
                buffer,
                Placement::default(),
            )?
            .into()]);
        }
        if inputs[0].allocation_domain() == backend.allocation_domain() {
            return inputs[0]
                .as_tensor()
                .cloned()
                .map(|tensor| vec![tensor])
                .ok_or_else(|| {
                    Error::Internal(
                        "allocation-domain test executor expected an owned tensor".into(),
                    )
                });
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
    builder.register_engine(cpu_registration(backend, true)?)?;
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
        storage.clone(),
        capabilities.build(),
    )
    .map(|registration| {
        let registration = registration
            .with_input_signature_validator({
                let storage = storage.clone();
                let allocation_domain = backend.allocation_domain();
                move |placement, family, domain, candidate| {
                    candidate == &storage
                        && test_cpu_input_signature(placement, family, domain, allocation_domain)
                }
            })
            .with_input_ingress_validator(
                {
                    let storage = storage.clone();
                    move |placement, candidate| {
                        test_cpu_placement(placement) && candidate == &storage
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
            );
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
        storage.clone(),
        capabilities.build(),
    )
    .map(|registration| {
        let registration = registration
            .with_input_signature_validator({
                let storage = storage.clone();
                let allocation_domain = backend.allocation_domain();
                move |placement, family, domain, candidate| {
                    candidate == &storage
                        && test_cpu_input_signature(placement, family, domain, allocation_domain)
                }
            })
            .with_input_ingress_validator(
                {
                    let storage = storage.clone();
                    move |placement, candidate| {
                        test_cpu_placement(placement) && candidate == &storage
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
                {
                    let storage = storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &storage && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
            );
        if include_execution_bridge {
            registration.with_tensor_backend_executor(backend.as_ref().clone())
        } else {
            registration
        }
    })
}

fn cpu_registration_with_storage_classes(
    backend: &CpuBackend,
    engine_id: &str,
    storage_classes: Vec<StorageClass>,
    default_storage: StorageClass,
    ingress_storage: StorageClass,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    EngineRegistration::new(
        EngineId::new(engine_id).map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(storage_classes),
        default_storage,
        CoreCapabilityBundle::builder().build(),
    )
    .map(|registration| {
        registration
            .with_input_signature_validator({
                let ingress_storage = ingress_storage.clone();
                let allocation_domain = backend.allocation_domain();
                move |placement, family, domain, candidate| {
                    candidate == &ingress_storage
                        && test_cpu_input_signature(placement, family, domain, allocation_domain)
                }
            })
            .with_input_ingress_validator(
                {
                    let ingress_storage = ingress_storage.clone();
                    move |placement, candidate| {
                        test_cpu_placement(placement) && candidate == &ingress_storage
                    }
                },
                {
                    let ingress_storage = ingress_storage.clone();
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &ingress_storage
                            && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
                {
                    let allocation_domain = backend.allocation_domain();
                    move |input: &TensorRead<'_>, candidate| {
                        candidate == &ingress_storage
                            && test_cpu_runtime_input(input, allocation_domain)
                    }
                },
            )
            .with_tensor_backend_executor(backend.as_ref().clone())
    })
}

#[test]
fn transfer_provider_registration_is_idempotent_and_rejects_conflicts(
) -> Result<(), Box<dyn StdError>> {
    let source = StorageClass::new("tenferro-test.storage.registry-source")?;
    let destination = StorageClass::new("tenferro-test.storage.registry-destination")?;
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
        source.clone(),
        destination.clone(),
        Arc::clone(&provider),
    )?;
    builder.register_transfer_provider(
        source.clone(),
        destination.clone(),
        Arc::clone(&provider),
    )?;
    let error = builder
        .register_transfer_provider(source.clone(), destination.clone(), conflicting)
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeConfigError::ConflictingRegistration {
            key: RegistrationKey::TransferProvider {
                source: actual_source,
                destination: actual_destination,
            },
        } if actual_source == source && actual_destination == destination
    ));
    Ok(())
}

#[test]
fn execution_bridge_registration_requires_explicit_ingress_contract(
) -> Result<(), Box<dyn StdError>> {
    let storage = StorageClass::new("tenferro-test.storage.missing-ingress.v1")?;
    let registration = EngineRegistration::new(
        EngineId::new("tenferro-test.engine.missing-ingress.v1")?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new("tenferro-test.hardware.missing-ingress.v1")?,
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    )?
    .with_tensor_backend_executor(CpuBackend::new());
    let mut builder = Runtime::builder();

    let error = builder.register_engine(registration).unwrap_err();

    assert!(matches!(
        error,
        RuntimeConfigError::MissingInputIngressValidator { engine_id }
            if engine_id.as_str() == "tenferro-test.engine.missing-ingress.v1"
    ));
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &second_backend,
        second_engine,
        false,
        true,
    )?)?;
    builder.register_transfer_provider(
        storage,
        StorageClass::new(CPU_STORAGE_CLASS_ID)?,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.b-routed-ingress.v1",
        routed_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &consumer_backend,
        "tenferro-test.z-routed-consumer.v1",
        consumer_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_transfer_provider(
        routed_storage.clone(),
        consumer_storage,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_backend,
        "tenferro-test.b-second-reachable.v1",
        shared_storage.as_str(),
        false,
        true,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_backend,
        "tenferro-test.b-second-storage-reachable.v1",
        CPU_STORAGE_CLASS_ID,
        false,
        true,
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
            false,
            true,
        )?)?;
    }
    builder.register_transfer_provider(b_storage.clone(), d_storage.clone(), transfer.clone())?;
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &second_backend,
        "tenferro-test.b-unreachable-capable.v1",
        false,
        true,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.b-split-routed.v1",
        routed_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &first_consumer_backend,
        "tenferro-test.y-split-first.v1",
        first_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &second_consumer_backend,
        "tenferro-test.z-split-second.v1",
        second_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_transfer_provider(
        dead_storage,
        first_storage.clone(),
        dead_to_first.clone(),
    )?;
    builder.register_transfer_provider(
        routed_storage.clone(),
        first_storage,
        routed_to_first.clone(),
    )?;
    builder.register_transfer_provider(
        routed_storage.clone(),
        second_storage,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &dead_backend,
        "tenferro-test.b-synth-dead.v1",
        dead_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &routed_backend,
        "tenferro-test.c-synth-routed.v1",
        routed_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &core_backend,
        "tenferro-test.z-synth-core.v1",
        core_storage.as_str(),
        true,
        true,
    )?)?;
    builder.register_transfer_provider(dead_storage, core_storage.clone(), dead_to_core.clone())?;
    builder.register_transfer_provider(
        routed_storage.clone(),
        root_storage.clone(),
        routed_to_root.clone(),
    )?;
    builder.register_transfer_provider(
        routed_storage.clone(),
        core_storage.clone(),
        routed_to_core.clone(),
    )?;
    builder.register_transfer_provider(root_storage, core_storage, root_to_core)?;
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
        true,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_id(
        &extension_backend,
        extension_engine_id,
        false,
        true,
    )?)?;
    builder.register_transfer_provider(storage.clone(), storage.clone(), transfer.clone())?;
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
            source_event_domain_id: source_event_domain,
            source_storage_class: storage.clone(),
            destination_engine_id: EngineId::new(extension_engine_id)?,
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
            source_event_domain_id: source_event_domain,
            source_storage_class: source_storage,
            destination_engine_id: EngineId::new(extension_engine_id)?,
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
        false,
        true,
    )?)?;
    builder.register_engine(cpu_registration_with_storage_id(
        &consumer_backend,
        consumer_engine_id,
        consumer_storage.as_str(),
        false,
        true,
    )?)?;
    builder.register_transfer_provider(
        ingress_storage.clone(),
        consumer_storage.clone(),
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
            source_event_domain_id: ingress_event_domain,
            source_storage_class: ingress_storage,
            destination_engine_id: EngineId::new(consumer_engine_id)?,
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
            false,
            true,
        )?)?;
        builder.register_engine(cpu_registration_with_storage_id(
            &consumer_backend,
            "tenferro-test.z-faulty-consumer.v1",
            consumer_storage.as_str(),
            false,
            true,
        )?)?;
        builder.register_transfer_provider(
            ingress_storage,
            consumer_storage,
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
        forward.clone(),
    )?;
    builder.register_transfer_provider(
        destination_storage.clone(),
        source_storage,
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

#[test]
fn transfer_failure_skips_downstream_execution_and_releases_located_values(
) -> Result<(), Box<dyn StdError>> {
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
        forward.clone(),
    )?;
    builder.register_transfer_provider(destination_storage, source_storage, failing.clone())?;
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

    assert!(error.to_string().contains("intentional transfer failure"));
    assert_eq!(forward.calls(), 1);
    assert_eq!(failing.calls.load(Ordering::SeqCst), 1);
    assert_eq!(upstream_counters.execute.load(Ordering::SeqCst), 1);
    assert_eq!(
        downstream_counters.execute.load(Ordering::SeqCst),
        0,
        "the downstream operation must not execute after its input transfer fails"
    );
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "the extension output retained at its source location must be released on failure"
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
    let source_storage = StorageClass::new(CPU_STORAGE_CLASS_ID)?;
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
    let prepare_error = error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("typed prepare error source");
    assert!(matches!(
        prepare_error,
        PrepareError::MissingTransferProvider {
            destination_storage_class,
            available_storage_classes,
            ..
        } if available_storage_classes == std::slice::from_ref(&source_storage)
            && destination_storage_class == &destination_storage
    ));
    assert_eq!(counters.execute.load(Ordering::SeqCst), 0);

    let submit_error = runtime.submit(&program, &[&input]).unwrap_err();
    let submit_prepare_error = submit_error
        .source()
        .and_then(StdError::source)
        .and_then(|source| source.downcast_ref::<PrepareError>())
        .expect("submit preserves the typed preparation error synchronously");
    assert!(matches!(
        submit_prepare_error,
        PrepareError::MissingTransferProvider {
            destination_storage_class,
            available_storage_classes,
            ..
        } if available_storage_classes == std::slice::from_ref(&source_storage)
            && destination_storage_class == &destination_storage
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

    let error = runtime.submit(&program, &[&input]).unwrap_err();

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
        } if placement == input.placement()
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
