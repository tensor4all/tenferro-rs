use std::any::Any;
use std::error::Error as StdError;
use std::io;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::thread;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendRuntimeCache, Engine, BackendSessionHost,
    BackendStorage, DType, HostAccessError, HostReadGuard, HostWriteGuard, MemoryKind, Placement,
    SessionEntryError, StorageBuffer, Tensor, TensorBackend, TensorDeviceTransfer, TensorRead,
    TensorValue, TensorView, TypedTensor,
};

use crate::exec::{ExecInstruction, ExecOp, ExecProgram, ExecSlot};
use crate::runtime::execution::{
    collect_scoped_outputs, collect_tensor_outputs_with, retain_instruction_results,
    spawn_in_flight, submit_with_spawner, BorrowedExecutorSessionState,
    ErasedTensorBackendExecutor, InFlightSubmission, LocatedExecSlot, OsThreadSpawner,
    RuntimeOutputMode, SubmissionSpawner,
};
use crate::runtime::schedule::ExecutionLocation;
use crate::runtime::{
    CacheOwnerError, CacheStats, CoreCapabilityBundle, CorePrepareContext, DotGeneralPreparation,
    DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime, EngineId,
    EngineRegistration, EventDomainDriver, EventDomainId, EventDomainRun, EventToken,
    ExecutableEngineContract, ExecutionContextIdentity, ExecutionInputs, HardwareClassId,
    ImmediateEventDomainDriver, IndexingPrepareRequest, IndexingRuntime, InputIngressContract,
    InputPlacementContract, InputSignatureContract, LayoutPrepareRequest, LayoutRuntime,
    PrepareCapability, PrepareError, PreparedOperation, PreparedOperationBinding,
    PreparedOperationPlan, ProviderDeviceIdentity, ProviderExecutableBinding, ProviderId,
    ReductionPrepareRequest, ReductionRuntime, RegistrationIdentity, ResidentOutputContract,
    Runtime, RuntimeConfigError, RuntimeEpoch, RuntimeId, RuntimeInputContract,
    ScopedExecutionOutcome, ScopedReadInputs, SpecializationProjection, StorageClass,
    SubmissionError,
};
use crate::{Error, ErrorPhase, GraphCompiler, TracedTensor};

fn test_provider_device_identity(target: &str) -> ProviderDeviceIdentity {
    ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.execution").expect("provider id"),
        target,
    )
    .expect("provider target")
}

fn qualified_domain(ordinal: u64) -> EventDomainId {
    EventDomainId::runtime_created_for_test(
        RuntimeId::from_nonzero(NonZeroU64::new(1).expect("runtime id")),
        RuntimeEpoch::from_nonzero(NonZeroU64::new(1).expect("runtime epoch")),
        RegistrationIdentity::new(
            NonZeroU64::new(1).expect("registration issuer"),
            NonZeroU64::new(ordinal).expect("registration ordinal"),
        ),
    )
}

static NEXT_TEST_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

fn next_test_allocation_id() -> AllocationId {
    AllocationId::from_backend_id(NEXT_TEST_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed))
}

#[derive(Debug)]
struct ForeignProbeBuffer {
    values: Arc<Mutex<Vec<f64>>>,
    domain: AllocationDomainId,
    allocation: AllocationId,
}

impl BackendStorage<f64> for ForeignProbeBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.foreign-output-probe"
    }

    fn len(&self) -> usize {
        self.values.lock().expect("probe buffer lock").len()
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, f64>, HostAccessError> {
        Ok(HostReadGuard::new(
            self.values.lock().expect("probe buffer lock"),
        ))
    }

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, f64>, HostAccessError> {
        let values = Arc::clone(&self.values);
        Ok(HostWriteGuard::new(self.len(), move |source| {
            values
                .lock()
                .expect("probe buffer lock")
                .clone_from_slice(source);
            Ok(())
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct DelayedSpawner {
    started: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl SubmissionSpawner for DelayedSpawner {
    fn spawn(&self, submission: Arc<InFlightSubmission>) -> io::Result<()> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        thread::spawn(move || {
            started.wait();
            release.wait();
            submission.run();
        });
        Ok(())
    }
}

#[derive(Debug)]
struct FailingSpawner;

impl SubmissionSpawner for FailingSpawner {
    fn spawn(&self, _submission: Arc<InFlightSubmission>) -> io::Result<()> {
        Err(io::Error::other("injected worker spawn failure"))
    }
}

#[derive(Debug)]
struct AdmissionTestExecutor {
    fail_materialize: bool,
    /// Ordered log shared with an [`OrderingProbeDriver`] so a test can assert
    /// that output materialization ran before the event domain was retired.
    events: Option<Arc<Mutex<Vec<&'static str>>>>,
}

impl ErasedTensorBackendExecutor for AdmissionTestExecutor {
    fn backend_type_name(&self) -> &'static str {
        "AdmissionTestExecutor"
    }

    fn try_lease_session_state<'a>(
        &'a self,
        caller: &'static str,
    ) -> Result<Box<dyn BorrowedExecutorSessionState + 'a>, Error> {
        Err(Error::unsupported(
            caller,
            ErrorPhase::Execution,
            "AdmissionTestExecutor does not lease a session state",
        ))
    }

    fn extension_cache_stats(&self) -> Result<CacheStats, CacheOwnerError> {
        Ok(CacheStats::default())
    }

    fn clear_extension_caches(&self) -> Result<(), CacheOwnerError> {
        Ok(())
    }

    fn execute(
        &self,
        _program: &ExecProgram,
        _operations: &[PreparedOperationPlan],
        inputs: Vec<Tensor>,
    ) -> crate::Result<Vec<Tensor>> {
        Ok(inputs)
    }

    fn execute_tensor_refs(
        &self,
        _program: &ExecProgram,
        _operations: &[PreparedOperationPlan],
        inputs: &[&Tensor],
    ) -> crate::Result<Vec<Tensor>> {
        inputs
            .iter()
            .map(|input| input.duplicate())
            .collect::<tenferro_tensor::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    fn execute_values(
        &self,
        _program: &ExecProgram,
        _operations: &[PreparedOperationPlan],
        inputs: Vec<Tensor>,
    ) -> crate::Result<Vec<TensorValue>> {
        Ok(inputs.into_iter().map(TensorValue::from_tensor).collect())
    }

    fn execute_value_refs(
        &self,
        _program: &ExecProgram,
        _operations: &[PreparedOperationPlan],
        inputs: &[&Tensor],
    ) -> crate::Result<Vec<TensorValue>> {
        inputs
            .iter()
            .map(|input| input.duplicate().map(TensorValue::from_tensor))
            .collect::<tenferro_tensor::Result<Vec<_>>>()
            .map_err(Error::from)
    }

    fn execute_elementwise_fusion_slots<'input>(
        &self,
        _input_slots: &[usize],
        _instruction_count: usize,
        _plan: &tenferro_tensor::backend::ElementwiseFusionPlan,
        _slots: &mut [Option<ExecSlot<'input>>],
        _output_slots: &[usize],
    ) -> crate::Result<bool> {
        // This executor is an admission/scheduling double; it declines fusion so
        // regions keep the per-instruction path.
        Ok(false)
    }

    fn execute_slot_instruction<'input>(
        &self,
        _instruction_index: usize,
        _instruction: &ExecInstruction,
        _operations: &[PreparedOperationPlan],
        _slots: &mut [Option<ExecSlot<'input>>],
        _output_mode: RuntimeOutputMode,
        _terminal_slots: &[bool],
    ) -> crate::Result<()> {
        Err(Error::runtime_state(
            "AdmissionTestExecutor",
            ErrorPhase::Execution,
            "operationless admission test unexpectedly executed an instruction",
        ))
    }

    fn execute_prepared_single_session(
        &self,
        _program: &ExecProgram,
        _schedule: &crate::runtime::schedule::ScheduledGraph,
        _regions: &[crate::runtime::region::ElementwiseRegion],
        _region_counters: &crate::runtime::region::RegionExecutionCounters,
        _operations: &[PreparedOperationPlan],
        _inputs: &[TensorRead<'_>],
    ) -> crate::Result<Option<Vec<Tensor>>> {
        // This double cannot lease a session state, so it declines the
        // single-session route and the graph keeps the per-instruction leased path.
        // The ordering test then observes materialization inside the same run's
        // event-domain lifetime, exactly as before.
        Ok(None)
    }

    fn materialize_slot<'input>(&self, slot: ExecSlot<'input>) -> crate::Result<Tensor> {
        if let Some(events) = &self.events {
            events.lock().expect("event log lock").push("materialize");
        }
        if self.fail_materialize {
            return Err(Error::runtime_state(
                "AdmissionTestExecutor",
                ErrorPhase::Execution,
                "replacement executor must not serve admitted work",
            ));
        }
        match slot {
            ExecSlot::Owned(tensor) => tensor.duplicate().map_err(Error::from),
            ExecSlot::Read(TensorRead::Tensor(tensor)) => tensor.duplicate().map_err(Error::from),
            ExecSlot::Read(TensorRead::View(TensorView::F64(view))) => {
                let data = view.as_slice().map_err(Error::from)?.to_vec();
                Tensor::from_vec_col_major(view.shape().to_vec(), data).map_err(Error::from)
            }
            _ => Err(Error::Internal(
                "AdmissionTestExecutor received unsupported materialization value".to_owned(),
            )),
        }
    }

    fn materialize_slot_value<'input>(&self, slot: ExecSlot<'input>) -> crate::Result<TensorValue> {
        self.materialize_slot(slot).map(TensorValue::from_tensor)
    }
}

fn admission_test_registration(
    fail_materialize: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    admission_test_registration_with(
        Arc::new(AdmissionTestExecutor {
            fail_materialize,
            events: None,
        }),
        Arc::new(ImmediateEventDomainDriver::new()),
    )
}

fn admission_test_registration_with(
    executor: Arc<AdmissionTestExecutor>,
    driver: Arc<dyn EventDomainDriver>,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let storage = StorageClass::new("tenferro-test.admission-storage.v1")?;
    let engine_id = EngineId::new("tenferro-test.admission-engine.v1")?;
    let registration = EngineRegistration::executable(ProviderExecutableBinding::new(
        engine_id,
        HardwareClassId::new("tenferro-test.admission-hardware.v1")?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        ExecutableEngineContract::from_erased_for_test(
            ProviderDeviceIdentity::new(
                ProviderId::new("tenferro.test.admission").expect("provider id"),
                "admission-engine",
            )
            .expect("provider target"),
            ExecutionContextIdentity::of::<AdmissionTestExecutor>(),
            CoreCapabilityBundle::builder().build(),
            executor,
            driver,
            InputIngressContract::new(
                InputPlacementContract::new({
                    let storage = storage.clone();
                    move |_, candidate| candidate == &storage
                }),
                InputSignatureContract::new({
                    let storage = storage.clone();
                    move |_, family, domain, candidate| {
                        candidate == &storage && family.is_none() && domain.is_none()
                    }
                }),
                RuntimeInputContract::new({
                    let storage = storage.clone();
                    move |input, candidate| {
                        candidate == &storage && input.backend_family().is_none()
                    }
                }),
                ResidentOutputContract::new({
                    let storage = storage.clone();
                    move |input, candidate| {
                        candidate == &storage && input.backend_family().is_none()
                    }
                }),
            ),
            None,
        ),
    )?);
    Ok(registration)
}

#[derive(Debug)]
struct OrderingProbeRun {
    domain: EventDomainId,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl EventDomainRun for OrderingProbeRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        assert!(
            dependencies.is_empty(),
            "the ordering probe has no inter-node dependencies"
        );
        launch()?;
        Ok(Arc::new(OrderingProbeToken {
            domain: self.domain,
        }))
    }

    fn drain(&mut self) -> crate::Result<()> {
        self.events.lock().expect("event log lock").push("drain");
        Ok(())
    }
}

#[derive(Debug)]
struct OrderingProbeToken {
    domain: EventDomainId,
}

impl EventToken for OrderingProbeToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.domain
    }

    fn wait(&self) -> crate::Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct OrderingProbeDriver {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl EventDomainDriver for OrderingProbeDriver {
    fn begin_run(&self, domain: EventDomainId) -> crate::Result<Box<dyn EventDomainRun>> {
        self.events.lock().expect("event log lock").push("begin");
        Ok(Box::new(OrderingProbeRun {
            domain,
            events: Arc::clone(&self.events),
        }))
    }
}

#[test]
fn prepared_materialization_runs_inside_the_event_domain_retirement(
) -> Result<(), Box<dyn StdError>> {
    // The identity graph has no scheduled node, so the only evidence that the
    // run owns a retirement barrier is the event-domain run seeded from the
    // scheduled locations. The shared log proves output materialization runs
    // before that run is retired; an identity or zero-node graph without the
    // location seed would never retire the copy at all.
    let events = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(AdmissionTestExecutor {
        fail_materialize: false,
        events: Some(Arc::clone(&events)),
    });
    let driver: Arc<dyn EventDomainDriver> = Arc::new(OrderingProbeDriver {
        events: Arc::clone(&events),
    });
    let mut builder = Runtime::builder();
    builder.register_engine(admission_test_registration_with(executor, driver)?)?;
    let runtime = builder.build()?;

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let prepared = runtime.prepare_compiled(&program, &[&input])?;

    let output = runtime.run_prepared(&prepared, &[&input])?;
    assert_eq!(output[0].as_slice::<f64>()?, &[1.0, 2.0]);
    assert_eq!(
        *events.lock().expect("event log lock"),
        vec!["begin", "materialize", "drain"],
        "output materialization must run before the run's event domain is retired"
    );
    Ok(())
}

#[test]
fn scoped_immediate_provider_returns_borrowed_output() -> Result<(), Box<dyn StdError>> {
    let mut builder = Runtime::builder();
    builder.register_engine(admission_test_registration(false)?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let view = TensorView::F64(
        input
            .as_typed::<f64>()
            .expect("unexpected input dtype")
            .as_view(),
    );

    let scoped = ScopedReadInputs::new(vec![view]);
    let outcome = match runtime.execute_scoped_read_only(&program, scoped) {
        Ok(outcome) => outcome,
        Err(rejected) => return Err(rejected.to_string().into()),
    };
    let ScopedExecutionOutcome::Completed(bundle) = outcome else {
        return Err("immediate provider did not complete scoped work".into());
    };
    match bundle.output(0)? {
        crate::runtime::OutputRef::Tensor(TensorView::F64(view)) => {
            assert_eq!(view.as_slice()?, &[1.0, 2.0]);
        }
        crate::runtime::OutputRef::Metadata(_) => return Err("unexpected metadata output".into()),
        _ => return Err("unexpected output dtype".into()),
    }
    assert!(matches!(
        bundle.into_owned_output(0),
        Err((_, crate::runtime::ScopedOutputExtractError::BorrowedOutput))
    ));
    Ok(())
}

#[test]
fn scoped_owned_output_is_bundled_and_extractable() -> Result<(), Box<dyn StdError>> {
    let location = ExecutionLocation::new(
        EngineId::new("tenferro-test.scoped-owned-engine")?,
        test_provider_device_identity("scoped-owned"),
        qualified_domain(3),
        StorageClass::new("tenferro-test.scoped-owned-storage")?,
    );
    let program = crate::exec::ExecProgram {
        instructions: Vec::new(),
        input_slots: Vec::new(),
        output_slots: vec![0],
        n_slots: 1,
        shape_guards: Vec::new(),
    };
    let output = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    let mut located = vec![Some(LocatedExecSlot {
        location,
        value: ExecSlot::Owned(output),
    })];
    let bundle = collect_scoped_outputs(&program, &mut located)?;
    match bundle.output(0)? {
        crate::runtime::OutputRef::Tensor(TensorView::F64(view)) => {
            assert_eq!(view.as_slice()?, &[3.0, 4.0]);
        }
        crate::runtime::OutputRef::Metadata(_) => return Err("unexpected metadata output".into()),
        _ => return Err("unexpected output dtype".into()),
    }
    let extracted = bundle.into_owned_output(0).map_err(|(_, error)| error)?;
    assert_eq!(extracted.as_slice::<f64>()?, &[3.0, 4.0]);
    Ok(())
}

#[test]
fn scoped_non_host_input_is_rejected_before_admission() -> Result<(), Box<dyn StdError>> {
    let mut builder = Runtime::builder();
    builder.register_engine(admission_test_registration(false)?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_typed::<f64>(TypedTensor::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(ForeignProbeBuffer {
            values: Arc::new(Mutex::new(vec![1.0, 2.0])),
            domain: AllocationDomainId::fresh(),
            allocation: next_test_allocation_id(),
        })),
        Placement {
            memory_kind: tenferro_tensor::MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        },
    )?);
    let view = TensorView::F64(
        input
            .as_typed::<f64>()
            .expect("unexpected input dtype")
            .as_view(),
    );
    let rejected = runtime
        .execute_scoped_read_only(&program, ScopedReadInputs::new(vec![view]))
        .expect_err("non-host scoped input must be rejected before admission");
    let (error, inputs) = rejected.into_parts();
    assert_eq!(inputs.len(), 1);
    assert!(error
        .to_string()
        .contains("scoped borrowed execution requires host/CPU storage"));
    Ok(())
}

#[test]
fn in_flight_worker_uses_state_captured_before_release() -> Result<(), Box<dyn StdError>> {
    let mut builder = Runtime::builder();
    builder.register_engine(admission_test_registration(false)?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&x, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let spawner = DelayedSpawner {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    };
    let admitted_epoch = runtime.epoch()?;

    let handle = submit_with_spawner(
        &runtime,
        &program,
        ExecutionInputs::new(vec![input.duplicate()?])?,
        &spawner,
    )?;
    started.wait();
    runtime.reconfigure(|edit| {
        edit.replace_engine(admission_test_registration(true)?)?;
        Ok(())
    })?;
    assert_ne!(runtime.epoch()?, admitted_epoch);
    release.wait();

    let output = match handle.wait()? {
        crate::runtime::ExecutionOutcome::Completed(output) => output,
        other => panic!("unexpected submission outcome: {other:?}"),
    };
    match output.output(0)? {
        crate::runtime::OutputRef::Tensor(tenferro_tensor::TensorView::F64(view)) => {
            assert_eq!(view.as_slice()?, &[1.0, 2.0]);
        }
        other => panic!("unexpected output view: {other:?}"),
    }
    let extracted = match output.into_output(0) {
        Ok(tensor) => tensor,
        Err((_bundle, error)) => return Err(Box::new(error)),
    };
    assert_eq!(extracted.as_slice::<f64>()?, &[1.0, 2.0]);
    Ok(())
}

#[test]
fn submission_spawn_failure_preserves_typed_source() -> Result<(), Box<dyn StdError>> {
    let submission = Arc::new(InFlightSubmission::for_test(|| Ok(Vec::new())));

    let (inputs, error) = *spawn_in_flight(submission, &FailingSpawner).unwrap_err();

    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<SubmissionError>())
        .expect("typed submission error source");
    assert!(matches!(source, SubmissionError::WorkerSpawn { .. }));
    assert!(format!("{inputs:?}").contains("ExecutionInputs"));
    Ok(())
}

#[test]
fn dropped_handle_does_not_cancel_blocked_worker() -> Result<(), Box<dyn StdError>> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let completed = Arc::new(AtomicBool::new(false));
    let worker_completed = Arc::clone(&completed);
    let submission = Arc::new(InFlightSubmission::for_test(move || {
        entered_tx.send(()).expect("test receiver remains alive");
        release_rx.recv().expect("test sender releases worker");
        worker_completed.store(true, Ordering::Release);
        Ok(Vec::new())
    }));

    let handle = spawn_in_flight(submission, &OsThreadSpawner).map_err(|failure| failure.1)?;
    entered_rx.recv()?;
    drop(handle);
    assert!(!completed.load(Ordering::Acquire));
    release_tx.send(())?;
    while !completed.load(Ordering::Acquire) {
        thread::yield_now();
    }
    Ok(())
}

#[test]
fn worker_panic_and_execution_error_complete_handle() -> Result<(), Box<dyn StdError>> {
    let panic_submission = Arc::new(InFlightSubmission::for_test(|| {
        panic!("injected submission panic")
    }));
    let panic_outcome = spawn_in_flight(panic_submission, &OsThreadSpawner)
        .map_err(|failure| failure.1)?
        .wait()?;
    match panic_outcome {
        crate::runtime::ExecutionOutcome::CompletionUnproven {
            error,
            diagnostic_keys,
        } => {
            assert!(error.to_string().contains("injected submission panic"));
            assert_eq!(diagnostic_keys.as_ref(), ["execution.retirement-unproven"]);
        }
        other => panic!("unexpected panic outcome: {other:?}"),
    }

    let error_submission = Arc::new(InFlightSubmission::for_test(|| {
        Err(Error::runtime_state(
            "submission-test",
            ErrorPhase::Execution,
            "injected execution error",
        ))
    }));
    let execution_error = spawn_in_flight(error_submission, &OsThreadSpawner)
        .map_err(|failure| failure.1)?
        .wait()
        .unwrap_err();
    assert!(execution_error
        .to_string()
        .contains("injected execution error"));
    Ok(())
}

#[test]
fn terminal_lazy_read_keeps_nonroot_location_for_materialization() -> Result<(), Box<dyn StdError>>
{
    let root_location = ExecutionLocation::new(
        EngineId::new("tenferro-test.output-root")?,
        test_provider_device_identity("output-root"),
        qualified_domain(1),
        StorageClass::new("tenferro-test.output-root-storage")?,
    );
    let output_location = ExecutionLocation::new(
        EngineId::new("tenferro-test.output-nonroot")?,
        test_provider_device_identity("output-nonroot"),
        qualified_domain(2),
        StorageClass::new("tenferro-test.output-nonroot-storage")?,
    );
    let domain = AllocationDomainId::fresh();
    let buffer = StorageBuffer::Backend(Box::new(ForeignProbeBuffer {
        values: Arc::new(Mutex::new(vec![1.0, 2.0, 3.0, 4.0])),
        domain,
        allocation: next_test_allocation_id(),
    }));
    let base: Tensor =
        TypedTensor::<f64>::from_buffer_col_major(vec![4], buffer, Placement::default())?.into();
    let view = TensorValue::from_parts(base, vec![2], vec![2], 0)?;
    let program = crate::exec::ExecProgram {
        instructions: Vec::new(),
        input_slots: vec![0],
        output_slots: vec![0],
        n_slots: 1,
        shape_guards: Vec::new(),
    };
    let mut outputs = vec![Some(LocatedExecSlot {
        location: output_location.clone(),
        value: ExecSlot::Read(view.tensor_read()),
    })];
    let mut materialized_at = None;

    let tensors = collect_tensor_outputs_with(&program, &mut outputs, |location, value| {
        assert_ne!(location, &root_location);
        assert_eq!(
            value.as_read().backend_family(),
            Some("tenferro-test.foreign-output-probe")
        );
        assert_eq!(value.as_read().allocation_domain(), Some(domain));
        materialized_at = Some(location.clone());
        Ok(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?)
    })?;

    assert_eq!(materialized_at, Some(output_location));
    assert_eq!(tensors[0].as_slice::<f64>()?, &[1.0, 3.0]);
    Ok(())
}

#[test]
fn result_retention_preserves_output_that_reuses_last_input_slot() {
    let location = ExecutionLocation::new(
        EngineId::new("tenferro-test.same-slot-engine").expect("engine id"),
        test_provider_device_identity("same-slot"),
        qualified_domain(1),
        StorageClass::new("tenferro-test.same-slot-storage").expect("storage class"),
    );
    let instruction = ExecInstruction {
        op: ExecOp::Negate,
        semantic_operation_index: None,
        input_slots: vec![0],
        output_slots: vec![0],
        dtype: DType::F64,
        output_shapes: Default::default(),
        output_extents: Default::default(),
        last_use: vec![true],
    };
    let output = Tensor::from_vec_col_major(vec![2], vec![-1.0_f64, -2.0]).expect("output tensor");
    let mut staged = vec![Some(ExecSlot::Owned(output))];
    let mut located: Vec<Vec<LocatedExecSlot<'_>>> = vec![Vec::new()];

    retain_instruction_results(&instruction, &location, &mut located, &mut staged)
        .expect("retain output");

    assert!(staged[0].is_none());
    assert_eq!(located[0].len(), 1);
    assert_eq!(located[0][0].location, location);
    let ExecSlot::Owned(output) = &located[0][0].value else {
        panic!("same-slot output must remain owned");
    };
    assert_eq!(output.as_slice::<f64>().expect("f64 output"), &[-1.0, -2.0]);
}

// --- one backend session per eligible prepared graph -------------------------

const COUNTING_ENGINE_ID: &str = "tenferro-test.session-counting-engine.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";

/// A `CpuBackend` host whose backend-session entries are counted.
///
/// `cached_entries` counts the runtime-cache-aware entry a graph-wide session
/// uses. `uncached_entries` counts the plain (non-cache-aware) entry; the leased
/// fallback also uses the cached entry, so a non-zero plain count means a node
/// or the output collection escaped the graph-wide cached session. No production
/// seam exposes either count, so the default `run_prepared` route is asserted
/// through this wrapper.
#[derive(Clone, Debug)]
struct CountingSessionHost {
    inner: CpuBackend,
    cached_entries: Arc<AtomicUsize>,
    uncached_entries: Arc<AtomicUsize>,
    /// Index of the cached entry that panics instead of admitting work.
    panic_on_cached_entry: Arc<AtomicUsize>,
}

impl CountingSessionHost {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
            cached_entries: Arc::new(AtomicUsize::new(0)),
            uncached_entries: Arc::new(AtomicUsize::new(0)),
            panic_on_cached_entry: Arc::new(AtomicUsize::new(usize::MAX)),
        }
    }

    fn cached_entries(&self) -> usize {
        self.cached_entries.load(Ordering::SeqCst)
    }

    fn uncached_entries(&self) -> usize {
        self.uncached_entries.load(Ordering::SeqCst)
    }

    /// Panic inside the session body of the next cached entry.
    fn panic_on_next_cached_entry(&self) {
        let next = self.cached_entries.load(Ordering::SeqCst);
        self.panic_on_cached_entry.store(next, Ordering::SeqCst);
    }
}

impl BackendRuntimeCache for CountingSessionHost {
    type RuntimeCache = <CpuBackend as BackendRuntimeCache>::RuntimeCache;
}

impl TensorBackend for CountingSessionHost {}

impl TensorDeviceTransfer for CountingSessionHost {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        self.inner.download_to_host(tensor)
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        self.inner.upload_host_tensor(tensor)
    }
}

impl BackendSessionHost for CountingSessionHost {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn Engine) -> R,
    ) -> Result<R, SessionEntryError> {
        self.uncached_entries.fetch_add(1, Ordering::SeqCst);
        self.inner.with_backend_session(f)
    }

    fn with_backend_session_cached<R>(
        &mut self,
        cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn Engine) -> R,
    ) -> Result<R, SessionEntryError> {
        let entry = self.cached_entries.fetch_add(1, Ordering::SeqCst);
        let panic_inside = self.panic_on_cached_entry.load(Ordering::SeqCst) == entry;
        self.inner
            .with_backend_session_cached(cache, move |session| {
                if panic_inside {
                    panic!("injected backend-session panic");
                }
                f(session)
            })
    }

    fn owns_session(&self, session: &mut dyn Engine) -> bool {
        self.inner.owns_session(session)
    }
}

/// A provider that prepares metadata-only plans, so a counting test can prepare
/// core operations for its engine.
///
/// The counting host is the *execution* backend, and the crate's unit-test build
/// cannot name the CPU provider's capability traits (the dev-dependency links a
/// separate `tenferro_runtime` instance), so this provider only returns the
/// binding and specialization the runtime asked for. Core operations dispatch on
/// the backend session during execution, so the prepared plan carries no
/// executable work of its own.
#[derive(Debug)]
struct MetadataOnlyProvider;

#[derive(Debug)]
struct MetadataOnlyOperation {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
}

impl MetadataOnlyOperation {
    fn new(context: &CorePrepareContext<'_>) -> Self {
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

fn metadata_only_plan(context: &CorePrepareContext<'_>) -> PrepareCapability {
    PrepareCapability::Prepared(PreparedOperationPlan::metadata(Arc::new(
        MetadataOnlyOperation::new(context),
    )))
}

impl ElementwiseRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl ReductionRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl IndexingRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl DotGeneralPreparation for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

impl LayoutRuntime for MetadataOnlyProvider {
    fn prepare(
        &self,
        request: LayoutPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(metadata_only_plan(request.context()))
    }
}

fn metadata_only_capabilities() -> CoreCapabilityBundle {
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

fn host_cpu_ingress(storage: &StorageClass) -> InputIngressContract {
    fn is_host(placement: &Placement) -> bool {
        matches!(
            placement.memory_kind,
            MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
        )
    }
    let placement_storage = storage.clone();
    let signature_storage = storage.clone();
    let runtime_storage = storage.clone();
    let resident_storage = storage.clone();
    InputIngressContract::new(
        InputPlacementContract::new(move |placement, candidate| {
            is_host(placement) && candidate == &placement_storage
        }),
        InputSignatureContract::new(move |placement, family, domain, candidate| {
            is_host(placement)
                && family.is_none()
                && domain.is_none()
                && candidate == &signature_storage
        }),
        RuntimeInputContract::new(move |input: &TensorRead<'_>, candidate| {
            is_host(input.placement())
                && input.backend_family().is_none()
                && candidate == &runtime_storage
        }),
        ResidentOutputContract::new(move |input: &TensorRead<'_>, candidate| {
            is_host(input.placement())
                && input.backend_family().is_none()
                && candidate == &resident_storage
        }),
    )
}

fn counting_engine_registration_with_driver(
    host: &CountingSessionHost,
    driver: Arc<dyn EventDomainDriver>,
) -> EngineRegistration {
    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).expect("cpu storage class");
    EngineRegistration::executable(
        ProviderExecutableBinding::new(
            EngineId::new(COUNTING_ENGINE_ID).expect("counting engine id"),
            HardwareClassId::new(CPU_HARDWARE_CLASS_ID).expect("cpu hardware class"),
            Arc::from(vec![storage.clone()]),
            storage.clone(),
            ExecutableEngineContract::new(
                test_provider_device_identity("session-counting"),
                metadata_only_capabilities(),
                host.clone(),
                driver,
                host_cpu_ingress(&storage),
                None,
            ),
        )
        .expect("counting provider binding"),
    )
}

fn counting_engine_registration(host: &CountingSessionHost) -> EngineRegistration {
    counting_engine_registration_with_driver(host, Arc::new(ImmediateEventDomainDriver::new()))
}

fn runtime_with_counting_host(host: &CountingSessionHost) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(counting_engine_registration(host))
        .expect("register counting engine");
    builder.build().expect("counting runtime")
}

#[derive(Debug)]
struct FailingDrainDriver {
    inner: ImmediateEventDomainDriver,
}

impl FailingDrainDriver {
    fn new() -> Self {
        Self {
            inner: ImmediateEventDomainDriver::new(),
        }
    }
}

#[derive(Debug)]
struct FailingDrainRun {
    inner: Box<dyn EventDomainRun>,
}

impl EventDomainRun for FailingDrainRun {
    fn domain(&self) -> EventDomainId {
        self.inner.domain()
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        self.inner.enqueue(dependencies, launch)
    }

    fn drain(&mut self) -> crate::Result<()> {
        Err(Error::runtime_state(
            "test event domain",
            ErrorPhase::Execution,
            "injected device-side retirement failure",
        ))
    }
}

impl EventDomainDriver for FailingDrainDriver {
    fn begin_run(&self, domain: EventDomainId) -> crate::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(FailingDrainRun {
            inner: self.inner.begin_run(domain)?,
        }))
    }
}

#[derive(Debug)]
struct CountingDrainDriver {
    inner: ImmediateEventDomainDriver,
    drains: Arc<AtomicUsize>,
}

impl CountingDrainDriver {
    fn new(drains: Arc<AtomicUsize>) -> Self {
        Self {
            inner: ImmediateEventDomainDriver::new(),
            drains,
        }
    }
}

#[derive(Debug)]
struct CountingDrainRun {
    inner: Box<dyn EventDomainRun>,
    drains: Arc<AtomicUsize>,
}

impl EventDomainRun for CountingDrainRun {
    fn domain(&self) -> EventDomainId {
        self.inner.domain()
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn EventToken>> {
        self.inner.enqueue(dependencies, launch)
    }

    fn drain(&mut self) -> crate::Result<()> {
        self.drains.fetch_add(1, Ordering::SeqCst);
        self.inner.drain()
    }
}

impl EventDomainDriver for CountingDrainDriver {
    fn begin_run(&self, domain: EventDomainId) -> crate::Result<Box<dyn EventDomainRun>> {
        Ok(Box::new(CountingDrainRun {
            inner: self.inner.begin_run(domain)?,
            drains: Arc::clone(&self.drains),
        }))
    }
}

#[test]
fn prepared_deferred_host_error_retires_before_returning() {
    let mut host = CountingSessionHost::new();
    let drains = Arc::new(AtomicUsize::new(0));
    let mut builder = Runtime::builder();
    builder
        .register_engine(counting_engine_registration_with_driver(
            &host,
            Arc::new(CountingDrainDriver::new(Arc::clone(&drains))),
        ))
        .expect("register counting-drain engine");
    let runtime = builder.build().expect("runtime");

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("input");
    let y = (&x + &x).expect("add");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .expect("compile");
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("input tensor");
    let wrong_shape =
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).expect("wrong input");
    let prepared = runtime
        .prepare_compiled(&program, &[&input])
        .expect("prepare");
    let inputs: [&Tensor; 1] = [&input];
    let bad_inputs: [&Tensor; 1] = [&wrong_shape];

    host.with_backend_session(|session| {
        let mut deferred = runtime
            .begin_prepared_deferred(session, &prepared)
            .expect("deferred guard");
        deferred
            .run_prepared_deferred(&inputs)
            .expect("first deferred run");
        assert_eq!(drains.load(Ordering::SeqCst), 0);
        let error = deferred
            .run_prepared_deferred(&bad_inputs)
            .expect_err("a wrong input shape must fail before enqueueing");
        assert!(
            error.to_string().contains("shape"),
            "unexpected host error: {error}"
        );
        assert_eq!(
            drains.load(Ordering::SeqCst),
            1,
            "the rejected call must retire the earlier deferred run before returning"
        );
        let runs = deferred
            .finish_prepared()
            .expect("finish after the early retirement");
        assert_eq!(runs.len(), 1);
        assert_eq!(
            runs[0][0].as_slice::<f64>().expect("host values"),
            &[2.0, 4.0]
        );
    })
    .expect("backend session entry");
}

#[test]
fn prepared_deferred_finish_returns_the_device_side_retirement_error() {
    let mut host = CountingSessionHost::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(counting_engine_registration_with_driver(
            &host,
            Arc::new(FailingDrainDriver::new()),
        ))
        .expect("register failing driver engine");
    let runtime = builder.build().expect("runtime");

    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("input");
    let y = (&x + &x).expect("add");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .expect("compile");
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("input tensor");
    let prepared = runtime
        .prepare_compiled(&program, &[&input])
        .expect("prepare");
    let inputs: [&Tensor; 1] = [&input];

    host.with_backend_session(|session| {
        let mut deferred = runtime
            .begin_prepared_deferred(session, &prepared)
            .expect("deferred guard");
        deferred
            .run_prepared_deferred(&inputs)
            .expect("the deferred run itself only enqueues");
        let error = deferred
            .finish_prepared()
            .expect_err("retirement must report the injected device-side failure");
        assert!(
            error
                .to_string()
                .contains("injected device-side retirement failure"),
            "unexpected finish error: {error}"
        );
    })
    .expect("backend session entry");
}

/// Several scheduled operations of an eligible graph share one session entry.
///
/// A reduction keeps the graph out of the elementwise-region planner, so every
/// instruction dispatches on its own and the per-instruction route would show
/// one cached entry per node.
#[test]
fn run_prepared_opens_one_session_for_several_scheduled_operations() -> Result<(), Box<dyn StdError>>
{
    let host = CountingSessionHost::new();
    let runtime = runtime_with_counting_host(&host);
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let squared = (&x * &x)?;
    let y = squared.reduce_sum(None)?;
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])?;
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    let prepared = runtime.prepare_compiled(&program, &[&input])?;
    // Count from after preparation: preparation and backend construction are not
    // what this test measures.
    let cached_after_preparation = host.cached_entries();
    let uncached_after_preparation = host.uncached_entries();

    for _ in 0..2 {
        let output = runtime.run_prepared(&prepared, &[&input])?;
        assert_eq!(output[0].as_slice::<f64>()?, &[14.0]);
    }

    assert_eq!(
        host.cached_entries() - cached_after_preparation,
        2,
        "each run_prepared call must open exactly one backend session for the graph"
    );
    assert_eq!(
        host.uncached_entries() - uncached_after_preparation,
        0,
        "no per-instruction or per-materialization session may run beside the graph session"
    );
    assert_eq!(
        prepared.elementwise_region_execution_counts(),
        (0, 0),
        "a reduction graph plans no elementwise region"
    );
    Ok(())
}

/// An accepted fusion, its per-instruction fallback, and the identity graph all
/// stay inside the one graph session.
#[test]
fn run_prepared_opens_one_session_for_fused_and_declined_fusions() -> Result<(), Box<dyn StdError>>
{
    let host = CountingSessionHost::new();
    let runtime = runtime_with_counting_host(&host);

    // Above the CPU fused kernel's element floor: the planned region executes as
    // one fused command.
    let fused_length = 16 * 1024usize;
    let x = TracedTensor::input_concrete_shape(DType::F64, &[fused_length])?;
    let doubled = (&x + &x)?;
    let chain = doubled.mul(&doubled)?.exp()?.tanh()?;
    let program = GraphCompiler::new()
        .compile_with_input_specs(&chain, &[(&x, DType::F64, &[fused_length])])?;
    let fused_data: Vec<f64> = (0..fused_length)
        .map(|index| 0.25 + (index % 17) as f64 * 0.125)
        .collect();
    let input = Tensor::from_vec_col_major(vec![fused_length], fused_data.clone())?;
    let prepared = runtime.prepare_compiled(&program, &[&input])?;
    let before_fused = host.cached_entries();
    let mut fused_output = runtime.run_prepared(&prepared, &[&input])?;
    assert_eq!(
        host.cached_entries() - before_fused,
        1,
        "an accepted fusion must not open a session beside the graph session"
    );
    assert_eq!(
        prepared.elementwise_region_execution_counts(),
        (1, 0),
        "the long elementwise chain must execute as one fused command"
    );
    let fused_output = fused_output.pop().expect("one output");
    for (index, value) in fused_output.as_slice::<f64>()?.iter().enumerate() {
        let doubled = fused_data[index] + fused_data[index];
        assert!((value - (doubled * doubled).exp().tanh()).abs() <= 1e-12);
    }

    // Below the floor: the same chain is planned, the backend declines it, and the
    // region falls back per instruction on the same session.
    let small_length = 4usize;
    let small_x = TracedTensor::input_concrete_shape(DType::F64, &[small_length])?;
    let small_doubled = (&small_x + &small_x)?;
    let small_chain = small_doubled.mul(&small_doubled)?.exp()?.tanh()?;
    let small_program = GraphCompiler::new()
        .compile_with_input_specs(&small_chain, &[(&small_x, DType::F64, &[small_length])])?;
    let small_data = [0.5_f64, 1.0, 1.5, 2.0];
    let small_input = Tensor::from_vec_col_major(vec![small_length], small_data.to_vec())?;
    let small_prepared = runtime.prepare_compiled(&small_program, &[&small_input])?;
    let before_declined = host.cached_entries();
    let mut declined_output = runtime.run_prepared(&small_prepared, &[&small_input])?;
    assert_eq!(
        host.cached_entries() - before_declined,
        1,
        "a declined fusion must fall back inside the graph session, not beside it"
    );
    assert_eq!(
        small_prepared.elementwise_region_execution_counts(),
        (0, 1),
        "the CPU backend declines fusion below its element floor"
    );
    let declined_output = declined_output.pop().expect("one output");
    for (index, value) in declined_output.as_slice::<f64>()?.iter().enumerate() {
        let doubled = small_data[index] + small_data[index];
        assert!((value - (doubled * doubled).exp().tanh()).abs() <= 1e-12);
    }

    // The identity graph has no scheduled node, so its only work is the output
    // copy. That copy must run on the same graph session: the per-instruction
    // route materializes on a fresh plain session instead, which would show up in
    // `uncached_entries`.
    let identity_x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let identity_program = GraphCompiler::new()
        .compile_with_input_specs(&identity_x, &[(&identity_x, DType::F64, &[2])])?;
    let identity_input = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 7.0])?;
    let identity_prepared = runtime.prepare_compiled(&identity_program, &[&identity_input])?;
    let before_identity = host.cached_entries();
    let before_identity_uncached = host.uncached_entries();
    let identity_output = runtime.run_prepared(&identity_prepared, &[&identity_input])?;
    assert_eq!(identity_output[0].as_slice::<f64>()?, &[4.0, 7.0]);
    assert_eq!(
        host.cached_entries() - before_identity,
        1,
        "the identity graph must open exactly one backend session"
    );
    assert_eq!(
        host.uncached_entries() - before_identity_uncached,
        0,
        "identity output materialization must stay inside the graph session"
    );
    Ok(())
}

/// An execution error inside the graph session releases the lease and the
/// session, so the same prepared handle runs again.
#[test]
fn run_prepared_reuses_the_executor_after_an_execution_error() -> Result<(), Box<dyn StdError>> {
    let host = CountingSessionHost::new();
    let runtime = runtime_with_counting_host(&host);
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x * &x)?;
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;
    let prepared = runtime.prepare_compiled(&program, &[&input])?;
    // Device-resident storage cannot enter the host ingress of this engine, so the
    // failure happens inside the run after its session was opened.
    let foreign_input = Tensor::from_typed::<f64>(TypedTensor::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(ForeignProbeBuffer {
            values: Arc::new(Mutex::new(vec![3.0, 5.0])),
            domain: AllocationDomainId::fresh(),
            allocation: next_test_allocation_id(),
        })),
        Placement {
            memory_kind: MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        },
    )?);
    let cached_before = host.cached_entries();

    let error = runtime
        .run_prepared(&prepared, &[&foreign_input])
        .expect_err("a device-resident input must fail host ingress inside the graph session");
    let message = error.to_string();
    assert!(
        message.contains("residency")
            || message.contains("allocation")
            || message.contains("ingress"),
        "unexpected ingress error: {message}"
    );
    assert_eq!(
        host.cached_entries() - cached_before,
        1,
        "the failed run still used exactly one graph session"
    );

    let output = runtime.run_prepared(&prepared, &[&input])?;
    assert_eq!(output[0].as_slice::<f64>()?, &[9.0, 25.0]);
    assert_eq!(
        host.cached_entries() - cached_before,
        2,
        "the executor must be reusable after an ordinary execution error"
    );
    Ok(())
}

/// A panic inside the graph session keeps the executor's established poisoned
/// behavior: session resources are released and later calls report the poison
/// instead of running on a half-released backend.
#[test]
fn run_prepared_poisons_the_executor_after_a_session_panic() -> Result<(), Box<dyn StdError>> {
    let host = CountingSessionHost::new();
    let runtime = runtime_with_counting_host(&host);
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let program = GraphCompiler::new().compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0])?;
    let prepared = runtime.prepare_compiled(&program, &[&input])?;
    host.panic_on_next_cached_entry();
    let cached_before = host.cached_entries();

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = runtime.run_prepared(&prepared, &[&input]);
    }));

    assert!(panic.is_err(), "an injected session panic must unwind");
    assert_eq!(
        host.cached_entries() - cached_before,
        1,
        "the panic must happen inside the single graph session"
    );

    let error = runtime
        .run_prepared(&prepared, &[&input])
        .expect_err("a panicked executor must stay poisoned");
    assert!(
        error.to_string().contains("poisoned"),
        "unexpected post-panic error: {error}"
    );
    Ok(())
}
