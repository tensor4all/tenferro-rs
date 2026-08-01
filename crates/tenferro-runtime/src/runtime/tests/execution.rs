use std::any::Any;
use std::error::Error as StdError;
use std::io;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::thread;

use tenferro_tensor::{
    AllocationDomainId, BackendBuffer, Buffer, DType, HostAccessError, HostReadGuard,
    HostWriteGuard, Placement, Tensor, TensorOwnedView, TensorValue, TypedTensor,
};

use crate::exec::{ExecInstruction, ExecOp, ExecProgram, ExecSlot};
use crate::runtime::execution::{
    collect_tensor_outputs_with, retain_instruction_results, spawn_in_flight, submit_with_spawner,
    ErasedTensorBackendExecutor, InFlightSubmission, LocatedExecSlot, OsThreadSpawner,
    RuntimeOutputMode, SubmissionSpawner,
};
use crate::runtime::schedule::ExecutionLocation;
use crate::runtime::{
    CacheOwnerError, CacheStats, CoreCapabilityBundle, EngineId, EngineRegistration, EventDomainId,
    ExecutionContextIdentity, HardwareClassId, PreparedOperationPlan, ProviderDeviceIdentity,
    ProviderId, RegistrationIdentity, Runtime, RuntimeConfigError, RuntimeEpoch, RuntimeId,
    StorageClass, SubmissionError,
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

#[derive(Debug)]
struct ForeignProbeBuffer {
    values: Arc<Mutex<Vec<f64>>>,
    domain: AllocationDomainId,
}

impl BackendBuffer<f64> for ForeignProbeBuffer {
    fn backend_family(&self) -> &'static str {
        "tenferro-test.foreign-output-probe"
    }

    fn len(&self) -> usize {
        self.values.lock().expect("probe buffer lock").len()
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, f64>, HostAccessError> {
        Ok(HostReadGuard::new(
            self.values.lock().expect("probe buffer lock"),
        ))
    }

    fn map_write(&self) -> Result<HostWriteGuard<'_, f64>, HostAccessError> {
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
}

impl ErasedTensorBackendExecutor for AdmissionTestExecutor {
    fn backend_type_name(&self) -> &'static str {
        "AdmissionTestExecutor"
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
        Ok(inputs.iter().map(|input| (*input).clone()).collect())
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
        Ok(inputs
            .iter()
            .map(|input| TensorValue::from_tensor((*input).clone()))
            .collect())
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

    fn materialize_slot<'input>(&self, slot: ExecSlot<'input>) -> crate::Result<Tensor> {
        if self.fail_materialize {
            return Err(Error::runtime_state(
                "AdmissionTestExecutor",
                ErrorPhase::Execution,
                "replacement executor must not serve admitted work",
            ));
        }
        slot.as_tensor("AdmissionTestExecutor::materialize_slot")
            .cloned()
    }

    fn materialize_slot_value<'input>(&self, slot: ExecSlot<'input>) -> crate::Result<TensorValue> {
        self.materialize_slot(slot).map(TensorValue::from_tensor)
    }
}

fn admission_test_registration(
    fail_materialize: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let storage = StorageClass::new("tenferro-test.admission-storage.v1")?;
    let engine_id = EngineId::new("tenferro-test.admission-engine.v1")?;
    let mut registration = EngineRegistration::new(
        engine_id,
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.admission").expect("provider id"),
            "admission-engine",
        )
        .expect("provider target"),
        ExecutionContextIdentity::of::<AdmissionTestExecutor>(),
        HardwareClassId::new("tenferro-test.admission-hardware.v1")?,
        Arc::from(vec![storage.clone()]),
        storage.clone(),
        CoreCapabilityBundle::builder().build(),
    )?
    .with_input_signature_validator({
        let storage = storage.clone();
        move |_, family, domain, candidate| {
            candidate == &storage && family.is_none() && domain.is_none()
        }
    })
    .with_input_ingress_validator(
        {
            let storage = storage.clone();
            move |_, candidate| candidate == &storage
        },
        {
            let storage = storage.clone();
            move |input, candidate| candidate == &storage && input.backend_family().is_none()
        },
        move |input, candidate| candidate == &storage && input.backend_family().is_none(),
    );
    registration.execution_engine = Some(Arc::new(AdmissionTestExecutor { fail_materialize }));
    Ok(registration)
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

    let handle = submit_with_spawner(&runtime, &program, &[&input], &spawner)?;
    started.wait();
    runtime.reconfigure(|edit| {
        edit.replace_engine(admission_test_registration(true)?)?;
        Ok(())
    })?;
    assert_ne!(runtime.epoch()?, admitted_epoch);
    release.wait();

    let output = handle.wait()?;
    assert_eq!(output[0].as_slice::<f64>()?, &[1.0, 2.0]);
    Ok(())
}

#[test]
fn submission_spawn_failure_preserves_typed_source() -> Result<(), Box<dyn StdError>> {
    let submission = Arc::new(InFlightSubmission::for_test(|| Ok(Vec::new())));

    let error = spawn_in_flight(submission, &FailingSpawner).unwrap_err();

    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<SubmissionError>())
        .expect("typed submission error source");
    assert!(matches!(source, SubmissionError::WorkerSpawn { .. }));
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

    let handle = spawn_in_flight(submission, &OsThreadSpawner)?;
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
    let panic_error = spawn_in_flight(panic_submission, &OsThreadSpawner)?
        .wait()
        .unwrap_err();
    assert!(panic_error
        .to_string()
        .contains("injected submission panic"));

    let error_submission = Arc::new(InFlightSubmission::for_test(|| {
        Err(Error::runtime_state(
            "submission-test",
            ErrorPhase::Execution,
            "injected execution error",
        ))
    }));
    let execution_error = spawn_in_flight(error_submission, &OsThreadSpawner)?
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
    let buffer = Buffer::Backend(Arc::new(ForeignProbeBuffer {
        values: Arc::new(Mutex::new(vec![1.0, 2.0, 3.0, 4.0])),
        domain,
    }));
    let base: Tensor =
        TypedTensor::<f64>::from_buffer_col_major(vec![4], buffer, Placement::default())?.into();
    let view = TensorOwnedView::from_parts(Arc::new(base), vec![2], vec![2], 0)?;
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
