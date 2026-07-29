use std::error::Error as StdError;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::thread;

use tenferro_tensor::{DType, Tensor};

use crate::exec::{ExecInstruction, ExecOp, ExecSlot};
use crate::runtime::execution::{
    retain_instruction_results, spawn_in_flight, InFlightSubmission, LocatedExecSlot,
    OsThreadSpawner, SubmissionSpawner,
};
use crate::runtime::schedule::ExecutionLocation;
use crate::runtime::{EngineId, EventDomainId, StorageClass, SubmissionError};
use crate::{Error, ErrorPhase};

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

#[test]
fn in_flight_worker_uses_state_captured_before_release() -> Result<(), Box<dyn StdError>> {
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let admitted_epoch = 7.0_f64;
    let current_epoch = Arc::new(std::sync::atomic::AtomicU64::new(7));
    let worker_epoch = Arc::clone(&current_epoch);
    let spawner = DelayedSpawner {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    };
    let submission = Arc::new(InFlightSubmission::for_test(move || {
        let _current_epoch = worker_epoch.load(Ordering::Acquire);
        Ok(vec![Tensor::from_vec_col_major(
            vec![1],
            vec![admitted_epoch],
        )?])
    }));

    let handle = spawn_in_flight(submission, &spawner)?;
    started.wait();
    current_epoch.store(8, Ordering::Release);
    release.wait();

    let output = handle.wait()?;
    assert_eq!(output[0].as_slice::<f64>()?, &[admitted_epoch]);
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
fn result_retention_preserves_output_that_reuses_last_input_slot() {
    let location = ExecutionLocation::new(
        EngineId::new("tenferro-test.same-slot-engine").expect("engine id"),
        EventDomainId::runtime_created_for_test(1),
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
