//! Session-route executor tests for the prepared execution boundary (#2182).
//!
//! These live in a module-local test file so `runtime/execution.rs` keeps only
//! production code, per `REPOSITORY_RULES.md` "Unit Test Organization". The
//! sibling `tests` module holds the pre-existing probe fixtures.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tenferro_cpu::CpuBackend;

use crate::runtime::PreparedOperationPlan;
use crate::{Error, ExtensionCacheStore};

use super::tests::{
    lock_probe_program, probe_binding, probe_specialization, SessionProbePreparedOperation,
};
use super::{ensure_instructions_support_session, TensorBackendExecutor};

#[test]
fn tensor_backend_executor_try_lease_reports_busy_without_blocking() {
    let executor = Arc::new(TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new()));
    let held = executor
        .try_lease_state("test.holder")
        .expect("the first try-lease succeeds");
    let contender = Arc::clone(&executor);
    let worker = std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let error = match contender.try_lease_state("test.contender") {
            Ok(_lease) => panic!("a contended try-lease must fail"),
            Err(error) => error,
        };
        (error, started.elapsed())
    });
    let (error, elapsed) = worker.join().expect("contender thread joins");
    assert!(
        error.to_string().contains("busy on another thread"),
        "{error}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "try-lease must not wait for the holder: {elapsed:?}"
    );
    drop(held);
    executor
        .try_lease_state("test.after")
        .expect("try-lease succeeds after the holder releases");
}

#[test]
fn session_preflight_rejects_a_metadata_only_extension_plan() {
    // A registered extension can hand the runtime a metadata-only plan (no
    // executor bridge). The instruction-level preflight must reject it
    // before an earlier node runs, not fail later at dispatch.
    let prepared = Arc::new(SessionProbePreparedOperation {
        binding: probe_binding(),
        specialization: probe_specialization(),
        observed_session: Arc::new(AtomicBool::new(false)),
    });
    let mut caches = ExtensionCacheStore::new();
    let metadata_only = vec![PreparedOperationPlan::metadata(prepared.clone())];
    let error = ensure_instructions_support_session(
        &lock_probe_program(),
        &metadata_only,
        &mut caches,
        "test",
    )
    .expect_err("a metadata-only extension plan must be rejected");
    assert!(matches!(error, Error::Unsupported { .. }), "{error}");

    let executable = vec![PreparedOperationPlan::executable(
        prepared.clone(),
        prepared,
    )];
    ensure_instructions_support_session(&lock_probe_program(), &executable, &mut caches, "test")
        .expect("a session-capable extension plan passes preflight");
}
