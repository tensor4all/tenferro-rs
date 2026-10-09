//! Child-execution handles: descendants join the issuing execution's reservation
//! and own their N-ary scratch, so concurrent children cannot race on one lease.

use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_tensor::{BackendSessionHost, SessionEntryError};

const CHILD_WAIT: Duration = Duration::from_secs(30);

/// A descendant session runs while the issuing execution is active.
#[test]
fn child_session_runs_on_a_scoped_worker() {
    let mut backend = CpuBackend::with_threads(2).expect("backend");
    let value = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let child = cpu.child_execution();
                thread::scope(|scope| {
                    let worker = scope.spawn(|| child.backend().with_backend_session(|_| 21usize));
                    worker.join().expect("child worker must not panic")
                })
            })
            .expect("CpuBackend exposes its CpuExecSession")
        })
        .expect("issuing session entry")
        .expect("child session entry");
    assert_eq!(value, 21);
}

/// The issuing session's N-ary scratch lease does not block a concurrent child:
/// the child owns its own store, so both children hold a lease at the same time
/// while the issuing execution still holds its own.
#[test]
fn concurrent_children_own_their_nary_scratch() {
    let mut backend = CpuBackend::with_threads(2).expect("backend");
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Arc::new(Mutex::new(release_rx));
    let total = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let child = cpu.child_execution();
                cpu.with_contraction_exec(|_exec, buffers, workspaces| {
                    let limit = buffers.max_retained_capacity_bytes();
                    // `with_scratch` holds this execution's N-ary lease for the
                    // whole closure. A child that shared the store could not take
                    // a lease of its own and would fail with the typed "CPU N-ary
                    // workspace" error, so the two children below would never both
                    // report entry.
                    workspaces.with_scratch::<f64, _>(8, limit, |scratch| {
                        assert_eq!(scratch.len(), 8);
                        let total = thread::scope(|scope| {
                            let workers = [0, 1].map(|_| {
                                let child = &child;
                                let entered_tx = entered_tx.clone();
                                let release_rx = Arc::clone(&release_rx);
                                scope.spawn(move || {
                                    child
                                        .backend()
                                        .with_backend_session(|session| {
                                            with_cpu_exec_session(session, |cpu| {
                                                cpu.with_contraction_exec(|_, buffers, store| {
                                                    store.with_scratch::<f64, _>(
                                                        8,
                                                        buffers.max_retained_capacity_bytes(),
                                                        |scratch| {
                                                            assert_eq!(scratch.len(), 8);
                                                            entered_tx
                                                                .send(())
                                                                .expect("report entry");
                                                            release_rx
                                                                .lock()
                                                                .expect("release channel")
                                                                .recv_timeout(CHILD_WAIT)
                                                                .expect("release while the issuing lease is held");
                                                            Ok(1usize)
                                                        },
                                                    )
                                                })
                                            })
                                            .expect("CpuBackend exposes its CpuExecSession")
                                        })
                                        .expect("child session entry")
                                        .expect("child N-ary scratch")
                                })
                            });
                            // Both children must be inside their own lease before
                            // either is released: this is the deterministic overlap
                            // the shared store cannot satisfy.
                            for _ in 0..2 {
                                entered_rx
                                    .recv_timeout(CHILD_WAIT)
                                    .expect("both children must enter concurrently");
                            }
                            for _ in 0..2 {
                                release_tx.send(()).expect("release child");
                            }
                            workers
                                .into_iter()
                                .map(|worker| worker.join().expect("child worker must not panic"))
                                .sum::<usize>()
                        });
                        Ok(total)
                    })
                })
            })
            .expect("CpuBackend exposes its CpuExecSession")
        })
        .expect("issuing session entry")
        .expect("concurrent children must own their N-ary scratch");
    assert_eq!(total, 2);
}

/// The issuing thread's own recursion stays rejected: a descendant entry from the
/// thread that already holds the execution reports the typed reentry error and
/// never runs the callback.
#[test]
fn same_thread_child_entry_stays_rejected() {
    let mut backend = CpuBackend::with_threads(1).expect("backend");
    backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let mut child_backend = cpu.child_execution().backend();
                let ran = std::cell::Cell::new(false);
                let outcome = child_backend.with_backend_session(|_| ran.set(true));
                assert!(
                    matches!(outcome, Err(SessionEntryError::Reentered { .. })),
                    "nested entry on the issuing thread must be a reentry error: {outcome:?}"
                );
                assert!(!ran.get(), "the rejected callback must not run");
            })
            .expect("CpuBackend exposes its CpuExecSession");
        })
        .expect("issuing session entry");
}

/// A handle whose issuing execution has finished is rejected with the typed
/// error instead of being admitted as a fresh independent owner.
#[test]
fn stale_child_handle_is_rejected_with_the_typed_error() {
    let mut backend = CpuBackend::with_threads(1).expect("backend");
    let mut child_backend = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| cpu.child_execution().backend())
                .expect("CpuBackend exposes its CpuExecSession")
        })
        .expect("issuing session entry");
    let outcome = child_backend.with_backend_session(|_| ());
    assert!(
        matches!(outcome, Err(SessionEntryError::Reentered { .. })),
        "a session from a finished execution must be a reentry error: {outcome:?}"
    );
}

/// A panic inside a child's N-ary scratch use is contained: a fresh child still
/// gets a working store while the issuing execution continues.
#[test]
fn panic_inside_a_child_nary_scratch_recovers() {
    let mut backend = CpuBackend::with_threads(1).expect("backend");
    backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let child = cpu.child_execution();
                let panicked = thread::scope(|scope| {
                    let worker = scope.spawn(|| {
                        child
                            .backend()
                            .with_backend_session(|session| {
                                with_cpu_exec_session(session, |cpu| {
                                    cpu.with_contraction_exec(|_, buffers, store| {
                                        store.with_scratch::<f64, _>(
                                            4,
                                            buffers.max_retained_capacity_bytes(),
                                            |_scratch| -> Result<(), tenferro_tensor::Error> {
                                                panic!("panic inside a child N-ary scratch")
                                            },
                                        )
                                    })
                                })
                                .expect("CpuBackend exposes its CpuExecSession")
                            })
                            .expect("child session entry")
                    });
                    worker.join()
                });
                assert!(panicked.is_err(), "the child panic must surface");

                let value = thread::scope(|scope| {
                    let worker = scope.spawn(|| {
                        child
                            .backend()
                            .with_backend_session(|session| {
                                with_cpu_exec_session(session, |cpu| {
                                    cpu.with_contraction_exec(|_, buffers, store| {
                                        store.with_scratch::<f64, _>(
                                            4,
                                            buffers.max_retained_capacity_bytes(),
                                            |scratch| Ok(scratch.len()),
                                        )
                                    })
                                })
                                .expect("CpuBackend exposes its CpuExecSession")
                            })
                            .expect("child session entry")
                    });
                    worker.join().expect("a fresh child must not panic")
                })
                .expect("a fresh child must own a usable N-ary store");
                assert_eq!(value, 4);
            })
            .expect("CpuBackend exposes its CpuExecSession");
        })
        .expect("issuing session entry");
}

/// The handle is shareable across workers and has a summary `Debug`.
#[test]
fn child_handle_is_sync_and_debug() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<tenferro_cpu::CpuChildExecution<'static>>();

    let mut backend = CpuBackend::with_threads(1).expect("backend");
    backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let rendered = format!("{:?}", cpu.child_execution());
                assert!(rendered.starts_with("CpuChildExecution"), "{rendered}");
            })
            .expect("CpuBackend exposes its CpuExecSession");
        })
        .expect("issuing session entry");
}

/// The motivating case: workers of a foreign Rayon pool (the consumer's own
/// pool) hold child sessions while the issuing execution is active.
#[test]
fn child_session_runs_on_foreign_rayon_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("foreign pool");
    let mut backend = CpuBackend::with_threads(2).expect("backend");
    let outcomes = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                let child = cpu.child_execution();
                let (tx, rx) = mpsc::channel();
                pool.scope(|scope| {
                    for _ in 0..2 {
                        let child = &child;
                        let tx = tx.clone();
                        scope.spawn(move |_| {
                            let outcome = child.backend().with_backend_session(|_| 5usize);
                            tx.send(outcome).expect("report child outcome");
                        });
                    }
                });
                (0..2)
                    .map(|_| {
                        rx.recv_timeout(CHILD_WAIT)
                            .expect("both Rayon workers must report")
                    })
                    .collect::<Vec<_>>()
            })
            .expect("CpuBackend exposes its CpuExecSession")
        })
        .expect("issuing session entry");
    assert_eq!(outcomes.len(), 2);
    for outcome in outcomes {
        assert_eq!(outcome.expect("child session under a Rayon worker"), 5);
    }
}
